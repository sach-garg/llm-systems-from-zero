import argparse
import torch
from torch.utils.checkpoint import checkpoint
from pathlib import Path
import pandas as pd

from llm_systems.optimizer import AdamW
from llm_systems.model import TransformerBlock, RotaryEmbedding
from llm_systems.benchmark_config import BenchmarkConfig


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--context_length", type=int)
    parser.add_argument("--d_model", type=int)
    parser.add_argument("--d_ff", type=int)
    parser.add_argument("--num_layers", type=int)
    parser.add_argument("--num_heads", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--warmup", type=int)
    parser.add_argument("--mode", type=str, choices=["FB", "FBO"]) ##activation checkpointing is mainly for "FB" mode
    parser.add_argument("--device", type=str)
    parser.add_argument("--memory_file", type=str)
    parser.add_argument("--chunk_size", type=int)
    parser.add_argument("--operator_fusion", action="store_true")
    parser.add_argument("--out_dir", type=str)
    return parser.parse_args()


def build_config():
    config = BenchmarkConfig()
    args = parse_args()
    for key, value in vars(args).items():
        if value is not None:
            setattr(config, key, value)
    return config

def append_benchmark_result(config, peak_memory):
    out_path = Path(config.out_dir) / "checkpoint_benchmark.csv"
    new_row = pd.DataFrame(
        [{"d_model": config.d_model, "d_ff": config.d_ff,
            "num_layers": config.num_layers,"num_heads": config.num_heads,
            "context_length":config.context_length, "batch_size": config.batch_size,
            "mode": config.mode, "warm_up": config.warmup, "torch_compile":config.operator_fusion, 
            "chunk_size":config.chunk_size, "peak_memory": peak_memory
              }])

    if out_path.exists():
        old_df = pd.read_csv(out_path)
        df = pd.concat([old_df, new_row], ignore_index=True)
    else:
        df = new_row
    return df


def run_chunk(blocks, x):
    for block in blocks:
        x = block(x)
    return x


def model_with_checkpointing(blocks, x, chunk_size):
    n = len(blocks)
    for start in range(0, n, chunk_size):
        chunk = blocks[start:start + chunk_size]

        def chunk_fn(x, chunk=chunk):
            return run_chunk(chunk, x)

        x = checkpoint(chunk_fn, x, use_reentrant=False)
    return x


def main():
    config = build_config()

    device = config.device if config.device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)


    if not hasattr(config, "chunk_size") or config.chunk_size is None:
        raise ValueError("Please provide --chunk_size")

    x = torch.randn(config.batch_size,config.context_length,config.d_model,device=device, dtype=torch.float32,requires_grad=False)

    rope_app = RotaryEmbedding(context_length=config.context_length,dim=config.d_model // config.num_heads,
                               theta=config.rope_theta, device=device)

    if config.operator_fusion:
        blocks = torch.nn.ModuleList([
            torch.compile(
                TransformerBlock(
                    d_model=config.d_model,
                    num_heads=config.num_heads,
                    d_ff=config.d_ff,
                    positional_encoder=rope_app,
                    device=device,
                    dtype=torch.float32,
                ).to(device),
                fullgraph=True,
            )
            for _ in range(config.num_layers)
        ])
    else:
        blocks = torch.nn.ModuleList([
            TransformerBlock(
                d_model=config.d_model,
                num_heads=config.num_heads,
                d_ff=config.d_ff,
                positional_encoder=rope_app,
                device=device,
                dtype=torch.float32,
            ).to(device)
            for _ in range(config.num_layers)
        ])

    optimizer = AdamW(blocks.parameters(),lr=config.lr_max,weight_decay=config.weight_decay,betas=(config.beta1, config.beta2),eps=config.eps)

    # Warmup
    for _ in range(config.warmup):
        y = model_with_checkpointing(blocks, x, config.chunk_size)
        loss = y.sum()
        optimizer.zero_grad()
        loss.backward()
        if config.mode=="FBO":
            optimizer.step()

    torch.cuda.synchronize() ### Make sure all GPUs kernels started during warmup get finished
    torch.cuda.reset_peak_memory_stats() ### start counting memory peak from here onwards
    torch.cuda.memory._record_memory_history(max_entries=100000) 

    ### Memory recording starts after warmup, so warmup allocation events are not logged.
    ### However, persistent state from warmup (parameters, gradients, optimizer state, allocator cache)
    ### may still affect the baseline memory before the measured step.  


    y = model_with_checkpointing(blocks, x, config.chunk_size)
    loss = y.sum()
    optimizer.zero_grad()
    loss.backward()
    if config.mode=="FBO":
        optimizer.step()

    torch.cuda.synchronize()
    peak_mib = torch.cuda.max_memory_allocated() / (1024 ** 2)

    torch.cuda.memory._dump_snapshot(config.memory_file)
    torch.cuda.memory._record_memory_history(enabled=None)

    print(f"chunk_size = {config.chunk_size}")
    print(f"operator_fusion = {getattr(config, 'operator_fusion', False)}")
    print(f"peak memory = {peak_mib:.2f} MiB")
    print(f"memory snapshot saved to {config.memory_file}")


    df = append_benchmark_result(config, peak_mib)
    df.to_csv(Path(config.out_dir) / "checkpoint_benchmark.csv", index=False)


if __name__ == "__main__":
    main()