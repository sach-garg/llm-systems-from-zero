from llm_systems.model import BasicsTransformerLM
from llm_systems.optimizer import AdamW
from llm_systems.nn_utils import cross_entropy
from pathlib import Path
import torch
from timeit import default_timer
import pandas as pd
from contextlib import nullcontext


from llm_systems.benchmark_config import BenchmarkConfig

import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--context_length", type=int)
    parser.add_argument("--d_model", type=int)
    parser.add_argument("--d_ff", type=int)
    parser.add_argument("--num_layers", type=int)
    parser.add_argument("--num_heads", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--measure_iters", type=int)
    parser.add_argument("--warmup", type=int)
    parser.add_argument("--mode", type=str, choices=["F", "FB", "FBO"])
    parser.add_argument("--device", type=str)
    parser.add_argument("--precision",type=str,choices = ["torch.float32","torch.float16","torch.bfloat16"])
    parser.add_argument("--out_dir", type=str)
    
    return parser.parse_args()


def BuildConfig():
    config = BenchmarkConfig()
    args = parse_args()
    for key, value in vars(args).items():
        if value is not None:
            setattr(config, key, value)
    
    return config


def append_benchmark_result(config, device, times):
    out_path = Path(config.out_dir) / "benchmark_history.csv"
    new_row = pd.DataFrame(
        [{"d_model": config.d_model, "d_ff": config.d_ff,
            "num_layers": config.num_layers,"num_heads": config.num_heads,
            "context_length":config.context_length, "batch_size": config.batch_size,
            "mode": config.mode, "warm_up": config.warmup, "device": device, "dtype":config.precision,
              "Avg_time": pd.Series(times).mean(), "Std_time": pd.Series(times).std(), "Iterations_measured":config.measure_iters}])

    if out_path.exists():
        old_df = pd.read_csv(out_path)
        df = pd.concat([old_df, new_row], ignore_index=True)
    else:
        df = new_row
    return df

def main():
    
    config = BuildConfig()
    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if config.precision == "torch.float16":
        #autocast = torch.autocast(device_type="cuda", dtype=torch.float16) ### Not a very clean way to store and keep using same context manager
        autocast = lambda: torch.autocast(device_type="cuda", dtype=torch.float16) ## Better way to create new context manager, whenever needed
    elif config.precision == "torch.bfloat16":
        #autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        autocast = lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    else:
        #autocast = nullcontext()
        autocast = lambda:nullcontext()

    if config.mode not in ["F", "FB", "FBO"]:
        raise ValueError("mode must be one of 'F', 'FB', or 'FBO'")

    grad_context = torch.no_grad if config.mode == "F" else nullcontext

    device = config.device if config.device is not None else ("cuda" if torch.cuda.is_available() else "cpu")

    data = torch.randint(high=config.vocab_size, size=(config.batch_size, config.context_length+1), 
                           dtype=torch.long, device =device)

    x = data[:, :-1]
    y = data[:,1:]


    model = BasicsTransformerLM(config.vocab_size, config.context_length,config.d_model,
                              config.num_layers,config.num_heads,config.d_ff,
                              config.rope_theta,device=device,dtype=torch.float32).to(device)
    
    optimizer = AdamW(model.parameters(), lr=config.lr_max, weight_decay=config.weight_decay,
                       betas=(config.beta1, config.beta2),eps=config.eps)

    

  
    for _ in range(config.warmup):
        with grad_context():
            with autocast():
                logits = model(x)
            if config.mode == "F":
                continue
            with autocast():
                loss = cross_entropy(logits.reshape(-1, logits.size(-1)),y.reshape(-1)) ### -> [B*T,V], [B*T]
            optimizer.zero_grad() ## Optimizer and backward use same dtypes as produced in forward pass, so no autocasting
            loss.backward() ## Also optimizer and backward produce and use gradients, so no autocasting on them
            if config.mode == "FB":
                continue
            optimizer.step()

    times =[]
    if "cuda" in device:
        torch.cuda.synchronize()

    
    for _ in range(config.measure_iters):
        if "cuda" in device:
            torch.cuda.synchronize()
        with grad_context():
            start = default_timer()

            with autocast():
                logits = model(x)

            if config.mode=="F":
                if "cuda" in device:
                    torch.cuda.synchronize()
                end = default_timer()
                times.append(end-start)
                continue
            with autocast():
                loss = cross_entropy(logits.reshape(-1, logits.size(-1)),y.reshape(-1)) ### -> [B*T,V], [B*T]
            optimizer.zero_grad()
            loss.backward()
            if config.mode == "FB":
                if "cuda" in device:
                    torch.cuda.synchronize()
                end = default_timer()
                times.append(end-start)
                continue
            optimizer.step()
        
            if config.mode == "FBO":
                if "cuda" in device:
                    torch.cuda.synchronize()
                end = default_timer()
                times.append(end-start)

    df = append_benchmark_result(config, device, times)
    print(df.tail())
    df.to_csv(Path(config.out_dir) / "benchmark_history.csv", index=False)
    print(f"mean time = {pd.Series(times).mean():.6f} s")
    print(f"std time = {pd.Series(times).std():.6f} s")


if __name__ == "__main__":
    main()



