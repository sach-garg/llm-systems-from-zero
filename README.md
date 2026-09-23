# LLM Systems from Zero

A hands-on project for learning **systems-level optimizations for large language models**.

The goal is to understand how implementation choices affect training speed, GPU memory usage, and communication overhead—from single-GPU execution to distributed training.

Built while studying Stanford CS336, this repository includes implementations and experiments covering mixed precision, activation checkpointing, compilation, Triton FlashAttention, distributed data parallelism, optimizer-state sharding, and fully sharded training.

The focus is on implementing these techniques, profiling their behavior, and understanding their trade-offs.

## What’s Implemented

### Model and Training Fundamentals

- Transformer language model with custom linear and embedding layers.
- Rotary positional embeddings, RMSNorm, and feed-forward blocks.
- AdamW optimizer and training utilities.
- Forward-only, forward–backward, and full training-step benchmarks.

### Single-GPU Optimizations and Profiling

- FP32 and BF16 mixed-precision benchmarking.
- NVIDIA Nsight Systems profiling with NVTX annotations.
- GPU memory snapshots and peak-memory measurements.
- Activation checkpointing with configurable chunk sizes.
- Eager versus `torch.compile` attention benchmarks.
- Triton FlashAttention-2 forward and backward implementations.
- Query/key tile-size sweeps across sequence lengths and precisions.

### Distributed Training

- Naïve DDP with per-parameter gradient all-reduce.
- Flattened-gradient all-reduce.
- Asynchronous gradient communication overlapping with backward computation.
- Optimizer-state sharding and parameter synchronization.
- Custom FSDP for this repository’s model layers.
- A memory-efficient FSDP variant with custom linear autograd and configurable gradient-buffer cleanup.

These are educational implementations, not drop-in replacements for production distributed-training libraries.

## Results and Report

A detailed report with benchmark results, profiling analysis, plots, and implementation trade-offs is coming shortly.

The report will cover single-GPU experiments on an NVIDIA B200 and distributed-training experiments on two NVIDIA H200 SXM GPUs.

## Repository Structure

```text
.
├── benchmarks/
│   ├── end_to_end.py                 # Full-model timing
│   ├── profiler.py                   # NVTX and memory profiling
│   ├── checkpointing.py              # Checkpoint chunk-size experiments
│   ├── attention_profiler.py         # Eager vs compiled attention
│   ├── flash_attention_benchmarking.py
│   ├── DDP.py                        # Individual vs flattened all-reduce
│   ├── OverlapDDP.py                  # Communication/computation overlap
│   ├── OptShard_DDP.py                # Optimizer-state sharding
│   └── FSDP_benchmarking.py           # Baseline and memory-efficient FSDP
├── src/llm_systems/
│   ├── model.py
│   ├── optimizer.py
│   ├── nn_utils.py
│   ├── data.py
│   ├── benchmark_config.py
│   ├── flashattention.py
│   ├── OptimizerSharding.py
│   ├── FSDP.py
│   └── FSDP_mem_efficient.py
├── docs/
├── results/
└── pyproject.toml
```

## Experimental Hardware

| Experiment Family | Hardware |
|---|---|
| Single-GPU timing, profiling, and attention | 1 × NVIDIA B200 |
| Distributed training | 2 × NVIDIA H200 SXM |

Experiments were run on RunPod. Results from different GPU families are not treated as direct single-GPU versus multi-GPU scaling comparisons.

## Setup

Use a Linux CUDA environment with a compatible NVIDIA driver and CUDA-enabled PyTorch installation. Python 3.11 or newer is required.

```bash
git clone https://github.com/sach-garg/llm-systems-from-zero.git
cd llm-systems-from-zero

python -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -e ".[benchmark,gpu]"
```

NVIDIA Nsight Systems is an additional requirement for collecting Nsight traces; it is not installed by the Python package.

## Running Benchmarks

Run commands from the repository root. The examples below use small configurations to get started; they do not reproduce the XL experiments.

### Single-GPU Training Step

```bash
python benchmarks/end_to_end.py \
  --d_model 64 \
  --d_ff 256 \
  --num_layers 2 \
  --num_heads 4 \
  --context_length 128 \
  --batch_size 2 \
  --mode FBO \
  --precision torch.float32 \
  --device cuda \
  --warmup 5 \
  --measure_iters 10 \
  --out_dir results/smoke_single_gpu
```

Supported modes:

| Mode | Workload |
|---|---|
| `F` | Forward only, without gradient tracking |
| `FB` | Forward, loss, and backward |
| `FBO` | Forward, loss, backward, and optimizer update |

### FlashAttention Tile Sweep

```bash
python -u benchmarks/flash_attention_benchmarking.py
```

The script benchmarks:

- Batch size 1 and one attention head.
- Causal attention.
- Sequence lengths from 128 through 65,536.
- Dimensions 16, 32, 64, and 128.
- BF16 and FP32.
- Query and key tile sizes from `{16, 32, 64, 128}`.
- Forward, backward, and combined latency using `triton.testing.do_bench`.

Results are written incrementally to:

```text
results/flash_attention_benchmark.csv
```

The full sweep can take substantial time and includes configurations that exceed kernel resource limits. Back up an existing results file before rerunning.

### Naïve DDP

Requires two visible CUDA GPUs. The script launches its own worker processes; do not wrap it in `torchrun`.

```bash
python benchmarks/DDP.py \
  --d_model 64 \
  --d_ff 256 \
  --num_layers 2 \
  --num_heads 4 \
  --context_length 128 \
  --batch_size 2 \
  --warmup 5 \
  --measure_iters 10 \
  --out_dir results/smoke_ddp
```

Add `--flatten_grads` to compare against flattened-gradient communication. Batch size is per rank in this benchmark.

### Memory-Efficient FSDP

Requires two visible CUDA GPUs and launches its own worker processes.

```bash
python benchmarks/FSDP_benchmarking.py \
  --d_model 64 \
  --d_ff 256 \
  --num_layers 2 \
  --num_heads 4 \
  --context_length 128 \
  --batch_size 2 \
  --precision torch.float32 \
  --warmup 5 \
  --measure_iters 10 \
  --FSDP_mem_efficient \
  --min_cleanup_distance 0 \
  --max_cleanup_distance 5 \
  --out_dir results/smoke_fsdp
```

Cleanup distance 0 disables early gradient-buffer cleanup within the memory-efficient variant; it does not switch to baseline FSDP.

For scripts exposing a command-line interface, use `--help` to inspect available arguments.

## Interpreting Measurements

- Check units: FlashAttention timings are in milliseconds; several other benchmark CSVs record seconds.
- Memory allocated before backward is not the same as peak memory over a complete training step.
- Summed CUDA kernel durations are not the same as elapsed iteration time.
- Compare identical workloads, precision, batch sizes, and compilation settings.
- Record OOM and resource failures rather than replacing missing timings with zero.
- The FlashAttention CSV label `Triton OOM` represents caught `triton.OutOfResources` errors, not necessarily exhausted GPU HBM.
- Best measured tile configurations are selected from a finite sweep, not proven globally optimal.

## Ongoing Work

- Publish the single-GPU and distributed-training analysis with plots.
- Complete the LaTeX report.
- Compare full-model eager and compiled execution.
- Integrate optimizations incrementally into an XL-sized model.
- Measure end-to-end throughput, iteration time, and peak memory—including FlashAttention memory savings.

## Acknowledgments

This project was developed while studying Stanford CS336, *Language Modeling from Scratch*, particularly its systems assignments. It uses PyTorch, Triton, and NVIDIA profiling tools.
