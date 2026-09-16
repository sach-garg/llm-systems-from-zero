from dataclasses import dataclass


@dataclass
class BenchmarkConfig:
    vocab_size: int = 10000

    # model
    context_length: int = 32
    d_model: int = 64
    num_layers: int = 2
    num_heads: int = 4
    d_ff: int = int(0.66*4*64)
    rope_theta: float = 10000.0

    # training
    batch_size: int = 8
    measure_iters: int = 100
    warmup: int = 0 ### warmup iterations
    mode:str = "FBO" ## either "F - for forward" or "FB - forward + backward", "FBO - forward + backward + optimization"

    # optimizer
    lr_max: float = 3e-4
    lr_min: float = 3e-5
    cosine_cycle_iters: int = 50
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8

    # system
    device: str | None = None
    precision: str = "torch.float32" ## Precision for autocasting
    out_dir: str ="results"
    profile_memory:bool=False
    memory_file:str = "memory_snapshot.pkl"

    ## Activation Checkpointing and operator fusion
    operator_fusion:bool=False
    chunk_size:int | None=None

    


