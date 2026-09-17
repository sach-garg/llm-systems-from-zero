from pathlib import Path

import pandas as pd
import torch
import triton
from triton.testing import do_bench

from llm_systems.flashattention import flash_attention
from llm_systems.model import scaled_dot_product_attention


def benchmark(T, D, dtype, implementation, row, Q_TILE, K_TILE):
    Q, K, V = [torch.randn(1, 1, T, D,device="cuda",dtype=dtype,requires_grad=True,) for _ in range(3)] ###(B,h,T,C)
    dO = torch.randn_like(Q)

    if implementation == "triton":
        def forward():
            return flash_attention(Q, K, V, Q_TILE = Q_TILE, K_TILE = K_TILE , is_causal = True) 
    else:
        mask = torch.ones(T, T, device="cuda", dtype=torch.bool).tril()

        def forward():
            return scaled_dot_product_attention(Q, K, V, mask)

    row["forward_ms"] = do_bench(forward, return_mode="mean") #### runs forward multiple times and stores mean time in row["forward_ms"]

    # Build one graph outside timing and reuse it for backward measurements.
    O = forward()

    def backward():
        O.backward(dO, retain_graph=True)

    row["backward_ms"] = do_bench(backward, grad_to_none=[Q, K, V],return_mode="mean") #### runs backward multiple times and stores mean time in row["backward_ms"]. 
    ### Because computational graph is destroyed after backward, we pass retain_graph = True in backward function

    #### Now benchmark combined forward and backward pass, no need to cretain graph here, because we will foreward and then backward and then repeat
    del O
    Q.grad = K.grad = V.grad = None

    def forward_backward():
        forward().backward(dO)

    row["forward_backward_ms"] = do_bench(forward_backward,grad_to_none=[Q, K, V],return_mode="mean")


def main():
    output_path = Path("results/flash_attention_benchmark.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    tile_sizes = [(q,k) for q in [16,32,64,128] for k in [16,32,64,128]]

    for dtype in [torch.bfloat16, torch.float32]:
        for D in [16, 32, 64, 128]:
            for T in [2**i for i in range(7, 17)]:
                for implementation in ["pytorch", "triton"]:
                    if implementation == "triton":
                        for (q_tile,k_tile) in tile_sizes:
                            row = {"implementation": implementation,"T": T, "D": D, "Q_TILE": q_tile, "K_TILE": k_tile, "dtype": str(dtype),"status": "OK"}
                            try:
                                benchmark(T, D, dtype, implementation, row,q_tile,k_tile)
                            except torch.OutOfMemoryError:
                                row["status"] = "Pytorch OOM"
                            except triton.OutOfResources:
                                row["status"] = "Triton OOM"
                            rows.append(row)
                            pd.DataFrame(rows).to_csv(output_path, index=False)
                            torch.cuda.empty_cache()
                    else:
                        q_tile=None
                        k_tile=None
                        row = {"implementation": implementation,"T": T, "D": D, "Q_TILE": q_tile, "K_TILE": k_tile, "dtype": str(dtype),"status": "OK"}
                        try:
                            benchmark(T, D, dtype, implementation, row,q_tile,k_tile)
                        except torch.OutOfMemoryError:
                            row["status"] = "Pytorch OOM"
                        except triton.OutOfResources:
                            row["status"] = "Triton OOM"
                        rows.append(row)
                        pd.DataFrame(rows).to_csv(output_path, index=False)
                        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()