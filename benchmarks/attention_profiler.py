from llm_systems.model import scaled_dot_product_attention
import torch
from timeit import default_timer
import pandas as pd
import argparse
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir",type=str,default="results")
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = out_dir / "attention_benchmark.csv"
  
    B = 8
    warmup=2
    profile_iterations=100
    stats={}
    device = "cuda"

    for operator_fuse in [True,False]:

        if operator_fuse:
            #scaled_dot_product_attention  = torch.compile(scaled_dot_product_attention) ## scoping issue comes UnboundLocalError: cannot access local variable 'scaled_dot_product_attention' where it is not associated with a value
            attn_func = torch.compile(scaled_dot_product_attention)
        else:
            attn_func = scaled_dot_product_attention
        
        
        for d_model in [16,32,64,128]:
            for T in [256, 1024, 4096, 8192, 16384]:
                print(f"Running d_model={d_model}, T={T}", flush=True)
                forward_times =[]
                backward_times =[]
                memory_before_backward=[]
                Q=None
                K=None
                V=None
                mask=None
                attn=None
                try:
                    Q = torch.randn(B,T,d_model,device=device, dtype=torch.float32,requires_grad=True)
                    K = torch.randn(B,T,d_model,device=device, dtype=torch.float32,requires_grad=True)
                    V = torch.randn(B,T,d_model,device=device, dtype=torch.float32,requires_grad=True)
                    mask = torch.tril(torch.ones(T,T,device=device,dtype=torch.bool))
                    for _ in range(warmup):
                        attn = attn_func(Q,K,V,mask,use_nvtx=False)
                        if Q.grad is not None: Q.grad = None
                        if K.grad is not None: K.grad = None
                        if V.grad is not None: V.grad = None
                        attn.sum().backward()

                    torch.cuda.synchronize()

                    for _ in range(profile_iterations):
                        if Q.grad is not None: Q.grad = None
                        if K.grad is not None: K.grad = None
                        if V.grad is not None: V.grad = None
                        start = default_timer()
                        attn = attn_func(Q,K,V,mask,use_nvtx=False)
                        torch.cuda.synchronize()
                        end = default_timer()
                        forward_times.append(end-start)

                        
                        m = torch.cuda.memory_allocated()/(1024**2)
                        memory_before_backward.append(m)
    
                        start = default_timer()
                        attn.sum().backward()
                        torch.cuda.synchronize()
                        end=default_timer()
                        backward_times.append(end-start)
                    stats[(d_model, T,operator_fuse,args.flash_attention)] = {
                                            "forward_mean": sum(forward_times) / len(forward_times),
                                            "forward_std": pd.Series(forward_times).std(),
                                            "backward_mean": sum(backward_times) / len(backward_times),
                                            "backward_std": pd.Series(backward_times).std(),
                                            "memory_before_backward_mean": sum(memory_before_backward) / len(memory_before_backward),
                                            "memory_before_backward_std": pd.Series(memory_before_backward).std()}
                
                except torch.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    stats[(d_model, T,operator_fuse)] = {"status":"OOM"}

                del Q, K, V,attn,mask
                torch.cuda.empty_cache()
    stats_df = pd.DataFrame.from_dict(stats, orient="index")
    stats_df.index = pd.MultiIndex.from_tuples(stats_df.index, names=["d_model", "T","compiled"])
    stats_df = stats_df.reset_index()
    stats_df.to_csv(output_path, index=False)
    return

if __name__ =="__main__":
    main()





