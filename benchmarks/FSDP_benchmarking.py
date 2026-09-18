from llm_systems.model import BasicsTransformerLM
from llm_systems.optimizer import AdamW
from llm_systems.nn_utils import cross_entropy
from llm_systems.benchmark_config import BenchmarkConfig
from llm_systems.FSDP import FSDP
from llm_systems.FSDP_mem_efficient import FSDP_mem_efficient

import torch
import argparse
import os
import torch.distributed as dist
import torch.multiprocessing as mp
from timeit import default_timer
import pandas as pd
from pathlib import Path


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
    #parser.add_argument("--mode", type=str, choices=["F", "FB", "FBO"])
    parser.add_argument("--precision",type=str,choices = ["torch.float32","torch.float16","torch.bfloat16"])
    parser.add_argument("--out_dir", type=str)
    parser.add_argument("--FSDP_mem_efficient", action="store_true")
    return parser.parse_args()


def BuildConfig():
    config = BenchmarkConfig()
    args = parse_args()
    for key, value in vars(args).items():
        if value is not None:
            setattr(config, key, value)
    return config

def setup(rank,world_size):
  os.environ["MASTER_ADDR"] = "localhost"
  os.environ["MASTER_PORT"] = "29501"
  device = torch.device(f"cuda:{rank}")
  torch.cuda.set_device(device)
  dist.init_process_group(backend="nccl",rank=rank,world_size=world_size,device_id=device) ## "nccl" for GPUs





def FSDP_training(rank,world_size,config,result_queue):
    setup(rank,world_size)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(f"cuda:{rank}")

    base_model = BasicsTransformerLM(config.vocab_size, config.context_length,config.d_model,
                                    config.num_layers,config.num_heads,config.d_ff,
                                    config.rope_theta,device=f"cuda:{rank}",dtype=torch.float32)

    dtypes = {"torch.float32": None, "torch.float16":torch.float16, "torch.bfloat16":torch.bfloat16}

    if config.FSDP_mem_efficient:
        model = FSDP_mem_efficient(base_model,dtypes[config.precision])
    else:
        model = FSDP(base_model,dtypes[config.precision])

    ### Record the memory right after model initialization
    torch.cuda.synchronize()
    
    mem_after_init = torch.cuda.memory_allocated()/(1024**2)
    pmem_after_init =  torch.cuda.max_memory_allocated()/(1024**2)
    torch.cuda.reset_peak_memory_stats(f"cuda:{rank}")

    optimizer = AdamW(model.parameters(), lr=config.lr_max, weight_decay=config.weight_decay,
                      betas=(config.beta1, config.beta2),eps=config.eps)
    
    data = torch.randint(high=config.vocab_size, size=(config.batch_size, config.context_length+1),
                        dtype=torch.long, device =f"cuda:{rank}")
    
            
    x = data[:, :-1]
    y = data[:,1:]

    
    torch.cuda.synchronize()
    mem_after_data = torch.cuda.memory_allocated()/(1024**2)
    pmem_after_data =  torch.cuda.max_memory_allocated()/(1024**2)
    torch.cuda.reset_peak_memory_stats(f"cuda:{rank}")


    #### Now memory profile iteration
    optimizer.zero_grad()
    logits = model(x)
    loss = cross_entropy(logits.reshape(-1, logits.size(-1)),y.reshape(-1))
    torch.cuda.synchronize()
    mem_after_forward = torch.cuda.memory_allocated()/(1024**2)
    pmem_after_forward =  torch.cuda.max_memory_allocated()/(1024**2)
    torch.cuda.reset_peak_memory_stats(f"cuda:{rank}")

    
    loss.backward()
    model.finish_gradient_synchronization()
    torch.cuda.synchronize()
    mem_after_backward = torch.cuda.memory_allocated()/(1024**2)
    pmem_after_backward =  torch.cuda.max_memory_allocated()/(1024**2)
    torch.cuda.reset_peak_memory_stats(f"cuda:{rank}")

    optimizer.step()
    torch.cuda.synchronize()
    mem_after_step = torch.cuda.memory_allocated()/(1024**2)
    pmem_after_step =  torch.cuda.max_memory_allocated()/(1024**2)
    torch.cuda.reset_peak_memory_stats(f"cuda:{rank}")

    


    for _ in range(config.warmup):
        optimizer.zero_grad()
        logits = model(x)
        loss = cross_entropy(logits.reshape(-1, logits.size(-1)),y.reshape(-1))
        
        loss.backward()
        model.finish_gradient_synchronization()
        optimizer.step()

    torch.cuda.synchronize() ## Finish warm up
    dist.barrier() ### Make sure no process proceeds to recorded iterations sooner than others
    torch.cuda.synchronize() ### Synchronize because the NCCL barrier itself may use the GPU. 
    iteration_times=[]


    for _ in range(config.measure_iters):

        start = default_timer()
        optimizer.zero_grad()
        logits = model(x)
        loss = cross_entropy(logits.reshape(-1, logits.size(-1)),y.reshape(-1))
        
        loss.backward()
        model.finish_gradient_synchronization()
        optimizer.step()
        torch.cuda.synchronize() ## make sure optimization step is over
        end = default_timer()
        iteration_times.append(end-start)

    torch.cuda.synchronize() ### make sure all iterations have run
    iteration_times = torch.tensor(iteration_times,device=f"cuda:{rank}",dtype=torch.float32)
    gather_iteration_times = [torch.empty_like(iteration_times) for _ in range(world_size)]

    memory_stats =torch.tensor([mem_after_init,pmem_after_init,mem_after_data,pmem_after_data, mem_after_forward,pmem_after_forward,
                                mem_after_backward,pmem_after_backward,mem_after_step,pmem_after_step],dtype=torch.float64,device=f"cuda:{rank}")

    gather_memory_stats = [torch.empty_like(memory_stats) for _ in range(world_size)]
    
    
    dist.all_gather(gather_iteration_times,iteration_times)
    dist.all_gather(gather_memory_stats,memory_stats)
    torch.cuda.synchronize()
    
    if rank==0:
        final_iteration_times = torch.stack(gather_iteration_times).cpu()
        final_memory_stats = torch.stack(gather_memory_stats).cpu()
        result_queue.put({"iteration_times":final_iteration_times.tolist(), "memory_stats":final_memory_stats.tolist()})
    dist.barrier()
    torch.cuda.synchronize()
    dist.destroy_process_group()
    return




def main():
    config = BuildConfig()
    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    world_size=2
    spawn_context = mp.get_context("spawn")
    result_queue = spawn_context.SimpleQueue()
    mp.spawn(fn=FSDP_training,args=(world_size,config,result_queue),nprocs=world_size,join=True)
    results = result_queue.get()
    final_iteration_times = results["iteration_times"]
    final_memory_stats = results["memory_stats"]
    rows = [{"rank": r, "iteration": i, "iteration_time": final_iteration_times[r][i]} for r in range(world_size) for i in range(config.measure_iters)]

    mem_rows = [{"rank":r, "mem_after_init": final_memory_stats[r][0], "pmem_after_init": final_memory_stats[r][1],
                    "mem_after_data": final_memory_stats[r][2], "pmem_after_data": final_memory_stats[r][3],
                    "mem_after_fwd": final_memory_stats[r][4], "pmem_after_fwd": final_memory_stats[r][5],
                    "mem_after_bkwd": final_memory_stats[r][6], "pmem_after_bkwd": final_memory_stats[r][7],
                    "mem_after_step": final_memory_stats[r][8], "pmem_after_step": final_memory_stats[r][9]} for r in range(world_size)]
    df = pd.DataFrame(rows)
    df_mem =  pd.DataFrame(mem_rows)
    df.to_csv(out_dir / f"Timing_FSDP_mem_efficient_{config.FSDP_mem_efficient}.csv", index=False)
    df_mem.to_csv(out_dir / f"Memory_FSDP_mem_efficient_{config.FSDP_mem_efficient}.csv", index=False)
    return

if __name__ == "__main__":
    main()









    
    

    
 

        

