from llm_systems.model import BasicsTransformerLM
from llm_systems.optimizer import AdamW
from llm_systems.nn_utils import cross_entropy
from llm_systems.benchmark_config import BenchmarkConfig

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
    parser.add_argument("--out_dir", type=str)
    parser.add_argument("--flatten_grads",action="store_true")
    #parser.add_argument("--precision",type=str,choices = ["torch.float32","torch.float16","torch.bfloat16"])
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



def NaiveDDP_training(rank,world_size,config,result_queue,flatten_grads):
    setup(rank,world_size)

    model = BasicsTransformerLM(config.vocab_size, config.context_length,config.d_model,
                                config.num_layers,config.num_heads,config.d_ff,
                                config.rope_theta,device=f"cuda:{rank}",dtype=torch.float32)

    with torch.no_grad(): 
        for parameter in model.parameters():
            dist.broadcast(parameter,src=0) ### if rank 1 reaches before rank=0, then ## rank-1 will wait for rank-0 to reach here and transfer its parameters

   
        
    optimizer = AdamW(model.parameters(), lr=config.lr_max, weight_decay=config.weight_decay,
                     betas=(config.beta1, config.beta2),eps=config.eps)

    data = torch.randint(high=config.vocab_size, size=(config.batch_size, config.context_length+1), 
                                   dtype=torch.long, device =f"cuda:{rank}")

        
    x = data[:, :-1]
    y = data[:,1:]
    for _ in range(config.warmup):
        logits = model(x)
        loss = cross_entropy(logits.reshape(-1, logits.size(-1)),y.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        if not flatten_grads:
            with torch.no_grad():
                for parameter in model.parameters():
                    if parameter.grad is None:
                        continue
                    dist.all_reduce(parameter.grad,op=dist.ReduceOp.SUM,async_op=False) ### all ranks will wait till all ranks reach here
                    parameter.grad.div_(world_size)
        else:
            parameter_with_grad = [p for p in model.parameters() if p.grad is not None]
            flat_grads = torch._utils._flatten_dense_tensors([p.grad for p in parameter_with_grad])
            dist.all_reduce(flat_grads,op=dist.ReduceOp.SUM,async_op=False)
            flat_grads.div_(world_size)
            grads = torch._utils._unflatten_dense_tensors(flat_grads,parameter_with_grad)
            for p,g in zip(parameter_with_grad,grads):
                p.grad.copy_(g)
            del flat_grads, grads, g ### suggested by chatgpt

        optimizer.step()

    torch.cuda.synchronize() ## Finish warm up
    dist.barrier() ### Make sure no process proceeds to recorded iterations sooner than others
    torch.cuda.synchronize() ### Synchronize because the NCCL barrier itself may use the GPU. 

    iteration_times=[]
    comm_times=[]

    

    for _ in range(config.measure_iters):
        it_start_time = default_timer()
        logits = model(x)
        loss = cross_entropy(logits.reshape(-1, logits.size(-1)),y.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        torch.cuda.synchronize() ### start recording communication time only once backward is finished
        comm_start_time = default_timer()
        if not flatten_grads:
             with torch.no_grad():
                for parameter in model.parameters():
                    if parameter.grad is None:
                        continue
                    dist.all_reduce(parameter.grad,op=dist.ReduceOp.SUM,async_op=False) ### all ranks will wait till all ranks reach here
                    parameter.grad.div_(world_size)
        else:
            parameter_with_grad = [p for p in model.parameters() if p.grad is not None]
            flat_grads = torch._utils._flatten_dense_tensors([p.grad for p in parameter_with_grad])
            dist.all_reduce(flat_grads,op=dist.ReduceOp.SUM,async_op=False)
            flat_grads.div_(world_size)
            grads = torch._utils._unflatten_dense_tensors(flat_grads,parameter_with_grad)
            for p,g in zip(parameter_with_grad,grads):
                p.grad.copy_(g)
            del flat_grads, grads, g ## suggested by chatgpt
  
        torch.cuda.synchronize() ### wait for communication to finish before stop recording communication time
        comm_end_time = default_timer()
        optimizer.step()
        torch.cuda.synchronize() ### wait for optimization to finish before stop recording iteration time
        it_end_time = default_timer()
        iteration_times.append(it_end_time-it_start_time)
        comm_times.append(comm_end_time-comm_start_time)


    torch.cuda.synchronize() ### make sure all iterations have run
    iteration_times = torch.tensor(iteration_times,device=f"cuda:{rank}",dtype=torch.float32)
    comm_times = torch.tensor(comm_times,device=f"cuda:{rank}",dtype=torch.float32)

    gather_iteration_times = [torch.empty_like(iteration_times) for _ in range(world_size)]
    gather_comm_times =  [torch.empty_like(comm_times) for _ in range(world_size)]
    dist.all_gather(gather_iteration_times,iteration_times)
    dist.all_gather(gather_comm_times,comm_times)
    torch.cuda.synchronize()

    if rank==0:
        final_iteration_times = torch.stack(gather_iteration_times).cpu()
        final_comm_times = torch.stack(gather_comm_times).cpu()
        result_queue.put({"iteration_times":final_iteration_times.tolist(), "comm_times": final_comm_times.tolist()})
  
    dist.barrier()
    torch.cuda.synchronize()
    dist.destroy_process_group()
    return 


def main():
    config = BuildConfig()
    flat_grads = config.flatten_grads

    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    world_size=2
    spawn_context = mp.get_context("spawn")
    result_queue = spawn_context.SimpleQueue()
    mp.spawn(fn=NaiveDDP_training,args=(world_size,config,result_queue,flat_grads),nprocs=world_size,join=True)
    results = result_queue.get()
    final_iteration_times = results["iteration_times"]
    final_comm_times = results["comm_times"]
    rows = [{"rank": r, "iteration": i, "iteration_time": final_iteration_times[r][i], "comm_time": final_comm_times[r][i]} for r in range(world_size) for i in range(config.measure_iters)]
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / f"naive_ddp_benchmark_flat_{flat_grads}.csv", index=False)
    return

if __name__ == "__main__":
    main()




        

        

        

    




        


        

