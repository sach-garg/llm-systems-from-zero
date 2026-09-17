from llm_systems.model import BasicsTransformerLM
from llm_systems.optimizer import AdamW
from llm_systems.nn_utils import cross_entropy
from llm_systems.benchmark_config import BenchmarkConfig
from pathlib import Path

import torch
import argparse
import os
import torch.distributed as dist
import torch.multiprocessing as mp
from timeit import default_timer
import pandas as pd


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



class DDP(torch.nn.Module):
    def __init__(self,module:torch.nn.Module):
        super().__init__()
        self.module = module
        self.world_size = dist.get_world_size() ### for dividing gradient before using all reduce
        self.pending_allreduce=[]

        with torch.no_grad(): ### Aligning initial weights across all ranks
            for parameter in self.module.parameters():
                dist.broadcast(parameter,src=0)

        for parameter in self.module.parameters():
            if parameter.requires_grad:
                parameter.register_post_accumulate_grad_hook(self.send_gradient) ### automatially calls to self.send_gradient(parameter) when parameter's gradient is computed

    def send_gradient(self,parameter):
        if parameter.grad is None:
            return
        else:
            parameter.grad.div_(self.world_size)
            queued_status= dist.all_reduce(parameter.grad,op=dist.ReduceOp.SUM,async_op=True)
            self.pending_allreduce.append(queued_status)
        return

    def forward(self,*inputs,**kwargs):
        return self.module(*inputs,**kwargs)

    def finish_gradient_synchronization(self):
        for queued_status in self.pending_allreduce:
            queued_status.wait()
        self.pending_allreduce.clear()
        return



# With async_op=False, this will be the order sequence:
# Find p2 gradient -> call hook for p2 -> call all_reduce for p2 -> wait for all_reduce to be queued 
# ->return from hook -> find p1 gradient -> call hook for p1 -> call all reduce for p1 -> wait for to be queued -> return back

# With async_op=True, this will be the order sequence (the wait removes):
# Find p2 gradient -> call hook for p2 -> call all_reduce for p2 ->  
# ->return from hook -> find p1 gradient -> call hook for p1 -> call all reduce for p1 -> return back

# The benefits in asynv_op=True is that backward for p1 continues even before p2's all reduce has been queued
# However, before taking optimizer step, we need to make sure p1 and p2 all_reduces have been queued, that's why we store queued_status and make sure all are queued before calling optimizer.step
# queued_status.wait() ensures the gradient is all reduces for subsequent CUDA opeartion on the gradient






def Overlapping_DDP_training(rank,world_size,config,result_queue):
    setup(rank,world_size)
    base_model = BasicsTransformerLM(config.vocab_size, config.context_length,config.d_model,
                                    config.num_layers,config.num_heads,config.d_ff,
                                    config.rope_theta,device=f"cuda:{rank}",dtype=torch.float32)

    model = DDP(base_model)

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
        model.finish_gradient_synchronization()
        optimizer.step()

    torch.cuda.synchronize() ## Finish warm up
    dist.barrier() ### Make sure no process proceeds to recorded iterations sooner than others
    torch.cuda.synchronize() ### Synchronize because the NCCL barrier itself may use the GPU. 
    iteration_times=[]

    for _ in range(config.measure_iters):
        start = default_timer()
        logits = model(x)
        loss = cross_entropy(logits.reshape(-1, logits.size(-1)),y.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        model.finish_gradient_synchronization()
        optimizer.step()
        torch.cuda.synchronize() ## make sure optimization step is over
        end = default_timer()
        iteration_times.append(end-start)

    torch.cuda.synchronize() ### make sure all iterations have run
    iteration_times = torch.tensor(iteration_times,device=f"cuda:{rank}",dtype=torch.float32)
    
    gather_iteration_times = [torch.empty_like(iteration_times) for _ in range(world_size)]
    dist.all_gather(gather_iteration_times,iteration_times)
    torch.cuda.synchronize()
    
    if rank==0:
        final_iteration_times = torch.stack(gather_iteration_times).cpu()
        result_queue.put({"iteration_times":final_iteration_times.tolist()})
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
    mp.spawn(fn=Overlapping_DDP_training,args=(world_size,config,result_queue),nprocs=world_size,join=True)
    results = result_queue.get()
    final_iteration_times = results["iteration_times"]
    rows = [{"rank": r, "iteration": i, "iteration_time": final_iteration_times[r][i]} for r in range(world_size) for i in range(config.measure_iters)]
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "Overlap_DDP_benchmark.csv", index=False)
    return

if __name__ == "__main__":
    main()









    
    

    
 

        

