from __future__ import annotations

import math
from collections.abc import Callable, Iterable

import torch

def get_cosine_lr(it:int, 
                     max_learning_rate: float,
                     min_learning_rate:float,
                     warmup_iters,
                     cosine_cycle_iters) -> float:
    if it < warmup_iters:
        return it*max_learning_rate/warmup_iters
    elif it > cosine_cycle_iters:
        return min_learning_rate

    else:
        temp = (it-warmup_iters)/(cosine_cycle_iters-warmup_iters)*math.pi
        return min_learning_rate + 0.5 *(1 + math.cos(temp) ) *(max_learning_rate - min_learning_rate)

    
class AdamW(torch.optim.Optimizer):
    def __init__(self,params:Iterable[torch.nn.parameter.Parameter],lr:float=1e-3,weight_decay:float=0.0,betas: tuple[float, float] = (0.9, 0.999),eps:float=1e-8):
        defaults = {"lr":lr,"betas":betas,"eps":eps,"weight_decay":weight_decay}
        super().__init__(params,defaults)
    
    @torch.no_grad()
    def step(self,closure: Callable | None=None):
        loss = None
        if closure is not None:  ### This if statement gets used for LBFGS type methods, but due to syntax and design requirement we need it
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta1 = group["betas"][0]
            beta2 = group["betas"][1]
            lr = group["lr"]
            eps = group["eps"]
            lamda = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                grad = p.grad
                t = state.get("t",1)
                m = beta1*state.get("momentum",torch.zeros_like(grad)) + (1-beta1)*grad
                v = beta2*state.get("v",torch.zeros_like(grad)) + (1-beta2)*(grad**2)
                update = m/(torch.sqrt(v)+eps)
                p-=(math.sqrt(1-beta2**t))/(1-beta1**t) *lr*update 
                p-=lr*lamda*p   
                state["momentum"] = m
                state["v"] = v
                state["t"] = t+1
        return loss
    

