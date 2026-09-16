import torch
from typing import List




def softmax(logits:torch.Tensor,dim: int =-1):

    max_logits = torch.max(logits,dim=dim,keepdim=True).values ### max returns tuple (values,indices)
    stable_logits = torch.exp(logits - max_logits)
    normalizer = torch.sum(stable_logits,dim=dim,keepdim=True)
    return stable_logits/normalizer



def cross_entropy(logits:torch.Tensor,y:torch.Tensor): ### logits <-- [B,vocab_size] ### y <- [B]
    B = logits.shape[0]
    max_logits = torch.max(logits,dim=1,keepdim=True).values #### <--[B,1]
    shifted_logits = logits-max_logits #### ---> [B,V]
    normalizer = torch.sum(torch.exp(shifted_logits),dim=1) #### --> [B]
    return -torch.mean(shifted_logits[torch.arange(B,device=logits.device), y]- torch.log(normalizer))



def clip_gradient(parameters: List[torch.Tensor],max_norm:float) -> None:
    device = parameters[0].device
    dtype = parameters[0].dtype
    temp = torch.sqrt(torch.sum(torch.tensor([torch.linalg.norm(p.grad)**2 for p in parameters if p.grad is not None],device = device, dtype = dtype)))
    if temp <=   max_norm:
        return
    scale = max_norm/(temp + 1e-6)
    for p in parameters:
        if p.grad is not None:
            p.grad*=scale
    return
    
