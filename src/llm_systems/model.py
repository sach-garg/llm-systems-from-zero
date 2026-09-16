from __future__ import annotations

import logging
import math
import os

import torch.cuda.nvtx as nvtx

import torch
import torch.nn as nn

from torch import Tensor


from llm_systems.nn_utils import softmax

logger = logging.getLogger(__name__)




class Linear(nn.Module):
    def __init__(self, d_in:int, d_out:int, device=None, dtype = None):
        super().__init__()
        self.device = device if device is not None else "cpu"
        self.weight = nn.Parameter(torch.empty(d_out,d_in,device=self.device,dtype=dtype)) ### always define in (d_out, d_in format)
        self.sigma = math.sqrt(2/(d_in+d_out))
        nn.init.trunc_normal_(self.weight, mean=0., std=self.sigma, a=-3*self.sigma, b=3*self.sigma)

    def forward(self,x: torch.Tensor) -> torch.Tensor:
        return x@self.weight.transpose(0,1)

    def extra_repr(self):
            return f"d_out={self.weight.shape[0]}, d_in={self.weight.shape[1]}"

    

class Embedding(nn.Module):
    def __init__(self,vocab_size:int,d_model:int,device=None,dtype=None):
        super().__init__()
        self.device = device if device is not None else "cpu"
        self.weight = nn.Parameter(torch.empty(vocab_size,d_model,device = self.device, dtype = dtype))
        self.sigma = 1
        nn.init.trunc_normal_(self.weight, mean=0., std=self.sigma, a=-3*self.sigma, b=3*self.sigma)
    
    def forward(self, token_ids:torch.Tensor):
        return self.weight[token_ids] ## (B,T) -> (B,T,C)

    def extra_repr(self):
            return f"vocab_size={self.weight.shape[0]}, d={self.weight.shape[1]}"



class RotaryEmbedding(nn.Module):
    def __init__(self, context_length:int, dim:int, theta: float =10000, device=None):
        super().__init__()
        if dim%2!=0:
            raise ValueError("RoPE is defined only for even dimensions")
        self.device = device if device is not None else "cpu"
        self.theta = theta
        theta_dim = torch.tensor([1/theta**(2*(k-1)/dim) for k in range(1,int(dim/2)+1)],device=self.device) #<- [D/2]
        pos = torch.arange(context_length,device=self.device) #<-[T]
        theta_pos_dim = torch.outer(pos,theta_dim) #<-[T,D/2]
        cos = theta_pos_dim.cos()
        sin = theta_pos_dim.sin()
        
        self.register_buffer('cos',cos) # [T,D/2]
        self.register_buffer('sin',sin)  # [T,D/2]
                
        self.register_buffer('theta_dim',theta_dim)


    def forward(self,x:torch.Tensor,pos_ids:torch.Tensor | None=None): ### Parameter could be K also Q:Query, K: Key
        
        T = x.shape[-2]
        x_even = x[...,0::2] ### a view so no memory cost
        x_odd = x[...,1::2] ### a view so no memory cost
        out = torch.empty_like(x)

      
        if pos_ids is not None: ## for tensors just writing if TP doesn't work
            theta_pos_dim1 = pos_ids.unsqueeze(-1) * self.theta_dim ### [... T] -->(unsqueeze) [... T,1] ---> (*) [... T, dh/2]
            cos1 =  theta_pos_dim1.cos()
            sin1 = theta_pos_dim1.sin()
            out[..., 0::2] = x_even*cos1 - x_odd*sin1
            out[...,1::2] = x_even*sin1 + x_odd*cos1
            return out
      

        out[..., 0::2] = x_even*self.cos[:T] - x_odd*self.sin[:T]
        out[...,1::2] = x_even*self.sin[:T] + x_odd*self.cos[:T]
        return out
    
    def extra_repr(self):
            return f"context_length={self.cos.shape[0]}, dim/2={self.cos.shape[1]}"


class RMSNorm(nn.Module): 
   
    def __init__(self,hidden_size:int,eps: float = 1e-5, device=None,dtype=None):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size,device=device,dtype=dtype)) ### not torch.empty or torch.zeros.
        self.eps = eps
        self.d_model = hidden_size
    
    def forward(self,x:torch.Tensor):  ### x is [B,T,C]
        in_dtype = x.dtype
        x = x.to(torch.float32)
        fac = torch.rsqrt(torch.mean(x**2,dim=-1,keepdim=True)+ self.eps) #### keeping the last dimension here for broadcasting, fac becomes [B,T,1]
        return (x*fac * self.weight).to(in_dtype)
    def extra_repr(self):
            return f"hidden_size={self.weight.shape[0]}, eps={self.eps}"





class BasicsTransformerLM(nn.Module):
    def __init__(self,vocab_size: int,context_length: int,d_model: int,num_layers: int,num_heads: int,d_ff: int,
        rope_theta: float | None = 10000.0,device=None,dtype=None):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by num_heads={num_heads}")

        self.context_length = context_length
        self.d_model = d_model
        self.device = device if device is not None else "cpu"
        self.dtype = dtype

        self.token_embeddings = Embedding(vocab_size, d_model, device=self.device, dtype=self.dtype)

        d_head = d_model // num_heads
        self.positional_encoder = (RotaryEmbedding(context_length, d_head, rope_theta, device=self.device)
            if rope_theta is not None else None)

        self.register_buffer('causal_mask',torch.tril(torch.ones(self.context_length,self.context_length,device = self.device,dtype=torch.bool)))
                

        self.layers = nn.ModuleList(
            [TransformerBlock(d_model=d_model,num_heads=num_heads,d_ff=d_ff,positional_encoder=self.positional_encoder,
                    device=self.device, dtype=self.dtype) for _ in range(num_layers)])

       

        self.ln_final = RMSNorm(hidden_size=d_model, device=self.device, dtype=self.dtype)
        self.lm_head = Linear(d_model, vocab_size, device=self.device, dtype=self.dtype)


    def forward(self, x: torch.Tensor,use_nvtx:bool=False) -> torch.Tensor:
        sequence_length = x.shape[-1]
        if sequence_length > self.context_length:
            raise ValueError(f"Sequence length exceeds context length {self.context_length}")

        x = self.token_embeddings(x)

        for layer in self.layers:
            x = layer(x,mask=self.causal_mask,use_nvtx = use_nvtx)

        x = self.ln_final(x)
        logits = self.lm_head(x)
        return logits
 


class TransformerBlock(nn.Module):
    def __init__(self,d_model: int, num_heads: int, d_ff: int,positional_encoder: RotaryEmbedding | None = None,
                 device=None,dtype=None):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by num_heads={num_heads}")

        self.device = device if device is not None else "cpu"
        self.dtype = dtype

        self.ln1 = RMSNorm(hidden_size=d_model,device=self.device,dtype=self.dtype)

        self.attn = CausalMultiHeadSelfAttention(d_model=d_model,num_heads=num_heads,positional_encoder=positional_encoder,
            device=self.device,dtype=self.dtype)

        self.ln2 = RMSNorm(hidden_size=d_model,device=self.device,dtype=self.dtype)

        self.ffn = SwiGLU(d_model=d_model,d_ff=d_ff,device=self.device,dtype=self.dtype)


    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None,mask:torch.Tensor | None=None,use_nvtx:bool=False):
        if use_nvtx:
            with nvtx.range("RMSNorm"):
                y = self.ln1(x)
        else:
            y = self.ln1(x)
        if use_nvtx:
            with nvtx.range("Attention"):
                x = x + self.attn(y,token_positions,mask,use_nvtx)
        else:
            x = x + self.attn(y,token_positions,mask,use_nvtx)
        if use_nvtx:
            with nvtx.range("RMSNorm"):
                y = self.ln2(x)
        else:
            y = self.ln2(x)
        if use_nvtx:
            with nvtx.range("FFN"):
                x = x + self.ffn(y)
        else:
             x = x + self.ffn(y)
        return x

       





def silu(x:torch.Tensor): ### x-> [B,T,C]
    return x*torch.sigmoid(x) #### [B,T,C]



class SwiGLU(nn.Module):
    def __init__(self,d_model:int,d_ff:int,device=None,dtype=None):
        super().__init__()
        self.device = device if device is not None else "cpu"
        self.w1 = Linear(d_model,d_ff,device=self.device,dtype=dtype)
        self.w3 = Linear(d_model,d_ff,device=self.device,dtype=dtype)
        self.w2 = Linear(d_ff,d_model,device=self.device,dtype=dtype)
        

    def forward(self,x:torch.Tensor): ### x-> [B,T,C]
        return self.w2(silu(self.w1(x)) * self.w3(x))



def scaled_dot_product_attention(Q:torch.Tensor,K:torch.Tensor,V:torch.Tensor,mask=None,use_nvtx:bool=False):
    Tq,Cq = Q.shape[-2], Q.shape[-1]
    Tk,Ck = K.shape[-2], K.shape[-1] #### If not self-attention then Tq and Tk could be different, but Cq,Ck would be same

    Tv, Cv = V.shape[-2], V.shape[-1] ### Tv would be same as Tk

    if use_nvtx:
        with nvtx.range("QK matmul"):
            attention_matrix = Q@K.transpose(-1,-2)/math.sqrt(Cq)
    else:
        attention_matrix = Q@K.transpose(-1,-2)/math.sqrt(Cq)

    if mask is not None:
        masked_attention = attention_matrix.masked_fill(~mask, float("-inf"))
    else:
        masked_attention = attention_matrix
    if use_nvtx:
        with nvtx.range("Softmax"):
            attention_scores = softmax(masked_attention,dim=-1)
    else:
        attention_scores = softmax(masked_attention,dim=-1)

    if use_nvtx:
        with nvtx.range("PV matmul"):
            output = attention_scores @ V
    else:
        output = attention_scores @ V



    return output




class CausalMultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        positional_encoder: RotaryEmbedding | None = None,
        device=None,
        dtype=None,
    ):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by num_heads={num_heads}")

        self.d_model = d_model
        self.num_heads = num_heads
        self.positional_encoder = positional_encoder
        self.device = device if device is not None else "cpu"

        self.q_proj = Linear(d_model, d_model, device=self.device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=self.device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=self.device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=self.device, dtype=dtype)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None,mask: torch.Tensor | None=None, use_nvtx:bool=False) -> torch.Tensor:
        C = x.shape[-1]
        T = x.shape[-2]

        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        d_head = self.d_model//self.num_heads


        Q = Q.view(*Q.shape[:-1],self.num_heads,d_head).transpose(-3,-2)
        K = K.view(*K.shape[:-1],self.num_heads,d_head).transpose(-3,-2)
        V = V.view(*V.shape[:-1],self.num_heads,d_head).transpose(-3,-2)



        if self.positional_encoder is not None:
            if token_positions is not None:
                token_positions = token_positions.unsqueeze(-2)  # [..., 1, seq]
            Q = self.positional_encoder(Q, token_positions)
            K = self.positional_encoder(K, token_positions)

        if mask is None:
            mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
        else:
            mask = mask[:T,:T]
        attention = scaled_dot_product_attention(Q, K, V, mask,use_nvtx)

        *batch_dims, h, T, d_head = attention.shape
        attn_out = attention.transpose(-3, -2).reshape(*batch_dims, T, h * d_head)

        return self.output_proj(attn_out)



