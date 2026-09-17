import torch
import math
import triton
import triton.language as tl

@triton.jit
def FAT2_triton_fwd(Q_ptr,Q_batch_stride, Q_token_stride, Q_dim_stride,
                    K_ptr, K_batch_stride, K_token_stride, K_dim_stride,
                    V_ptr, V_batch_stride, V_token_stride, V_dim_stride,
                    O_ptr, O_batch_stride, O_token_stride, O_dim_stride,
                    L_ptr , L_batch_stride, L_token_stride,
                    TQ, TK,scale,
                    D:tl.constexpr,
                    Q_TILE_SIZE: tl.constexpr,
                    K_TILE_SIZE: tl.constexpr,
                    is_causal:tl.constexpr):

  MY_QTILE_ID = tl.program_id(0)
  MY_BATCH_ID = tl.program_id(1)



  Q_block_ptr = tl.make_block_ptr(Q_ptr + MY_BATCH_ID*Q_batch_stride,
                                  shape = (TQ,D),
                                  strides = (Q_token_stride,Q_dim_stride),
                                  offsets = (MY_QTILE_ID*Q_TILE_SIZE,0),
                                  block_shape = (Q_TILE_SIZE,D),
                                  order = (1,0))

  O_block_ptr = tl.make_block_ptr(O_ptr + MY_BATCH_ID*O_batch_stride,
                                  shape = (TQ,D),
                                  strides = (O_token_stride,O_dim_stride),
                                  offsets = (MY_QTILE_ID*Q_TILE_SIZE,0),
                                  block_shape = (Q_TILE_SIZE,D),
                                  order = (1,0))

  L_block_ptr = tl.make_block_ptr(L_ptr + MY_BATCH_ID*L_batch_stride,
                                  shape = (TQ,),
                                  strides = (L_token_stride,),
                                  offsets = (MY_QTILE_ID*Q_TILE_SIZE,),
                                  block_shape = (Q_TILE_SIZE,),
                                  order = (0,))

  K_block_ptr = tl.make_block_ptr(K_ptr + MY_BATCH_ID*K_batch_stride,
                                  shape = (TK,D),
                                  strides = (K_token_stride,K_dim_stride),
                                  offsets = (0,0),
                                  block_shape = (K_TILE_SIZE,D),
                                  order=(1,0))

  V_block_ptr = tl.make_block_ptr(V_ptr + MY_BATCH_ID*V_batch_stride,
                                  shape = (TK,D),
                                  strides = (V_token_stride,V_dim_stride),
                                  offsets = (0,0),
                                  block_shape = (K_TILE_SIZE,D),
                                  order = (1,0))



  O_acc = tl.zeros((Q_TILE_SIZE,D),dtype=tl.float32) ### on chip accumulator
  L_acc = tl.zeros((Q_TILE_SIZE,),dtype=tl.float32) ### on chip running normalizer
  M_acc = tl.full((Q_TILE_SIZE,),-float("inf"), dtype=tl.float32) ### on chip running maximum

  Q_rows = tl.load(Q_block_ptr,boundary_check=(0,),padding_option="zero") ### ROW tile of Q -> [Q_TILE_SIZE, D]

  ### tl.arange should take compile take constant argument, so following line doesn't work
  #Q_ids = tl.arange(MY_QTILE_ID*Q_TILE_SIZE,MY_QTILE_ID*Q_TILE_SIZE+Q_TILE_SIZE) ### [Q_TILE_SIZE,]

  if is_causal:
    Q_ids = MY_QTILE_ID*Q_TILE_SIZE + tl.arange(0,Q_TILE_SIZE) ### [Q_TILE_SIZE,]


  for i in range(tl.cdiv(TK,K_TILE_SIZE)):
    K_rows = tl.load(K_block_ptr,boundary_check=(0,),padding_option="zero") ### ROW tile of K -> [K_TILE_SIZE, D]
    V_rows = tl.load(V_block_ptr,boundary_check=(0,),padding_option="zero") ### ROW tile of V -> [V_TILE_SIZE, D]

    #K_ids = tl.arange(i*K_TILE_SIZE,i*K_TILE_SIZE + K_TILE_SIZE) ### [K_TILE_SIZE,]

    K_ids = i*K_TILE_SIZE + tl.arange(0,K_TILE_SIZE) ### [K_TILE_SIZE,]
    valid_K_ids = K_ids < TK ### last few padded rows from K can pollute the softmax calculation, so make them -inf, instead of zero

    if is_causal:
      mask = (valid_K_ids[None,:] & (Q_ids[:,None] >= K_ids[None,:]))
    else:
      mask = valid_K_ids[None,:]


    S = tl.dot(Q_rows,K_rows.T,input_precision="ieee") * scale ### [Q_TILE_SIZE, D] @ [D,K_TILE_SIZE] -> [Q_TILE_SIZE, K_TILE_SIZE]

    S= tl.where(mask,S,-float("inf"))

    M_new = tl.maximum(M_acc,tl.max(S,axis=-1)) ### tl.max([Q_TILE_SIZE, K_TILE_SIZE]) -> [Q_TILE_SIZE,], then tl.maximum([Q_TILE_SIZE,], [Q_TILE_SIZE,] -> [Q_TILE_SIZE,])
    ## P = tl.exp(S - M_new[...,None]) Gives compilation error: Triton deosn't support ...

    P = tl.exp(S - M_new[:,None]) ### M_new: [Q_TILE_SIZE,] -> [Q_TILE_SIZE,1], then [Q_TILE_SIZE, K_TILE_SIZE] - [Q_TILE_SIZE,1] -> [Q_TILE_SIZE, K_TILE_SIZE]
    L_acc = L_acc * tl.exp(M_acc-M_new) + tl.sum(P,axis=-1) ### [Q_TILE_SIZE,]*[Q_TILE_SIZE,] + [Q_TILE_SIZE,] -> [Q_TILE_SIZE,]

    O_acc = O_acc * tl.exp(M_acc[:,None] - M_new[:,None]) ### [Q_TILE_SIZE, D] * [Q_TILE_SIZE, 1] -> [Q_TILE_SIZE, D]
    O_acc = tl.dot(P.to(V_rows.dtype),V_rows,acc = O_acc,input_precision="ieee") ### P = [Q_TILE_SIZE, K_TILE_SIZE], V_rows = [K_TILE_SIZE,D]

    M_acc = M_new

    K_block_ptr = K_block_ptr.advance((K_TILE_SIZE,0))
    V_block_ptr = V_block_ptr.advance((K_TILE_SIZE,0))

  O_acc = O_acc /L_acc[:,None] ### [Q_TILE_SIZE, D] * [Q_TILE_SIZE, 1] -> [Q_TILE_SIZE, D]
  tl.store(L_block_ptr, M_acc + tl.log(L_acc),boundary_check = (0,))
  tl.store(O_block_ptr, O_acc.to(O_block_ptr.type.element_ty),boundary_check=(0,))
  return





@triton.jit
def FAT2_triton_bwd_find_Delta(O_ptr,O_batch_stride,O_token_stride,O_dim_stride,
                               dO_ptr,dO_batch_stride,dO_token_stride,dO_dim_stride,
                               Delta_ptr,Delta_batch_stride,Delta_token_stride,
                               TQ,D: tl.constexpr,Q_TILE_SIZE: tl.constexpr):
    MY_QTILE_ID = tl.program_id(0)
    MY_BATCH_ID = tl.program_id(1)

    O_block_ptr = tl.make_block_ptr(base=O_ptr + MY_BATCH_ID * O_batch_stride,
                                    shape=(TQ, D),strides=(O_token_stride, O_dim_stride),
                                    offsets=(MY_QTILE_ID * Q_TILE_SIZE, 0),
                                    block_shape=(Q_TILE_SIZE, D),
                                    order=(1, 0))

    dO_block_ptr = tl.make_block_ptr(base=dO_ptr + MY_BATCH_ID * dO_batch_stride,
                                         shape=(TQ, D),strides=(dO_token_stride, dO_dim_stride),
                                         offsets=(MY_QTILE_ID * Q_TILE_SIZE, 0),
                                         block_shape=(Q_TILE_SIZE, D),
                                         order=(1, 0))

    Delta_block_ptr = tl.make_block_ptr(base=Delta_ptr + MY_BATCH_ID * Delta_batch_stride,
                                        shape=(TQ,),strides=(Delta_token_stride,),
                                        offsets=(MY_QTILE_ID * Q_TILE_SIZE,),
                                        block_shape=(Q_TILE_SIZE,),
                                        order=(0,))

    O_rows = tl.load(O_block_ptr,boundary_check=(0,),padding_option="zero").to(tl.float32)

    dO_rows = tl.load(dO_block_ptr,boundary_check=(0,),padding_option="zero").to(tl.float32)

    Delta_rows = tl.sum(O_rows * dO_rows,axis=1)

    tl.store(Delta_block_ptr,Delta_rows,boundary_check=(0,))


@triton.jit
def FAT2_triton_bwd_dKdV(Q_ptr,Q_batch_stride,Q_token_stride,Q_dim_stride,
                         K_ptr,K_batch_stride,K_token_stride,K_dim_stride,
                         V_ptr,V_batch_stride,V_token_stride,V_dim_stride,
                         dO_ptr,dO_batch_stride,dO_token_stride,dO_dim_stride,
                         L_ptr,L_batch_stride,L_token_stride,
                         Delta_ptr,Delta_batch_stride,Delta_token_stride,
                         dK_ptr,dK_batch_stride,dK_token_stride,dK_dim_stride,
                         dV_ptr,dV_batch_stride,dV_token_stride,dV_dim_stride,
                         TQ,TK,scale,
                         D: tl.constexpr,Q_TILE_SIZE: tl.constexpr,
                         K_TILE_SIZE: tl.constexpr,is_causal: tl.constexpr):
    MY_KTILE_ID = tl.program_id(0)
    MY_BATCH_ID = tl.program_id(1)

    K_block_ptr = tl.make_block_ptr(base=K_ptr + MY_BATCH_ID * K_batch_stride,
                                    shape=(TK, D),strides=(K_token_stride, K_dim_stride),
                                    offsets=(MY_KTILE_ID * K_TILE_SIZE, 0),
                                    block_shape=(K_TILE_SIZE, D),
                                    order=(1, 0))

    V_block_ptr = tl.make_block_ptr(base=V_ptr + MY_BATCH_ID * V_batch_stride,
                                    shape=(TK, D),strides=(V_token_stride, V_dim_stride),
                                    offsets=(MY_KTILE_ID * K_TILE_SIZE, 0),
                                    block_shape=(K_TILE_SIZE, D),
                                    order=(1, 0))

    dK_block_ptr = tl.make_block_ptr(base=dK_ptr + MY_BATCH_ID * dK_batch_stride,shape=(TK, D),
                                     strides=(dK_token_stride, dK_dim_stride),
                                     offsets=(MY_KTILE_ID * K_TILE_SIZE, 0),
                                     block_shape=(K_TILE_SIZE, D),
                                     order=(1, 0))

    dV_block_ptr = tl.make_block_ptr(base=dV_ptr + MY_BATCH_ID * dV_batch_stride,shape=(TK, D),
                                     strides=(dV_token_stride, dV_dim_stride),
                                     offsets=(MY_KTILE_ID * K_TILE_SIZE, 0),
                                     block_shape=(K_TILE_SIZE, D),
                                     order=(1, 0),)

    Q_block_ptr = tl.make_block_ptr(base=Q_ptr + MY_BATCH_ID * Q_batch_stride,shape=(TQ, D),
                                    strides=(Q_token_stride, Q_dim_stride),
                                    offsets=(0, 0),
                                    block_shape=(Q_TILE_SIZE, D),
                                    order=(1, 0))

    dO_block_ptr = tl.make_block_ptr(base=dO_ptr + MY_BATCH_ID * dO_batch_stride,
                                         shape=(TQ, D),
                                         strides=(dO_token_stride, dO_dim_stride),
                                         offsets=(0, 0),
                                         block_shape=(Q_TILE_SIZE, D),
                                         order=(1, 0))

    L_block_ptr = tl.make_block_ptr(base=L_ptr + MY_BATCH_ID * L_batch_stride,shape=(TQ,),
                                    strides=(L_token_stride,),
                                    offsets=(0,),
                                    block_shape=(Q_TILE_SIZE,),
                                    order=(0,))

    Delta_block_ptr = tl.make_block_ptr(base=Delta_ptr + MY_BATCH_ID * Delta_batch_stride,shape=(TQ,),
                                        strides=(Delta_token_stride,),
                                        offsets=(0,),
                                        block_shape=(Q_TILE_SIZE,),
                                        order=(0,))

    K_rows = tl.load(K_block_ptr,boundary_check=(0,),padding_option="zero") ##[K_TILE_SIZE,D]

    V_rows = tl.load(V_block_ptr,boundary_check=(0,),padding_option="zero") ##[K_TILE_SIZE,D]

    dK_acc = tl.zeros((K_TILE_SIZE, D),dtype=tl.float32) ##[K_TILE_SIZE,D]

    dV_acc = tl.zeros((K_TILE_SIZE, D),dtype=tl.float32) ##[K_TILE_SIZE,D]

    K_ids = MY_KTILE_ID*K_TILE_SIZE + tl.arange(0, K_TILE_SIZE) ## [K_TILE_SIZE,]


    for i in range(0, tl.cdiv(TQ, Q_TILE_SIZE)):
        Q_rows = tl.load(Q_block_ptr,boundary_check=(0,),padding_option="zero") ## [Q_TILE_SIZE,D]
        dO_rows = tl.load(dO_block_ptr,boundary_check=(0,),padding_option="zero") ## [Q_TILE_SIZE,D]
        L_rows = tl.load(L_block_ptr,boundary_check=(0,),padding_option="zero") ##[Q_TILE_SIZE,]
        Delta_rows = tl.load(Delta_block_ptr,boundary_check=(0,),padding_option="zero") ##[Q_TILE_SIZE,]

        Q_ids = i * Q_TILE_SIZE+ tl.arange(0, Q_TILE_SIZE)  ##[Q_TILE_SIZE,]

        mask = (Q_ids[:, None] < TQ) & (K_ids[None, :] < TK) ## -> [Q_TILE_SIZE, K_TILE_SIZE]

        if is_causal:
            mask = (mask & (Q_ids[:, None] >= K_ids[None, :])) ## [Q_TILE_SIZE, K_TILE_SIZE]

        S = tl.dot(Q_rows,K_rows.T,input_precision="ieee") * scale ## -> [Q_TILE_SIZE, D] @ [D,K_TILE_SIZE] -> [Q_TILE_SIZE, K_TILE_SIZE]

        S = tl.where(mask,S,-float("inf"))  ## [Q_TILE_SIZE, K_TILE_SIZE]

        # Recompute P tile using saved logsumexp.
        P = tl.exp(S - L_rows[:, None]) ## [Q_TILE_SIZE, K_TILE_SIZE] - [Q_TILE_SIZE,1] -> [Q_TILE_SIZE, K_TILE_SIZE]

        # dV += P^T @ dO
        PT = P.T.to(dO_rows.dtype) ### [K_TILE_SIZE, Q_TILE_SIZE]
        dV_acc = tl.dot(PT,dO_rows,acc=dV_acc,input_precision="ieee") ### [K_TILE_SIZE, Q_TILE_SIZE] @ [Q_TILE_SIZE,D] -> [K_TILE_SIZE,D]

        # dP = dO @ V^T
        dP = tl.dot(dO_rows,V_rows.T,input_precision="ieee") ### [Q_TILE_SIZE,D] @ [D,K_TILE_SIZE] ->[Q_TILE_SIZE, K_TILE_SIZE]

        # dS = P * (dP - Delta)
        dS = P * (dP - Delta_rows[:, None]) ### [Q_TILE_SIZE, K_TILE_SIZE] - [Q_TILE_SIZE, 1] ->[Q_TILE_SIZE, K_TILE_SIZE]

        # dK += (dS^T @ Q) * scale
        dS_for_dot = (dS.T * scale).to(Q_rows.dtype)

        dK_acc = tl.dot(dS_for_dot,Q_rows,acc=dK_acc,input_precision="ieee")

        Q_block_ptr = Q_block_ptr.advance((Q_TILE_SIZE, 0))
        dO_block_ptr = dO_block_ptr.advance((Q_TILE_SIZE, 0))
        L_block_ptr = L_block_ptr.advance((Q_TILE_SIZE,))
        Delta_block_ptr = Delta_block_ptr.advance((Q_TILE_SIZE,))

    tl.store(dK_block_ptr,dK_acc.to(dK_block_ptr.type.element_ty),boundary_check=(0,))

    tl.store(dV_block_ptr,dV_acc.to(dV_block_ptr.type.element_ty),boundary_check=(0,))


@triton.jit
def FAT2_triton_bwd_dQ(Q_ptr,Q_batch_stride,Q_token_stride,Q_dim_stride,
                       K_ptr,K_batch_stride,K_token_stride,K_dim_stride,
                       V_ptr,V_batch_stride,V_token_stride,V_dim_stride,
                       dO_ptr,dO_batch_stride,dO_token_stride,dO_dim_stride,
                       L_ptr,L_batch_stride,L_token_stride,
                       Delta_ptr,Delta_batch_stride,Delta_token_stride,
                       dQ_ptr,dQ_batch_stride,dQ_token_stride,dQ_dim_stride,
                       TQ,TK,scale,
                       D: tl.constexpr,Q_TILE_SIZE: tl.constexpr,
                       K_TILE_SIZE: tl.constexpr,is_causal: tl.constexpr):
    MY_QTILE_ID = tl.program_id(0)
    MY_BATCH_ID = tl.program_id(1)

    Q_block_ptr = tl.make_block_ptr(base=Q_ptr + MY_BATCH_ID * Q_batch_stride,shape=(TQ, D),
                                    strides=(Q_token_stride, Q_dim_stride),
                                    offsets=(MY_QTILE_ID * Q_TILE_SIZE, 0),
                                    block_shape=(Q_TILE_SIZE, D),
                                    order=(1, 0))

    dO_block_ptr = tl.make_block_ptr(base=dO_ptr + MY_BATCH_ID * dO_batch_stride,shape=(TQ, D),
                                     strides=(dO_token_stride, dO_dim_stride),
                                     offsets=(MY_QTILE_ID * Q_TILE_SIZE, 0),
                                     block_shape=(Q_TILE_SIZE, D),
                                     order=(1, 0))

    L_block_ptr = tl.make_block_ptr(base=L_ptr + MY_BATCH_ID * L_batch_stride,shape=(TQ,),
                                    strides=(L_token_stride,),
                                    offsets=(MY_QTILE_ID * Q_TILE_SIZE,),
                                    block_shape=(Q_TILE_SIZE,),
                                    order=(0,))

    Delta_block_ptr = tl.make_block_ptr(base=Delta_ptr + MY_BATCH_ID * Delta_batch_stride,shape=(TQ,),
                                        strides=(Delta_token_stride,),
                                        offsets=(MY_QTILE_ID * Q_TILE_SIZE,),
                                        block_shape=(Q_TILE_SIZE,),
                                        order=(0,))

    dQ_block_ptr = tl.make_block_ptr(base=dQ_ptr + MY_BATCH_ID * dQ_batch_stride, shape=(TQ, D),
                                     strides=(dQ_token_stride, dQ_dim_stride),
                                     offsets=(MY_QTILE_ID * Q_TILE_SIZE, 0),
                                     block_shape=(Q_TILE_SIZE, D),
                                     order=(1, 0))

    K_block_ptr = tl.make_block_ptr(base=K_ptr + MY_BATCH_ID * K_batch_stride,shape=(TK, D),
                                    strides=(K_token_stride, K_dim_stride),
                                    offsets=(0, 0),
                                    block_shape=(K_TILE_SIZE, D),
                                    order=(1, 0))

    V_block_ptr = tl.make_block_ptr(base=V_ptr + MY_BATCH_ID * V_batch_stride,
                                    shape=(TK, D),strides=(V_token_stride, V_dim_stride),
                                    offsets=(0, 0),
                                    block_shape=(K_TILE_SIZE, D),
                                    order=(1, 0))

    Q_rows = tl.load(Q_block_ptr,boundary_check=(0,),padding_option="zero")
    dO_rows = tl.load(dO_block_ptr,boundary_check=(0,),padding_option="zero")
    L_rows = tl.load(L_block_ptr,boundary_check=(0,),padding_option="zero")
    Delta_rows = tl.load(Delta_block_ptr,boundary_check=(0,),padding_option="zero")

    dQ_acc = tl.zeros((Q_TILE_SIZE, D),dtype=tl.float32)

    Q_ids = MY_QTILE_ID * Q_TILE_SIZE+ tl.arange(0, Q_TILE_SIZE)

    for j in range(0, tl.cdiv(TK, K_TILE_SIZE)):
        K_rows = tl.load(K_block_ptr,boundary_check=(0,),padding_option="zero")
        V_rows = tl.load(V_block_ptr,boundary_check=(0,),padding_option="zero")

        K_ids = j * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)
        mask = (Q_ids[:, None] < TQ) & (K_ids[None, :] < TK)


        if is_causal:
            mask =mask & (Q_ids[:, None] >= K_ids[None, :])

        S = tl.dot(Q_rows,K_rows.T,input_precision="ieee") * scale
        S = tl.where(mask,S,-float("inf"))
        P = tl.exp(S - L_rows[:, None])

        dP = tl.dot(dO_rows,V_rows.T,input_precision="ieee")
        dS = P * (dP - Delta_rows[:, None])
        dS_for_dot = (dS * scale).to(K_rows.dtype)
        dQ_acc = tl.dot(dS_for_dot,K_rows,acc=dQ_acc,input_precision="ieee")

        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))

    tl.store(dQ_block_ptr,dQ_acc.to(dQ_block_ptr.type.element_ty),boundary_check=(0,))

class FlashAttention2Triton(torch.autograd.Function):
  @staticmethod
  def forward(ctx,Q:torch.Tensor,K:torch.Tensor,V:torch.Tensor,
              Q_TILE_SIZE: int=16, K_TILE_SIZE: int =16, is_causal:bool=False):
    #### Expects Q.shape as [B,h,TQ,dh] and K,V.shape as [B,h,TK,dh]

    assert Q.is_cuda and Q.device == K.device == V.device and Q.ndim==4 and Q.ndim == K.ndim == V.ndim
    assert Q.shape[:-2] == K.shape[:-2] == V.shape[:-2] ## same batch_size and num_head
    assert Q.shape[-1] == K.shape[-1] == V.shape[-1] ### same dimension for all
    assert K.shape[-2] == V.shape[-2] ### context length of K and V
    assert Q.dtype == K.dtype == V.dtype

    B,h,TQ,dh = Q.shape
    TK = K.shape[-2]



    Q_flat = Q.reshape(-1,TQ,dh)
    K_flat = K.reshape(-1,TK,dh)
    V_flat = V.reshape(-1,TK,dh)


    O = torch.empty_like(Q_flat, dtype = Q.dtype)
    L = torch.zeros((Q_flat.shape[0],TQ), dtype=torch.float32 ,device = Q.device)


    scale = 1/math.sqrt(dh)
    FAT2_triton_fwd[(triton.cdiv(TQ,Q_TILE_SIZE),Q_flat.shape[0])](Q_flat,Q_flat.stride(0),Q_flat.stride(1),Q_flat.stride(2),
                                                                    K_flat,K_flat.stride(0),K_flat.stride(1),K_flat.stride(2),
                                                                    V_flat,V_flat.stride(0),V_flat.stride(1),V_flat.stride(2),
                                                                    O,O.stride(0),O.stride(1),O.stride(2),
                                                                    L, L.stride(0), L.stride(1),
                                                                    TQ,TK,scale,
                                                                    D = dh,
                                                                    Q_TILE_SIZE = Q_TILE_SIZE,
                                                                    K_TILE_SIZE = K_TILE_SIZE,
                                                                   is_causal = is_causal)

    O = O.reshape((B,h,TQ,dh))
    L = L.reshape((B,h,TQ))
    ctx.save_for_backward(L,Q,K,V,O)
    ctx.Q_TILE_SIZE = Q_TILE_SIZE
    ctx.K_TILE_SIZE = K_TILE_SIZE
    ctx.is_causal= is_causal
    return O

  @staticmethod
  def backward(ctx, dO):
      L, Q, K, V, O = ctx.saved_tensors
      is_causal = ctx.is_causal
      B, H, TQ, D = Q.shape
      TK = K.shape[-2]
      Q_flat = Q.reshape(-1, TQ, D)
      K_flat = K.reshape(-1, TK, D)
      V_flat = V.reshape(-1, TK, D)
      O_flat = O.reshape(-1, TQ, D)
      dO_flat = dO.reshape(-1, TQ, D).contiguous()

      dQ_flat = torch.empty_like(Q_flat)
      dK_flat = torch.empty_like(K_flat)
      dV_flat = torch.empty_like(V_flat)

      Delta = torch.empty((Q_flat.shape[0], TQ),dtype=torch.float32,device=Q.device)

      Q_TILE_SIZE = ctx.Q_TILE_SIZE
      K_TILE_SIZE = ctx.K_TILE_SIZE
      scale = 1.0 / math.sqrt(D)
      q_grid = (triton.cdiv(TQ, Q_TILE_SIZE),Q_flat.shape[0])

      k_grid = (triton.cdiv(TK, K_TILE_SIZE),Q_flat.shape[0])

      FAT2_triton_bwd_find_Delta[q_grid](O_flat,O_flat.stride(0),O_flat.stride(1),O_flat.stride(2),
                                        dO_flat,dO_flat.stride(0),dO_flat.stride(1),dO_flat.stride(2),
                                        Delta,Delta.stride(0),Delta.stride(1),
                                        TQ,D=D,
                                        Q_TILE_SIZE=Q_TILE_SIZE)
      L_flat = L.reshape(-1, TQ)

      FAT2_triton_bwd_dKdV[k_grid](Q_flat,Q_flat.stride(0),Q_flat.stride(1),Q_flat.stride(2),
                                  K_flat,K_flat.stride(0),K_flat.stride(1),K_flat.stride(2),
                                  V_flat,V_flat.stride(0),V_flat.stride(1),V_flat.stride(2),
                                  dO_flat,dO_flat.stride(0),dO_flat.stride(1),dO_flat.stride(2),
                                  L_flat, L_flat.stride(0),L_flat.stride(1),
                                  Delta,Delta.stride(0),Delta.stride(1),
                                  dK_flat,dK_flat.stride(0),dK_flat.stride(1),dK_flat.stride(2),
                                  dV_flat,dV_flat.stride(0),dV_flat.stride(1),dV_flat.stride(2),
                                  TQ,TK,scale,
                                  D=D,Q_TILE_SIZE=Q_TILE_SIZE,K_TILE_SIZE=K_TILE_SIZE,is_causal=is_causal)

      FAT2_triton_bwd_dQ[q_grid](Q_flat,Q_flat.stride(0),Q_flat.stride(1),Q_flat.stride(2),
                                K_flat,K_flat.stride(0),K_flat.stride(1),K_flat.stride(2),
                                V_flat,V_flat.stride(0),V_flat.stride(1),V_flat.stride(2),
                                dO_flat,dO_flat.stride(0),dO_flat.stride(1),dO_flat.stride(2),
                                L_flat,L_flat.stride(0),L_flat.stride(1),
                                Delta,Delta.stride(0),Delta.stride(1),
                                dQ_flat,dQ_flat.stride(0),dQ_flat.stride(1),dQ_flat.stride(2),
                                TQ,TK,scale,D=D,
                                Q_TILE_SIZE=Q_TILE_SIZE,K_TILE_SIZE=K_TILE_SIZE,is_causal=is_causal)

      dQ = dQ_flat.reshape_as(Q)
      dK = dK_flat.reshape_as(K)
      dV = dV_flat.reshape_as(V)

      return dQ, dK, dV, None, None, None ### PyTorch expects backward() to return one gradient entry for every argument passed to .apply().


def flash_attention(Q: torch.Tensor,K: torch.Tensor,V: torch.Tensor,
                    Q_TILE: int =16, K_TILE: int=16, is_causal: bool = True) -> torch.Tensor:
    return FlashAttention2Triton.apply(Q,K,V,Q_TILE,K_TILE,is_causal) ###.apply only takes positional argument so writing Q_TILE= Q_TILE would be wrong

















