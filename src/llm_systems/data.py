from __future__ import annotations

import numpy as np
import numpy.typing as npt
import torch




def get_batch(dataset: npt.NDArray, batch_size:int, context_length:int,device:str | None):
  last_index = len(dataset) - context_length
  start_idx = np.random.randint(0,last_index,size=batch_size,dtype=np.int64)
  offsets = np.arange(context_length+1,dtype= np.int64)
  ids = start_idx[:,None] + offsets[None,:]
  batch_np = np.asarray(dataset[ids], order="C")          # (B, T+1), likely uint16, order="C" makes the row major contiguous layout in memory
  batch = torch.from_numpy(batch_np).long()            # CPU torch.long, long() in int64, which is usually input into the model

  fast_transfer = device is not None and "cuda" in device

  if fast_transfer:
      batch = batch.pin_memory()

  X = batch[:, :-1]
  Y = batch[:,  1:]


  if device is not None:
      X = X.to(device, non_blocking=fast_transfer)
      Y = Y.to(device, non_blocking=fast_transfer)
  return (X,Y)
