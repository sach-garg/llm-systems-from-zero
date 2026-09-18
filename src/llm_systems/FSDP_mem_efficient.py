import torch
import torch.distributed as dist
from llm_systems.model import Linear, Embedding


from types import MethodType



class FSDP_Linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, layer):
        ctx.layer = layer
        ctx.input_dtype = x.dtype

        x_compute = x.to(weight.dtype)

        # Save the activation, for backward
        ctx.save_for_backward(x_compute)

        return x_compute @ weight.transpose(0, 1)

    @staticmethod
    def backward(ctx, grad_output):
        (x_compute,) = ctx.saved_tensors

        # The existing backward-pre hook would already have gathered the weight 
        weight = ctx.layer.weight
        grad_output = grad_output.to(weight.dtype)

        grad_x = None
        grad_weight = None

        grad_x = (grad_output @ weight).to(ctx.input_dtype)

        x_flat = x_compute.reshape(-1, x_compute.shape[-1])
        grad_flat = grad_output.reshape(-1, grad_output.shape[-1])
        grad_weight = grad_flat.transpose(0, 1) @ x_flat
        return grad_x, grad_weight, None


def custom_FSDP_Linear(layer, x):
    return FSDP_Linear.apply(x, layer.weight, layer)



class FSDP_mem_efficient(torch.nn.Module):
    def __init__(self,module : torch.nn.Module, compute_dtype : torch.dtype | None=None, cleanup_distance: int =1):
        super().__init__()
        self.module = module
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.compute_dtype = compute_dtype
        self.prefetch_distance = 2
        self.cleanup_distance = cleanup_distance
                  
        

        #### Broadcast weights at initialization
        with torch.no_grad():
            for parameter in self.module.parameters():
                parameter.data = parameter.data.to(torch.float32)
                dist.broadcast(parameter.data,src=0)

        # module.modules() recursively traverses the complete module hierarchy, not just
        # the immediate children. Therefore, it enters each TransformerBlock and its
        # nested attention/MLP modules, allowing us to find every Linear and Embedding
        # layer while excluding other layers like RoPE, RMS NORM, and higher modules like CausalAttention, TransformerBlock

        self.sharded_layers =[]
        for child in self.module.modules():
            if isinstance(child,(Linear,Embedding)):
                self.sharded_layers.append(child)

        self.sharded_layer_indices= {layer:id for id,layer in enumerate(self.sharded_layers)}
      
      
        
        ### Creating a metadata dictionary which stores the original shape of the sharded weight tensors
        self.sharded_metadata = {}
        for layer in self.sharded_layers:
            self.sharded_metadata[layer]={}
            self.sharded_metadata[layer]["original_shape"] = layer.weight.shape
            self.sharded_metadata[layer]["original_numel"] = layer.weight.numel()
            assert layer.weight.numel() % self.world_size==0, "Number of Elements in a weight matrix are not divisible by number of workers."
            self.sharded_metadata[layer]["shard_numel"] = layer.weight.numel() // self.world_size

        ### Sharding the weight matrices and storing in the metadata,
        ### The metadata[local_shard] is used to restore the rank's local weight shard after, forward pass using all_gathered weight is finished
        ### gather_handle is handle which tells us whether all_gather has been queued on GPU or not
        ### comm_shard is the shard that will be communicated by this rank, basically it could be a downcasted version of local_shard
        ### gathered_flag to check if the full_weight has been all_gathered or not
 
     
        for layer in self.sharded_layers:
            metadata = self.sharded_metadata[layer]
            flattened_weight = layer.weight.data.reshape(-1)
            shard_start = self.rank * metadata["shard_numel"]
            shard_end = shard_start + metadata["shard_numel"]
            local_shard = flattened_weight[shard_start:shard_end].clone() ###.clone() is necessary because the local shard must own independent storage rather than remain a view of the full flattened weight.
            layer.weight.data = local_shard
            metadata["local_shard"] = local_shard
            metadata["gather_handle"] = None
            metadata["gathered_shards"] = None
            metadata["comm_shard"] = None
            metadata["gathered_flag"] = False

            metadata["grad_handle"] = None
            metadata["grad_shard"] = None
            metadata["grad_chunks"] = None

        self.sharded_parameter_to_layer = {layer.weight: layer for layer in self.sharded_layers}


        with torch.no_grad():
            for layer in self.sharded_layers:
                if isinstance(layer,Linear):
                    layer.forward = MethodType(custom_FSDP_Linear, layer)
                layer.register_forward_pre_hook(self.pre_forwardPass)
                layer.register_forward_hook(self.post_forwardPass)
                layer.register_full_backward_pre_hook(self.pre_backwardPass)
                layer.weight.register_post_accumulate_grad_hook(self.post_accumulate_sharded_gradient)

        self.module.register_full_backward_pre_hook(self.pre_model_backward)


        ### For unsharded parameters, we need to perform all reduce, so create handles for allreduce and register hooks

        self.unsharded_params = []
        for p in self.module.parameters():
            if p.requires_grad and p not in self.sharded_parameter_to_layer:
                self.unsharded_params.append(p)

        self.unsharded_grad_handles = {p:None for p in self.unsharded_params}
                
        
        for p in self.unsharded_params:
            p.register_post_accumulate_grad_hook(self.post_accumulate_unsharded_gradient)

     

    def start_layer_all_gather(self,layer):
        metadata = self.sharded_metadata[layer]
        if metadata["gather_handle"] is not None or metadata["gathered_flag"]:
            return

        local_shard = metadata["local_shard"]
        if self.compute_dtype is None:
            comm_shard = local_shard
        else:
            comm_shard = local_shard.to(self.compute_dtype)

        gathered_shards = [torch.empty_like(comm_shard) for _ in range(self.world_size)]
        gather_handle = dist.all_gather(gathered_shards,comm_shard,async_op=True)
        metadata["comm_shard"] = comm_shard ### necessary here, because layer_all_gather function might return before all_gather is complete and we lose reference to comm_shard
        metadata["gathered_shards"] = gathered_shards
        metadata["gather_handle"] = gather_handle
        return

    def end_layer_all_gather(self,layer):
        metadata = self.sharded_metadata[layer]
        if metadata["gathered_flag"]:
            return
        assert metadata["gather_handle"] is not None, "Gather handle absent, can not run end_layer_all_gather"

        metadata["gather_handle"].wait()
        full_flat_weight = torch.cat(metadata["gathered_shards"],dim=0)

        layer.weight.data = full_flat_weight.reshape(metadata["original_shape"])
        metadata["gather_handle"] = None
        metadata["gathered_shards"] = None
        metadata["comm_shard"] = None
        metadata["gathered_flag"] = True
        return

    def reshard_layer(self,layer):
        metadata = self.sharded_metadata[layer]
        assert metadata["gathered_flag"] , "Weight not fully gathered to reshard again"
        layer.weight.data = metadata["local_shard"]
        metadata["gathered_flag"] = False
        return

    def pre_forwardPass(self,layer,inputs): ### becaue in pre forward hook, Pytorch also passes inputs. We don't use inputs anywhere here, but have to take as an argument
        self.start_layer_all_gather(layer)

        #####
        current_id = self.sharded_layer_indices[layer]
        prefetch_id = current_id + self.prefetch_distance
        if prefetch_id < len(self.sharded_layers):
            self.start_layer_all_gather(self.sharded_layers[prefetch_id])

        self.end_layer_all_gather(layer)

        return

    def post_forwardPass(self,layer,inputs,outputs): ### becaue in post forward hook, Pytorch passes inputs, outputs. We don't use them anywhere here, but have to take as arguments.
        self.reshard_layer(layer)
        return

    def forward(self,*inputs,**kwargs):
        for layer in self.sharded_layers[:self.prefetch_distance]:
            self.start_layer_all_gather(layer)
        return self.module(*inputs,**kwargs)


    def pre_backwardPass(self,layer,gradOutput):
        self.start_layer_all_gather(layer)
        current_id = self.sharded_layer_indices[layer]
        prefetch_id = current_id - self.prefetch_distance
        if prefetch_id >=0:
            self.start_layer_all_gather(self.sharded_layers[prefetch_id])
        self.end_layer_all_gather(layer)
        return

    def start_layer_grad_reduce_scatter(self,layer):
        metadata = self.sharded_metadata[layer]
        parameter = layer.weight

        if metadata["grad_handle"] is not None or parameter.grad is None:
            return

        local_data_full_grad = (parameter.grad.detach().reshape(-1))
        grad_chunks = list(local_data_full_grad.chunk(self.world_size))
        grad_shard = torch.empty(metadata["shard_numel"],dtype=local_data_full_grad.dtype,device=local_data_full_grad.device)
        grad_handle = dist.reduce_scatter(grad_shard,grad_chunks,op=dist.ReduceOp.SUM,async_op=True)
        metadata["grad_chunks"] = grad_chunks
        metadata["grad_shard"] = grad_shard
        metadata["grad_handle"] = grad_handle
        return

    def post_accumulate_sharded_gradient(self,parameter):
        layer = self.sharded_parameter_to_layer[parameter]
        self.release_completed_gradient_buffers(layer)
        metadata = self.sharded_metadata[layer]
        self.start_layer_grad_reduce_scatter(layer)
        assert metadata["grad_handle"] is not None
        parameter.grad = None
        self.reshard_layer(layer)
        return

    def release_completed_gradient_buffers(self, current_layer):
        if self.cleanup_distance ==0:
            return
        current_id = self.sharded_layer_indices[current_layer]

        for layer_id in range(current_id + self.cleanup_distance, len(self.sharded_layers), self.cleanup_distance):
            layer = self.sharded_layers[layer_id]
            metadata = self.sharded_metadata[layer]
            handle = metadata["grad_handle"]

            if handle is None or metadata["grad_chunks"] is None:
                continue

            if handle.is_completed():
                handle.wait()
                metadata["grad_chunks"] = None

    def finish_gradient_synchronization(self):
        ### For sharded weights
        for layer in self.sharded_layers:
            metadata = self.sharded_metadata[layer]
            if metadata["grad_handle"] is None:
                continue
            metadata["grad_handle"].wait()
            grad_shard = metadata["grad_shard"]
            grad_shard.div_(self.world_size)
            layer.weight.grad = grad_shard.to(dtype=metadata["local_shard"].dtype)
            metadata["grad_handle"] = None
            metadata["grad_shard"] = None
            metadata["grad_chunks"] = None

        
        #### For unsharded weights
        for parameter in self.unsharded_params:
            if self.unsharded_grad_handles[parameter] is None:
                continue
            self.unsharded_grad_handles[parameter].wait()
            assert parameter.grad is not None, "For unsharded weight gradient not found"
            parameter.grad.div_(self.world_size)
            self.unsharded_grad_handles[parameter]= None

        return

    def post_accumulate_unsharded_gradient(self,parameter):
        if parameter.grad is None or self.unsharded_grad_handles[parameter] is not None:
            return

        grad_handle = dist.all_reduce(parameter.grad,op=dist.ReduceOp.SUM,async_op=True)
        self.unsharded_grad_handles[parameter] = grad_handle
        return

    def pre_model_backward(self,module,grad_output):
        last_layers = self.sharded_layers[-self.prefetch_distance:]

        for layer in reversed(last_layers):
            self.start_layer_all_gather(layer)
        return






        



    
    
    









                        


