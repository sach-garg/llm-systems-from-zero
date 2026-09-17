import torch
import torch.distributed as dist




class ShardedOptimizer(torch.optim.Optimizer):
    def __init__(self,params,optimizer_cls,**kwargs): ### params would be model.parameters() or a list of parameter group dictionaries, ### optimizer_cls is a class object e.g. torch.optim.AdamW, and not a class instance.
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.optimizer_cls = optimizer_cls
        self.next_param_index = 0 ## Our logic: if (paramater index)% world_size ==i, then rank i will own that parameter. We need to iterate through parameters (in same order on all ranks) and advance self.next_param_index and assign ownership of that parameter to rank self.next_param_index % world_size
        self.params_with_owners = [] ## contains mapping from all parameters to their owner's rank. This will be used in broadcast after optimizer step
        self.local_param_groups =[] ## list of local parameter group dictionaries
        self.local_optimizer = None

        ## The following line should only come after all the previous lines. This is because the following line will automatically call add_param_group, but as we have provided our own add_param_group,
        ## we need all things like self.rank ---- self.local_optimizer to exist.
        ## Note that we need to provide our own add_param_group function because we are sharding the optimizer 
        ## Had we done, super().__init__(params,defaults=kwargs) and not define add_param_group, an optimizer would have been created at all ranks with all parameters, as in DDP
        super().__init__(params,defaults=kwargs) 
        self.local_optimizer = self.optimizer_cls(self.local_param_groups,**kwargs)
   
    def step(self,closure=None):
        loss = self.local_optimizer.step(closure=closure)
        with torch.no_grad():
            for parameter,owner_rank in self.params_with_owners:
                dist.broadcast(parameter,src=owner_rank)
        return loss

    def add_param_group(self,param_group):
        super().add_param_group(param_group) ### preprocessing the parameter group
        ## A parameter group is a dictionary like {"params":model.parameters(), "lr":0.1, "weight_decay": 0.1}
        ## it normalizes param_group and appends it to self.param_groups as { "params": [p0, p1, p2], "lr": 0.001,"weight_decay": 0.1}
        ## this self.param_groups will be returned at all the ranks
        ## We access the latest validated parameter group through self.param_groups[-1]
        ## this whole thing is called as "normalizing" the parameter group

        full_group = self.param_groups[-1]
        local_parameters = []
        for parameter in full_group["params"]:
            owner_rank = (self.next_param_index % self.world_size)
            self.params_with_owners.append((parameter,owner_rank))
            if owner_rank == self.rank:
                local_parameters.append(parameter)
            self.next_param_index +=1
        local_group = {key:value for key,value in full_group.items() if key !="params"}
        local_group["params"] = local_parameters
        self.local_param_groups.append(local_group)
        if self.local_optimizer is not None:
            self.local_optimizer.add_param_group(local_group)

        


