from util import *
import torch.nn.functional as F
import torch.distributed as dist

def data_parallelism(rank: int, world_size: int, data: torch.Tensor, num_layers: int, num_steps: int):
    
    setup(rank, world_size)
    
    # Get the slice of data for this rank (in practice, each rank should load only its own data)
    batch_size = data.size(0)  # @inspect batch_size
    num_dim = data.size(1)  # @inspect num_dim
    local_batch_size = batch_size // world_size  # @inspect local_batch_size
    start_index = rank * local_batch_size  # @inspect start_index
    end_index = start_index + local_batch_size  # @inspect end_index
    data = data[start_index : end_index].to(get_device(rank))
    
    # Create MLP parameters params[0], ..., params[num_layers - 1] (each rank has all parameters)
    params = [get_init_params(num_dim, num_dim, rank) for i in range(num_layers)]
    optimizer = torch.optim.AdamW(params, lr=1e-3)  # Each rank has own optimizer state
    
    for step in range(num_steps):
        # Forward pass
        x = data
        for param in params:
            x = x @ param
            x = F.gelu(x)
        loss = x.square().mean()  # Loss function is average squared magnitude
        
        # Backward pass
        loss.backward()
        
        # Sync gradients across workers (only difference between standard training and DDP)
        for param in params:
            dist.all_reduce(tensor=param.grad, op=dist.ReduceOp.AVG, async_op=False)
        
        # Update parameters
        optimizer.step()
        print(f"[data_parallelism] Rank {rank}: step = {step}, loss = {loss.item()}, params = {[summarize_tensor(params[i]) for i in range(num_layers)]}", flush=True)
    
    cleanup()


def tensor_parallelism(rank: int, world_size: int, data: torch.Tensor, num_layers: int):
    
    setup(rank, world_size)
    
    data = data.to(get_device(rank))
    batch_size = data.size(0)  # @inspect batch_size
    num_dim = data.size(1)  # @inspect num_dim
    local_num_dim = num_dim // world_size  # Shard `num_dim`  @inspect local_num_dim
    
    # Create model (each rank gets 1/world_size of the parameters)
    params = [get_init_params(local_num_dim, num_dim, rank) for i in range(num_layers)]
    
    # Forward pass
    x = data
    for i in range(num_layers):
        # Compute activations (batch_size x local_num_dim)
        x = x @ params[i]  # Note: this is only on a slice of the parameters
        x = F.gelu(x)
        
        # Allocate memory for activations (world_size x batch_size x local_num_dim)
        activations = [torch.empty(batch_size, local_num_dim, device=get_device(rank)) for _ in range(world_size)]
        
        # Send activations via all gather
        dist.all_gather(tensor_list=activations, tensor=x, async_op=False)
        
        # Concatenate them to get batch_size x num_dim
        x = torch.cat(activations, dim=1)
    
    print(f"[tensor_parallelism] Rank {rank}: forward pass produced activations {summarize_tensor(x)}", flush=True)
    
    # Backward pass

    
    
    cleanup()



def pipeline_parallelism(rank: int, world_size: int, data: torch.Tensor, num_layers: int, num_micro_batches: int):
    
    setup(rank, world_size)
    
    # Use all the data
    data = data.to(get_device(rank))
    batch_size = data.size(0)  # @inspect batch_size
    num_dim = data.size(1)  # @inspect num_dim
    
    
    # Split up layers
    local_num_layers = num_layers // world_size  # @inspect local_num_layers
    

    # Each rank gets a subset of layers
    local_params = [get_init_params(num_dim, num_dim, rank) for i in range(local_num_layers)]
    
    # Forward pass
    
    # Break up into micro batches to minimize the bubble
    micro_batch_size = batch_size // num_micro_batches  # @inspect micro_batch_size
    if rank == 0:
        # The data
        micro_batches = data.chunk(chunks=num_micro_batches, dim=0)
    else:
        # Allocate memory for activations
        micro_batches = [torch.empty(micro_batch_size, num_dim, device=get_device(rank)) for _ in range(num_micro_batches)]
    
    for x in micro_batches:
        
        # Get activations from previous rank
        if rank - 1 >= 0:
            dist.recv(tensor=x, src=rank - 1)
        
        # Compute layers assigned to this rank
        for param in local_params:
            x = x @ param
            x = F.gelu(x)
        
        # Send to the next rank
        if rank + 1 < world_size:
            print(f"[pipeline_parallelism] Rank {rank}: sending {summarize_tensor(x)} to rank {rank + 1}", flush=True)
            dist.send(tensor=x, dst=rank + 1)

    # 可以优化的点：重叠通信/计算消除流水线气泡

    # Backward pass




    cleanup()



if __name__ == "__main__":
    data = generate_sample_data()

    # spawn(data_parallelism, 2, data, 4, 1)
    
    # spawn(tensor_parallelism, 2, data, 4)
    
    spawn(pipeline_parallelism, 2, data, 4, 4)