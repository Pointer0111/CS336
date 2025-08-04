import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


class DDPIndividualParameters(torch.nn.Module):
    """
    Returns a torch.nn.Module container that handles
    parameter broadcasting and gradient synchronization for
    distributed data parallel training.

    This container should overlaps communication with backprop computation
    by asynchronously communicating gradients as they are ready
    in the backward pass. The gradient for each parameter tensor
    is individually communicated.

    Args:
        module: torch.nn.Module
            Underlying model to wrap with DDP.
    Returns:
        Instance of a DDP class.
    """
    
    def __init__(self, module):
        super().__init__()
        self.module = module
        self.param_handles = {}
        self.hook_handles = []
        
        # Broadcast parameters from rank 0 to all other ranks
        self._broadcast_parameters()
        
        # Register post accumulate grad hooks for asynchronous gradient communication
        self._register_post_accumulate_grad_hooks()
    
    def _broadcast_parameters(self):
        """Broadcast parameters from rank 0 to all other ranks"""
        for param in self.module.parameters():
            dist.broadcast(param.data, src=0)
    
    def _register_post_accumulate_grad_hooks(self):
        """Register post accumulate grad hooks to start async gradient communication"""
        for name, param in self.module.named_parameters():
            if param.requires_grad:
                hook_handle = param.register_post_accumulate_grad_hook(
                    lambda p, param_name=name: self._start_async_allreduce(param_name, p)
                )
                self.hook_handles.append(hook_handle)
    
    def _start_async_allreduce(self, name, param):
        """Start async all-reduce for a parameter's gradient"""
        if param.grad is not None:
            # Make a copy of the gradient to avoid in-place operations
            grad_copy = param.grad.clone()
            # Start async all-reduce for this parameter's gradient
            handle = dist.all_reduce(grad_copy, async_op=True)
            self.param_handles[name] = (handle, grad_copy, param)
    
    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)
    
    def finish_gradient_synchronization(self):
        """Wait for all async gradient communications to complete"""
        world_size = dist.get_world_size()
        
        # Wait for all handles and update gradients
        for _, (handle, grad_copy, param) in self.param_handles.items():
            handle.wait()
            # Average the gradient and update the parameter's gradient
            grad_copy.div_(world_size)
            param.grad.copy_(grad_copy)
        
        # Clear handles for next iteration
        self.param_handles.clear()
    
    def __del__(self):
        """Cleanup hook handles when the object is destroyed"""
        for handle in self.hook_handles:
            handle.remove()


def ddp_individual_parameters_on_after_backward(ddp_model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    """
    Code to run after the backward pass is completed, but before we take
    an optimizer step.

    Args:
        ddp_model: torch.nn.Module
            DDP-wrapped model.
        optimizer: torch.optim.Optimizer
            Optimizer being used with the DDP-wrapped model.
    """
    # Wait for all gradient synchronization to complete
    ddp_model.finish_gradient_synchronization()


class DDPBucketed(torch.nn.Module):
    """
    Returns a torch.nn.Module container that handles
    parameter broadcasting and gradient synchronization for
    distributed data parallel training.

    This container should overlaps communication with backprop computation
    by asynchronously communicating buckets of gradients as they are ready
    in the backward pass.

    Args:
        module: torch.nn.Module
            Underlying model to wrap with DDP.
        bucket_size_mb: The bucket size, in megabytes. If None, use a single
            bucket of unbounded size.
    Returns:
        Instance of a DDP class.
    """

    def __init__(self, module: torch.nn.Module, bucket_size_mb: float):
        super().__init__()
        self.module = module
        self.bucket_size_mb = bucket_size_mb
        
        # 存储桶信息
        self.buckets = []  # 每个桶包含参数列表
        self.param_to_bucket = {}  # 参数名到桶索引的映射
        self.bucket_handles = {}  # 存储每个桶的异步通信句柄
        self.bucket_ready_params = {}  # 跟踪每个桶中已准备好的参数
        self.hook_handles = []
        
        # 广播参数并创建桶
        self._broadcast_parameters()
        self._create_buckets()
        self._register_post_accumulate_grad_hooks()
    
    def _broadcast_parameters(self):
        """从rank 0广播参数到所有其他rank"""
        for param in self.module.parameters():
            dist.broadcast(param.data, src=0)
    
    def _create_buckets(self):
        """将参数分组到桶中"""
        current_bucket = []
        current_bucket_size = 0.0
        bucket_idx = 0
        
        # 按参数的逆序遍历（类似PyTorch DDP的做法）
        param_list = list(self.module.named_parameters())
        param_list.reverse()
        
        for name, param in param_list:
            if not param.requires_grad:
                continue
                
            # 计算参数大小（以MB为单位）
            param_size_mb = param.numel() * param.element_size() / (1024 * 1024)
            
            # 如果当前桶已经有参数且添加这个参数会超过桶大小限制，则创建新桶
            if current_bucket and current_bucket_size + param_size_mb > self.bucket_size_mb:
                self.buckets.append(current_bucket)
                self.bucket_ready_params[bucket_idx] = set()
                bucket_idx += 1
                current_bucket = []
                current_bucket_size = 0.0
            
            # 将参数添加到当前桶
            current_bucket.append((name, param))
            current_bucket_size += param_size_mb
            self.param_to_bucket[name] = bucket_idx
        
        # 添加最后一个桶
        if current_bucket:
            self.buckets.append(current_bucket)
            self.bucket_ready_params[bucket_idx] = set()
    
    def _register_post_accumulate_grad_hooks(self):
        """注册post accumulate grad hooks来启动异步梯度通信"""
        for name, param in self.module.named_parameters():
            if param.requires_grad:
                hook_handle = param.register_post_accumulate_grad_hook(
                    lambda p, param_name=name: self._on_gradient_ready(param_name, p)
                )
                self.hook_handles.append(hook_handle)
    
    def _on_gradient_ready(self, param_name: str, param: torch.Tensor):
        """当参数梯度准备好时调用"""
        if param.grad is None:
            return
            
        bucket_idx = self.param_to_bucket[param_name]
        self.bucket_ready_params[bucket_idx].add(param_name)
        
        # 检查这个桶中的所有参数是否都准备好了
        bucket_param_names = {name for name, _ in self.buckets[bucket_idx]}
        if self.bucket_ready_params[bucket_idx] == bucket_param_names:
            # 桶中所有参数都准备好了，启动异步all-reduce
            self._start_bucket_allreduce(bucket_idx)
    
    def _start_bucket_allreduce(self, bucket_idx: int):
        """为指定桶启动异步all-reduce"""
        if bucket_idx in self.bucket_handles:
            return  # 已经启动了
        
        # 收集桶中所有参数的梯度
        bucket_grads = []
        bucket_params = []
        
        for name, param in self.buckets[bucket_idx]:
            if param.grad is not None:
                bucket_grads.append(param.grad.clone().flatten())
                bucket_params.append(param)
        
        if bucket_grads:
            # 将所有梯度拼接成一个tensor
            flattened_grad = torch.cat(bucket_grads)
            
            # 启动异步all-reduce
            handle = dist.all_reduce(flattened_grad, async_op=True)
            self.bucket_handles[bucket_idx] = (handle, flattened_grad, bucket_params, bucket_grads)
    
    def forward(self, *args, **kwargs):
        """前向传播"""
        return self.module(*args, **kwargs)
    
    def finish_gradient_synchronization(self):
        """等待所有异步梯度通信完成"""
        world_size = dist.get_world_size()
        
        # 等待所有桶的通信完成并更新梯度
        for bucket_idx, (handle, flattened_grad, bucket_params, original_grads) in self.bucket_handles.items():
            handle.wait()
            
            # 平均梯度
            flattened_grad.div_(world_size)
            
            # 将平均后的梯度分配回对应的参数
            offset = 0
            for i, param in enumerate(bucket_params):
                grad_size = original_grads[i].numel()
                param.grad.copy_(flattened_grad[offset:offset + grad_size].view_as(param.grad))
                offset += grad_size
        
        # 清理状态，为下一次迭代做准备
        self.bucket_handles.clear()
        for bucket_idx in self.bucket_ready_params:
            self.bucket_ready_params[bucket_idx].clear()
    
    def __del__(self):
        """清理hook handles"""
        for handle in self.hook_handles:
            handle.remove()


def ddp_bucketed_on_after_backward(ddp_model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    """
    Code to run after the backward pass is completed, but before we take
    an optimizer step.

    Args:
        ddp_model: torch.nn.Module
            DDP-wrapped model.
        optimizer: torch.optim.Optimizer
            Optimizer being used with the DDP-wrapped model.
    """
    # For example: ddp_model.finish_gradient_synchronization()
    ddp_model.finish_gradient_synchronization()


def ddp_bucketed_on_train_batch_start(ddp_model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    """
    Code to run at the very start of the training step.

    Args:
        ddp_model: torch.nn.Module
            DDP-wrapped model.
        optimizer: torch.optim.Optimizer
            Optimizer being used with the DDP-wrapped model.
    """
    # 在训练批次开始时，我们需要确保模型处于准备状态
    # 对于分桶DDP，通常不需要特殊操作，因为状态在finish_gradient_synchronization中已经清理
    pass