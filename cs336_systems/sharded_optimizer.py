import torch
import torch.distributed as dist
from typing import Dict, List, Any, Optional, Callable, Union
from collections import defaultdict

class ShardedOptimizer(torch.optim.Optimizer):
    """
    分片优化器实现，将优化器状态分片到不同的rank上，
    每个rank只处理部分参数的优化，然后同步更新后的参数。
    """
    
    def __init__(self, params, optimizer_cls: type, **kwargs):
        """
        初始化分片优化器
        
        Args:
            params: 要优化的参数集合
            optimizer_cls: 要包装的优化器类型 (如 torch.optim.AdamW)
            **kwargs: 传递给优化器构造函数的关键字参数
        """
        # 将参数转换为列表以便索引
        if hasattr(params, '__iter__') and not isinstance(params, (torch.Tensor, dict)):
            self.all_params = [p for p in params if isinstance(p, torch.Tensor)]
        else:
            self.all_params = [params] if isinstance(params, torch.Tensor) else []
        
        # 获取分布式信息
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        
        # 计算当前rank应该处理的参数分片
        self.param_shards = self._shard_parameters(self.all_params)
        
        # 为分片的参数创建本地优化器
        if self.param_shards:
            self.local_optimizer = optimizer_cls(self.param_shards, **kwargs)
            # 直接使用本地优化器的参数组和默认值，不调用父类构造函数
            self.param_groups = self.local_optimizer.param_groups
            self.defaults = self.local_optimizer.defaults
            self.state = self.local_optimizer.state
        else:
            # 如果当前rank没有分配到参数，创建空的优化器
            self.local_optimizer = None
            self.param_groups = []
            self.defaults = kwargs
            self.state = defaultdict(dict)
    
    def _shard_parameters(self, params):
        """
        将参数分片到不同的rank
        使用轮询方式分配参数，确保负载均衡
        """
        my_params = []
        for i, param in enumerate(params):
            if param.requires_grad and i % self.world_size == self.rank:
                my_params.append(param)
        return my_params
    
    def zero_grad(self, set_to_none: bool = False):
        """清零梯度"""
        if self.local_optimizer is not None:
            self.local_optimizer.zero_grad(set_to_none=set_to_none)
    
    def step(self, closure: Optional[Callable] = None):
        """
        执行优化步骤
        1. 在本地参数上执行优化器步骤
        2. 将更新后的参数广播到所有rank以保持同步
        """
        loss = None
        
        # 在本地参数上执行优化步骤
        if self.local_optimizer is not None:
            loss = self.local_optimizer.step(closure)
        
        # 同步所有参数到所有rank
        if dist.is_initialized():
            self._synchronize_parameters()
        
        return loss
    
    def _synchronize_parameters(self):
        """
        将每个rank更新的参数广播到所有其他rank
        确保所有rank的模型参数保持一致
        """
        for i, param in enumerate(self.all_params):
            if not param.requires_grad:
                continue
            
            # 确定哪个rank拥有这个参数
            owner_rank = i % self.world_size
            
            # 参数的拥有者将其广播给所有其他rank
            dist.broadcast(param.data, src=owner_rank)
    
    def state_dict(self):
        """获取优化器状态字典"""
        if self.local_optimizer is not None:
            return self.local_optimizer.state_dict()
        else:
            return {'state': {}, 'param_groups': []}
    
    def load_state_dict(self, state_dict):
        """加载优化器状态字典"""
        if self.local_optimizer is not None:
            self.local_optimizer.load_state_dict(state_dict)
    
    def add_param_group(self, param_group: dict):
        """
        添加参数组到分片优化器
        这在训练过程中动态添加参数时很有用（如渐进式解冻）
        """
        # 确保param_group包含'params'键
        if 'params' not in param_group:
            raise ValueError("param group must contain 'params' key")
        
        # 获取新的参数列表
        new_params = param_group['params']
        if not isinstance(new_params, list):
            new_params = list(new_params)
        
        # 将新参数添加到总参数列表
        param_start_idx = len(self.all_params)
        self.all_params.extend(new_params)
        
        # 计算当前rank应该处理的新参数
        my_new_params = []
        for i, param in enumerate(new_params):
            if param.requires_grad and (param_start_idx + i) % self.world_size == self.rank:
                my_new_params.append(param)
        
        # 如果有新的参数分配给当前rank，添加到本地优化器
        if my_new_params:
            if self.local_optimizer is None:
                # 如果之前没有本地优化器，现在需要创建一个
                # 这种情况比较复杂，为简化实现，我们假设至少有一个初始参数组
                raise RuntimeError("Cannot add param group to optimizer with no initial parameters")
            
            # 创建新的参数组，只包含分配给当前rank的参数
            new_param_group = param_group.copy()
            new_param_group['params'] = my_new_params
            
            # 添加到本地优化器
            self.local_optimizer.add_param_group(new_param_group)
            
            # 更新自己的参数组引用
            self.param_groups = self.local_optimizer.param_groups 