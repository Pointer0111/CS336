# 辅助函数
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os


def generate_sample_data():
    batch_size = 128
    num_dim = 1024
    data = torch.randn(batch_size, num_dim)
    return data


def setup(rank: int, world_size: int):
    # Specify where master lives (rank 0), used to coordinate (actual data goes through NCCL)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    if torch.cuda.is_available():
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
    else:
        dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def get_device(rank: int):
    return torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")


def spawn(func, world_size: int, *args):
    """
    辅助函数，用于启动分布式训练进程
    
    Args:
        func: 要在每个进程中执行的函数，必须接受rank和world_size作为前两个参数
        world_size: 进程数量
        *args: 传递给func的额外位置参数
    """
    mp.spawn(
        func,
        args=(world_size,) + args,
        nprocs=world_size,
        join=True
    )


def render_duration(duration):
    """将时间格式化为可读字符串"""
    if duration < 1e-6:
        return f"{duration * 1e9:.2f} ns"
    elif duration < 1e-3:
        return f"{duration * 1e6:.2f} μs"
    elif duration < 1:
        return f"{duration * 1e3:.2f} ms"
    else:
        return f"{duration:.2f} s"


def get_init_params(d_in: int, d_out: int, rank: int):
    """
    初始化MLP参数
    
    Args:
        d_in: 输入维度
        d_out: 输出维度  
        rank: 进程rank，用于设置随机种子确保不同进程有不同的初始化
    
    Returns:
        torch.nn.Parameter: 可训练的参数张量
    """
    # 设置随机种子，确保不同rank有不同的参数初始化
    torch.manual_seed(42 + rank)
    
    # 获取设备
    device = get_device(rank)
    
    # 使用Xavier初始化
    std = torch.sqrt(torch.tensor(2.0 / (d_in + d_out)))
    param = torch.randn(d_out, d_in, device=device) * std
    
    # 转换为可训练参数
    return torch.nn.Parameter(param, requires_grad=True)


def summarize_tensor(tensor):
    """
    总结张量的统计信息
    
    Args:
        tensor: 要总结的张量
        
    Returns:
        str: 包含张量形状、均值、标准差等信息的字符串
    """
    if tensor is None:
        return "None"
    
    with torch.no_grad():
        mean_val = tensor.mean().item()
        std_val = tensor.std().item()
        min_val = tensor.min().item()
        max_val = tensor.max().item()
        
    return f"shape={list(tensor.size())}, mean={mean_val:.4f}, std={std_val:.4f}, range=[{min_val:.4f}, {max_val:.4f}]"