import torch
import time
from typing import Callable
from statistics import mean

def benchmark(description: str, run: Callable, num_warmups: int = 1, num_trials: int = 3):
    """对函数进行基准测试，运行num_trials次并返回所有时间"""
    # 预热：前几次可能因为编译和缓存问题较慢
    # 由于我们会多次运行内核，重要的是稳态时间
    for _ in range(num_warmups):
        run()
    if torch.cuda.is_available():
        torch.cuda.synchronize()  # 等待CUDA线程完成（重要！）
    # 现在开始真正计时！
    times: list[float] = []
    for trial in range(num_trials):  # 多次运行以捕获方差
        start_time = time.time()
        run()  # 实际执行计算
        if torch.cuda.is_available():
            torch.cuda.synchronize()  # 等待CUDA线程完成（重要！）
        end_time = time.time()
        times.append((end_time - start_time) * 1000)
    mean_time = mean(times)
    return mean_time

from torch.profiler import ProfilerActivity

def profile(description: str, run: Callable, num_warmups: int = 1, with_stack: bool = False):
    # 预热
    for _ in range(num_warmups):
        run()
    if torch.cuda.is_available():
        torch.cuda.synchronize()  # 等待CUDA线程完成（重要！）
    # 使用分析器运行代码
    with torch.profiler.profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        # 输出堆栈跟踪用于可视化
        with_stack=with_stack,
        # 启用详细模式，输出详细报告
        experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True)
    ) as prof:
        run()
        if torch.cuda.is_available():
            torch.cuda.synchronize()  # 等待CUDA线程完成（重要！）
    # 打印表格
    table = prof.key_averages().table(
        sort_by="cuda_time_total",
        max_name_column_width=80,
        row_limit=10
    )
    # 写入堆栈跟踪可视化
    if with_stack:
        text_path = f"var/stacks_{description}.txt"
        svg_path = f"var/stacks_{description}.svg"
        prof.export_stacks(text_path, "self_cuda_time_total")
    return table

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_operation1(dim: int, operation: Callable) -> Callable:
    # 设置：创建一个随机dim x dim矩阵
    x = torch.randn(dim, dim, device=get_device())
    # 返回执行操作的函数
    return lambda: operation(x)

def run_operation2(dim: int, operation: Callable) -> Callable:
    # 设置：创建两个随机dim x dim矩阵
    x = torch.randn(dim, dim, device=get_device())
    y = torch.randn(dim, dim, device=get_device())
    # 返回执行操作的函数
    return lambda: operation(x, y)

def check_equal(op1: Callable, op2: Callable):
    x = torch.randn(1024, 1024, device=get_device())
    y1 = op1(x)
    y2 = op2(x)
    # 使用更宽松的容差来处理浮点运算的微小差异
    return torch.allclose(y1, y2, rtol=1e-4, atol=1e-4)
