import torch
import triton
import triton.language as tl


# gelu激活函数
def pytorch_gelu(x: torch.Tensor):
    # 使用tanh近似来匹配我们的实现
    return torch.nn.functional.gelu(x, approximate="tanh")

def manual_gelu(x: torch.Tensor):
    return 0.5 * x * (1 + torch.tanh(0.79788456 * (x + 0.044715 * x * x * x)))

@triton.jit
def triton_gelu_kernel(x_ptr, y_ptr, numel, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    blk_start = pid*BLOCK_SIZE

    offsets = blk_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel

    x = tl.load(x_ptr + offsets, mask=mask)

    a = 0.79788456 * (x + 0.044715 * x * x * x)

    exp = tl.exp(2 * a)
    tanh = (exp - 1) / (exp + 1)
    y = 0.5 * x * (1 + tanh)

    tl.store(y_ptr + offsets, y, mask=mask)

def triton_gelu(x: torch.Tensor):
    assert x.is_cuda
    assert x.is_contiguous()

    y = torch.empty_like(x)
    numel = x.numel()
    block_size = 1024
    num_blks = triton.cdiv(numel, block_size)

    triton_gelu_kernel[(num_blks,)](x, y, numel, BLOCK_SIZE=block_size)  # 这里直接用字面量

    return y

# softmax函数

def pytorch_softmax(x: torch.Tensor):
    return torch.nn.functional.softmax(x, dim=-1)

def manual_softmax(x: torch.Tensor):
    # M: 行数, N: 列数
    M, N = x.shape
    # 计算每行的最大值（MN次读取，M次写入）
    x_max = x.max(dim=1)[0]
    # 减去最大值（MN + M次读取，MN次写入）
    x = x - x_max[:, None]
    # 指数化（MN次读取，MN次写入）
    numerator = torch.exp(x)
    # 计算归一化常数（MN次读取，M次写入）
    denominator = numerator.sum(dim=1)
    # 归一化（MN次读取，MN次写入）
    y = numerator / denominator[:, None]
    # 总计：5MN + M次读取，3MN + 2M次写入
    # 理论上应该有MN次读取，MN次写入（4倍加速！）
    return y

@triton.jit
def triton_softmax_kernel(x_ptr, y_ptr, x_row_stride, y_row_stride, num_cols, BLOCK_SIZE: tl.constexpr):
    assert num_cols <= BLOCK_SIZE

    # 独立处理每一行
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)

    # 从全局内存读取
    x_start_ptr = x_ptr + row_idx * x_row_stride
    x_ptrs = x_start_ptr + col_offsets
    x_row = tl.load(x_ptrs, mask=col_offsets < num_cols, other=float("-inf"))

    # 计算
    x_row = x_row - tl.max(x_row, axis=0)
    numerator = tl.exp(x_row)
    denominator = tl.sum(numerator, axis=0)
    y_row = numerator / denominator

    # 写回全局内存
    y_start_ptr = y_ptr + row_idx * y_row_stride
    y_ptrs = y_start_ptr + col_offsets
    tl.store(y_ptrs, y_row, mask=col_offsets < num_cols)

def triton_softmax(x: torch.Tensor):
    # 分配输出张量
    y = torch.empty_like(x)
    # 确定网格
    M, N = x.shape                          # 行数 x 列数
    block_size = triton.next_power_of_2(N)  # 每个块包含所有列
    num_blocks = M                          # 每个块是一行
    # 启动内核
    triton_softmax_kernel[(M,)](
        x_ptr=x, y_ptr=y,
        x_row_stride=x.stride(0), y_row_stride=y.stride(0),
        num_cols=N, BLOCK_SIZE=block_size
    )
    return y



