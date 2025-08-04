import torch
import time
import itertools
import gc
import os

# 假设attention函数为标准的scaled dot-product attention
# Q, K, V: [batch, seq_len, d_model]
def attention(Q, K, V):
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)
    attn = torch.softmax(scores, dim=-1)
    return torch.matmul(attn, V)

# 固定batch size
batch_size = 8
# 不使用多头
# d_model和seq_len的组合
D_MODELS = [16, 32, 64, 128]
SEQ_LENS = [256, 1024, 4096, 8192, 16384]

# 检查是否有GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 记录结果
def get_mem():
    if device.type == 'cuda':
        return torch.cuda.max_memory_allocated(device) / 1024 / 1024  # MB
    else:
        return 0

def reset_mem():
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)

def benchmark():
    print(f"device: {device}")
    for d_model, seq_len in itertools.product(D_MODELS, SEQ_LENS):
        print(f"\n==== d_model={d_model}, seq_len={seq_len} ====")
        # 创建输入
        Q = torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=True)
        K = torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=True)
        V = torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=True)
        # warmup
        for _ in range(10):
            out = attention(Q, K, V)
            if device.type == 'cuda':
                torch.cuda.synchronize()
        # 前向
        fwd_times = []
        reset_mem()
        for _ in range(100):
            start = time.time()
            out = attention(Q, K, V)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            fwd_times.append(time.time() - start)
        mem_used = get_mem()
        print(f"前向平均耗时: {sum(fwd_times)/len(fwd_times):.6f}s, 峰值显存: {mem_used:.2f}MB")
        # 反向
        bwd_times = []
        for _ in range(100):
            out = attention(Q, K, V)
            grad = torch.randn_like(out)
            start = time.time()
            out.backward(grad, retain_graph=True)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            bwd_times.append(time.time() - start)
            if Q.grad is not None:
                Q.grad.zero_()
            if K.grad is not None:
                K.grad.zero_()
            if V.grad is not None:
                V.grad.zero_()
        print(f"反向平均耗时: {sum(bwd_times)/len(bwd_times):.6f}s")
        # 清理
        del Q, K, V, out, grad
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()

if __name__ == '__main__':
    benchmark()
    