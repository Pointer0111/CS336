import argparse
import timeit
import torch
import numpy as np
import sys
import os
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from contextlib import nullcontext


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../cs336-basics')))
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.optimizer import AdamW
from cs336_basics.nn_utils import cross_entropy

def main():
    parser = argparse.ArgumentParser(description="Transformer基准测试脚本")
    parser.add_argument('--vocab_size', type=int, default=1000)
    parser.add_argument('--context_length', type=int, default=128)
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--d_ff', type=int, default=256)
    parser.add_argument('--rope_theta', type=float, default=10000.0)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--warmup_steps', type=int, default=5)
    parser.add_argument('--n_steps', type=int, default=10)
    parser.add_argument('--mode', type=str, choices=['forward', 'forward_backward'], default='forward')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--autocast', type=bool, default=False)
    parser.add_argument('--profile_memory', action='store_true', help='是否监控CUDA内存')
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    # 初始化模型
    model = BasicsTransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta
    ).to(args.device)
    model.train()

    optimizer = AdamW(model.parameters(), lr=1e-3)

    # 随机生成一批数据
    x = torch.randint(0, args.vocab_size, (args.batch_size, args.context_length), device=args.device)
    y = torch.randint(0, args.vocab_size, (args.batch_size, args.context_length), device=args.device)

    # 预热
    for _ in range(args.warmup_steps):
        out = model(x)
        if args.mode == 'forward_backward':
            loss = cross_entropy(out, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        if args.device.startswith('cuda'):
            torch.cuda.synchronize()

    # 正式计时
    forward_times = []
    backward_times = []
    for _ in range(args.n_steps):
        # 前向计时
        start_forward = timeit.default_timer()
        autocast_context = autocast(device_type=args.device, dtype=torch.bfloat16) if args.autocast else nullcontext()
        with autocast_context:
            if args.profile_memory and args.device.startswith('cuda'):
                torch.cuda.memory._record_memory_history(max_entries=1000000)
                out = model(x)
                if args.mode == "forward":
                    torch.cuda.memory._dump_snapshot("forward.pickle")
                    torch.cuda.memory._record_memory_history(enabled=None)
            else:
                out = model(x)
        if args.device.startswith('cuda'):
            torch.cuda.synchronize()
        end_forward = timeit.default_timer()
        forward_times.append(end_forward - start_forward)

        # 后向计时（仅在 forward_backward 模式下）
        if args.mode == 'forward_backward':
            with autocast_context:
                loss = cross_entropy(out, y)
            start_backward = timeit.default_timer()
            if args.autocast:
                scaler = GradScaler()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                if args.profile_memory and args.device.startswith('cuda'):
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    torch.cuda.memory._dump_snapshot("full_step.pickle")
                    torch.cuda.memory._record_memory_history(enabled=None)
                else:
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
            if args.device.startswith('cuda'):
                torch.cuda.synchronize()
            end_backward = timeit.default_timer()
            backward_times.append(end_backward - start_backward)

    print(f"前向传播: 平均每步耗时: {np.mean(forward_times):.6f} 秒, 标准差: {np.std(forward_times):.6f} 秒")
    if args.mode == 'forward_backward':
        print(f"反向传播: 平均每步耗时: {np.mean(backward_times):.6f} 秒, 标准差: {np.std(backward_times):.6f} 秒")
    # print(f"参数: {vars(args)}")

if __name__ == '__main__':
    main()
