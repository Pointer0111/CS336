import subprocess
import pandas as pd
import re
from tabulate import tabulate

model_configs = [
    {"size": "small", "d_model": 768, "d_ff": 3072, "num_layers": 12, "num_heads": 12},
    # {"size": "medium", "d_model": 1024, "d_ff": 4096, "num_layers": 24, "num_heads": 16},
    # {"size": "large", "d_model": 1280, "d_ff": 5120, "num_layers": 36, "num_heads": 20},
    # {"size": "xl", "d_model": 1600, "d_ff": 6400, "num_layers": 48, "num_heads": 25},
    # {"size": "2.7B", "d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32},
]

common_args = [
    '--warmup_steps', '5',
    '--n_steps', '10',
    '--mode', 'forward_backward',
    '--batch_size', '8',
    '--context_length', '32',
    '--vocab_size', '1000',
    '--device', 'cuda', 
]

def parse_output(output):
    m = re.search(r"平均每步耗时: ([0-9.]+) 秒, 标准差: ([0-9.]+) 秒", output)
    if m:
        return float(m.group(1)), float(m.group(2))
    return None, None

results = []
for cfg in model_configs:
    args = [
        'python3', 'cs336_systems/benchmark.py',
        '--d_model', str(cfg['d_model']),
        '--d_ff', str(cfg['d_ff']),
        '--num_layers', str(cfg['num_layers']),
        '--num_heads', str(cfg['num_heads']),
    ] + common_args
    print(f"正在测试: {cfg['size']} ...")
    proc = subprocess.run(args, capture_output=True, text=True)
    avg, std = parse_output(proc.stdout)
    results.append({
        'size': cfg['size'],
        'd_model': cfg['d_model'],
        'd_ff': cfg['d_ff'],
        'num_layers': cfg['num_layers'],
        'num_heads': cfg['num_heads'],
        'avg_time': avg,
        'std_time': std
    })

df = pd.DataFrame(results)
md_table = str(df.to_markdown(index=False))
with open('benchmark_results.md', 'w') as f:
    f.write(md_table)
print(md_table)
print("结果已保存到 benchmark_results.md") 