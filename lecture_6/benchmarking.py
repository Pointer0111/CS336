from util import *
from mlp import run_mlp

def benchmarking():
    benchmark("sleep", lambda : time.sleep(50 / 1000))

    if torch.cuda.is_available():
        dims = (1024, 2048, 4096, 8192, 16384) 
    else:
        dims = (1024, 2048) 
    matmul_results = [] 
    for dim in dims:
        result = benchmark(f"matmul(dim={dim})", run_operation2(dim=dim, operation=lambda a, b: a @ b))
        matmul_results.append((dim, result))  
        
    dim = 256  
    num_layers = 4   
    batch_size = 256  
    num_steps = 2 
    mlp_base = benchmark("run_mlp", run_mlp(dim=dim, num_layers=num_layers, batch_size=batch_size, num_steps=num_steps)) 
    # 缩放步数
    step_results = []
    for scale in (2, 3, 4, 5):
        result = benchmark(f"run_mlp({scale}x num_steps)", 
                         run_mlp(dim=dim, num_layers=num_layers, 
                                batch_size=batch_size, num_steps=scale * num_steps)) 
        step_results.append((scale, result)) 
    # 缩放层数
    layer_results = []
    for scale in (2, 3, 4, 5):
        result = benchmark(f"run_mlp({scale}x num_layers)", 
                         run_mlp(dim=dim, num_layers=scale * num_layers, 
                                batch_size=batch_size, num_steps=num_steps)) 
        layer_results.append((scale, result)) 
    # 缩放批次大小
    batch_results = []
    for scale in (2, 3, 4, 5):
        result = benchmark(f"run_mlp({scale}x batch_size)", 
                         run_mlp(dim=dim, num_layers=num_layers, 
                                batch_size=scale * batch_size, num_steps=num_steps)) 
        batch_results.append((scale, result))  
    # 缩放维度
    dim_results = []
    for scale in (2, 3, 4, 5):
        result = benchmark(f"run_mlp({scale}x dim)", 
                         run_mlp(dim=scale * dim, num_layers=num_layers, 
                                batch_size=batch_size, num_steps=num_steps)) 
        dim_results.append((scale, result)) 

    print("matmul_results:")
    print(matmul_results)
    print("--------------------------------")
    print("step_results:")
    print(step_results)
    print("--------------------------------")
    print("layer_results:")
    print(layer_results)
    print("--------------------------------")
    print("batch_results:")
    print(batch_results)
    print("--------------------------------")
    print("dim_results:")
    print(dim_results)


if __name__ == "__main__":
    benchmarking()