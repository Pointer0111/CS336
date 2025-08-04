# 系统需要安装Ninja和nvcc编译器，并且要正确
# 配置CUDA_HOME环境变量（一般是/usr/local/cuda）
# 请先创建一个cuda_gelu文件夹，编译后会生成inline_gelu.so文件


import os
import sys
from util import *
from ops import pytorch_gelu, manual_gelu, triton_gelu
from torch.utils.cpp_extension import load_inline
from tabulate import tabulate

def create_cuda_gelu():
    print("正在尝试创建CUDA版GELU函数...")
    
    # 检查是否已经编译过
    build_dir = "cuda_gelu"
    so_file = os.path.join(build_dir, "inline_gelu.so")
    
    if os.path.exists(so_file):
        print("检测到已编译的CUDA GELU模块，正在加载...")
        try:
            # 将build_dir添加到Python路径
            if build_dir not in sys.path:
                sys.path.insert(0, build_dir)
            
            # 动态导入已编译的模块
            import importlib.util
            spec = importlib.util.spec_from_file_location("inline_gelu", so_file)
            if spec is not None and spec.loader is not None:
                inline_gelu = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(inline_gelu)
                cuda_gelu = getattr(inline_gelu, "gelu")
                print("CUDA GELU模块加载成功。")
                return cuda_gelu
            else:
                raise ImportError("无法创建模块规范")
        except Exception as e:
            print(f"加载已编译模块失败: {e}")
            print("将重新编译CUDA GELU模块...")
    
    if not torch.cuda.is_available():
        print("未检测到可用的CUDA设备，无法创建CUDA GELU。")
        return None
    
    print("CUDA设备可用，正在编译CUDA GELU模块...")
    cuda_gelu_src = open("gelu.cu").read()
    cpp_gelu_src = "torch::Tensor gelu(torch::Tensor x);"
    
    module = load_inline(
        cuda_sources=[cuda_gelu_src],
        cpp_sources=[cpp_gelu_src],
        functions=["gelu"],
        extra_cflags=["-O2"],
        verbose=True,
        name="inline_gelu",
        build_directory=build_dir,
    )
    print("CUDA GELU模块编译完成。")
    cuda_gelu = getattr(module, "gelu")
    return cuda_gelu

def main():
    assert check_equal(pytorch_gelu, manual_gelu)
    assert check_equal(manual_gelu, triton_gelu)

    print("开始GELU性能测试...")
    cuda_gelu = create_cuda_gelu()
    print("正在测试PyTorch自带的GELU实现...")
    pytorch_time = benchmark("pytorch_gelu", run_operation1(dim=16384, operation=pytorch_gelu)) # @inspect pytorch_time
    print(f"PyTorch GELU耗时: {pytorch_time} ms")

    print("正在测试手动实现的GELU...")
    manual_time = benchmark("manual_gelu", run_operation1(dim=16384, operation=manual_gelu)) # @inspect manual_time
    print(f"Manual GELU耗时: {manual_time} ms")

    cuda_time = None
    if cuda_gelu is not None:
        assert check_equal(manual_gelu, cuda_gelu)
        print("正在测试CUDA实现的GELU...")
        cuda_time = benchmark("cuda_gelu", run_operation1(dim=16384, operation=cuda_gelu)) # @inspect cuda_time 
        print(f"CUDA GELU耗时: {cuda_time} ms")
        print("正在进行CUDA GELU的性能分析...")
        cuda_gelu_profile = profile("cuda_gelu", run_operation1(dim=16384, operation=cuda_gelu))
        # print("CUDA GELU性能分析结果：")
        # print(cuda_gelu_profile)
    else:
        print("CUDA GELU不可用，跳过CUDA测试")



    triton_time = benchmark("triton_gelu", run_operation1(dim=16384, operation=triton_gelu)) # @inspect triton_time
    triton_gelu_profile = profile("triton_gelu", run_operation1(dim=16384, operation=triton_gelu))
    # print("Triton GELU性能分析结果：")
    # print(triton_gelu_profile)

    compiled_gelu = torch.compile(manual_gelu)
    assert check_equal(compiled_gelu, manual_gelu)
    compiled_time = benchmark("compiled_gelu", run_operation1(dim=16384, operation=compiled_gelu)) # @inspect compiled_time
    print(f"Compiled GELU耗时: {compiled_time} ms")
    compiled_gelu_profile = profile("compiled_gelu", run_operation1(dim=16384, operation=compiled_gelu))
    print("Compiled GELU性能分析结果：")
    print(compiled_gelu_profile)



    # 制成表格
    print("各GELU实现的性能对比如下：")
    print(tabulate([
        ["PyTorch GELU", pytorch_time],
        ["Manual GELU", manual_time],
        ["CUDA GELU", cuda_time],
        ["Triton GELU", triton_time],
        ["Compiled GELU", compiled_time]
    ], headers=["GELU实现", "时间(ms)"]))

if __name__ == "__main__":
    main()