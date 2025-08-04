from ops import pytorch_softmax, manual_softmax, triton_softmax
from util import *
from tabulate import tabulate

def main():
    compiled_softmax = torch.compile(manual_softmax)
    methods = [
        ("PyTorch Softmax", pytorch_softmax),
        ("Manual Softmax", manual_softmax),
        ("Triton Softmax", triton_softmax),
        ("Compiled Softmax", compiled_softmax)
    ]

    assert check_equal(pytorch_softmax, manual_softmax)
    assert check_equal(manual_softmax, triton_softmax)
    assert check_equal(manual_softmax, compiled_softmax)

    results = []
    profiles = {}

    for name, func in methods:
        time = benchmark(name, run_operation1(dim=16384, operation=func))
        prof = profile(name, run_operation1(dim=16384, operation=func))
        results.append([name, time])
        print(f"{name} 性能分析结果：")
        print(prof)

    print(tabulate(results, headers=["Softmax实现", "时间(ms)"]))

if __name__ == "__main__":
    main()