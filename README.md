# CS336 Spring 2025 Assignment 5: Alignment

For a full description of the assignment, see the assignment handout at
[cs336_spring2025_assignment5_alignment.pdf](./cs336_spring2025_assignment5_alignment.pdf)

We include a supplemental (and completely optional) assignment on safety alignment, instruction tuning, and RLHF at [cs336_spring2025_assignment5_supplement_safety_rlhf.pdf](./cs336_spring2025_assignment5_supplement_safety_rlhf.pdf)

If you see any issues with the assignment handout or code, please feel free to
raise a GitHub issue or open a pull request with a fix.

## Setup

As in previous assignments, we use `uv` to manage dependencies.

1. Install all packages except `flash-attn` and `xformers`, then all packages (`flash-attn` is weird, and `xformers` needs to be installed after `torch`)
```
uv sync --no-install-package flash-attn --no-install-package xformers
uv sync
```

2. Run unit tests:

``` sh
uv run pytest
```

Initially, all tests should fail with `NotImplementedError`s.
To connect your implementation to the tests, complete the
functions in [./tests/adapters.py](./tests/adapters.py).

## 注意事项
25.7.23：
- 由于实验室服务器CUDA驱动太低，用不了flash-attn和vllm（但 vllm 的预编译包要求必须是 11.8）

- 由于部署LLaMA-3.1-70B-Instruct模型需要至少两块H100，因此没法做A5的附加作业。