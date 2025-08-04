from jaxtyping import Float, Int
from torch import Tensor
import torch
from typing import Iterable, Optional, Callable
import math

def cross_entropy(inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]) -> Float[Tensor, ""]:
    """Given a tensor of inputs and targets, compute the average cross-entropy
    loss across examples.

    Args:
        inputs (Float[Tensor, "batch_size vocab_size"]): inputs[i][j] is the
            unnormalized logit of jth class for the ith example.
        targets (Int[Tensor, "batch_size"]): Tensor of shape (batch_size,) with the index of the correct class.
            Each value must be between 0 and `num_classes - 1`.

    Returns:
        Float[Tensor, ""]: The average cross-entropy loss across examples.
    """
    
    # 减去最大元素以确保数值稳定性
    max_logits = inputs.max(dim=-1, keepdim=True)[0]
    shifted_logits = inputs - max_logits
    
    # 计算 log_softmax，这避免了直接计算 softmax 再取 log
    # log_softmax(x) = x - log(sum(exp(x)))
    log_softmax = shifted_logits - torch.logsumexp(shifted_logits, dim=-1, keepdim=True)
    
    # 获取目标位置的 log_softmax 值
    target_log_softmax = torch.gather(log_softmax, dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    
    # 交叉熵损失 = -log_softmax
    return -target_log_softmax.mean()


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(AdamW, self).__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]  # 获取学习率
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]  # 获取与参数p相关的状态
                t = state.get("t", 1)  # 获取迭代次数，默认为1
                grad = p.grad.data  # 获取梯度
                m = state.get("m", torch.zeros_like(p.data))  # 获取动量
                v = state.get("v", torch.zeros_like(p.data))  # 获取速度
                beta1, beta2 = group["betas"]
                eps = group["eps"]
                weight_decay = group["weight_decay"]

                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * grad ** 2

                lr_t = lr * math.sqrt(1 - beta2 ** t) / (1 - beta1 ** t)


                p.data -= lr_t * m / (torch.sqrt(v) + eps) 
                p.data -= lr * weight_decay * p.data

                # 保存状态
                state["m"] = m
                state["v"] = v
                state["t"] = t + 1  # 迭代次数加1

        return loss
    
        


def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    if it < warmup_iters:
        return max_learning_rate * it / warmup_iters
    elif it <= cosine_cycle_iters:
        return min_learning_rate + (max_learning_rate - min_learning_rate) * (1 + math.cos(math.pi * (it - warmup_iters) / (cosine_cycle_iters - warmup_iters))) / 2
    else:
        return min_learning_rate



def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """Given a set of parameters, clip their combined gradients to have l2 norm at most max_l2_norm.

    Args:
        parameters (Iterable[torch.nn.Parameter]): collection of trainable parameters.
        max_l2_norm (float): a positive value containing the maximum l2-norm.

    The gradients of the parameters (parameter.grad) should be modified in-place.
    """
    # 收集所有非空梯度
    grads = []
    for p in parameters:
        if p.grad is not None:
            grads.append(p.grad.view(-1))
    
    if not grads:
        return
    
    # 计算所有梯度的总L2范数，而不是计算一个参数一个参数地算梯度范数！
    total_norm = torch.cat(grads).norm(2)
    
    eps = 1e-6
    clip_coef = max_l2_norm / (total_norm + eps)
    
    # 对所有梯度进行缩放
    for p in parameters:
        if p.grad is not None:
            p.grad.data.mul_(clip_coef)