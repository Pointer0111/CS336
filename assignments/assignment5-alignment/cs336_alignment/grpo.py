import os
from typing import Any, Callable, Literal

import torch
from torch import Tensor
from torch.utils.data import Dataset
from transformers.tokenization_utils_base import PreTrainedTokenizerBase


def masked_mean(tensor: torch.Tensor, mask: torch.Tensor, dim: int | None = None) -> torch.Tensor:
    """Compute the mean of the tensor along a dimension,
    considering only the elements with mask value 1.

    Args:
        tensor: torch.Tensor, the tensor to compute the mean of.
        mask: torch.Tensor, the mask. We only take the mean over
            the elements with mask value 1.
        dim: int | None, the dimension to compute the mean along.
            If None, sum over all non-masked elements and average
            by their total count.

    Returns:
        torch.Tensor, the mean of the tensor along the specified
            dimension, considering only the elements with mask value 1.
    """
    
    # 计算mask中非零元素的数量，避免除零错误
    mask_count = torch.count_nonzero(mask, dim=dim)
    
    # 计算masked sum并除以非零元素数量
    return torch.sum(tensor.masked_fill(~mask, 0), dim=dim) / mask_count



def compute_group_normalized_rewards(
    reward_fn: Callable,
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    advantage_eps: float,
    normalize_by_std: bool,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """
    Compute rewards for each group of rollout responses, 
    normalized by the group size.

    For more on GRPO, see:
        DeepSeekMath: https://arxiv.org/abs/2402.03300
        DeepSeek-R1: https://arxiv.org/abs/2501.12948

    Args:
        reward_fn: Callable[[str, str], dict[str, float]], 
            scores the rollout responses against the ground truths, 
            producing a dict with keys 
            "reward", "format_reward", and "answer_reward".
        rollout_responses: list[str], rollouts from the policy. 
            The length of this list is 
            `rollout_batch_size = n_prompts_per_rollout_batch * group_size`.
        repeated_ground_truths: list[str], the ground truths for the examples. 
            The length of this list is `rollout_batch_size`, 
            because the ground truth for each example is repeated `group_size` times.
        group_size: int, number of rollouts per group.
        advantage_eps: float, epsilon to avoid division by zero
            during group normalization.
        normalize_by_std: bool, whether to normalize the rewards by
            std(rewards).

    Returns:
        tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
            torch.Tensor of shape (rollout_batch_size,): 
                group-normalized rewards for each rollout response.
            torch.Tensor of shape (rollout_batch_size,): 
                raw rewards for each rollout response.
            dict[str, float]: metadata for the rewards of the rollout batch.
                You may choose what you wish to log here
                (some statistics of the rewards, etc.).
    """

    metadata = {}
    rollout_batch_size = len(rollout_responses)
    rewards = [reward_fn(rollout_responses[i], repeated_ground_truths[i])['reward'] for i in range(rollout_batch_size)]

    raw_rewards = torch.tensor(rewards)

    # 计算原始奖励的统计信息
    metadata['raw_rewards_mean'] = float(raw_rewards.mean())
    metadata['raw_rewards_std'] = float(raw_rewards.std())
    metadata['raw_rewards_min'] = float(raw_rewards.min())
    metadata['raw_rewards_max'] = float(raw_rewards.max())

    advantages = raw_rewards.reshape(-1, group_size).clone()
    if normalize_by_std:
        advantages = (advantages - advantages.mean(-1, keepdim=True))/(advantage_eps + advantages.std(-1, keepdim=True))
    else:
        advantages -= advantages.mean(-1, keepdim=True)

    # 计算标准化后奖励的统计信息
    normalized_rewards = advantages.reshape(-1)
    metadata['normalized_rewards_mean'] = float(normalized_rewards.mean())
    metadata['normalized_rewards_std'] = float(normalized_rewards.std())
    metadata['normalized_rewards_min'] = float(normalized_rewards.min())
    metadata['normalized_rewards_max'] = float(normalized_rewards.max())


    return (normalized_rewards, raw_rewards, metadata)



def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
) -> torch.Tensor:
    """Compute policy gradient loss using either raw rewards or advantages.

    Args:
        raw_rewards_or_advantages: torch.Tensor of shape (batch_size, 1): 
            the raw rewards or advantages for each rollout response.
        policy_log_probs: torch.Tensor of shape (batch_size, sequence_length): 
            the log-probs of the policy.

    Returns:
        torch.Tensor of shape (batch_size, sequence_length): 
            the policy gradient per-token loss.
    """
    
    seq_len = policy_log_probs.shape[-1]
    raw_rewards_or_advantages = raw_rewards_or_advantages.expand(-1, seq_len)

    return - raw_rewards_or_advantages * policy_log_probs



def compute_grpo_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the GRPO-Clip loss.

    Args:
        advantages: torch.Tensor of shape (batch_size, 1): 
            the advantages for each rollout response.
        policy_log_probs: torch.Tensor of shape (batch_size, sequence_length): 
            the log-probs of the policy.
        old_log_probs: torch.Tensor of shape (batch_size, sequence_length): 
            the log-probs of the old policy.
        cliprange: float, the clip range for the ratio.

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]]:
            torch.Tensor of shape (batch_size, sequence_length): 
                the GRPO-Clip per-token loss.
            dict[str, torch.Tensor]: metadata for the GRPO-Clip loss 
                (used to compute clip fraction).
    """
    
    metadata = {}
    seq_len = policy_log_probs.shape[-1]
    advantages = advantages.expand(-1, seq_len)

    ratios = torch.exp(policy_log_probs - old_log_probs)
    ratios_clipped = torch.clamp(ratios, 1 - cliprange, 1 + cliprange)

    # 计算未裁剪和裁剪的损失
    unclipped_loss = ratios * advantages
    clipped_loss = ratios_clipped * advantages
    
    # 记录每个token是否被裁剪
    # 如果clipped_loss < unclipped_loss，说明该token被裁剪了
    clipped_tokens = clipped_loss < unclipped_loss
    
    # 计算裁剪统计信息
    metadata['clip_fraction'] = float(clipped_tokens.float().mean())
    metadata['num_clipped_tokens'] = int(clipped_tokens.sum().item())
    metadata['total_tokens'] = int(clipped_tokens.numel())
    
    # 记录裁剪相关的统计信息
    metadata['ratios_mean'] = float(ratios.mean())
    metadata['ratios_std'] = float(ratios.std())
    metadata['ratios_min'] = float(ratios.min())
    metadata['ratios_max'] = float(ratios.max())
    
    # 记录损失统计信息
    metadata['unclipped_loss_mean'] = float(unclipped_loss.mean())
    metadata['clipped_loss_mean'] = float(clipped_loss.mean())
    metadata['final_loss_mean'] = float(-torch.minimum(unclipped_loss, clipped_loss).mean())

    loss = - torch.minimum(unclipped_loss, clipped_loss)

    return (loss, metadata)
    



def compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: str,
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Wrapper that delegates to the appropriate policy gradient loss function above.
    """
    
    metadata = {}

    if loss_type == "no_baseline":
        assert raw_rewards is not None
        loss = compute_naive_policy_gradient_loss(raw_rewards, policy_log_probs)
        
        # 收集 no_baseline 的 metadata
        metadata['raw_rewards_mean'] = float(raw_rewards.mean())
        metadata['raw_rewards_std'] = float(raw_rewards.std())
        metadata['raw_rewards_min'] = float(raw_rewards.min())
        metadata['raw_rewards_max'] = float(raw_rewards.max())
        metadata['loss_mean'] = float(loss.mean())
        metadata['loss_std'] = float(loss.std())
        metadata['loss_min'] = float(loss.min())
        metadata['loss_max'] = float(loss.max())

    elif loss_type == "reinforce_with_baseline":
        assert advantages is not None
        loss = compute_naive_policy_gradient_loss(advantages, policy_log_probs)
        
        # 收集 reinforce_with_baseline 的 metadata
        metadata['advantages_mean'] = float(advantages.mean())
        metadata['advantages_std'] = float(advantages.std())
        metadata['advantages_min'] = float(advantages.min())
        metadata['advantages_max'] = float(advantages.max())
        metadata['loss_mean'] = float(loss.mean())
        metadata['loss_std'] = float(loss.std())
        metadata['loss_min'] = float(loss.min())
        metadata['loss_max'] = float(loss.max())

    elif loss_type == "grpo_clip":
        assert advantages is not None
        assert old_log_probs is not None and cliprange is not None
        loss, clip_metadata = compute_grpo_clip_loss(advantages, 
            policy_log_probs, old_log_probs, cliprange)
        
        # 合并 GRPO-Clip 的 metadata
        metadata.update(clip_metadata)
        metadata['advantages_mean'] = float(advantages.mean())
        metadata['advantages_std'] = float(advantages.std())
        metadata['advantages_min'] = float(advantages.min())
        metadata['advantages_max'] = float(advantages.max())
        metadata['loss_mean'] = float(loss.mean())
        metadata['loss_std'] = float(loss.std())
        metadata['loss_min'] = float(loss.min())
        metadata['loss_max'] = float(loss.max())
    
    else:
        raise Warning(f"未知的损失类型: {loss_type}，请检查输入参数。")

    
    return (loss, metadata)



def grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the policy gradient loss and backprop its gradients for a microbatch.

    Args:
        policy_log_probs: torch.Tensor of shape (batch_size, sequence_length): 
            the log-probs of the policy.
        response_mask: torch.Tensor of shape (batch_size, sequence_length): 
            the mask for the response.
        gradient_accumulation_steps: int, the number of gradient accumulation steps.
        loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"], 
            the type of loss function to use.
        raw_rewards: torch.Tensor | None, the raw rewards for each rollout response.
            Needed for loss_type="no_baseline".
        advantages: torch.Tensor | None, the advantages for each rollout response.
            Needed for loss_type in {"reinforce_with_baseline", "grpo_clip"}.
        old_log_probs: torch.Tensor | None, the log-probs of the old policy.
            Needed for loss_type="grpo_clip".
        cliprange: float | None, the clip range for the ratio. 
            Needed for loss_type="grpo_clip".
        constant_normalize_factor: int | None, provided if we want to sum over 
            the sequence dimension and normalize by this constant factor
            (as in Dr. GRPO).

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]]: 
            the policy gradient loss and its metadata.
    """
    
    # 获取损失和底层损失函数的metadata
    losses, loss_metadata = compute_policy_gradient_loss(
        policy_log_probs, 
        loss_type, 
        raw_rewards, 
        advantages, 
        old_log_probs, 
        cliprange)

    # 计算masked mean损失
    masked_losses = masked_mean(losses, response_mask, dim=-1)
    loss = masked_losses.mean(0) / gradient_accumulation_steps

    # 收集microbatch训练的metadata
    metadata = {}
    metadata.update(loss_metadata)  # 包含底层损失函数的metadata
    
    # 添加microbatch特定的统计信息
    metadata['microbatch_loss'] = float(loss.item())
    metadata['masked_losses_mean'] = float(masked_losses.mean())
    metadata['masked_losses_std'] = float(masked_losses.std())
    metadata['masked_losses_min'] = float(masked_losses.min())
    metadata['masked_losses_max'] = float(masked_losses.max())
    
    # 添加序列长度和mask统计信息
    metadata['sequence_length'] = int(policy_log_probs.shape[-1])
    metadata['batch_size'] = int(policy_log_probs.shape[0])
    metadata['response_mask_sum'] = int(response_mask.sum().item())
    metadata['response_mask_mean'] = float(response_mask.float().mean())
    
    # 添加梯度累积相关信息
    metadata['gradient_accumulation_steps'] = gradient_accumulation_steps
    metadata['effective_batch_size'] = int(policy_log_probs.shape[0] * gradient_accumulation_steps)

    loss.backward()

    return (loss, metadata)



    





