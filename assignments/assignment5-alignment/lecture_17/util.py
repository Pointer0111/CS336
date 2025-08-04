from torch import nn
import torch
import math
import torch.nn.functional as F
from einops import repeat, einsum, rearrange
from typing import Callable


class Model(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, prompt_length: int, response_length: int):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # For each position, we have a matrix for encoding and a matrix for decoding
        self.encode_weights = nn.Parameter(torch.randn(prompt_length, embedding_dim, embedding_dim) / math.sqrt(embedding_dim))
        self.decode_weights = nn.Parameter(torch.randn(response_length, embedding_dim, embedding_dim) / math.sqrt(embedding_dim))
    
    def forward(self, prompts: torch.Tensor) -> torch.Tensor:
        """
        Args:
            prompts: int[batch pos]
        Returns:
            logits: float[batch pos vocab]
        """
        # Embed the prompts
        embeddings = self.embedding(prompts)   # [batch pos dim]
        
        # Transform using per prompt position matrix, collapse into one vector
        encoded = einsum(embeddings, self.encode_weights, "batch pos dim1, pos dim1 dim2 -> batch dim2")
        
        # Turn into one vector per response position
        decoded = einsum(encoded, self.decode_weights, "batch dim2, pos dim2 dim1 -> batch pos dim1")
        
        # Convert to logits (input and output share embeddings)
        logits = einsum(decoded, self.embedding.weight, "batch pos dim1, vocab dim1 -> batch pos vocab")
        return logits

def compute_log_probs(prompts: torch.Tensor, responses: torch.Tensor, model: Model) -> torch.Tensor:

    logits = model(prompts)
    log_probs = F.log_softmax(logits, dim=-1)

    num_responses = responses.shape[1]
    log_probs = repeat(log_probs, "batch pos vocab -> batch trial pos vocab", trial=num_responses)

    log_probs = log_probs.gather(dim=-1, index=responses.long().unsqueeze(-1)).squeeze(-1)

    return log_probs


def compute_deltas(rewards: torch.Tensor, mode: str) -> torch.Tensor:
    """
    Args:
        rewards (float[batch trial])
    Returns:
        deltas (float[batch trial]) which are advantage-like quantities for updating
    """
    if mode == "rewards":
        return rewards

    if mode == "centered_rewards":
        # Compute mean over all the responses (trial) for each prompt (batch)
        mean_rewards = rewards.mean(dim=-1, keepdim=True)
        centered_rewards = rewards - mean_rewards
        return centered_rewards

    if mode == "normalized_rewards":
        assert rewards.numel() >= 2, "rewards must have at least 2 elements"
        mean_rewards = rewards.mean(dim=-1, keepdim=True)
        std_rewards = rewards.std(dim=-1, keepdim=True)
        centered_rewards = rewards - mean_rewards
        normalized_rewards = centered_rewards / (std_rewards + 1e-5)
        return normalized_rewards

    if mode == "max_rewards":
        # Zero out any reward that isn't the maximum for each batch
        max_rewards = rewards.max(dim=-1, keepdim=True)[0]
        max_rewards = torch.where(rewards == max_rewards, rewards, torch.zeros_like(rewards))
        return max_rewards

    raise ValueError(f"Unknown mode: {mode}")



def compute_loss(log_probs: torch.Tensor, deltas: torch.Tensor, mode: str, old_log_probs: torch.Tensor | None = None) -> torch.Tensor:
    if mode == "naive":
        return -einsum(log_probs, deltas, "batch trial pos, batch trial -> batch trial pos").mean()

    assert old_log_probs is not None

    if mode == "unclipped":
        ratios = log_probs / old_log_probs  # [batch trial]
        return -einsum(ratios, deltas, "batch trial pos, batch trial -> batch trial pos").mean()

    if mode == "clipped":
        epsilon = 0.01
        unclipped_ratios = log_probs / old_log_probs  # [batch trial]
        unclipped = einsum(unclipped_ratios, deltas, "batch trial pos, batch trial -> batch trial pos")
        
        clipped_ratios = torch.clamp(unclipped_ratios, min=1 - epsilon, max=1 + epsilon)
        clipped = einsum(clipped_ratios, deltas, "batch trial pos, batch trial -> batch trial pos")
        return -torch.minimum(unclipped, clipped).mean()

    raise ValueError(f"Unknown mode: {mode}")


def compute_kl_penalty(log_probs: torch.Tensor, ref_log_probs: torch.Tensor) -> torch.Tensor:
    """
    Compute an estimate of KL(model | ref_model), where the models are given by:
        log_probs [batch trial pos vocab]
        ref_log_probs [batch trial pos vocab]
    Use the estimate:
        KL(p || q) = E_p[q/p - log(q/p) - 1]
    """

    return (torch.exp(ref_log_probs - log_probs) - (ref_log_probs - log_probs) - 1).sum(dim=-1).mean()



def compute_reward(prompts: torch.Tensor, responses: torch.Tensor, reward_fn: Callable[[torch.Tensor, torch.Tensor], float]) -> torch.Tensor:
    """
    Args:
        prompts (int[batch pos])
        responses (int[batch trial pos])
    Returns:
        rewards (float[batch trial])
    """
    batch_size, num_responses, _ = responses.shape
    rewards = torch.empty(batch_size, num_responses, dtype=torch.float32)
    for i in range(batch_size):
        for j in range(num_responses):
            rewards[i, j] = reward_fn(prompts[i, :], responses[i, j, :])
    return rewards


def generate_responses(prompts: torch.Tensor, model: Model, num_responses: int) -> torch.Tensor:

    logits = model(prompts)
    batch_size = prompts.shape[0]

    flattened_logits = rearrange(logits, "batch pos vocab -> (batch pos) vocab")
    # 对[batch * pos] 个 logits，按概率分布采样 num_responses 个 token
    flattened_reponses = torch.multinomial(flattened_logits.softmax(dim=-1), num_responses, replacement=True)

    responses = rearrange(flattened_reponses, "(batch pos) trial -> batch trial pos", batch=batch_size)

    return responses



def sort_inclusion_ordering_reward(prompt: torch.Tensor, response: torch.Tensor) -> float:
    """
    Return how close response is to ground_truth = sorted(prompt).
    """
    assert len(prompt) == len(response)
    
    # 用torch改写
    # 计算inclusion_reward：prompt中每个token是否出现在response中
    inclusion_reward = (prompt.unsqueeze(1) == response).any(dim=1).sum().item()

    # 计算ordering_reward：response中相邻元素是否递增
    ordering_reward = (response[:-1] <= response[1:]).sum().item()
    return inclusion_reward + ordering_reward


def print_information(epoch: int, step: int, loss: torch.Tensor, prompts: torch.Tensor, rewards: torch.Tensor, responses: torch.Tensor, log_probs: torch.Tensor, deltas: torch.Tensor):
    print(f"epoch = {epoch}, step = {step}, loss = {loss:.3f}, reward = {rewards.mean():.3f}")
    if epoch % 1 == 0 and step % 5 == 0:
        for batch in range(prompts.shape[0]):
            print(f"  prompt = {prompts[batch, :]}")
            for trial in range(responses.shape[1]):
                print(f"    response = {responses[batch, trial, :]}, log_probs = {tstr(log_probs[batch, trial])}, reward = {rewards[batch, trial]}, delta = {deltas[batch, trial]:.3f}")

def tstr(x: torch.Tensor) -> str:
    return "[" + ", ".join(f"{x[i]:.3f}" for i in range(x.shape[0])) + "]"
