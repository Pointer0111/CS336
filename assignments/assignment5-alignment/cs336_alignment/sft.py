from importlib import metadata
import os
from typing import Any, Callable, Literal

import torch
from torch import Tensor
from torch.utils.data import Dataset
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
import json
import random


def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, Tensor]:
    """Tokenize the prompt and output strings, and construct a mask that is 1
    for the response tokens and 0 for other tokens (prompt or padding).

    Args:
        prompt_strs: list[str], the prompt strings.
        output_strs: list[str], the output strings.
        tokenizer: PreTrainedTokenizer, the tokenizer to use.

    Returns:
        dict[str, torch.Tensor]:
            "input_ids": torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
                the tokenized prompt and output strings, with the final token sliced off.
            "labels": torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
                shifted input_ids (i.e., the input_ids without the first token).
            "response_mask": torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
                a mask on the response tokens in `labels`.
    """
    ans = {}
    prompt_tokenized = tokenizer(prompt_strs)['input_ids']
    output_tokenized = tokenizer(output_strs)['input_ids']
     
    tokenized = [ p + o for (p, o) in zip(prompt_tokenized, output_tokenized)]
    max_len = max([len(t) for t in tokenized]) - 1
    bs = len(tokenized)

    input_ids = torch.zeros((bs, max_len), dtype=torch.long)
    labels = torch.zeros((bs, max_len), dtype=torch.long) 
    response_mask = torch.zeros((bs, max_len), dtype=torch.bool)

    for i, tokens in enumerate(tokenized):
        input_ids[i, :len(tokens)-1] = torch.tensor(tokens[:-1])
        labels[i, :len(tokens)-1] = torch.tensor(tokens[1:])
        if len(tokens) < max_len:
            labels[i, len(tokens)-1:] = tokenizer.eos_token_id
        response_mask[i, len(prompt_tokenized[i])-1:len(tokens)-1] = True

    # 将input_ids最后一列等于0的地方替换为tokenizer.eos_token_id
    last_col_idx = max_len - 1
    mask = input_ids[:, last_col_idx] == 0
    input_ids[mask, last_col_idx] = tokenizer.eos_token_id

    ans["input_ids"] = input_ids
    ans["labels"] = labels
    ans["response_mask"] = response_mask

    return ans




def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Get the entropy of the logits (i.e., entropy of the final dimension)."""

    return torch.logsumexp(logits, dim=-1) - (logits * logits.softmax(dim=-1)).sum(dim=-1)


def get_response_log_probs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool,
) -> torch.Tensor:
    """Get the conditional log-probs of the response given the prompt,
        and optionally the entropy of the next token predictions.

    Args:
        model: PreTrainedModel, the model to score.
        input_ids: torch.Tensor of shape (batch_size, sequence_length):
            the tokenized prompt and output.
        labels: torch.Tensor of shape (batch_size, sequence_length):
            shifted input_ids.
        return_token_entropy: bool, whether to return the entropy of the
            next token predictions.

    Returns:
        dict[str, torch.Tensor]:
            "log_probs": torch.Tensor of shape (batch_size, sequence_length):
                the conditional log-probs of the response given the prompt.
                Note that we have not masked out the token indices corresponding
                to the prompt or padding; that is done in the train loop.
            "token_entropy": Optional[torch.Tensor] of shape (batch_size, sequence_length):
                the entropy of the next token predictions. As with the log-probs,
                we have not masked out the token indices corresponding to the prompt
                or padding; that is done in the train loop.
    """
    ans = {}
    logits = model(input_ids).logits
    ans["log_probs"] = torch.log(torch.gather(logits.softmax(dim=-1), dim=-1, index=labels.unsqueeze(-1)).squeeze(-1))
    if return_token_entropy:
        ans["token_entropy"] = compute_entropy(logits)

    return ans



def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
    normalize_constant: float = 1.0,
) -> torch.Tensor:
    """Sum over a dimension and normalize by a constant,
    considering only the elements with mask value 1.

    Args:
        tensor: torch.Tensor, the tensor to sum and normalize.
        mask: torch.Tensor, the mask. We only consider elements
            with mask value 1.
        dim: int | None, the dimension to sum along before
            normalization. If None, sum over all dimensions.
        normalize_constant: float, the constant to divide by
            for normalization.

    Returns:
        torch.Tensor, the normalized sum, where masked elements
            (mask=0) don't contribute to the sum.
    """

    return torch.sum(tensor.masked_fill(~mask, 0), dim=dim) / normalize_constant





def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the policy gradient loss and backprop its gradients for a microbatch.
    """

    # 对每个样本分别计算 masked sum，然后除以 normalize_constant，最后求平均
    # 在梯度累积中，需要将损失除以累积步数
    loss = - masked_normalize(policy_log_probs, response_mask, dim=-1, normalize_constant=normalize_constant).mean() / gradient_accumulation_steps
    loss.backward() 

    metadata = {
        "loss": loss.detach().cpu(),
        "policy_log_probs_mean": policy_log_probs.mean().detach().cpu(),
        "policy_log_probs_std": policy_log_probs.std().detach().cpu(),
        "num_masked_tokens": response_mask.sum().item(),
        "normalize_constant": normalize_constant,
    }

    return (loss, metadata)


class PackedSFTDataset(Dataset):
    def __init__(self, tokenizer, dataset_path, seq_length, shuffle):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.dataset_path = dataset_path
        
        # 读取数据集
        with open(dataset_path, "r", encoding="utf-8") as f:
            examples = [json.loads(line) for line in f]
        
        # 如果需要，打乱文档顺序
        if shuffle:
            random.shuffle(examples)
        
        # 使用Alpaca模板格式化每个(prompt, response)对
        template = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n{response}"
        
        # 获取特殊token ID
        bos_token_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id
        eos_token_id = tokenizer.eos_token_id
        
        # 构建所有token序列，将所有文档连接成一个单一序列
        self.all_token_ids = []
        
        # 处理每个example
        for i, example in enumerate(examples):
            # 格式化文本
            formatted_text = template.format(instruction=example["prompt"], response=example["response"])
            
            # 对文本进行tokenization
            tokens = tokenizer(formatted_text, add_special_tokens=False)["input_ids"]
            
            # 如果是第一个example，在开头添加BOS token
            if i == 0 and bos_token_id is not None:
                self.all_token_ids.append(bos_token_id)
            
            # 添加当前example的tokens
            self.all_token_ids.extend(tokens)
            
            # 在每个example后添加EOS token作为分隔符
            if eos_token_id is not None:
                self.all_token_ids.append(eos_token_id)
                
            # 为下一个example添加BOS token（除了最后一个）
            if i < len(examples) - 1 and bos_token_id is not None:
                self.all_token_ids.append(bos_token_id)
        
        # 将长序列分割成固定长度的块（non-overlapping chunks）
        self.sequences = []
        total_tokens = len(self.all_token_ids)
        
        # 使用非重叠的方式分割序列
        i = 0
        while i + seq_length <= total_tokens:
            chunk = self.all_token_ids[i:i + seq_length]
            self.sequences.append(chunk)
            i += seq_length
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        input_ids = torch.tensor(sequence, dtype=torch.long)
        
        # 对于语言建模，labels是input_ids向左移位一位
        # input_ids: [token1, token2, token3, ..., tokenN]
        # labels:    [token2, token3, token4, ..., tokenN+1]
        
        # labels的最后一个位置应该是下一个token
        if idx < len(self.sequences) - 1:
            next_sequence = self.sequences[idx + 1]
            labels = torch.tensor(sequence[1:] + [next_sequence[0]], dtype=torch.long)
        else:
            # 对于最后一个序列，从原始token序列中获取下一个token
            start_pos = idx * self.seq_length
            end_pos = start_pos + self.seq_length
            
            # 获取下一个token
            if end_pos < len(self.all_token_ids):
                next_token = self.all_token_ids[end_pos]
                labels = torch.tensor(sequence[1:] + [next_token], dtype=torch.long)
            else:
                # 如果真的没有下一个token，使用-100
                labels = torch.tensor(sequence[1:] + [-100], dtype=torch.long)
        
        return {
            "input_ids": input_ids,
            "labels": labels
        }







