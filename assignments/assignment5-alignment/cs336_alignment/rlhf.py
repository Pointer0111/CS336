"""
The below adapters are used in the optional 
RLHF / safety part of the Alignment assignment.
"""

import re
from typing import Any
import torch.nn.functional as F

from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from .sft import *

import json
import random
import torch
from torch.utils.data import Dataset



def iterate_batches(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
):
    """
    Given a PyTorch Dataset, return an iterable over batches of size `batch_size`.
    Iterating through the returned iterable should constitute one epoch over the Dataset.

    Args:
        dataset: Dataset
            Dataset to emit batches from.
        batch_size: int
            Number of examples to include per batch.
        shuffle: bool
            If true, shuffle examples before batching them.

    Returns:
        Iterable over batches, where each batch has size `batch_size`.
    """
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def parse_mmlu_response(
    mmlu_example: dict[str, Any],
    model_output: str,
) -> str | None:
    """
    Given an MMLU example and a model output, parse the model output into a
    predicted option letter (i.e., 'A', 'B', 'C', or 'D'). If the model output
    cannot be parsed into a prediction option letter, return None.

    mmlu_example: dict[str, Any]
        Dictionary with an MMLU example. Contains the following keys:
        - "subject": str with the subject of the question.
        - "question": str with the text of the question.
        - "options": list[str] with the four answer options (in order).
                     The first option refers to letter "A", the second to "B", etc.
        - "answer": str with the option of the correct answer (e.g., "A")
    model_output: str
        str with the model's output to the MMLU example.

    Returns:
        str (one of "A", "B", "C", or "D") if the model output can be parsed into a prediction,
        else None.
    """
    # 使用正则表达式匹配 "The correct answer is X" 格式
    # 其中 X 应该是 A、B、C 或 D 中的一个字母
    pattern = r"The correct answer is\s+([ABCD])"
    match = re.search(pattern, model_output, re.IGNORECASE)
    
    if match:
        return match.group(1).upper()
    
    return None


def parse_gsm8k_response(
    model_output: str,
) -> str | None:
    """
    Given a GSM8K model output, parse the model output into a predicted numeric answer by
    taking the last number that occurs in the output.

    model_output: str
        str with the model's output to a GSM8K example.

    Returns:
        str with the predicted numeric answer if the model output can be parsed into a prediction,
        else None.
    """
    # 使用正则表达式匹配所有数字（包括整数和小数）
    # \d+ 匹配一个或多个数字
    # (?:\.\d+)? 匹配可选的小数部分
    pattern = r'\d+(?:\.\d+)?'
    matches = re.findall(pattern, model_output)
    
    # 如果找到数字，返回最后一个
    if matches:
        return matches[-1]
    
    return None



template = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n{response}"

def compute_per_instance_dpo_loss(
    lm: torch.nn.Module,
    lm_ref: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    beta: float,
    prompt: str,
    response_chosen: str,
    response_rejected: str,
) -> torch.Tensor:
    """
    Given two language models (`lm`, and the "reference model" `lm_ref`),
    their tokenizer, the DPO beta hyperparameter, a prompt and a pair
    of responses to the prompt, computes the value of the DPO loss for this example.

    lm: torch.nn.Module
        Language model being trained.
    lm_ref: torch.nn.Module
        Reference language model.
    tokenizer: PreTrainedTokenizerBase
        Tokenizer for both language models.
    beta: float
        DPO beta hyperparameter.
    prompt: str
        Prompt for this instance of preference pair.
    response_chosen: str
        Preferred response to the prompt.
    response_rejected: str
        Rejected response to the prompt.

    Returns:
        torch.Tensor with the DPO loss for this example.
    """
    
    # 设置模型为评估模式
    lm.eval()
    lm_ref.eval()
    
    # 使用Alpaca模板格式化prompt和response，并添加EOS token
    eos_token = tokenizer.eos_token if tokenizer.eos_token is not None else "<|endoftext|>"
    if isinstance(eos_token, list):
        eos_token = eos_token[0]
    text_chosen = template.format(instruction=prompt, response=response_chosen) + eos_token
    text_rejected = template.format(instruction=prompt, response=response_rejected) + eos_token
    
    # 对完整文本进行tokenization
    tokens_chosen = tokenizer(text_chosen, return_tensors="pt", add_special_tokens=False)["input_ids"]
    tokens_rejected = tokenizer(text_rejected, return_tensors="pt", add_special_tokens=False)["input_ids"]
    
    # 确保tokens是1维张量 (去掉batch维度)
    if tokens_chosen.dim() > 1:
        tokens_chosen = tokens_chosen.squeeze(0)
    if tokens_rejected.dim() > 1:
        tokens_rejected = tokens_rejected.squeeze(0)
    
    # 计算logits (需要重新添加batch维度)
    with torch.no_grad():
        logits_lm_chosen = lm(tokens_chosen.unsqueeze(0)).logits.squeeze(0)  # [seq_len, vocab_size]
        logits_ref_chosen = lm_ref(tokens_chosen.unsqueeze(0)).logits.squeeze(0)
        
        logits_lm_rejected = lm(tokens_rejected.unsqueeze(0)).logits.squeeze(0)
        logits_ref_rejected = lm_ref(tokens_rejected.unsqueeze(0)).logits.squeeze(0)
    
    # 计算对数概率
    log_probs_lm_chosen = torch.log_softmax(logits_lm_chosen, dim=-1)
    log_probs_ref_chosen = torch.log_softmax(logits_ref_chosen, dim=-1)
    log_probs_lm_rejected = torch.log_softmax(logits_lm_rejected, dim=-1)
    log_probs_ref_rejected = torch.log_softmax(logits_ref_rejected, dim=-1)
    
    # 计算序列对数概率：对于每个位置预测下一个token
    # tokens[1:]是target，因为我们要预测每个位置的下一个token
    # log_probs[:-1]是logits，因为最后一个位置没有下一个token可预测
    chosen_target_ids = tokens_chosen[1:]  # [seq_len-1]
    rejected_target_ids = tokens_rejected[1:]  # [seq_len-1]
    
    # 获取每个位置的对数概率
    log_probs_lm_chosen = log_probs_lm_chosen[:-1].gather(-1, chosen_target_ids.unsqueeze(-1)).squeeze(-1)
    log_probs_ref_chosen = log_probs_ref_chosen[:-1].gather(-1, chosen_target_ids.unsqueeze(-1)).squeeze(-1)
    log_probs_lm_rejected = log_probs_lm_rejected[:-1].gather(-1, rejected_target_ids.unsqueeze(-1)).squeeze(-1)
    log_probs_ref_rejected = log_probs_ref_rejected[:-1].gather(-1, rejected_target_ids.unsqueeze(-1)).squeeze(-1)
    
    # 计算整个序列的对数概率（求和）
    log_prob_lm_chosen = log_probs_lm_chosen.sum()
    log_prob_ref_chosen = log_probs_ref_chosen.sum()
    log_prob_lm_rejected = log_probs_lm_rejected.sum()
    log_prob_ref_rejected = log_probs_ref_rejected.sum()
    
    # 计算对数概率比率
    log_ratio_chosen = log_prob_lm_chosen - log_prob_ref_chosen
    log_ratio_rejected = log_prob_lm_rejected - log_prob_ref_rejected
    
    # 计算DPO损失：-log(σ(β * (log_ratio_chosen - log_ratio_rejected)))
    loss = -F.logsigmoid(beta * (log_ratio_chosen - log_ratio_rejected))
    
    return loss 

