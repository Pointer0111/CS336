import os
import regex as re
from collections import defaultdict, Counter
from typing import Iterable, Iterator
import pickle
from concurrent.futures import ThreadPoolExecutor

# 导入分块函数
from .pretokenization_example import find_chunk_boundaries

def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """训练BPE tokenizer
    
    Args:
        input_path: 输入文本文件路径
        vocab_size: 最终词汇表大小（包括字节、合并token和特殊token）
        special_tokens: 特殊token列表
    
    Returns:
        vocab: 词汇表 {token_id: token_bytes}
        merges: 合并列表 [(token1, token2), ...]
    """
    
    # 读取文件
    with open(input_path, "rb") as f:
        content = f.read()
    
    # 移除特殊token，按特殊token分割
    text = content.decode("utf-8", errors="ignore")
    if special_tokens:
        pattern = "|".join(re.escape(token) for token in special_tokens)
        parts = re.split(pattern, text)
    else:
        parts = [text]
    
    # 预分词
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    word_freqs = Counter()
    for part in parts:
        if part:
            tokens = re.findall(PAT, part)
            word_freqs.update(tokens)
    
    # 初始化词汇表：256个字节 + 特殊token
    vocab = {i: bytes([i]) for i in range(256)}
    next_id = 256
    
    # 添加特殊token到词汇表
    for token in special_tokens:
        vocab[next_id] = token.encode('utf-8')
        next_id += 1
    
    # 将每个word转换为字节序列并初始化token列表
    word_tokens = {}
    for word, freq in word_freqs.items():
        word_bytes = word.encode('utf-8')
        word_tokens[word] = [bytes([b]) for b in word_bytes]
    
    # BPE合并算法
    merges = []
    
    while len(vocab) < vocab_size:
        # 统计所有相邻字节对的频率 - 优化版本
        pair_freqs = defaultdict(int)
        for word, freq in word_freqs.items():
            tokens = word_tokens[word]
            if len(tokens) < 2:
                continue
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                pair_freqs[pair] += freq
        
        if not pair_freqs:
            break
        
        # 找到频率最高的pair，如果有平局则选择字典序最大的
        best_pair = max(pair_freqs.items(), key=lambda x: (x[1], x[0]))[0]
        
        # 记录合并
        merges.append(best_pair)
        
        # 创建新token
        new_token = best_pair[0] + best_pair[1]
        vocab[next_id] = new_token
        next_id += 1
        
        # 更新所有words中的pair - 进一步优化
        words_to_update = []
        for word, tokens in word_tokens.items():
            if len(tokens) >= 2:
                # 快速检查是否包含目标pair
                for i in range(len(tokens) - 1):
                    if tokens[i] == best_pair[0] and tokens[i + 1] == best_pair[1]:
                        words_to_update.append(word)
                        break
        
        # 只更新包含目标pair的words
        for word in words_to_update:
            tokens = word_tokens[word]
            new_tokens = []
            i = 0
            while i < len(tokens):
                if (i < len(tokens) - 1 and 
                    tokens[i] == best_pair[0] and 
                    tokens[i + 1] == best_pair[1]):
                    new_tokens.append(new_token)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            word_tokens[word] = new_tokens
    
    return vocab, merges




class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens
        
        # 创建反向映射以提高查找效率
        self.vocab_reverse = {token: token_id for token_id, token in vocab.items()}
        
        # 创建特殊token的字节到ID的映射
        self.special_token_ids = {}
        if special_tokens:
            for token in special_tokens:
                token_bytes = token.encode('utf-8')
                if token_bytes in self.vocab_reverse:
                    self.special_token_ids[token] = self.vocab_reverse[token_bytes]
        
        # 创建合并规则的查找表以提高BPE效率
        self.merge_lookup = {}
        for pair in merges:
            self.merge_lookup[pair] = pair[0] + pair[1]
        
        # 创建合并规则优先级映射，用于快速查找优先级
        self.merge_priority = {pair: i for i, pair in enumerate(merges)}

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)
        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)
        return cls(vocab, merges, special_tokens)

    def encode(self, text: str, verbose: bool = False) -> list[int]:
        """Encode an input text into a sequence of token IDs."""
        # 步骤1: 预分词 - 使用与训练时相同的正则表达式
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        
        # 如果有特殊token，先处理特殊token
        if self.special_tokens:
            # 按特殊token分割文本，优先匹配最长的token
            # 按长度降序排序特殊token
            sorted_special_tokens = sorted(self.special_tokens, key=len, reverse=True)
            pattern = "|".join(re.escape(token) for token in sorted_special_tokens)
            
            # 找到所有特殊token的位置
            special_token_positions = []
            for match in re.finditer(pattern, text):
                special_token_positions.append((match.start(), match.end(), match.group()))
            
            # 按位置排序
            special_token_positions.sort()
            
            # 重新构建文本和特殊token序列
            result_ids = []
            current_pos = 0
            
            for start, end, special_token in special_token_positions:
                # 处理特殊token之前的文本
                if start > current_pos:
                    normal_text = text[current_pos:start]
                    result_ids.extend(self._encode_normal_text(normal_text, verbose))
                
                # 添加特殊token的ID
                special_token_id = self._get_special_token_id(special_token)
                if special_token_id is not None:
                    result_ids.append(special_token_id)
                
                current_pos = end
            
            # 处理最后剩余的文本
            if current_pos < len(text):
                normal_text = text[current_pos:]
                result_ids.extend(self._encode_normal_text(normal_text, verbose))
            
            return result_ids
        else:
            return self._encode_normal_text(text, verbose)
    
    def _apply_bpe_merges(self, tokens: list[bytes], verbose: bool = False) -> list[bytes]:
        """优化的BPE合并算法"""
        if len(tokens) <= 1:
            return tokens
        
        original_length = len(tokens)
        merge_count = 0
        max_merges = len(tokens) - 1  # 最多可能的合并次数
        
        while True:
            # 找到最优先的合并对
            best_pair = None
            best_pos = -1
            best_merge_index = -1
            
            # 遍历所有可能的相邻token对
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                # 使用哈希表快速查找优先级
                if pair in self.merge_priority:
                    merge_index = self.merge_priority[pair]
                    # 如果这个pair的优先级更高（索引更小），则选择它
                    if best_merge_index == -1 or merge_index < best_merge_index:
                        best_pair = pair
                        best_pos = i
                        best_merge_index = merge_index
            
            if best_pair is None:
                break
            
            # 执行合并
            merged_token = self.merge_lookup[best_pair]
            tokens[best_pos] = merged_token
            tokens.pop(best_pos + 1)
            merge_count += 1
        
        return tokens

    def _encode_normal_text(self, text: str, verbose: bool = False) -> list[int]:
        """编码普通文本（不包含特殊token）"""
        # 预分词
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        pre_tokens = re.findall(PAT, text)
        
        result_ids = []
        for i, pre_token in enumerate(pre_tokens):
            # 将预分词转换为字节序列
            pre_token_bytes = pre_token.encode('utf-8')
            tokens = [bytes([b]) for b in pre_token_bytes]
            
            # 应用BPE合并规则 - 优化版本
            tokens = self._apply_bpe_merges(tokens, verbose)
            
            # 将token转换为ID
            for token in tokens:
                token_id = self._get_token_id(token)
                result_ids.append(token_id)
        
        return result_ids
    
    def _get_token_id(self, token: bytes) -> int:
        """获取token对应的ID - 优化版本"""
        # 使用反向映射快速查找
        if token in self.vocab_reverse:
            return self.vocab_reverse[token]
        
        # 如果token不在词汇表中，按字节处理
        if len(token) == 1:
            return token[0]
        else:
            # 对于多字节token，按字节分解
            return token[0] if token else 0
    
    def _get_special_token_id(self, special_token: str) -> int | None:
        """获取特殊token的ID - 优化版本"""
        return self.special_token_ids.get(special_token)

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """Given an iterable of strings (e.g., a Python file handle), return a generator that lazily yields token IDs. This is
        required for memory-efficient tokenization of large files that we cannot directly load into
        memory."""
        
        # 检查是否是文件对象，如果是则使用分块处理
        if hasattr(iterable, 'read') and hasattr(iterable, 'seek'):
            # 这是一个文件对象，使用分块处理
            yield from self._encode_file_in_chunks(iterable)
        else:
            # 这是一个普通的字符串迭代器，使用原来的方法
            for text in iterable:
                token_ids = self.encode(text)
                for token_id in token_ids:
                    yield token_id
    
    def _encode_file_in_chunks(self, file_obj) -> Iterator[int]:
        """使用分块策略编码大文件"""
        # 检查文件模式，如果是文本模式需要特殊处理
        if hasattr(file_obj, 'mode') and 'b' not in file_obj.mode:
            # 文本文件，需要重新以二进制模式打开
            file_path = file_obj.name
            file_obj.close()
            file_obj = open(file_path, 'rb')
        
        # 确定分块数量（可以根据文件大小调整）
        file_obj.seek(0, os.SEEK_END)
        file_size = file_obj.tell()
        file_obj.seek(0)
        
        # 根据文件大小确定分块数量
        if file_size < 1024 * 1024:  # 小于1MB
            num_chunks = 1
        elif file_size < 4 * 1024 * 1024:  # 小于4MB
            num_chunks = 4
        else:  # 大于4MB
            num_chunks = 8
        
        # 使用特殊token作为分块边界
        split_token = b"<|endoftext|>" if self.special_tokens and "<|endoftext|>" in self.special_tokens else b"\n"
        
        # 找到分块边界
        boundaries = find_chunk_boundaries(file_obj, num_chunks, split_token)
        
        # 定义处理单个分块的函数
        def process_chunk(chunk_id, start, end):
            # 只对第一个分块启用详细模式
            verbose = (chunk_id == 0)
            
            try:
                # 重新打开文件以避免线程冲突
                file_path = file_obj.name
                with open(file_path, 'rb') as chunk_file:
                    chunk_file.seek(start)
                    chunk_data = chunk_file.read(end - start)
                    
                    # 解码分块数据
                    try:
                        chunk_text = chunk_data.decode("utf-8", errors="ignore")
                    except UnicodeDecodeError:
                        chunk_text = chunk_data.decode("utf-8", errors="replace")
                    
                    # 编码这个分块
                    token_ids = self.encode(chunk_text, verbose=verbose)
                    
                    return chunk_id, token_ids
            except Exception as e:
                return chunk_id, [f"Error processing chunk {chunk_id}: {str(e)}"]
        
        # 使用线程池并行处理分块
        max_workers = num_chunks
        chunk_tasks = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有分块任务
            for i, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
                future = executor.submit(process_chunk, i, start, end)
                chunk_tasks.append((i, future))
            
            # 收集结果并按顺序输出
            results = {}
            for chunk_id, future in chunk_tasks:
                try:
                    chunk_id, token_ids = future.result()
                    results[chunk_id] = token_ids
                except Exception as e:
                    results[chunk_id] = [f"Error in chunk {chunk_id}: {str(e)}"]
            
            # 按chunk_id顺序输出结果
            for i in range(len(results)):
                if i in results:
                    for token_id in results[i]:
                        yield token_id

    def decode(self, ids: list[int]) -> str:
        """Decode a sequence of token IDs into text"""
        # 将ID转换为token
        tokens = []
        for token_id in ids:
            if token_id in self.vocab:
                tokens.append(self.vocab[token_id])
            else:
                # 如果ID不在词汇表中，按字节处理
                tokens.append(bytes([token_id]))
        
        # 合并所有token的字节
        result_bytes = b''.join(tokens)
        
        # 解码为字符串
        try:
            return result_bytes.decode('utf-8')
        except UnicodeDecodeError:
            # 如果解码失败，使用错误处理
            return result_bytes.decode('utf-8', errors='replace')


