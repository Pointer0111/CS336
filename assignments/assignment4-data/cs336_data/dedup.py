import os
import hashlib
import mmh3
import re
import unicodedata
from collections import defaultdict
from typing import Set, List, Tuple, Dict


def exact_line_deduplication(
    input_files: list[os.PathLike], output_directory: os.PathLike
):
    """
    执行精确行去重，去除在整个语料库中出现超过一次的行。

    """
    # 确保输出目录存在
    os.makedirs(output_directory, exist_ok=True)
    
    # 第一遍扫描：统计每行的出现次数
    line_counts = {}
    
    for file_path in input_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                # 使用行的哈希值作为键来节省内存
                line_hash = hashlib.md5(line.encode('utf-8')).hexdigest()
                line_counts[line_hash] = line_counts.get(line_hash, 0) + 1
    
    # 第二遍扫描：重写每个文件，只保留唯一的行
    for file_path in input_files:
        # 构建输出文件路径
        output_file_path = os.path.join(output_directory, os.path.basename(file_path))
        
        with open(file_path, 'r', encoding='utf-8') as input_file, \
             open(output_file_path, 'w', encoding='utf-8') as output_file:
            
            for line in input_file:
                # 检查这行是否只出现一次
                line_hash = hashlib.md5(line.encode('utf-8')).hexdigest()
                if line_counts[line_hash] == 1:
                    output_file.write(line)


def normalize_text(text: str) -> str:
    """
    规范化文本：小写、去标点、规范化空白、去音标、NFD Unicode规范化
    """
    # NFD Unicode规范化
    text = unicodedata.normalize('NFD', text)
    
    # 去除音标（combining characters）
    text = ''.join(char for char in text if unicodedata.category(char) != 'Mn')
    
    # 转换为小写
    text = text.lower()
    
    # 去除标点符号，保留字母、数字和空白字符
    text = re.sub(r'[^\w\s]', '', text)
    
    # 规范化空白字符（将多个空白字符替换为单个空格）
    text = re.sub(r'\s+', ' ', text)
    
    # 去除首尾空白
    text = text.strip()
    
    return text


def generate_ngrams(text: str, n: int) -> Set[str]:
    """
    生成文本的n-grams集合
    """
    words = text.split()
    ngrams = set()
    
    for i in range(len(words) - n + 1):
        ngram = ' '.join(words[i:i + n])
        ngrams.add(ngram)
    
    return ngrams


def compute_minhash_signature(ngrams: Set[str], num_hashes: int) -> List[int]:
    """
    计算n-grams集合的MinHash签名
    """
    # 初始化签名为无穷大
    signature = [float('inf')] * num_hashes
    
    for ngram in ngrams:
        ngram_bytes = ngram.encode('utf-8')
        
        for i in range(num_hashes):
            # 使用不同的种子计算哈希值
            hash_value = mmh3.hash(ngram_bytes, seed=i, signed=False)
            signature[i] = min(signature[i], hash_value)
    
    # 将无穷大替换为0（空集合的情况）
    signature = [int(val) if val != float('inf') else 0 for val in signature]
    
    return signature


def lsh_candidates(signatures: Dict[str, List[int]], num_bands: int) -> Set[Tuple[str, str]]:
    """
    使用LSH找到候选重复对
    """
    num_hashes = len(next(iter(signatures.values())))
    rows_per_band = num_hashes // num_bands
    
    candidates = set()
    
    # 为每个band创建哈希表
    for band_idx in range(num_bands):
        band_buckets = defaultdict(list)
        
        start_idx = band_idx * rows_per_band
        end_idx = start_idx + rows_per_band
        
        for doc_id, signature in signatures.items():
            # 提取当前band的签名部分
            band_signature = tuple(signature[start_idx:end_idx])
            # 将文档添加到对应的bucket
            band_buckets[band_signature].append(doc_id)
        
        # 在同一bucket中的文档是候选重复对
        for bucket_docs in band_buckets.values():
            if len(bucket_docs) > 1:
                for i in range(len(bucket_docs)):
                    for j in range(i + 1, len(bucket_docs)):
                        candidates.add((bucket_docs[i], bucket_docs[j]))
    
    return candidates


def jaccard_similarity(set1: Set[str], set2: Set[str]) -> float:
    """
    计算两个集合的Jaccard相似度
    """
    if not set1 and not set2:
        return 1.0
    
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    
    return intersection / union if union > 0 else 0.0


def cluster_duplicates(candidates: Set[Tuple[str, str]], 
                      document_ngrams: Dict[str, Set[str]], 
                      jaccard_threshold: float) -> List[Set[str]]:
    """
    基于Jaccard相似度对候选重复进行聚类
    """
    # 构建相似度图
    similarity_graph = defaultdict(set)
    
    for doc1, doc2 in candidates:
        ngrams1 = document_ngrams[doc1]
        ngrams2 = document_ngrams[doc2]
        
        similarity = jaccard_similarity(ngrams1, ngrams2)
        
        if similarity >= jaccard_threshold:
            similarity_graph[doc1].add(doc2)
            similarity_graph[doc2].add(doc1)
    
    # 使用深度优先搜索找到连通分量
    visited = set()
    clusters = []
    
    def dfs(node: str, cluster: Set[str]):
        if node in visited:
            return
        visited.add(node)
        cluster.add(node)
        
        for neighbor in similarity_graph[node]:
            dfs(neighbor, cluster)
    
    for doc_id in similarity_graph:
        if doc_id not in visited:
            cluster = set()
            dfs(doc_id, cluster)
            if cluster:
                clusters.append(cluster)
    
    return clusters


def minhash_deduplication(
    input_files: list[os.PathLike],
    num_hashes: int,
    num_bands: int,
    ngrams: int,
    jaccard_threshold: float,
    output_directory: os.PathLike,
):
    """
    使用MinHash和LSH进行模糊文档去重
    """
    # 确保输出目录存在
    os.makedirs(output_directory, exist_ok=True)
    
    # 读取所有文档并计算n-grams
    documents = {}
    document_ngrams = {}
    signatures = {}
    
    for file_path in input_files:
        doc_id = os.path.basename(file_path)
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            documents[doc_id] = content
            
            # 规范化文本
            normalized_content = normalize_text(content)
            
            # 生成n-grams
            doc_ngrams = generate_ngrams(normalized_content, ngrams)
            document_ngrams[doc_id] = doc_ngrams
            
            # 计算MinHash签名
            signature = compute_minhash_signature(doc_ngrams, num_hashes)
            signatures[doc_id] = signature
    
    # 使用LSH找到候选重复对
    candidates = lsh_candidates(signatures, num_bands)
    
    # 基于真实Jaccard相似度进行聚类
    clusters = cluster_duplicates(candidates, document_ngrams, jaccard_threshold)
    
    # 确定要保留的文档
    documents_to_keep = set(documents.keys())
    
    for cluster in clusters:
        if len(cluster) > 1:
            # 在每个聚类中随机保留一个文档，移除其他文档
            cluster_list = list(cluster)
            # 保留第一个，移除其他
            for doc_to_remove in cluster_list[1:]:
                documents_to_keep.discard(doc_to_remove)
    
    # 写入保留的文档到输出目录
    for doc_id in documents_to_keep:
        output_file_path = os.path.join(output_directory, doc_id)
        with open(output_file_path, 'w', encoding='utf-8') as f:
            f.write(documents[doc_id])