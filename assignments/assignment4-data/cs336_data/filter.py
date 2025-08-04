import re
import fasttext
from typing import Any
from resiliparse.extract.html2text import extract_plain_text
from resiliparse.parse.encoding import detect_encoding


ASSETS_DIR = "cs336_data/assets"

def extract_text_from_html_bytes(html_bytes: bytes) -> str | None:
    try:
        # 检测编码
        detected_encoding = detect_encoding(html_bytes)
        
        # 如果检测到编码，使用检测到的编码解码
        if detected_encoding:
            html_string = html_bytes.decode(detected_encoding)
        else:
            # 如果检测失败，尝试使用 UTF-8 解码
            html_string = html_bytes.decode('utf-8')
        
        # 提取纯文本
        text = extract_plain_text(html_string)
        return text
        
    except (UnicodeDecodeError, Exception):
        # 如果解码失败或其他错误，返回 None
        return None



def identify_language(text: str) -> tuple[Any, float]:
    model_path = f"{ASSETS_DIR}/lid.176.bin"
    model = fasttext.load_model(model_path)

    # 处理换行符：将文本中的换行符替换为空格
    text_processed = text.replace('\n', ' ')
    
    # 预测语言
    labels, probs = model.predict(text_processed, k=1)
    
    # 语言标签格式通常是 "__label__en"，我们需要提取 "en" 部分
    predicted_language = labels[0].replace('__label__', '') if labels else 'unknown'
    confidence_score = float(probs[0]) if probs else 0.0  # 转换为 Python float

    return (predicted_language, confidence_score)


# 学会写相应的正则表达式！！！
def _mask_pattern(text: str, pattern: str, replacement: str) -> tuple[str, int]:
    """
    通用的模式匹配和替换函数
    
    Args:
        text: 输入文本
        pattern: 正则表达式模式
        replacement: 替换字符串
    
    Returns:
        tuple[str, int]: (替换后的文本, 匹配数量)
    """
    # 找到所有匹配的内容
    matches = re.findall(pattern, text)
    num_matches = len(matches)
    
    # 替换所有匹配的内容
    masked_text = re.sub(pattern, replacement, text)
    
    return (masked_text, num_matches)


def mask_emails(text: str) -> tuple[str, int]:
    # 邮箱地址的正则表达式模式
    # 匹配格式：用户名@域名.顶级域名
    # 用户名可以包含字母、数字、点号、下划线、连字符
    # 域名可以包含字母、数字、连字符
    # 顶级域名至少2个字符
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b'
    
    return _mask_pattern(text, email_pattern, "|||EMAIL_ADDRESS|||")


def mask_phone_numbers(text: str) -> tuple[str, int]:
    # 美国电话号码的正则表达式模式
    # 支持以下格式：
    # - 纯数字：2831823829
    # - 带括号和连字符：(283)-182-3829
    # - 带括号和空格：(283) 182 3829
    # - 带连字符：283-182-3829
    # - 带空格：283 182 3829
    # - 带点号：283.182.3829
    
    # 使用一个综合的正则表达式来匹配所有格式
    phone_pattern = r'\(\d{3}\)[-\s.]?\d{3}[-\s.]?\d{4}|\b\d{3}[-\s.]?\d{3}[-\s.]?\d{4}\b'
    
    return _mask_pattern(text, phone_pattern, "|||PHONE_NUMBER|||")


def mask_ips(text: str) -> tuple[str, int]:
    # 匹配IPv4地址
    ip_pattern = r'\b(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b'
    
    return _mask_pattern(text, ip_pattern, "|||IP_ADDRESS|||")



def classify_nsfw(text: str) -> tuple[Any, float]:
    model_path = f"{ASSETS_DIR}/dolma_fasttext_nsfw_jigsaw_model.bin"
    model = fasttext.load_model(model_path)

    labels, probs = model.predict(text, k=1)
    
    # 提取标签（去掉 "__label__" 前缀）
    predicted_label = labels[0].replace('__label__', '') if labels else 'unknown'
    confidence_score = float(probs[0]) if probs else 0.0
    
    return (predicted_label, confidence_score)


def classify_toxic_speech(text: str) -> tuple[Any, float]:
    model_path = f"{ASSETS_DIR}/dolma_fasttext_hatespeech_jigsaw_model.bin"
    model = fasttext.load_model(model_path)

    labels, probs = model.predict(text, k=1)
    
    predicted_label = labels[0].replace('__label__', '') if labels else 'unknown'
    confidence_score = float(probs[0]) if probs else 0.0
    
    return (predicted_label, confidence_score)



def gopher_quality_filter(text: str) -> bool:
    
    # 使用简单的空格分割
    tokens = text.split()
    if len(tokens) < 50 or len(tokens) > 100000:
        return False

    # 检查平均词长是否在3到10之间
    mean_word_len = sum(len(token) for token in tokens) / len(tokens)
    if mean_word_len < 3 or mean_word_len > 10:
        return False

    # 检查超过30%的行以省略号结尾
    lines = text.splitlines()
    if lines:
        ellipsis_count = sum(1 for line in lines if line.rstrip().endswith("..."))
        if ellipsis_count / len(lines) > 0.3:
            return False

    # 检查至少80%的词包含字母
    word_has_alpha = [bool(re.search(r'[A-Za-z]', token)) for token in tokens]
    if len(word_has_alpha) == 0 or sum(word_has_alpha) / len(word_has_alpha) < 0.8:
        return False
    
    return True


def classify_quality(text: str) -> tuple[Any, float]:
    """
    使用训练好的FastText模型对文本质量进行分类
    
    Args:
        text: 要分类的文本
    
    Returns:
        tuple[Any, float]: (预测标签, 置信度分数)
    """
    model_path = f"{ASSETS_DIR}/quality_classifier.bin"
    
    try:
        # 加载质量分类模型
        model = fasttext.load_model(model_path)
        
        # 处理文本：移除换行符，规范化空格
        processed_text = text.replace('\n', ' ').replace('\r', ' ')
        processed_text = re.sub(r'\s+', ' ', processed_text.strip())
        
        # 预测
        labels, probs = model.predict(processed_text, k=1)
        
        # 提取标签（去掉 "__label__" 前缀）
        predicted_label = labels[0].replace('__label__', '') if labels else 'unknown'
        confidence_score = float(probs[0]) if probs else 0.0
        
        return (predicted_label, confidence_score)
        
    except Exception as e:
        # 如果模型不存在或加载失败，使用简单的启发式方法
        # 基于文本长度和结构特征进行简单分类
        
        # 计算基本特征
        word_count = len(text.split())
        line_count = len(text.splitlines())
        avg_line_length = len(text) / max(line_count, 1)
        
        # 简单的启发式规则
        quality_score = 0.0
        
        # 基于长度的评分
        if 100 < word_count < 10000:
            quality_score += 0.3
        
        # 基于平均行长度的评分
        if avg_line_length > 50:
            quality_score += 0.3
        
        # 基于结构的评分（是否有段落）
        if '\n\n' in text:
            quality_score += 0.2
        
        # 基于内容质量的评分（大写字母比例）
        uppercase_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
        if 0.02 < uppercase_ratio < 0.1:
            quality_score += 0.2
        
        # 根据评分决定标签
        if quality_score > 0.5:
            return ("wiki", quality_score)
        else:
            return ("cc", 1.0 - quality_score)