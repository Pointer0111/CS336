from cs336_data.filter import extract_text_from_html_bytes
from fastwarc.warc import WarcRecordType, ArchiveIterator
import os
import json
import random
from typing import Optional

def process_warc_file(warc_file_path: str, output_file_path: Optional[str] = None, max_records: int = 500):
    """
    处理 WARC 文件，提取前 max_records 个 response 记录的文本内容并保存
    
    Args:
        warc_file_path: WARC 文件路径
        output_file_path: 输出文件路径，如果为 None 则自动生成
        max_records: 最大处理记录数
    """
    if output_file_path is None:
        # 自动生成输出文件名
        base_name = os.path.splitext(os.path.basename(warc_file_path))[0]
        if base_name.endswith('.warc'):
            base_name = base_name[:-5]
        output_file_path = f"{base_name}_extracted_text.txt"
    
    record_count = 0
    response_count = 0
    extracted_texts = []
    
    print(f"Processing WARC file: {warc_file_path}")
    print(f"Max records to process: {max_records}")
    
    with open(warc_file_path, 'rb') as f:
        for record in ArchiveIterator(f):
            record_count += 1
            
            # 只处理有用的记录类型
            if record.record_type in [WarcRecordType.response, WarcRecordType.conversion]:
                response_count += 1
                
                # 使用 reader 来获取内容
                content = record.reader.read()
                text = extract_text_from_html_bytes(content)
                
                if text:
                    extracted_texts.append(text)
                    if response_count <= 10:  # 只打印前10个的简要信息
                        record_type_name = "response" if record.record_type == WarcRecordType.response else "conversion"
                        print(f"Record {response_count} ({record_type_name}): {len(content)} bytes -> {len(text)} chars")
                
                # 达到最大记录数后停止
                if response_count >= max_records:
                    break
    
    print(f"Processed {record_count} total records, found {response_count} response records")
    print(f"Successfully extracted text from {len(extracted_texts)} records")
    
    # 保存提取的文本到文件
    with open(output_file_path, 'w', encoding='utf-8') as f:
        for text in extracted_texts:
            f.write(text)
    
    print(f"Extracted text saved to: {output_file_path}")
    
    # 随机选择20个样本并保存为JSON
    if len(extracted_texts) >= 20:
        sample_size = 20
    else:
        sample_size = len(extracted_texts)
    
    random.seed(42)  # 设置随机种子以确保结果可重现
    sample_indices = random.sample(range(len(extracted_texts)), sample_size)
    sample_texts = [extracted_texts[i] for i in sample_indices]
    
    # 创建样本数据
    samples = []
    for i, (idx, text) in enumerate(zip(sample_indices, sample_texts)):
        sample = {
            "sample_id": i + 1,
            "original_index": idx,
            "text_length": len(text),
            "text_preview": text[:200] + "..." if len(text) > 200 else text,
            "full_text": text
        }
        samples.append(sample)
    
    # 保存样本到JSON文件
    sample_output_path = output_file_path.replace('.txt', '_samples.json')
    with open(sample_output_path, 'w', encoding='utf-8') as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)
    
    print(f"Random samples saved to: {sample_output_path}")
    print(f"Selected {sample_size} random samples from {len(extracted_texts)} total texts")
    
    # 返回统计信息
    total_text_length = sum(len(text) for text in extracted_texts)
    print(f"Total extracted text length: {total_text_length} characters")
    
    return {
        'total_records': record_count,
        'response_records': response_count,
        'extracted_records': len(extracted_texts),
        'total_text_length': total_text_length,
        'output_file': output_file_path,
        'extracted_texts': extracted_texts
    }


def compare_with_wet_file(warc_file_path: str, wet_file_path: str, max_records: int = 500):
    """
    比较 WARC 文件提取的文本与对应的 WET 文件
    
    Args:
        warc_file_path: WARC 文件路径
        wet_file_path: 对应的 WET 文件路径
        max_records: 最大处理记录数
    """
    print("=" * 50)
    print("WARC vs WET 文件比较")
    print("=" * 50)
    
    # 处理 WARC 文件
    warc_stats = process_warc_file(warc_file_path, max_records=max_records)
    
    # 读取 WET 文件进行比较
    print(f"\n读取 WET 文件: {wet_file_path}")
    wet_texts = []
    
    try:
        import gzip
        with gzip.open(wet_file_path, 'rt', encoding='utf-8') as f:
            current_text = ""
            record_count = 0
            in_header = True
            in_content = False
            
            for line in f:
                line = line.rstrip('\n')
                
                # 检测新记录的开始
                if line.startswith('WARC/1.0'):
                    # 保存前一个记录的内容
                    if current_text.strip() and in_content:
                        wet_texts.append(current_text.strip())
                        record_count += 1
                        if record_count >= max_records:
                            break
                    
                    # 开始新记录
                    current_text = ""
                    in_header = True
                    in_content = False
                    continue
                
                # 跳过头部信息
                if in_header:
                    if line.strip() == '':
                        # 空行表示头部结束，接下来是内容
                        in_header = False
                        in_content = True
                    continue
                
                # 收集内容
                if in_content:
                    current_text += line + '\n'
            
            # 添加最后一个文本块
            if current_text.strip() and in_content and record_count < max_records:
                wet_texts.append(current_text.strip())
        
        print(f"WET 文件中的文本块数量: {len(wet_texts)}")
        wet_total_length = sum(len(text) for text in wet_texts)
        print(f"WET 文件总文本长度: {wet_total_length} 字符")
        
        # 比较结果
        print("\n比较结果:")
        print(f"WARC 提取的文本块: {warc_stats['extracted_records']}")
        print(f"WET 文件中的文本块: {len(wet_texts)}")
        print(f"WARC 总文本长度: {warc_stats['total_text_length']} 字符")
        print(f"WET 总文本长度: {wet_total_length} 字符")
        
        if warc_stats['extracted_records'] > 0 and len(wet_texts) > 0:
            # 比较前几个文本块
            print("\n前3个文本块的比较:")
            for i in range(min(3, warc_stats['extracted_records'], len(wet_texts))):
                print(f"\n--- 文本块 {i+1} ---")
                warc_text = warc_stats['extracted_texts'][i]
                wet_text = wet_texts[i]
                print(f"WARC: {len(warc_text)} 字符")
                print(f"WET:  {len(wet_text)} 字符")
                print(f"WARC 前50字符: {warc_text[:50]}...")
                print(f"WET  前50字符: {wet_text[:50]}...")
        
    except Exception as e:
        print(f"读取 WET 文件时出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    warc_file_path = "example.warc.gz"
    wet_file_path = "example.warc.wet.gz"
    
    # 处理 WARC 文件（只处理前500条记录）
    stats = process_warc_file(warc_file_path, max_records=500)
    
    # 与 WET 文件比较
    compare_with_wet_file(warc_file_path, wet_file_path, max_records=500)










