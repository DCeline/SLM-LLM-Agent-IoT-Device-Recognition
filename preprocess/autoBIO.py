import json
import re
import os
from tqdm import tqdm
from transformers import RobertaTokenizerFast

def auto_label_device(jsonl_path, output_jsonl_path, tokenizer):
    """
    自动标注报文中的设备型号实体，每条样本都标注BIOE，如果没有识别到设备型号，全标O
    :param jsonl_path: 输入JSONL文件路径（每行一个JSON对象）
    :param output_jsonl_path: 输出标注后的JSONL文件路径
    :param tokenizer: 使用的分词器（RobertaTokenizerFast）
    """
    # 读取JSONL文件
    dataset = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                item = json.loads(line.strip())
                dataset.append(item)
    
    # 存储所有标注后的样本
    all_samples = []
    
    for idx, item in enumerate(tqdm(dataset, desc="Processing samples")):
        # 获取文本和设备型号
        banner = item.get('banner', '')  # 使用get避免KeyError
        if not banner:  # 如果banner为空，跳过
            continue
        model_str = item['product'].strip()
        sample_type = item.get('sample_type', '')  # 如果没有sample_type，则为空字符串

        # 跳过空设备型号（但仍标注为O）
        if not model_str:
            # 使用 RobertaTokenizerFast 获取分词
            encoding = tokenizer(banner, return_offsets_mapping=True, add_special_tokens=False)
            tokens = encoding.tokens()
            tags = ['O'] * len(tokens)
            sample_data = {
                'id': str(item['id']),
                'banner': banner,
                'model': model_str,
                'tokens': tokens,
                'labels': tags,
                'sample_type': sample_type,
                'pos': False,
                'protocol': item.get('protocol', ''),
                'req_name': item.get('req_name', ''),
                'type': item.get('type', ''),
                'brand': item.get('oem_brand', '') or ''
            }
            all_samples.append(sample_data)
            continue
            
        # 使用 RobertaTokenizerFast 获取分词和位置信息
        encoding = tokenizer(banner, return_offsets_mapping=True, add_special_tokens=False)
        tokens = encoding.tokens()
        positions = encoding.offset_mapping  # 每个 token 的起止位置 (start, end)

        # 初始化标签
        tags = ['O'] * len(tokens)
        found = False
        
        # 查找设备型号出现位置（支持多次出现）
        for match in re.finditer(re.escape(model_str), banner):
            start_idx, end_idx = match.span()
            # 记录匹配到的 token 索引
            match_indices = []
            
            for i, (token_start, token_end) in enumerate(positions):
                if (token_start >= start_idx and token_end <= end_idx):
                    match_indices.append(i)
            
            # 检查连续性
            if match_indices and match_indices == list(range(match_indices[0], match_indices[-1] + 1)):
                # 标注 B-I-E 标签
                for j, idx in enumerate(match_indices):
                    if j == 0:
                        tags[idx] = 'B-DEVICE'
                    elif j == len(match_indices) - 1:
                        tags[idx] = 'E-DEVICE'
                    else:
                        tags[idx] = 'I-DEVICE'
                found = True
        
        # 样本数据
        sample_data = {
            'id': str(item['id']),
            'banner': banner,
            'model': model_str,
            'tokens': tokens,
            'labels': tags,
            'sample_type': sample_type,
            'pos': found,
            'protocol': item.get('protocol', ''),
            'req_name': item.get('req_name', ''),
            'type': item.get('type', ''),
            'brand': item.get('oem_brand', '') or ''
        }
        all_samples.append(sample_data)
    
    # 保存为JSONL格式
    with open(output_jsonl_path, 'w', encoding='utf-8') as f:
        for sample in all_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print(f"标注完成！总样本：{len(all_samples)}条")

# 配置路径并执行
if __name__ == "__main__":
    # 初始化 RobertaTokenizerFast
    tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
    
    auto_label_device(
        jsonl_path="../DATAsets/sampled_8p_cleaned.jsonl",
        output_jsonl_path="../OUTPUTs/sampled_8p_labeled.jsonl",
        tokenizer=tokenizer
    )

# python autoBIO.py