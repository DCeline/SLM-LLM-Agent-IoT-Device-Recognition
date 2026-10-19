import json
import torch
from torch.utils.data import Dataset
import os
import random

class DeviceNERDataset(Dataset):
    def __init__(self, json_path, tokenizer, seqlen=128, stride=64, debug=False):
        """
        加载并预处理NER数据集，支持滑动窗口机制
        :param json_path: 数据集路径
        :param tokenizer: 分词器
        :param seqlen: 滑动窗口的长度
        :param stride: 滑动窗口的步幅
        :param debug: 若为 True，则在发现标签与 token 数不匹配时打印调试信息
        """
        # 兼容 JSON 数组 和 JSONL（每行一个 JSON）
        data = []
        with open(json_path, 'r', encoding='utf-8') as f:
            head = f.read(1024)
            first_nonspace = next((c for c in head if not c.isspace()), None)
            f.seek(0)
            if first_nonspace == '[':
                # 整个文件是一个 JSON 数组
                data = json.load(f)
            else:
                # 假定为 JSONL：每行一个 JSON
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    data.append(json.loads(line))
        
        self.tokenizer = tokenizer
        self.seqlen = seqlen
        self.stride = stride
        self.debug = debug
        self.label_map = {"O": 0, "B-DEVICE": 1, "I-DEVICE": 2, "E-DEVICE": 3}
        self.inverse_label_map = {v: k for k, v in self.label_map.items()}
        
        self.samples = []
        mismatch_count = 0
        for item in data:
            banner = item['banner']
            labels = item.get('labels', [])
            
            # 对文本进行分词和编码
            encoding = self.tokenizer(
                banner,
                add_special_tokens=True,
                return_offsets_mapping=True,
                return_tensors='pt'
            )
            
            input_ids = encoding['input_ids'].squeeze(0)
            attention_mask = encoding['attention_mask'].squeeze(0)
            offset_mapping = encoding['offset_mapping'].squeeze(0).tolist()
            
            # 生成标签序列
            label_ids = self._generate_labels(offset_mapping, labels)
            
            # 调试：检查 token-level 标签与非-special token 数量是否对齐
            if self.debug and labels and isinstance(labels[0], str):
                non_special_tokens = [o for o in offset_mapping if not (o[0] == o[1] == 0)]
                if len(labels) != len(non_special_tokens):
                    mismatch_count += 1
                    if mismatch_count <= 10:  # 只打印前 10 个供排查
                        tokens = tokenizer.convert_ids_to_tokens(input_ids.tolist())
                        print("===== LABEL/TOKEN MISMATCH =====")
                        print(f"文本前100: {banner[:100]!r}")
                        print(f"token 总数 (含 special): {len(offset_mapping)}, 非-special token 数: {len(non_special_tokens)}")
                        print(f"labels 长度: {len(labels)}")
                        print(f"tokens (前50): {tokens[:50]}")
                        print(f"offsets (前50): {offset_mapping[:50]}")
                        print(f"labels (前50): {labels[:50]}")
                        print("================================")
            
            # 滑动窗口切片
            for start in range(0, len(input_ids), self.stride):
                end = start + self.seqlen
                if start >= len(input_ids):
                    break
                
                slice_input_ids = input_ids[start:end]
                slice_attention_mask = attention_mask[start:end]
                slice_labels = label_ids[start:end]
                
                # 确保 mask 的第一个时间步全为 1
                if slice_attention_mask.size(0) > 0:
                    slice_attention_mask[0] = 1
                
                # 填充到固定长度
                padding_length = self.seqlen - slice_input_ids.size(0)
                if padding_length > 0:
                    slice_input_ids = torch.cat([slice_input_ids, torch.zeros(padding_length, dtype=torch.long)])
                    slice_attention_mask = torch.cat([slice_attention_mask, torch.zeros(padding_length, dtype=torch.long)])
                    slice_labels = torch.cat([slice_labels, torch.zeros(padding_length, dtype=torch.long)])
                
                self.samples.append({
                    'input_ids': slice_input_ids,
                    'attention_mask': slice_attention_mask,
                    'labels': slice_labels,
                    'banner': banner
                })

    def _generate_labels(self, offset_mapping, labels):
        """
        根据 offset_mapping 和原始标签生成标签序列
        支持两种 labels 格式：
         - span 列表：[(start_char, end_char, label_str), ...]
         - token-level 列表：["O","B-DEVICE",...]（与 tokenizer 的非-special tokens 对齐）
        :param offset_mapping: 每个 token 的起止位置
        :param labels: 原始标签
        :return: 标签序列
        """
        label_ids = [self.label_map['O']] * len(offset_mapping)

        # 如果 labels 是 span triples 格式 (start,end,label)
        if labels and isinstance(labels[0], (list, tuple)) and len(labels[0]) >= 3:
            for label in labels:
                if not (isinstance(label, (list, tuple)) and len(label) >= 3):
                    continue
                label_start, label_end, _ = label
                entity_token_indices = []
                for i, (token_start, token_end) in enumerate(offset_mapping):
                    if token_start == token_end == 0:
                        continue
                    # 判断 token 与实体 span 是否有交集（半开区间）
                    if max(token_start, label_start) < min(token_end, label_end):
                        entity_token_indices.append(i)

                if entity_token_indices:
                    if len(entity_token_indices) == 1:
                        # 单 token 实体，标为 B-DEVICE（CRF transition 已允许 B->O 表示单字实体）
                        label_ids[entity_token_indices[0]] = self.label_map['B-DEVICE']
                    else:
                        label_ids[entity_token_indices[0]] = self.label_map['B-DEVICE']
                        for idx in entity_token_indices[1:-1]:
                            label_ids[idx] = self.label_map['I-DEVICE']
                        label_ids[entity_token_indices[-1]] = self.label_map['E-DEVICE']

        # 如果 labels 是 token-level 字符串列表（来自 autoBIO 的输出）
        elif labels and isinstance(labels[0], str):
            j = 0
            for i, (token_start, token_end) in enumerate(offset_mapping):
                if token_start == token_end == 0:
                    # special token（例如 <s>, </s>, pad），保持 O
                    continue
                if j >= len(labels):
                    # 如果标签数量比非-special token 少，剩余保持 O
                    break
                tag = labels[j]
                label_ids[i] = self.label_map.get(tag, self.label_map['O'])
                j += 1
            # 如果 labels 比非-special token 多，超出的部分忽略

        # 其他不符合格式的情况保持全 O
        return torch.tensor(label_ids, dtype=torch.long)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def iter_json_items(path):
    """按流式方式迭代大 JSON 文件的元素：
       - 如果文件以'['开头，尝试用 ijson 解析大数组（需 pip install ijson）
       - 否则按 JSONL（每行一个 JSON）逐行解析
    """
    with open(path, 'r', encoding='utf-8') as f:
        head = f.read(1024)
        first_nonspace = next((c for c in head if not c.isspace()), None)
        f.seek(0)
        if first_nonspace == '[':
            try:
                import ijson
            except Exception:
                raise RuntimeError("输入是大 JSON 数组，需安装 ijson：pip install ijson，或将文件转为 JSONL（一行一个 JSON）。")
            for item in ijson.items(f, 'item'):
                yield item
        else:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)

def split_dataset_to_files(single_path, out_dir=None, ratios=(95, 5, 0.5), seed=42):
    """
    single_path: 输入数据集路径（JSONL或JSON数组）
    out_dir: 输出目录（若 None 则与 single_path 同目录或 DATAsets）
    ratios: 三元组，支持百分比或小数（例如 (95,5,0.5) 或 (0.95,0.05,0.005)）
    返回: (train_path, val_path, test_path, counts_dict)
    实现：优先尝试按 labels 做分层抽样（使用 sklearn.model_selection.train_test_split）
    若不可用或分层条件不满足（类别样本过少），回退到随机切分实现。
    """
    if out_dir is None:
        out_dir = os.path.join(os.path.dirname(single_path) or ".", "DATAsets")
    os.makedirs(out_dir, exist_ok=True)

    # 标准化 ratios：如果任一值 >= 1，则认为是百分比，先除以100
    fracs = []
    for r in ratios:
        if r >= 1:
            fracs.append(r / 100.0)
        else:
            fracs.append(r)
    total = sum(fracs)
    if total <= 0:
        raise ValueError("无效的划分比例")
    fracs = [f / total for f in fracs]  # 归一化为和为1

    # 读取所有样本到列表（若数据极大需注意内存）
    items = []
    for it in iter_json_items(single_path):
        items.append(it)

    n = len(items)
    if n == 0:
        raise ValueError("输入数据集为空")

    # 计算用于分层的标签（stratify keys）
    stratify_keys = []
    for it in items:
        labels = it.get('labels', [])
        key = 'no_entity'
        if labels:
            # span triples 格式
            if isinstance(labels[0], (list, tuple)) and len(labels[0]) >= 3:
                if len(labels) > 0:
                    key = 'has_entity'
            # token-level 字符串列表
            elif isinstance(labels[0], str):
                # 判断是否存在非-O 标签（视为含实体）
                has = any(tag != 'O' for tag in labels)
                key = 'has_entity' if has else 'no_entity'
        stratify_keys.append(key)

    # 计算第一步（train_val vs test）的 test_size 与第二步（train vs val）的相对比例
    test_frac = fracs[2]
    train_val_frac = fracs[0] + fracs[1]
    val_relative_frac = 0.0
    if train_val_frac > 0:
        val_relative_frac = fracs[1] / train_val_frac

    # 尝试使用 sklearn 分层抽样
    try:
        from sklearn.model_selection import train_test_split
        # 先判断每个类的样本数是否足够做分层（每类至少2个通常为安全阈值）
        from collections import Counter
        cnt = Counter(stratify_keys)
        enough_for_stratify = all(v >= 2 for v in cnt.values())

        if enough_for_stratify:
            # train_val 与 test
            train_val_items, test_items, train_val_keys, test_keys = train_test_split(
                items,
                stratify_keys,
                test_size=test_frac,
                random_state=seed,
                shuffle=True,
            )
            # train 与 val（在 train_val 内继续分层）
            if val_relative_frac > 0 and len(train_val_items) > 0:
                # 进一步检查 train_val 中每类样本是否足够
                tv_cnt = Counter(train_val_keys)
                if all(v >= 2 for v in tv_cnt.values()):
                    train_items, val_items, _, _ = train_test_split(
                        train_val_items,
                        train_val_keys,
                        test_size=val_relative_frac,
                        stratify=train_val_keys,
                        random_state=seed,
                        shuffle=True,
                    )
                else:
                    # 回退到不分层的随机划分（在 train_val 内）
                    random.seed(seed)
                    random.shuffle(train_val_items)
                    n_val = int(round(val_relative_frac * len(train_val_items)))
                    val_items = train_val_items[:n_val]
                    train_items = train_val_items[n_val:]
            else:
                # 无验证集或 val 比例为0
                train_items = train_val_items
                val_items = []
        else:
            # 类别过少，回退到随机划分
            raise ValueError("类别样本数不足以进行分层划分")
    except Exception:
        # 回退到原先的随机划分逻辑（基于归一化 fracs）
        random.seed(seed)
        random.shuffle(items)
        n_train = int(round(fracs[0] * n))
        n_val = int(round(fracs[1] * n))
        n_test = n - n_train - n_val
        if n_test < 0:
            n_test = 0
            rem = n - n_train - n_val
            if rem < 0:
                n_train += rem
                if n_train < 0:
                    n_train = 0
        train_items = items[:n_train]
        val_items = items[n_train:n_train + n_val]
        test_items = items[n_train + n_val:]

    # 写出为 JSONL
    train_path = os.path.join(out_dir, "train_split.json")
    val_path = os.path.join(out_dir, "val_split.json")
    test_path = os.path.join(out_dir, "test_split.json")

    def write_jsonl(path, seq):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            for it in seq:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")

    write_jsonl(train_path, train_items)
    write_jsonl(val_path, val_items)
    write_jsonl(test_path, test_items)

    counts = {"total": n, "train": len(train_items), "val": len(val_items), "test": len(test_items)}
    return train_path, val_path, test_path, counts

def create_k_fold_files_from_items(items, out_dir, k=5, seed=42):
    os.makedirs(out_dir, exist_ok=True)
    random.seed(seed)
    n = len(items)
    indices = list(range(n))
    random.shuffle(indices)
    fold_sizes = [n // k + (1 if x < (n % k) else 0) for x in range(k)]
    folds = []
    cursor = 0
    for size in fold_sizes:
        folds.append(indices[cursor:cursor+size])
        cursor += size

    fold_paths = []
    for i in range(k):
        val_idx = set(folds[i])
        train_seq = [items[j] for j in range(n) if j not in val_idx]
        val_seq = [items[j] for j in folds[i]]
        train_path = os.path.join(out_dir, f"train_fold_{i}.json")
        val_path = os.path.join(out_dir, f"val_fold_{i}.json")
        with open(train_path, 'w', encoding='utf-8') as f:
            for it in train_seq:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
        with open(val_path, 'w', encoding='utf-8') as f:
            for it in val_seq:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
        fold_paths.append((train_path, val_path, len(train_seq), len(val_seq)))
    return fold_paths
