import json
import torch
import os
from transformers import RobertaTokenizerFast
from model import BERT_CRF_NER
from train import train_model
from predict import predict_devices_with_sliding_window
from dataset import iter_json_items, split_dataset_to_files
import random

if __name__ == "__main__":
    # --- 配置参数 ---
    MODEL_NAME = 'roberta-base'
    SINGLE_DATA_PATH = 'OUTPUTs/sampled_8p_labeled.jsonl'  # 更改为当前规模的训练集文件，例如 sampled_0.5p_cleaned.json 等
    MODEL_SAVE_PATH = "models/per8_model.pth"  # 更改为对应规模的路径，例如 per0.5_model.pth
    RESULTS_SAVE_PATH = 'OUTPUTs/8per_model.jsonl'  # 更改为对应规模的输出路径
    SEQLEN = 64
    STRIDE = 32
    EPOCHS = 5
    BATCH_SIZE = 16
    LEARNING_RATE = 5e-5
    EARLY_STOPPING_PATIENCE = 3
    STREAM_BATCH_SIZE = 64

    # 从训练集中划分验证集：90% 训练，10% 验证，0% 测试（测试集预先生成）
    SPLIT_RATIOS = (0.9, 0.1, 0.0)
    print(f"--- 使用预先训练集 {SINGLE_DATA_PATH}，按比例 {SPLIT_RATIOS} 划分（训练+验证） ---")
    out_dir = os.path.join("DATAsets", "datasplit")
    os.makedirs(out_dir, exist_ok=True)
    train_path = os.path.join(out_dir, "train_split.json")
    val_path = os.path.join(out_dir, "val_split.json")
    # 测试集预先生成：test.jsonl = all_sample - 训练集样本
    TEST_DATA_PATH = 'DATAsets/testset_cleaned.jsonl'  # 预先生成的测试集路径

    # 划分训练集为 train_split 和 val_split
    if os.path.exists(train_path) and os.path.exists(val_path):
        def _count_items(path):
            return sum(1 for _ in iter_json_items(path)) if os.path.exists(path) else 0

        counts = {
            "train": _count_items(train_path),
            "val": _count_items(val_path),
        }
        counts["total"] = counts["train"] + counts["val"]
        print(f"已存在划分文件，跳过划分：train={train_path} ({counts['train']}), val={val_path} ({counts['val']})")
    else:
        train_path, val_path, _, counts = split_dataset_to_files(
            SINGLE_DATA_PATH,
            out_dir=out_dir,
            ratios=SPLIT_RATIOS,
            seed=42
        )
        print(f"划分完成：总 {counts['total']} -> train {counts['train']}, val {counts['val']}")

    TRAIN_DATA_PATH = train_path
    VAL_DATA_PATH = val_path

    label_map = {"O": 0, "B-DEVICE": 1, "I-DEVICE": 2, "E-DEVICE": 3}

    # 训练流程
    tokenizer = RobertaTokenizerFast.from_pretrained(MODEL_NAME)
    model = BERT_CRF_NER(roberta_model=MODEL_NAME, num_tags=len(label_map))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    map_loc = "cuda" if torch.cuda.is_available() else "cpu"
    if os.path.exists(MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=map_loc))
        print(f"--- 已有模型，加载完成: {MODEL_SAVE_PATH} ---")
    else:
        print("--- 开始训练模型 ---")
        val_path_arg = VAL_DATA_PATH if os.path.exists(VAL_DATA_PATH) else None
        train_model(
            model=model,
            tokenizer=tokenizer,
            train_json_path=TRAIN_DATA_PATH,
            val_json_path=val_path_arg,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            lr=LEARNING_RATE,
            model_save_path=MODEL_SAVE_PATH,
            seqlen=SEQLEN,
            stride=STRIDE,
            early_stopping_patience=EARLY_STOPPING_PATIENCE
        )
        model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=map_loc))
        print(f"--- 训练完成，加载最后模型: {MODEL_SAVE_PATH} ---")

    # --- 流式加载测试数据并分批预测、写出为 JSONL ---
    print(f"--- 流式读取测试数据从 {TEST_DATA_PATH} 并按批预测，输出到 {RESULTS_SAVE_PATH} ---")
    buffer = []
    count = 0
    os.makedirs(os.path.dirname(RESULTS_SAVE_PATH) or ".", exist_ok=True)
    with open(RESULTS_SAVE_PATH, 'w', encoding='utf-8') as outf:
        for item in iter_json_items(TEST_DATA_PATH):
            buffer.append(item)
            if len(buffer) >= STREAM_BATCH_SIZE:
                banners = [it['banner'] for it in buffer]
                batch_results = predict_devices_with_sliding_window(
                    model=model,
                    tokenizer=tokenizer,
                    banners=banners,
                    seqlen=SEQLEN,
                    stride=STRIDE,
                    verbose=False
                )
                for it, res in zip(buffer, batch_results):
                    it["bio_tags"] = res.get("bio_tags", [])
                    it["extracted_models"] = res.get("extracted_models", [])
                    outf.write(json.dumps(it, ensure_ascii=False) + "\n")
                    count += 1
                    if count % 100 == 0:
                        print(f"已写出 {count} 条")
                buffer = []

        # 处理剩余
        if buffer:
            banners = [it['banner'] for it in buffer]
            batch_results = predict_devices_with_sliding_window(
                model=model,
                tokenizer=tokenizer,
                banners=banners,
                seqlen=SEQLEN,
                stride=STRIDE,
                verbose=False
            )
            for it, res in zip(buffer, batch_results):
                it["bio_tags"] = res.get("bio_tags", [])
                it["extracted_models"] = res.get("extracted_models", [])
                outf.write(json.dumps(it, ensure_ascii=False) + "\n")
                count += 1

    print(f"--- 全部完成，已输出 {count} 条到 {RESULTS_SAVE_PATH}（JSONL） ---")
    print("说明：输出为 JSONL（每行一个 JSON），可按需合并为 JSON 数组")
