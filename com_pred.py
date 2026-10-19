"""
用于比较不同训练集比例设置的模型的预测效果，避免重复训练
输入：模型path、测试集数据path
输出：预测结果文件

方法和predict.py中一致
"""

import os
import json
import argparse
import time
import torch
from transformers import RobertaTokenizerFast
from model import BERT_CRF_NER
from predict import predict_devices_with_sliding_window
from dataset import iter_json_items

def safe_name(p):
    name = os.path.basename(p)
    name = name.replace(os.path.sep, "_")
    return name

def process_model(model_path, test_path, out_path, tokenizer_name="roberta-base",
                  seqlen=128, stride=64, batch_size=64, device=None, verbose=True):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    map_loc = "cuda" if torch.cuda.is_available() else "cpu"

    if verbose:
        print(f"[com_pred] 加载 tokenizer {tokenizer_name}")
    tokenizer = RobertaTokenizerFast.from_pretrained(tokenizer_name)

    if verbose:
        print(f"[com_pred] 初始化模型结构并加载权重 {model_path} -> device={device}")
    model = BERT_CRF_NER(roberta_model=tokenizer_name, num_tags=4)
    state = torch.load(model_path, map_location=map_loc)
    # 支持直接保存 whole model 或 state_dict 的情况
    if isinstance(state, dict) and all(k in state for k in ("roberta", "crf", "state_dict")):
        model.load_state_dict(state.get("state_dict", state))
    else:
        try:
            model.load_state_dict(state)
        except Exception:
            model = state
    model.to(device)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    outf = open(out_path, "w", encoding="utf-8")

    buffer = []
    count = 0
    t0 = time.time()
    last_report_time = t0       # 新增：上次打印时间
    last_report_count = 0       # 新增：上次打印时的已处理数量
    next_report = 10  # 首次在 10 条时打印，之后 <100 时每 10 条，>=100 后每 100 条

    # 新增：用于统计模型推理耗时（仅模型计算，不含写文件等开销）
    total_model_time = 0.0
    model_processed_count = 0

    for item in iter_json_items(test_path):
        buffer.append(item)
        if len(buffer) >= batch_size:
            texts = [it.get("text", "") for it in buffer]

            # 计时：模型推理开始
            t_model_start = time.time()
            try:
                results = predict_devices_with_sliding_window(
                    model=model,
                    tokenizer=tokenizer,
                    texts=texts,
                    seqlen=seqlen,
                    stride=stride,
                    verbose=False
                )
            except Exception as e:
                # 若模型推理失败，记录并跳过此次 batch（也可以选择重试）    
                print(f"[com_pred] predict failed for batch starting at {count+1}, error: {e}", flush=True)
                results = [ {"bio_tags": [], "extracted_models": []} for _ in texts ]
            t_model_elapsed = time.time() - t_model_start

            # 累加模型耗时与样本数
            total_model_time += t_model_elapsed
            model_processed_count += len(texts)

            for it, res in zip(buffer, results):
                it["bio_tags"] = res.get("bio_tags", [])
                it["extracted_models"] = res.get("extracted_models", [])
                it["_pred_model"] = model_path
                outf.write(json.dumps(it, ensure_ascii=False) + "\n")
                count += 1

                # 更可靠的进度打印：简化为只输出模型总耗时和模型均耗时
                if verbose and count >= next_report:
                    now = time.time()   # <- 确保 now 已定义
                    avg_model_per_sample = (total_model_time / model_processed_count) if model_processed_count > 0 else 0.0
                    print(f"[com_pred] 已写出 {count} 条 (最近模型 {safe_name(model_path)})，模型累计耗时 {total_model_time:.2f}s，模型均耗时 {avg_model_per_sample:.4f}s/条", flush=True)
                    # 更新上次打印标记（保留 next_report 逻辑）
                    last_report_time = now
                    last_report_count = count
                    if next_report < 100:
                        next_report += 10
                        if next_report > 100:
                            next_report = ((count // 100) + 1) * 100
                    else:
                        next_report = ((count // 100) + 1) * 100

            buffer = []
    
    # 处理剩余
    if buffer:
        texts = [it.get("text", "") for it in buffer]
        t_model_start = time.time()
        try:
            results = predict_devices_with_sliding_window(
                model=model,
                tokenizer=tokenizer,
                texts=texts,
                seqlen=seqlen,
                stride=stride,
                verbose=False
            )
        except Exception as e:
            print(f"[com_pred] predict failed for final batch, error: {e}", flush=True)
            results = [ {"bio_tags": [], "extracted_models": []} for _ in texts ]
        t_model_elapsed = time.time() - t_model_start
        total_model_time += t_model_elapsed
        model_processed_count += len(texts)

        for it, res in zip(buffer, results):
            it["bio_tags"] = res.get("bio_tags", [])
            it["extracted_models"] = res.get("extracted_models", [])
            it["_pred_model"] = model_path
            outf.write(json.dumps(it, ensure_ascii=False) + "\n")
            count += 1
            # 处理剩余时也检查是否需要打印进度
            if verbose and count >= next_report:
                now = time.time()
                elapsed_total = now - t0
                avg_model_per_sample = (total_model_time / model_processed_count) if model_processed_count > 0 else 0.0
                avg_total_per_sample = elapsed_total / count if count > 0 else 0.0
                print(f"[com_pred] 已写出 {count} 条 (最近模型 {safe_name(model_path)})，总耗时 {elapsed_total:.1f}s，模型均耗时 {avg_model_per_sample:.4f}s/条，总均耗时 {avg_total_per_sample:.4f}s/条", flush=True)
                if next_report < 100:
                    next_report += 10
                    if next_report > 100:
                        next_report = ((count // 100) + 1) * 100
                else:
                    next_report = ((count // 100) + 1) * 100

    outf.close()

    # 结束汇总打印：只输出模型相关耗时指标和总体耗时
    total_elapsed = time.time() - t0
    avg_model = (total_model_time / model_processed_count) if model_processed_count > 0 else 0.0
    avg_total = (total_elapsed / count) if count > 0 else 0.0
    if verbose:
        print(f"[com_pred] 模型 {model_path} 预测完成，已写出 {count} 条 到 {out_path}", flush=True)
        print(f"[com_pred] 模型累计耗时: {total_model_time:.2f}s；处理样本数: {model_processed_count}；模型均耗时: {avg_model:.4f}s/条；总耗时(含IO等): {total_elapsed:.2f}s", flush=True)

def main():
    parser = argparse.ArgumentParser(description="批量对比模型预测（同步版本，避免重复训练）")
    parser.add_argument("--models", "-m", nargs="+", required=True,
                        help="一个或多个模型文件路径（.pth），脚本会并发对每个模型生成预测结果文件）")
    parser.add_argument("--test", "-t", required=True, default="DATAsets/test_sample_10000.jsonl", help="测试数据 JSONL/JSON 路径（支持 dataset.iter_json_items 读取）")
    parser.add_argument("--out-dir", "-o", default="OUTPUTs/com_pred", help="输出目录（每个模型生成单独 JSONL）")
    parser.add_argument("--tokenizer", default="roberta-base", help="tokenizer 名称或路径（默认 roberta-base）")
    parser.add_argument("--seqlen", type=int, default=128)  # 修改默认值为128，与predict.py同步
    parser.add_argument("--stride", type=int, default=64)  # 修改默认值为64，与predict.py同步
    parser.add_argument("--batch-size", type=int, default=64, help="流式批量大小（每次 predict 的文本数）")
    parser.add_argument("--no-cuda", action="store_true", help="禁用 CUDA")
    parser.add_argument("--quiet", action="store_true", default=False, help="安静模式，减少输出")
    args = parser.parse_args()

    use_cuda = (not args.no_cuda) and torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)

    for m in args.models:
        safe_out = os.path.join(args.out_dir, f"{safe_name(m)}.pred.jsonl")
        start = time.time()
        if not args.quiet:
            print(f"[com_pred] 开始对模型 {m} 进行预测，输出 -> {safe_out}")
        process_model(
            model_path=m,
            test_path=args.test,
            out_path=safe_out,
            tokenizer_name=args.tokenizer,
            seqlen=args.seqlen,
            stride=args.stride,
            batch_size=args.batch_size,
            device=device,     
            verbose=(not args.quiet)
        )
        if not args.quiet:
            print(f"[com_pred] 模型 {m} 完成，耗时 {time.time() - start:.1f}s")
    
    if not args.quiet:
        print(f"[com_pred] 所有模型预测完成")

if __name__ == "__main__":
    main()

# 示例调用（同步）
# python com_pred.py -m models/per5_model.pth.best.pth -t DATAsets/sampled1xx_10000.jsonl -o OUTPUTs/com_pred
# python com_pred.py -m models/per5_model.pth.best.pth -t DATAsets/samples_2xx.jsonl -o OUTPUTs/neg