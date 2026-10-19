import os
import json
import csv
import re
import argparse
from collections import Counter

# 在此处设定输出文件标识（修改为你想要的字符串）
EVAL_TAG = "testset"  # <-- 把 "default" 改为标识(测试样本数)，例如 "1000"

def _norm(s):
    if s is None:
        return ""
    s = str(s).strip()
    return s  

def normalize(text: str) -> str:
    """规范化字符串：转为小写、去除空格和连字符（与 langref.py 的 judge_correctness 一致）"""
    return text.lower().strip().replace("-", "").replace(" ", "")

def iter_jsonl(path):
    """
    按行解析JSONL文件（每行一个JSON对象）。
    """
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue

def _extract_extracted_field(item):
    # 兼容多个可能的字段名
    extracted = item.get("extracted_models", None)
    if extracted is None:
        extracted = item.get("extracted_models", item.get("predicted", item.get("extractions", []))) or []

    # 规范化：支持多种格式
    # - 列表中可能为字符串、dict 或 [字符串, 置信度] 列表
    # - 单个 dict -> 提取其 "product" 字段（若存在）
    # - 单个字符串 -> 包装为列表
    if isinstance(extracted, dict):
        # 如果是单个 dict，优先取 product 字段
        if "product" in extracted:
            extracted = [extracted.get("product")]
        else:
            # 否则把整个 dict 转为字符串作为候选
            extracted = [json.dumps(extracted, ensure_ascii=False)]
    elif isinstance(extracted, list):
        norm_list = []
        for x in extracted:
            if isinstance(x, dict):
                # 优先取常见的字段名
                m = x.get("product") or x.get("text") or x.get("value") or x.get("name")
                if m is not None:
                    norm_list.append(m)
                else:
                    # 无明显字段时把 dict 序列化为字符串保留信息
                    norm_list.append(json.dumps(x, ensure_ascii=False))
            elif isinstance(x, list) and len(x) > 0:
                # 处理 [字符串, 置信度] 格式，取第一个元素作为字符串
                norm_list.append(str(x[0]).strip())
            else:
                norm_list.append(str(x).strip())
        extracted = norm_list
    else:
        # 其他类型（比如字符串/数字），转为单元素列表
        extracted = [str(extracted).strip()]

    # 去除空白并确保字符串类型
    extracted = [str(x).strip() for x in extracted if str(x).strip()]
    return extracted

def evaluate_extractions(items_source, save_dir="OUTPUTs", top_errors=20):
    """
    按行（JSONL）评估提取结果。
    评估逻辑（按你的需求）：

      - 每行代表一条样本（JSON 对象）
      - 若 extracted_models 中存在任一字符串与 product 字段完全相等（strip 后），则该样本计为召回命中 +1（仅对正样本，即product非空样本计算召回率）
      - 召回率 = 召回命中样本数 / 正样本数（product非空）
      - 精确度（precision）按提取项层面计算：
          correct_score = 提取列表中与 product 完全相等的字符串数量 * 置信度
          false_score = 提取列表中与 product 不相等的字符串数量 * 置信度
          precision = correct_score / (correct_score + false_score)
        这样若某样本既包含正确提取也包含错误提取，错误提取会降低 precision
      - 另外计算 exact_match_samples：提取列表非空且所有提取字符串都等于 product
      - 记录 extracted_models 为空 / 非空 的样本数
    保存：
      - per-sample JSONL -> eval_{EVAL_TAG}_per_sample.jsonl
      - metrics JSON -> eval_{EVAL_TAG}_metrics.json
      - summary CSV -> eval_{EVAL_TAG}_summary.csv
    返回：metrics, fp_examples, fn_examples, out_json, csv_path
    """
    os.makedirs(save_dir, exist_ok=True)
    per_sample_path = os.path.join(save_dir, f"eval_{EVAL_TAG}_per_sample.jsonl")
    out_json = os.path.join(save_dir, f"eval_{EVAL_TAG}_metrics.json")
    csv_path = os.path.join(save_dir, f"eval_{EVAL_TAG}_summary.csv")

    total_samples = 0
    positive_samples = 0  # 正样本数（product非空）
    recall_hit_samples = 0          # 样本级召回（任一提取等于 model，仅对正样本）
    correct_score = 0.0
    false_score = 0.0
    exact_match_samples = 0         # 提取非空且所有提取均等于 product
    extracted_empty_count = 0
    extracted_nonempty_count = 0

    fp_examples = []
    fn_examples = []

    # 支持传入路径或可迭代对象
    if isinstance(items_source, str) and os.path.exists(items_source):
        iterator = iter_jsonl(items_source)
    else:
        iterator = items_source

    with open(per_sample_path, 'w', encoding='utf-8') as per_out:
        for item in iterator:
            total_samples += 1
            product = str(item.get("product", "")).strip()
            extracted = item.get("extracted_models", [])
            # 兼容格式：extracted_models 可能是 [["xxx", 0.99], ...]
            norm_extracted = []
            for x in extracted:
                if isinstance(x, list) and len(x) > 1:
                    s, conf = str(x[0]).strip(), float(x[1])
                else:
                    s, conf = str(x).strip(), 1.0
                norm_extracted.append((s, conf))

            if len(norm_extracted) == 0:
                extracted_empty_count += 1
            else:
                extracted_nonempty_count += 1

            n_model = normalize(product)
            this_correct_score = sum(conf for s, conf in norm_extracted if normalize(s) == n_model and normalize(s) != "")
            this_false_score = sum(conf for s, conf in norm_extracted if (normalize(s) != n_model) and (normalize(s) != ""))

            correct_score += this_correct_score
            false_score += this_false_score

            # 样本级召回：仅对正样本（product非空）计算
            if product:
                positive_samples += 1
                if this_correct_score > 0:
                    recall_hit_samples += 1

            # exact match 样本：提取非空且全部等于 product
            if len(norm_extracted) > 0 and this_false_score == 0 and this_correct_score > 0:
                exact_match_samples += 1

            # 收集错误示例（便于人工检查）
            matched_sample = (this_correct_score > 0)
            if (not matched_sample) and len(fn_examples) < top_errors and str(item.get("sample_type","")).startswith("1xx"):
                fn_examples.append({
                    "id": item.get("id", total_samples-1),
                    "product": product,
                    "extracted": extracted,
                    "banner": item.get("banner","")[:400]
                })
            if (len(extracted) > 0 and (not matched_sample or (matched_sample and this_false_score > 0))) and len(fp_examples) < top_errors:
                fp_examples.append({
                    "id": item.get("id", total_samples-1),
                    "product": product,
                    "extracted": extracted,
                    "banner": item.get("banner","")[:400]
                })

            per_sample_result = {
                "id": item.get("id", total_samples-1),
                "product": product,
                "extracted": extracted,
                "sample_matched": bool(this_correct_score > 0),
                "this_correct_score": this_correct_score,
                "this_false_score": this_false_score
            }
            per_out.write(json.dumps(per_sample_result, ensure_ascii=False) + "\n")

    # 计算指标
    recall = (recall_hit_samples / positive_samples) if positive_samples > 0 else 0.0
    precision = (correct_score / (correct_score + false_score)) if (correct_score + false_score) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    exact_match_rate = (exact_match_samples / total_samples) if total_samples > 0 else 0.0

    metrics = {
        "total_samples": int(total_samples),
        "positive_samples": int(positive_samples),
        "recall_hit_samples": int(recall_hit_samples),
        "correct_strings": int(correct_score),
        "false_positive_strings": int(false_score),
        "extracted_empty_count": int(extracted_empty_count),
        "extracted_nonempty_count": int(extracted_nonempty_count),
        "exact_match_samples": int(exact_match_samples),
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "exact_match_rate": exact_match_rate
    }

    # 保存 JSON 报告（包含少量 FP/FN 示例）
    with open(out_json, "w", encoding="utf-8") as jf:
        json.dump({"metrics": metrics, "fp_examples": fp_examples, "fn_examples": fn_examples, "per_sample_path": per_sample_path}, jf, ensure_ascii=False, indent=2)

    # 保存 CSV 摘要
    with open(csv_path, "w", newline="", encoding="utf-8") as cf:
        writer = csv.writer(cf)
        writer.writerow(list(metrics.keys()))
        writer.writerow([metrics[k] for k in metrics.keys()])

    return metrics, fp_examples, fn_examples, out_json, csv_path

def main():
    parser = argparse.ArgumentParser(description="Evaluate extraction results against test set products.")
    parser.add_argument("--pred", "-p", type=str, default=None, help="预测结果 JSONL 文件路径（每行一个 JSON）")
    parser.add_argument("--save_dir", "-s", type=str, default=".", help="评估结果保存目录（默认当前目录）")
    parser.add_argument("--top_errors", "-t", type=int, default=20, help="保存的错误示例数量上限")
    args = parser.parse_args()

    if args.pred is None or not os.path.exists(args.pred):
        raise FileNotFoundError("请通过 --pred 指定预测结果 JSONL 文件路径（每行一条 JSON 对象）。")

    metrics, fp_examples, fn_examples, out_json, csv_path = evaluate_extractions(args.pred, save_dir=args.save_dir, top_errors=args.top_errors)

    print("Evaluation finished.")
    print(f"Samples: {metrics['total_samples']}")
    print(f"Positive samples: {metrics['positive_samples']}")
    print(f"Recall-hit samples: {metrics['recall_hit_samples']} / {metrics['positive_samples']} -> Recall: {metrics['recall']:.4f}")
    print(f"Correct strings: {metrics['correct_strings']}, False positive strings: {metrics['false_positive_strings']}")
    print(f"Precision (string-level): {metrics['precision']:.4f}, F1(using sample-recall & string-precision): {metrics['f1']:.4f}")
    print(f"Exact-match samples (all extracted == product): {metrics['exact_match_samples']} -> exact_match_rate: {metrics['exact_match_rate']:.4f}")
    print(f"Extracted empty: {metrics['extracted_empty_count']}, Extracted non-empty: {metrics['extracted_nonempty_count']}")
    print(f"Saved JSON -> {out_json}")
    print(f"Saved CSV  -> {csv_path}")
    if fp_examples:
        print(f"\n示例 FP (最多 {len(fp_examples)}):")
        for e in fp_examples[:args.top_errors]:
            print(e)
    if fn_examples:
        print(f"\n示例 FN (最多 {len(fn_examples)}):")
        for e in fn_examples[:args.top_errors]:
            print(e)

if __name__ == "__main__":
    main()
# 运行示例：python slm_evalue.py --pred ..\OUTPUTs\final_llm_testset.jsonl -s compare
"""注意改标记（避免覆盖之前文件）"""