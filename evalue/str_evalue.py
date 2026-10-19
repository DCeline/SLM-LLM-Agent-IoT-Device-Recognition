import jsonlines
from sklearn.metrics import accuracy_score

def normalize(text: str) -> str:
    """规范化字符串：转为小写、去除空格和连字符（与 langref.py 的 judge_correctness 一致）"""
    return text.lower().strip().replace("-", "").replace(" ", "")

def evaluate_lowconf_accuracy(result_path="OUTPUTs/sllm_combine.jsonl"):
    """检查大模型生成结果的准确率（仅统计使用了大模型的样本）"""
    l_preds, s_preds, trues = [], [], []
    with jsonlines.open(result_path, "r") as f:
        for sample in f:
            # 假设所有样本都使用了LLM，检查call_agents是否包含'1'（表示调用了Agent）
            if '1' in sample.get("call_agents", ""):
                l_preds.append(normalize(sample.get("llm_result", "")))
                s_preds.append(normalize(sample.get("extracted_models", {}).get("model", "")))
                trues.append(normalize(sample.get("product", "")))

    large_acc = accuracy_score(trues, l_preds)
    small_acc = accuracy_score(trues, s_preds)
    print("低置信度样本大模型准确率:", large_acc)
    print("低置信度样本小模型准确率:", small_acc)
    return large_acc, small_acc

def evaluate_all_samples(result_path="OUTPUTs/sllm_combine.jsonl"):
    """统计小模型、大小模型协同的输出准确率"""
    s_preds, ls_preds, trues = [], [], []
    with jsonlines.open(result_path, "r") as f:
        for sample in f:
            # 小模型结果
            s_preds.append(normalize(sample.get("extracted_models", {}).get("model", "")))
            # LLM结果
            ls_preds.append(normalize(sample.get("llm_result", "")))
            # 真实值
            trues.append(normalize(sample.get("product", "")))

    small_acc = accuracy_score(trues, s_preds)
    combined_acc = accuracy_score(trues, ls_preds)
    print("所有样本小模型准确率:", small_acc)
    print("所有样本大小模型协同准确率:", combined_acc)
    return small_acc, combined_acc

def evaluate_large_model_on_unextracted(result_path="OUTPUTs/sllm_combine.jsonl"):
    """比较小模型没有提取出来，但是大模型提取出来的型号的准确率"""
    l_preds, trues = [], []
    cnt = 0
    correct = 0
    with jsonlines.open(result_path, "r") as f:
        for sample in f:
            # 小模型未提取：extracted_models为空或model为空
            if not sample.get("extracted_models", {}).get("model", "").strip():
                cnt += 1
                l_pred = normalize(sample.get("llm_result", ""))
                l_preds.append(l_pred)
                true_sample = normalize(sample.get("product", ""))
                trues.append(true_sample)
                if l_pred == true_sample:
                    correct += 1

    acc = correct / cnt if cnt else 0
    print("小模型未提取样本数:", cnt)
    print("大模型正确数（小模型未提取样本）:", correct)
    print("大模型准确率（小模型未提取样本）:", acc)
    return cnt, correct, acc

if __name__ == "__main__":
    RESULT_FILENAME = "sllm_combine.jsonl"  # 更新为实际文件名
    RESULT_PATH = f"../OUTPUTs/{RESULT_FILENAME}"
    lowconf_l_acc, lowconf_s_acc = evaluate_lowconf_accuracy(RESULT_PATH)
    all_s_acc, combined_acc = evaluate_all_samples(RESULT_PATH)
    null_cnt, null_correct, null_acc = evaluate_large_model_on_unextracted(RESULT_PATH)

    # 保存评估结果到txt文件
    EVA_FILENAME = "slm_summary.txt"
    EVA_PATH = f"../OUTPUTs/{EVA_FILENAME}"
    with open(EVA_PATH, "w", encoding="utf-8") as f:
        f.write(f"低置信度样本大模型准确率: {lowconf_l_acc}\n")
        f.write(f"低置信度样本小模型准确率: {lowconf_s_acc}\n")
        f.write(f"所有样本小模型准确率: {all_s_acc}\n")
        f.write(f"所有样本大小模型协同准确率: {combined_acc}\n")
        f.write(f"小模型未提取样本数: {null_cnt}\n")
        f.write(f"大模型正确数（小模型未提取样本）: {null_correct}\n")
        f.write(f"大模型准确率（小模型未提取样本）: {null_acc}\n")