import os
import jsonlines
from sklearn.metrics import accuracy_score
import sys

# 导入llm_evalue和str_evalue的功能
try:
    from llm_evalue import is_same_model
    from str_evalue import normalize

except ImportError:
    # 如果import失败，手动定义函数
    def normalize(text: str) -> str:
        """规范化字符串：转为小写、去除空格和连字符"""
        return text.lower().strip().replace("-", "").replace(" ", "")

    def is_same_model(pred: str, gt: str, use_llm: bool = True):
        # 简化版，直接返回False（实际应调用llm_evalue）
        return False, "llm_not_available"

class DeviceEvaluer:
    def __init__(self, result_path: str):
        self.result_path = result_path
        self.samples = []
        self.load_samples()

    def load_samples(self):
        """加载JSONL样本"""
        with jsonlines.open(self.result_path, "r") as f:
            self.samples = list(f)

    def evaluate_sample(self, sample):
        """对单个样本进行评估：先字符串匹配，失败则LLM判断"""  
        pred = sample.get("llm_result", "")
        gt = sample.get("product", "")

        if not pred:
            return False, "no_llm_result"
    
        # 先字符串匹配
        if normalize(pred) == normalize(gt):
            print("模型识别型号正确（字符串匹配）")
            return True, "string_match"
        # 字符串匹配失败，调用LLM
        same, reason = is_same_model(pred, gt, use_llm=True)
        if same:
            print("模型识别型号正确（LLM判断）")
        else:
            print("模型识别型号错误，结果为:", pred, "正确型号为:", gt)
        return same, reason

    def evaluate_all(self):
        """评估所有样本，更新样本字典"""
        smp_cnt = 0
        for sample in self.samples:
            smp_cnt += 1
            print(f"\n评估第{smp_cnt}个样本:")
            same, reason = self.evaluate_sample(sample)
            sample["final_same"] = same
            sample["final_reason"] = reason

    def evaluate_lowconf_accuracy(self):
        """检查大模型生成结果的准确率（仅统计使用了大模型的样本）"""
        large_correct = 0
        small_correct = 0
        total_large = 0
        for sample in self.samples:
            if '1' in sample.get("call_agents", ""):  # 修改：使用call_agents判断是否使用了LLM
                total_large += 1
                if sample["final_same"]:
                    large_correct += 1
                # 小模型准确率：比较normalize(small_model_candidate) == normalize(gt)
                small_pred = sample.get("extracted_models", {}).get("model", "")
                if normalize(small_pred) == normalize(sample.get("product", "")):
                    small_correct += 1

        large_acc = large_correct / total_large if total_large else 0
        small_acc = small_correct / total_large if total_large else 0
        print("低置信度样本大模型准确率:", large_acc)
        print("低置信度样本小模型准确率:", small_acc)
        return large_acc, small_acc

    def evaluate_all_samples(self):
        """统计小模型、大小模型协同的输出准确率"""
        small_correct = 0
        combined_correct = 0
        total = len(self.samples)
        for sample in self.samples:
            # 小模型准确率：比较normalize(small_candidate) == normalize(gt)
            small_pred = sample.get("extracted_models", {}).get("model", "")
            if normalize(small_pred) == normalize(sample.get("product", "")):
                small_correct += 1
            # 大小模型协同准确率：使用final_same
            if sample["final_same"]:
                combined_correct += 1

        small_acc = small_correct / total if total else 0
        combined_acc = combined_correct / total if total else 0
        print("所有样本小模型准确率:", small_acc)
        print("所有样本大小模型协同准确率:", combined_acc)
        
        return small_acc, combined_acc

    def evaluate_large_model_on_unextracted(self):
        """比较小模型没有提取出来，但是大模型提取出来的型号的准确率"""
        cnt = 0
        correct = 0
        for sample in self.samples:
            small_pred = sample.get("extracted_models", {}).get("model", "")
            if not small_pred.strip():  # 小模型未提取
                cnt += 1
                # 使用final_same判断大模型是否正确
                if sample["final_same"]:
                    correct += 1

        acc = correct / cnt if cnt else 0
        print("小模型未提取样本数:", cnt)
        print("大模型正确数（小模型未提取样本）:", correct)
        print("大模型准确率（小模型未提取样本）:", acc)
        return cnt, correct, acc

    def save_results(self, output_path: str):
        """保存评估结果到JSONL"""
        with jsonlines.open(output_path, "w") as writer:
            for sample in self.samples:  # 修改：循环self.samples
                writer.write(sample)

    def run_evaluation(self, output_path: str = None):
        """运行完整评估"""
        self.evaluate_all()
        lowconf_l_acc, lowconf_s_acc = self.evaluate_lowconf_accuracy()
        all_s_acc, combined_acc = self.evaluate_all_samples()
        null_cnt, null_correct, null_acc = self.evaluate_large_model_on_unextracted()

        # 保存评估结果到txt文件
        EVA_FILENAME = "sllm_summary.txt"  # 修改文件名
        EVA_PATH = f"../LOGs/{EVA_FILENAME}"  # 修改路径为相对于项目根目录
        with open(EVA_PATH, "w", encoding="utf-8") as f:
            f.write(f"低置信度样本大模型准确率: {lowconf_l_acc}\n")
            f.write(f"低置信度样本小模型准确率: {lowconf_s_acc}\n")
            f.write(f"所有样本小模型准确率: {all_s_acc}\n")
            f.write(f"所有样本大小模型协同准确率: {combined_acc}\n")
            f.write(f"小模型未提取样本数: {null_cnt}\n")
            f.write(f"大模型正确数（小模型未提取样本）: {null_correct}\n")
            f.write(f"大模型准确率（小模型未提取样本）: {null_acc}\n")

        if output_path:
            self.save_results(output_path)

if __name__ == "__main__":
    RESULT_DIR = "../OUTPUTs"
    RESULT_FILENAME = "sllm_combine.jsonl"
    # RESULT_FILENAME = "tn_nt_res.jsonl"
    RESULT_PATH = f"{RESULT_DIR}/{RESULT_FILENAME}"

    OUTPUT_PATH = "evaluated_sllm_res.jsonl"

    evaluer = DeviceEvaluer(RESULT_PATH)
    evaluer.run_evaluation(OUTPUT_PATH)