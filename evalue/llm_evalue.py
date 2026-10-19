import os
import re
import json
import hashlib
import time
import jsonlines

# 导入LangChain组件
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

# 配置参数
MODEL_NAME = "deepseek-v3:671b"
API_KEY = "c98d40b746766d61a7c087b77def7b2be757ead6d64bee57b3cdb2bbe2b125da"  # 火山引擎API Key，DeepSeekV3.1
BASE_URL = "https://uni-api.cstcloud.cn/v1"

llm = ChatOpenAI(
    model_name=MODEL_NAME,
    openai_api_key=API_KEY,
    openai_api_base=BASE_URL,
    temperature=0.0,
    max_tokens=1024,
    max_retries=3
)

# 定义Prompt模板（用于判断型号是否相同）
PROMPT_TEMPLATE = """
你是设备型号匹配专家。输入两个字符串：model_pred 和 model_gt。判断它们是否表示相同型号
注意：
- 考虑厂商前缀（如 'dh-'、'hik-'）、大小写、分隔符、同一型号对外宣称不同等原因。
- 忽略版本尾缀（如 '-v2'、'mk ii'、'-i'）。
要求：仅返回 JSON：{{"same": true/false, "reason": "简短理由"}}，不要额外输出。

model_pred: "{pred}"
model_gt: "{gt}"

请据此返回 JSON。
"""

prompt = PromptTemplate(
    input_variables=["pred", "gt"],
    template=PROMPT_TEMPLATE
)

# 使用RunnableSequence替换deprecated的LLMChain
chain = prompt | llm

CACHE_FILE = os.path.join(os.path.dirname(__file__), "..", "LOGs", "llm_evalue_cache.json")

def load_cache():
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_cache(c):
    try:
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(c, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

def key_for_pair(pred: str, gt: str) -> str:
    return hashlib.sha256((pred + "\t" + gt).encode("utf-8")).hexdigest()

def call_llm(pred: str, gt: str, model: str = "deepseek-v3-1-250821", max_retries: int = 3):
    # 返回 (same: bool, reason: str)
    cache = load_cache()
    k = key_for_pair(pred, gt)
    if k in cache:
        return cache[k]["same"], cache[k].get("reason", "from_cache")
    try:
        # 使用RunnableSequence的invoke调用
        result = chain.invoke({"pred": pred, "gt": gt}).content
        content = result.strip()
        m = re.search(r"\{.*\}", content, flags=re.S)
        jtext = m.group(0) if m else content
        parsed = json.loads(jtext)
        same = bool(parsed.get("same", False))
        reason = parsed.get("reason", "") if isinstance(parsed.get("reason", ""), str) else ""
        cache[k] = {"pred": pred, "gt": gt, "same": same, "reason": reason}
        save_cache(cache)
        return same, reason
    except Exception as e:
        return False, f"llm_error:{e}"

def is_same_model(pred: str, gt: str, use_llm: bool = True, model: str = "deepseek-v3-1-250821"):
    # 直接调用大模型判断（去掉启发式）
    if not use_llm:
        return False, "llm_disabled"
    same, reason = call_llm(pred, gt, model=model)
    return bool(same), reason

def evaluate_jsonl(input_path: str, output_path: str, use_llm: bool = True):
    """
    读取JSONL文件，对每个样本调用LLM判断model_output与ground_truth是否相同，
    输出带llm_same和llm_reason的JSONL。
    """
    with jsonlines.open(input_path, "r") as reader, jsonlines.open(output_path, "w") as writer:
        sample_cnt = 0
        smp_rgt, smp_wrg = 0, 0
        for sample in reader:
            sample_cnt += 1
            pred = sample["model_output_info"]["model_output"]
            gt = sample["input_info"]["ground_truth"]
            same, reason = is_same_model(pred, gt, use_llm=use_llm)
            if same:
                smp_rgt += 1
                print("正在处理第{}条样本，型号正确".format(sample_cnt))
            else:
                smp_wrg += 1
                print("正在处理第{}条样本，型号错误".format(sample_cnt))
            sample["llm_same"] = same
            sample["llm_reason"] = reason
            writer.write(sample)
        print(f"处理完成，共{sample_cnt}条样本，正确{str(smp_rgt)}，错误{str(smp_wrg)}")

if __name__ == "__main__":
    # 示例用法
    OUTPUT_DIR = "OUTPUTs"
    INPUT_FILE = "llm_wrong_samples.jsonl"
    OUTPUT_FILE = "llm_evalues.jsonl"  # 更新为实际文件名（例如 "llm_results_reflective.jsonl"）
    
    INPUT_PATH = os.path.join(OUTPUT_DIR, INPUT_FILE)
    OUTPUT_PATH = os.path.join(OUTPUT_DIR, OUTPUT_FILE)

    evaluate_jsonl(INPUT_PATH, OUTPUT_PATH, use_llm=True)