import os
import time
import json
import requests
import jsonlines
import numpy as np
import asyncio  # 新增导入
import re  # 新增导入（如果未导入）
from typing import Dict, List, Tuple, Optional, Any
from tenacity import retry, stop_after_attempt, wait_exponential
from concurrent.futures import ThreadPoolExecutor, as_completed  # 可移除，如果全异步

# 导入LangChain核心组件
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain  
from langchain_core.callbacks import CallbackManager, FileCallbackHandler
from langchain.schema import LLMResult
from langchain.tools import Tool
from langchain.agents import AgentExecutor, create_react_agent
from langchain.llms.base import LLM  # 新增导入
from langchain_core.runnables import RunnablePassthrough, RunnableSequence  # 新增导入

# -------------------------- 1. 配置参数 --------------------------
API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise ValueError("环境变量 API_KEY 未设置")
BASE_URL = "https://ark.cn-beijing.volces.com/api/v3" # 火山引擎
# BASE_URL = "https://uni-api.cstcloud.cn/v1"  # 中国科技云的BASE_URL
MODEL_NAME = "deepseek-v3-1-250821" # r1太慢了

TEST_DATA_PATH = "DATASETs/h_smp_10000.jsonl"  # 与之前相同的测试集格式
RESULT_SAVE_PATH = "OUTPUTs/sllm_combine.jsonl"
LOG_PATH = "LOGs/comb_10000.log"  # LangChain日志文件（便于调试）

HONEY_FLAG = False  # 新增：蜜罐标志

# -------------------------- 2. 定义工具（模拟AI搜索） --------------------------
def simulate_ai_search(query):
    """
    模拟AI搜索工具：不实际联网，而是返回提示，让模型在推理中描述搜索过程。
    这依赖于模型的自带联网能力（如DeepSeek或Doubao的联网版本）。
    """
    return f"模拟搜索结果：基于模型内置知识，查询'{query}'的验证过程需在模型推理中描述。实际链接和结论由模型生成。"

tools = [
    Tool(name="AISearch", func=simulate_ai_search, description="模拟AI搜索验证型号真实性和相关性（依赖模型自带联网）")
]

# -------------------------- 3. 初始化LangChain组件 --------------------------
# 初步预测Prompt：生成初始设备识别结果
initial_prediction_prompt = PromptTemplate(
    template=""" 
    任务：从报文中提取设备型号信息。
    输入报文：{raw_packet}
    小模型候选：{small_model_candidate}（置信度：{small_model_conf}）
    输出为严格的 JSON 格式，字段包括type、vendor、model、confidence。不要添加任何额外文本。要求返回结果以{{开头
    输出示例：{{"type": "Network Camera", "vendor": "Hikvision", "model": "DS-2CD2142FWD-I", "confidence": 0.92}}
    """,
    input_variables=["raw_packet", "small_model_candidate", "small_model_conf"]
)

# 反思Prompt：输出模型推理状态和对最终识别结果预测的建议
reflection_prompt = PromptTemplate(
    template=""" 
    任务：对初步预测进行反思。对于置信度低的预测，尝试搜索已有知识验证型号真实性和相关性，并根据搜索结果调整预测。
    初步预测：{initial_prediction}
    报文：{raw_packet}
    输出为严格的 JSON 格式，内容包括status（success/unreliable）、warnings、key_evidence、recommendation。不要添加任何额外文本。要求返回结果以{{开头
    - 如果初步预测合理，设置为 success；不合理时设置为 unreliable。
    - 成功示例：{{"status": "success", "warnings": [], "key_evidence": ["UI: 48:5a:3f (Hikvision)"]}}
    - 失败示例：{{"status": "unreliable", "warnings": ["LOW_CONFIDENCE"], "key_evidence": ["Server: ABC Embedded Server"], "recommendation": "Perform deeper inspection"}}
    """,
    input_variables=["initial_prediction", "raw_packet"]
)

# 修改输出Prompt：基于反思，生成最终结果
final_output_prompt = PromptTemplate(
    template=""" 
    任务：基于反思结果，生成最终设备识别输出。即使反思status为unreliable，也必须尝试修正并输出一个预测结果（不能置为null，除非完全无法提取）。
    反思结果：{reflection_result}
    初步预测：{initial_prediction}
    - 如果反思status为success，直接使用初步预测。
    - 如果反思status为unreliable，基于反思的key_evidence和recommendation，尝试修正初步预测（例如，调整model或vendor），但必须输出一个结果，以最可靠、最常见的结果为主。如果无法修正，保留初步预测但降低confidence到0.5。
    输出为严格的 JSON 格式，字段为prediction（必须包含type、vendor、model）、confidence、reflection。不要添加任何额外文本。要求返回结果以{{开头。
    - 成功示例输出：{{"prediction": {{"type": "Network Camera", "vendor": "Hikvision", "model": "DS-2CD2142FWD-I"}}, "confidence": 0.92, "reflection": {{"status": "success", "warnings": [], "key_evidence": ["Server: App-webs/"]}}}}
    - 不可靠但修正示例输出：{{"prediction": {{"type": "Network Camera", "vendor": "Hikvision", "model": "DS-2CD2142FWD-I"}}, "confidence": 0.5, "reflection": {{"status": "unreliable", "warnings": ["LOW_CONFIDENCE"], "key_evidence": ["Server: ABC Embedded Server"], "recommendation": "Perform deeper inspection"}}}}
    - 完全无法提取时才输出null（罕见）：{{"prediction": null, "confidence": 0.0, "reflection": {{"status": "failed", "warnings": ["NO_EVIDENCE"], "key_evidence": [], "recommendation": "No identifiable device"}}}}
    """,
    input_variables=["reflection_result", "initial_prediction"]
)

# -------------------------- 4. 核心功能（复用部分逻辑，简化调用） --------------------------
def judge_correctness(model_output: str, ground_truth: str) -> Tuple[bool, str]:
    def normalize(text: str) -> str:
        return text.lower().strip().replace("-", "").replace(" ", "")
    
    # 修复：处理 model_output 为 None 的情况
    if model_output is None:
        model_output = "无"
    
    if normalize(model_output) == normalize(ground_truth):
        return True, f"匹配正确：{model_output}"
    return False, f"匹配错误：{model_output}（正确值：{ground_truth}）"

# -------------------------- 5. 实验主流程（反思版，异步） --------------------------
def clean_and_parse_llm_output(output_str: str, default_json: Dict = None) -> Dict:
    """
    通用函数：清洗 LLM 输出，提取并解析 JSON。
    - 处理 <think> 标签（DeepSeek R1）。
    - 清理 ```json 和 ``` 标记。
    - 提取 JSON 子串。
    - 解析为字典，如果失败返回默认值。
    """
    if default_json is None:
        default_json = {"status": "failed", "warnings": ["PARSE_ERROR"], "key_evidence": [], "recommendation": "Output parsing failed"}
    
    try:
        # 解析 DeepSeek R1 响应（提取 <think> 后的答案）
        if '<think>' in output_str:
            think_end = output_str.find('</think>')
            if think_end != -1:
                output_str = output_str[think_end + len('</think>'):].strip()
            else:
                output_str = output_str.split('<think>')[-1].strip()
        
        # 清理代码块标记
        output_str = re.sub(r'^```json\s*', '', output_str)
        output_str = re.sub(r'```\s*$', '', output_str)
        
        # 提取 JSON 子串（模仿 device_identify_by_llm.py 的正则）
        json_match = re.search(r'\{.*\}', output_str, re.DOTALL)
        if json_match:
            output_str = json_match.group(0)
        else:
            print(f"未找到 JSON 子串：{output_str[:200]}...")
            return default_json
        
        # 解析 JSON
        return json.loads(output_str)
    except json.JSONDecodeError as e:
        print(f"JSON 解析失败：{str(e)}，输出内容：{output_str[:200]}...")
        return default_json
    except Exception as e:
        print(f"清洗解析异常：{str(e)}")
        return default_json

async def process_sample(sample: Dict, idx: int, file_lock: asyncio.Lock, writer) -> Optional[Dict]:
    """处理单个样本的异步函数"""
    # 为每个样本创建独立的 LLM 和链（确保隔离）
    llm = ChatOpenAI(
        model_name=MODEL_NAME,
        temperature=0.0,
        max_tokens=65536,
        openai_api_key=API_KEY,
        base_url=BASE_URL,
        streaming=False,
        max_retries=5,
        request_timeout=300
    )
    #honeypot_chain = honeypot_prompt | llm  # 新增：蜜罐判断链
    initial_chain = initial_prediction_prompt | llm
    reflection_chain = reflection_prompt | llm
    final_chain = final_output_prompt | llm
    
    sample_id = sample["id"]
    if not sample.get("extracted_models"):
        small_model_candidate = "无"
        small_model_conf = "无"
    else:
        small_model_candidate = sample["extracted_models"][0][0]
        small_model_conf = float(sample["extracted_models"][0][1])
    
    raw_packet = sample["text"]
    ground_truth = sample["model"]
    
    print(f"\n处理样本 {idx}（ID：{sample_id}）")
    
    if HONEY_FLAG:
        # 如果是蜜罐，直接返回结果
        model_output = "无"
        model_confidence = 0.0
        call_time = 0.0  # 蜜罐判断耗时，可计算
        used_large_model = True
        search_results = honeypot_result.get("reason", [])
        reflection = {"status": "honeypot", "warnings": ["HONEYPOT_DETECTED"], "key_evidence": search_results, "recommendation": "Skip device identification"}
        print(f"检测到蜜罐：{honeypot_result.get('reason')}，跳过后续识别")
    else:
        # 如果不是蜜罐，继续原有流程
        if isinstance(small_model_conf, (int, float)) and small_model_conf >= 0.965:
            model_output = small_model_candidate
            model_confidence = small_model_conf
            call_time = 0.0
            used_large_model = False
            search_results = "无（未使用搜索）"
            reflection = {"status": "success", "warnings": [], "key_evidence": [], "alternative_predictions": []}
            print(f"使用小模型结果：{model_output}（置信度：{model_confidence}）")
        else:
            used_large_model = True
            start_time = time.time()
            try:
                # 新增：测试简单 API 调用（仅第一个样本）
                if idx == 1:
                    test_llm = ChatOpenAI(
                        model_name=MODEL_NAME,
                        temperature=0.0,
                        max_tokens=50,
                        openai_api_key=API_KEY,
                        base_url=BASE_URL,
                        max_retries=1,
                        request_timeout=30
                    )
                    test_response = test_llm.invoke("Hello, test message.")
                    print(f"API 测试响应：{test_response.content}")
                
                # 异步调用链
                initial_message = await initial_chain.ainvoke({
                    "raw_packet": raw_packet,
                    "small_model_candidate": small_model_candidate,
                    "small_model_conf": small_model_conf
                })
                initial_result = clean_and_parse_llm_output(initial_message.content.strip(), 
                                                       default_json={"type": "Unknown", "vendor": "Unknown", "model": "Unknown", "confidence": 0.0})
            
                reflection_message = await reflection_chain.ainvoke({
                    "initial_prediction": json.dumps(initial_result),  # 传入 JSON 字符串
                    "raw_packet": raw_packet
                })
                reflection_result = clean_and_parse_llm_output(reflection_message.content.strip(), 
                                                              default_json={"status": "unreliable", "warnings": ["PARSE_ERROR"], "key_evidence": [], "recommendation": "Reflection parse failed"})
            
                final_message = await final_chain.ainvoke({
                    "reflection_result": json.dumps(reflection_result),
                    "initial_prediction": json.dumps(initial_result)
                })
                final_output = clean_and_parse_llm_output(final_message.content.strip(), 
                                         default_json={"prediction": None, "confidence": 0.0, "reflection": {"status": "failed", "warnings": ["PARSE_ERROR"], "key_evidence": [], "recommendation": "Final parse failed"}})
    
                # 提取结果（逻辑不变）
                model_output = final_output.get("prediction", {}).get("model", "无") if final_output.get("prediction") else "无"
                model_confidence = final_output.get("confidence", 0.0)
                reflection = final_output.get("reflection", reflection_result)
                search_results = reflection.get("key_evidence", [])
                call_time = time.time() - start_time
                print(f"反思链输出：{model_output}（耗时：{call_time}s，反思状态：{reflection.get('status')}）")
            except json.JSONDecodeError as json_error:
                print(f"JSON解析失败：{str(json_error)}，输出内容：{final_message.content.strip()[:200]}...")  # 修复：使用 final_message.content.strip() 替换未定义的 final_output_str
                model_output = "无"
                model_confidence = 0.0
                reflection = {"status": "json_failed", "warnings": ["JSON_PARSE_ERROR"], "key_evidence": [], "recommendation": "Check LLM output format"}
                search_results = []
                call_time = time.time() - start_time
            except Exception as e:
                print(f"LLM 调用异常（样本 {idx}）：{str(e)}，类型：{type(e).__name__}")
                reflection_json = {"status": "call_failed", "warnings": ["LLM_CALL_ERROR"], "key_evidence": [], "recommendation": "Check API connection"}
                final_output = {"prediction": None, "confidence": 0.0, "reflection": {"status": "failed", "warnings": ["LLM_ERROR"], "key_evidence": [], "recommendation": "LLM call failed"}}
                model_output = "无"
                model_confidence = 0.0
                reflection = reflection_json
                search_results = []
                call_time = time.time() - start_time if 'start_time' in locals() else 0.0
                # 跳过后续处理，直接返回结果
    
    is_correct, reason = judge_correctness(model_output, ground_truth)
    
    result = {
        "sample_id": sample_id,
        "input_info": {
            "small_model_candidate": small_model_candidate,
            "raw_packet": raw_packet[:100] + "...",
            "ground_truth": ground_truth
        },
        "model_output_info": {
            "model_output": model_output,
            "confidence": model_confidence,
            "call_time": call_time,
            "used_large_model": used_large_model,
            "reflection": reflection,
            "search_results": search_results
        },
        "evaluation": {"is_correct": is_correct, "reason": reason}
    }
    
    # 新增：实时写入结果（使用锁确保并发安全）
    async with file_lock:
        writer.write(result)
    
    return result  # 可选：仍返回结果用于统计

async def run_reflective_experiment() -> None:
    total_start_time = time.time()
    samples = list(jsonlines.open(TEST_DATA_PATH, "r"))
    total_samples = len(samples)
    small_model_used = 0
    large_model_used = 0
    
    # 新增：创建文件锁和 writer（追加模式）
    file_lock = asyncio.Lock()
    writer = jsonlines.open(RESULT_SAVE_PATH, "a")  # 追加模式
    
    # 异步并发执行样本（限制并发数为30，避免API限制）
    semaphore = asyncio.Semaphore(30)
    async def limited_process(sample, idx):
        async with semaphore:
            await asyncio.sleep(1.0)  # 延迟1s
            return await process_sample(sample, idx, file_lock, writer)
    
    tasks = [limited_process(sample, idx) for idx, sample in enumerate(samples, 1)]
    results = await asyncio.gather(*tasks)
    
    for result in results:
        if result:
            if result.get("model_output_info", {}).get("used_large_model", False):
                large_model_used += 1
            else:
                small_model_used += 1
    
    total_prediction_time = round(time.time() - total_start_time, 2)
    print(f"\n实验统计：")
    print(f"- 总样本数：{total_samples}")
    print(f"- 使用小模型结果的样本数：{small_model_used}（占比：{small_model_used / total_samples * 100:.2f}%）")
    print(f"- 使用反思链识别的样本数：{large_model_used}（占比：{large_model_used / total_samples * 100:.2f}%）")
    print(f"- 总预测时间：{total_prediction_time}s")
    print(f"实验完成，结果保存至：{RESULT_SAVE_PATH}")

if __name__ == "__main__":
    asyncio.run(run_reflective_experiment())
