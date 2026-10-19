import os
import time
import json
import requests
import jsonlines
import numpy as np
import asyncio
import re
from typing import Dict, List, Tuple, Optional, Any
from tenacity import retry, stop_after_attempt, wait_exponential
from concurrent.futures import ThreadPoolExecutor, as_completed

# 导入LangChain核心组件
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain_core.callbacks import CallbackManager, FileCallbackHandler
from langchain.schema import LLMResult
from langchain.tools import Tool
from langchain.agents import AgentExecutor, create_react_agent
from langchain.llms.base import LLM
from langchain_core.runnables import RunnablePassthrough, RunnableSequence

# -------------------------- 1. 配置参数 --------------------------
API_KEY = "cbcb991a-3607-42a5-8e36-38ea9c40b191"  # 火山引擎API Key，DeepSeekV3.1
BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"  # 火山引擎
MODEL_NAME = "deepseek-v3-1-250821"  # r1太慢了

TEST_DATA_PATH = "OUTPUTs/final_llm_testset.jsonl"  # 测试集
RESULT_SAVE_PATH = "OUTPUTs/ma_sllm_combine.jsonl"
LOG_PATH = "LOGs/multiagent_10000.log"

# -------------------------- 2. 黑板类 --------------------------
class Blackboard:
    def __init__(self):
        self.content: Dict[str, Any] = {
            "current_result": None,
            "key_evidence": [],
            "warnings": [],
            "rationale": [],
            "search_results": []
        }

    def update(self, key: str, value: Any):
        if key in self.content:
            if isinstance(self.content[key], list):
                self.content[key].append(value)
            else:
                self.content[key] = value

    def get_content(self) -> str:
        return json.dumps(self.content, ensure_ascii=False)

# -------------------------- 3. 定义Agents的Prompts --------------------------
# Router Agent Prompt
router_prompt = PromptTemplate(
    template="""
    你是Router Agent，决策中心。
    输入：
    - current_result: {current_result} (初始为SLM输出，后续为上一轮结果)
    - blackboard_content: {blackboard_content} (黑板内容，压缩版)
    - raw_packet: {raw_packet} (报文片段)

    输出严格JSON格式：
    - decision: "reliable" 或 "unreliable"
    - final_result: (若reliable) 最终设备型号
    - next_agents: (若unreliable) 3位二进制字符串，如"101"表示调用Deep-Reasoner(1)和Searcher(1)，不调用Reflector(0)
    - agent_instructions: (若unreliable) 字典，键为agent名，值为指令字符串
    - keywords: (若unreliable) 关键词列表
    - warnings: (若unreliable) 警告列表

    示例输出：{{"decision": "unreliable", "next_agents": "110", "agent_instructions": {{"Deep-Reasoner": "深度推理型号", "Reflector": "验证品牌匹配"}}, "keywords": ["Hikvision"], "warnings": ["置信度低"]}}
    """,
    input_variables=["current_result", "blackboard_content", "raw_packet"]
)

# Reflection Agent Prompt
reflection_prompt = PromptTemplate(
    template="""
    你是Reflection Agent，快速校验员。
    输入：
    - instruction_from_router: {instruction_from_router}
    - syn_product: {syn_product}

    输出严格JSON格式：
    - verified_result: 验证或修正后的型号
    - rationale: 简要理由

    示例输出：{{"verified_result": "DS-2CD2142FWD-I", "rationale": "格式符合Hikvision规范"}}
    """,
    input_variables=["instruction_from_router", "syn_product"]
)

# Deep-Reasoner Agent Prompt
deep_reasoner_prompt = PromptTemplate(
    template="""
    你是Deep-Reasoner Agent，资深专家。
    输入：
    - instruction_from_router: {instruction_from_router}
    - key_fragments: {key_fragments}

    输出严格JSON格式：
    - reasoned_result: 推理后型号
    - rationale: 详细推理链

    示例输出：{{"reasoned_result": "AC10", "rationale": "结合Tenda和Ver:1.0推理出Tenda AC10"}}
    """,
    input_variables=["instruction_from_router", "key_fragments"]
)

# Searcher Agent Prompt
searcher_prompt = PromptTemplate(
    template="""
    你是Searcher Agent，知识库专员。从你已知知识中检索。
    输入：
    - instruction_from_router: {instruction_from_router}
    - search_query: {search_query}

    输出严格JSON格式：
    - search_results: 检索信息
    - rationale: 关键词和结果

    示例输出：{{"search_results": "TP-Link Archer AX100存在", "rationale": "关键词Archer AX100匹配"}}
    """,
    input_variables=["instruction_from_router", "search_query"]
)

# -------------------------- 4. 核心功能 --------------------------
def clean_and_parse_llm_output(output_str: str, default_json: Dict = None) -> Dict:
    # 复用原函数，略
    if default_json is None:
        default_json = {"decision": "unreliable", "warnings": ["PARSE_ERROR"]}
    
    try:
        if '<think>' in output_str:
            think_end = output_str.find('</think>')
            if think_end != -1:
                output_str = output_str[think_end + len('</think>'):].strip()
            else:
                output_str = output_str.split('<think>')[-1].strip()
        
        output_str = re.sub(r'^```json\s*', '', output_str)
        output_str = re.sub(r'```\s*$', '', output_str)
        
        json_match = re.search(r'\{.*\}', output_str, re.DOTALL)
        if json_match:
            output_str = json_match.group(0)
        else:
            return default_json
        
        return json.loads(output_str)
    except json.JSONDecodeError:
        return default_json
    except Exception:
        return default_json

def judge_correctness(model_output: str, ground_truth: str) -> Tuple[bool, str]:
    # 复用原函数
    def normalize(text: str) -> str:
        return text.lower().strip().replace("-", "").replace(" ", "")
    
    if model_output is None:
        model_output = "无"
    
    if normalize(model_output) == normalize(ground_truth):
        return True, f"匹配正确：{model_output}"
    return False, f"匹配错误：{model_output}（正确值：{ground_truth}）"

# -------------------------- 5. 多Agent流程 --------------------------
async def process_sample(sample: Dict, idx: int, file_lock: asyncio.Lock, writer) -> Optional[Dict]:
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
    
    router_chain = router_prompt | llm
    reflection_chain = reflection_prompt | llm
    deep_reasoner_chain = deep_reasoner_prompt | llm
    searcher_chain = searcher_prompt | llm
    
    sample_id = sample["id"]
    small_model_candidate = sample["extracted_models"][0][0] if sample.get("extracted_models") else "无"
    raw_packet = sample["text"]
    ground_truth = sample["model"]
    
    blackboard = Blackboard()
    blackboard.update("current_result", small_model_candidate)
    
    max_rounds = 2
    for round_num in range(max_rounds):
        print(f"样本 {idx} 轮 {round_num + 1}")
        
        # Router 判断
        router_input = {
            "current_result": blackboard.content["current_result"],
            "blackboard_content": blackboard.get_content(),
            "raw_packet": raw_packet[:500]  # 压缩
        }
        router_message = await router_chain.ainvoke(router_input)
        router_output = clean_and_parse_llm_output(router_message.content.strip())
        
        if router_output.get("decision") == "reliable":
            final_result = router_output.get("final_result", blackboard.content["current_result"])
            break
        
        # 调用Agents
        next_agents = router_output.get("next_agents", "000")
        instructions = router_output.get("agent_instructions", {})
        keywords = router_output.get("keywords", [])
        warnings = router_output.get("warnings", [])
        
        blackboard.update("warnings", warnings)
        
        # Deep-Reasoner (位0)
        if next_agents[0] == '1':
            key_fragments = [frag for frag in raw_packet.split() if any(kw in frag for kw in keywords)]
            dr_input = {
                "instruction_from_router": instructions.get("Deep-Reasoner", "深度推理"),
                "key_fragments": json.dumps(key_fragments)
            }
            dr_message = await deep_reasoner_chain.ainvoke(dr_input)
            dr_output = clean_and_parse_llm_output(dr_message.content.strip())
            blackboard.update("current_result", dr_output.get("reasoned_result"))
            blackboard.update("rationale", dr_output.get("rationale"))
        
        # Reflector (位1)
        if next_agents[1] == '1':
            ref_input = {
                "instruction_from_router": instructions.get("Reflector", "验证"),
                "syn_product": blackboard.content["current_result"]
            }
            ref_message = await reflection_chain.ainvoke(ref_input)
            ref_output = clean_and_parse_llm_output(ref_message.content.strip())
            blackboard.update("current_result", ref_output.get("verified_result"))
            blackboard.update("rationale", ref_output.get("rationale"))
        
        # Searcher (位2)
        if next_agents[2] == '1':
            search_query = " ".join(keywords)
            sr_input = {
                "instruction_from_router": instructions.get("Searcher", "检索"),
                "search_query": search_query
            }
            sr_message = await searcher_chain.ainvoke(sr_input)
            sr_output = clean_and_parse_llm_output(sr_message.content.strip())
            blackboard.update("search_results", sr_output.get("search_results"))
            blackboard.update("rationale", sr_output.get("rationale"))
    
    else:
        final_result = blackboard.content["current_result"]  # 两轮后默认
    
    is_correct, reason = judge_correctness(final_result, ground_truth)
    
    result = {
        "sample_id": sample_id,
        "final_result": final_result,
        "blackboard": blackboard.content,
        "evaluation": {"is_correct": is_correct, "reason": reason}
    }
    
    async with file_lock:
        writer.write(result)
    
    return result

async def run_multiagent_experiment() -> None:
    total_start_time = time.time()
    samples = list(jsonlines.open(TEST_DATA_PATH, "r"))
    
    file_lock = asyncio.Lock()
    writer = jsonlines.open(RESULT_SAVE_PATH, "a")
    
    semaphore = asyncio.Semaphore(30)
    async def limited_process(sample, idx):
        async with semaphore:
            await asyncio.sleep(1.0)
            return await process_sample(sample, idx, file_lock, writer)
    
    tasks = [limited_process(sample, idx) for idx, sample in enumerate(samples, 1)]
    results = await asyncio.gather(*tasks)
    
    total_time = round(time.time() - total_start_time, 2)
    print(f"实验完成，总时间：{total_time}s，结果保存至：{RESULT_SAVE_PATH}")

if __name__ == "__main__":
    asyncio.run(run_multiagent_experiment())