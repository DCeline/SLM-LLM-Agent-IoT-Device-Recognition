import json
import jsonlines
import os
import logging
import asyncio  # 新增导入
import sys
import time  # 新增导入，用于记录时间
from typing import Dict, List, Any, Optional
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain  # 确保导入

# -------------------------- 配置参数 --------------------------
API_KEY = "c98d40b746766d61a7c087b77def7b2be757ead6d64bee57b3cdb2bbe2b125da"  # 替换为中国科技云提供的Token
# BASE_URL = "https://ark.cn-beijing.volces.com/api/v3" # 火山引擎
BASE_URL = "https://uni-api.cstcloud.cn/v1"  # 中国科技云的BASE_URL
MODEL_NAME = "deepseek-v3:671b"
#MODEL_NAME = "deepseek-v3-1-250821"

# TEST_DATA_PATH = "OUTPUTs/final_llm_testset.jsonl"
TEST_DATA_PATH = "OUTPUTs/tn_nt_samples.jsonl"  
# RESULT_SAVE_PATH = "OUTPUTs/sllm_combine.jsonl"
RESULT_SAVE_PATH = "OUTPUTs/tn_nt_res.jsonl"
LOG_PATH = "LOGs/mas2.log"  # LangChain日志文件（便于调试）

# -------------------------- 辅助函数 --------------------------
def load_sllm_results(path: str) -> List[Dict[str, Any]]:
    """
    加载SLM结果文件，返回对齐的样本列表。
    只把banner、extracted_models和product传入后续流程。
    """   
    aligned_data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            banner = data.get('banner', '')  # 修改：使用get避免KeyError
            extracted_models = data.get('extracted_models', [])  # 修改：使用get
            product = data.get('product', '')
            sample_type = data.get('sample_type', '')  # 新增：加载sample_type
            aligned_data.append({
                'banner': banner,
                'extracted_models': extracted_models,
                'product': product,
                'sample_type': sample_type  # 新增：添加到字典中
            })
    return aligned_data

def clean_and_parse_llm_output(output_str: str, default_json: Dict = None) -> Dict:
    """
    通用函数：清洗 LLM 输出，提取并解析 JSON。
    - 处理 <think> 标签（DeepSeek R1）。
    - 清理 ```json 和 ``` 标记。
    - 提取 JSON 子串。
    - 解析为字典，如果失败返回默认值。
    """
    import re
    if default_json is None:
        default_json = {"status": "failed", "warnings": ["PARSE_ERROR"], "recommendation": "Output parsing failed"}
    
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
        
        # 提取 JSON 子串
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

# -------------------------- 1. 初始化 LLM -------------------------- 
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

# -------------------------- 2. 定义 4 个 PromptTemplate -------------------------- 
# Router Prompt（评估三维标签）
router_prompt = PromptTemplate(
    template="""
    任务：从原始报文中提取设备型号信息。
    输入信息：
    - 原始报文：{raw_banner}
    - SLM识别结果：{slm_result}
    - 当前推理状态：{current_state}
    - 已有信息：{existing_info}

    处理原则：
    1. 自身输出与SLM结果一致，那么结果置信度很高，尽可能直接输出。
    2. 自身输出与SLM结果不一致时，评估SLM结果。

    对SLM结果的评估维度与触发规则：
    1. 格式合规（Perception）：当SLM结果格式明显错误（如型号包含非法字符或不匹配banner结构）时，输出fail。
    2. 知识回忆一致性（Recollection）：认为需要搜索具体设备信息，输出fail。
    3. 逻辑推理自洽性（Reasoning）：和banner上下文存在矛盾。如果SLM结果为空，但banner中有明显的型号字符串，输出fail以触发DeepReasoner提取。

    输出为严格的 JSON，必须以{{开头，不要添加任何额外文本。json字段包括：
    - eva_tag：对象，包含perception/recollection/reasoning，每项为 "pass" 或 "fail"。
    - reason：触发该动作的简要说明。
    - key_evidence：列表，列出支持评估的关键证据片段（如banner中的匹配字符串）。

    输出示例：
    - 需要反思（perception fail）：
    {{
        "eva_tag": {{"perception": "fail", "recollection": "pass", "reasoning": "pass"}}, 
        "reason": "型号格式明显错误", 
        "key_evidence": ["banner中型号: 'ipc-hdw1230c-a'", "SLM结果格式错误"] 
    }}
    - 潜在漏判（reasoning fail）：
    {{
        "eva_tag": {{"perception": "pass", "recollection": "pass", "reasoning": "fail"}},
        "reason": "SLM结果为空，但banner中有型号字符串",
        "key_evidence": ["banner中型号: 'p1447-le'", "SLM结果为空"]
    }}  
    - 全部通过：
    {{
        "eva_tag": {{"perception": "pass", "recollection": "pass", "reasoning": "pass"}},
        "reason": "所有维度均通过",
        "key_evidence": []
    }}
    """,
    input_variables=["raw_banner", "slm_result", "current_state", "existing_info"]
)

# Reflector Prompt
reflector_prompt = PromptTemplate(
    template="""
    输入信息：
    - 当前结果：{current_result}
    - 已有信息：{existing_info}
    任务：聚焦设备型号的 “外在结构特征识别”。评估是否符合设备型号典型外在结构（字母-数字组合、品牌前缀匹配、版本号 / 规格参数完整性），
    是否具备设备型号的显性特征标识（如 “Model:”“Ver:” 后的结构化字符串）。

    输出为严格的 JSON 格式，字段包括：
    - verified_result：对象，包含 model、confidence。
    - add_evidence：列表，给出简要验证理由、返回补充的证据片段。
    不要添加任何额外文本。要求返回结果以{{开头。

    输出示例：
    {{
        "verified_result": {{"model":"DS-2CD2142FWD-I","confidence":0.65}},
        "add_evidence": ["型号格式符合banner中的字符串，仅修正大小写"]
    }}
    {{
        "verified_result": {{"model":"","confidence":0.0}},
        "add_evidence": ["banner中无型号字符串，返回空"]
    }}
    """,
    input_variables=["current_result", "existing_info"]
)

# Searcher Prompt
searcher_prompt = PromptTemplate(
    template="""
    任务：尝试搜索已有知识验证型号真实性和相关性，并根据搜索结果调整预测。
    输入信息：
    - 待验证内容：{target}
    - 已有信息：{existing_info}

    **重要：如果banner中无型号字符串，不要搜索，返回空。**

    输出为严格的 JSON 格式，字段包括：
    - search_results：结果列表，每条包含 add_info(type or vendor)、model、source（KB/条目）、confidence（匹配置信度或相似度）。
    - summary：对search_results的简要总结，只保留高置信度信息。
    不要添加任何额外文本。要求返回结果以{{开头。

    输出示例：
    {{
        "search_results": [  
            {{"add_info":"Network Camera; vendor Hikvision","model":"DS-2CD2142FWD-I","source":"Hikvision官网","confidence":0.95}},
            {{"add_info":"Network Camera","model":"DS-2CD2142FWD-I","source":"banner中的型号字段","confidence":0.95}}
        ],
        "summary": "banner中找到高置信匹配"
    }}
    """,
    input_variables=["target", "existing_info"]
)

# DeepReasoner Prompt
deep_reasoner_prompt = PromptTemplate(
    template="""
    任务：对SLM结果进行反思验证，基于banner生成最终设备型号识别结果。
    输入信息：
    - 当前结果：{current_result} 
    - 已有信息：{existing_info}
    - 原始报文：{raw_banner}

    **反思步骤**：
    1. 检查banner中是否有明显的设备型号字符串（如字母-数字组合）。
    2. 如果有，提取核心型号：忽略厂商前缀（如 'dh-'、'hik-'）和版本尾缀（如 '-v2'、'mk ii'），匹配核心字母-数字组合。
    3. 如果SLM结果为空，尝试从banner提取型号；如果不匹配，修正。
    4. 如果banner中无型号字符串，返回空结果（model: ""）。

    输出为严格的 JSON 格式，字段包括：
    - reasoned_result：推理后得出的型号model、confidence。
    - rationale：列表，分步说明验证依据。
    不要添加任何额外文本。要求返回结果以{{开头。

    输出示例：
    {{
        "reasoned_result": {{"model":"DS-2CD2142FWD-I","confidence":0.998}},
        "rationale": [
            "1. banner中有型号字符串匹配SLM结果；",
            "2. 保持SLM结果。"
        ]
    }}
    - 如果无型号：
    {{
        "reasoned_result": {{"model":"","confidence":0.0}},
        "rationale": [
            "1. banner中无型号字符串；",
            "2. 返回空结果。"
        ]
    }}
    """,
    input_variables=["current_result", "existing_info", "raw_banner"]
)

# -------------------------- 3. 核心 MAS 流程函数 -------------------------- 
async def run_mas_for_sample(sample: Dict[str, Any], idx: int, round_limit: int = 1, enable_log: bool = False) -> Dict[str, Any]:
    """
    对单个样本运行 MAS 流程，支持多轮迭代：
    1. Router 评估当前结果。
    2. 如果有fail维度，调用对应 Agent 并更新结果。
    3. 重复直到轮次达到 round_limit 或所有pass。
    4. 返回最终决策。
    """
    start_time = time.time()  # 记录开始时间
    
    raw_banner = sample.get("banner", "")
    extracted_models = sample.get("extracted_models", [])
    if extracted_models:
        slm_result = {"model": extracted_models[0][0], "confidence": extracted_models[0][1]}
    else:
        slm_result = {}
    blackboard: Dict[str, Dict[str, Any]] = {}  # 黑板用于agents间信息共享，键为轮次（如"round_1"），值包含"router"评估和"agents"输出，便于多轮迭代中传递信息
    current_result = slm_result  # 当前结果初始化为SLM结果

    router_chain = LLMChain(llm=llm, prompt=router_prompt)
    reflector_chain = LLMChain(llm=llm, prompt=reflector_prompt)
    searcher_chain = LLMChain(llm=llm, prompt=searcher_prompt)
    deep_reasoner_chain = LLMChain(llm=llm, prompt=deep_reasoner_prompt)

    for round_num in range(1, round_limit + 1):
        existing_info = blackboard  # 每轮开始时，existing_info 为当前 blackboard
        
        current_state = f"Round {round_num}" # 控制轮次
        
        # 步骤 1: 调用 Router Prompt（改为异步）
        router_input = {
            "raw_banner": raw_banner[:512],
            "slm_result": json.dumps(current_result, ensure_ascii=False),
            "current_state": current_state,
            "existing_info": json.dumps(existing_info, ensure_ascii=False)
        }
        router_message = await router_chain.ainvoke(router_input)  # 使用 ainvoke
        router_output = clean_and_parse_llm_output(router_message.get('text', '').strip(), default_json={"eva_tag": {"perception": "pass", "recollection": "pass", "reasoning": "pass"}, "reason": "", "key_evidence": []})

        if enable_log:
            logging.info(f"Sample {idx} Round {round_num}: Router Input: {json.dumps(router_input, ensure_ascii=False)}")
            logging.info(f"Sample {idx} Round {round_num}: Router Output: {json.dumps(router_output, ensure_ascii=False)}")

        eva_tag = router_output.get("eva_tag", {})
        # 设置call_agents
        call_agents = "000"
        if eva_tag.get("perception") == "fail":
            call_agents = "1" + call_agents[1:]
        if eva_tag.get("recollection") == "fail":
            call_agents = call_agents[0] + "1" + call_agents[2]
        if eva_tag.get("reasoning") == "fail":
            call_agents = call_agents[:2] + "1"
        
        blackboard[f"round_{round_num}"] = {
            "router": {
                "reason": router_output.get("reason", ""),
                "key_evidence": router_output.get("key_evidence", [])
            }
        }

        # 检查是否所有pass
        if all(tag == "pass" for tag in eva_tag.values()):
            break  # 如果所有pass，停止迭代

        # 步骤 2: 根据fail维度调用 Agent（改为异步）
        agent_outputs = {}

        # 更新 existing_info 包含当前 blackboard
        existing_info = blackboard
        if eva_tag.get("perception") == "fail":
            reflector_input = {
                "current_result": json.dumps(current_result, ensure_ascii=False),
                "existing_info": json.dumps(existing_info, ensure_ascii=False)  # 添加existing_info
            }
            reflector_message = await reflector_chain.ainvoke(reflector_input)
            agent_outputs["reflector"] = clean_and_parse_llm_output(reflector_message.get('text', '').strip(), default_json={"verified_result": {}, "add_evidence": []})

            # verified_result和add_evidence加入blackboard
            blackboard[f"round_{round_num}"] = {
                "reflector": {
                    "add_evidence": agent_outputs["reflector"].get("add_evidence", [])
                }
            }
            existing_info = blackboard  # 更新 existing_info 包含 reflector 输出

            if enable_log:
                logging.info(f"Sample {idx} Round {round_num}: Reflector Input: {json.dumps(reflector_input, ensure_ascii=False)}")
                logging.info(f"Sample {idx} Round {round_num}: Reflector Output: {json.dumps(agent_outputs['reflector'], ensure_ascii=False)}")

        if eva_tag.get("recollection") == "fail":
            searcher_input = {
                "target": json.dumps(current_result, ensure_ascii=False),
                #"search_goal": router_output.get("reason", ""),
                "existing_info": json.dumps(existing_info, ensure_ascii=False)  # 使用 blackboard 作为已有信息
            }
            searcher_message = await searcher_chain.ainvoke(searcher_input)
            agent_outputs["searcher"] = clean_and_parse_llm_output(searcher_message.get('text', '').strip(), default_json={"search_queries": [], "search_results": [], "summary": ""})

            # search_results和summary加入blackboard
            blackboard[f"round_{round_num}"]["searcher"] = {
                "summary": agent_outputs["searcher"].get("summary", "")
            }
            existing_info = blackboard  # 更新 existing_info 包含 searcher 输出

            if enable_log:
                logging.info(f"Sample {idx} Round {round_num}: Searcher Input: {json.dumps(searcher_input, ensure_ascii=False)}")
                logging.info(f"Sample {idx} Round {round_num}: Searcher Output: {json.dumps(agent_outputs['searcher'], ensure_ascii=False)}")

        if eva_tag.get("reasoning") == "fail":
            deep_reasoner_input = {
                "current_result": json.dumps(current_result, ensure_ascii=False),  # 修正键名为 current_result
                "existing_info": json.dumps(existing_info, ensure_ascii=False),  # 使用 blackboard 作为已有信息
                "raw_banner": raw_banner[:512]
            }
            deep_reasoner_message = await deep_reasoner_chain.ainvoke(deep_reasoner_input)
            agent_outputs["deep_reasoner"] = clean_and_parse_llm_output(deep_reasoner_message.get('text', '').strip(), default_json={"reasoned_result": {}, "rationale": []})

            # reasoned_result和rationale加入blackboard
            blackboard[f"round_{round_num}"]["deep_reasoner"] = {
                "rationale": agent_outputs["deep_reasoner"].get("rationale", [])
            }
            existing_info = blackboard  # 更新 existing_info 包含 deep_reasoner 输出

            if enable_log:
                logging.info(f"Sample {idx} Round {round_num}: DeepReasoner Input: {json.dumps(deep_reasoner_input, ensure_ascii=False)}")
                logging.info(f"Sample {idx} Round {round_num}: DeepReasoner Output: {json.dumps(agent_outputs['deep_reasoner'], ensure_ascii=False)}")

        # 步骤 3: 更新当前结果（优先 DeepReasoner，其次 Reflector/Searcher）
        if "deep_reasoner" in agent_outputs and agent_outputs["deep_reasoner"].get("reasoned_result"):
            current_result = agent_outputs["deep_reasoner"]["reasoned_result"]
        elif "reflector" in agent_outputs and agent_outputs["reflector"].get("verified_result"):
            current_result = agent_outputs["reflector"]["verified_result"]
        elif "searcher" in agent_outputs and agent_outputs["searcher"].get("search_results"):
            sr = agent_outputs["searcher"]["search_results"]
            if sr:
                current_result = sr[0]

    # 最终结果为最后更新的current_result
    final_result = current_result

    end_time = time.time()  # 记录结束时间
    processing_time = end_time - start_time  # 计算处理时间（秒）
    
    # 构建输出：只保留指定字段
    mas_output = {
        "banner": sample.get("banner", ""),
        "product": sample.get("product", ""),
        "extracted_models": slm_result,
        "llm_result": final_result.get("model", ""),
        "call_agents": call_agents,
        "processing_time": processing_time,  # 添加处理时间
        "sample_type": sample.get("sample_type", "")  # 新增：保存sample_type字段
    }
    return mas_output

# -------------------------- 4. 主流程 -------------------------- 
async def main_async():
    start_total = time.time()  # 记录整个脚本开始时间
    
    # 设置日志
    logging.basicConfig(filename=LOG_PATH, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # 加载 SLM 结果
    sllm_path = TEST_DATA_PATH
    if not os.path.exists(sllm_path):
        print(f"SLM 结果文件不存在：{sllm_path}")
        return

    samples = load_sllm_results(sllm_path)
    round_limit = 1  # 默认轮次限制，可手动调整

    # 保存 MAS 结果（实时写入）
    output_path = RESULT_SAVE_PATH
    lock = asyncio.Lock()  # 用于线程安全的写入和时间累加
    total_time = 0.0  # 初始化总处理时间
    with jsonlines.open(output_path, "w") as writer:
        # 并发处理样本（限制并发数为30）
        semaphore = asyncio.Semaphore(30)
        async def limited_process(sample, idx):
            nonlocal total_time
            async with semaphore:
                await asyncio.sleep(1.0)  # 延迟1s，避免API过载
                enable_log = (idx == 1)  # 只对第一个样本启用详细日志
                result = await run_mas_for_sample(sample, idx, round_limit, enable_log)
                async with lock:
                    writer.write(result)
                    total_time += result.get("processing_time", 0.0)  # 累加总时间

        tasks = [limited_process(sample, idx) for idx, sample in enumerate(samples, 1)]
        await asyncio.gather(*tasks)

    end_total = time.time()  # 记录整个脚本结束时间
    total_execution_time = end_total - start_total  # 计算总执行时间
    
    print(f"MAS 流程完成，结果保存至：{output_path}，总处理时间：{total_time:.2f} 秒，总执行时间：{total_execution_time:.2f} 秒")

def process_banner(banner):
    # Placeholder for original banner processing logic
    # In the original script, this might extract models from banner
    # Now, we skip this and use SLM results directly
    return None  # Not used since we use extracted_models

def main():
    asyncio.run(main_async())

if __name__ == "__main__":
    main()