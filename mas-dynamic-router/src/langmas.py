import json
import jsonlines
import os
from typing import Dict, List, Any, Optional
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain

# 导入配置和工具函数
from langref import API_KEY, BASE_URL, MODEL_NAME, clean_and_parse_llm_output, load_sllm_results

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
    任务：基于三维评估（格式合规感知 / 知识回忆一致性 / 推理自洽性）对 SLM 的识别结果进行结构化判断。
    输入信息：
    - 原始报文：{raw_packet}
    - SLM结果：{slm_result}
    - 当前推理状态：{current_state}
    - 已有信息：{existing_info}

    评估维度与触发规则（Router 内部规则）：
    1. 格式合规感知（Perception）：检测型号外在结构（字母-数字组合、品牌前缀、Model:/Ver: 标识等）。输出合格/不合格。
    2. 知识回忆一致性（Recollection）：核验与设备知识库/品牌型号体系匹配。输出合格/不合格。
    3. 推理驱动自洽性（Reasoning）：检查识别结果与报文协议/参数/应用场景逻辑一致性。输出合格/不合格。

    输出为严格的 JSON，必须以{{开头。字段包括：
    - eva_tag：对象，包含perception/recollection/reasoning，每项为 "合格" 或 "不合格"。
    - reason：触发该动作的简要说明（中文）。
    - key_evidence：列表，列出支持评估的关键证据片段（如报文字段、匹配条目等）。

    不要添加任何额外文本。要求返回结果以{{开头。

    输出示例：
    - 需要搜索（回忆不一致）：
    {{
        "eva_tag": {{"perception": "合格", "recollection": "不合格", "reasoning": "合格"}},
        "reason": "识别结果与已知品牌型号体系不匹配或在知识库中未命中",
        "key_evidence": ["型号 'ABC-1234' 未在设备知识库中检索到", "存在相似前缀但编号不匹配"]
    }}
    - 需要反思（格式不合规）：
    {{
        "eva_tag": {{"perception": "不合格", "recollection": "合格", "reasoning": "合格"}},
        "reason": "型号格式异常，缺失标准标识（如 Model: / Ver:）或命名不符合厂商习惯",
        "key_evidence": ["报文字段 'DeviceModel: ??'", "缺少品牌前缀"]
    }}
    - 需要深度推理（推理不自洽或复杂场景）：
    {{
        "eva_tag": {{"perception": "合格", "recollection": "合格", "reasoning": "不合格"}},
        "reason": "识别结果与报文协议/参数存在逻辑矛盾或需要多源融合验证",
        "key_evidence": ["报文协议为工业协议但型号属消费级系列"]
    }}
    """,
    input_variables=["raw_packet", "slm_result", "current_state", "existing_info"]
)

# Reflector Prompt
reflector_prompt = PromptTemplate(
    template="""
    任务：根据 Router 给出的建议，快速验证并（如有必要）修正型号格式或小范围拼写/结构问题（低成本校验）。
    输入信息：
    - 当前结果：{current_result}  （结构化对象或字符串）
    - router 的说明：{reason}
    - 关键证据：{key_evidence}

    输出为严格的 JSON 格式，字段包括：
    - verified_result：对象，包含 type、vendor、model、confidence。
    - rationale：列表，给出简要验证理由与证据引用。
    - add_evidence：列表，返回补充的证据片段（例如从报文中提取的字段）。
    不要添加任何额外文本。要求返回结果以{{开头。

    输出示例：
    {{
        "verified_result": {{"type":"Network Camera","vendor":"Hikvision","model":"DS-2CD2142FWD-I","confidence":0.65}},
        "rationale": ["型号格式符合厂商惯例，修正了少量拼写错误"],
        "add_evidence": ["在报文中找到字段 'Model: DS-2CD2142FWD-I'"]
    }}
    """,
    input_variables=["current_result", "reason", "key_evidence"]
)

# Searcher Prompt
searcher_prompt = PromptTemplate(
    template="""
    任务：根据搜索目标，搜索内部知识并检查该型号是否存在。
    输入信息：
    - 待验证内容：{target}（如设备型号、厂商信息等）
    - 搜索目标说明：{search_goal}
    - 可使用的上下文片段：{context_snippet}

    输出为严格的 JSON 格式，字段包括：
    - search_queries：生成的检索关键词列表（用于审计）。
    - search_results：结果列表，每条包含 type、vendor、model、source（KB/条目）、score（匹配置信度或相似度）。
    - summary：对检索结果的简要总结与建议（是否存在高置信匹配）。
    不要添加任何额外文本。要求返回结果以{{开头。

    输出示例：
    {{
        "search_queries": ["Hikvision DS-2CD2142FWD-I 官方参数", "DS-2CD2142FWD-I 报文特征"],
        "search_results": [
            {{"type":"Network Camera","vendor":"Hikvision","model":"DS-2CD2142FWD-I","source":"Hikvision官网","score":0.95}},
            {{"type":"Network Camera","vendor":"设备特征库","model":"DS-2CD2142FWD-I","source":"内部KB","score":0.88}}
        ],
        "summary": "检索确认 DS-2CD2142FWD-I 为海康威视网络摄像机，存在高置信匹配"
    }}
    """,
    input_variables=["target", "search_goal", "context_snippet"]
)

# DeepReasoner Prompt
deep_reasoner_prompt = PromptTemplate(
    template="""
    任务：基于提示和banner和设备型号，进行深度推理，生成设备型号识别结果。
    输入信息：
    - 初始结果：{initial_result}
    - 已有信息：{existing_info}（可能包含 searcher/reflector 的输出）
    - 原始报文（片段）：{raw_packet}

    输出为严格的 JSON 格式，字段包括：
    - reasoned_result：对象，包含 type、vendor、model、confidence（高权重可信度）。
    - rationale：列表，分步说明推理链与依据。
    - notes：可选备注（如不确定项或需人工介入提示）。
    不要添加任何额外文本。要求返回结果以{{开头。

    输出示例：
    {{
        "reasoned_result": {{"type":"Network Camera","vendor":"Hikvision","model":"DS-2CD2142FWD-I","confidence":0.92}},
        "rationale": [
            "1. 初始预测为 DS-2CD2142FWD-I；",
            "2. Searcher 输出与官网条目高度匹配；",
            "3. 报文包含 RTSP/ONVIF 特征，与摄像机场景一致；因此提高置信度。"
        ],
        "notes": "建议直接采用该结论作为最终结果"
    }}
    """,
    input_variables=["initial_result", "existing_info", "raw_packet"]
)

# -------------------------- 3. 核心 MAS 流程函数 -------------------------- 
def run_mas_for_sample(sample: Dict[str, Any], idx: int) -> Dict[str, Any]:
    """
    对单个样本运行 MAS 流程：
    1. Router 评估三维标签。
    2. 根据不合格维度调用对应 Agent。
    3. 合并结果，返回最终决策。
    """
    raw_packet = sample.get("text", "")
    slm_result = sample.get("model_output_info", {}).get("model_output", {})
    current_state = "初始评估"
    existing_info = {}  # 黑板初始为空

    # 步骤 1: 调用 Router Prompt
    router_chain = LLMChain(llm=llm, prompt=router_prompt)
    router_input = {
        "raw_packet": raw_packet[:512],  # 截断避免过长
        "slm_result": json.dumps(slm_result, ensure_ascii=False),
        "current_state": current_state,
        "existing_info": json.dumps(existing_info, ensure_ascii=False)
    }
    router_raw = router_chain.run(router_input)
    router_output = clean_and_parse_llm_output(router_raw, default_json={"eva_tag": {"perception": "合格", "recollection": "合格", "reasoning": "合格"}, "reason": "", "key_evidence": []})

    eva_tag = router_output.get("eva_tag", {})
    blackboard = {"router_output": router_output, "agent_outputs": []}

    # 步骤 2: 根据不合格维度调用 Agent
    agent_outputs = {}
    if eva_tag.get("perception") == "不合格":
        reflector_chain = LLMChain(llm=llm, prompt=reflector_prompt)
        reflector_input = {
            "current_result": json.dumps(slm_result, ensure_ascii=False),
            "reason": router_output.get("reason", ""),
            "key_evidence": json.dumps(router_output.get("key_evidence", []), ensure_ascii=False)
        }
        reflector_raw = reflector_chain.run(reflector_input)
        agent_outputs["reflector"] = clean_and_parse_llm_output(reflector_raw, default_json={"verified_result": {}, "rationale": [], "add_evidence": []})

    if eva_tag.get("recollection") == "不合格":
        searcher_chain = LLMChain(llm=llm, prompt=searcher_prompt)
        searcher_input = {
            "target": json.dumps(slm_result, ensure_ascii=False),
            "search_goal": router_output.get("reason", ""),
            "context_snippet": raw_packet[:512]
        }
        searcher_raw = searcher_chain.run(searcher_input)
        agent_outputs["searcher"] = clean_and_parse_llm_output(searcher_raw, default_json={"search_queries": [], "search_results": [], "summary": ""})

    if eva_tag.get("reasoning") == "不合格":
        deep_reasoner_chain = LLMChain(llm=llm, prompt=deep_reasoner_prompt)
        deep_reasoner_input = {
            "initial_result": json.dumps(slm_result, ensure_ascii=False),
            "existing_info": json.dumps(agent_outputs, ensure_ascii=False),
            "raw_packet": raw_packet[:512]
        }
        deep_reasoner_raw = deep_reasoner_chain.run(deep_reasoner_input)
        agent_outputs["deep_reasoner"] = clean_and_parse_llm_output(deep_reasoner_raw, default_json={"reasoned_result": {}, "rationale": [], "notes": ""})

    blackboard["agent_outputs"] = agent_outputs

    # 步骤 3: 合并最终结果（优先 DeepReasoner，其次 Reflector/Searcher，最后 SLM）
    final_result = slm_result
    if "deep_reasoner" in agent_outputs and agent_outputs["deep_reasoner"].get("reasoned_result"):
        final_result = agent_outputs["deep_reasoner"]["reasoned_result"]
    elif "reflector" in agent_outputs and agent_outputs["reflector"].get("verified_result"):
        final_result = agent_outputs["reflector"]["verified_result"]
    elif "searcher" in agent_outputs and agent_outputs["searcher"].get("search_results"):
        # 取第一个高分结果
        sr = agent_outputs["searcher"]["search_results"]
        if sr:
            final_result = sr[0]

    # 构建输出
    mas_output = {
        "sample_id": sample.get("sample_id"),
        "router_evaluation": router_output,
        "agent_outputs": agent_outputs,
        "final_result": final_result,
        "blackboard": blackboard
    }
    return mas_output

# -------------------------- 4. 主流程 -------------------------- 
def main():
    # 加载 SLM 结果
    base_dir = os.path.dirname(__file__)
    sllm_path = os.path.join(base_dir, "..", "data", "sllm_results.jsonl")
    sllm_path = os.path.normpath(sllm_path)
    if not os.path.exists(sllm_path):
        print(f"SLM 结果文件不存在：{sllm_path}")
        return

    samples = load_sllm_results(sllm_path)
    mas_results = []

    for idx, sample in enumerate(samples, 1):
        print(f"处理样本 {idx}：{sample.get('sample_id')}")
        mas_result = run_mas_for_sample(sample, idx)
        mas_results.append(mas_result)

    # 保存 MAS 结果
    output_path = os.path.join(base_dir, "..", "data", "mas_results.jsonl")
    with jsonlines.open(output_path, "w") as writer:
        for result in mas_results:
            writer.write(result)

    print(f"MAS 流程完成，结果保存至：{output_path}")

if __name__ == "__main__":
    main()