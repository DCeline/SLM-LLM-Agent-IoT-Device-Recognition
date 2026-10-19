# Router
PromptTemplate(
    template="""
    任务：基于三维评估（格式合规感知 / 知识回忆一致性 / 推理自洽性）对 SLM 的识别结果进行结构化判断。
    输入信息：
    - 原始报文：{raw_packet}
    - SLM结果：{slm_result}
    - 当前推理状态：{current_state}
    - 已有信息：{existing_info}

    评估维度与触发规则（Router 内部规则）：
    1. 格式合规感知（Perception）：检测型号外在结构（字母-数字组合、品牌前缀、Model:/Ver: 标识等）。输出pass/fail。
    2. 知识回忆一致性（Recollection）：核验与设备知识库/品牌型号体系匹配。输出pass/fail。
    3. 推理驱动自洽性（Reasoning）：检查识别结果与报文协议/参数/应用场景逻辑一致性。输出pass/fail。

    输出为严格的 JSON，必须以{{开头。字段包括：
    - eva_tag：对象，包含perception/recollection/reasoning，每项为 "pass" 或 "fail"。
    - reason：触发该动作的简要说明（中文）。
    - key_evidence：列表，列出支持评估的关键证据片段（如报文字段、匹配条目等）。

    不要添加任何额外文本。要求返回结果以{{开头。

    输出示例：
    - 需要搜索（回忆不一致）：
    {{
        "eva_tag": {{"perception": "pass", "recollection": "fail", "reasoning": "pass"}},
        "reason": "识别结果与已知品牌型号体系不匹配或在知识库中未命中",
        "key_evidence": ["型号 'ABC-1234' 未在设备知识库中检索到", "存在相似前缀但编号不匹配"]
    }}
    - 需要反思（格式不合规）：
    {{
        "eva_tag": {{"perception": "fail", "recollection": "pass", "reasoning": "pass"}},
        "reason": "型号格式异常，缺失标准标识（如 Model: / Ver:）或命名不符合厂商习惯",
        "key_evidence": ["报文字段 'DeviceModel: ??'", "缺少品牌前缀"]
    }}
    - 需要深度推理（推理不自洽或复杂场景）：
    {{
        "eva_tag": {{"perception": "pass", "recollection": "pass", "reasoning": "fail"}},
        "reason": "识别结果与报文协议/参数存在逻辑矛盾或需要多源融合验证",
        "key_evidence": ["报文协议为工业协议但型号属消费级系列"]
    }}
    """,
    input_variables=["raw_packet", "slm_result", "current_state", "existing_info"]
)

# Reflector
PromptTemplate(
    template="""
    任务：根据 Router 给出的建议，快速验证并（如有必要）修正型号格式或小范围拼写/结构问题（低成本校验）。
    输入信息：
    - 当前结果：{current_result}  （结构化对象或字符串）
    - router 的说明：{reason}
    - 关键证据：{key_evidence}

    输出为严格的 JSON 格式，字段包括：
    - verified_result：对象，包含 type、vendor、model。
    - add_evidence：列表，反回补充的证据片段（例如从报文中提取的字段）。
    不要添加任何额外文本。要求返回结果以{{开头。

    输出示例：
    {{
        "verified_result": {{"type":"Network Camera","vendor":"Hikvision","model":"DS-2CD2142FWD-I"}},
        "add_evidence": ["型号格式符合厂商惯例，修正了少量拼写错误"],
    }}
    """,
    input_variables=["current_result", "reason", "key_evidence"]
)

# Searcher
PromptTemplate(
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

# DeepReasoner
PromptTemplate(
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
        "reasoned_result": "DS-2CD2142FWD-I",
        "rationale": [
            "1. 初始预测为 DS-2CD2142FWD-I；",
            "2. Searcher 输出与官网条目高度匹配；",
            "3. 报文包含 RTSP/ONVIF 特征，与摄像机场景一致；因此提高置信度。"
        ],
    }}
    """,
    input_variables=["initial_result", "existing_info", "raw_packet"]
)