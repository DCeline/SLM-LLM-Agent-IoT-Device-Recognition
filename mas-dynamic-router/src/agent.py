import json
import time
from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langref import API_KEY, BASE_URL, MODEL_NAME, clean_and_parse_llm_output

# 简单抽象基类（可扩展）
class BaseAgent:
    def __init__(self, name: str, temperature: float = 0.0):
        self.name = name
        self.llm = ChatOpenAI(
            model_name=MODEL_NAME,
            temperature=temperature,
            openai_api_key=API_KEY,
            base_url=BASE_URL,
            streaming=False
        )

    def _call_llm(self, prompt: str) -> str:
        try:
            # 这里使用 predict 接口（根据实际 langchain 版本调整）
            return self.llm.predict(prompt)
        except Exception as e:
            # 容错：返回简短说明，clean_and_parse_llm_output 会处理
            return json.dumps({"error": f"llm_call_failed: {str(e)}"})

    def process(self, instruction: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError


class Reflector(BaseAgent):
    def __init__(self):
        super().__init__("reflector", temperature=0.0)

    def process(self, instruction: Dict[str, Any]) -> Dict[str, Any]:
        prompt = f"""
        任务：快速校验并修正型号格式（低功耗）。
        输入 current_result：{json.dumps(instruction.get('syn_product') or instruction.get('current_result') or {}, ensure_ascii=False)}
        Router 说明：{instruction.get('from_router') or instruction.get('reason')}
        关键证据：{json.dumps(instruction.get('evidence') or instruction.get('key_evidence') or [], ensure_ascii=False)}
        报文片段：{instruction.get('raw_packet_snippet','')}

        请严格返回 JSON，包含 verified_result（对象：type/vendor/model/confidence）、rationale（列表）、add_evidence（列表）。
        """
        raw = self._call_llm(prompt)
        parsed = clean_and_parse_llm_output(raw, default_json={"verified_result": None, "rationale": [], "add_evidence": []})
        # 保证 verified_result 有标准子字段
        vr = parsed.get("verified_result") or {}
        if isinstance(vr, str):
            # 尝试简单封装字符串为 model 字段
            vr = {"type": None, "vendor": None, "model": vr, "confidence": 0.0}
        parsed["verified_result"] = vr
        return parsed


class Searcher(BaseAgent):
    def __init__(self):
        super().__init__("searcher", temperature=0.0)

    def process(self, instruction: Dict[str, Any]) -> Dict[str, Any]:
        target = instruction.get("syn_product") or instruction.get("target") or ""
        prompt = f"""
任务：在内置知识库或近似检索机制中查找与目标匹配的型号/厂商条目。
输入 target：{json.dumps(target, ensure_ascii=False)}
检索说明：{instruction.get('from_router') or instruction.get('search_goal','')}
上下文片段：{instruction.get('raw_packet_snippet','')}

请严格返回 JSON，包含 search_queries（列表）、search_results（列表，元素包含type/vendor/model/source/score）、summary（字符串）。
"""
        raw = self._call_llm(prompt)
        parsed = clean_and_parse_llm_output(raw, default_json={"search_queries": [], "search_results": [], "summary": ""})
        # 保证 search_results 是列表且元素为 dict
        sr = parsed.get("search_results") or []
        if isinstance(sr, dict):
            sr = [sr]
        parsed["search_results"] = sr
        return parsed


class DeepReasoner(BaseAgent):
    def __init__(self):
        super().__init__("deep_reasoner", temperature=0.0)

    def process(self, instruction: Dict[str, Any]) -> Dict[str, Any]:
        prompt = f"""
任务：基于已有信息与报文片段进行深度推理，给出最终 reasoned_result（对象：type/vendor/model/confidence）与详细推理链。
初始结果：{json.dumps(instruction.get('syn_product') or instruction.get('initial_result') or {}, ensure_ascii=False)}
已有信息（黑板）：{json.dumps(instruction.get('existing_info') or instruction.get('evidence') or {}, ensure_ascii=False)}
报文片段：{instruction.get('raw_packet_snippet','')}

请严格返回 JSON，包含 reasoned_result、rationale（步骤链）、notes（可选）。
"""
        raw = self._call_llm(prompt)
        parsed = clean_and_parse_llm_output(raw, default_json={"reasoned_result": None, "rationale": [], "notes": ""})
        rr = parsed.get("reasoned_result") or {}
        if isinstance(rr, str):
            rr = {"type": None, "vendor": None, "model": rr, "confidence": 0.0}
        parsed["reasoned_result"] = rr
        return parsed