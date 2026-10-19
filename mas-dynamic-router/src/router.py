import os
import re
import json
import jsonlines
from agent import Reflector, Searcher, DeepReasoner
from langref import clean_and_parse_llm_output

class Router:
    """
    Router: 根据三维认知评估（感知 Perception / 回忆 Recollection / 推理 Reasoning）
    的合格/不合格标签决定是否调度对应 Agent。
    不再直接使用 SLM 的单一置信度来路由。
    """

    def __init__(self, agents, known_brands: list = None):
        self.agents = agents
        # 简易知识库（用于 recollection 层的轻量判断），可扩展为文件加载
        self.known_brands = known_brands or ["huawei", "hikvision", "tp-link", "tplink", "tenda", "dlink", "zyxel"]

    def evaluate_three_dimensions(self, slm_result: dict) -> dict:
        """
        基于规则的三维度快速评估（用于产生 Router 的 decision 输入）
        返回结构示例：
        {
            "perception": {"label": "合格"/"不合格", "evidence": [...]},
            "recollection": {"label": "...", "evidence": [...]},
            "reasoning": {"label": "...", "evidence": [...]},
            "final_results_hint": {...}  # Router 的初步修正建议（可选）
        }
        """
        pred = slm_result.get("prediction") or {}
        model_str = (pred.get("model") or "") if isinstance(pred, dict) else str(pred or "")
        vendor = (pred.get("vendor") or "") if isinstance(pred, dict) else ""
        raw = slm_result.get("raw_packet", slm_result.get("text", ""))

        # 感知（格式合规性）：检查是否匹配常见型号结构（字母+数字组合、含Model:/Ver:等）
        evidence_p = []
        label_p = "合格"
        if not model_str or len(model_str) < 3:
            label_p = "不合格"
            evidence_p.append("型号字符串过短或为空")
        if not re.search(r'[A-Za-z].*\d|\d.*[A-Za-z]', model_str):
            # 没有字母数字混合的典型结构
            label_p = "不合格"
            evidence_p.append("型号缺乏典型字母-数字组合")
        if re.search(r'(Model:|Ver:|version|v\.)', raw, re.IGNORECASE):
            evidence_p.append("报文中存在 Model/Ver 标记（有利于格式感知）")

        # 回忆（知识库一致性）：简单检查 vendor 是否为已知品牌或 model 包含品牌前缀
        evidence_r = []
        label_r = "合格"
        mv_lower = model_str.lower()
        vendor_lower = vendor.lower()
        known_hit = any(b in vendor_lower or b in mv_lower for b in self.known_brands)
        if not known_hit:
            label_r = "不合格"
            evidence_r.append("厂商/型号与本地已知品牌库不匹配")
        else:
            evidence_r.append("在轻量品牌列表中找到匹配项")

        # 推理（逻辑一致性）：检查报文协议/关键词是否与类型/型号逻辑矛盾
        evidence_w = []
        label_w = "合格"
        inst_type = pred.get("type", "") if isinstance(pred, dict) else ""
        # 简单规则：如果类型为 Network Camera，但报文缺少常见摄像机关键词则警示
        if inst_type and re.search(r'camera|ipc|rtsp|onvif', inst_type, re.IGNORECASE):
            if not re.search(r'rtsp|onvif|video|snapshot', raw, re.IGNORECASE):
                label_w = "不合格"
                evidence_w.append("预测为摄像机但报文中缺少摄像机典型协议关键词")
        # 其它通用矛盾检测（占位）
        if "sensor" in mv_lower and "http" in raw.lower():
            # 假设某些工业传感器可能不会出现 http banner —— 仅示例规则
            evidence_w.append("型号含 'sensor'，但报文包含 http，可能与应用场景不一致")

        return {
            "perception": {"label": label_p, "evidence": evidence_p},
            "recollection": {"label": label_r, "evidence": evidence_r},
            "reasoning": {"label": label_w, "evidence": evidence_w},
            "final_results_hint": {
                "initial_prediction": pred
            }
        }

    def route(self, slm_result: dict, blackboard: dict = None, max_rounds: int = 2) -> dict:
        """
        核心路由逻辑：
        1) 评估三维度
        2) 对不合格维度按需调用对应 Agent（Reflector / Searcher / DeepReasoner）
        3) 读取黑板并裁定最终结果（DeepReasoner 的输出优先）
        返回结构化结果，方便程序解析与记录（包含 decision、final_result、agents调用记录等）
        """
        if blackboard is None:
            blackboard = {"notes": [], "evidence": [], "agent_outputs": []}

        evals = self.evaluate_three_dimensions(slm_result)
        # 决策结构初始
        decision_obj = {
            "decision": "reliable" if all(v["label"] == "合格" for v in [evals["perception"], evals["recollection"], evals["reasoning"]]) else "unreliable",
            "evals": evals,
            "next_agents": [],
            "agent_instructions": [],
            "final_result": None,
            "blackboard": blackboard
        }

        if decision_obj["decision"] == "reliable":
            # 全合格，直接输出初始预测作为 final_result
            decision_obj["final_result"] = evals["final_results_hint"]["initial_prediction"]
            decision_obj["blackboard"]["notes"].append("三维度均合格，直接终止路由。")
            return decision_obj

        # 根据不合格维度生成 agent 调度并调用
        mapping = {
            "perception": ("reflector", "请针对型号格式与外观结构进行快速校验，修正或确认型号字符串的格式与完整性。"),
            "recollection": ("searcher", "请在内置设备知识库中检索厂商/型号是否存在并返回匹配结果及证据。"),
            "reasoning": ("deep_reasoner", "请基于报文上下文进行深度推理，判断型号与报文是否逻辑自洽并给出推理链。")
        }

        # 最多做 max_rounds 轮（可循环，但当前实现为单轮调用所有需要 agent）
        for dim in ("perception", "recollection", "reasoning"):
            if evals[dim]["label"] == "不合格":
                agent_key, instr = mapping[dim]
                decision_obj["next_agents"].append(agent_key)
                # 精准 instruction 带上 evidence 与当前预测信息
                instruction = {
                    "from_router": instr,
                    "evidence": evals[dim]["evidence"],
                    "syn_product": evals["final_results_hint"]["initial_prediction"],
                    "raw_packet_snippet": slm_result.get("raw_packet", slm_result.get("text", ""))[:512]
                }
                decision_obj["agent_instructions"].append({agent_key: instruction})

                # 调用 agent：统一以 instruction dict 作为单参数传入（agent 需实现兼容）
                try:
                    agent = self.agents.get(agent_key)
                    if agent is None:
                        raise KeyError(f"Agent {agent_key} 未注册")
                    output = agent.process(instruction)  # agent.process 应返回 dict，可扩展
                    # 记录黑板与 agent 输出
                    decision_obj["blackboard"]["agent_outputs"].append({agent_key: output})
                    decision_obj["blackboard"]["evidence"].append({agent_key: output.get("rationale", output)})
                except Exception as e:
                    # 若 agent 调用失败，记录并继续
                    decision_obj["blackboard"]["notes"].append(f"调用 {agent_key} 失败：{type(e).__name__} {str(e)}")
                    decision_obj["blackboard"]["agent_outputs"].append({agent_key: {"error": str(e)}})

        # 裁定最终结果：若 DeepReasoner 返回非空，则优先采用；否则合并其它 agent 的修正建议
        final = None
        # 查找 deep_reasoner 输出
        for out in decision_obj["blackboard"]["agent_outputs"]:
            if "deep_reasoner" in out:
                dr_out = out["deep_reasoner"]
                if dr_out and dr_out.get("reasoned_result"):
                    final = dr_out["reasoned_result"]
                    decision_obj["blackboard"]["notes"].append("采用 Deep-Reasoner 输出作为最终结果（优先级最高）。")
                    break

        if final is None:
            # 合并 reflector / searcher 的 verified_result 或 search_results
            merged = evals["final_results_hint"]["initial_prediction"] or {}
            for out in decision_obj["blackboard"]["agent_outputs"]:
                if "reflector" in out:
                    r = out["reflector"]
                    if r and r.get("verified_result"):
                        merged = r["verified_result"]
                        decision_obj["blackboard"]["notes"].append("使用 Reflector 的修正结果。")
                if "searcher" in out:
                    s = out["searcher"]
                    # 如果 searcher 返回明确存在匹配，采用其最可信条目
                    if s and s.get("search_results"):
                        # 优先使用第一条匹配
                        sr = s["search_results"][0]
                        if sr:
                            merged = sr
                            decision_obj["blackboard"]["notes"].append("使用 Searcher 的检索结果作为最终候选。")
            final = merged

        decision_obj["final_result"] = final
        # 最终判断是否可靠：若经过 agent 校正后与原预测有合理匹配，则标为 reliable（可自定义阈值）
        decision_obj["decision"] = "reliable" if final and final != {} else "unreliable"
        return decision_obj


def load_sllm_results(file_path):
    results = []
    if not os.path.exists(file_path):
        print(f"结果文件不存在：{file_path}")
        return results
    with jsonlines.open(file_path) as reader:
        for obj in reader:
            results.append(obj)
    return results


def main():
    agents = {
        "reflector": Reflector(),
        "searcher": Searcher(),
        "deep_reasoner": DeepReasoner()
    }

    router = Router(agents)
    # 期望 data 目录位于项目根（src/../data）
    base_dir = os.path.dirname(__file__)
    sllm_path = os.path.join(base_dir, "..", "data", "sllm_results.jsonl")
    sllm_path = os.path.normpath(sllm_path)
    slm_results = load_sllm_results(sllm_path)

    for result in slm_results:
        routed = router.route(result)
        # 输出为 JSON 结构，便于后续记录或黑板可视化
        print(json.dumps(routed, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()