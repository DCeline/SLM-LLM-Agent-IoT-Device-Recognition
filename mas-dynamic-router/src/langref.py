import json
import jsonlines
from typing import Dict, Any

def load_sllm_results(file_path: str) -> Dict[str, Any]:
    results = []
    with jsonlines.open(file_path, 'r') as reader:
        for obj in reader:
            results.append(obj)
    return results

def parse_sllm_result(result: Dict[str, Any]) -> Dict[str, Any]:
    parsed_result = {
        "sample_id": result.get("sample_id"),
        "model_output": result.get("model_output_info", {}).get("model_output", "无"),
        "confidence": result.get("model_output_info", {}).get("confidence", 0.0),
        "is_correct": result.get("evaluation", {}).get("is_correct", False),
        "reason": result.get("evaluation", {}).get("reason", "")
    }
    return parsed_result

def format_result_as_json(result: Dict[str, Any]) -> str:
    return json.dumps(result, ensure_ascii=False)

def main():
    sllm_results_path = "data/sllm_results.jsonl"
    results = load_sllm_results(sllm_results_path)
    
    for result in results:
        parsed_result = parse_sllm_result(result)
        formatted_result = format_result_as_json(parsed_result)
        print(formatted_result)

if __name__ == "__main__":
    main()