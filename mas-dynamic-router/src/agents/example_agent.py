import json
from src.agent import BaseAgent

class ExampleAgent(BaseAgent):
    def __init__(self, name: str):
        super().__init__(name)

    def process_input(self, input_data: dict) -> dict:
        # 处理输入数据并生成响应
        prompt = self.create_prompt(input_data)
        response = self.call_llm(prompt)
        return self.format_response(response)

    def create_prompt(self, input_data: dict) -> str:
        # 根据输入数据生成提示
        return f"根据以下信息生成响应：{json.dumps(input_data)}"

    def call_llm(self, prompt: str) -> str:
        # 调用LLM并返回结果
        # 这里可以添加与LLM的实际交互逻辑
        return "示例响应"

    def format_response(self, response: str) -> dict:
        # 格式化LLM的响应为字典
        return {"response": response}