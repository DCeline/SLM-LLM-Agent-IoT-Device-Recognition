class BaseAgent:
    def __init__(self, agent_name: str):
        self.agent_name = agent_name

    def process_input(self, input_data: dict) -> dict:
        raise NotImplementedError("Subclasses should implement this method.")

    def generate_prompt(self, context: str) -> str:
        raise NotImplementedError("Subclasses should implement this method.")

    def parse_output(self, output: str) -> dict:
        raise NotImplementedError("Subclasses should implement this method.")

    def run(self, input_data: dict) -> dict:
        prompt = self.generate_prompt(input_data.get("context", ""))
        output = self.call_llm(prompt)
        return self.parse_output(output)

    def call_llm(self, prompt: str) -> str:
        # Placeholder for LLM call logic
        return "LLM response based on prompt"