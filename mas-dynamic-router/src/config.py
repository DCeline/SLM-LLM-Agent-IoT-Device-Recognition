import os

# -------------------------- 1. 配置参数 --------------------------
API_KEY = os.getenv("API_KEY", "your_api_key_here")  # API密钥
BASE_URL = os.getenv("BASE_URL", "https://api.example.com")  # API基础URL
MODEL_NAME = os.getenv("MODEL_NAME", "your_model_name_here")  # 模型名称

# -------------------------- 2. 运行时设置 --------------------------
TIMEOUT = int(os.getenv("TIMEOUT", 30))  # 请求超时时间（秒）
MAX_RETRIES = int(os.getenv("MAX_RETRIES", 5))  # 最大重试次数
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")  # 日志级别

# -------------------------- 3. 数据文件路径 --------------------------
SLLM_RESULTS_PATH = os.path.join("data", "sllm_results.jsonl")  # SLM结果数据文件路径
OUTPUT_PATH = os.path.join("output", "results.jsonl")  # 输出结果文件路径

# -------------------------- 4. 其他配置 --------------------------
AGENT_TIMEOUT = int(os.getenv("AGENT_TIMEOUT", 60))  # Agent处理超时时间（秒）
ENABLE_LOGGING = os.getenv("ENABLE_LOGGING", "true").lower() == "true"  # 是否启用日志记录

# -------------------------- 5. 代理配置 --------------------------
AGENT_TYPES = ["Reflector", "Searcher", "DeepReasoner"]  # 可用的代理类型

# -------------------------- 6. 其他常量 --------------------------
DEFAULT_PROMPT_TEMPLATE = "default_prompt.tpl"  # 默认提示模板文件名