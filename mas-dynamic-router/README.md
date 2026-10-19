# mas-dynamic-router

## 项目简介
该项目实现了一个基于动态路由的多智能体系统（MAS），旨在通过智能体与大型语言模型（LLM）交互来处理和分析SLM结果。系统的核心组件包括Router和多个Agent，Router负责接收和调度任务，而Agent则执行具体的操作并返回结果。

## 功能
- **动态路由**：根据SLM结果动态选择合适的Agent进行处理。
- **多智能体交互**：通过LLM与多个Agent进行交互，支持复杂的任务处理。
- **结果分析**：对SLM结果进行解析和格式化，提供最终的输出。

## 目录结构
```
mas-dynamic-router
├── src
│   ├── main.py          # 应用程序入口点
│   ├── router.py        # Router Agent逻辑
│   ├── agent.py         # Agent基类和实现
│   ├── langref.py       # SLM相关逻辑
│   ├── config.py        # 系统配置
│   ├── prompts
│   │   └── agent_prompt.tpl  # Agent提示模板
│   ├── agents
│   │   ├── __init__.py  # agents模块初始化 
│   │   ├── base_agent.py # Agent基类
│   │   └── example_agent.py # 示例Agent实现
│   └── utils
│       ├── io.py        # 输入输出工具函数
│       └── parsing.py   # 解析工具函数
├── data
│   └── sllm_results.jsonl # SLM结果数据
├── configs
│   └── settings.yaml    # 系统配置文件
├── tests
│   ├── test_router.py   # Router单元测试
│   └── test_agents.py    # Agent单元测试
├── requirements.txt      # Python依赖包
├── pyproject.toml       # 项目配置文件
├── .env.example          # 环境变量示例配置
└── README.md             # 项目文档
```

## 安装
1. 克隆项目：
   ```
   git clone <repository-url>
   cd mas-dynamic-router
   ```

2. 安装依赖：
   ```
   pip install -r requirements.txt
   ```

3. 配置环境变量：
   复制`.env.example`为`.env`并根据需要修改配置。

## 使用
1. 运行主程序：
   ```
   python src/main.py
   ```

2. 系统将自动加载SLM结果并开始处理。

## 贡献
欢迎提交问题和建议，或直接提交代码贡献。请遵循项目的贡献指南。

## 许可证
该项目遵循MIT许可证。