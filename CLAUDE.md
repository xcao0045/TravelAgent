# CLAUDE.md

## Project

Multi-Agent 智能旅行规划 Web 系统 — Streamlit + LangGraph + 阿里百炼千问 + 高德地图 + ChromaDB。

## Setup & Run

```bash
conda activate TravelAgent
pip install -e .
streamlit run Home.py        # http://localhost:8501
python -m pytest tests/ -v   # 40 tests
```

## API Keys

在项目根目录 `.env` 中配置（不提交）：
```
BAILIAN_API_KEY=sk-xxx
AMAP_API_KEY=xxx
```

## Architecture

```
Home.py  →  agents/graph.py  →  agents/{orchestrator, weather, attraction, hotel, synthesizer}
                                  ├── tools/amap_tools.py (5个 LangChain @tool)
                                  └── rag/retriever.py → rag/vector_store.py (ChromaDB)
```

- **LangGraph 状态机**: 1 主控 → 3 并行子 Agent (Send API 扇出) → 1 汇总 → END
- **共享状态**: `agents/state.py` — `TravelPlanState(TypedDict)`, 14 字段
- **循环依赖解决**: `agents/graph.py` 提供 `_get_llm()`, `_get_tools()`, `_get_retriever()` 单例；子 Agent 在函数体内 lazy import
- **RAG 双库**: `user_preferences` (偏好标签) + `travel_cases` (历史案例)，入库三级去重 (MD5 → 字段匹配 → 语义)

## Key Conventions

- 所有 Agent 通过 LangGraph State 通信，不直接 import 彼此
- `config.Settings.from_env()` 读取所有配置
- 高德 API 当前封装为 LangChain Tool，非标准 MCP 协议
- 提交信息使用中文
- `.env` / `.superpowers/` / `*.egg-info/` / ChromaDB 数据 / IDE 配置 已在 .gitignore

## Do NOT

- 在 UI 暴露 API Key
- 让 Agent 直接调用高德 HTTP API（必须通过 tools/amap_tools.py 的 @tool）
- 在子 Agent 中修改 State 的输入字段（只写自己的输出字段）
