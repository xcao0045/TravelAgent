# 🧭 Multi-Agent 智能旅行规划系统

基于 **LangGraph + 阿里百炼千问 + 高德地图** 的 Multi-Agent 旅行规划 Web 应用。用户在页面输入目的地、出行日期和偏好，AI 自动生成覆盖天气、景点、酒店、餐饮、交通、预算的完整 Markdown 旅行方案。

## 系统架构

```
Streamlit UI (3 页面)
       │
       ▼
LangGraph 编排层
  ├── 主控 Agent ──→ 解析需求、调度子 Agent
  ├── 天气 Agent ──→ 高德天气 API + RAG 案例库
  ├── 景点餐饮 Agent → 高德 POI 搜索 + RAG 偏好库 + RAG 案例库
  ├── 酒店 Agent ──→ 高德 POI 搜索 + RAG 偏好库 + RAG 案例库
  └── 汇总 Agent ──→ 整合输出 Markdown 报告
       │
       ▼
┌─────────────┬──────────────────┐
│  ChromaDB   │  高德 LangChain  │
│  双向量库    │  Tools (5个)     │
└─────────────┴──────────────────┘
```

### 分层职责

| 层 | 技术 | 职责 |
|------|------|------|
| UI | Streamlit | 旅行规划首页、知识库管理页、历史记录页 |
| 编排 | LangGraph | 状态机编排 1 主控 + 3 并行子 Agent + 1 汇总 |
| 能力底座 | 高德 API + ChromaDB | POI 搜索、天气查询、路线规划、RAG 向量检索 |

### 两套 RAG 知识库

| 知识库 | 内容 | 用途 |
|------|------|------|
| **用户偏好库** | 酒店/景点/餐厅的评价文本 + 标签 | 按用户偏好标签精准匹配推荐 |
| **优质案例库** | 历史高分旅行方案 | 相似目的地/天数的历史方案参考 |

## 快速开始

### 环境要求

- Python >= 3.10
- [Conda](https://docs.conda.io/) (推荐) 或 venv
- 阿里百炼 API Key ([获取地址](https://bailian.console.aliyun.com/))
- 高德地图 API Key ([获取地址](https://console.amap.com/dev/key/app))

### 安装

```bash
# 1. 克隆项目
git clone git@github.com:xcao0045/TravelAgent.git
cd TravelAgent

# 2. 创建虚拟环境
conda create -n TravelAgent python=3.10 -y
conda activate TravelAgent

# 3. 安装依赖
pip install -e .

# 4. 配置 API Key
cp .env.example .env
# 编辑 .env 填入你的 Key:
#   BAILIAN_API_KEY=sk-xxx
#   AMAP_API_KEY=xxx
```

### 启动

```bash
streamlit run Home.py
```

浏览器访问 `http://localhost:8501`

### 运行测试

```bash
python -m pytest tests/ -v
```

## 使用指南

### 页面 1：旅行规划

1. **输入需求** — 目的地、日期、天数、预算、偏好（如"亲子、安静、美食"）
2. **AI 执行** — 实时显示各 Agent 执行状态
3. **查看方案** — 结构化 Markdown 报告，支持下载
4. **微调优化** — 通过聊天框反复对话修改方案，直到满意后保存

### 页面 2：知识库管理

- 手动录入酒店/景点/餐厅的评价和标签
- 支持 CSV、JSON、Markdown 文件批量导入
- 三级去重保护（MD5 + 字段匹配 + 语义相似度）
- 已有数据搜索、编辑、删除

### 页面 3：历史记录

- 浏览所有历史旅行方案
- 展开查看完整对话记录和方案版本演进
- 下载 Markdown / 删除记录

## 项目结构

```
TravelAgent/
├── Home.py                      # 首页 — 旅行规划
├── config.py                    # 配置中心 (Settings)
├── pyproject.toml
├── requirements.txt
│
├── agents/                      # LangGraph Agent 层
│   ├── state.py                 # TravelPlanState 共享状态
│   ├── graph.py                 # 状态图构建 + 依赖初始化
│   ├── orchestrator.py          # 主控 Agent
│   ├── weather_agent.py         # 天气 Agent
│   ├── attraction_agent.py      # 景点餐饮 Agent
│   ├── hotel_agent.py           # 酒店 Agent
│   └── synthesizer.py           # 汇总 Agent
│
├── tools/                       # 高德 LangChain Tool 层
│   ├── amap_client.py           # HTTP 客户端
│   └── amap_tools.py            # 5 个 @tool
│
├── rag/                         # RAG 知识库层
│   ├── embedding.py             # 百炼 text-embedding-v3
│   ├── vector_store.py          # ChromaDB 双 Collection
│   ├── retriever.py             # 统一检索接口
│   ├── data_loader.py           # CSV/JSON/MD/TXT 解析
│   └── dedup.py                 # 三级去重
│
├── pages/                       # Streamlit 子页面
│   ├── 01_Knowledge_Base.py     # 知识库管理
│   └── 02_History.py            # 历史记录
│
├── tests/                       # 40 个单元测试
├── docs/superpowers/            # 设计文档 + 实现计划
├── storage/                     # ChromaDB 本地持久化
└── data/history/                # 用户方案 JSON 存档
```

## 技术栈

| 组件 | 技术 |
|------|------|
| 前端 | Streamlit |
| LLM | 阿里百炼千问 (qwen-max) |
| Agent 编排 | LangGraph (状态机 + Send 并行扇出) |
| 向量数据库 | ChromaDB |
| Embedding | 阿里百炼 text-embedding-v3 |
| 地图服务 | 高德开放平台 Web API |
| 测试 | pytest (40 tests) |

## License

MIT
