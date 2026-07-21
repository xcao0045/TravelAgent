# 🧭 Multi-Agent 智能旅行规划系统

基于 **LangGraph + 阿里百炼千问 + 高德地图 + ChromaDB** 的 Multi-Agent 旅行规划 Web 应用。

用户在页面输入目的地、出行日期和偏好，AI 自动生成覆盖天气、景点、酒店、餐饮、交通、预算的完整 Markdown 旅行方案。

## 系统架构

```
Streamlit UI (3 页面)
       │
       ▼
LangGraph 编排层 (Supervisor-Worker + Send 并行扇出)
  ├── orchestrator ──→ 验证 destination，扇出 3 路 Send
  ├── weather_agent ──→ RAG 案例库检索 + LLM 推理
  ├── attraction_agent → RAG 双库检索 + LLM 推理
  ├── hotel_agent ──→ RAG 双库检索 (category=hotel) + LLM 推理
  └── synthesizer ──→ 汇聚子 Agent 输出，生成 Markdown 报告
       │
       ▼
┌─────────────┬──────────────────┬──────────────────┐
│  ChromaDB   │  高德 LangChain  │  阿里百炼 Embedding │
│  双向量库    │  Tools (5个)     │  text-embedding-v3 │
└─────────────┴──────────────────┴──────────────────┘
```

### 分层职责

| 层 | 技术 | 职责 |
| --- | --- | --- |
| **表示层** | Streamlit | 旅行规划首页、知识库管理页、历史记录页；线程模型持久化 |
| **编排引擎层** | LangGraph | 静态 DAG 状态机，1 主控 + 3 并行 Worker + 1 汇总 |
| **能力组件层** | 高德 API + ChromaDB + 百炼 | POI/天气/路线查询、双库 RAG 向量检索、文本嵌入 |

### 两套 RAG 知识库

| 知识库 | 内容 | 检索方式 | 用途 |
| --- | --- | --- | --- |
| **用户偏好库** (`user_preferences`) | 酒店/景点/餐厅的评价 + 标签 | 纯向量检索 (余弦相似度, threshold=0.45) | 按偏好标签精准匹配推荐 |
| **优质案例库** (`travel_cases`) | 历史高分旅行方案全文 | 纯向量检索 (余弦相似度, threshold=0.45) | 同目的地/同天数的历史方案参考 |

> 📖 详细架构分析见 [docs/architecture.md](docs/architecture.md)、[docs/rag-pipeline.md](docs/rag-pipeline.md)、[docs/agent-orchestration.md](docs/agent-orchestration.md)

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
python -m pytest tests/ -v   # 40 tests
```

## 使用指南

### 页面 1：旅行规划 (`Home.py`)

1. **输入需求** — 目的地、日期、天数、预算、偏好（如"亲子、安静、美食"）
2. **AI 执行** — LangGraph 并行调度 3 个子 Agent，RAG 检索 + LLM 推理
3. **查看方案** — 结构化 Markdown 报告 + RAG 引用来源 expander，支持下载
4. **微调优化** — 聊天框反复对话修改方案，最终确认后保存到历史记录

### 页面 2：知识库管理 (`pages/01_Knowledge_Base.py`)

- **手动录入** — 表单提交酒店/景点/餐厅评价或旅行案例
- **文件上传** — 支持 CSV、JSON、Markdown、TXT 批量导入，自动归档到 `data/knowledge_base/`
- **三级去重** — MD5 精确匹配（默认开启），可选字段匹配和语义去重
- **文档管理** — 按 source_md5 分组展示，分页浏览，支持语义搜索和删除（同步清理 chunk + 归档文件）

### 页面 3：历史记录 (`pages/02_History.py`)

- 浏览所有已确认的旅行方案
- 展开查看完整对话历史（含 RAG 引用 popover）
- 对比最终方案与初始方案的演进
- 下载 Markdown / 删除记录

## 项目结构

```
TravelAgent/
├── Home.py                          # 首页 — 旅行规划
├── config.py                        # 配置中心 (Settings dataclass)
├── pyproject.toml
├── requirements.txt
│
├── agents/                          # LangGraph Agent 层
│   ├── state.py                     # TravelPlanState (TypedDict + Annotated reducer)
│   ├── graph.py                     # 状态图构建 + 模块级单例 + 依赖初始化
│   ├── orchestrator.py              # 主控 Agent (验证 + 扇出)
│   ├── weather_agent.py             # 天气 Agent (案例库 RAG + LLM)
│   ├── attraction_agent.py          # 景点餐饮 Agent (双库 RAG + LLM)
│   ├── hotel_agent.py               # 酒店 Agent (双库 RAG + LLM, category=hotel)
│   └── synthesizer.py               # 汇总 Agent (汇聚 + Markdown 排版)
│
├── tools/                           # 高德 LangChain Tool 层
│   ├── amap_client.py               # HTTP 客户端 (weather/poi/direction/geo)
│   └── amap_tools.py                # 5 个 @tool 工厂函数
│
├── rag/                             # RAG 知识库层
│   ├── embedding.py                 # 百炼 text-embedding-v3 (1536 维)
│   ├── vector_store.py              # ChromaDB 双 Collection + chunk 管理
│   ├── retriever.py                 # DualRetriever 统一检索接口
│   ├── data_loader.py               # CSV / JSON / MD / TXT 解析器
│   └── dedup.py                     # 三级去重流水线 (MD5 → 字段 → 语义)
│
├── pages/                           # Streamlit 子页面
│   ├── 01_Knowledge_Base.py         # 知识库管理
│   └── 02_History.py                # 历史记录
│
├── docs/                            # 架构文档
│   ├── architecture.md              # 全局架构、生命周期、技术选型
│   ├── rag-pipeline.md              # RAG 全链路策略审计
│   └── agent-orchestration.md       # 多智能体编排与状态机设计
│
├── tests/                           # 40 个单元测试 (pytest)
├── storage/                         # ChromaDB 本地持久化
└── data/
    ├── history/                     # 用户方案 JSON 存档
    └── knowledge_base/              # 上传文件归档 (preferences/cases)
```

## 技术栈

| 组件 | 技术 | 说明 |
| --- | --- | --- |
| 前端 | Streamlit ≥1.28 | 纯 Python Web UI，st.session_state + st.cache_resource |
| LLM | 阿里百炼 qwen-max | ChatTongyi 集成，中文理解能力业界领先 |
| Agent 编排 | LangGraph ≥0.2 | Send API 并行扇出，Annotated reducer 解决并发写入 |
| Embedding | text-embedding-v3 | 百炼 DashScope，1536 维，余弦相似度 threshold=0.45 |
| 向量数据库 | ChromaDB ≥0.5 | PersistentClient + HNSW 索引，双 Collection |
| 地图服务 | 高德开放平台 Web API | 5 个 LangChain @tool 封装 |
| 去重 | 自研三级流水线 | MD5 → 字段匹配 → 语义相似度 |
| 测试 | pytest ≥8.0 | 40 个单元测试 |

## 设计关键点

- **Send API 扇出：** `_continue_to_sub_agents()` 返回 3 个 `Send(node, state)`，LangGraph 并行调度
- **Annotated Reducer：** `rag_refs: Annotated[dict, _merge_dict]` 保护 3 路并行写入不冲突
- **Lazy Import：** 子 Agent 在函数体内 `from agents.graph import _get_*`，解决循环依赖
- **source_md5：** 原始文档 MD5 标记，chunk 切分后所有子片段继承，支持跨 chunk 去重和批量删除
- **元数据清洗：** `_sanitize_metadata()` 剔除空列表字段，规避 ChromaDB 拒绝入库
- **Key Rotation：** 文件上传后 `uploader_key += 1` + `st.rerun()`，清除 uploader widget 残留

## License

MIT
