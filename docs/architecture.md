# 🧭 全局架构、运行生命周期与技术选型

## 1. 架构模式

本系统采用 **Streamlit-Over-LangGraph 双层分层架构**。前端（表示层）与后端核心逻辑（编排引擎层 + 能力组件层）通过 LangGraph State dict 进行数据交换，Streamlit 不感知 LangGraph 内部状态机，LangGraph 不感知 Streamlit 会话周期。

```
┌─ 表示层 (Streamlit UI) ───────────────────────────────────────────┐
│  Home.py              ← 规划入口 + 微调对话                          │
│  pages/01_Knowledge_Base.py  ← 知识库管理（录入/上传/列表/删除）       │
│  pages/02_History.py  ← 历史记录浏览（搜索/下载/删除）                 │
│                                                                   │
│  st.session_state.history[thread_id]  ← 线程模型，跨 rerun 持久化   │
└───────────────────────────────────────────────────────────────────┘
        │                                   ▲
        │ run_travel_plan(user_input, cfg)   │ result (TravelPlanState)
        ▼                                   │
┌─ 编排引擎层 (LangGraph StateGraph) ────────────────────────────────┐
│  agents/graph.py → build_graph() → compile() → graph.invoke()     │
│                                                                   │
│  orchestrator ──Send──→ weather_agent     ┐                       │
│              ──Send──→ attraction_agent   ├─ 并行扇出 (3 路)       │
│              ──Send──→ hotel_agent        ┘                       │
│                                    ↓                               │
│                              synthesizer (汇总)                     │
└───────────────────────────────────────────────────────────────────┘
        │                                         ▲
        │ lazy import (避免循环依赖)                 │
        ▼                                         │
┌─ 能力组件层 ───────────────────────────────────────────────────────┐
│  tools/amap_client.py   ← 高德 REST API 封装 (requests)            │
│  tools/amap_tools.py    ← 5 个 @tool 工厂函数                      │
│                                                                   │
│  rag/embedding.py       ← DashScopeEmbeddings (text-embedding-v3) │
│  rag/vector_store.py    ← ChromaDB PersistentClient + chunk 管理   │
│  rag/retriever.py       ← DualRetriever (双库统一检索接口)          │
│  rag/dedup.py           ← 三级去重流水线 (MD5 → 字段 → 语义)        │
│  rag/data_loader.py     ← CSV / JSON / Markdown 文件解析器          │
└───────────────────────────────────────────────────────────────────┘
```

### 关键分层决策

| 决策 | 说明 |
|---|---|
| **Streamlit 不感知 Graph** | `Home.py` 仅调用 `run_travel_plan()` 单一入口，拿到 State dict 后写入自己的线程模型 |
| **Graph 不感知 Streamlit** | 所有 Agent 通过 `TravelPlanState` TypedDict 通信，不访问 `st.session_state` |
| **循环依赖解决** | `agents/graph.py` 提供 `_get_llm()` / `_get_tools()` / `_get_retriever()` 模块级单例；子 Agent 在函数体内 `from agents.graph import _get_*`（lazy import），不做模块顶层 import |
| **配置中心化** | `config.Settings.from_env()` 从 `.env` 读取全部配置，所有组件通过 dataclass 实例获取参数 |

---

## 2. 完整运行生命周期

### 2.1 规划链路 (Home.py)

```
用户填写表单 (destination, date, days, budget, preferences)
  → 点击 "🚀 开始规划"
  → st.session_state.is_processing = True (锁定按钮, 防止重复提交)
  → run_travel_plan(user_input, settings)
      → build_graph(settings)
          → _init_dependencies(settings)
              ├─ LLM:       ChatTongyi(qwen-max) 初始化
              ├─ Tools:     AmapClient + create_amap_tools() → 5 个 @tool
              └─ RAG:       DashScopeEmbeddings → VectorStoreManager → DualRetriever
          → StateGraph(TravelPlanState)
              ├─ add_node × 5
              ├─ set_entry_point("orchestrator")
              ├─ add_conditional_edges("orchestrator", _continue_to_sub_agents)
              ├─ add_edge("weather_agent"    → "synthesizer")
              ├─ add_edge("attraction_agent" → "synthesizer")
              ├─ add_edge("hotel_agent"      → "synthesizer")
              └─ add_edge("synthesizer"      → END)
          → compile()
      → graph.invoke(initial_state)
          ├─ orchestrator_node:    验证 destination 非空
          │   ├─ 空 → error_log 追加, 图结束
          │   └─ 非空 → 透传
          ├─ _continue_to_sub_agents(): 返回 3 个 Send(state)
          ├─ [并行] weather_agent_node:
          │     lazy import → retriever.retrieve_cases(k=3)
          │     → 构建 case_rag → llm.invoke(prompt)
          │     → return {"weather_report": ..., "rag_refs": {...}}
          ├─ [并行] attraction_agent_node:
          │     → retriever.retrieve_both(k_prefs=5, k_cases=3)
          │     → llm.invoke(prompt) → regex 提取 JSON
          │     → return {"attractions": [...], "restaurants": [...], "rag_refs": {...}}
          ├─ [并行] hotel_agent_node:
          │     → retriever.retrieve_both(category="hotel", k_prefs=5, k_cases=3)
          │     → llm.invoke(prompt) → regex 提取 JSON
          │     → return {"hotels": [...], "rag_refs": {...}}
          ├─ 3 子 Agent 完成 → LangGraph _merge_dict reducer 合并 rag_refs
          └─ synthesizer_node:
                → _format_list() 渲染 attractions/restaurants/hotels
                → llm.invoke(prompt) → 生成完整 Markdown
                → return {"final_report": ..., "routes": []}
      → result (TravelPlanState) 返回 Home.py
  → _new_thread(meta) → 写入 final_report + initial_report + rag_refs
  → is_processing = False
  → 步骤 3: 渲染 Markdown + RAG expander
  → 步骤 4: 微调对话 (chat_input → llm.invoke → st.rerun)
  → 确认按钮: JSON 写入 data/history/{thread_id}_{dest}_{days}天.json
```

### 2.2 知识库链路 (01_Knowledge_Base.py)

```
用户选择目标库 (preferences / cases)
  ├─ 手动录入 Tab:
  │     st.form → Document 构造 → dedup_pipeline (仅 MD5)
  │     → 通过: add_to_preferences/cases → chunk → ChromaDB upsert
  │     → 重复: st.warning + 跳过
  │
  ├─ 文件上传 Tab:
  │     st.file_uploader → 本地归档 data/knowledge_base/{type}/
  │     → load_file_to_docs() 按扩展名调度解析器 (CSV/JSON/MD/TXT)
  │     → dedup_pipeline 逐条检查 (MD5)
  │     → add_to_preferences/cases (标记 source_md5 + source_file)
  │     → uploader_key += 1 → st.rerun() (清除 widget)
  │
  └─ 文档列表:
        list_documents() → 按 source_md5 分组 → 分页 (PAGE_SIZE=5)
        → 语义搜索: similarity_search_with_relevance_scores(threshold=0.3)
        → 删除: delete_by_source(source_md5) → 清除所有 chunk + 磁盘归档文件
```

### 2.3 历史记录链路 (02_History.py)

```
进入页面 → 扫描 data/history/*.json
  → 搜索框过滤 destination
  → expander × N 条记录, 每个含 3 Tab:
      Tab 0: 最终方案 (Markdown + 下载)
      Tab 1: 对话历史 (消息列表 + RAG popover)
      Tab 2: 初始方案 (首次生成版本)
  → 删除: os.remove(filepath) → st.rerun()
```

---

## 3. 技术选型清册

| 层级 | 组件 | 版本 | 不可替代性 |
|---|---|---|---|
| **前端** | Streamlit | ≥1.28 | 纯 Python Web UI，零前端代码；`st.session_state` 跨 rerun 持久化；`st.cache_resource` 保障 ChromaDB 单例 |
| **LLM 底座** | 阿里百炼 ChatTongyi (qwen-max) | — | 中文理解能力业界领先，原生 Markdown 输出；通过 DashScope API 统一管理 LLM + Embedding |
| **图谱编排** | LangGraph | ≥0.2 | 唯一提供 `Send` API 的 Python 编排框架，支持确定性并行扇出；`Annotated[dict, reducer]` 解决并行写入冲突 |
| **LLM 抽象** | LangChain | ≥0.3 | 提供 `ChatTongyi` 集成、`RecursiveCharacterTextSplitter`、`@tool` 装饰器、`DashScopeEmbeddings` 封装 |
| **向量嵌入** | text-embedding-v3 (DashScope) | — | 阿里百炼最新嵌入模型，中文语义捕获能力强，1,536 维向量 |
| **向量库** | ChromaDB (PersistentClient) | ≥0.5 | 嵌入式向量数据库，零运维；`langchain-chroma` 桥接层提供 `similarity_search_with_relevance_scores` |
| **地图 API** | 高德 Maps REST API | — | 国内 POI / 天气 / 路线数据源；封装为 LangChain `@tool` 而非原生 MCP 协议 |
| **去重** | 自研三级流水线 | — | MD5 (全文本) → 字段匹配 (category+name 或 destination+days+title) → 语义 (余弦相似度 ≥0.95) |
| **测试** | pytest | ≥8.0 | 40 个单元测试，覆盖 config / state / agents / RAG / tools |
