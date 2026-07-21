# 🤖 多智能体编排与状态机策略

## 1. 智能体拓扑

**拓扑类型：Supervisor-Worker + LangGraph Send API 并行扇出**

```
                    ┌─────────────┐
                    │ orchestrator │  ← 主控 (验证 destination + 调度)
                    └──────┬──────┘
               ┌───────────┼───────────┐
               │  Send API 扇出 (3 路并行) │
               ▼           ▼           ▼
        ┌─────────┐ ┌──────────┐ ┌──────────┐
        │ weather  │ │attraction│ │  hotel   │  ← Worker (独立检索 + 推理)
        └────┬─────┘ └────┬─────┘ └────┬─────┘
               │           │           │
               └───────────┼───────────┘
                           ▼
                    ┌──────────────┐
                    │ synthesizer  │  ← 汇总 (汇聚 + 排版)
                    └──────┬───────┘
                           ▼
                          END
```

### 与经典 Supervisor-Worker 的差异

| 维度 | 经典 Supervisor-Worker | 本项目 |
|---|---|---|
| 调度次数 | Supervisor 反复调度 Worker，直到任务完成 | orchestrator 只验证一次就扇出，不监控结果 |
| 图结构 | 动态循环 (Agent Loop) | 静态 DAG (有向无环图) |
| 子 Agent 通信 | 通过 Supervisor 中转 | 完全不通信，独立执行 |
| 适用场景 | 复杂多步任务，需要动态调整 | 确定性流水线，每步职责明确 |

### 节点职责矩阵

| 节点 | 读取 State 字段 | 写入 State 字段 | 外部调用 |
|---|---|---|---|
| orchestrator | `destination` | `error_log` | 无（纯验证） |
| weather_agent | `destination`, `travel_date`, `preferences` | `weather_report`, `rag_refs` | RAG 案例库 (retrieve_cases) |
| attraction_agent | `destination`, `days`, `preferences` | `attractions`, `restaurants`, `rag_refs` | RAG 双库 (retrieve_both) |
| hotel_agent | `destination`, `preferences` | `hotels`, `rag_refs` | RAG 双库 (retrieve_both, category=hotel) |
| synthesizer | 全部子 Agent 输出 | `final_report`, `routes` | 无（纯 LLM 排版） |

**注意：** 高德 API 的 5 个 LangChain `@tool` 在子 Agent 中被 import（`tools = _get_tools()`），但当前实现未通过 `llm.bind_tools()` + Tool Calling 循环执行它们。Agent 实际依赖：(1) LLM 训练数据中的世界知识 + (2) RAG 检索的历史案例生成推荐。这是一个已完成封装但运行时未激活的能力模块。

---

## 2. LangGraph 状态机设计

### 2.1 图结构

```python
# 节点注册 (5 节点, 1 条件边)
graph = StateGraph(TravelPlanState)

graph.add_node("orchestrator", orchestrator_node)          # 主控
graph.add_node("weather_agent", weather_agent_node)        # 天气
graph.add_node("attraction_agent", attraction_agent_node)  # 景点餐饮
graph.add_node("hotel_agent", hotel_agent_node)            # 酒店
graph.add_node("synthesizer", synthesizer_node)            # 汇总

# 边 (静态拓扑)
graph.set_entry_point("orchestrator")
graph.add_conditional_edges("orchestrator", _continue_to_sub_agents)  # 唯一条件分支
graph.add_edge("weather_agent",     "synthesizer")   # 无条件 → 汇总
graph.add_edge("attraction_agent",  "synthesizer")   # 无条件 → 汇总
graph.add_edge("hotel_agent",       "synthesizer")   # 无条件 → 汇总
graph.add_edge("synthesizer",       END)              # 汇总 → 结束
```

### 2.2 条件边：Send API 扇出

```python
def _continue_to_sub_agents(state: TravelPlanState):
    """orchestrator 之后的唯一条件边"""
    if state.get("destination", "").strip() == "":
        return []  # 空 destination → 不发 Send, 图结束
    return [
        Send("weather_agent",     state),   # ← 传递完整 state (非 {})
        Send("attraction_agent",  state),   #    确保子 Agent 可读取全部输入字段
        Send("hotel_agent",       state),
    ]
```

**关键设计点：** `Send` 的第二个参数是完整 `state` 对象。若传递 `{}`，子 Agent 无法读取 `destination`、`days` 等输入字段。

### 2.3 空目的地回退路径

```
destination 为空
  → orchestrator_node 追加 error_log
  → _continue_to_sub_agents 返回 []
  → 图直接结束 (synthesizer 不执行)
  → 返回的 State 含 error_log, 无 final_report
```

### 2.4 依赖初始化 (模块级单例)

```python
# agents/graph.py — 模块级惰性单例
_llm       = None   # ChatTongyi(qwen-max)
_tools     = None   # [amap_weather, amap_poi_search, ...] (5 个 @tool)
_retriever = None   # DualRetriever
_settings  = None

def _init_dependencies(settings):
    """build_graph() 时调用一次，初始化全部单例"""
    global _llm, _tools, _retriever, _settings
    _llm       = ChatTongyi(...)
    _tools     = create_amap_tools(AmapClient(...))
    _retriever = DualRetriever(VectorStoreManager(...))

def _get_llm():       return _llm
def _get_tools():     return _tools
def _get_retriever(): return _retriever

# 子 Agent 中:
def weather_agent_node(state):
    from agents.graph import _get_llm, _get_retriever  # lazy import
    llm = _get_llm()
    ...
```

这解决了 `graph.py` → 子 Agent → `graph.py` 的循环依赖问题。

---

## 3. State 规约策略

### 3.1 TravelPlanState 定义

```python
from typing import TypedDict, Annotated

def _merge_dict(left: dict, right: dict) -> dict:
    """LangGraph reducer：并行分支写入 dict 时合并而非覆盖。"""
    return {**left, **right}

class TravelPlanState(TypedDict):
    # 用户输入 (只读)
    destination:    str
    travel_date:    str
    days:           int
    preferences:    str
    budget_total:   float

    # 子 Agent 输出 (各只有一个 writer)
    weather_report: str
    attractions:    list[dict]
    restaurants:    list[dict]
    hotels:         list[dict]
    routes:         list[dict]

    # 汇总输出
    final_report:   str
    rag_refs:       Annotated[dict, _merge_dict]  # ← reducer 保护
    error_log:      list[str]

    # 微调对话
    conversation:   list[dict]
    is_finalized:   bool
```

### 3.2 `INVALID_CONCURRENT_GRAPH_UPDATE` 与 Reducer 保护

**问题：** 3 个并行子 Agent 同时写入 `rag_refs` 字段，LangGraph 检测到并发写入同一 key 且无 reducer，抛出错误。

**解决：** 对 `rag_refs` 声明 `Annotated[dict, _merge_dict]`。

```
并行写入时序:
  weather_agent      → {"rag_refs": {"[RAG-C1]": "chunk1...", "[RAG-C2]": "chunk2..."}}
  attraction_agent   → {"rag_refs": {"[RAG-P1]": "chunk3...", "[RAG-C1]": "chunk4..."}}
  hotel_agent        → {"rag_refs": {"[RAG-P1]": "chunk5...", "[RAG-C2]": "chunk6..."}}

LangGraph 自动归约:
  _merge_dict({}, weather_rag)          → {C1, C2}
  _merge_dict({C1, C2}, attraction_rag) → {C1(覆盖!), C2, P1}
  _merge_dict({C1, C2, P1}, hotel_rag)  → {C1, C2(覆盖!), P1(覆盖!)}

最终 rag_refs: hotel 的 C2 覆盖 weather 的 C2; hotel 的 P1 覆盖 attraction 的 P1
```

**已知风险：** `_merge_dict` 是纯字典合并（`{**left, **right}`），当两个 Agent 使用相同的 key（如都从案例库检索产生 `[RAG-C1]`），后者会覆盖前者。当前 mitigation：key 空间通过前缀分隔（P vs C），但 attraction 和 hotel 共用 C 前缀，存在潜在覆盖。

**其他字段无需 reducer 的原因：** `weather_report`、`attractions`、`hotels` 等字段各自只有唯一 writer，不存在并发写入同一 key 的场景。

### 3.3 Agent 通信规约

```
✅ 允许:
  - 子 Agent 读取 State 中任何字段
  - 子 Agent 写入自己的输出字段
  - 通过 State 传递数据给下游 (synthesizer)

❌ 禁止:
  - 子 Agent 直接 import 另一个子 Agent
  - 子 Agent 修改其他 Agent 的输出字段
  - 子 Agent 修改用户输入字段
```

---

## 4. LangChain 抽象层应用

| 抽象 | 位置 | 使用方式 |
|---|---|---|
| **Chat Model** | `ChatTongyi` (langchain_community) | `llm.invoke(prompt)` 直接调用，未使用 LCEL Pipe |
| **Embedding** | `DashScopeEmbeddings` (langchain_community) | 由 langchain-chroma 自动调用 |
| **Text Splitter** | `RecursiveCharacterTextSplitter` (langchain_text_splitters) | chunk 切分核心组件 |
| **Document** | `langchain_core.documents.Document` | 全局数据交换单元，贯穿 data_loader → vector_store → retriever → Agent |
| **@tool 装饰器** | `langchain.tools.tool` | 5 个高德 API Tool 工厂，已封装但未在 Agent 推理循环中激活 |
| **LCEL** | 未使用 | 所有调用为 `llm.invoke()` 直接调用，未使用 `|` 管道语法 |
| **Prompt Template** | 未使用 | 所有 prompt 通过 f-string 拼接构造，未使用 `ChatPromptTemplate` |

### Tool Calling 现状

5 个高德 API `@tool` 已封装并被加载到模块级单例 `_tools` 中，但任何子 Agent 都未执行以下模式：

```python
# 标准 LangChain Tool Calling 模式 (未使用):
llm_with_tools = llm.bind_tools(tools)
response = llm_with_tools.invoke(prompt)
# ... Tool execution loop ...
```

当前 Agent 生成推荐时依赖：
1. **LLM 训练数据中的世界知识**（如"成都 3 天怎么玩"）
2. **RAG 检索的历史案例**（"别人去成都是怎么玩的"）

高德 API 的实时 POI / 天气 / 路线数据未进入推理管道。这是架构上已完成封装但运行时未激活的能力模块，也是后续优化的优先切入点。
