# Multi-Agent 智能旅行规划 Web 系统 — 设计文档

> **日期**: 2026-06-24
> **状态**: 已确认
> **技术栈**: Streamlit + Python + 阿里百炼千问 (LangChain) + ChromaDB + 高德地图 API

---

## 一、系统整体架构

```
┌─────────────────────────────────────────────────┐
│                  Streamlit UI                    │
│  ┌──────────┐  ┌──────────────┐  ┌───────────┐  │
│  │ 旅行规划  │  │ 知识库管理页  │  │ 历史记录页  │  │
│  │ Home.py  │  │             │  │           │  │
│  └─────┬────┘  └──────────────┘  └───────────┘  │
└────────┼────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────┐
│              LangGraph 编排层                     │
│                                                 │
│   ┌──────────┐                                  │
│   │ 主控Agent │──── 根据状态决定下一节点           │
│   └────┬─────┘                                  │
│        │                                         │
│   ┌────┴────┬──────────┬──────────┐              │
│   ▼         ▼          ▼          ▼              │
│ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────────┐        │
│ │天气   │ │景点   │ │酒店   │ │汇总报告   │        │
│ │Agent  │ │餐饮   │ │Agent  │ │Agent     │        │
│ │       │ │Agent  │ │       │ │          │        │
│ └──┬───┘ └──┬───┘ └──┬───┘ └────┬─────┘        │
│    │        │        │          │                │
└────┼────────┼────────┼──────────┼────────────────┘
     │        │        │          │
     └────────┴────────┴──────────┘
              │
     ┌────────┴────────┐
     ▼                 ▼
┌──────────┐   ┌──────────────┐
│ ChromaDB │   │ 高德 LangChain │
│ 双向量库  │   │ Tool (5个)    │
└──────────┘   └──────────────┘
```

### 分层职责

| 层 | 职责 |
|------|------|
| **UI 层** (Streamlit) | 三个页面：旅行规划主页面、知识库管理页、历史记录页 |
| **编排层** (LangGraph) | 主控 Agent 定义状态图，调度子 Agent 并行执行，汇总输出 |
| **子 Agent 层** | 天气、景点餐饮、酒店 3 个专业 Agent，各自负责搜索 + 匹配 + RAG 检索 |
| **能力底座层** | 高德 LangChain Tool (5个) + ChromaDB 双向量库 |

### 关键设计决策

- 高德 API 当前封装为 LangChain Tool，未来如需多系统共用再重构为标准 MCP Server
- 两个 ChromaDB Collection 独立管理：`user_preferences` 和 `travel_cases`
- Streamlit session_state 持有 LangGraph 运行实例，一次用户请求 = 一次 Graph run

---

## 二、LangGraph 状态机设计

### State Schema

```python
class TravelPlanState(TypedDict):
    # === 用户输入 ===
    destination: str          # 目的地
    travel_date: str          # 出发日期
    days: int                 # 天数
    preferences: str          # 偏好描述
    budget_total: float       # 总预算

    # === 子 Agent 输出 ===
    weather_report: str
    attractions: list[dict]
    restaurants: list[dict]
    hotels: list[dict]
    routes: list[dict]

    # === 汇总输出 ===
    final_report: str
    error_log: list[str]

    # === 微调对话 ===
    conversation: list[dict]  # 第4步聊天记录
    is_finalized: bool        # 用户是否已确认
```

### 状态图结构

```
START → 主控Agent → [并行] 天气Agent + 景点餐饮Agent + 酒店Agent → 汇总Agent → END
```

### 各节点职责

| 节点 | 动作 | 工具调用 |
|------|------|------|
| **主控Agent** | 解析输入，拆分任务 | 无 |
| **天气Agent** | 调用高德天气 API，参考 RAG 案例库天气应对策略 | 高德天气 Tool + RAG 案例库 |
| **景点餐饮Agent** | 高德 POI 搜索 + RAG 偏好库标签匹配 + RAG 案例库参考 | 高德 POI Tool + RAG 偏好库 + RAG 案例库 |
| **酒店Agent** | 高德 POI 搜索 + RAG 偏好库标签匹配 + RAG 案例库参考 | 高德 POI Tool + RAG 偏好库 + RAG 案例库 |
| **汇总Agent** | 整合子 Agent 输出，计算预算，生成 Markdown | 高德路线规划 Tool |

### 并行策略

- 天气、景点、酒店 3 个 Agent 互不依赖，通过 LangGraph `Send` API 扇出并行执行
- 汇总 Agent 等待前三者全部完成后触发（条件边 `check_all_done`）

### RAG 调用矩阵

三个子 Agent 均可调用两个 RAG 库：

| Agent | 偏好库用途 | 案例库用途 |
|-------|-----------|-----------|
| **天气Agent** | 查询该目的地/季节用户评价中提到的天气注意事项 | 查询历史案例中同季节同目的地的天气应对策略 |
| **景点餐饮Agent** | 按标签筛选匹配偏好的景点+餐厅 | 检索相似天数/目的地的历史案例，参考景点路线和餐厅选择 |
| **酒店Agent** | 按标签筛选匹配偏好的酒店 | 检索历史案例中同目的地的高分酒店选择 |

每个子 Agent 内部执行两阶段：先 RAG 检索获取约束条件和参考经验 → 再高德查询真实数据后筛选排序。

---

## 三、RAG 知识库设计

### ChromaDB 双 Collection

**偏好库 `user_preferences`**

```python
metadata = {
    "category": "hotel" | "restaurant" | "attraction",
    "name": "XX酒店",
    "tags": ["亲子", "隔音好", "卫生好"],
    "rating": 4.5,
    "source": "user_upload",
    "created_at": "2026-07-15"
}
# 文本内容: 用户评价原文 → embedding
```

**案例库 `travel_cases`**

```python
metadata = {
    "destination": "成都",
    "days": 3,
    "season": "秋季",
    "budget_range": "3000-5000",
    "tags": ["美食", "休闲"],
    "rating": 4.8,
    "created_at": "2025-10-01"
}
# 文本内容: 完整旅行方案 Markdown → embedding
```

### 文档切分策略

入库时使用 `RecursiveCharacterTextSplitter` 切分文档：

| 参数 | 值 |
|------|------|
| chunk_size | 500 |
| chunk_overlap | 50 (10%) |
| 分隔符 | `\n\n` → `\n` → `。` → `，` → ` ` → `` |

### Embedding 与检索策略

| 项 | 选择 |
|----|------|
| Embedding 模型 | 阿里百炼 `text-embedding-v3` (1024维) |
| 检索方式 | 余弦相似度 + 阈值过滤 |
| 检索 Top-K | 偏好库 K=5，案例库 K=3 |
| 相似度阈值 | 0.7（低于此分数的结果被丢弃，实际返回数量可能小于 K） |

### RAG 引用格式

Agent 将 RAG 检索结果以编号标记注入 prompt：
- `[RAG-P1]` ... `[RAG-P5]` — 偏好库来源
- `[RAG-C1]` ... `[RAG-C3]` — 案例库来源

LLM 生成的推荐中须引用这些编号，并在输出的 JSON 中包含 `sources` 字段列出实际引用的来源。

### 数据录入（两个库均支持手动 + 文件上传）

| 方式 | 偏好库 | 案例库 |
|------|------|------|
| **手动表单** | category + name + tags + rating + 评价文本 | destination + days + season + budget + tags + rating + 方案Markdown |
| **文件上传** | CSV / JSON | JSON / Markdown / TXT / CSV(元数据) |

### 去重策略（三级防线）

```
第1关：MD5 快速去重
  → 命中 → 跳过，提示"已存在（完全一致）"
  → 未命中 → 进入第2关

第2关：关键字段匹配（可选开启）
  → 偏好库：category + name + text 相同
  → 案例库：destination + days + title 相同

第3关：语义近重复检测（可选开启）
  → embedding 余弦相似度 ≥ 0.95
  → 列入"疑似重复"，用户确认处理
```

### 检索触发流程

```
1. 子Agent收到任务
2. 同时发起双检索：
   - 偏好库.query(偏好关键词, filter={category}, k=5)
   - 案例库.query(目的地+天数+偏好, k=3)
3. 偏好库结果 → 标签匹配形成筛选条件
4. 案例库结果 → 历史方案参考
5. 结合高德POI结果综合排序 → 输出推荐
```

---

## 四、高德服务层设计

### 当前方案：LangChain Tool 封装

直接封装为 5 个 `@tool`，不引入 MCP 协议层开销。未来如需多系统共用再重构为标准 MCP Server。

### 5 个 Tool 定义

| Tool | 功能 | 对应高德 API |
|------|------|-------------|
| `amap_weather(city, date)` | 查询指定城市日期天气 | `weatherInfo` |
| `amap_poi_search(city, keyword, category, radius)` | 搜索 POI（酒店/餐厅/景点） | `place/text` |
| `amap_route_plan(origin, destination, mode)` | 两点间路线规划 | `direction/transit` 等 |
| `amap_multi_route(waypoints, mode)` | 多点串联路线 | `direction/driving` |
| `amap_geo_code(address)` | 地址→经纬度 | `geocode/geo` |

### 各 Agent 工具映射

| Agent | 使用 Tool |
|-------|-----------|
| 天气Agent | `amap_weather` |
| 景点餐饮Agent | `amap_poi_search` + `amap_multi_route` |
| 酒店Agent | `amap_poi_search` |
| 汇总Agent | `amap_route_plan` + `amap_geo_code` |

### 底层实现

```python
class AmapClient:
    BASE_URL = "https://restapi.amap.com/v3"

    def __init__(self, api_key: str):
        self.api_key = api_key

    def weather(self, city: str) -> dict: ...
    def poi_search(self, keywords: str, types: str, city: str) -> dict: ...
    def direction(self, origin: str, destination: str, type: str) -> dict: ...
```

---

## 五、Streamlit UI 设计

### 页面结构

```
Home.py                      # 旅行规划主页面
pages/
  01_Knowledge_Base.py       # 知识库管理页
  02_History.py              # 历史记录页
```

> Home.py 必须放在根目录，这是 Streamlit 多页面路由的技术要求。

### 页面 1：旅行规划主页面

**四步交互流程：**

1. **输入需求** — 目的地、日期、天数、预算区间、偏好描述 → 点击"开始规划"
2. **AI 执行过程** — 实时显示 LangGraph 节点执行状态（天气 → 景点 → 酒店 → 汇总）
3. **旅行方案** — 结构化 Markdown 报告渲染，包含天气、酒店、景点、美食、交通、预算明细。支持复制和下载 Markdown。
4. **微调优化** — 聊天框供用户反复提修改意见，AI 增量更新方案（如"换酒店"只重跑酒店Agent），直到用户点击"确认最终方案"保存。

### 页面 2：知识库管理页

- 选择目标库（偏好库/案例库）
- 手动录入表单（字段随库类型变化）
- 文件上传（CSV/JSON/MD/TXT）
- 去重策略开关（MD5必开/语义可选）
- 已有数据列表（搜索、筛选、编辑、删除、分页）
- 疑似重复维护面板

### 页面 3：历史记录页

- 所有历史会话列表（时间、目的地、天数、预算、对话轮数、状态）
- 展开查看完整会话：按版本切换查看各轮方案 Markdown + 对话记录
- 下载最终 Markdown / 删除记录

### 侧边栏（全局）

- LLM 模型选择（qwen-max / qwen-plus 等）
- 系统状态指示（ChromaDB / 高德API / 百炼API 连接状态）
- 页面导航

### API Key 管理

- 通过 `.env` 文件管理，不暴露在 UI
- `BAILIAN_API_KEY` + `AMAP_API_KEY`
- 启动时检测，缺失则在首页显示引导提示

---

## 六、项目目录结构

```
TravelAgent/
├── .env.example
├── .gitignore
├── requirements.txt
├── Home.py                         # Streamlit 首页 → 旅行规划
├── config.py                       # 配置中心

├── pages/
│   ├── 01_Knowledge_Base.py
│   └── 02_History.py

├── agents/                         # LangGraph Agent 层
│   ├── __init__.py
│   ├── state.py                    # TravelPlanState
│   ├── orchestrator.py             # 主控Agent
│   ├── weather_agent.py
│   ├── attraction_agent.py
│   ├── hotel_agent.py
│   ├── synthesizer.py              # 汇总Agent
│   └── graph.py                    # 状态图构建

├── tools/                          # LangChain Tool 层
│   ├── __init__.py
│   └── amap_tools.py               # 5个@tool + AmapClient

├── rag/                            # RAG 知识库层
│   ├── __init__.py
│   ├── embedding.py                # 百炼embedding
│   ├── vector_store.py             # ChromaDB管理
│   ├── retriever.py                # 检索统一接口
│   ├── dedup.py                    # MD5 + 语义去重
│   └── data_loader.py              # CSV/JSON/MD/TXT解析

├── storage/
│   ├── chroma_preferences/
│   └── chroma_cases/

├── data/
│   └── history/                    # 完整会话JSON（含对话历史+方案版本演进）

└── docs/
    └── superpowers/
        └── specs/
            └── 2026-06-24-travel-agent-design.md
```

### 分层依赖关系

```
Home.py, pages/*.py   (UI层)
       │
       ▼
agents/graph.py        (编排层)
       │
  ┌────┼────┬──────────┐
  ▼    ▼    ▼          ▼
agents/*  tools/       rag/
```

子 Agent 不直接 import 彼此，仅通过 LangGraph State 通信。

### 历史存储格式

每次会话保存为完整 JSON：

```json
{
    "session_id": "20260715_143022",
    "created_at": "2026-07-15T14:30:22",
    "status": "confirmed",
    "input": { "destination": "成都", "travel_date": "2026-07-20", ... },
    "initial_report": "# 初始方案 Markdown\n...",
    "conversation": [
        {
            "role": "user",
            "content": "我想把第二天酒店换成离春熙路更近的",
            "timestamp": "14:31:05"
        },
        {
            "role": "assistant",
            "content": "好的，已更新...",
            "agent_trace": ["hotel_agent: re-run"],
            "report_snapshot": "# 更新后方案\n...",
            "timestamp": "14:31:18"
        }
    ],
    "final_report": "# 最终方案 Markdown\n...",
    "agent_trace": [...]
}
```

---

## 七、错误处理与边界情况

### 分层错误策略

| 层 | 策略 |
|----|------|
| **UI 层** | 所有错误友好展示，不崩溃，不暴露技术细节 |
| **LangGraph 编排层** | 子Agent失败不影响其他Agent，error_log 收集后继续 |
| **子Agent层** | 工具调用失败 → 降级处理（RAG兜底 / 标注"暂不可用"） |
| **能力底座层** | 网络异常/超时重试1次，仍失败返回明确错误码 |

### 关键场景

| 场景 | 处理 |
|------|------|
| 高德 API 超时 | 重试1次(间隔2s)，仍失败 → 降级结果，标注"⚠️ 数据暂不可用" |
| 高德 POI 无结果 | Agent 转用 RAG 案例库历史推荐兜底 |
| 百炼 LLM 失败 | 重试1次 → 回退至 qwen-plus → 全失败则提示用户稍后重试 |
| ChromaDB 为空 | RAG 返回空，Agent 仅依赖高德 POI + LLM 自身知识 |
| 用户输入模糊 | 主控Agent 要求 LLM 识别并追问，不直接进入子Agent |
| 某子Agent失败 | error_log 记录，其余继续；汇总Agent 标注"⚠️ XX推荐暂不可用" |
| 整体流程失败 | 显示部分结果 + 错误说明，不展示空白或崩溃 |

### 微调对话（第4步）特殊处理

| 场景 | 处理 |
|------|------|
| 部分修改（如"换酒店"） | 只重跑对应子Agent，复用其他 Agent 缓存 |
| 推翻重来（如"换目的地"） | 清空 State，完整重跑 |
| 对话超上下文窗口 | 保留最近10轮，超出部分摘要后注入 |
| 用户未确认就离开 | 会话保存为 status=abandoned，可在历史页找回 |
