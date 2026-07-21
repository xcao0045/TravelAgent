# 🔍 RAG 检索增强生成 — 全链路策略

## 管道总览

```
文件上传 (CSV/JSON/MD/TXT)
  → 格式解析 (data_loader.py)
  → 三级去重 (dedup.py: MD5 → 字段匹配 → 语义)
  → Chunk 切分 (RecursiveCharacterTextSplitter, chunk=500, overlap=50)
  → 向量嵌入 (text-embedding-v3, 1536 维)
  → ChromaDB 双库存储 (user_preferences + travel_cases)
  → 检索召回 (余弦相似度, threshold=0.45)
  → Prompt 拼接 (rid 标注, [RAG-P1..N] / [RAG-C1..N])
  → LLM 生成 (qwen-max, 要求在理由中引用 rid)
```

---

## 1. 文档预处理

**入口：** `rag/data_loader.py` → `load_file_to_docs(file_path, collection_type)` 按扩展名调度。

| 格式 | 解析器 | 目标库 | Document 构造 |
|---|---|---|---|
| CSV | `parse_csv_to_docs` | preferences | 每行 → `Document(text=text, meta={category, name, tags, rating})` |
| JSON | `parse_json_to_docs` | preferences / cases | 按 `collection_type` 分支构建不同 metadata schema |
| MD / TXT | `parse_md_to_doc` | cases | 全文 → 单 Document，title 从首行 `#` 标题或文件名提取 |

**清洗策略：** 不做 NLP 清洗（不去停用词、不纠错）。`text-embedding-v3` 面向原始文本优化，且 LLM 在 prompt 中会二次理解，清洗反而可能丢失语义信号。

**归档策略：** 上传文件写一份到 `data/knowledge_base/{type}/`。Document 的 `source_file` 元数据指向此路径，删除时同步清理。

---

## 2. Chunk 切分

**切分器：** `langchain_text_splitters.RecursiveCharacterTextSplitter`

```python
RecursiveCharacterTextSplitter(
    chunk_size=500,         # 配置项 chunk_size (config.py)
    chunk_overlap=50,       # 配置项 chunk_overlap, 即 10% overlap
    separators=["\n\n", "\n", "。", "，", " ", ""],
)
```

**策略分析：**

| 维度 | 取值 | 理由 |
|---|---|---|
| chunk_size | 500 字符 (中文约 250-300 token) | 小 chunk 策略，适合偏好库中单条评价的精确匹配 |
| chunk_overlap | 50 (10%) | 偏低但可接受 — Markdown 方案的 `##` 标题天然提供切分点，`separators` 优先在段落和句子边界断开 |
| 中文 separators | `。` `，` 优先级高于空格 | 确保在句子边界切分，避免字词中间截断 |

**source_md5 跨 chunk 追踪：**
```python
# 切分前: 对原始 Document 的完整 page_content 取 MD5
doc.metadata["source_md5"] = hashlib.md5(doc.page_content.encode()).hexdigest()
# 切分后: 所有子 chunk 继承同一 source_md5 → 可归组、可批量删除
```

---

## 3. 文本向量嵌入

**实现：** `rag/embedding.py`

| 参数 | 值 |
|---|---|
| 模型 | 阿里百炼 text-embedding-v3 |
| 维度 | 1,536 |
| 调用方式 | `langchain-chroma` 在 `collection.add_documents()` 时自动调用 `embedding_function.embed_documents()` |
| API Key | 与 LLM 共享 `BAILIAN_API_KEY` |

**实证调优（来自测试）：** 该模型在中文旅行领域的实际余弦相似度落在 **0.55–0.67** 范围，低于常见默认阈值 0.7。系统将 `similarity_threshold` 下调至 **0.45**，知识库搜索使用 **0.3**。

---

## 4. 向量数据库存储

**实现：** `rag/vector_store.py` → `VectorStoreManager`

```python
PersistentClient(
    path=persist_dir,       # 默认 ./storage
    settings=ChromaSettings(anonymized_telemetry=False),
)
```

**双库结构：**

| 集合 | Chroma 标识 | 内容 | 典型 chunk 数/文档 |
|---|---|---|---|
| 用户偏好库 | `user_preferences` | 酒店/景点/餐厅评价 | 1–3 chunk |
| 旅行案例库 | `travel_cases` | 优质旅行方案全文 | 10–20 chunk |

**索引类型：** ChromaDB 默认使用 **HNSW** (Hierarchical Navigable Small World) 近似最近邻索引。

**关键工程细节：**

1. **元数据清洗 (`_sanitize_metadata`)：** ChromaDB 拒绝值为空列表的元数据字段（如 `tags: []`），写入前必须剔除
2. **单例管理：** 知识库页通过 `st.cache_resource` 确保 `VectorStoreManager` 只初始化一次
3. **source_md5 分组：** `list_documents()` 按 `source_md5` 聚合 chunk，返回去重后的文档摘要列表

---

## 5. 检索与召回

**实现：** `rag/retriever.py` → `DualRetriever`

### 检索策略：纯向量检索（当前无混合检索）

```
查询构造 (在各子 Agent 中):
  weather_agent:     f"{destination} {travel_date} 天气 出行准备"
  attraction_agent:  f"{destination} {preferences} 景点 美食 餐厅"
  hotel_agent:       f"{destination} {preferences} 酒店 住宿"
  KB 语义搜索:        用户输入关键词

检索执行:
  ChromaDB.similarity_search_with_relevance_scores(query, k=k)
    → 返回 [(Document, score), ...]
    → 过滤: score >= similarity_threshold (0.45 for Agent, 0.3 for KB search)
    → 无结果时 rag_refs 为空 dict
```

### 当前局限

- **无混合检索：** 未引入 BM25 或关键词倒排索引。纯向量检索在短地名查询（如"苏州"）时，与长文档的语义向量相似度偏低，可能漏召回
- **无 Category 过滤（attraction agent）：** `attraction_agent` 调用 `retrieve_both` 时 `preferences_category=None`，偏好库检索不过滤品类，可能将酒店评价错误喂给景点推荐 Agent

---

## 6. 召回结果融合与重排

**当前状态：无显式融合，无 Reranker。**

三个子 Agent 各自独立检索，各自在自己的 prompt 中消费检索结果，**不跨 Agent 融合结果**。

```
检索结果生命周期:

子 Agent prompt:  chunks 按 ChromaDB 默认得分排序 → LLM 自行决定引用
LangGraph 层:     _merge_dict 合并 3 个 Agent 的 rag_refs
                  (key 不冲突: C/P 前缀区分, 但 attraction 和 hotel 共用 C 前缀存在潜在覆盖)
前端 Render:      st.expander("RAG 引用来源") → 按 rid 遍历 → text[:300]
```

**未引入的优化项：**
- Cross-Encoder Reranker（如 bge-reranker-v2）对检索结果精排
- LLM 在长 prompt 中可能忽略低排名 chunk，有效召回窗口受限于 LLM 上下文注意力分布
- 前端展示是原始检索顺序，不是 LLM 实际引用的重排顺序

---

## 7. 最终提示词上下文集成

### rag_refs 数据结构

```python
# 每个子 Agent 内部构建:
rag = {}
for i, d in enumerate(rag_results["preferences"]):
    rid = f"[RAG-P{i+1}]"
    rag[rid] = d.page_content[:300]   # 偏好库截取 300 字符
for i, d in enumerate(rag_results["cases"]):
    rid = f"[RAG-C{i+1}]"
    rag[rid] = d.page_content[:400]   # 案例库截取 400 字符
```

### Prompt 集成方式

```
偏好库匹配（用户评价标签）:
[RAG-P1] hotel·希尔顿 [标签:亲子,安静,服务好]: 房间宽敞...
[RAG-P2] restaurant·海底捞 [标签:亲子,服务好]: 服务贴心...

历史优秀案例参考:
[RAG-C1] 目的地:成都 天数:3天 成都3日亲子游完整方案...

请直接推荐，并在理由中引用RAG来源编号如 [RAG-P1][RAG-C2]
sources字段: 列出本次推荐实际引用的RAG来源编号列表
```

### 前端展示

| 场景 | 展示方式 |
|---|---|
| **初始报告** | `st.expander("📖 RAG 引用来源（N 条）")` 展示全部 rag_refs |
| **对话消息** | 每条 assistant 消息携带 `sources` 字段，通过 `st.popover("📖 引用来源")` 展示 |
| **无匹配** | 显示 "💡 当前知识库未匹配到相关内容" |

### 关键设计约束

**RAG 引用是 chunk 而非原文。** 每条 `[RAG-P1]` 对应的文本是单个 500-char chunk 的前 300 或 400 字符，不是完整原始文档。用户看到的是碎片化的 chunk 摘要。如需完整原文，需通过 `source_md5` 回查原始归档文件。

---

## 8. 去重流水线

**实现：** `rag/dedup.py` → `dedup_pipeline()`

```python
# 三级去重，可选开关
options = {"md5": True, "field": False, "semantic": False}

# 第1关: MD5 精确匹配
#   优先比对 source_md5（跨 chunk 的原文哈希），兼容 chunk 切分后的去重
#   回退方案: 比对文本的 MD5
#
# 第2关: 字段匹配 (默认关闭)
#   preferences: category + name 相同 → 疑似重复
#   cases:       destination + days + title 相同 → 疑似重复
#
# 第3关: 语义去重 (默认关闭)
#   similarity_search_with_relevance_scores(text, k=5, threshold=0.95)
```

| 关卡 | 默认状态 | 阈值 | 适用场景 |
|---|---|---|---|
| MD5 | ✅ 开启 | 精确匹配 | 完全相同的文档重复上传 |
| 字段 | ❌ 关闭 | category+name 或 dest+days+title | 同一实体的不同评价版本 |
| 语义 | ❌ 关闭 | 0.95 | 内容高度相似但表述不同的文档 |
