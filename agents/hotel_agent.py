"""酒店 Agent — amap_poi_search + amap_geo_code + RAG 双库检索。"""
from langchain_core.messages import SystemMessage
from agents.state import TravelPlanState


_HOTEL_SYSTEM = """你是酒店推荐专家。按以下规则工作：

## 工具使用规则（必须遵守）
1. 可用工具:
   - amap_poi_search: 搜索酒店 (category="hotel")
   - amap_geo_code: 地址→经纬度坐标
2. **你必须调用 amap_poi_search 至少一次**。禁止跳过工具调用直接生成结果。
   搜索后可用 amap_geo_code 查询酒店坐标以便后续路线计算。

## API 调用规则
1. 每个工具最多调用 1 次，禁止重试。
2. 工具返回含义约定：
   - "❌" 开头 → 系统故障，跳过该数据源，用 RAG 偏好库和你的知识补充
   - "⚠️" 开头 → 该词未匹配结果，换一个关键词重新搜索
   - 其他 → 正常数据，解析并整合
3. 无论 API 是否成功，都必须输出酒店推荐列表。

## 输出格式（严格 JSON）
输出纯 JSON（不要包裹在 ```json``` 中），格式:
{{"hotels": [...], "sources": [...]}}
每个推荐必须包含以下字段:
- name: 真实酒店名称（优先使用高德POI或RAG案例中的真实名称）
- address: 地址
- rating: 评分字符串如"4.7"
- price_range: 价格区间如"500-800"
- reason: 推荐理由（必须引用来源: [RAG-P1] 或标注 [高德POI]）
- matched_tags: 匹配用户偏好的标签数组

## 硬性约束（禁止违反）
- 禁止输出 "自行搜索"、"根据预算选择" 等模糊内容
- **必须推荐至少 3 家不同档次的酒店**，每家含具体理由（真实设施描述、距景区距离）
- **多样性要求**：3 家酒店必须覆盖不同档次（如豪华五星1家 + 舒适四星1家 + 经济/民宿1家）
  且尽量覆盖不同区域（如姑苏区、工业园区、太湖周边各1家）
- 每家酒店的 reason 必须引用具体的 RAG 来源编号或高德 POI 数据

## 降级规则（RAG 数据不足时）
如果知识库匹配到的酒店少于 3 家，你必须：
1. 优先输出知识库中已有的酒店（标注 [RAG-P1] 等来源）
2. **用你的训练知识补全剩余部分**，推荐苏州真实存在的知名酒店
3. 补全的酒店标注为 **[基于通用知识推荐]**
4. 绝对禁止只输出 1 家酒店或输出"暂无其他推荐"

目的地: {destination}
用户偏好: {preferences}
偏好库匹配（用户评价+标签，请在推荐中引用 [RAG-P1]～[RAG-P{N_prefs}]）:
{prefs_context}
历史优秀案例参考（请在推荐中引用 [RAG-C1]～[RAG-C{N_cases}]）:
{cases_context}"""


def hotel_agent_node(state: TravelPlanState) -> dict:
    from agents.graph import _get_llm, _get_tools, _get_retriever, _execute_tool_calls

    llm = _get_llm()
    all_tools = _get_tools()
    retriever = _get_retriever()

    destination = state["destination"]
    preferences = state.get("preferences", "")

    # RAG 双检索
    rag_results = retriever.retrieve_both(
        f"{destination} {preferences} 酒店 住宿",
        preferences_category="hotel",
        k_prefs=5,
        k_cases=3,
    )
    rag = {}
    prefs_lines = []
    prefs_count = 0
    for i, d in enumerate(rag_results["preferences"]):
        prefs_count += 1
        rid = f"[RAG-P{prefs_count}]"
        rag[rid] = d.page_content[:1200]
        prefs_lines.append(
            f"{rid} {d.metadata.get('name','')}: {d.page_content[:1200]} "
            f"[标签: {d.metadata.get('tags',[])}]"
        )
    cases_lines = []
    cases_count = 0
    for i, d in enumerate(rag_results["cases"]):
        cases_count += 1
        rid = f"[RAG-C{cases_count}]"
        rag[rid] = d.page_content[:1200]
        cases_lines.append(f"{rid} {d.page_content[:1200]}")

    prefs_context = "\n".join(prefs_lines) if prefs_lines else "(无相关偏好数据)"
    cases_context = "\n".join(cases_lines) if cases_lines else "(无相关历史案例)"

    agent_tools = [t for t in all_tools if t.name in ("amap_poi_search", "amap_geo_code")]

    system_prompt = _HOTEL_SYSTEM.format(
        destination=destination,
        preferences=preferences,
        N_prefs=prefs_count,
        N_cases=cases_count,
        prefs_context=prefs_context,
        cases_context=cases_context,
    )

    # ── Tool Calling (最多 2 轮) ──
    llm_with_tools = llm.bind_tools(agent_tools)
    messages = [SystemMessage(content=system_prompt)]
    response = llm_with_tools.invoke(messages)

    for _round in range(2):
        if hasattr(response, "tool_calls") and response.tool_calls:
            tool_messages = _execute_tool_calls(response, agent_tools)
            if not tool_messages:
                break
            messages.extend([response, *tool_messages])
            response = llm_with_tools.invoke(messages)
        else:
            break

    import json as _json
    import re
    try:
        json_match = re.search(r"\{[\s\S]*\}", response.content)
        data = _json.loads(json_match.group()) if json_match else {}
    except _json.JSONDecodeError:
        data = {}

    hotels = data.get("hotels", [])
    print(f"[hotel_agent] RAG prefs={prefs_count} cases={cases_count} "
          f"hotels={len(hotels)}")

    return {"hotels": hotels, "rag_refs": rag}
