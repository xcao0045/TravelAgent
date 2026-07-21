"""景点餐饮 Agent — amap_poi_search + RAG 双库检索。"""
from langchain_core.messages import SystemMessage
from agents.state import TravelPlanState


_ATTRACTION_SYSTEM = """你是旅游规划专家。按以下规则工作：

## 工具使用规则（必须遵守）
1. 你有一个工具: amap_poi_search (搜索景点/餐厅)。
   参数: city (城市名称), keyword (搜索关键词), category (attraction/restaurant)
2. **你必须调用 amap_poi_search 至少一次**。禁止跳过工具调用直接生成结果。
   先搜索景点 (category="attraction")，再搜索餐厅 (category="restaurant")。
   每次搜索使用与用户偏好相关的关键词。

## API 调用规则
1. 每个工具最多调用 1 次，禁止重试。
2. 工具返回含义约定：
   - "❌" 开头 → 系统故障，跳过该数据源，用 RAG 案例库和你的知识补充
   - "⚠️" 开头 → 该词未匹配结果，换一个关键词重新搜索
   - 其他 → 正常数据，解析并整合
3. 无论 API 是否成功，都必须输出完整的推荐列表。

## 输出格式（严格 JSON）
输出纯 JSON（不要包裹在 ```json``` 中），格式:
{{"attractions": [...], "restaurants": [...], "sources": [...]}}
每个推荐必须包含以下字段:
- name: 真实景点/餐厅名称（优先使用高德POI搜索结果或RAG案例中的真实名称）
- address: 地址（如已知）
- rating: 评分字符串如"4.5"
- reason: 推荐理由（必须引用来源: [RAG-P1]/[RAG-C2] 或标注 [高德POI]）
- tags: 标签数组如 ["亲子", "园林"]

## 硬性约束（禁止违反）
- 禁止输出 "自行探索"、"根据兴趣选择" 等模糊内容
- 必须为每一天填充至少 2 个具体景点和 2 家具体餐厅
- 推荐数量: 景点 ≥ {days}*2 个, 餐厅 ≥ {days}+2 家
- 每个推荐的理由必须是具体的（含真实菜品名、游玩亮点、交通方式）

目的地: {destination}
天数: {days}天
用户偏好: {preferences}
偏好库匹配（用户评价+标签，请在推荐中引用 [RAG-P1]～[RAG-P{N_prefs}]）:
{prefs_context}
历史优秀案例参考（请在推荐中引用 [RAG-C1]～[RAG-C{N_cases}]）:
{cases_context}"""


def attraction_agent_node(state: TravelPlanState) -> dict:
    from agents.graph import _get_llm, _get_tools, _get_retriever, _execute_tool_calls

    llm = _get_llm()
    all_tools = _get_tools()
    retriever = _get_retriever()

    destination = state["destination"]
    days = state["days"]
    preferences = state.get("preferences", "")

    # RAG 双检索
    rag_results = retriever.retrieve_both(
        f"{destination} {preferences} 景点 美食 餐厅",
        preferences_category=None,
        k_prefs=5,
        k_cases=3,
    )
    rag = {}
    # 后置过滤: attraction 只需要 attraction + restaurant 类型的偏好
    prefs_lines = []
    prefs_count = 0
    for i, d in enumerate(rag_results["preferences"]):
        cat = d.metadata.get("category", "")
        if cat == "hotel":
            continue  # 酒店评价不给景点Agent
        prefs_count += 1
        rid = f"[RAG-P{prefs_count}]"
        rag[rid] = d.page_content[:1200]
        prefs_lines.append(
            f"{rid} {cat}·{d.metadata.get('name','')} "
            f"[标签:{d.metadata.get('tags',[])}]: {d.page_content[:1200]}"
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

    agent_tools = [t for t in all_tools if t.name in ("amap_poi_search",)]

    system_prompt = _ATTRACTION_SYSTEM.format(
        destination=destination,
        days=days,
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

    for _round in range(2):  # 最多 2 轮
        if hasattr(response, "tool_calls") and response.tool_calls:
            tool_messages = _execute_tool_calls(response, agent_tools)
            if not tool_messages:
                break
            messages.extend([response, *tool_messages])
            response = llm_with_tools.invoke(messages)
        else:
            break

    # 解析 JSON
    import json as _json
    import re
    try:
        json_match = re.search(r"\{[\s\S]*\}", response.content)
        data = _json.loads(json_match.group()) if json_match else {}
    except _json.JSONDecodeError:
        data = {}

    attractions = data.get("attractions", [])
    restaurants = data.get("restaurants", [])

    # 诊断日志
    print(f"[attraction_agent] RAG prefs={prefs_count} cases={cases_count} "
          f"attractions={len(attractions)} restaurants={len(restaurants)}")

    return {
        "attractions": attractions,
        "restaurants": restaurants,
        "rag_refs": rag,
    }
