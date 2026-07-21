"""酒店 Agent — amap_poi_search + amap_geo_code + RAG 双库检索。"""
from langchain_core.messages import SystemMessage
from agents.state import TravelPlanState


_HOTEL_SYSTEM = """你是酒店推荐专家。按以下规则工作：

## 工具使用规则
1. 可用工具:
   - amap_poi_search: 搜索酒店 (category="hotel")
   - amap_geo_code: 地址→经纬度坐标 (可用于查询酒店坐标方便后续路线计算)
2. 先调用 amap_poi_search 搜索酒店，若需坐标可再调用 amap_geo_code。

## API 调用规则
1. 每个工具最多调用 1 次，禁止重试。
2. 工具返回含义约定：
   - "❌" 开头 → 系统故障，跳过该数据源，用 RAG 偏好库和你的知识补充
   - "⚠️" 开头 → 该词未匹配结果，用 RAG 偏好库补充替代推荐
   - 其他 → 正常数据，解析并整合
3. 无论 API 是否成功，都必须输出酒店推荐列表。

## 输出格式
输出 JSON 格式: {{"hotels": [...], "sources": [...]}}
每个推荐含: name, address, rating, price_range, reason(在理由中引用RAG来源编号), matched_tags
sources字段: 列出实际引用的RAG来源编号列表

目的地: {destination}
用户偏好: {preferences}
偏好库匹配（用户评价+标签）:
{prefs_context}
历史优秀案例参考:
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
    for i, d in enumerate(rag_results["preferences"]):
        rid = f"[RAG-P{i+1}]"
        rag[rid] = d.page_content[:300]
        prefs_lines.append(f"{rid} {d.metadata.get('name','')}: {d.page_content[:300]} [标签: {d.metadata.get('tags',[])}]")
    cases_lines = []
    for i, d in enumerate(rag_results["cases"]):
        rid = f"[RAG-C{i+1}]"
        rag[rid] = d.page_content[:400]
        cases_lines.append(f"{rid} {d.page_content[:400]}")

    prefs_context = "\n".join(prefs_lines) if prefs_lines else "(无相关偏好数据)"
    cases_context = "\n".join(cases_lines) if cases_lines else "(无相关历史案例)"

    agent_tools = [t for t in all_tools if t.name in ("amap_poi_search", "amap_geo_code")]

    system_prompt = _HOTEL_SYSTEM.format(
        destination=destination,
        preferences=preferences,
        prefs_context=prefs_context,
        cases_context=cases_context,
    )

    llm_with_tools = llm.bind_tools(agent_tools)
    messages = [SystemMessage(content=system_prompt)]
    response = llm_with_tools.invoke(messages)

    # Tool Calling loop
    if hasattr(response, "tool_calls") and response.tool_calls:
        tool_messages = _execute_tool_calls(response, agent_tools)
        messages.extend([response, *tool_messages])
        final = llm_with_tools.invoke(messages)
    else:
        final = response

    import json as _json
    import re
    try:
        json_match = re.search(r"\{[\s\S]*\}", final.content)
        data = _json.loads(json_match.group()) if json_match else {}
    except _json.JSONDecodeError:
        data = {}

    return {"hotels": data.get("hotels", []), "rag_refs": rag}
