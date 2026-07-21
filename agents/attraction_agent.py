"""景点餐饮 Agent — amap_poi_search + RAG 双库检索。"""
from langchain_core.messages import SystemMessage
from agents.state import TravelPlanState


_ATTRACTION_SYSTEM = """你是旅游规划专家。按以下规则工作：

## 工具使用规则
1. 可用工具: amap_poi_search (搜索景点/餐厅)。
   参数: city, keyword, category (attraction/restaurant)
2. 先搜索景点 (category="attraction")，再搜索餐厅 (category="restaurant")。
   每次搜索使用与用户偏好相关的关键词。

## API 调用规则
1. 每个工具最多调用 1 次，禁止重试。
2. 工具返回含义约定：
   - "❌" 开头 → 系统故障，跳过该数据源，用 RAG 案例库和你的知识补充
   - "⚠️" 开头 → 该词未匹配结果，用 RAG 案例库补充替代推荐
   - 其他 → 正常数据，解析并整合
3. 无论 API 是否成功，都必须输出推荐列表。

## 输出格式
输出 JSON 格式: {{"attractions": [...], "restaurants": [...], "sources": [...]}}
每个推荐含: name, address, rating, reason(在理由中引用RAG来源编号如[RAG-P1][RAG-C2]), tags
sources字段: 列出本次推荐实际引用的RAG来源编号列表

目的地: {destination}
天数: {days}天
用户偏好: {preferences}
偏好库匹配（用户评价标签）:
{prefs_context}
历史优秀案例参考:
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
    prefs_lines = []
    for i, d in enumerate(rag_results["preferences"]):
        rid = f"[RAG-P{i+1}]"
        rag[rid] = d.page_content[:300]
        prefs_lines.append(f"{rid} {d.metadata.get('category','')}·{d.metadata.get('name','')} [标签:{d.metadata.get('tags',[])}]: {d.page_content[:300]}")
    cases_lines = []
    for i, d in enumerate(rag_results["cases"]):
        rid = f"[RAG-C{i+1}]"
        rag[rid] = d.page_content[:400]
        cases_lines.append(f"{rid} {d.page_content[:400]}")

    prefs_context = "\n".join(prefs_lines) if prefs_lines else "(无相关偏好数据)"
    cases_context = "\n".join(cases_lines) if cases_lines else "(无相关历史案例)"

    # 取 POI + multi_route tools
    agent_tools = [t for t in all_tools if t.name in ("amap_poi_search",)]

    system_prompt = _ATTRACTION_SYSTEM.format(
        destination=destination,
        days=days,
        preferences=preferences,
        prefs_context=prefs_context,
        cases_context=cases_context,
    )

    llm_with_tools = llm.bind_tools(agent_tools)
    messages = [SystemMessage(content=system_prompt)]
    response = llm_with_tools.invoke(messages)

    # Tool Calling loop (最多 1 轮)
    if hasattr(response, "tool_calls") and response.tool_calls:
        tool_messages = _execute_tool_calls(response, agent_tools)
        messages.extend([response, *tool_messages])
        final = llm_with_tools.invoke(messages)
    else:
        final = response

    # 解析 JSON
    import json as _json
    import re
    try:
        json_match = re.search(r"\{[\s\S]*\}", final.content)
        data = _json.loads(json_match.group()) if json_match else {}
    except _json.JSONDecodeError:
        data = {}

    return {
        "attractions": data.get("attractions", []),
        "restaurants": data.get("restaurants", []),
        "rag_refs": rag,
    }
