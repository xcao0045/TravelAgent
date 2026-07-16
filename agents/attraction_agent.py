from agents.state import TravelPlanState


def attraction_agent_node(state: TravelPlanState) -> dict:
    """
    景点餐饮Agent：高德POI搜索 + RAG偏好库匹配 + RAG案例库参考，
    输出景点推荐和餐厅推荐。
    """
    from agents.graph import _get_llm, _get_tools, _get_retriever

    llm = _get_llm()
    tools = _get_tools()
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
    prefs_context = "\n".join(
        [f"[RAG-P{i+1}] {d.metadata.get('category','')}·{d.metadata.get('name','')} [标签:{d.metadata.get('tags',[])}]: {d.page_content[:300]}"
         for i, d in enumerate(rag_results["preferences"])]
    )
    cases_context = "\n".join(
        [f"[RAG-C{i+1}] {d.page_content[:400]}"
         for i, d in enumerate(rag_results["cases"])]
    )

    prompt = f"""你是旅游规划专家。为{destination}规划{days}天的景点和餐厅。

用户偏好: {preferences}
偏好库匹配（用户评价标签）:
{prefs_context}
历史优秀案例参考:
{cases_context}

请直接推荐景点和餐厅，输出JSON格式: {{"attractions": [...], "restaurants": [...], "sources": [...]}}
每个推荐含: name, address, rating, reason(请在理由中引用RAG来源编号如[RAG-P1][RAG-C2]), tags
sources字段: 列出本次推荐实际引用的RAG来源编号列表，如 ["RAG-P1", "RAG-C2", "RAG-P3"]
"""
    response = llm.invoke(prompt)

    # 解析JSON
    import json
    import re
    try:
        json_match = re.search(r"\{[\s\S]*\}", response.content)
        data = json.loads(json_match.group()) if json_match else {}
    except json.JSONDecodeError:
        data = {}

    return {
        "attractions": data.get("attractions", []),
        "restaurants": data.get("restaurants", []),
    }
