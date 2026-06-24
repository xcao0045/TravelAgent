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
        [f"[{d.metadata.get('category','')}] {d.metadata.get('name','')}: {d.page_content[:300]}"
         for d in rag_results["preferences"]]
    )
    cases_context = "\n".join([d.page_content[:500] for d in rag_results["cases"]])

    relevant_tools = [t for t in tools if t.name in ("amap_poi_search", "amap_multi_route")]
    llm_with_tools = llm.bind_tools(relevant_tools)

    prompt = f"""你是旅游规划专家。为{destination}规划{days}天的景点和餐厅。

用户偏好: {preferences}
偏好库匹配（用户评价标签）:
{prefs_context}
历史优秀案例参考:
{cases_context}

请:
1. 用amap_poi_search搜索景点(category=attraction)和餐厅(category=restaurant)
2. 结合偏好库标签筛选（如标签"适合看日落"优先推荐对应景点）
3. 用amap_multi_route规划每日景点串联路线
4. 输出JSON格式: {{"attractions": [...], "restaurants": [...]}}
每个推荐含: name, address, rating, reason(推荐理由), tags
"""
    response = llm_with_tools.invoke(prompt)

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
