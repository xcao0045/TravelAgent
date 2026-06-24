from agents.state import TravelPlanState


def hotel_agent_node(state: TravelPlanState) -> dict:
    """
    酒店Agent：高德POI搜索 + RAG偏好库标签匹配 + RAG案例库参考，
    输出酒店推荐列表。
    """
    from agents.graph import _get_llm, _get_tools, _get_retriever

    llm = _get_llm()
    tools = _get_tools()
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
    prefs_context = "\n".join(
        [f"{d.metadata.get('name','')}: {d.page_content[:300]} [标签: {d.metadata.get('tags',[])}]"
         for d in rag_results["preferences"]]
    )
    cases_context = "\n".join([d.page_content[:500] for d in rag_results["cases"]])

    relevant_tools = [t for t in tools if t.name == "amap_poi_search"]
    llm_with_tools = llm.bind_tools(relevant_tools)

    prompt = f"""你是酒店推荐专家。为{destination}筛选合适的酒店。

用户偏好: {preferences}
偏好库匹配（用户评价+标签）:
{prefs_context}
历史优秀案例参考:
{cases_context}

请:
1. 用amap_poi_search搜索酒店(category=hotel)
2. 结合偏好库标签筛选（如用户偏好"亲子"→优先推荐标签含"隔音好""儿童乐园"的酒店）
3. 输出JSON格式: {{"hotels": [...]}}
每个推荐含: name, address, rating, price_range, reason(推荐理由，需引用偏好库标签), matched_tags
"""
    response = llm_with_tools.invoke(prompt)

    import json
    import re
    try:
        json_match = re.search(r"\{[\s\S]*\}", response.content)
        data = json.loads(json_match.group()) if json_match else {}
    except json.JSONDecodeError:
        data = {}

    return {"hotels": data.get("hotels", [])}
