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
        [f"[RAG-P{i+1}] {d.metadata.get('name','')}: {d.page_content[:300]} [标签: {d.metadata.get('tags',[])}]"
         for i, d in enumerate(rag_results["preferences"])]
    )
    cases_context = "\n".join(
        [f"[RAG-C{i+1}] {d.page_content[:400]}"
         for i, d in enumerate(rag_results["cases"])]
    )

    prompt = f"""你是酒店推荐专家。为{destination}筛选合适的酒店。

用户偏好: {preferences}
偏好库匹配（用户评价+标签）:
{prefs_context}
历史优秀案例参考:
{cases_context}

请直接推荐酒店，输出JSON格式: {{"hotels": [...], "sources": [...]}}
每个推荐含: name, address, rating, price_range, reason(请在理由中引用RAG来源编号如[RAG-P1]), matched_tags
sources字段: 列出本次推荐实际引用的RAG来源编号列表，如 ["RAG-P1", "RAG-P3"]
"""
    response = llm.invoke(prompt)

    import json
    import re
    try:
        json_match = re.search(r"\{[\s\S]*\}", response.content)
        data = json.loads(json_match.group()) if json_match else {}
    except json.JSONDecodeError:
        data = {}

    return {"hotels": data.get("hotels", [])}
