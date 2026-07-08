from agents.state import TravelPlanState


def weather_agent_node(state: TravelPlanState) -> dict:
    """
    天气Agent：调用高德天气API + RAG案例库，
    生成目的地天气报告和出行建议。
    """
    from agents.graph import _get_llm, _get_tools, _get_retriever

    llm = _get_llm()
    tools = _get_tools()
    retriever = _get_retriever()

    destination = state["destination"]
    travel_date = state["travel_date"]
    preferences = state.get("preferences", "")

    # RAG 检索案例库中同目的地的天气应对策略
    case_docs = retriever.retrieve_cases(
        f"{destination} {travel_date} 天气 出行准备", k=3
    )
    case_context = "\n".join([d.page_content[:500] for d in case_docs])

    prompt = f"""你是天气查询专家。请根据你的知识，为{destination}在{travel_date}前后的天气情况给出预测和建议。

用户偏好: {preferences}
历史案例参考:
{case_context}

请输出:
1. 预测天气（温度、降水、风力）
2. 穿衣建议
3. 对行程的影响提示
"""
    response = llm.invoke(prompt)
    return {"weather_report": response.content}
