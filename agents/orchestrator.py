from agents.state import TravelPlanState


def orchestrator_node(state: TravelPlanState) -> dict:
    """
    主控Agent：解析用户输入，验证必要字段，传递任务给子Agent。
    如果输入不完整或模糊，要求LLM追问用户。
    """
    destination = state.get("destination", "").strip()
    if not destination:
        error_log = state.get("error_log", [])
        error_log.append("主控Agent: 目的地为空，需要用户补充")
        return {"error_log": error_log}

    # 一切正常，透传state
    return {"destination": destination}
