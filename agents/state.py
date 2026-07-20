from typing import TypedDict, Annotated


def _merge_dict(left: dict, right: dict) -> dict:
    """LangGraph reducer：并行分支写入 dict 字段时合并而非覆盖。"""
    return {**left, **right}


class TravelPlanState(TypedDict):
    # === 用户输入 ===
    destination: str
    travel_date: str
    days: int
    preferences: str
    budget_total: float

    # === 子Agent输出 ===
    weather_report: str
    attractions: list[dict]
    restaurants: list[dict]
    hotels: list[dict]
    routes: list[dict]

    # === 汇总输出 ===
    final_report: str
    rag_refs: Annotated[dict, _merge_dict]  # 多 Agent 并行写入，reducer 合并
    error_log: list[str]

    # === 微调对话 ===
    conversation: list[dict]
    is_finalized: bool
