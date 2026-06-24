from typing import TypedDict


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
    error_log: list[str]

    # === 微调对话 ===
    conversation: list[dict]
    is_finalized: bool
