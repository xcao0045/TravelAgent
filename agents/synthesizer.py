from agents.state import TravelPlanState


def synthesizer_node(state: TravelPlanState) -> dict:
    """
    汇总Agent：整合天气、景点、酒店、餐厅数据，
    结合高德路线规划Tool生成完整Markdown旅行方案。
    """
    from agents.graph import _get_llm, _get_tools

    llm = _get_llm()
    tools = _get_tools()
    route_tools = [t for t in tools if t.name in ("amap_route_plan", "amap_geo_code")]
    llm_with_tools = llm.bind_tools(route_tools) if route_tools else llm

    destination = state["destination"]
    days = state["days"]
    budget = state["budget_total"]
    weather = state.get("weather_report", "")
    attractions = state.get("attractions", [])
    restaurants = state.get("restaurants", [])
    hotels = state.get("hotels", [])
    error_log = state.get("error_log", [])

    # 构建各部分文本
    weather_section = weather if weather else "⚠️ 天气数据暂不可用"
    attr_text = _format_list(attractions, "景点")
    rest_text = _format_list(restaurants, "餐厅")
    hotel_text = _format_list(hotels, "酒店")

    # 错误提示
    warnings = ""
    if error_log:
        warnings = "\n".join([f"> ⚠️ {e}" for e in error_log])

    prompt = f"""你是旅行方案汇总专家。请将以下信息整合为一份结构化的Markdown旅行方案。

目的地: {destination}
天数: {days}天
总预算: {budget}元

## 天气
{weather_section}

## 景点推荐
{attr_text}

## 餐厅推荐
{rest_text}

## 酒店推荐
{hotel_text}

{warnings}

请生成Markdown格式的完整旅行方案，包含:
1. 天气概况
2. 每日行程安排 (Day1, Day2, ...)
3. 酒店推荐
4. 美食推荐
5. 交通建议
6. 预算明细
7. 注意事项
"""
    response = llm_with_tools.invoke(prompt)
    return {
        "final_report": response.content,
        "routes": [],  # 路线已在子Agent中通过amap_multi_route获取
    }


def _format_list(items: list[dict], label: str) -> str:
    if not items:
        return f"⚠️ {label}推荐暂不可用"
    lines = []
    for i, item in enumerate(items, 1):
        name = item.get("name", "未知")
        reason = item.get("reason", "")
        lines.append(f"{i}. **{name}** - {reason}")
    return "\n".join(lines)
