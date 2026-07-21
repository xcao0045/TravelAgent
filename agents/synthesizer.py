"""汇总 Agent — 整合子Agent输出 + 高德路线规划 Tool + 生成 Markdown 方案。"""
from langchain_core.messages import SystemMessage
from agents.state import TravelPlanState


_SYNTHESIZER_SYSTEM = """你是旅行方案汇总专家。按以下规则工作：

## 工具使用规则
1. 可用工具:
   - amap_multi_route: 规划多点串联路线 (waypoints: 逗号分隔的地点列表, mode: transit/driving/walking)
   - amap_route_plan: 规划两点间路线 (origin, destination, mode)
   - amap_geo_code: 地址→经纬度坐标

2. 行程合理性检查:
   - 对每个 Day 的景点列表，调用 amap_multi_route 计算总距离和总耗时
   - 若单日总距离 > 50km，在方案中标注 "⚠️ 该日行程较远（XXkm），建议调整"
   - 若酒店位置已知，调用 amap_route_plan 计算酒店到第一个景点的交通

## API 调用规则
1. 每个工具最多调用 1 次，禁止重试。
2. 工具返回含义约定：
   - "❌" 开头 → 系统故障，跳过路线验证，直接生成报告
   - "⚠️" 开头 → 该路数据缺失，标注 "路线数据暂不可用"
   - 其他 → 正常数据，整合到报告中
3. 无论 API 是否成功，都必须输出完整报告。

## 输出格式
生成完整的 Markdown 旅行方案，包含:
1. 天气概况
2. 每日行程安排 (Day1, Day2, ...) — 含路线时间/距离
3. 酒店推荐
4. 美食推荐
5. 交通建议 — 用路线规划结果
6. 预算明细
7. 注意事项

目的地: {destination}
天数: {days}天
总预算: {budget}元
{warnings}

## 天气
{weather_section}

## 景点推荐
{attr_text}

## 餐厅推荐
{rest_text}

## 酒店推荐
{hotel_text}"""


def synthesizer_node(state: TravelPlanState) -> dict:
    from agents.graph import _get_llm, _get_tools, _execute_tool_calls

    llm = _get_llm()
    all_tools = _get_tools()

    destination = state["destination"]
    days = state["days"]
    budget = state["budget_total"]
    weather = state.get("weather_report", "")
    attractions = state.get("attractions", [])
    restaurants = state.get("restaurants", [])
    hotels = state.get("hotels", [])
    error_log = state.get("error_log", [])

    weather_section = weather if weather else "⚠️ 天气数据暂不可用"
    attr_text = _format_list(attractions, "景点")
    rest_text = _format_list(restaurants, "餐厅")
    hotel_text = _format_list(hotels, "酒店")

    # 构建景点名称列表，用于路线规划
    attraction_names = [a.get("name", "") for a in attractions if a.get("name")]
    route_hint = ""
    if len(attraction_names) >= 2:
        route_hint = (
            f"\n\n## 路线规划提示\n"
            f"以下景点需要路线规划: {', '.join(attraction_names)}。\n"
            f"请用 amap_multi_route 规划这些景点的串联路线，验证行程合理性。"
        )

    warnings = ""
    if error_log:
        warnings = "## ⚠️ 系统告警\n" + "\n".join([f"> ⚠️ {e}" for e in error_log])

    system_prompt = _SYNTHESIZER_SYSTEM.format(
        destination=destination,
        days=days,
        budget=budget,
        weather_section=weather_section,
        attr_text=attr_text,
        rest_text=rest_text,
        hotel_text=hotel_text,
        warnings=warnings,
    ) + route_hint

    # 路线规划相关 tools
    route_tools = [t for t in all_tools if t.name in ("amap_multi_route", "amap_route_plan", "amap_geo_code")]

    llm_with_tools = llm.bind_tools(route_tools)
    messages = [SystemMessage(content=system_prompt)]
    response = llm_with_tools.invoke(messages)

    # Tool Calling loop (最多 1 轮)
    if hasattr(response, "tool_calls") and response.tool_calls:
        tool_messages = _execute_tool_calls(response, route_tools)
        messages.extend([response, *tool_messages])
        final = llm_with_tools.invoke(messages)
    else:
        final = response

    return {
        "final_report": final.content,
        "routes": [],  # 路线数据已整合到 Markdown 报告中
    }


def _format_list(items: list[dict], label: str) -> str:
    if not items:
        return f"⚠️ {label}推荐暂不可用"
    lines = []
    for i, item in enumerate(items, 1):
        name = item.get("name", "未知")
        reason = item.get("reason", "")
        address = item.get("address", "")
        location = f" ({address})" if address else ""
        lines.append(f"{i}. **{name}**{location} - {reason}")
    return "\n".join(lines)
