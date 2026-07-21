"""汇总 Agent — 整合子Agent输出 + 高德路线 + RAG上下文 + 生成详细 Markdown。"""
from langchain_core.messages import SystemMessage
from agents.state import TravelPlanState


_SYNTHESIZER_SYSTEM = """你是旅行方案汇总专家。按以下规则工作：

## 工具使用规则
1. 可用工具: amap_multi_route (多点路线) / amap_route_plan (两点路线)
2. **禁止直接调用 amap_geo_code** — 路线工具内部已自动完成地名→坐标解析和 QPS 优化，无需手动 geocode。
3. **必须对每一天的景点调用 amap_multi_route**，景点名直接传入即可（工具会自动解析坐标）。
4. 工具返回 "⚠️ 坐标异常提醒" 或 "⚠️ 存在异常长距离路段" 时，该路段数据不可用，用你的知识估算替代。
5. 工具返回错误时标注 "⚠️ 路线数据暂不可用"，不中断报告生成。

## 输出模板（强制遵守——输出必须且仅包含以下 5 个一级标题）

你的输出必须严格按以下模板结构生成，**不可增减或重排序标题**：

---

# ☀️ 一、天气概况与出行建议
- 日期范围与每日预报（温度、天气现象、风力）
- 针对天气的穿搭建议（如"建议携带雨具""注意防晒"）
- 天气对行程的影响与备选方案（如"若下雨可将户外景点改为室内博物馆"）

# 🗺️ 二、每日详细行程安排
Day1 ~ Day{days}，每一天必须包含：
- **上午 (08:00-12:00)**: 景点名 + 游玩时长 + 亮点描述
- **午餐 (12:00-13:30)**: 餐厅名 + 推荐菜 + 人均消费 + 到达方式
- **下午 (13:30-18:00)**: 景点名 + 游玩时长 + 亮点描述
- **晚餐 (18:00-19:30)**: 餐厅名 + 推荐菜 + 人均消费 + 到达方式
- **晚间 (19:30- )**: 可选活动建议
- 每两个地点之间标注: 🚗 路线数据（距离/耗时/方式），引用Tool返回的真实数据
- **路线数据格式要求**: 必须直接引用 amap_multi_route 或 amap_route_plan 返回的真实数据
  正确示例: 🚗 [驾车] 5.0km (12分钟) 或 🚌 [公交] 3站 (20分钟)
  错误示例: ❌ "步行约15分钟" ❌ "大概几站路" ❌ 不标注距离只写耗时

禁止项：
- ❌ 禁止"自行探索""根据兴趣选择""等地标""等景点"
- ❌ 禁止空的景点名或餐厅名
- ❌ 禁止"Day X 请自行安排"

# 🏨 三、酒店住宿推荐
- 至少推荐 2-3 家不同档次的酒店，标注来源：
  - 引用 RAG 来源的标注 [RAG-P1] 等
  - 基于高德 POI 搜索的标注 [高德POI]
  - 基于通用知识补全的标注 [通用知识]
- 每家含: 名称、地址、价格区间、推荐理由、距景区距离

# 💰 四、预算参考明细
分项列出预估费用:
| 项目 | 明细 | 预估费用 |
|------|------|----------|
| 交通 | 往返大交通 + 市内交通 | XX元 |
| 住宿 | {days}晚 × XX酒店 | XX元 |
| 门票 | 各景点门票汇总 | XX元 |
| 餐饮 | 正餐 + 小吃零食 | XX元 |
| **总计** | | **XX元** |

# ⚠️ 五、出行注意事项
- 针对目的地与人群的专属贴士（至少5条）
- 高德 API 数据可用性说明（如有降级）
- 预约提醒、避坑指南、最佳拍照点

---

## 可用数据汇总

目的地: {destination} | 天数: {days}天 | 预算: {budget}元
{warnings}

## 天气数据
{weather_section}

## 景点推荐
{attr_text}

## 餐厅推荐
{rest_text}

## 酒店推荐
{hotel_text}

## 知识库参考
{rag_context}
{route_hint}

---
**再次强调：输出必须严格使用上述5个一级标题（一、二、三、四、五），每个板块必须填满具体内容。禁止输出空白板块或"暂无"占位。**"""


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
    rag_refs = state.get("rag_refs", {})

    weather_section = weather if weather else "⚠️ 天气数据暂不可用"
    attr_text = _format_list(attractions, "景点")
    rest_text = _format_list(restaurants, "餐厅")
    hotel_text = _format_list(hotels, "酒店")

    # 注入 RAG 检索结果到 prompt 中
    rag_context = ""
    if rag_refs:
        rag_lines = ["以下是从知识库中检索到的相关资料，请在方案中引用:"]
        for rid, text in rag_refs.items():
            rag_lines.append(f"**{rid}**: {text[:800]}")
        rag_context = "\n".join(rag_lines)
    else:
        rag_context = "(知识库未匹配到相关内容)"

    # 路线规划提示：从 attractions 中提取真实景点名
    attraction_names = [a.get("name", "") for a in attractions if a.get("name", "").strip()]
    # 过滤掉明显是编造的名称（太长的、不含中文的、含"推荐"等词的）
    valid_names = [
        n for n in attraction_names
        if len(n) <= 20 and any('一' <= c <= '鿿' for c in n)
        and "推荐" not in n and "景点" not in n
    ]
    route_hint = ""
    if len(valid_names) >= 2:
        route_hint = (
            f"\n\n## 路线规划任务（必须执行）\n"
            f"以下景点需要路线规划: {', '.join(valid_names[:10])}。\n"
            f"**必须对每一天的景点调用 amap_multi_route**，将真实的距离和耗时写入行程中。\n"
            f"若某天仅 1 个景点，改用 amap_route_plan 查询酒店到该景点的路线。"
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
        rag_context=rag_context,
        warnings=warnings,
        route_hint=route_hint,
    )

    route_tools = [t for t in all_tools if t.name in ("amap_multi_route", "amap_route_plan")]

    # ── Tool Calling (最多 2 轮) ──
    llm_with_tools = llm.bind_tools(route_tools)
    messages = [SystemMessage(content=system_prompt)]
    response = llm_with_tools.invoke(messages)

    for _round in range(2):
        if hasattr(response, "tool_calls") and response.tool_calls:
            tool_messages = _execute_tool_calls(response, route_tools)
            if not tool_messages:
                break
            messages.extend([response, *tool_messages])
            response = llm_with_tools.invoke(messages)
        else:
            break

    report = response.content
    print(f"[synthesizer] RAG={len(rag_refs)} attrs={len(attractions)} "
          f"rests={len(restaurants)} hotels={len(hotels)} valid_names={len(valid_names)} "
          f"report_len={len(report)} tool_calls={bool(response.tool_calls if hasattr(response,'tool_calls') else False)}")

    return {
        "final_report": report,
        "routes": [],
    }


def _format_list(items: list[dict], label: str) -> str:
    if not items:
        return f"⚠️ {label}推荐暂不可用（请检查高德API Key或知识库数据）"
    lines = []
    for i, item in enumerate(items, 1):
        name = item.get("name", "未知")
        reason = item.get("reason", "")
        address = item.get("address", "")
        rating = item.get("rating", "")
        tags = item.get("tags", [])
        tag_str = f" [{'·'.join(tags)}]" if tags else ""
        location = f" 📍{address}" if address else ""
        score = f" ⭐{rating}" if rating else ""
        lines.append(f"{i}. **{name}**{score}{tag_str}{location}\n   {reason}")
    return "\n".join(lines)
