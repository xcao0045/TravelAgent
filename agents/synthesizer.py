"""汇总 Agent — 整合子Agent输出 + 高德路线 + RAG上下文 + 生成详细 Markdown。"""
from langchain_core.messages import SystemMessage
from agents.state import TravelPlanState


_SYNTHESIZER_SYSTEM = """你是旅行方案汇总专家。按以下规则工作：

## 工具使用规则（必须遵守）
1. 可用工具:
   - amap_multi_route: 多点路线 (waypoints: 逗号分隔地点列表, mode: transit/driving/walking)
   - amap_route_plan: 两点路线 (origin, destination, mode)
   - amap_geo_code: 地址→坐标

2. **你必须对每一天的景点列表调用 amap_multi_route**。禁止跳过路线验证。
   若单日景点 ≤1 个无法调用 multi_route，改用 amap_route_plan 查酒店到景点。

## API 调用规则
1. 每个工具最多调用 1 次，禁止重试。
2. 工具返回含义约定：
   - "❌" 开头 → 系统故障，跳过路线验证，标注 "⚠️ 路线数据暂不可用"
   - "⚠️" 开头 → 数据缺失，标注 "路线数据暂不可用"
   - 其他 → 正常数据，整合到每日行程中
3. 无论 API 是否成功，都必须输出完整报告。

## 输出格式 — 硬性约束（禁止违反）
生成完整的 Markdown 旅行方案。以下每条规则必须严格遵守：

**禁止项（违反即为失败）:**
- ❌ 禁止输出 "自行探索"、"根据个人兴趣选择"、"等地标"、"等景点"
- ❌ 禁止出现 "Day X 请自行安排" 或 "其他时间自由活动"
- ❌ 禁止输出空的景点名或餐厅名

**必须项（缺少即为失败）:**
- ✅ 每一天 (Day1, Day2, ... Day{days}) 必须包含具体行程:
  * 上午: 景点名 + 游玩时长（如2小时）+ 具体亮点
  * 午餐: 餐厅名 + 推荐菜名 + 人均消费
  * 下午: 景点名 + 游玩时长 + 具体亮点
  * 晚餐: 餐厅名 + 推荐菜名 + 人均消费
  * 晚间（可选）: 活动建议
- ✅ 每个景点/餐厅后标注路线信息（如 [步行5分钟]、[公交20分钟 2元]）
- ✅ 引用所有可用的数据来源: [高德实时数据]、[RAG-P1]、[RAG-C2] 等

## 高质量示例
以下是期望的输出格式（以苏州2日游为例，基于真实数据）：

```markdown
# 苏州2日园林美食之旅

## 天气概况
7月22日 晴 28-35°C（高德实时数据）。建议携带防晒和遮阳伞。

## Day1: 姑苏区经典园林线
### 上午: 拙政园 (2.5小时)
中国四大名园之一，夏日荷花盛开。[高德实时数据]
> 🚗 酒店→拙政园: 公交2站 约15分钟

### 午餐: 松鹤楼 (观前街店)
松鼠桂鱼(158元)是招牌，响油鳝糊也很地道。人均120元。[步行5分钟]
> 📖 [RAG-P1] 食客评价: "老字号苏帮菜，松鼠桂鱼必点"

### 下午: 苏州博物馆 (1.5小时) → 狮子林 (1小时)
贝聿铭设计，免费需预约。[RAG-C3] [步行3分钟到狮子林]
> 🚗 拙政园→苏博: 步行2分钟

### 晚餐: 同得兴面馆
枫镇大肉面(28元)夏季限定。[公交10分钟]
> 📖 [RAG-P2] "苏州最好吃的面馆之一"

...
```

## 可用数据汇总

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
{hotel_text}

## 知识库参考资料
{rag_context}
{route_hint}"""


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

    route_tools = [t for t in all_tools if t.name in ("amap_multi_route", "amap_route_plan", "amap_geo_code")]

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
