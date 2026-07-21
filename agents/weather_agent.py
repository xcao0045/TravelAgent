"""天气 Agent — amap_weather Tool Calling + RAG 案例库检索。"""
from langchain_core.messages import SystemMessage
from agents.state import TravelPlanState


_WEATHER_SYSTEM = """你是天气查询专家。按以下规则工作：

## 工具使用规则（必须遵守）
1. 你有一个工具: amap_weather，用于查询城市实时天气。参数: city (城市名称)。
2. **你必须调用 amap_weather** 查询目的地天气。禁止跳过工具调用直接生成结果。
   拿到结果后再结合历史案例生成报告。

## API 调用规则
1. 每个工具最多调用 1 次，禁止重试。
2. 工具返回含义约定：
   - "❌" 开头 → 系统故障（网络/密钥），跳过该数据源，用你的知识补充，
     标注 "⚠️ 实时天气数据暂不可用，以下为模型预测"
   - "⚠️" 开头 → 数据缺失，跳过，用你的知识补充
   - 其他 → 正常数据，解析并整合
3. 无论 API 是否成功，都必须输出天气报告。禁止因 API 失败而中断。

## 输出格式
输出 Markdown 格式的天气报告:
1. 预测天气（温度、降水、风力）— 标注数据来源（高德实时 / 模型预测）
2. 穿衣建议 — 必须引用RAG来源编号如 [RAG-C1]
3. 对行程的影响提示 — 具体说明哪些日子适合户外活动

目的地: {destination}
出行日期: {travel_date}
用户偏好: {preferences}
历史案例参考（请在建议中引用来源编号 [RAG-C1]～[RAG-C{N_cases}]）:
{case_context}"""


def weather_agent_node(state: TravelPlanState) -> dict:
    from agents.graph import _get_llm, _get_tools, _get_retriever, _execute_tool_calls

    llm = _get_llm()
    all_tools = _get_tools()
    retriever = _get_retriever()

    destination = state["destination"]
    travel_date = state["travel_date"]
    preferences = state.get("preferences", "")

    # RAG 检索案例库
    case_docs = retriever.retrieve_cases(
        f"{destination} {travel_date} 天气 出行准备", k=3
    )
    case_rag = {}
    case_lines = []
    case_count = 0
    for i, d in enumerate(case_docs):
        case_count += 1
        rid = f"[RAG-C{case_count}]"
        case_rag[rid] = d.page_content[:1200]
        case_lines.append(
            f"{rid} 目的地:{d.metadata.get('destination','?')} "
            f"天数:{d.metadata.get('days','?')}天\n{d.page_content[:1200]}"
        )
    case_context = "\n".join(case_lines) if case_lines else "(无相关历史案例)"

    weather_tools = [t for t in all_tools if t.name == "amap_weather"]

    system_prompt = _WEATHER_SYSTEM.format(
        destination=destination,
        travel_date=travel_date,
        preferences=preferences,
        N_cases=case_count,
        case_context=case_context,
    )

    llm_with_tools = llm.bind_tools(weather_tools)
    messages = [SystemMessage(content=system_prompt)]
    response = llm_with_tools.invoke(messages)

    # Tool Calling loop (最多 1 轮)
    if hasattr(response, "tool_calls") and response.tool_calls:
        tool_messages = _execute_tool_calls(response, weather_tools)
        messages.extend([response, *tool_messages])
        final = llm_with_tools.invoke(messages)
    else:
        final = response

    print(f"[weather_agent] RAG cases={case_count} "
          f"tool_called={bool(response.tool_calls if hasattr(response,'tool_calls') else False)}")

    return {"weather_report": final.content, "rag_refs": case_rag}
