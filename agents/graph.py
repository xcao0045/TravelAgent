from langgraph.graph import StateGraph, END
from langgraph.types import Send
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.messages import SystemMessage, ToolMessage, AIMessage
from agents.state import TravelPlanState
from agents.orchestrator import orchestrator_node
from agents.weather_agent import weather_agent_node
from agents.attraction_agent import attraction_agent_node
from agents.hotel_agent import hotel_agent_node
from agents.synthesizer import synthesizer_node
from tools.amap_client import AmapClient
from tools.amap_tools import create_amap_tools
from rag.embedding import create_embeddings
from rag.vector_store import VectorStoreManager
from rag.retriever import DualRetriever
from config import Settings

# Module-level singletons (initialized on first use)
_llm = None
_tools = None
_retriever = None
_settings = None


def _init_dependencies(settings: Settings):
    global _llm, _tools, _retriever, _settings
    _settings = settings

    # LLM
    _llm = ChatTongyi(
        dashscope_api_key=settings.bailian_api_key,
        model=settings.llm_model,
    )

    # Tools
    amap_client = AmapClient(api_key=settings.amap_api_key)
    _tools = create_amap_tools(amap_client)

    # RAG
    embeddings = create_embeddings(
        api_key=settings.bailian_api_key,
        model=settings.embedding_model,
    )
    vector_store = VectorStoreManager(
        persist_dir=settings.chroma_persist_dir,
        embeddings=embeddings,
        child_chunk_size=settings.child_chunk_size,
        child_chunk_overlap=settings.child_chunk_overlap,
        parent_chunk_size=settings.parent_chunk_size,
        parent_chunk_overlap=settings.parent_chunk_overlap,
    )
    _retriever = DualRetriever(
        vector_store,
        similarity_threshold=settings.similarity_threshold,
        search_type=settings.search_type,
        rrf_k=settings.rrf_k,
    )


def _get_llm():
    return _llm


def _get_tools():
    return _tools


def _get_retriever():
    return _retriever


# ── Tool Calling 公共辅助 ─────────────────────────────────────

def _tool_by_name(tools: list, name: str):
    """按名称查找 tool。"""
    for t in tools:
        if t.name == name:
            return t
    return None


def _execute_tool_calls(response, available_tools: list) -> list[ToolMessage]:
    """执行 LLM 返回的 tool_calls，返回 ToolMessage 列表。

    每个 Tool 只调用一次。若 tool 不存在或执行异常，以 ToolMessage 包装错误信息。
    """
    messages = []
    for tc in response.tool_calls:
        tool_name = tc.get("name", "")
        tool_args = tc.get("args", {})
        tool = _tool_by_name(available_tools, tool_name)
        if tool is None:
            messages.append(ToolMessage(
                content=f"❌ 未知工具: {tool_name}",
                tool_call_id=tc["id"],
            ))
            continue
        try:
            result = tool.invoke(tool_args)
        except Exception as e:
            result = f"❌ 工具执行异常: {e}"
        messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
    return messages


def _continue_to_sub_agents(state: TravelPlanState):
    """条件边：将任务扇出到三个子Agent"""
    if state.get("destination", "").strip() == "":
        return []
    return [
        Send("weather_agent", state),
        Send("attraction_agent", state),
        Send("hotel_agent", state),
    ]


def build_graph(settings: Settings):
    """构建LangGraph状态图"""
    _init_dependencies(settings)

    graph = StateGraph(TravelPlanState)

    # 添加节点
    graph.add_node("orchestrator", orchestrator_node)
    graph.add_node("weather_agent", weather_agent_node)
    graph.add_node("attraction_agent", attraction_agent_node)
    graph.add_node("hotel_agent", hotel_agent_node)
    graph.add_node("synthesizer", synthesizer_node)

    # 边
    graph.set_entry_point("orchestrator")
    graph.add_conditional_edges("orchestrator", _continue_to_sub_agents)
    graph.add_edge("weather_agent", "synthesizer")
    graph.add_edge("attraction_agent", "synthesizer")
    graph.add_edge("hotel_agent", "synthesizer")
    graph.add_edge("synthesizer", END)

    return graph.compile()


def run_travel_plan(user_input: dict, settings: Settings) -> TravelPlanState:
    """执行一次旅行规划"""
    graph = build_graph(settings)
    initial_state = TravelPlanState(
        destination=user_input.get("destination", ""),
        travel_date=user_input.get("travel_date", ""),
        days=user_input.get("days", 3),
        preferences=user_input.get("preferences", ""),
        budget_total=float(user_input.get("budget_total", 0)),
        weather_report="",
        attractions=[],
        restaurants=[],
        hotels=[],
        routes=[],
        final_report="",
        rag_refs={},
        error_log=[],
        conversation=[],
        is_finalized=False,
    )
    result = graph.invoke(initial_state)
    return result
