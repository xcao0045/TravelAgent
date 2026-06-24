from langgraph.graph import StateGraph, END
from langgraph.types import Send
from langchain_community.chat_models.tongyi import ChatTongyi
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
    )
    _retriever = DualRetriever(vector_store)


def _get_llm():
    return _llm


def _get_tools():
    return _tools


def _get_retriever():
    return _retriever


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
        error_log=[],
        conversation=[],
        is_finalized=False,
    )
    result = graph.invoke(initial_state)
    return result
