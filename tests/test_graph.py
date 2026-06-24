from unittest.mock import patch, Mock
from agents.state import TravelPlanState


def test_build_graph_compiles():
    from agents.graph import build_graph

    # Mock all dependencies to avoid real API calls
    with patch("agents.graph.ChatTongyi") as mock_llm, \
         patch("agents.graph.AmapClient") as mock_amap, \
         patch("agents.graph.create_amap_tools") as mock_create_tools, \
         patch("agents.graph.create_embeddings") as mock_embeddings, \
         patch("agents.graph.VectorStoreManager") as mock_vs, \
         patch("agents.graph.DualRetriever") as mock_retriever:

        mock_llm.return_value = Mock()
        mock_amap.return_value = Mock()
        mock_create_tools.return_value = [Mock(name=f"tool_{i}") for i in range(5)]
        mock_embeddings.return_value = Mock()
        mock_vs.return_value = Mock()
        mock_retriever.return_value = Mock()

        from config import Settings
        settings = Settings(
            bailian_api_key="test-key",
            amap_api_key="test-key",
        )

        graph = build_graph(settings)
        assert graph is not None
        # Graph should have the key nodes (LangGraph adds __start__ and __end__ internally)
        node_names = list(graph.get_graph().nodes.keys())
        user_nodes = {n for n in node_names if not n.startswith("__")}
        assert user_nodes == {"orchestrator", "weather_agent", "attraction_agent", "hotel_agent", "synthesizer"}, f"Unexpected nodes: {node_names}"


def test_orchestrator_node_parses_input():
    from agents.orchestrator import orchestrator_node

    state = TravelPlanState(
        destination="成都",
        travel_date="2026-07-15",
        days=3,
        preferences="亲子",
        budget_total=5000.0,
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
    # Orchestrator just validates and passes through
    result = orchestrator_node(state)
    assert "destination" in result
    assert result["destination"] == "成都"
