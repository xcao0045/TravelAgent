import sys
from unittest.mock import MagicMock, Mock

from agents.state import TravelPlanState
from agents.weather_agent import weather_agent_node


def test_weather_agent_returns_report():
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

    mock_graph = MagicMock()
    mock_llm_response = Mock(content="成都7月15日天气：晴，28°C，适宜出行",
                             tool_calls=None)
    mock_llm_with_tools = Mock()
    mock_llm_with_tools.invoke.return_value = mock_llm_response
    mock_llm = Mock()
    mock_llm.bind_tools.return_value = mock_llm_with_tools
    mock_tools = [Mock(name="amap_weather")]
    mock_retriever = Mock()
    mock_retriever.retrieve_cases.return_value = []
    mock_graph._get_llm = Mock(return_value=mock_llm)
    mock_graph._get_tools = Mock(return_value=mock_tools)
    mock_graph._get_retriever = Mock(return_value=mock_retriever)
    mock_graph._execute_tool_calls = Mock(return_value=[])

    sys.modules["agents.graph"] = mock_graph

    try:
        result = weather_agent_node(state)
        assert "weather_report" in result
        assert isinstance(result["weather_report"], str)
        assert len(result["weather_report"]) > 0
    finally:
        sys.modules.pop("agents.graph", None)


def make_state(**overrides) -> TravelPlanState:
    """Helper to build a default TravelPlanState for tests."""
    defaults = dict(
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
    defaults.update(overrides)
    return TravelPlanState(**defaults)


def _inject_mock_graph(llm_response_content: str, tools: list | None = None,
                       retrieve_cases_return: list | None = None,
                       retrieve_both_return: dict | None = None):
    """Inject a mock agents.graph module into sys.modules and return the mocks."""
    mock_graph = MagicMock()
    mock_llm_response = Mock(content=llm_response_content, tool_calls=None)
    mock_llm_with_tools = Mock()
    mock_llm_with_tools.invoke.return_value = mock_llm_response
    mock_llm = Mock()
    mock_llm.bind_tools.return_value = mock_llm_with_tools
    mock_tools = tools or [Mock(name="amap_weather")]
    mock_retriever = Mock()
    if retrieve_cases_return is not None:
        mock_retriever.retrieve_cases.return_value = retrieve_cases_return
    if retrieve_both_return is not None:
        mock_retriever.retrieve_both.return_value = retrieve_both_return
    mock_graph._get_llm = Mock(return_value=mock_llm)
    mock_graph._get_tools = Mock(return_value=mock_tools)
    mock_graph._get_retriever = Mock(return_value=mock_retriever)
    mock_graph._execute_tool_calls = Mock(return_value=[])
    sys.modules["agents.graph"] = mock_graph
    return mock_graph


def _cleanup_mock_graph():
    sys.modules.pop("agents.graph", None)


def test_attraction_agent_returns_lists():
    from agents.attraction_agent import attraction_agent_node

    state = make_state()
    json_output = '{"attractions": [{"name": "宽窄巷子", "rating": "4.5"}], "restaurants": [{"name": "小龙坎", "rating": "4.3"}]}'

    _inject_mock_graph(
        llm_response_content=json_output,
        tools=[Mock(name="amap_poi_search"), Mock(name="amap_multi_route")],
        retrieve_both_return={"preferences": [], "cases": []},
    )
    try:
        result = attraction_agent_node(state)
        assert "attractions" in result
        assert "restaurants" in result
        assert isinstance(result["attractions"], list)
        assert isinstance(result["restaurants"], list)
        assert len(result["attractions"]) == 1
        assert result["attractions"][0]["name"] == "宽窄巷子"
    finally:
        _cleanup_mock_graph()


def test_hotel_agent_returns_list():
    from agents.hotel_agent import hotel_agent_node

    state = make_state()
    json_output = '{"hotels": [{"name": "锦江宾馆", "rating": "4.7", "price_range": "500-800"}]}'

    _inject_mock_graph(
        llm_response_content=json_output,
        tools=[Mock(name="amap_poi_search")],
        retrieve_both_return={"preferences": [], "cases": []},
    )
    try:
        result = hotel_agent_node(state)
        assert "hotels" in result
        assert isinstance(result["hotels"], list)
        assert len(result["hotels"]) == 1
        assert result["hotels"][0]["name"] == "锦江宾馆"
    finally:
        _cleanup_mock_graph()
