from unittest.mock import patch
from tools.amap_client import AmapClient
from tools.amap_tools import create_amap_tools, amap_weather_factory


def test_amap_weather_returns_formatted_string():
    client = AmapClient(api_key="test-key")
    mock_response = {
        "status": "1",
        "lives": [
            {"city": "成都", "temperature": "28", "weather": "晴", "winddirection": "东", "windpower": "≤3"}
        ],
    }

    with patch.object(client, "weather", return_value=mock_response):
        tool_fn = amap_weather_factory(client)
        result = tool_fn.invoke("成都")

    assert "成都" in result
    assert "28" in result
    assert "晴" in result


def test_create_amap_tools_returns_five_tools():
    client = AmapClient(api_key="test-key")
    tools = create_amap_tools(client)

    assert len(tools) == 5
    tool_names = {t.name for t in tools}
    assert tool_names == {
        "amap_weather",
        "amap_poi_search",
        "amap_route_plan",
        "amap_multi_route",
        "amap_geo_code",
    }
