from unittest.mock import patch, Mock
from tools.amap_client import AmapClient


def test_weather_returns_parsed_response():
    client = AmapClient(api_key="test-key")
    mock_response = {
        "status": "1",
        "lives": [
            {
                "city": "成都",
                "temperature": "28",
                "weather": "晴",
                "winddirection": "东",
                "windpower": "≤3",
            }
        ],
    }

    with patch("tools.amap_client.requests.get") as mock_get:
        mock_get.return_value = Mock(status_code=200, json=lambda: mock_response)
        result = client.weather("成都")

    assert result["status"] == "1"
    assert result["lives"][0]["city"] == "成都"
    mock_get.assert_called_once()
    call_args = mock_get.call_args[1]["params"]
    assert call_args["city"] == "成都"
    assert call_args["key"] == "test-key"


def test_poi_search_returns_parsed_response():
    client = AmapClient(api_key="test-key")
    mock_response = {
        "status": "1",
        "pois": [
            {"id": "B0FFF", "name": "成都大熊猫繁育研究基地", "type": "风景名胜"}
        ],
    }

    with patch("tools.amap_client.requests.get") as mock_get:
        mock_get.return_value = Mock(status_code=200, json=lambda: mock_response)
        result = client.poi_search(keywords="熊猫基地", types="风景名胜", city="成都")

    assert len(result["pois"]) == 1
    call_args = mock_get.call_args[1]["params"]
    assert call_args["keywords"] == "熊猫基地"


def test_amap_client_handles_http_error_gracefully():
    client = AmapClient(api_key="test-key")

    with patch("tools.amap_client.requests.get") as mock_get:
        mock_get.return_value = Mock(status_code=500, json=lambda: {"status": "0"})
        result = client.weather("成都")

    assert result["status"] == "0"
