from unittest.mock import patch, Mock
import json
import requests
from tools.amap_client import AmapClient


def test_weather_returns_parsed_response():
    client = AmapClient(api_key="test-key")
    mock_response = {
        "status": "1",
        "lives": [{"city": "成都", "temperature": "28", "weather": "晴"}],
    }
    with patch("tools.amap_client.requests.get") as mock_get:
        mock_get.return_value = Mock(status_code=200, json=lambda: mock_response)
        result = client.weather("成都")
    assert result["status"] == "1"
    assert result["lives"][0]["city"] == "成都"


def test_poi_search_returns_parsed_response():
    client = AmapClient(api_key="test-key")
    mock_response = {
        "status": "1",
        "pois": [{"id": "B0FFF", "name": "成都大熊猫繁育研究基地"}],
    }
    with patch("tools.amap_client.requests.get") as mock_get:
        mock_get.return_value = Mock(status_code=200, json=lambda: mock_response)
        result = client.poi_search(keywords="熊猫基地", types="风景名胜", city="成都")
    assert len(result["pois"]) == 1


# ── 新增: 异常处理测试 ──

class TestAmapClientErrorHandling:
    """验证 _get() 统一错误返回格式"""

    def test_timeout_returns_error_dict(self):
        """请求超时 → 返回 {"error": True, "info": "..."}"""
        client = AmapClient(api_key="test-key")
        with patch("tools.amap_client.requests.get") as mock_get:
            mock_get.side_effect = requests.Timeout()
            result = client.weather("成都")
        assert result.get("error") is True
        assert "超时" in result.get("info", "")

    def test_network_error_returns_error_dict(self):
        """网络异常 → 返回 {"error": True, "info": "..."}"""
        client = AmapClient(api_key="test-key")
        with patch("tools.amap_client.requests.get") as mock_get:
            mock_get.side_effect = requests.ConnectionError()
            result = client.weather("成都")
        assert result.get("error") is True
        assert "网络异常" in result.get("info", "")

    def test_invalid_json_returns_error_dict(self):
        """非法 JSON → 返回 {"error": True, "info": "..."}"""
        client = AmapClient(api_key="test-key")
        with patch("tools.amap_client.requests.get") as mock_get:
            mock_get.return_value = Mock(status_code=200, json=lambda: (_ for _ in ()).throw(json.JSONDecodeError("", "", 0)))
            result = client.weather("成都")
        assert result.get("error") is True
        assert "格式异常" in result.get("info", "")

    def test_api_status_not_1_returns_error_dict(self):
        """高德返回 status!="1" → 返回 {"error": True, "info": "..."}"""
        client = AmapClient(api_key="test-key")
        with patch("tools.amap_client.requests.get") as mock_get:
            mock_get.return_value = Mock(status_code=200, json=lambda: {"status": "0", "info": "INVALID_USER_KEY"})
            result = client.weather("成都")
        assert result.get("error") is True
        assert "INVALID_USER_KEY" in result.get("info", "")

    def test_normal_response_no_error_flag(self):
        """正常响应 → 不含 error 字段"""
        client = AmapClient(api_key="test-key")
        with patch("tools.amap_client.requests.get") as mock_get:
            mock_get.return_value = Mock(status_code=200, json=lambda: {"status": "1", "lives": []})
            result = client.weather("成都")
        assert "error" not in result
