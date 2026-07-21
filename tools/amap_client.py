import json as _json
import requests


class AmapClient:
    BASE_URL = "https://restapi.amap.com/v3"

    def __init__(self, api_key: str, timeout: int = 10):
        self.api_key = api_key
        self.timeout = timeout

    def _get(self, path: str, params: dict) -> dict:
        """统一 GET 请求 + 错误处理。

        正常响应（status="1"）→ 原样返回高德 dict，不含 "error" 字段。
        任何异常（超时/网络/JSON/API错误）→ 返回 {"error": True, "info": "..."}.
        """
        params["key"] = self.api_key
        url = f"{self.BASE_URL}{path}"
        try:
            resp = requests.get(url, params=params, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
        except requests.Timeout:
            return {"error": True, "info": "高德 API 请求超时"}
        except requests.RequestException as e:
            return {"error": True, "info": f"高德 API 网络异常: {e}"}
        except (_json.JSONDecodeError, ValueError):
            return {"error": True, "info": "高德 API 返回数据格式异常"}

        if data.get("status") != "1":
            return {"error": True, "info": data.get("info", "未知错误")}
        return data

    def weather(self, city: str) -> dict:
        """查询城市实时天气"""
        return self._get("/weather/weatherInfo", {"city": city, "extensions": "all"})

    def poi_search(
        self,
        keywords: str,
        types: str,
        city: str,
        offset: int = 10,
    ) -> dict:
        """POI搜索：types可取 风景名胜|餐饮服务|住宿服务"""
        return self._get(
            "/place/text",
            {"keywords": keywords, "types": types, "city": city, "offset": offset},
        )

    def direction(
        self, origin: str, destination: str, mode: str = "transit"
    ) -> dict:
        """路线规划：mode = transit|driving|walking"""
        return self._get(
            f"/direction/{mode}",
            {"origin": origin, "destination": destination},
        )

    def geo_code(self, address: str) -> dict:
        """地理编码：地址→经纬度"""
        return self._get("/geocode/geo", {"address": address})

    def resolve_coord(self, place: str) -> str | None:
        """将地名解析为 'lon,lat' 坐标字符串。解析失败返回 None。"""
        result = self.geo_code(place)
        if result.get("error"):
            return None
        geocodes = result.get("geocodes", [])
        if not geocodes:
            return None
        location = geocodes[0].get("location", "")
        return location if location else None
