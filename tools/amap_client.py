import requests


class AmapClient:
    BASE_URL = "https://restapi.amap.com/v3"

    def __init__(self, api_key: str):
        self.api_key = api_key

    def _get(self, path: str, params: dict) -> dict:
        params["key"] = self.api_key
        url = f"{self.BASE_URL}{path}"
        resp = requests.get(url, params=params, timeout=10)
        return resp.json()

    def weather(self, city: str) -> dict:
        """查询城市实时天气"""
        return self._get("/weather/weatherInfo", {"city": city, "extensions": "base"})

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
