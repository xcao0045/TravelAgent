import json as _json
import requests


# 主要旅游城市经纬度范围 (用于校验坐标是否在正确城市)
_CITY_BOUNDS: dict[str, tuple[float, float, float, float]] = {
    "苏州": (120.3, 121.0, 30.9, 31.7),   "北京": (115.4, 117.5, 39.4, 41.1),
    "上海": (120.8, 122.0, 30.6, 31.6),   "杭州": (118.3, 120.8, 29.1, 30.6),
    "南京": (118.3, 119.3, 31.1, 32.6),   "成都": (103.0, 104.6, 30.0, 31.0),
    "广州": (112.9, 114.0, 22.5, 23.9),   "深圳": (113.7, 114.6, 22.4, 22.9),
    "西安": (108.7, 109.8, 33.6, 34.7),   "重庆": (105.2, 110.2, 28.1, 32.2),
    "昆明": (102.1, 103.3, 24.3, 26.1),   "厦门": (117.9, 118.5, 24.3, 24.8),
    "青岛": (119.5, 121.0, 35.6, 36.9),   "武汉": (113.6, 115.1, 29.9, 31.4),
    "长沙": (112.5, 114.3, 27.8, 28.6),   "三亚": (108.9, 109.8, 18.0, 18.5),
    "大理": (99.9, 100.5, 25.3, 26.0),    "丽江": (99.8, 100.8, 26.5, 27.8),
    "桂林": (110.0, 110.7, 24.6, 25.8),    "贵阳": (106.2, 107.3, 26.0, 27.2),
    "天津": (116.6, 118.1, 38.5, 40.3),    "大连": (121.0, 122.2, 38.6, 39.5),
}
_ANYWHERE_BOUNDS = (73.0, 135.0, 17.0, 54.0)  # 中国全境范围


class AmapClient:
    BASE_URL = "https://restapi.amap.com/v3"

    def __init__(self, api_key: str, timeout: int = 10):
        self.api_key = api_key
        self.timeout = timeout
        self._geocode_cache: dict[str, str | None] = {}  # 地名→坐标 缓存

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

    def resolve_coord(self, place: str, city: str = "") -> str | None:
        """将地名解析为 'lon,lat' 坐标。解析失败返回 None。

        解析链: 缓存 → 坐标检测 → POI文本提取 → '名称;lng,lat' 分隔 → geocode API → 城市范围校验
        """
        if place in self._geocode_cache:
            return self._geocode_cache[place]
        # 处理 '名称;lng,lat' 格式（LLM 显式传入坐标）
        semicolon = _split_semicolon_coord(place)
        if semicolon:
            name, coord = semicolon
            self._geocode_cache[name] = coord  # 缓存名称→坐标
            self._geocode_cache[place] = coord
            return self._validate_coord(coord, city)
        if _looks_like_coord(place):
            self._geocode_cache[place] = place
            return self._validate_coord(place, city)
        # 尝试从 POI 文本中提取坐标 (格式: "XXX | 坐标: lng,lat")
        coord_from_text = _extract_coord_from_text(place)
        if coord_from_text:
            self._geocode_cache[place] = coord_from_text
            return self._validate_coord(coord_from_text, city)
        # geocode API (最后手段)
        result = self.geo_code(place)
        if result.get("error"):
            self._geocode_cache[place] = None
            return None
        geocodes = result.get("geocodes", [])
        if not geocodes:
            self._geocode_cache[place] = None
            return None
        location = geocodes[0].get("location", "")
        if not location:
            self._geocode_cache[place] = None
            return None
        validated = self._validate_coord(location, city)
        self._geocode_cache[place] = validated
        return validated

    def _validate_coord(self, coord: str, city: str) -> str | None:
        """校验坐标是否在指定城市范围内。超出范围但仍在中国的返回坐标(标注异常)，
        超出中国范围的返回 None。"""
        import re
        m = re.match(r'(\d{2,3}\.\d+),(\d{2,3}\.\d+)', coord)
        if not m:
            return None
        lng, lat = float(m.group(1)), float(m.group(2))
        # 中国范围校验
        min_lng, max_lng, min_lat, max_lat = _ANYWHERE_BOUNDS
        if not (min_lng <= lng <= max_lng and min_lat <= lat <= max_lat):
            return None
        # 城市范围校验（宽松警告，不拒绝）
        if city and city in _CITY_BOUNDS:
            c_min_lng, c_max_lng, c_min_lat, c_max_lat = _CITY_BOUNDS[city]
            if not (c_min_lng <= lng <= c_max_lng and c_min_lat <= lat <= c_max_lat):
                return f"{coord} ⚠️坐标超出{city}范围"  # 仍然可用, 标注异常
        return coord


def _looks_like_coord(text: str) -> bool:
    """检测文本是否已经是 'lng,lat' 坐标格式。"""
    import re
    return bool(re.match(r'^\d{2,3}\.\d+,\d{2,3}\.\d+$', text.strip()))


def _split_semicolon_coord(text: str) -> tuple[str, str] | None:
    """解析 '名称;lng,lat' 格式。返回 (name, coord) 或 None。"""
    import re
    if ';' not in text:
        return None
    parts = text.rsplit(';', 1)
    if len(parts) != 2:
        return None
    name, coord_part = parts[0].strip(), parts[1].strip()
    if re.match(r'^\d{2,3}\.\d+,\d{2,3}\.\d+$', coord_part):
        return name, coord_part
    return None


def _extract_coord_from_text(text: str) -> str | None:
    """从 POI 搜索结果文本中提取坐标 (格式: '坐标: lng,lat')。
    用于 geocode API 失败时的回退方案。"""
    import re
    m = re.search(r'坐标[：:]\s*(\d{2,3}\.\d+),(\d{2,3}\.\d+)', text)
    if m:
        return f"{m.group(1)},{m.group(2)}"
    return None
