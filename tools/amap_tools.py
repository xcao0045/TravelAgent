"""高德地图 LangChain Tool — 含 Pydantic Input Schema + 统一错误处理。"""
from langchain.tools import tool
from pydantic import BaseModel, Field
from tools.amap_client import AmapClient


# ── Input Schemas ────────────────────────────────────────────

class WeatherInput(BaseModel):
    city: str = Field(description="城市名称，如'成都'、'苏州'")


class POISearchInput(BaseModel):
    city: str = Field(description="城市名称，如'成都'")
    keyword: str = Field(description="搜索关键词，如'亲子景点'、'希尔顿酒店'")
    category: str = Field(
        default="",
        description="POI类别: hotel(酒店)/restaurant(餐厅)/attraction(景点)，留空则搜索全部",
    )


class RoutePlanInput(BaseModel):
    origin: str = Field(description="起点地址，如'宽窄巷子'")
    destination: str = Field(description="终点地址，如'锦里'")
    mode: str = Field(
        default="transit",
        description="出行方式: transit(公交)/driving(驾车)/walking(步行)",
    )


class MultiRouteInput(BaseModel):
    waypoints: str = Field(description="用逗号分隔的地点列表，如'宽窄巷子,锦里,武侯祠'")
    mode: str = Field(default="driving", description="出行方式: transit/driving/walking")


class GeoCodeInput(BaseModel):
    address: str = Field(description="地址名称，如'成都市锦江区春熙路'")


# ── Helper ───────────────────────────────────────────────────

def _is_error(result: dict) -> bool:
    """检测 AmapClient 返回的统一错误格式。"""
    return result.get("error", False)


# ── Tool Factory Functions ───────────────────────────────────

def amap_weather_factory(client: AmapClient):
    @tool(args_schema=WeatherInput)
    def amap_weather(city: str) -> str:
        """查询指定城市的实时天气，返回温度、天气状况、风力等信息。"""
        result = client.weather(city)
        if _is_error(result):
            return f"❌ {result['info']}"
        if not result.get("lives"):
            return f"⚠️ 未获取到 {city} 的天气数据"
        live = result["lives"][0]
        return (
            f"城市: {live['city']}\n"
            f"天气: {live['weather']}\n"
            f"温度: {live['temperature']}°C\n"
            f"风向: {live.get('winddirection', '未知')}\n"
            f"风力: {live.get('windpower', '未知')}"
        )
    return amap_weather


def amap_poi_search_factory(client: AmapClient):
    @tool(args_schema=POISearchInput)
    def amap_poi_search(city: str, keyword: str, category: str = "") -> str:
        """搜索指定城市的POI（酒店/景点/餐厅），返回名称、地址、评分。"""
        types_map = {
            "hotel": "住宿服务",
            "restaurant": "餐饮服务",
            "attraction": "风景名胜",
        }
        types = types_map.get(category, "风景名胜|餐饮服务|住宿服务")
        result = client.poi_search(keywords=keyword, types=types, city=city)
        if _is_error(result):
            return f"❌ {result['info']}"
        if not result.get("pois"):
            return f"⚠️ 未搜索到与'{keyword}'相关的{category or 'POI'}信息"
        pois = result["pois"][:10]
        lines = [f"搜索'{keyword}'结果:"]
        for i, poi in enumerate(pois, 1):
            lines.append(
                f"{i}. {poi['name']} | "
                f"地址: {poi.get('address', '未知')} | "
                f"评分: {poi.get('biz_ext', {}).get('rating', '暂无')}"
            )
        return "\n".join(lines)
    return amap_poi_search


def amap_route_plan_factory(client: AmapClient):
    @tool(args_schema=RoutePlanInput)
    def amap_route_plan(origin: str, destination: str, mode: str = "transit") -> str:
        """规划两点之间的出行路线，返回距离、耗时、费用。"""
        result = client.direction(origin=origin, destination=destination, mode=mode)
        if _is_error(result):
            return f"❌ {result['info']}"
        route = result.get("route", {})
        if mode == "transit" and route.get("transits"):
            transit = route["transits"][0]
            return (
                f"从 {origin} → {destination}\n"
                f"方式: 公交/地铁\n"
                f"耗时: {int(transit.get('duration', 0)) // 60}分钟\n"
                f"费用: {transit.get('cost', '未知')}元"
            )
        if mode in ("driving", "walking") and route.get("paths"):
            path = route["paths"][0]
            return (
                f"从 {origin} → {destination}\n"
                f"距离: {int(path.get('distance', 0))}米\n"
                f"耗时: {int(path.get('duration', 0)) // 60}分钟"
            )
        return f"⚠️ 从 {origin} → {destination}: 未找到路线"
    return amap_route_plan


def amap_multi_route_factory(client: AmapClient):
    @tool(args_schema=MultiRouteInput)
    def amap_multi_route(waypoints: str, mode: str = "driving") -> str:
        """规划多点串联路线（如一日游景点顺序），返回逐段距离、耗时和汇总。"""
        points = [w.strip() for w in waypoints.split(",")]
        if len(points) < 2:
            return "⚠️ 至少需要两个地点"
        lines = [f"📍 多点路线规划 ({mode}):"]
        total_distance = 0
        total_duration = 0
        has_error = False
        for i in range(len(points) - 1):
            result = client.direction(origin=points[i], destination=points[i + 1], mode=mode)
            if _is_error(result):
                lines.append(f"  {points[i]} → {points[i+1]}: ❌ {result['info']}")
                has_error = True
                continue
            route = result.get("route", {})
            if mode == "transit" and route.get("transits"):
                transit = route["transits"][0]
                dist = 0
                dur = int(transit.get("duration", 0))
            elif route.get("paths"):
                path = route["paths"][0]
                dist = int(path.get("distance", 0))
                dur = int(path.get("duration", 0))
            else:
                lines.append(f"  {points[i]} → {points[i+1]}: 路线计算失败")
                has_error = True
                continue
            total_distance += dist
            total_duration += dur
            lines.append(f"  {points[i]} → {points[i+1]}: {dist}米, {int(dur // 60)}分钟")
        lines.append(f"总距离: {total_distance}米, 总耗时: {int(total_duration // 60)}分钟")
        if has_error:
            lines.append("⚠️ 部分路段计算失败，总距离/耗时仅供参考")
        return "\n".join(lines)
    return amap_multi_route


def amap_geo_code_factory(client: AmapClient):
    @tool(args_schema=GeoCodeInput)
    def amap_geo_code(address: str) -> str:
        """将地址转换为经纬度坐标，用于后续路线计算。"""
        result = client.geo_code(address)
        if _is_error(result):
            return f"❌ {result['info']}"
        if not result.get("geocodes"):
            return f"⚠️ 无法解析地址: {address}"
        geo = result["geocodes"][0]
        return f"地址: {address}\n坐标: {geo['location']}"
    return amap_geo_code


def create_amap_tools(client: AmapClient) -> list:
    """创建全部5个高德LangChain Tool"""
    return [
        amap_weather_factory(client),
        amap_poi_search_factory(client),
        amap_route_plan_factory(client),
        amap_multi_route_factory(client),
        amap_geo_code_factory(client),
    ]
