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

# 城市名 → adcode 映射（高德天气 API 要求 adcode）
_CITY_ADCODE = {
    "北京": "110000", "上海": "310000", "广州": "440100", "深圳": "440300",
    "成都": "510100", "杭州": "330100", "苏州": "320500", "南京": "320100",
    "武汉": "420100", "西安": "610100", "重庆": "500000", "厦门": "350200",
    "青岛": "370200", "大连": "210200", "三亚": "460200", "昆明": "530100",
    "长沙": "430100", "郑州": "410100", "天津": "120000", "桂林": "450300",
    "丽江": "530700", "大理": "532900", "拉萨": "540100", "贵阳": "520100",
}


def amap_weather_factory(client: AmapClient):
    @tool(args_schema=WeatherInput)
    def amap_weather(city: str) -> str:
        """查询指定城市的实时天气，返回温度、天气状况、风力等信息。"""
        # 高德天气 API 优先使用 adcode
        city_param = _CITY_ADCODE.get(city, city)
        result = client.weather(city_param)
        if _is_error(result):
            return f"❌ {result['info']}"
        # 优先取实时天气 lives，其次取预报 forecasts
        if result.get("lives"):
            live = result["lives"][0]
            return (
                f"城市: {live.get('city', city)}\n"
                f"天气: {live.get('weather', '未知')}\n"
                f"温度: {live.get('temperature', '未知')}°C\n"
                f"风向: {live.get('winddirection', '未知')}\n"
                f"风力: {live.get('windpower', '未知')}\n"
                f"湿度: {live.get('humidity', '未知')}%"
            )
        if result.get("forecasts"):
            fc = result["forecasts"][0]
            casts = fc.get("casts", [])
            if casts:
                today = casts[0]
                return (
                    f"城市: {fc.get('city', city)}\n"
                    f"日期: {today.get('date', '未知')}\n"
                    f"白天天气: {today.get('dayweather', '未知')}\n"
                    f"夜间天气: {today.get('nightweather', '未知')}\n"
                    f"温度: {today.get('nighttemp', '?')}°C ~ {today.get('daytemp', '?')}°C\n"
                    f"风向: {today.get('daywind', '未知')} {today.get('daypower', '')}"
                )
        return f"⚠️ 未获取到 {city} 的天气数据（已尝试 adcode={city_param}）"
    return amap_weather


# 抽象/感性词汇 → 具体搜索词的映射表
_KEYWORD_EXPAND = {
    # 酒店
    "情侣酒店": "高档酒店|精品酒店|度假酒店|湖景酒店",
    "浪漫酒店": "高档酒店|湖景酒店|精品民宿",
    "亲子酒店": "亲子酒店|家庭房|儿童乐园酒店",
    "安静酒店": "精品酒店|商务酒店|度假村",
    # 餐厅
    "浪漫餐厅": "西餐厅|日料|景观餐厅|黑珍珠|江浙菜",
    "情侣餐厅": "西餐厅|景观餐厅|日料|法餐|私房菜",
    "亲子餐厅": "亲子餐厅|家庭套餐|儿童友好",
    "高档餐厅": "黑珍珠|米其林|私房菜|粤菜|法餐",
    # 景点
    "情侣景点": "园林|古镇|湖景|夜景|摩天轮",
    "亲子景点": "乐园|动物园|科技馆|公园|博物馆",
}
_EMPTY_FALLBACK = {
    "hotel": "豪华酒店|精品酒店", "restaurant": "人气餐厅|特色美食",
    "attraction": "热门景点|5A景区", "": "热门推荐",
}


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

        # 关键词映射：将抽象词展开为具体搜索词
        search_keyword = _KEYWORD_EXPAND.get(keyword, keyword)
        result = client.poi_search(keywords=search_keyword, types=types, city=city)

        if _is_error(result):
            return f"❌ {result['info']}"
        if not result.get("pois"):
            # 首次搜索无结果 → 用兜底词重试
            fallback = _EMPTY_FALLBACK.get(category, "热门推荐")
            result2 = client.poi_search(keywords=fallback, types=types, city=city)
            if not _is_error(result2) and result2.get("pois"):
                result = result2
                keyword = fallback
            else:
                return f"⚠️ 未搜索到与'{keyword}'相关的{category or 'POI'}信息（已尝试: {search_keyword}）"
        pois = result["pois"][:10]
        lines = [f"搜索'{keyword}'结果（关键词: {search_keyword}）:"]
        for i, poi in enumerate(pois, 1):
            lines.append(
                f"{i}. {poi['name']} | "
                f"地址: {poi.get('address', '未知')} | "
                f"评分: {poi.get('biz_ext', {}).get('rating', '暂无')}"
            )
        return "\n".join(lines)
    return amap_poi_search


def _try_direction_with_fallback(client: AmapClient, origin: str, destination: str,
                                 mode: str) -> tuple[dict | None, str]:
    """尝试路线规划，文本名失败时自动用坐标重试。返回 (route_dict, error_msg)。"""
    result = client.direction(origin=origin, destination=destination, mode=mode)
    if not result.get("error"):
        return result.get("route", {}), ""

    # 文本名失败 → 尝试地理编码转为坐标后重试
    coord_o = client.resolve_coord(origin)
    coord_d = client.resolve_coord(destination)
    if coord_o and coord_d:
        result2 = client.direction(origin=coord_o, destination=coord_d, mode=mode)
        if not result2.get("error"):
            return result2.get("route", {}), ""
    return None, result.get("info", "未知错误")


def amap_route_plan_factory(client: AmapClient):
    @tool(args_schema=RoutePlanInput)
    def amap_route_plan(origin: str, destination: str, mode: str = "transit") -> str:
        """规划两点之间的出行路线，返回距离、耗时、费用。"""
        route, error = _try_direction_with_fallback(client, origin, destination, mode)
        if route is None:
            # transit 模式失败 → 尝试 driving
            if mode == "transit":
                route2, _ = _try_direction_with_fallback(client, origin, destination, "driving")
                if route2 is not None:
                    if route2.get("paths"):
                        path = route2["paths"][0]
                        return (
                            f"从 {origin} → {destination}\n"
                            f"方式: 驾车（公交路线暂无数据）\n"
                            f"距离: {int(path.get('distance', 0))}米\n"
                            f"耗时: {int(path.get('duration', 0)) // 60}分钟"
                        )
            return f"❌ 从 {origin} 到 {destination} 路线查询失败: {error}"

        if mode == "transit" and route.get("transits"):
            transit = route["transits"][0]
            return (
                f"从 {origin} → {destination}\n"
                f"方式: 公交/地铁\n"
                f"耗时: {int(transit.get('duration', 0)) // 60}分钟\n"
                f"费用: {transit.get('cost', '未知')}元"
            )
        if route.get("paths"):
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
        # 预解析所有地名 → 坐标（缓存，避免重复 geocode）
        coord_cache: dict[str, str | None] = {}
        for p in points:
            coord_cache[p] = client.resolve_coord(p)

        lines = [f"📍 多点路线规划 ({mode}):"]
        total_distance = 0
        total_duration = 0
        has_error = False
        for i in range(len(points) - 1):
            route, error = _try_direction_with_fallback(client, points[i], points[i + 1], mode)
            if route is None and mode == "transit":
                route, error = _try_direction_with_fallback(client, points[i], points[i + 1], "driving")
            if route is None:
                lines.append(f"  {points[i]} → {points[i+1]}: ❌ {error}")
                has_error = True
                continue
            dist = 0
            dur = 0
            if mode == "transit" and route.get("transits"):
                transit = route["transits"][0]
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
            dur_min = int(dur // 60) if dur else 0
            dist_km = f"{dist/1000:.1f}km" if dist else "N/A"
            lines.append(f"  {points[i]} → {points[i+1]}: {dist_km} ({dur_min}分钟)")
        lines.append(f"总距离: {'{:.1f}'.format(total_distance/1000)}km, 总耗时: {int(total_duration // 60)}分钟")
        if has_error:
            lines.append("⚠️ 部分路段无法识别，总距离/耗时仅供参考")
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
