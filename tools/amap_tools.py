from langchain.tools import tool
from tools.amap_client import AmapClient


def amap_weather_factory(client: AmapClient):
    @tool
    def amap_weather(city: str) -> str:
        """查询指定城市的实时天气，返回温度、天气状况、风力等信息。
        :param city: 城市名称，如"成都"、"北京"
        """
        result = client.weather(city)
        if result.get("status") != "1" or not result.get("lives"):
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
    @tool
    def amap_poi_search(city: str, keyword: str, category: str = "") -> str:
        """搜索指定城市的POI（酒店/景点/餐厅）。
        :param city: 城市名称
        :param keyword: 搜索关键词
        :param category: 类别，可选 hotel/restaurant/attraction
        """
        types_map = {
            "hotel": "住宿服务",
            "restaurant": "餐饮服务",
            "attraction": "风景名胜",
        }
        types = types_map.get(category, "风景名胜|餐饮服务|住宿服务")
        result = client.poi_search(keywords=keyword, types=types, city=city)
        if result.get("status") != "1" or not result.get("pois"):
            return f"⚠️ 未搜索到与'{keyword}'相关的{category}信息"
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
    @tool
    def amap_route_plan(origin: str, destination: str, mode: str = "transit") -> str:
        """规划两点之间的出行路线。
        :param origin: 起点地址
        :param destination: 终点地址
        :param mode: 出行方式 transit=公交 driving=驾车 walking=步行
        """
        result = client.direction(origin=origin, destination=destination, mode=mode)
        if result.get("status") != "1":
            return f"⚠️ 路线规划失败: {origin} → {destination}"
        route = result.get("route", {})
        if mode == "transit" and route.get("transits"):
            transit = route["transits"][0]
            return (
                f"从 {origin} → {destination}\n"
                f"方式: 公交/地铁\n"
                f"耗时: {transit.get('duration', '未知')}秒\n"
                f"费用: {transit.get('cost', '未知')}元"
            )
        if mode in ("driving", "walking") and route.get("paths"):
            path = route["paths"][0]
            return (
                f"从 {origin} → {destination}\n"
                f"距离: {path.get('distance', '未知')}米\n"
                f"耗时: {path.get('duration', '未知')}秒"
            )
        return f"从 {origin} → {destination}: 未找到路线"
    return amap_route_plan


def amap_multi_route_factory(client: AmapClient):
    @tool
    def amap_multi_route(waypoints: str, mode: str = "driving") -> str:
        """规划多点串联路线（如一日游景点顺序）。
        :param waypoints: 用逗号分隔的地点列表，如"宽窄巷子,锦里,武侯祠"
        :param mode: 出行方式
        """
        points = [w.strip() for w in waypoints.split(",")]
        if len(points) < 2:
            return "⚠️ 至少需要两个地点"
        lines = [f"📍 多点路线规划 ({mode}):"]
        total_distance = 0
        total_duration = 0
        for i in range(len(points) - 1):
            result = client.direction(origin=points[i], destination=points[i + 1], mode=mode)
            if result.get("status") != "1":
                lines.append(f"  {points[i]} → {points[i+1]}: 路线计算失败")
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
                continue
            total_distance += dist
            total_duration += dur
            lines.append(f"  {points[i]} → {points[i+1]}: {dist}米, {dur}秒")
        lines.append(f"总距离: {total_distance}米, 总耗时: {total_duration}秒")
        return "\n".join(lines)
    return amap_multi_route


def amap_geo_code_factory(client: AmapClient):
    @tool
    def amap_geo_code(address: str) -> str:
        """将地址转换为经纬度坐标。
        :param address: 地址名称
        """
        result = client.geo_code(address)
        if result.get("status") != "1" or not result.get("geocodes"):
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
