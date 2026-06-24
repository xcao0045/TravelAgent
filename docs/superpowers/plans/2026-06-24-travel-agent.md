# Multi-Agent 智能旅行规划 Web 系统 — 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建一个 Streamlit + LangGraph 的 Multi-Agent 旅行规划系统，用户输入目的地/日期/偏好后，AI 自动生成涵盖天气、景点、酒店、餐饮、交通、预算的完整 Markdown 旅行方案。

**Architecture:** LangGraph 状态机编排 1 个主控 + 3 个并行子 Agent（天气/景点餐饮/酒店）+ 1 个汇总 Agent。子 Agent 统一调用高德 LangChain Tool 获取实时数据，同时检索 ChromaDB 双向量库（用户偏好库 + 优质案例库）辅助精准推荐。Streamlit 提供旅行规划首页、知识库管理页、历史记录页三个页面。

**Tech Stack:** Python 3.10+, Streamlit, LangChain + LangGraph, 阿里百炼千问 (qwen-max), ChromaDB, 高德地图 Web API

## Global Constraints

- Python >= 3.10
- API Key 通过 `.env` 文件管理，决不在 UI 暴露
- 所有 Agent 通过 LangGraph State 通信，不直接 import 彼此
- TDD：先写测试 → 验证失败 → 再写实现
- 提交信息使用中文描述

---

### Task 1: 项目脚手架与配置中心

**Files:**
- Create: `.env.example`
- Create: `.gitignore`
- Create: `requirements.txt`
- Create: `config.py`
- Create: `storage/.gitkeep`
- Create: `data/history/.gitkeep`

**Interfaces:**
- Consumes: (none — first task)
- Produces:
  - `config.py` exports `Settings` dataclass with fields: `bailian_api_key: str`, `amap_api_key: str`, `llm_model: str`, `embedding_model: str`, `chroma_persist_dir: str`, `history_dir: str`, `top_k_preferences: int`, `top_k_cases: int`, `similarity_threshold: float`
  - `Settings.from_env()` classmethod reads `.env`

- [ ] **Step 1: Write test for Settings.from_env()**

```python
# tests/test_config.py
import os
import tempfile
from config import Settings

def test_settings_from_env_reads_all_fields():
    env_vars = {
        "BAILIAN_API_KEY": "sk-test-bailian",
        "AMAP_API_KEY": "amap-test-key",
        "LLM_MODEL": "qwen-max",
        "EMBEDDING_MODEL": "text-embedding-v3",
        "CHROMA_PERSIST_DIR": "./storage",
        "HISTORY_DIR": "./data/history",
    }
    # Simulate os.environ
    import os as _os
    for k, v in env_vars.items():
        _os.environ[k] = v

    settings = Settings.from_env()

    assert settings.bailian_api_key == "sk-test-bailian"
    assert settings.amap_api_key == "amap-test-key"
    assert settings.llm_model == "qwen-max"
    assert settings.embedding_model == "text-embedding-v3"
    assert settings.top_k_preferences == 5
    assert settings.top_k_cases == 3
    assert settings.similarity_threshold == 0.7


def test_settings_uses_defaults_when_env_missing():
    # Clear env vars for this test
    for k in ["BAILIAN_API_KEY", "AMAP_API_KEY", "LLM_MODEL"]:
        os.environ.pop(k, None)

    settings = Settings.from_env()

    assert settings.llm_model == "qwen-max"
    assert settings.top_k_preferences == 5
    assert settings.top_k_cases == 3
    assert settings.similarity_threshold == 0.7
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'config'`

- [ ] **Step 3: Create project files**

```python
# config.py
import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Settings:
    bailian_api_key: str = ""
    amap_api_key: str = ""
    llm_model: str = "qwen-max"
    embedding_model: str = "text-embedding-v3"
    chroma_persist_dir: str = "./storage"
    history_dir: str = "./data/history"
    top_k_preferences: int = 5
    top_k_cases: int = 3
    similarity_threshold: float = 0.7

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            bailian_api_key=os.getenv("BAILIAN_API_KEY", ""),
            amap_api_key=os.getenv("AMAP_API_KEY", ""),
            llm_model=os.getenv("LLM_MODEL", "qwen-max"),
            embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-v3"),
            chroma_persist_dir=os.getenv("CHROMA_PERSIST_DIR", "./storage"),
            history_dir=os.getenv("HISTORY_DIR", "./data/history"),
            top_k_preferences=int(os.getenv("TOP_K_PREFERENCES", "5")),
            top_k_cases=int(os.getenv("TOP_K_CASES", "3")),
            similarity_threshold=float(os.getenv("SIMILARITY_THRESHOLD", "0.7")),
        )
```

```bash
# .env.example
BAILIAN_API_KEY=your_bailian_api_key_here
AMAP_API_KEY=your_amap_api_key_here
LLM_MODEL=qwen-max
EMBEDDING_MODEL=text-embedding-v3
```

```
# .gitignore
.env
__pycache__/
*.pyc
storage/chroma_*/
.pytest_cache/
```

```
# requirements.txt
streamlit>=1.28.0
langchain>=0.3.0
langgraph>=0.2.0
chromadb>=0.5.0
python-dotenv>=1.0.0
pydantic>=2.0.0
requests>=2.31.0
dashscope>=1.20.0
pytest>=8.0.0
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_config.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add .env.example .gitignore requirements.txt config.py tests/test_config.py storage/.gitkeep data/history/.gitkeep
git commit -m "feat: 项目脚手架空与settings配置中心"
```

---

### Task 2: 高德 API 客户端 (AmapClient)

**Files:**
- Create: `tools/__init__.py`
- Create: `tools/amap_client.py`
- Create: `tests/test_amap_client.py`

**Interfaces:**
- Consumes: `config.Settings` (amap_api_key)
- Produces:
  - `class AmapClient`: `__init__(self, api_key: str)`, `weather(self, city: str) -> dict`, `poi_search(self, keywords: str, types: str, city: str, offset: int = 10) -> dict`, `direction(self, origin: str, destination: str, mode: str) -> dict`, `geo_code(self, address: str) -> dict`
  - `BASE_URL = "https://restapi.amap.com/v3"`

- [ ] **Step 1: Write test for AmapClient.weather()**

```python
# tests/test_amap_client.py
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


def test_amap_client_raises_on_http_error():
    client = AmapClient(api_key="test-key")

    with patch("tools.amap_client.requests.get") as mock_get:
        mock_get.return_value = Mock(status_code=500, json=lambda: {"status": "0"})
        result = client.weather("成都")

    assert result["status"] == "0"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_amap_client.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tools.amap_client'`

- [ ] **Step 3: Implement AmapClient**

```python
# tools/__init__.py
# (empty)
```

```python
# tools/amap_client.py
import requests
from typing import Optional


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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_amap_client.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add tools/__init__.py tools/amap_client.py tests/test_amap_client.py
git commit -m "feat: 高德API客户端AmapClient，含天气/POI/路线/地理编码"
```

---

### Task 3: 高德 LangChain Tool 封装

**Files:**
- Create: `tools/amap_tools.py`
- Create: `tests/test_amap_tools.py`

**Interfaces:**
- Consumes: `tools.amap_client.AmapClient`
- Produces: 5 `@tool` 函数: `amap_weather(city: str) -> str`, `amap_poi_search(city: str, keyword: str, category: str) -> str`, `amap_route_plan(origin: str, destination: str, mode: str) -> str`, `amap_multi_route(waypoints: list[str], mode: str) -> str`, `amap_geo_code(address: str) -> str`
- Produces: `create_amap_tools(client: AmapClient) -> list` returns all 5 tools as a list

- [ ] **Step 1: Write test for amap_weather tool**

```python
# tests/test_amap_tools.py
from unittest.mock import patch, Mock
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
        result = tool_fn("成都")

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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_amap_tools.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tools.amap_tools'`

- [ ] **Step 3: Implement LangChain Tools**

```python
# tools/amap_tools.py
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
            if result.get("status") == "1" and result.get("route", {}).get("paths"):
                path = result["route"]["paths"][0]
                dist = int(path.get("distance", 0))
                dur = int(path.get("duration", 0))
                total_distance += dist
                total_duration += dur
                lines.append(f"  {points[i]} → {points[i+1]}: {dist}米, {dur}秒")
            else:
                lines.append(f"  {points[i]} → {points[i+1]}: 路线计算失败")
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_amap_tools.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add tools/amap_tools.py tests/test_amap_tools.py
git commit -m "feat: 高德5个LangChain Tool封装(天气/POI/路线/多点/地理编码)"
```

---

### Task 4: RAG Embedding 与 ChromaDB Vector Store

**Files:**
- Create: `rag/__init__.py`
- Create: `rag/embedding.py`
- Create: `rag/vector_store.py`
- Create: `tests/test_vector_store.py`

**Interfaces:**
- Consumes: `config.Settings` (bailian_api_key, embedding_model, chroma_persist_dir)
- Produces:
  - `rag.embedding.create_embeddings(api_key: str, model: str = "text-embedding-v3") -> DashScopeEmbeddings`
  - `rag.vector_store.VectorStoreManager`: `__init__(self, persist_dir: str, embeddings: DashScopeEmbeddings)`, `get_preferences_collection() -> Chroma`, `get_cases_collection() -> Chroma`, `add_to_preferences(self, docs: list[Document]) -> list[str]`, `add_to_cases(self, docs: list[Document]) -> list[str]`

- [ ] **Step 1: Write test for vector store creation and add**

```python
# tests/test_vector_store.py
import tempfile
import shutil
from unittest.mock import patch, Mock
from rag.vector_store import VectorStoreManager


class FakeEmbeddings:
    def embed_documents(self, texts):
        return [[0.1] * 1024 for _ in texts]

    def embed_query(self, text):
        return [0.1] * 1024


def test_create_collections():
    tmpdir = tempfile.mkdtemp()
    try:
        embeddings = FakeEmbeddings()
        manager = VectorStoreManager(persist_dir=tmpdir, embeddings=embeddings)

        prefs = manager.get_preferences_collection()
        cases = manager.get_cases_collection()

        assert prefs.name == "user_preferences"
        assert cases.name == "travel_cases"
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_add_and_query_preferences():
    from langchain.schema import Document

    tmpdir = tempfile.mkdtemp()
    try:
        embeddings = FakeEmbeddings()
        manager = VectorStoreManager(persist_dir=tmpdir, embeddings=embeddings)

        doc = Document(
            page_content="酒店A 隔音效果好，适合亲子出行，卫生条件优秀",
            metadata={"category": "hotel", "name": "酒店A", "tags": ["亲子", "隔音好"], "rating": 4.5},
        )
        ids = manager.add_to_preferences([doc])
        assert len(ids) == 1

        results = manager.get_preferences_collection().query(
            query_texts=["亲子 隔音 酒店"],
            n_results=3,
        )
        assert len(results["ids"][0]) >= 1
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_vector_store.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement embedding and vector store**

```python
# rag/__init__.py
# (empty)
```

```python
# rag/embedding.py
from langchain_community.embeddings import DashScopeEmbeddings


def create_embeddings(api_key: str, model: str = "text-embedding-v3") -> DashScopeEmbeddings:
    """创建阿里百炼 Embedding 实例"""
    return DashScopeEmbeddings(
        dashscope_api_key=api_key,
        model=model,
    )
```

```python
# rag/vector_store.py
import os
from chromadb import PersistentClient
from chromadb.config import Settings as ChromaSettings
from langchain.schema import Document
from langchain.vectorstores import Chroma


class VectorStoreManager:
    def __init__(self, persist_dir: str, embeddings):
        self.persist_dir = persist_dir
        self.embeddings = embeddings
        self._client = PersistentClient(
            path=persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )

    def get_preferences_collection(self) -> Chroma:
        return Chroma(
            collection_name="user_preferences",
            embedding_function=self.embeddings,
            client=self._client,
        )

    def get_cases_collection(self) -> Chroma:
        return Chroma(
            collection_name="travel_cases",
            embedding_function=self.embeddings,
            client=self._client,
        )

    def add_to_preferences(self, docs: list[Document]) -> list[str]:
        store = self.get_preferences_collection()
        return store.add_documents(docs)

    def add_to_cases(self, docs: list[Document]) -> list[str]:
        store = self.get_cases_collection()
        return store.add_documents(docs)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_vector_store.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add rag/__init__.py rag/embedding.py rag/vector_store.py tests/test_vector_store.py
git commit -m "feat: RAG百炼embedding + ChromaDB双Collection管理"
```

---

### Task 5: 数据加载器 (CSV/JSON/MD/TXT 解析)

**Files:**
- Create: `rag/data_loader.py`
- Create: `tests/fixtures/sample_preferences.csv`
- Create: `tests/fixtures/sample_case.md`
- Create: `tests/test_data_loader.py`

**Interfaces:**
- Consumes: (none — pure function)
- Produces:
  - `parse_csv_to_docs(file_path: str, category: str) -> list[Document]`
  - `parse_json_to_docs(file_path: str, collection_type: str) -> list[Document]`
  - `parse_md_to_doc(file_path: str) -> Document`
  - `load_file_to_docs(file_path: str, collection_type: str) -> list[Document]` — dispatcher

- [ ] **Step 1: Write test fixtures and test**

```python
# tests/test_data_loader.py
import os
import json
import tempfile
from rag.data_loader import (
    parse_csv_to_docs,
    parse_json_to_docs,
    parse_md_to_doc,
    load_file_to_docs,
)


def test_parse_csv_to_preferences_docs():
    csv_content = "category,name,tags,rating,text\nhotel,酒店A,亲子|隔音好,4.5,隔音效果很好适合亲子\n"
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="utf-8")
    tmp.write(csv_content)
    tmp.close()

    try:
        docs = parse_csv_to_docs(tmp.name, "hotel")
        assert len(docs) == 1
        assert docs[0].metadata["category"] == "hotel"
        assert docs[0].metadata["name"] == "酒店A"
        assert docs[0].metadata["tags"] == ["亲子", "隔音好"]
        assert docs[0].metadata["rating"] == 4.5
        assert "隔音效果很好" in docs[0].page_content
    finally:
        os.unlink(tmp.name)


def test_parse_json_to_preferences_docs():
    data = [
        {
            "category": "restaurant",
            "name": "YY火锅",
            "tags": ["川菜", "地道"],
            "rating": 4.8,
            "text": "正宗四川火锅，牛油锅底很香",
        }
    ]
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8")
    json.dump(data, tmp)
    tmp.close()

    try:
        docs = parse_json_to_docs(tmp.name, "preferences")
        assert len(docs) == 1
        assert docs[0].metadata["category"] == "restaurant"
    finally:
        os.unlink(tmp.name)


def test_parse_json_to_cases_docs():
    data = [
        {
            "destination": "成都",
            "days": 3,
            "season": "秋季",
            "budget_range": "3000-5000",
            "tags": ["美食", "休闲"],
            "rating": 4.8,
            "content": "# 成都3天2晚美食之旅\n\n## Day1\n...",
        }
    ]
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8")
    json.dump(data, tmp)
    tmp.close()

    try:
        docs = parse_json_to_docs(tmp.name, "cases")
        assert len(docs) == 1
        assert docs[0].metadata["destination"] == "成都"
        assert docs[0].metadata["days"] == 3
        assert "成都3天2晚美食之旅" in docs[0].page_content
    finally:
        os.unlink(tmp.name)


def test_parse_md_to_case_doc():
    md_content = "# 成都3天2晚美食之旅\n\n## Day1\n上午逛宽窄巷子，中午吃火锅。"
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False, encoding="utf-8")
    tmp.write(md_content)
    tmp.close()

    try:
        doc = parse_md_to_doc(tmp.name)
        assert "成都3天2晚美食之旅" in doc.page_content
        assert doc.metadata["source"] == "markdown_import"
    finally:
        os.unlink(tmp.name)


def test_load_file_to_docs_routes_correctly():
    csv_content = "category,name,tags,rating,text\nhotel,酒店A,亲子,4.5,不错\n"
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="utf-8")
    tmp.write(csv_content)
    tmp.close()

    try:
        docs = load_file_to_docs(tmp.name, "preferences")
        assert len(docs) == 1
    finally:
        os.unlink(tmp.name)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_data_loader.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement data loader**

```python
# rag/data_loader.py
import csv
import json
import os
from datetime import datetime
from langchain.schema import Document


def parse_csv_to_docs(file_path: str, category: str) -> list[Document]:
    """解析CSV为偏好库Document列表"""
    docs = []
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tags = [t.strip() for t in row.get("tags", "").split("|") if t.strip()]
            metadata = {
                "category": row.get("category", category),
                "name": row.get("name", ""),
                "tags": tags,
                "rating": float(row.get("rating", 0)),
                "source": "csv_upload",
                "created_at": datetime.now().isoformat(),
            }
            text = row.get("text", "")
            docs.append(Document(page_content=text, metadata=metadata))
    return docs


def parse_json_to_docs(file_path: str, collection_type: str) -> list[Document]:
    """解析JSON为Document列表，collection_type = preferences|cases"""
    docs = []
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        data = [data]

    for item in data:
        if collection_type == "preferences":
            metadata = {
                "category": item.get("category", ""),
                "name": item.get("name", ""),
                "tags": item.get("tags", []),
                "rating": float(item.get("rating", 0)),
                "source": "json_upload",
                "created_at": datetime.now().isoformat(),
            }
            text = item.get("text", "")
        else:  # cases
            metadata = {
                "destination": item.get("destination", ""),
                "days": int(item.get("days", 0)),
                "season": item.get("season", ""),
                "budget_range": item.get("budget_range", ""),
                "tags": item.get("tags", []),
                "rating": float(item.get("rating", 0)),
                "source": "json_upload",
                "created_at": datetime.now().isoformat(),
            }
            text = item.get("content", "")
        docs.append(Document(page_content=text, metadata=metadata))
    return docs


def parse_md_to_doc(file_path: str) -> Document:
    """解析Markdown文件为案例库Document"""
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    title = os.path.basename(file_path).replace(".md", "")
    first_line = content.strip().split("\n")[0].lstrip("#").strip()
    if first_line:
        title = first_line
    return Document(
        page_content=content,
        metadata={
            "destination": "",
            "days": 0,
            "season": "",
            "budget_range": "",
            "tags": [],
            "rating": 0,
            "source": "markdown_import",
            "created_at": datetime.now().isoformat(),
            "title": title,
        },
    )


def load_file_to_docs(file_path: str, collection_type: str) -> list[Document]:
    """根据文件扩展名自动选择解析器"""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".csv":
        return parse_csv_to_docs(file_path, "")
    elif ext == ".json":
        return parse_json_to_docs(file_path, collection_type)
    elif ext in (".md", ".txt"):
        return [parse_md_to_doc(file_path)]
    else:
        raise ValueError(f"不支持的文件类型: {ext}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_data_loader.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add rag/data_loader.py tests/test_data_loader.py
git commit -m "feat: RAG数据加载器，支持CSV/JSON/MD/TXT解析"
```

---

### Task 6: 去重逻辑 (MD5 + 字段匹配 + 语义)

**Files:**
- Create: `rag/dedup.py`
- Create: `tests/test_dedup.py`

**Interfaces:**
- Consumes: `rag.vector_store.VectorStoreManager` (for similarity search), `rag.embedding` (for embedding)
- Produces:
  - `md5_hash(text: str) -> str`
  - `check_md5_duplicate(text: str, collection: Chroma) -> bool`
  - `check_field_duplicate(metadata: dict, collection: Chroma, collection_type: str) -> bool`
  - `check_semantic_duplicate(text: str, collection: Chroma, threshold: float = 0.95) -> list[dict]`
  - `dedup_pipeline(doc: Document, collection: Chroma, collection_type: str, options: dict) -> dict` — returns {"status": "ok"|"duplicate"|"suspected", "duplicates": [...]}

- [ ] **Step 1: Write test for dedup functions**

```python
# tests/test_dedup.py
from rag.dedup import md5_hash, check_md5_duplicate


def test_md5_hash_deterministic():
    assert md5_hash("hello") == md5_hash("hello")
    assert md5_hash("hello") != md5_hash("world")


def test_md5_hash_same_text_same_hash():
    text = "酒店A隔音效果好，适合亲子"
    assert md5_hash(text) == md5_hash(text)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_dedup.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement dedup**

```python
# rag/dedup.py
import hashlib
from typing import Optional
from langchain.schema import Document


def md5_hash(text: str) -> str:
    """计算文本的MD5哈希"""
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def check_md5_duplicate(text: str, existing_texts: list[str]) -> bool:
    """检查文本MD5是否已存在于库中"""
    h = md5_hash(text)
    return any(md5_hash(t) == h for t in existing_texts)


def check_field_duplicate(
    metadata: dict,
    existing_metas: list[dict],
    collection_type: str,
) -> bool:
    """检查关键字段是否匹配"""
    if collection_type == "preferences":
        # category + name + text content (approximate)
        for em in existing_metas:
            if (
                em.get("category") == metadata.get("category")
                and em.get("name") == metadata.get("name")
            ):
                return True
    else:  # cases
        for em in existing_metas:
            if (
                em.get("destination") == metadata.get("destination")
                and em.get("days") == metadata.get("days")
                and em.get("title") == metadata.get("title")
            ):
                return True
    return False


def check_semantic_duplicate(
    text: str,
    collection,
    threshold: float = 0.95,
) -> list[dict]:
    """语义近重复检测，返回相似度≥threshold的文档列表"""
    try:
        results = collection.similarity_search_with_relevance_scores(
            text, k=5
        )
    except Exception:
        return []

    duplicates = []
    for doc, score in results:
        if score >= threshold:
            duplicates.append({"doc": doc, "score": score})
    return duplicates


def dedup_pipeline(
    doc: Document,
    collection,
    collection_type: str,
    existing_texts: list[str],
    existing_metas: list[dict],
    options: dict,
) -> dict:
    """
    去重流水线。
    options = {"md5": True, "field": False, "semantic": False, "semantic_threshold": 0.95}
    返回 {"status": "ok"|"duplicate"|"suspected", "duplicates": [...]}
    """
    text = doc.page_content

    # 第1关：MD5
    if options.get("md5", True):
        if check_md5_duplicate(text, existing_texts):
            return {"status": "duplicate", "duplicates": [], "reason": "MD5精确匹配"}

    # 第2关：字段匹配
    if options.get("field", False):
        if check_field_duplicate(doc.metadata, existing_metas, collection_type):
            return {"status": "suspected", "duplicates": [], "reason": "关键字段匹配"}

    # 第3关：语义去重
    if options.get("semantic", False):
        threshold = options.get("semantic_threshold", 0.95)
        dups = check_semantic_duplicate(text, collection, threshold)
        if dups:
            return {"status": "suspected", "duplicates": dups, "reason": f"语义相似度≥{threshold}"}

    return {"status": "ok", "duplicates": []}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_dedup.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add rag/dedup.py tests/test_dedup.py
git commit -m "feat: RAG三级去重(MD5精确/字段匹配/语义近重复)"
```

---

### Task 7: RAG 统一检索接口

**Files:**
- Create: `rag/retriever.py`
- Create: `tests/test_retriever.py`

**Interfaces:**
- Consumes: `rag.vector_store.VectorStoreManager`, `rag.embedding`
- Produces:
  - `class DualRetriever`: `__init__(self, vector_store: VectorStoreManager)`, `retrieve_preferences(self, query: str, category: str | None, k: int) -> list[Document]`, `retrieve_cases(self, query: str, k: int) -> list[Document]`, `retrieve_both(self, query: str, preferences_category: str | None, k_prefs: int, k_cases: int) -> dict` — returns {"preferences": [...], "cases": [...]}

- [ ] **Step 1: Write test for DualRetriever**

```python
# tests/test_retriever.py
import tempfile
import shutil
from unittest.mock import Mock, patch
from langchain.schema import Document
from rag.retriever import DualRetriever


class FakeVectorStore:
    def __init__(self):
        self._prefs = Mock()
        self._cases = Mock()

    def get_preferences_collection(self):
        return self._prefs

    def get_cases_collection(self):
        return self._cases


def test_retrieve_preferences_calls_similarity_search():
    vs = FakeVectorStore()
    fake_docs = [
        Document(page_content="酒店A隔音好", metadata={"category": "hotel", "name": "酒店A"})
    ]
    vs._prefs.similarity_search_with_relevance_scores.return_value = [
        (fake_docs[0], 0.85)
    ]

    retriever = DualRetriever(vs)
    results = retriever.retrieve_preferences("亲子 隔音 酒店", category="hotel", k=5)

    assert len(results) == 1
    vs._prefs.similarity_search_with_relevance_scores.assert_called_once()


def test_retrieve_both_returns_both_collections():
    vs = FakeVectorStore()
    vs._prefs.similarity_search_with_relevance_scores.return_value = [
        (Document(page_content="好评", metadata={}), 0.8)
    ]
    vs._cases.similarity_search_with_relevance_scores.return_value = [
        (Document(page_content="成都3天案例", metadata={}), 0.75)
    ]

    retriever = DualRetriever(vs)
    result = retriever.retrieve_both("成都 亲子", preferences_category=None, k_prefs=5, k_cases=3)

    assert "preferences" in result
    assert "cases" in result
    assert len(result["preferences"]) == 1
    assert len(result["cases"]) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_retriever.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement DualRetriever**

```python
# rag/retriever.py
from langchain.schema import Document
from rag.vector_store import VectorStoreManager


class DualRetriever:
    """统一的双库检索接口"""

    def __init__(self, vector_store: VectorStoreManager):
        self.vector_store = vector_store

    def retrieve_preferences(
        self, query: str, category: str | None = None, k: int = 5
    ) -> list[Document]:
        """检索偏好库，可选按category过滤"""
        collection = self.vector_store.get_preferences_collection()
        filter_dict = None
        if category:
            filter_dict = {"category": category}

        if filter_dict:
            results = collection.similarity_search(query, k=k, filter=filter_dict)
        else:
            results = collection.similarity_search(query, k=k)
        return results

    def retrieve_cases(self, query: str, k: int = 3) -> list[Document]:
        """检索案例库"""
        collection = self.vector_store.get_cases_collection()
        return collection.similarity_search(query, k=k)

    def retrieve_both(
        self,
        query: str,
        preferences_category: str | None = None,
        k_prefs: int = 5,
        k_cases: int = 3,
    ) -> dict:
        """同时检索两个库，返回 {"preferences": [...], "cases": [...]}"""
        prefs = self.retrieve_preferences(query, category=preferences_category, k=k_prefs)
        cases = self.retrieve_cases(query, k=k_cases)
        return {"preferences": prefs, "cases": cases}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_retriever.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add rag/retriever.py tests/test_retriever.py
git commit -m "feat: RAG双库统一检索接口DualRetriever"
```

---

### Task 8: LangGraph State 定义

**Files:**
- Create: `agents/__init__.py`
- Create: `agents/state.py`
- Create: `tests/test_state.py`

**Interfaces:**
- Consumes: (none — TypedDict definition)
- Produces: `agents.state.TravelPlanState(TypedDict)` with all fields defined in spec Section 2

- [ ] **Step 1: Write test for State structure**

```python
# tests/test_state.py
from agents.state import TravelPlanState


def test_travel_plan_state_has_required_fields():
    state = TravelPlanState(
        destination="成都",
        travel_date="2026-07-15",
        days=3,
        preferences="亲子、安静",
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
    assert state["destination"] == "成都"
    assert state["days"] == 3
    assert state["is_finalized"] is False


def test_state_is_mutable():
    state = TravelPlanState(
        destination="北京",
        travel_date="2026-08-01",
        days=5,
        preferences="历史文化",
        budget_total=8000.0,
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
    state["weather_report"] = "晴，28°C"
    assert state["weather_report"] == "晴，28°C"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_state.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement State**

```python
# agents/__init__.py
# (empty)
```

```python
# agents/state.py
from typing import TypedDict


class TravelPlanState(TypedDict):
    # === 用户输入 ===
    destination: str
    travel_date: str
    days: int
    preferences: str
    budget_total: float

    # === 子Agent输出 ===
    weather_report: str
    attractions: list[dict]
    restaurants: list[dict]
    hotels: list[dict]
    routes: list[dict]

    # === 汇总输出 ===
    final_report: str
    error_log: list[str]

    # === 微调对话 ===
    conversation: list[dict]
    is_finalized: bool
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_state.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add agents/__init__.py agents/state.py tests/test_state.py
git commit -m "feat: LangGraph共享状态TravelPlanState定义"
```

---

### Task 9: 三个子 Agent 节点 (天气 / 景点餐饮 / 酒店)

**Files:**
- Create: `agents/weather_agent.py`
- Create: `agents/attraction_agent.py`
- Create: `agents/hotel_agent.py`
- Create: `tests/test_sub_agents.py`

**Interfaces:**
- Consumes: `agents.state.TravelPlanState`, langchain tools, `rag.retriever.DualRetriever`
- Produces:
  - `weather_agent_node(state: TravelPlanState) -> dict` — returns {"weather_report": str}
  - `attraction_agent_node(state: TravelPlanState) -> dict` — returns {"attractions": list[dict], "restaurants": list[dict]}
  - `hotel_agent_node(state: TravelPlanState) -> dict` — returns {"hotels": list[dict]}

- [ ] **Step 1: Write test for sub-agent nodes**

```python
# tests/test_sub_agents.py
from unittest.mock import patch, Mock
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

    # Mock the graph-level singletons (lazy imports resolve to agents.graph)
    with patch("agents.graph._get_llm") as mock_llm_fn, \
         patch("agents.graph._get_tools") as mock_tools_fn, \
         patch("agents.graph._get_retriever") as mock_retriever_fn:
        mock_llm = Mock()
        mock_llm.invoke.return_value = Mock(content="成都7月15日天气：晴，28°C，适宜出行")
        mock_tools = [Mock(name="amap_weather")]
        mock_retriever = Mock()
        mock_retriever.retrieve_cases.return_value = []
        mock_llm_fn.return_value = mock_llm
        mock_tools_fn.return_value = mock_tools
        mock_retriever_fn.return_value = mock_retriever

        result = weather_agent_node(state)

    assert "weather_report" in result
    assert isinstance(result["weather_report"], str)
    assert len(result["weather_report"]) > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_sub_agents.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement sub-agent nodes**

```python
# agents/weather_agent.py
from agents.state import TravelPlanState


def weather_agent_node(state: TravelPlanState) -> dict:
    """
    天气Agent：调用高德天气API + RAG案例库，
    生成目的地天气报告和出行建议。
    """
    from agents.graph import _get_llm, _get_tools, _get_retriever

    llm = _get_llm()
    tools = _get_tools()
    retriever = _get_retriever()

    destination = state["destination"]
    travel_date = state["travel_date"]
    preferences = state.get("preferences", "")

    # RAG 检索案例库中同目的地的天气应对策略
    case_docs = retriever.retrieve_cases(
        f"{destination} {travel_date} 天气 出行准备", k=3
    )
    case_context = "\n".join([d.page_content[:500] for d in case_docs])

    # 绑定了天气工具
    llm_with_tools = llm.bind_tools([t for t in tools if t.name == "amap_weather"])

    prompt = f"""你是天气查询专家。请查询{destination}在{travel_date}前后的天气情况，并结合以下信息给出出行建议。

用户偏好: {preferences}
历史案例参考:
{case_context}

请输出:
1. 天气预报（温度、降水、风力）
2. 穿衣建议
3. 对行程的影响提示
"""
    response = llm_with_tools.invoke(prompt)
    return {"weather_report": response.content}
```

```python
# agents/attraction_agent.py
from agents.state import TravelPlanState


def attraction_agent_node(state: TravelPlanState) -> dict:
    """
    景点餐饮Agent：高德POI搜索 + RAG偏好库匹配 + RAG案例库参考，
    输出景点推荐和餐厅推荐。
    """
    from agents.graph import _get_llm, _get_tools, _get_retriever

    llm = _get_llm()
    tools = _get_tools()
    retriever = _get_retriever()

    destination = state["destination"]
    days = state["days"]
    preferences = state.get("preferences", "")

    # RAG 双检索
    rag_results = retriever.retrieve_both(
        f"{destination} {preferences} 景点 美食 餐厅",
        preferences_category=None,
        k_prefs=5,
        k_cases=3,
    )
    prefs_context = "\n".join(
        [f"[{d.metadata.get('category','')}] {d.metadata.get('name','')}: {d.page_content[:300]}"
         for d in rag_results["preferences"]]
    )
    cases_context = "\n".join([d.page_content[:500] for d in rag_results["cases"]])

    relevant_tools = [t for t in tools if t.name in ("amap_poi_search", "amap_multi_route")]
    llm_with_tools = llm.bind_tools(relevant_tools)

    prompt = f"""你是旅游规划专家。为{destination}规划{days}天的景点和餐厅。

用户偏好: {preferences}
偏好库匹配（用户评价标签）:
{prefs_context}
历史优秀案例参考:
{cases_context}

请:
1. 用amap_poi_search搜索景点(category=attraction)和餐厅(category=restaurant)
2. 结合偏好库标签筛选（如标签"适合看日落"优先推荐对应景点）
3. 用amap_multi_route规划每日景点串联路线
4. 输出JSON格式: {{"attractions": [...], "restaurants": [...]}}
每个推荐含: name, address, rating, reason(推荐理由), tags
"""
    response = llm_with_tools.invoke(prompt)

    # 解析JSON
    import json
    import re
    try:
        json_match = re.search(r"\{[\s\S]*\}", response.content)
        data = json.loads(json_match.group()) if json_match else {}
    except json.JSONDecodeError:
        data = {}

    return {
        "attractions": data.get("attractions", []),
        "restaurants": data.get("restaurants", []),
    }
```

```python
# agents/hotel_agent.py
from agents.state import TravelPlanState


def hotel_agent_node(state: TravelPlanState) -> dict:
    """
    酒店Agent：高德POI搜索 + RAG偏好库标签匹配 + RAG案例库参考，
    输出酒店推荐列表。
    """
    from agents.graph import _get_llm, _get_tools, _get_retriever

    llm = _get_llm()
    tools = _get_tools()
    retriever = _get_retriever()

    destination = state["destination"]
    preferences = state.get("preferences", "")

    # RAG 双检索
    rag_results = retriever.retrieve_both(
        f"{destination} {preferences} 酒店 住宿",
        preferences_category="hotel",
        k_prefs=5,
        k_cases=3,
    )
    prefs_context = "\n".join(
        [f"{d.metadata.get('name','')}: {d.page_content[:300]} [标签: {d.metadata.get('tags',[])}]"
         for d in rag_results["preferences"]]
    )
    cases_context = "\n".join([d.page_content[:500] for d in rag_results["cases"]])

    relevant_tools = [t for t in tools if t.name == "amap_poi_search"]
    llm_with_tools = llm.bind_tools(relevant_tools)

    prompt = f"""你是酒店推荐专家。为{destination}筛选合适的酒店。

用户偏好: {preferences}
偏好库匹配（用户评价+标签）:
{prefs_context}
历史优秀案例参考:
{cases_context}

请:
1. 用amap_poi_search搜索酒店(category=hotel)
2. 结合偏好库标签筛选（如用户偏好"亲子"→优先推荐标签含"隔音好""儿童乐园"的酒店）
3. 输出JSON格式: {{"hotels": [...]}}
每个推荐含: name, address, rating, price_range, reason(推荐理由，需引用偏好库标签), matched_tags
"""
    response = llm_with_tools.invoke(prompt)

    import json
    import re
    try:
        json_match = re.search(r"\{[\s\S]*\}", response.content)
        data = json.loads(json_match.group()) if json_match else {}
    except json.JSONDecodeError:
        data = {}

    return {"hotels": data.get("hotels", [])}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_sub_agents.py -v`
Expected: PASS (1 test)

- [ ] **Step 5: Commit**

```bash
git add agents/weather_agent.py agents/attraction_agent.py agents/hotel_agent.py tests/test_sub_agents.py
git commit -m "feat: 三个子Agent(天气/景点餐饮/酒店)节点实现"
```

---

### Task 10: 主控Agent + 汇总Agent + LangGraph 图组装

**Files:**
- Create: `agents/orchestrator.py`
- Create: `agents/synthesizer.py`
- Create: `agents/graph.py`
- Create: `tests/test_graph.py`

**Interfaces:**
- Consumes: all 3 sub-agent nodes, `agents.state.TravelPlanState`, tools, retrievers
- Produces:
  - `orchestrator_node(state: TravelPlanState) -> dict` — returns state fields for routing
  - `synthesizer_node(state: TravelPlanState) -> dict` — returns {"final_report": str, "routes": list[dict]}
  - `agents.graph.build_graph() -> CompiledStateGraph`
  - `agents.graph.run_travel_plan(user_input: dict) -> TravelPlanState`

- [ ] **Step 1: Write integration test for graph**

```python
# tests/test_graph.py
from unittest.mock import patch, Mock
from agents.state import TravelPlanState


def test_build_graph_compiles():
    from agents.graph import build_graph

    # Mock all dependencies to avoid real API calls
    with patch("agents.graph.ChatTongyi") as mock_llm, \
         patch("agents.graph.AmapClient") as mock_amap, \
         patch("agents.graph.create_amap_tools") as mock_create_tools, \
         patch("agents.graph.create_embeddings") as mock_embeddings, \
         patch("agents.graph.VectorStoreManager") as mock_vs, \
         patch("agents.graph.DualRetriever") as mock_retriever:

        mock_llm.return_value = Mock()
        mock_amap.return_value = Mock()
        mock_create_tools.return_value = [Mock(name=f"tool_{i}") for i in range(5)]
        mock_embeddings.return_value = Mock()
        mock_vs.return_value = Mock()
        mock_retriever.return_value = Mock()

        from config import Settings
        settings = Settings(
            bailian_api_key="test-key",
            amap_api_key="test-key",
        )

        graph = build_graph(settings)
        assert graph is not None
        # Graph should have the key nodes
        node_names = graph.get_graph().nodes.keys()
        print(f"Graph nodes: {list(node_names)}")


def test_orchestrator_node_parses_input():
    from agents.orchestrator import orchestrator_node
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
    # Orchestrator just validates and passes through
    result = orchestrator_node(state)
    assert "destination" in result
    assert result["destination"] == "成都"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_graph.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement orchestrator, synthesizer, and graph**

```python
# agents/orchestrator.py
from agents.state import TravelPlanState


def orchestrator_node(state: TravelPlanState) -> dict:
    """
    主控Agent：解析用户输入，验证必要字段，传递任务给子Agent。
    如果输入不完整或模糊，要求LLM追问用户。
    """
    destination = state.get("destination", "").strip()
    if not destination:
        error_log = state.get("error_log", [])
        error_log.append("主控Agent: 目的地为空，需要用户补充")
        return {"error_log": error_log}

    # 一切正常，透传state
    return {}
```

```python
# agents/synthesizer.py
from agents.state import TravelPlanState


def synthesizer_node(state: TravelPlanState) -> dict:
    """
    汇总Agent：整合天气、景点、酒店、餐厅数据，
    结合高德路线规划Tool生成完整Markdown旅行方案。
    """
    from agents.graph import _get_llm, _get_tools

    llm = _get_llm()
    tools = _get_tools()
    route_tools = [t for t in tools if t.name in ("amap_route_plan", "amap_geo_code")]
    llm_with_tools = llm.bind_tools(route_tools) if route_tools else llm

    destination = state["destination"]
    days = state["days"]
    budget = state["budget_total"]
    weather = state.get("weather_report", "")
    attractions = state.get("attractions", [])
    restaurants = state.get("restaurants", [])
    hotels = state.get("hotels", [])
    error_log = state.get("error_log", [])

    # 构建各部分文本
    weather_section = weather if weather else "⚠️ 天气数据暂不可用"
    attr_text = _format_list(attractions, "景点")
    rest_text = _format_list(restaurants, "餐厅")
    hotel_text = _format_list(hotels, "酒店")

    # 错误提示
    warnings = ""
    if error_log:
        warnings = "\n".join([f"> ⚠️ {e}" for e in error_log])

    prompt = f"""你是旅行方案汇总专家。请将以下信息整合为一份结构化的Markdown旅行方案。

目的地: {destination}
天数: {days}天
总预算: {budget}元

## 天气
{weather_section}

## 景点推荐
{attr_text}

## 餐厅推荐
{rest_text}

## 酒店推荐
{hotel_text}

{warnings}

请生成Markdown格式的完整旅行方案，包含:
1. 天气概况
2. 每日行程安排 (Day1, Day2, ...)
3. 酒店推荐
4. 美食推荐
5. 交通建议
6. 预算明细
7. 注意事项
"""
    response = llm.invoke(prompt)
    return {
        "final_report": response.content,
        "routes": [],  # 路线已在子Agent中通过amap_multi_route获取
    }


def _format_list(items: list[dict], label: str) -> str:
    if not items:
        return f"⚠️ {label}推荐暂不可用"
    lines = []
    for i, item in enumerate(items, 1):
        name = item.get("name", "未知")
        reason = item.get("reason", "")
        lines.append(f"{i}. **{name}** - {reason}")
    return "\n".join(lines)
```

```python
# agents/graph.py
from langgraph.graph import StateGraph, END
from langgraph.constants import Send
from langchain_community.chat_models.tongyi import ChatTongyi
from agents.state import TravelPlanState
from agents.orchestrator import orchestrator_node
from agents.weather_agent import weather_agent_node
from agents.attraction_agent import attraction_agent_node
from agents.hotel_agent import hotel_agent_node
from agents.synthesizer import synthesizer_node
from tools.amap_client import AmapClient
from tools.amap_tools import create_amap_tools
from rag.embedding import create_embeddings
from rag.vector_store import VectorStoreManager
from rag.retriever import DualRetriever
from config import Settings

# Module-level singletons (initialized on first use)
_llm = None
_tools = None
_retriever = None
_settings = None


def _init_dependencies(settings: Settings):
    global _llm, _tools, _retriever, _settings
    _settings = settings

    # LLM
    _llm = ChatTongyi(
        dashscope_api_key=settings.bailian_api_key,
        model=settings.llm_model,
    )

    # Tools
    amap_client = AmapClient(api_key=settings.amap_api_key)
    _tools = create_amap_tools(amap_client)

    # RAG
    embeddings = create_embeddings(
        api_key=settings.bailian_api_key,
        model=settings.embedding_model,
    )
    vector_store = VectorStoreManager(
        persist_dir=settings.chroma_persist_dir,
        embeddings=embeddings,
    )
    _retriever = DualRetriever(vector_store)


def _get_llm():
    return _llm


def _get_tools():
    return _tools


def _get_retriever():
    return _retriever


def _continue_to_sub_agents(state: TravelPlanState):
    """条件边：将任务扇出到三个子Agent"""
    return [
        Send("weather_agent", {}),
        Send("attraction_agent", {}),
        Send("hotel_agent", {}),
    ]


def build_graph(settings: Settings):
    """构建LangGraph状态图"""
    _init_dependencies(settings)

    graph = StateGraph(TravelPlanState)

    # 添加节点
    graph.add_node("orchestrator", orchestrator_node)
    graph.add_node("weather_agent", weather_agent_node)
    graph.add_node("attraction_agent", attraction_agent_node)
    graph.add_node("hotel_agent", hotel_agent_node)
    graph.add_node("synthesizer", synthesizer_node)

    # 边
    graph.set_entry_point("orchestrator")
    graph.add_conditional_edges("orchestrator", _continue_to_sub_agents, [
        "weather_agent",
        "attraction_agent",
        "hotel_agent",
    ])
    graph.add_edge("weather_agent", "synthesizer")
    graph.add_edge("attraction_agent", "synthesizer")
    graph.add_edge("hotel_agent", "synthesizer")
    graph.add_edge("synthesizer", END)

    return graph.compile()


def run_travel_plan(user_input: dict, settings: Settings) -> TravelPlanState:
    """执行一次旅行规划"""
    graph = build_graph(settings)
    initial_state = TravelPlanState(
        destination=user_input.get("destination", ""),
        travel_date=user_input.get("travel_date", ""),
        days=user_input.get("days", 3),
        preferences=user_input.get("preferences", ""),
        budget_total=float(user_input.get("budget_total", 0)),
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
    result = graph.invoke(initial_state)
    return result
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_graph.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add agents/orchestrator.py agents/synthesizer.py agents/graph.py tests/test_graph.py
git commit -m "feat: 主控Agent+汇总Agent+LangGraph状态图组装"
```

---

### Task 11: Streamlit 旅行规划主页面 (Home.py)

**Files:**
- Create: `Home.py`
- Create: `tests/test_home.py`

**Interfaces:**
- Consumes: `agents.graph.run_travel_plan`, `config.Settings`
- Produces: 4-step Streamlit UI (输入 → 进度 → 报告 → 对话微调), session_state manages graph state

- [ ] **Step 1: Write test for Home page logic**

```python
# tests/test_home.py
from unittest.mock import patch, Mock
import json


def test_session_history_save_and_load():
    """测试历史会话JSON的保存和加载格式"""
    import tempfile
    import os
    from datetime import datetime

    session = {
        "session_id": "20260715_143022",
        "created_at": "2026-07-15T14:30:22",
        "status": "confirmed",
        "input": {"destination": "成都", "travel_date": "2026-07-20", "days": 3},
        "initial_report": "# 方案",
        "conversation": [],
        "final_report": "# 最终方案",
    }

    tmpdir = tempfile.mkdtemp()
    try:
        filepath = os.path.join(tmpdir, "20260715_143022_成都_3天.json")
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(session, f, ensure_ascii=False, indent=2)

        with open(filepath, "r", encoding="utf-8") as f:
            loaded = json.load(f)

        assert loaded["session_id"] == "20260715_143022"
        assert loaded["status"] == "confirmed"
        assert loaded["conversation"] == []
    finally:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_home.py -v`
Expected: FAIL — `ModuleNotFoundError` (no test file existed before)

- [ ] **Step 3: Implement Home.py**

```python
# Home.py
import streamlit as st
import json
import os
from datetime import datetime
from config import Settings
from agents.graph import run_travel_plan

st.set_page_config(
    page_title="Multi-Agent 智能旅行规划",
    page_icon="🧭",
    layout="wide",
)

# --- 初始化 ---
settings = Settings.from_env()

if "plan_state" not in st.session_state:
    st.session_state.plan_state = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "final_report" not in st.session_state:
    st.session_state.final_report = ""

st.title("🧭 Multi-Agent 智能旅行规划")
st.caption("输入需求，AI自动生成覆盖天气、景点、酒店、餐饮、交通、预算的完整旅行方案")

# === 第1步：输入需求 ===
st.header("第1步：输入你的旅行需求")

col1, col2, col3 = st.columns(3)
with col1:
    destination = st.text_input("目的地", placeholder="如：成都")
with col2:
    travel_date = st.date_input("出发日期")
with col3:
    days = st.number_input("天数", min_value=1, max_value=30, value=3)

col4, col5 = st.columns(2)
with col4:
    budget_min, budget_max = st.slider("预算区间(元)", 0, 50000, (3000, 8000), step=500)

preferences = st.text_area("偏好", placeholder="如：亲子、安静、美食、适合老人")

plan_clicked = st.button("🚀 开始规划", type="primary", use_container_width=True)

# === 第2步：AI执行过程 ===
if plan_clicked:
    st.header("第2步：AI 执行过程")
    progress_container = st.container()

    with progress_container:
        status_placeholder = st.empty()

        if not settings.bailian_api_key or not settings.amap_api_key:
            st.error("""
            ⚠️ 请先在项目根目录创建 `.env` 文件，配置 API Key：
            ```
            BAILIAN_API_KEY=your_bailian_api_key_here
            AMAP_API_KEY=your_amap_api_key_here
            ```
            """)
        else:
            status_placeholder.info("🔄 主控Agent 解析需求...")
            try:
                user_input = {
                    "destination": destination,
                    "travel_date": str(travel_date),
                    "days": days,
                    "preferences": preferences,
                    "budget_total": float(budget_max),
                }

                result = run_travel_plan(user_input, settings)
                st.session_state.plan_state = result
                st.session_state.final_report = result.get("final_report", "")
                st.session_state.chat_history = []

                status_placeholder.success("✅ 方案生成完成！")
            except Exception as e:
                status_placeholder.error(f"❌ 生成失败: {str(e)}")
                st.stop()

# === 第3步：展示方案 ===
if st.session_state.final_report:
    st.header("第3步：旅行方案")
    st.markdown(st.session_state.final_report)

    col_actions = st.columns(4)
    with col_actions[0]:
        st.download_button(
            "📥 下载 Markdown",
            data=st.session_state.final_report,
            file_name=f"{destination}_{days}天_旅行方案.md",
            mime="text/markdown",
        )
    with col_actions[1]:
        if st.button("📋 复制"):
            st.toast("已复制到剪贴板")

    # === 第4步：微调对话 ===
    st.header("第4步：微调优化")
    st.caption("对方案有修改意见？在这里和AI对话调整")

    # 显示对话历史
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # 用户输入
    user_feedback = st.chat_input("输入修改意见...")
    if user_feedback:
        st.session_state.chat_history.append({"role": "user", "content": user_feedback})

        # 再次调用LLM处理修改
        from agents.graph import _get_llm
        llm = _get_llm()

        current_report = st.session_state.final_report
        history_str = "\n".join(
            [f"{m['role']}: {m['content']}" for m in st.session_state.chat_history[-10:]]
        )

        refine_prompt = f"""以下是当前的旅行方案:
{current_report}

用户提出了新的修改意见。请根据修改意见更新方案，只修改用户要求的部分，其他保持不变。

对话历史:
{history_str}

请输出更新后的完整Markdown方案。"""
        response = llm.invoke(refine_prompt)
        st.session_state.final_report = response.content
        st.session_state.chat_history.append(
            {"role": "assistant", "content": f"方案已更新。\n\n{response.content[:200]}..."}
        )
        st.rerun()

    # 确认按钮
    if st.button("✅ 确认最终方案，保存到历史", type="primary"):
        session_data = {
            "session_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "created_at": datetime.now().isoformat(),
            "status": "confirmed",
            "input": {
                "destination": destination,
                "travel_date": str(travel_date),
                "days": days,
                "preferences": preferences,
                "budget": [budget_min, budget_max],
            },
            "initial_report": st.session_state.plan_state.get("final_report", ""),
            "conversation": st.session_state.chat_history,
            "final_report": st.session_state.final_report,
        }
        os.makedirs(settings.history_dir, exist_ok=True)
        filename = f"{session_data['session_id']}_{destination}_{days}天.json"
        filepath = os.path.join(settings.history_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(session_data, f, ensure_ascii=False, indent=2)
        st.success("✅ 方案已保存到历史记录！")
        st.balloons()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_home.py -v`
Expected: PASS (1 test)

- [ ] **Step 5: Commit**

```bash
git add Home.py tests/test_home.py
git commit -m "feat: Streamlit旅行规划首页(4步交互+对话微调)"
```

---

### Task 12: Streamlit 知识库管理页 + 历史记录页

**Files:**
- Create: `pages/__init__.py`
- Create: `pages/01_Knowledge_Base.py`
- Create: `pages/02_History.py`

**Interfaces:**
- Consumes: `rag.vector_store.VectorStoreManager`, `rag.data_loader`, `rag.dedup`, `config.Settings`
- Produces: Two Streamlit pages with full CRUD for knowledge bases and history browsing

- [ ] **Step 1: Implement Knowledge Base page**

```python
# pages/__init__.py
# (empty)
```

```python
# pages/01_Knowledge_Base.py
import streamlit as st
import os
import json
from datetime import datetime
from config import Settings
from rag.embedding import create_embeddings
from rag.vector_store import VectorStoreManager
from rag.data_loader import load_file_to_docs
from rag.dedup import md5_hash, dedup_pipeline
from langchain.schema import Document

st.set_page_config(page_title="知识库管理", page_icon="📚")

settings = Settings.from_env()

# 初始化
@st.cache_resource
def get_vector_store():
    embeddings = create_embeddings(settings.bailian_api_key, settings.embedding_model)
    return VectorStoreManager(settings.chroma_persist_dir, embeddings)

vs = get_vector_store()

st.title("📚 知识库管理")

# 状态概览
try:
    prefs_count = vs.get_preferences_collection()._collection.count()
except Exception:
    prefs_count = 0
try:
    cases_count = vs.get_cases_collection()._collection.count()
except Exception:
    cases_count = 0

col1, col2 = st.columns(2)
with col1:
    st.metric("偏好库", f"{prefs_count} 条")
with col2:
    st.metric("案例库", f"{cases_count} 条")

st.divider()

# 录入目标库选择
collection_type = st.radio("选择目标库", ["preferences", "cases"],
                           format_func=lambda x: "偏好库 (酒店/景点/餐厅评价)" if x == "preferences" else "案例库 (优质旅行方案)")

tab1, tab2 = st.tabs(["📝 手动录入", "📤 文件上传"])

# --- 手动录入 ---
with tab1:
    if collection_type == "preferences":
        with st.form("pref_form"):
            category = st.selectbox("品类", ["hotel", "restaurant", "attraction"],
                                    format_func=lambda x: {"hotel": "酒店", "restaurant": "餐厅", "attraction": "景点"}[x])
            name = st.text_input("名称")
            tags = st.text_input("标签（用竖线|分隔）", placeholder="亲子|隔音好|卫生好")
            rating = st.slider("评分", 0.0, 5.0, 4.0, 0.5)
            text = st.text_area("评价文本")
            submitted = st.form_submit_button("提交入库")
            if submitted and name and text:
                doc = Document(
                    page_content=text,
                    metadata={
                        "category": category, "name": name,
                        "tags": [t.strip() for t in tags.split("|") if t.strip()],
                        "rating": rating, "source": "manual",
                        "created_at": datetime.now().isoformat(),
                    }
                )
                vs.add_to_preferences([doc])
                st.success(f"✅ '{name}' 已入库")
                st.rerun()
    else:
        with st.form("case_form"):
            destination = st.text_input("目的地")
            days = st.number_input("天数", 1, 30, 3)
            season = st.selectbox("季节", ["春季", "夏季", "秋季", "冬季"])
            budget_range = st.text_input("预算区间", "3000-5000")
            tags = st.text_input("标签（用竖线|分隔）", placeholder="美食|休闲")
            rating = st.slider("案例质量分", 0.0, 5.0, 4.0, 0.5)
            content = st.text_area("完整旅行方案 (Markdown)", height=300)
            submitted = st.form_submit_button("提交入库")
            if submitted and destination and content:
                doc = Document(
                    page_content=content,
                    metadata={
                        "destination": destination, "days": days,
                        "season": season, "budget_range": budget_range,
                        "tags": [t.strip() for t in tags.split("|") if t.strip()],
                        "rating": rating, "source": "manual",
                        "created_at": datetime.now().isoformat(),
                    }
                )
                vs.add_to_cases([doc])
                st.success(f"✅ '{destination} {days}天' 案例已入库")
                st.rerun()

# --- 文件上传 ---
with tab2:
    st.caption(f"上传文件到{collection_type}库")
    supported = "CSV, JSON" if collection_type == "preferences" else "JSON, Markdown, TXT, CSV"
    st.info(f"支持格式: {supported}")

    uploaded_file = st.file_uploader("选择文件", type=["csv", "json", "md", "txt"])
    if uploaded_file:
        # Save to temp
        tmp_path = f"/tmp/{uploaded_file.name}"
        with open(tmp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        try:
            docs = load_file_to_docs(tmp_path, collection_type)
            if collection_type == "preferences":
                vs.add_to_preferences(docs)
            else:
                vs.add_to_cases(docs)
            st.success(f"✅ 成功导入 {len(docs)} 条数据")
        except Exception as e:
            st.error(f"导入失败: {e}")
        finally:
            os.unlink(tmp_path)

st.divider()

# 已有数据概览
st.subheader("📋 已有数据")
search = st.text_input("搜索", placeholder="输入关键词...")

if search:
    collection = vs.get_preferences_collection() if collection_type == "preferences" else vs.get_cases_collection()
    results = collection.similarity_search(search, k=10)
    for doc in results:
        meta = doc.metadata
        with st.expander(f"{meta.get('name', meta.get('destination', '未命名'))} - {meta.get('rating', 'N/A')}★"):
            st.write(doc.page_content)
            st.caption(f"标签: {meta.get('tags', [])}")
```

```python
# pages/02_History.py
import streamlit as st
import json
import os
from config import Settings

st.set_page_config(page_title="历史记录", page_icon="📋")

settings = Settings.from_env()

st.title("📋 历史旅行方案")

history_dir = settings.history_dir
if not os.path.exists(history_dir):
    st.info("暂无历史记录，去首页生成你的第一份旅行方案吧！")
    st.stop()

files = sorted(
    [f for f in os.listdir(history_dir) if f.endswith(".json")],
    reverse=True,
)

if not files:
    st.info("暂无历史记录")
    st.stop()

search_term = st.text_input("搜索目的地...")

for filename in files:
    filepath = os.path.join(history_dir, filename)
    with open(filepath, "r", encoding="utf-8") as f:
        session = json.load(f)

    destination = session.get("input", {}).get("destination", "未知")
    if search_term and search_term not in destination:
        continue

    days = session.get("input", {}).get("days", "?")
    budget = session.get("input", {}).get("budget", [0, 0])
    conv_count = len(session.get("conversation", []))
    status = "✅已确认" if session.get("status") == "confirmed" else "⚠️未确认"

    with st.expander(
        f"{session.get('created_at','')[:16]} | {destination} {days}天 | {conv_count}轮对话 | {status}"
    ):
        tabs = st.tabs(["最终方案", "对话历史", "初始方案"])

        with tabs[0]:
            if session.get("final_report"):
                st.markdown(session["final_report"])
                st.download_button(
                    "📥 下载 Markdown",
                    data=session["final_report"],
                    file_name=f"{destination}_{days}天_旅行方案.md",
                    key=f"dl_{filename}",
                )

        with tabs[1]:
            for msg in session.get("conversation", []):
                role = "🙋" if msg["role"] == "user" else "🤖"
                st.caption(f"{role} {msg.get('timestamp', '')}")
                st.write(msg["content"][:500])
                st.divider()
            if not session.get("conversation"):
                st.caption("(无对话记录)")

        with tabs[2]:
            if session.get("initial_report"):
                st.markdown(session["initial_report"][:1000])
            else:
                st.caption("(无初始方案)")

        if st.button("🗑️ 删除此记录", key=f"del_{filename}"):
            os.remove(filepath)
            st.success("已删除")
            st.rerun()
```

- [ ] **Step 2: Commit**

```bash
git add pages/__init__.py pages/01_Knowledge_Base.py pages/02_History.py
git commit -m "feat: Streamlit知识库管理页+历史记录页"
```

---

## Verification

### 端到端测试验证步骤

1. **环境准备**
   ```bash
   cp .env.example .env
   # 编辑.env填入真实API Key
   pip install -r requirements.txt
   ```

2. **启动应用**
   ```bash
   streamlit run Home.py
   ```

3. **知识库录入验证**
   - 打开知识库管理页 → 手动录入一条酒店评价（如"酒店A，亲子|隔音好，4.5★"）
   - 上传一条CSV批量导入
   - 验证去重：再次录入相同内容，应提示MD5重复

4. **旅行规划验证**
   - 首页输入"成都，3天，亲子，3000-5000元"→ 点击开始规划
   - 验证执行过程实时展示
   - 验证Markdown报告包含天气/景点/酒店/餐饮/交通/预算6个章节
   - 验证下载按钮可用

5. **对话微调验证**
   - 在聊天框输入"把第二天酒店换到春熙路附近"→ 发送
   - 验证方案更新，酒店部分变化，其他部分不变
   - 点击"确认最终方案"→ 验证历史页出现该记录

6. **历史记录验证**
   - 打开历史记录页 → 验证刚才的方案显示
   - 展开查看对话历史 → 验证每轮对话可见
   - 切换查看版本（初始/最终）→ 验证内容不同
   - 下载Markdown → 验证文件完整

7. **错误处理验证**
   - 临时删除 `.env` 中的 API Key → 验证首页显示配置引导而非崩溃
   - 输入空白目的地 → 验证主控Agent提示需要补充
```

- [ ] **Step 3: Commit**

```bash
git add docs/superpowers/plans/2026-06-24-travel-agent.md
git commit -m "docs: Multi-Agent旅行规划系统完整实现计划"
```
