from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

from rag.dedup import (
    check_field_duplicate,
    check_md5_duplicate,
    check_semantic_duplicate,
    dedup_pipeline,
    md5_hash,
)


# ---------------------------------------------------------------------------
# md5_hash
# ---------------------------------------------------------------------------

def test_md5_hash_deterministic():
    assert md5_hash("hello") == md5_hash("hello")
    assert md5_hash("hello") != md5_hash("world")


def test_md5_hash_same_text_same_hash():
    text = "酒店A隔音效果好，适合亲子"
    assert md5_hash(text) == md5_hash(text)


# ---------------------------------------------------------------------------
# check_md5_duplicate
# ---------------------------------------------------------------------------

def test_check_md5_duplicate_when_duplicate():
    """文本已存在于列表中时应返回 True"""
    existing = ["hello", "world"]
    assert check_md5_duplicate("hello", existing) is True


def test_check_md5_duplicate_when_not_duplicate():
    """文本不存在于列表中时应返回 False"""
    existing = ["hello", "world"]
    assert check_md5_duplicate("foo", existing) is False


def test_check_md5_duplicate_empty_list():
    """空列表时应返回 False"""
    assert check_md5_duplicate("hello", []) is False


# ---------------------------------------------------------------------------
# check_field_duplicate
# ---------------------------------------------------------------------------

class TestCheckFieldDuplicate:
    """check_field_duplicate — preferences 分支"""

    def test_prefs_matching_category_and_name(self):
        meta = {"category": "hotel", "name": "希尔顿"}
        existing = [
            {"category": "hotel", "name": "希尔顿", "other": "无关字段"},
            {"category": "restaurant", "name": "麦当劳"},
        ]
        assert check_field_duplicate(meta, existing, "preferences") is True

    def test_prefs_non_matching(self):
        meta = {"category": "hotel", "name": "希尔顿"}
        existing = [
            {"category": "hotel", "name": "万豪"},
            {"category": "restaurant", "name": "麦当劳"},
        ]
        assert check_field_duplicate(meta, existing, "preferences") is False

    """check_field_duplicate — cases 分支"""

    def test_cases_matching_destination_days_title(self):
        meta = {"destination": "东京", "days": 5, "title": "东京五日游"}
        existing = [
            {"destination": "东京", "days": 5, "title": "东京五日游", "other": "无关"},
            {"destination": "大阪", "days": 3, "title": "大阪三日"},
        ]
        assert check_field_duplicate(meta, existing, "cases") is True

    def test_cases_non_matching(self):
        meta = {"destination": "东京", "days": 5, "title": "东京五日游"}
        existing = [
            {"destination": "东京", "days": 3, "title": "东京三日"},
            {"destination": "大阪", "days": 5, "title": "大阪五日游"},
        ]
        assert check_field_duplicate(meta, existing, "cases") is False

    """check_field_duplicate — unknown collection_type"""

    def test_unknown_collection_type_returns_false(self):
        meta = {"category": "hotel", "name": "希尔顿"}
        existing = [{"category": "hotel", "name": "希尔顿"}]
        assert check_field_duplicate(meta, existing, "unknown") is False


# ---------------------------------------------------------------------------
# check_semantic_duplicate
# ---------------------------------------------------------------------------

def test_check_semantic_duplicate_returns_empty_on_exception():
    """异常时应 graceful 返回 [] 而不是崩溃"""
    coll = MagicMock()
    coll.similarity_search_with_relevance_scores.side_effect = Exception("boom")
    result = check_semantic_duplicate("some text", coll)
    assert result == []


# ---------------------------------------------------------------------------
# dedup_pipeline
# ---------------------------------------------------------------------------

class TestDedupPipeline:
    def test_md5_ok_path(self):
        """MD5 未命中时 pipeline 应返回 status=ok"""
        doc = Document(page_content="unique text", metadata={})
        result = dedup_pipeline(
            doc=doc,
            collection=MagicMock(),
            collection_type="preferences",
            existing_texts=["other text"],
            existing_metas=[],
            options={"md5": True, "field": False, "semantic": False},
        )
        assert result["status"] == "ok"

    def test_md5_duplicate_path(self):
        """MD5 命中时应返回 status=duplicate"""
        doc = Document(page_content="dupe", metadata={})
        result = dedup_pipeline(
            doc=doc,
            collection=MagicMock(),
            collection_type="preferences",
            existing_texts=["dupe"],
            existing_metas=[],
            options={"md5": True, "field": False, "semantic": False},
        )
        assert result["status"] == "duplicate"
        assert result["reason"] == "MD5精确匹配"

    def test_field_duplicate_path(self):
        """字段匹配命中时应返回 status=suspected"""
        doc = Document(
            page_content="some content",
            metadata={"category": "hotel", "name": "希尔顿"},
        )
        existing_metas = [{"category": "hotel", "name": "希尔顿"}]
        result = dedup_pipeline(
            doc=doc,
            collection=MagicMock(),
            collection_type="preferences",
            existing_texts=["other text"],
            existing_metas=existing_metas,
            options={"md5": True, "field": True, "semantic": False},
        )
        assert result["status"] == "suspected"
        assert result["reason"] == "关键字段匹配"

    def test_semantic_duplicate_path(self):
        """语义匹配命中时应返回 status=suspected"""
        doc = Document(page_content="similar text", metadata={})
        coll = MagicMock()
        coll.similarity_search_with_relevance_scores.return_value = [
            (Document(page_content="existing"), 0.97),
        ]
        result = dedup_pipeline(
            doc=doc,
            collection=coll,
            collection_type="cases",
            existing_texts=["other text"],
            existing_metas=[],
            options={"md5": True, "field": False, "semantic": True, "semantic_threshold": 0.95},
        )
        assert result["status"] == "suspected"
        assert result["reason"] == "语义相似度≥0.95"

    def test_md5_disabled_then_no_duplicate(self):
        """MD5 关卡关闭后，即使有相同文本也不报 duplicate"""
        doc = Document(page_content="same", metadata={})
        result = dedup_pipeline(
            doc=doc,
            collection=MagicMock(),
            collection_type="preferences",
            existing_texts=["same"],
            existing_metas=[],
            options={"md5": False, "field": False, "semantic": False},
        )
        assert result["status"] == "ok"
