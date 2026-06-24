import hashlib
from typing import Optional
from langchain_core.documents import Document


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
