import csv
import json
import os
from datetime import datetime
from langchain_core.documents import Document


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
