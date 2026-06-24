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
