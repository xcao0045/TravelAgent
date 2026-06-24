from rag.dedup import md5_hash, check_md5_duplicate


def test_md5_hash_deterministic():
    assert md5_hash("hello") == md5_hash("hello")
    assert md5_hash("hello") != md5_hash("world")


def test_md5_hash_same_text_same_hash():
    text = "酒店A隔音效果好，适合亲子"
    assert md5_hash(text) == md5_hash(text)
