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
