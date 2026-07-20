import os
from config import Settings

import pytest


@pytest.fixture(autouse=True)
def _protect_env():
    snapshot = dict(os.environ)
    yield
    os.environ.clear()
    os.environ.update(snapshot)


def test_settings_from_env_reads_all_fields():
    env_vars = {
        "BAILIAN_API_KEY": "sk-test-bailian",
        "AMAP_API_KEY": "amap-test-key",
        "LLM_MODEL": "qwen-max",
        "EMBEDDING_MODEL": "text-embedding-v3",
        "CHROMA_PERSIST_DIR": "./storage",
        "HISTORY_DIR": "./data/history",
    }
    for k, v in env_vars.items():
        os.environ[k] = v

    settings = Settings.from_env()

    assert settings.bailian_api_key == "sk-test-bailian"
    assert settings.amap_api_key == "amap-test-key"
    assert settings.llm_model == "qwen-max"
    assert settings.embedding_model == "text-embedding-v3"
    assert settings.top_k_preferences == 5
    assert settings.top_k_cases == 3
    assert settings.similarity_threshold == 0.45


def test_settings_uses_defaults_when_env_missing():
    for k in ["BAILIAN_API_KEY", "AMAP_API_KEY", "LLM_MODEL"]:
        os.environ.pop(k, None)

    settings = Settings.from_env()

    assert settings.llm_model == "qwen-max"
    assert settings.top_k_preferences == 5
    assert settings.top_k_cases == 3
    assert settings.similarity_threshold == 0.45
