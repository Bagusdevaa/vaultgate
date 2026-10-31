from pathlib import Path

from vault_rag.config import load_config


def test_repo_config_loads():
    cfg = load_config(env_path=Path("/nonexistent/.env"))
    assert cfg["llm"]["temperature"] == 0
    assert "qwen" not in cfg["llm"]["judge"].lower()
    assert cfg["gate"]["type"] in {"jev", "bge", "none"}


def test_blank_keys_count_as_missing(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    env = tmp_path / ".env"
    env.write_text("OPENROUTER_API_KEY=\n")
    cfg = load_config(env_path=env)
    assert cfg.openrouter_api_key is None


def test_qdrant_url_can_come_from_the_environment(monkeypatch):
    monkeypatch.setenv("VAULTGATE_QDRANT_URL", "http://qdrant:6333")
    assert load_config(env_path=Path("/nonexistent/.env"))["qdrant"]["url"] == "http://qdrant:6333"
