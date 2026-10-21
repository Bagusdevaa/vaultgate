"""Load config.yaml and secrets from .env into one object."""

import os
from dataclasses import dataclass
from pathlib import Path

import yaml
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Config:
    settings: dict
    openrouter_api_key: str | None
    vault_source_path: Path | None

    def __getitem__(self, section: str) -> dict:
        return self.settings[section]


def load_config(config_path: Path | None = None, env_path: Path | None = None) -> Config:
    config_path = config_path or REPO_ROOT / "config.yaml"
    load_dotenv(env_path or REPO_ROOT / ".env")
    settings = yaml.safe_load(config_path.read_text())
    vault = os.getenv("VAULT_SOURCE_PATH")
    return Config(
        settings=settings,
        # Empty strings from a copied .env.example count as "not set".
        openrouter_api_key=os.getenv("OPENROUTER_API_KEY") or None,
        vault_source_path=Path(vault) if vault else None,
    )
