"""Project configuration helpers."""
from __future__ import annotations

import os
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"


def load_yaml(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_paths(env: str | None = None) -> dict:
    """Return the `paths.yaml` section for the current machine (LIVER_ENV, default 'local')."""
    cfg = load_yaml(CONFIG_DIR / "paths.yaml")
    env = env or os.environ.get("LIVER_ENV", "local")
    section = dict(cfg[env])
    section["project"] = {k: PROJECT_ROOT / v for k, v in cfg["project"].items()}
    return section
