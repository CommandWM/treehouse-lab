from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

SETTINGS_DIRNAME = ".treehouse_lab"
SETTINGS_FILENAME = "llm_settings.json"

DEFAULT_LLM_SETTINGS: dict[str, Any] = {
    "provider": "",
    "model": "",
    "loop_llm_selection": False,
    "ollama_base_url": "",
    "ollama_api_key": "",
    "agent_cli": "",
    "openai_compatible_base_url": "",
    "openai_compatible_api_key": "",
    "openai_api_key": "",
}


def llm_settings_path(project_root: Path) -> Path:
    return project_root / SETTINGS_DIRNAME / SETTINGS_FILENAME


def load_llm_settings(project_root: Path) -> dict[str, Any]:
    settings_path = llm_settings_path(project_root)
    if not settings_path.exists():
        return dict(DEFAULT_LLM_SETTINGS)

    try:
        payload = json.loads(settings_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return dict(DEFAULT_LLM_SETTINGS)

    return _normalize_settings(payload)


def save_llm_settings(project_root: Path, payload: dict[str, Any]) -> Path:
    normalized = _normalize_settings(payload)
    settings_path = llm_settings_path(project_root)
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    settings_path.write_text(json.dumps(normalized, indent=2, sort_keys=True), encoding="utf-8")
    return settings_path


def effective_llm_settings(project_root: Path, defaults: dict[str, Any]) -> dict[str, Any]:
    saved = load_llm_settings(project_root)
    resolved = dict(DEFAULT_LLM_SETTINGS)
    env_map = {
        "provider": "TREEHOUSE_LAB_LLM_PROVIDER",
        "model": "TREEHOUSE_LAB_LLM_MODEL",
        "ollama_base_url": "TREEHOUSE_LAB_OLLAMA_BASE_URL",
        "ollama_api_key": "OLLAMA_API_KEY",
        "agent_cli": "TREEHOUSE_LAB_AGENT_CLI",
        "openai_compatible_base_url": "TREEHOUSE_LAB_OPENAI_COMPATIBLE_BASE_URL",
        "openai_compatible_api_key": "TREEHOUSE_LAB_OPENAI_COMPATIBLE_API_KEY",
        "openai_api_key": "OPENAI_API_KEY",
    }

    for key, default_value in defaults.items():
        saved_value = saved.get(key, "")
        if isinstance(default_value, bool):
            resolved[key] = bool(saved_value)
            continue
        if str(saved_value).strip():
            resolved[key] = str(saved_value).strip()
            continue
        env_name = env_map.get(key)
        env_value = "" if env_name is None else os.getenv(env_name, "").strip()
        resolved[key] = env_value or default_value

    return resolved


def _normalize_settings(payload: dict[str, Any] | None) -> dict[str, Any]:
    raw = {} if payload is None else dict(payload)
    normalized = dict(DEFAULT_LLM_SETTINGS)
    normalized.update(
        {
            "provider": str(raw.get("provider", "")).strip(),
            "model": str(raw.get("model", "")).strip(),
            "loop_llm_selection": bool(raw.get("loop_llm_selection", False)),
            "ollama_base_url": str(raw.get("ollama_base_url", "")).strip(),
            "ollama_api_key": str(raw.get("ollama_api_key", "")).strip(),
            "agent_cli": str(raw.get("agent_cli", "")).strip(),
            "openai_compatible_base_url": str(raw.get("openai_compatible_base_url", "")).strip(),
            "openai_compatible_api_key": str(raw.get("openai_compatible_api_key", "")).strip(),
            "openai_api_key": str(raw.get("openai_api_key", "")).strip(),
        }
    )
    return normalized
