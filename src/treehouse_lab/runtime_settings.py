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

OLLAMA_API_KEY_ENV_VARS = ("OLLAMA_API_KEY", "OLLAMA_CLOUD_KEY", "VIOLAAMA_CLOUD_KEY")


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
    has_saved_settings = llm_settings_path(project_root).exists()
    resolved = dict(DEFAULT_LLM_SETTINGS)
    env_map: dict[str, str | tuple[str, ...]] = {
        "provider": "TREEHOUSE_LAB_LLM_PROVIDER",
        "model": "TREEHOUSE_LAB_LLM_MODEL",
        "loop_llm_selection": "TREEHOUSE_LAB_LOOP_LLM_SELECTION",
        "ollama_base_url": "TREEHOUSE_LAB_OLLAMA_BASE_URL",
        "ollama_api_key": OLLAMA_API_KEY_ENV_VARS,
        "agent_cli": "TREEHOUSE_LAB_AGENT_CLI",
        "openai_compatible_base_url": "TREEHOUSE_LAB_OPENAI_COMPATIBLE_BASE_URL",
        "openai_compatible_api_key": "TREEHOUSE_LAB_OPENAI_COMPATIBLE_API_KEY",
        "openai_api_key": "OPENAI_API_KEY",
    }

    for key, default_value in defaults.items():
        if isinstance(default_value, bool):
            saved_bool = _coerce_bool(saved.get(key))
            if has_saved_settings and saved_bool is not None:
                resolved[key] = saved_bool
                continue
            env_bool = _coerce_bool(_resolve_env_value(env_map.get(key)))
            resolved[key] = default_value if env_bool is None else env_bool
            continue
        saved_value = saved.get(key, "")
        if str(saved_value).strip():
            resolved[key] = str(saved_value).strip()
            continue
        env_value = _resolve_env_value(env_map.get(key))
        resolved[key] = env_value or default_value

    return resolved


def _normalize_settings(payload: dict[str, Any] | None) -> dict[str, Any]:
    raw = {} if payload is None else dict(payload)
    normalized = dict(DEFAULT_LLM_SETTINGS)
    normalized.update(
        {
            "provider": str(raw.get("provider", "")).strip(),
            "model": str(raw.get("model", "")).strip(),
            "loop_llm_selection": _coerce_bool(raw.get("loop_llm_selection")) or False,
            "ollama_base_url": str(raw.get("ollama_base_url", "")).strip(),
            "ollama_api_key": str(raw.get("ollama_api_key", "")).strip(),
            "agent_cli": str(raw.get("agent_cli", "")).strip(),
            "openai_compatible_base_url": str(raw.get("openai_compatible_base_url", "")).strip(),
            "openai_compatible_api_key": str(raw.get("openai_compatible_api_key", "")).strip(),
            "openai_api_key": str(raw.get("openai_api_key", "")).strip(),
        }
    )
    return normalized


def _resolve_first_env(env_names: tuple[str, ...]) -> str:
    for env_name in env_names:
        env_value = os.getenv(env_name, "").strip()
        if env_value:
            return env_value
    return ""


def _resolve_env_value(env_name: str | tuple[str, ...] | None) -> str:
    if isinstance(env_name, tuple):
        return _resolve_first_env(env_name)
    if isinstance(env_name, str):
        return os.getenv(env_name, "").strip()
    return ""


def _coerce_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return None
