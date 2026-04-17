from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def ensure_run_directories(project_root: Path) -> Path:
    runs_dir = project_root / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    return runs_dir


def append_journal_entry(project_root: Path, entry: dict[str, Any]) -> Path:
    runs_dir = ensure_run_directories(project_root)
    journal_path = runs_dir / "journal.jsonl"
    with journal_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, sort_keys=True) + "\n")
    return journal_path


def load_journal_entries(project_root: Path) -> list[dict[str, Any]]:
    journal_path = ensure_run_directories(project_root) / "journal.jsonl"
    if not journal_path.exists():
        return []
    with journal_path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_incumbent(project_root: Path, registry_key: str) -> dict[str, Any] | None:
    incumbent_path = _incumbent_path(project_root, registry_key)
    if not incumbent_path.exists():
        return None
    with incumbent_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_incumbent(project_root: Path, registry_key: str, entry: dict[str, Any]) -> Path:
    incumbent_path = _incumbent_path(project_root, registry_key)
    with incumbent_path.open("w", encoding="utf-8") as handle:
        json.dump(entry, handle, indent=2, sort_keys=True)
    return incumbent_path


def _incumbent_path(project_root: Path, registry_key: str) -> Path:
    runs_dir = ensure_run_directories(project_root)
    incumbent_dir = runs_dir / "incumbents"
    incumbent_dir.mkdir(parents=True, exist_ok=True)
    return incumbent_dir / f"{registry_key}.json"
