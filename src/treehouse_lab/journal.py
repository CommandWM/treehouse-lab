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


def load_journal_entries(project_root: Path, registry_key: str | None = None) -> list[dict[str, Any]]:
    journal_path = ensure_run_directories(project_root) / "journal.jsonl"
    if not journal_path.exists():
        return []
    with journal_path.open("r", encoding="utf-8") as handle:
        entries = [json.loads(line) for line in handle if line.strip()]
    if registry_key is None:
        return entries
    return [entry for entry in entries if infer_registry_key(entry) == registry_key]


def load_run_entry(project_root: Path, run_id: str) -> dict[str, Any] | None:
    for entry in reversed(load_journal_entries(project_root)):
        if entry.get("run_id") == run_id:
            return entry
    return None


def update_journal_entry(project_root: Path, run_id: str, updates: dict[str, Any]) -> Path:
    journal_path = ensure_run_directories(project_root) / "journal.jsonl"
    entries = load_journal_entries(project_root)
    updated_entries: list[dict[str, Any]] = []
    found = False
    for entry in entries:
        if entry.get("run_id") == run_id:
            merged = dict(entry)
            merged.update(updates)
            updated_entries.append(merged)
            found = True
        else:
            updated_entries.append(entry)

    if not found:
        msg = f"Run id '{run_id}' was not found in the journal."
        raise ValueError(msg)

    with journal_path.open("w", encoding="utf-8") as handle:
        for entry in updated_entries:
            handle.write(json.dumps(entry, sort_keys=True) + "\n")
    return journal_path


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


def infer_registry_key(entry: dict[str, Any]) -> str | None:
    if "registry_key" in entry:
        return str(entry["registry_key"])
    metadata = entry.get("metadata", {})
    if isinstance(metadata, dict) and "dataset_key" in metadata:
        return str(metadata["dataset_key"])
    config_path = entry.get("config_path")
    if config_path:
        return Path(str(config_path)).stem
    return None
