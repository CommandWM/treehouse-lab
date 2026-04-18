from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import yaml

from treehouse_lab.config import load_experiment_config
from treehouse_lab.datasets import inspect_binary_target
from treehouse_lab.journal import load_incumbent, load_journal_entries, load_run_entry
from treehouse_lab.loop import AutonomousLoopController
from treehouse_lab.runner import DEFAULT_MODEL_PARAMS, TreehouseLabRunner

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_CONFIG_DIR = PROJECT_ROOT / "configs" / "datasets"
GLOSSARY_PATH = PROJECT_ROOT / "docs" / "glossary.md"

app = FastAPI(title="Treehouse Lab API", version="0.9.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class CandidateRequest(BaseModel):
    mutation_name: str = Field(min_length=1)
    hypothesis: str | None = None
    overrides: dict[str, Any] = Field(default_factory=dict)


class LoopRequest(BaseModel):
    steps: int = Field(default=1, ge=1, le=10)


class DatasetInspectRequest(BaseModel):
    path: str = Field(min_length=1)
    target_column: str | None = None


class DatasetCreateRequest(BaseModel):
    path: str = Field(min_length=1)
    target_column: str = Field(min_length=1)
    name: str = Field(min_length=1)
    description: str = ""
    config_key: str | None = None
    primary_metric: str = Field(default="roc_auc", min_length=1)
    objective: str = ""
    promote_if_delta_at_least: float = Field(default=0.003, gt=0)
    max_runtime_minutes: int = Field(default=10, ge=1, le=240)
    seed: int = Field(default=42)
    validation_size: float = Field(default=0.2, gt=0, lt=1)
    test_size: float = Field(default=0.2, gt=0, lt=1)
    stratify: bool = True


def _config_path_from_key(config_key: str) -> Path:
    config_path = (DATASET_CONFIG_DIR / f"{config_key}.yaml").resolve()
    if not config_path.exists():
        raise HTTPException(status_code=404, detail=f"Unknown config key: {config_key}")
    return config_path


def _load_glossary_sections() -> list[dict[str, str]]:
    sections: list[dict[str, str]] = []
    heading: str | None = None
    body: list[str] = []

    for raw_line in GLOSSARY_PATH.read_text(encoding="utf-8").splitlines():
        line = raw_line.rstrip()
        if line.startswith("## "):
            if heading is not None:
                sections.append({"term": heading, "definition": "\n".join(part for part in body if part.strip()).strip()})
            heading = line.removeprefix("## ").strip()
            body = []
            continue
        if heading is not None and not line.startswith("# "):
            body.append(line)

    if heading is not None:
        sections.append({"term": heading, "definition": "\n".join(part for part in body if part.strip()).strip()})
    return sections


def _load_run_artifact_text(artifact_dir: Path, filename: str) -> str | None:
    path = artifact_dir / filename
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8")


def _load_run_artifact_json(artifact_dir: Path, filename: str) -> dict[str, Any] | list[Any] | None:
    path = artifact_dir / filename
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_dataset_path(raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    resolved = path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()
    if not resolved.exists():
        raise HTTPException(status_code=404, detail=f"Dataset not found: {resolved}")
    if resolved.suffix.lower() != ".csv":
        raise HTTPException(status_code=400, detail="Dataset intake currently supports CSV files only.")
    return resolved


def _read_csv_frame(raw_path: str) -> tuple[Path, pd.DataFrame]:
    dataset_path = _resolve_dataset_path(raw_path)
    try:
        frame = pd.read_csv(dataset_path)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to read CSV dataset: {exc}") from exc
    if frame.empty:
        raise HTTPException(status_code=400, detail="Dataset is empty.")
    return dataset_path, frame


def _serialize_preview_rows(frame: pd.DataFrame, limit: int = 5) -> list[dict[str, Any]]:
    preview = frame.head(limit)
    return json.loads(preview.to_json(orient="records"))


def _inspect_frame(dataset_path: Path, frame: pd.DataFrame, target_column: str | None = None) -> dict[str, Any]:
    payload = {
        "path": str(dataset_path),
        "row_count": int(len(frame)),
        "column_count": int(len(frame.columns)),
        "columns": [
            {
                "name": column,
                "dtype": str(frame[column].dtype),
                "missing_count": int(frame[column].isna().sum()),
                "missing_rate": float(frame[column].isna().mean()),
                "unique_count": int(frame[column].nunique(dropna=True)),
            }
            for column in frame.columns
        ],
        "preview_rows": _serialize_preview_rows(frame),
    }
    if target_column is None:
        return payload

    if target_column not in frame.columns:
        raise HTTPException(status_code=400, detail=f"Unknown target column: {target_column}")

    try:
        payload["target"] = inspect_binary_target(frame[target_column], target_column)
    except ValueError as exc:
        payload["target"] = {
            "column": target_column,
            "binary_supported": False,
            "error": str(exc),
            "distinct_labels": [str(value) for value in pd.unique(frame[target_column].dropna())[:10]],
        }
    else:
        payload["target"]["binary_supported"] = True
        payload["feature_count"] = int(len(frame.columns) - 1)
    return payload


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")
    if not slug:
        raise HTTPException(status_code=400, detail="Unable to derive a valid config key.")
    return slug


def _config_storage_path(dataset_path: Path) -> str:
    try:
        return dataset_path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(dataset_path)


def _build_dataset_config_payload(payload: DatasetCreateRequest, dataset_path: Path) -> dict[str, Any]:
    description = payload.description.strip() or f"User-provided CSV dataset from {dataset_path.name}."
    objective = payload.objective.strip() or (
        "Establish a clean incumbent for a user-provided dataset before bounded autoresearch begins."
    )
    return {
        "dataset": {
            "source": {
                "kind": "csv",
                "name": payload.name.strip(),
                "target_column": payload.target_column,
                "path": _config_storage_path(dataset_path),
            },
            "split": {
                "validation_size": payload.validation_size,
                "test_size": payload.test_size,
                "stratify": payload.stratify,
            },
        },
        "benchmark": {
            "pack": "user",
            "profile": "dataset_intake",
            "objective": objective,
        },
        "evaluation_policy": {
            "require_promotion_for_readiness": True,
        },
        "experiment": {
            "name": f"{_slugify(payload.name)}-baseline",
            "description": description,
            "primary_metric": payload.primary_metric,
            "promote_if_delta_at_least": payload.promote_if_delta_at_least,
            "max_runtime_minutes": payload.max_runtime_minutes,
            "seed": payload.seed,
            "baseline_hypothesis": "A disciplined XGBoost baseline should establish a credible incumbent before bounded mutations begin.",
        },
        "model": {
            "params": dict(DEFAULT_MODEL_PARAMS),
        },
    }


def _serialize_config(config_key: str) -> dict[str, Any]:
    config_path = _config_path_from_key(config_key)
    config = load_experiment_config(config_path)
    return {
        "key": config_key,
        "path": str(config_path),
        "name": config.name,
        "description": config.description,
        "primary_metric": config.primary_metric,
        "source": {
            "kind": config.source.kind,
            "name": config.source.name,
            "target_column": config.source.target_column,
            "path": config.source.path,
        },
        "benchmark": {
            "pack": config.benchmark.pack,
            "profile": config.benchmark.profile,
            "objective": config.benchmark.objective,
        },
        "evaluation_policy": {
            "minimum_primary_metric": config.evaluation_policy.minimum_primary_metric,
            "max_train_validation_gap": config.evaluation_policy.max_train_validation_gap,
            "max_validation_test_gap": config.evaluation_policy.max_validation_test_gap,
            "max_runtime_seconds": config.evaluation_policy.max_runtime_seconds,
            "max_feature_count": config.evaluation_policy.max_feature_count,
            "require_promotion_for_readiness": config.evaluation_policy.require_promotion_for_readiness,
        },
        "raw": config.raw,
    }


def _serialize_state(config_key: str) -> dict[str, Any]:
    config_path = _config_path_from_key(config_key)
    controller = AutonomousLoopController(config_path)
    incumbent = load_incumbent(PROJECT_ROOT, config_key)
    diagnosis_preview = controller.diagnose().to_dict()
    journal_entries = load_journal_entries(PROJECT_ROOT, config_key)
    latest_run = None if not journal_entries else journal_entries[-1]
    return {
        "config": _serialize_config(config_key),
        "incumbent": incumbent,
        "diagnosis_preview": diagnosis_preview,
        "journal_count": len(journal_entries),
        "latest_run_id": None if latest_run is None else latest_run.get("run_id"),
    }


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/configs")
def list_configs() -> list[dict[str, Any]]:
    configs: list[dict[str, Any]] = []
    for path in sorted(DATASET_CONFIG_DIR.glob("*.yaml")):
        key = path.stem
        config_payload = _serialize_config(key)
        incumbent = load_incumbent(PROJECT_ROOT, key)
        configs.append(
            {
                **config_payload,
                "incumbent": incumbent,
            }
        )
    return configs


@app.get("/api/configs/{config_key}")
def get_config(config_key: str) -> dict[str, Any]:
    return _serialize_config(config_key)


@app.get("/api/configs/{config_key}/state")
def get_state(config_key: str) -> dict[str, Any]:
    return _serialize_state(config_key)


@app.get("/api/configs/{config_key}/journal")
def get_journal(config_key: str) -> list[dict[str, Any]]:
    return list(reversed(load_journal_entries(PROJECT_ROOT, config_key)))


@app.get("/api/configs/{config_key}/diagnose")
def get_diagnosis(config_key: str) -> dict[str, Any]:
    config_path = _config_path_from_key(config_key)
    controller = AutonomousLoopController(config_path)
    return controller.diagnose().to_dict()


@app.get("/api/runs/{run_id}")
def get_run(run_id: str) -> dict[str, Any]:
    entry = load_run_entry(PROJECT_ROOT, run_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"Unknown run id: {run_id}")
    artifact_dir = Path(str(entry["artifact_dir"]))
    return {
        "entry": entry,
        "artifacts": {
            "summary_markdown": _load_run_artifact_text(artifact_dir, "summary.md"),
            "narrative_markdown": _load_run_artifact_text(artifact_dir, "narrative.md"),
            "proposal": _load_run_artifact_json(artifact_dir, "proposal.json"),
            "assessment": _load_run_artifact_json(artifact_dir, "assessment.json"),
            "diagnosis": _load_run_artifact_json(artifact_dir, "diagnosis.json"),
            "run_context": _load_run_artifact_json(artifact_dir, "run_context.json"),
        },
    }


@app.get("/api/glossary")
def get_glossary() -> list[dict[str, str]]:
    return _load_glossary_sections()


@app.post("/api/intake/inspect")
def inspect_dataset(payload: DatasetInspectRequest) -> dict[str, Any]:
    dataset_path, frame = _read_csv_frame(payload.path)
    return _inspect_frame(dataset_path, frame, target_column=payload.target_column)


@app.post("/api/intake/create")
def create_dataset_config(payload: DatasetCreateRequest) -> dict[str, Any]:
    if payload.validation_size + payload.test_size >= 1:
        raise HTTPException(status_code=400, detail="validation_size and test_size must sum to less than 1.")

    dataset_path, frame = _read_csv_frame(payload.path)
    inspection = _inspect_frame(dataset_path, frame, target_column=payload.target_column)
    if not inspection.get("target", {}).get("binary_supported", False):
        detail = inspection["target"].get("error", "Target column is not valid for binary classification.")
        raise HTTPException(status_code=400, detail=detail)

    config_key = _slugify(payload.config_key or payload.name)
    config_path = DATASET_CONFIG_DIR / f"{config_key}.yaml"
    if config_path.exists():
        raise HTTPException(status_code=409, detail=f"Config already exists: {config_key}.yaml")

    config_blob = _build_dataset_config_payload(payload, dataset_path)
    config_path.write_text(
        yaml.safe_dump(config_blob, sort_keys=False, allow_unicode=False),
        encoding="utf-8",
    )

    return {
        "key": config_key,
        "path": str(config_path),
        "config": _serialize_config(config_key),
        "inspection": inspection,
    }


@app.post("/api/configs/{config_key}/baseline")
def run_baseline(config_key: str) -> dict[str, Any]:
    config_path = _config_path_from_key(config_key)
    runner = TreehouseLabRunner(config_path)
    return runner.run_baseline().to_dict()


@app.post("/api/configs/{config_key}/candidate")
def run_candidate(config_key: str, payload: CandidateRequest) -> dict[str, Any]:
    config_path = _config_path_from_key(config_key)
    runner = TreehouseLabRunner(config_path)
    return runner.run_candidate(
        mutation_name=payload.mutation_name,
        overrides=payload.overrides,
        hypothesis=payload.hypothesis,
    ).to_dict()


@app.post("/api/configs/{config_key}/loop")
def run_loop(config_key: str, payload: LoopRequest) -> dict[str, Any]:
    config_path = _config_path_from_key(config_key)
    controller = AutonomousLoopController(config_path)
    return controller.run_loop(max_steps=payload.steps).to_dict()


def main() -> None:
    import uvicorn

    uvicorn.run("treehouse_lab.api:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    main()
