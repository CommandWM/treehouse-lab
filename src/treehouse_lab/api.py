from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from treehouse_lab.config import load_experiment_config
from treehouse_lab.journal import load_incumbent, load_journal_entries, load_run_entry
from treehouse_lab.loop import AutonomousLoopController
from treehouse_lab.runner import TreehouseLabRunner

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_CONFIG_DIR = PROJECT_ROOT / "configs" / "datasets"
GLOSSARY_PATH = PROJECT_ROOT / "docs" / "glossary.md"

app = FastAPI(title="Treehouse Lab API", version="0.1.0")
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


def _serialize_config(config_key: str) -> dict[str, Any]:
    config_path = _config_path_from_key(config_key)
    config = load_experiment_config(config_path)
    return {
        "key": config_key,
        "path": str(config_path),
        "name": config.name,
        "description": config.description,
        "primary_metric": config.primary_metric,
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
