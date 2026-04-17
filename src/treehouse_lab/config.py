from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class DatasetSourceConfig:
    kind: str
    name: str | None = None
    target_column: str | None = None
    path: str | None = None
    rows: int = 1000
    random_state: int = 42


@dataclass(slots=True)
class SplitConfig:
    validation_size: float = 0.2
    test_size: float = 0.2
    stratify: bool = True


@dataclass(slots=True)
class ModelConfig:
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ExperimentConfig:
    name: str
    description: str
    primary_metric: str
    promote_if_delta_at_least: float
    max_runtime_minutes: int
    seed: int
    hypothesis: str
    source: DatasetSourceConfig
    split: SplitConfig
    model: ModelConfig
    raw: dict[str, Any]


def load_yaml_file(path: str | Path) -> dict[str, Any]:
    with Path(path).expanduser().open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        msg = f"Expected YAML object in {path}"
        raise ValueError(msg)
    return data


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path).expanduser().resolve()
    raw = load_yaml_file(config_path)

    source_raw = raw.get("dataset", {}).get("source", {})
    split_raw = raw.get("dataset", {}).get("split", {})
    experiment_raw = raw.get("experiment", {})
    model_raw = raw.get("model", {})

    return ExperimentConfig(
        name=experiment_raw.get("name", config_path.stem),
        description=experiment_raw.get("description", ""),
        primary_metric=experiment_raw.get("primary_metric", "roc_auc"),
        promote_if_delta_at_least=float(experiment_raw.get("promote_if_delta_at_least", 0.002)),
        max_runtime_minutes=int(experiment_raw.get("max_runtime_minutes", 20)),
        seed=int(experiment_raw.get("seed", 42)),
        hypothesis=experiment_raw.get(
            "baseline_hypothesis",
            "A strong default XGBoost baseline should be hard to beat and easy to audit.",
        ),
        source=DatasetSourceConfig(
            kind=source_raw["kind"],
            name=source_raw.get("name"),
            target_column=source_raw.get("target_column"),
            path=source_raw.get("path"),
            rows=int(source_raw.get("rows", 1000)),
            random_state=int(source_raw.get("random_state", experiment_raw.get("seed", 42))),
        ),
        split=SplitConfig(
            validation_size=float(split_raw.get("validation_size", 0.2)),
            test_size=float(split_raw.get("test_size", 0.2)),
            stratify=bool(split_raw.get("stratify", True)),
        ),
        model=ModelConfig(params=dict(model_raw.get("params", {}))),
        raw=raw,
    )
