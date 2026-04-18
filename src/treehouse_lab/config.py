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
    variant: str | None = None
    rows: int = 1000
    random_state: int = 42


@dataclass(slots=True)
class SplitConfig:
    validation_size: float = 0.2
    test_size: float = 0.2
    stratify: bool = True


@dataclass(slots=True)
class TaskConfig:
    kind: str = "binary_classification"


@dataclass(slots=True)
class ModelConfig:
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class BenchmarkConfig:
    pack: str = "default"
    profile: str = "custom"
    objective: str = ""


@dataclass(slots=True)
class EvaluationPolicyConfig:
    minimum_primary_metric: float | None = None
    max_train_validation_gap: float = 0.05
    max_validation_test_gap: float = 0.04
    max_runtime_seconds: float | None = None
    max_feature_count: int | None = None
    require_promotion_for_readiness: bool = True


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
    task: TaskConfig
    model: ModelConfig
    benchmark: BenchmarkConfig
    evaluation_policy: EvaluationPolicyConfig
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
    task_raw = raw.get("task", {})
    experiment_raw = raw.get("experiment", {})
    model_raw = raw.get("model", {})
    benchmark_raw = raw.get("benchmark", {})
    evaluation_raw = raw.get("evaluation_policy", {})
    max_runtime_minutes = int(experiment_raw.get("max_runtime_minutes", 20))

    return ExperimentConfig(
        name=experiment_raw.get("name", config_path.stem),
        description=experiment_raw.get("description", ""),
        primary_metric=experiment_raw.get("primary_metric", "roc_auc"),
        promote_if_delta_at_least=float(experiment_raw.get("promote_if_delta_at_least", 0.002)),
        max_runtime_minutes=max_runtime_minutes,
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
            variant=source_raw.get("variant"),
            rows=int(source_raw.get("rows", 1000)),
            random_state=int(source_raw.get("random_state", experiment_raw.get("seed", 42))),
        ),
        split=SplitConfig(
            validation_size=float(split_raw.get("validation_size", 0.2)),
            test_size=float(split_raw.get("test_size", 0.2)),
            stratify=bool(split_raw.get("stratify", True)),
        ),
        task=TaskConfig(
            kind=str(task_raw.get("kind", "binary_classification")),
        ),
        model=ModelConfig(params=dict(model_raw.get("params", {}))),
        benchmark=BenchmarkConfig(
            pack=str(benchmark_raw.get("pack", "default")),
            profile=str(benchmark_raw.get("profile", "custom")),
            objective=str(benchmark_raw.get("objective", "")),
        ),
        evaluation_policy=EvaluationPolicyConfig(
            minimum_primary_metric=None
            if evaluation_raw.get("minimum_primary_metric") is None
            else float(evaluation_raw["minimum_primary_metric"]),
            max_train_validation_gap=float(evaluation_raw.get("max_train_validation_gap", 0.05)),
            max_validation_test_gap=float(evaluation_raw.get("max_validation_test_gap", 0.04)),
            max_runtime_seconds=None
            if evaluation_raw.get("max_runtime_seconds") is None
            else float(evaluation_raw["max_runtime_seconds"]),
            max_feature_count=None
            if evaluation_raw.get("max_feature_count") is None
            else int(evaluation_raw["max_feature_count"]),
            require_promotion_for_readiness=bool(evaluation_raw.get("require_promotion_for_readiness", True)),
        ),
        raw=raw,
    )
