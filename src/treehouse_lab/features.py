from __future__ import annotations

from dataclasses import asdict, dataclass, field
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_selection import f_classif


DEFAULT_FEATURE_GENERATION_STRATEGY = "train_only_supervised_numeric_interactions"
DEFAULT_FEATURE_GENERATION_OPERATIONS = ("square", "product")


@dataclass(slots=True)
class FeatureGenerationPlan:
    enabled: bool
    reason: str
    max_new_features: int
    tool: str | None = None
    strategy: str = DEFAULT_FEATURE_GENERATION_STRATEGY
    top_k_numeric: int = 4
    operations: list[str] = field(default_factory=lambda: list(DEFAULT_FEATURE_GENERATION_OPERATIONS))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class FeatureGenerationResult:
    applied: bool
    reason: str
    new_feature_count: int = 0
    generated_feature_specs: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def should_enable_feature_generation(history: list[dict[str, Any]], incumbent: dict[str, Any] | None) -> bool:
    if incumbent is None or len(history) < 2:
        return False
    recent_deltas = [step["result"]["comparison_to_incumbent"].get("delta") for step in history[-2:]]
    bounded_deltas = [abs(float(delta)) for delta in recent_deltas if delta is not None]
    if len(bounded_deltas) < 2:
        return False
    return max(bounded_deltas) < 0.001


def build_feature_plan(config: dict[str, Any], enabled: bool) -> FeatureGenerationPlan:
    feature_config = config.get("feature_generation", {})
    max_new_features = int(feature_config.get("max_new_features", 0 if not enabled else 8))
    operations = [str(item) for item in feature_config.get("operations", DEFAULT_FEATURE_GENERATION_OPERATIONS)]
    if not enabled:
        return FeatureGenerationPlan(
            enabled=False,
            reason="Parameter-only templates have not plateaued yet, so feature generation remains off.",
            max_new_features=max_new_features,
            top_k_numeric=int(feature_config.get("top_k_numeric", 4)),
            operations=operations,
        )
    tools = feature_config.get("tools", [])
    return FeatureGenerationPlan(
        enabled=True,
        reason="Parameter-only templates plateaued, so a bounded feature-generation branch is now eligible.",
        max_new_features=max_new_features,
        tool=str(tools[0]) if tools else None,
        top_k_numeric=max(2, int(feature_config.get("top_k_numeric", 4))),
        operations=operations,
    )


def feature_plan_from_payload(payload: dict[str, Any] | None) -> FeatureGenerationPlan | None:
    if not payload or not bool(payload.get("enabled")):
        return None
    operations = [str(item) for item in payload.get("operations", DEFAULT_FEATURE_GENERATION_OPERATIONS)]
    return FeatureGenerationPlan(
        enabled=True,
        reason=str(payload.get("reason", "A bounded feature-generation branch was selected for execution.")),
        max_new_features=max(0, int(payload.get("max_new_features", 0))),
        tool=None if payload.get("tool") in (None, "") else str(payload.get("tool")),
        strategy=str(payload.get("strategy", DEFAULT_FEATURE_GENERATION_STRATEGY)),
        top_k_numeric=max(2, int(payload.get("top_k_numeric", 4))),
        operations=operations or list(DEFAULT_FEATURE_GENERATION_OPERATIONS),
    )


def run_feature_plan(
    plan: FeatureGenerationPlan,
    train_frame: pd.DataFrame | None = None,
    target: pd.Series | None = None,
) -> FeatureGenerationResult:
    if not plan.enabled:
        return FeatureGenerationResult(applied=False, reason=plan.reason)
    if train_frame is None or target is None:
        return FeatureGenerationResult(
            applied=False,
            reason="Feature-generation execution now exists, but this artifact only records plan readiness outside a bounded run.",
        )
    generated_feature_specs = fit_generated_feature_specs(train_frame, target, plan)
    if not generated_feature_specs:
        return FeatureGenerationResult(
            applied=False,
            reason="Feature generation was enabled, but the training split did not contain enough usable numeric signal to add bounded features.",
            generated_feature_specs=[],
        )
    return FeatureGenerationResult(
        applied=True,
        reason="Generated bounded train-only numeric interaction features from the highest-signal numeric columns.",
        new_feature_count=len(generated_feature_specs),
        generated_feature_specs=generated_feature_specs,
    )


def fit_generated_feature_specs(
    train_frame: pd.DataFrame,
    target: pd.Series,
    plan: FeatureGenerationPlan,
) -> list[dict[str, Any]]:
    if not plan.enabled or plan.max_new_features <= 0:
        return []

    numeric_frame = train_frame.reset_index(drop=True).copy()
    numeric_frame = numeric_frame.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan)
    if numeric_frame.empty:
        return []

    numeric_frame = numeric_frame.loc[:, numeric_frame.nunique(dropna=False) > 1]
    if numeric_frame.empty:
        return []

    fill_values = numeric_frame.median(numeric_only=True).fillna(0.0)
    numeric_frame = numeric_frame.fillna(fill_values)
    ranked_columns = _rank_numeric_columns(numeric_frame, target, top_k=min(plan.top_k_numeric, len(numeric_frame.columns)))
    if len(ranked_columns) < 2:
        return []

    specs: list[dict[str, Any]] = []
    operations = set(plan.operations or DEFAULT_FEATURE_GENERATION_OPERATIONS)

    if "square" in operations:
        for column in ranked_columns:
            specs.append(
                {
                    "name": f"fg__square__{column}",
                    "operation": "square",
                    "columns": [column],
                }
            )
            if len(specs) >= plan.max_new_features:
                return specs

    if "product" in operations:
        for left, right in combinations(ranked_columns, 2):
            specs.append(
                {
                    "name": f"fg__product__{left}__{right}",
                    "operation": "product",
                    "columns": [left, right],
                }
            )
            if len(specs) >= plan.max_new_features:
                return specs

    return specs


def apply_generated_features(numeric_frame: pd.DataFrame, specs: list[dict[str, Any]]) -> pd.DataFrame:
    if not specs:
        return numeric_frame

    base_frame = numeric_frame.reset_index(drop=True).copy()
    generated_columns: dict[str, pd.Series] = {}
    for spec in specs:
        name = str(spec["name"])
        columns = [str(column) for column in spec.get("columns", [])]
        missing_columns = [column for column in columns if column not in base_frame.columns]
        if missing_columns:
            msg = f"Generated feature '{name}' requires missing columns: {', '.join(missing_columns)}"
            raise ValueError(msg)

        operation = str(spec.get("operation", ""))
        if operation == "square":
            generated_columns[name] = base_frame[columns[0]].astype(float).pow(2)
            continue
        if operation == "product":
            generated_columns[name] = base_frame[columns[0]].astype(float) * base_frame[columns[1]].astype(float)
            continue

        msg = f"Unsupported generated feature operation: {operation}"
        raise ValueError(msg)

    if not generated_columns:
        return base_frame
    return pd.concat([base_frame, pd.DataFrame(generated_columns)], axis=1)


def _rank_numeric_columns(numeric_frame: pd.DataFrame, target: pd.Series, top_k: int) -> list[str]:
    frame = numeric_frame.reset_index(drop=True)
    target_series = target.reset_index(drop=True)

    try:
        scores, _ = f_classif(frame, target_series)
        ranked = [
            column
            for column, _score in sorted(
                zip(frame.columns.tolist(), np.nan_to_num(scores, nan=-1.0, posinf=-1.0, neginf=-1.0), strict=False),
                key=lambda item: (float(item[1]), float(frame[item[0]].var())),
                reverse=True,
            )
        ]
    except Exception:
        ranked = sorted(frame.columns.tolist(), key=lambda column: float(frame[column].var()), reverse=True)

    return ranked[:top_k]
