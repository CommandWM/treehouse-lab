from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from treehouse_lab.config import ExperimentConfig


@dataclass(slots=True)
class RunDiagnosis:
    primary_tag: str
    tags: list[str]
    summary: str
    recommended_direction: str
    preferred_mutations: list[str]
    avoided_mutations: list[str]
    reason_codes: list[str]
    evidence: dict[str, float | int | str | None]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def diagnose_run_state(
    config: ExperimentConfig,
    metrics: dict[str, float],
    split_summary: dict[str, Any],
    recent_entries: list[dict[str, Any]] | None = None,
) -> RunDiagnosis:
    recent_entries = recent_entries or []
    if not metrics:
        return RunDiagnosis(
            primary_tag="no_incumbent",
            tags=["no_incumbent"],
            summary="No incumbent metrics exist yet, so the loop should establish a baseline before diagnosis-aware mutation selection.",
            recommended_direction="Run the baseline first.",
            preferred_mutations=[],
            avoided_mutations=[],
            reason_codes=["diagnosis_no_incumbent"],
            evidence={},
        )

    metric_key = config.primary_metric
    metric_value = float(metrics.get(metric_key, metrics.get(f"validation_{metric_key}", 0.0)))
    train_roc_auc = float(metrics.get("train_roc_auc", metric_value))
    validation_roc_auc = float(metrics.get("validation_roc_auc", metric_value))
    test_roc_auc = float(metrics.get("test_roc_auc", metric_value))
    overfit_gap = train_roc_auc - validation_roc_auc
    validation_test_gap = abs(validation_roc_auc - test_roc_auc)
    positive_rate = float(split_summary.get("validation_positive_rate", split_summary.get("train_positive_rate", 0.5)))

    tags: list[str] = []
    reason_codes: list[str] = []

    if overfit_gap > config.evaluation_policy.max_train_validation_gap:
        tags.append("overfit")
        reason_codes.append("diagnosis_overfit")

    minimum_primary_metric = config.evaluation_policy.minimum_primary_metric
    if minimum_primary_metric is not None and metric_value < minimum_primary_metric:
        tags.append("quality_floor_miss")
        reason_codes.append("diagnosis_quality_floor_miss")

    if validation_test_gap > config.evaluation_policy.max_validation_test_gap:
        tags.append("generalization_risk")
        reason_codes.append("diagnosis_generalization_risk")

    if abs(0.5 - positive_rate) > 0.15:
        tags.append("class_imbalance")
        reason_codes.append("diagnosis_class_imbalance")

    if "overfit" not in tags and "quality_floor_miss" in tags and train_roc_auc < metric_value + 0.06:
        tags.append("underfit")
        reason_codes.append("diagnosis_underfit")

    if _is_plateauing(recent_entries, config.promote_if_delta_at_least):
        tags.append("plateau")
        reason_codes.append("diagnosis_plateau")

    if not tags:
        tags.append("healthy")
        reason_codes.append("diagnosis_healthy")

    primary_tag = _choose_primary_tag(tags)
    preferred_mutations, avoided_mutations = _mutation_preferences(tags)
    summary = _build_summary(tags, metric_key, metric_value, overfit_gap, validation_test_gap, positive_rate)
    recommended_direction = _recommended_direction(primary_tag, tags)

    return RunDiagnosis(
        primary_tag=primary_tag,
        tags=tags,
        summary=summary,
        recommended_direction=recommended_direction,
        preferred_mutations=preferred_mutations,
        avoided_mutations=avoided_mutations,
        reason_codes=reason_codes,
        evidence={
            "metric_value": round(metric_value, 4),
            "train_roc_auc": round(train_roc_auc, 4),
            "validation_roc_auc": round(validation_roc_auc, 4),
            "test_roc_auc": round(test_roc_auc, 4),
            "overfit_gap": round(overfit_gap, 4),
            "validation_test_gap": round(validation_test_gap, 4),
            "positive_rate": round(positive_rate, 4),
            "minimum_primary_metric": None if minimum_primary_metric is None else round(float(minimum_primary_metric), 4),
        },
    )


def build_reason_codes(
    promoted: bool,
    comparison: dict[str, Any],
    assessment: dict[str, Any],
    diagnosis: RunDiagnosis,
) -> list[str]:
    codes: list[str] = []
    if comparison.get("incumbent_metric") is None:
        codes.append("baseline_established")
    elif promoted:
        codes.append("promoted_metric_gain")
    else:
        codes.append("rejected_below_threshold")

    codes.extend(diagnosis.reason_codes)
    for check in assessment.get("checks", []):
        check_name = str(check["name"])
        prefix = "passed" if bool(check["passed"]) else "failed"
        codes.append(f"{prefix}_{check_name}")
    return sorted(dict.fromkeys(codes))


def _is_plateauing(recent_entries: list[dict[str, Any]], threshold: float) -> bool:
    deltas: list[float] = []
    for entry in recent_entries[-3:]:
        comparison = entry.get("comparison_to_incumbent", {})
        delta = comparison.get("delta")
        if delta is not None:
            deltas.append(abs(float(delta)))
    return len(deltas) >= 2 and all(delta < threshold for delta in deltas[-2:])


def _choose_primary_tag(tags: list[str]) -> str:
    priority = [
        "overfit",
        "underfit",
        "quality_floor_miss",
        "generalization_risk",
        "class_imbalance",
        "plateau",
        "healthy",
    ]
    for tag in priority:
        if tag in tags:
            return tag
    return tags[0]


def _mutation_preferences(tags: list[str]) -> tuple[list[str], list[str]]:
    preferred: list[str] = []
    avoided: list[str] = []

    if "overfit" in tags:
        preferred.extend(["regularization_tighten", "learning_rate_tradeoff"])
        avoided.append("capacity_increase")
    if "underfit" in tags:
        preferred.extend(["capacity_increase", "learning_rate_tradeoff"])
        avoided.append("regularization_tighten")
    if "class_imbalance" in tags:
        preferred.append("imbalance_adjustment")
    if "plateau" in tags:
        preferred.append("learning_rate_tradeoff")
    if tags == ["healthy"]:
        preferred.append("learning_rate_tradeoff")

    return sorted(dict.fromkeys(preferred)), sorted(dict.fromkeys(avoided))


def _build_summary(
    tags: list[str],
    metric_key: str,
    metric_value: float,
    overfit_gap: float,
    validation_test_gap: float,
    positive_rate: float,
) -> str:
    clauses = [f"Validation {metric_key} is {metric_value:.4f}."]
    if "overfit" in tags:
        clauses.append(f"Train-validation gap is {overfit_gap:.4f}, which points to overfitting.")
    if "underfit" in tags:
        clauses.append("The model looks capacity-limited rather than just noisy.")
    if "quality_floor_miss" in tags:
        clauses.append("The run is still below the configured quality floor.")
    if "generalization_risk" in tags:
        clauses.append(f"Validation-test gap is {validation_test_gap:.4f}, so holdout behavior looks unstable.")
    if "class_imbalance" in tags:
        clauses.append(f"Positive rate is {positive_rate:.4f}, so imbalance handling is worth considering.")
    if "plateau" in tags:
        clauses.append("Recent deltas are small, so the loop may be plateauing.")
    if tags == ["healthy"]:
        clauses.append("No major diagnosis flags are active.")
    return " ".join(clauses)


def _recommended_direction(primary_tag: str, tags: list[str]) -> str:
    if primary_tag == "overfit":
        return "Favor simpler ensembles and smoother boosting before adding capacity."
    if primary_tag == "underfit":
        return "Recover signal with a modest capacity increase or a slower boosting tradeoff."
    if "class_imbalance" in tags:
        return "Test bounded class weighting before broader changes."
    if primary_tag == "plateau":
        return "Prefer attributable parameter trades over repeating the same mutation family."
    if primary_tag == "generalization_risk":
        return "Prioritize stability and avoid mutations that widen validation-holdout drift."
    return "Keep the next mutation small, attributable, and inside the declared search space."
