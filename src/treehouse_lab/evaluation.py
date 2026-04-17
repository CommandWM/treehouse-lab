from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from treehouse_lab.config import ExperimentConfig


@dataclass(slots=True)
class EvaluationCheck:
    name: str
    passed: bool
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class RunAssessment:
    benchmark_status: str
    benchmark_summary: str
    implementation_ready: bool
    implementation_readiness: str
    checks: list[EvaluationCheck]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["checks"] = [check.to_dict() for check in self.checks]
        return payload


def assess_run(
    config: ExperimentConfig,
    metrics: dict[str, float],
    split_summary: dict[str, float | int],
    runtime_seconds: float,
    comparison: dict[str, Any],
    promoted: bool,
) -> RunAssessment:
    benchmark_status, benchmark_summary = _benchmark_status(config.primary_metric, metrics, comparison, promoted)
    checks = _build_checks(config, metrics, split_summary, runtime_seconds, comparison, promoted)
    implementation_ready = all(check.passed for check in checks)
    readiness = "implementation_ready" if implementation_ready else "needs_more_work"
    return RunAssessment(
        benchmark_status=benchmark_status,
        benchmark_summary=benchmark_summary,
        implementation_ready=implementation_ready,
        implementation_readiness=readiness,
        checks=checks,
    )


def _benchmark_status(
    primary_metric: str,
    metrics: dict[str, float],
    comparison: dict[str, Any],
    promoted: bool,
) -> tuple[str, str]:
    incumbent_metric = comparison.get("incumbent_metric")
    if incumbent_metric is None:
        metric_value = metrics.get(primary_metric, metrics.get(f"validation_{primary_metric}", 0.0))
        return (
            "baseline_established",
            f"Baseline established at validation {primary_metric} {metric_value:.4f}.",
        )
    delta = float(comparison.get("delta", 0.0))
    if promoted:
        return (
            "better_than_incumbent",
            f"Candidate improved validation {primary_metric} by {delta:.4f} and cleared the promotion bar.",
        )
    return (
        "not_better_than_incumbent",
        f"Candidate changed validation {primary_metric} by {delta:.4f} and did not clear the promotion bar.",
    )


def _build_checks(
    config: ExperimentConfig,
    metrics: dict[str, float],
    split_summary: dict[str, float | int],
    runtime_seconds: float,
    comparison: dict[str, Any],
    promoted: bool,
) -> list[EvaluationCheck]:
    policy = config.evaluation_policy
    checks: list[EvaluationCheck] = []

    incumbent_metric = comparison.get("incumbent_metric")
    if policy.require_promotion_for_readiness:
        passed = promoted or incumbent_metric is None
        detail = "Run either established the baseline or beat the incumbent under the configured promotion policy."
        if incumbent_metric is not None and not promoted:
            detail = "Run did not beat the incumbent, so it is not implementation-ready yet."
        checks.append(EvaluationCheck(name="promotion_bar", passed=passed, detail=detail))

    metric_value = float(metrics.get(config.primary_metric, metrics.get(f"validation_{config.primary_metric}", 0.0)))
    if policy.minimum_primary_metric is not None:
        checks.append(
            EvaluationCheck(
                name="minimum_primary_metric",
                passed=metric_value >= policy.minimum_primary_metric,
                detail=(
                    f"Validation {config.primary_metric} is {metric_value:.4f}; "
                    f"target is at least {policy.minimum_primary_metric:.4f}."
                ),
            )
        )

    train_gap = float(metrics["train_roc_auc"]) - float(metrics["validation_roc_auc"])
    checks.append(
        EvaluationCheck(
            name="train_validation_gap",
            passed=train_gap <= policy.max_train_validation_gap,
            detail=(
                f"Train-validation ROC AUC gap is {train_gap:.4f}; "
                f"allowed maximum is {policy.max_train_validation_gap:.4f}."
            ),
        )
    )

    validation_test_gap = abs(float(metrics["validation_roc_auc"]) - float(metrics["test_roc_auc"]))
    checks.append(
        EvaluationCheck(
            name="validation_test_gap",
            passed=validation_test_gap <= policy.max_validation_test_gap,
            detail=(
                f"Validation-test ROC AUC gap is {validation_test_gap:.4f}; "
                f"allowed maximum is {policy.max_validation_test_gap:.4f}."
            ),
        )
    )

    runtime_limit = policy.max_runtime_seconds
    if runtime_limit is None:
        runtime_limit = float(config.max_runtime_minutes * 60)
    checks.append(
        EvaluationCheck(
            name="runtime_budget",
            passed=runtime_seconds <= runtime_limit,
            detail=f"Runtime is {runtime_seconds:.2f}s; allowed maximum is {runtime_limit:.2f}s.",
        )
    )

    if policy.max_feature_count is not None:
        feature_count = int(split_summary["feature_count"])
        checks.append(
            EvaluationCheck(
                name="feature_budget",
                passed=feature_count <= policy.max_feature_count,
                detail=f"Feature count is {feature_count}; allowed maximum is {policy.max_feature_count}.",
            )
        )

    return checks
