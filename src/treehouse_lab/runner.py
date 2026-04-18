from __future__ import annotations

import json
import shutil
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

from treehouse_lab.config import load_experiment_config
from treehouse_lab.datasets import DatasetSplit, load_dataset, split_dataset
from treehouse_lab.diagnosis import build_reason_codes, diagnose_run_state
from treehouse_lab.evaluation import assess_run
from treehouse_lab.exporting import ExportedModelBundle, save_exported_model_bundle
from treehouse_lab.journal import (
    append_journal_entry,
    ensure_run_directories,
    load_incumbent,
    load_journal_entries,
    save_incumbent,
)

try:
    import mlflow
except ImportError:  # pragma: no cover - optional at runtime
    mlflow = None

try:
    from xgboost import XGBClassifier
except Exception as exc:  # pragma: no cover - platform-specific at runtime
    XGBClassifier = None
    XGBOOST_IMPORT_ERROR = exc
else:  # pragma: no cover - trivial assignment
    XGBOOST_IMPORT_ERROR = None


DEFAULT_MODEL_PARAMS: dict[str, Any] = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.05,
    "min_child_weight": 1,
    "subsample": 0.9,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
    "gamma": 0.0,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "random_state": 42,
    "tree_method": "hist",
    "n_jobs": 4,
}


@dataclass(slots=True)
class ExperimentResult:
    name: str
    backend: str
    metric: float
    promoted: bool
    notes: str
    run_id: str
    artifact_dir: str
    config_path: str
    hypothesis: str
    decision_reason: str
    runtime_seconds: float
    params: dict[str, Any]
    metrics: dict[str, float]
    split_summary: dict[str, float | int]
    comparison_to_incumbent: dict[str, Any]
    assessment: dict[str, Any]
    diagnosis: dict[str, Any]
    reason_codes: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TreehouseLabRunner:
    """Run disciplined baseline and bounded candidate experiments."""

    def __init__(self, config_path: str | Path):
        self.config_path = Path(config_path).expanduser().resolve()
        self.project_root = self.config_path.parents[2] if len(self.config_path.parents) >= 3 else Path(__file__).resolve().parents[2]
        self.config = load_experiment_config(self.config_path)
        self.registry_key = self.config_path.stem

    def run_baseline(self, metadata: dict[str, Any] | None = None) -> ExperimentResult:
        return self._run_experiment(
            mutation_name="baseline",
            overrides={},
            hypothesis=self.config.hypothesis,
            metadata=metadata,
        )

    def run_candidate(
        self,
        mutation_name: str,
        overrides: dict[str, Any],
        hypothesis: str | None = None,
        metadata: dict[str, Any] | None = None,
        base_params: dict[str, Any] | None = None,
    ) -> ExperimentResult:
        candidate_hypothesis = hypothesis or "A bounded parameter mutation can outperform the incumbent."
        return self._run_experiment(
            mutation_name=mutation_name,
            overrides=overrides,
            hypothesis=candidate_hypothesis,
            metadata=metadata,
            base_params=base_params,
        )

    def _run_experiment(
        self,
        mutation_name: str,
        overrides: dict[str, Any],
        hypothesis: str,
        metadata: dict[str, Any] | None,
        base_params: dict[str, Any] | None = None,
    ) -> ExperimentResult:
        run_started = time.perf_counter()
        dataset = load_dataset(self.config, self.project_root)
        split = split_dataset(dataset, self.config)
        params = self._resolve_model_params(overrides, base_params=base_params)
        model, backend = self._build_model(params)
        model.fit(split.X_train, split.y_train)

        metrics = self._compute_metrics(model, split)
        runtime_seconds = time.perf_counter() - run_started
        promoted, comparison, decision_reason = self._promotion_decision(metrics[self.config.primary_metric])
        split_summary = split.summary()
        assessment = assess_run(
            self.config,
            metrics=metrics,
            split_summary=split_summary,
            runtime_seconds=runtime_seconds,
            comparison=comparison,
            promoted=promoted,
        )
        recent_entries = load_journal_entries(self.project_root, self.registry_key)
        diagnosis = diagnose_run_state(self.config, metrics, split_summary, recent_entries=recent_entries)
        reason_codes = build_reason_codes(promoted, comparison, assessment.to_dict(), diagnosis)

        run_id = self._build_run_id(mutation_name)
        artifact_dir = ensure_run_directories(self.project_root) / run_id
        artifact_dir.mkdir(parents=True, exist_ok=False)
        self._write_artifacts(
            artifact_dir=artifact_dir,
            run_id=run_id,
            mutation_name=mutation_name,
            hypothesis=hypothesis,
            params=params,
            metrics=metrics,
            split_summary=split_summary,
            feature_names=list(split.X_train.columns),
            target_name=dataset.target_name,
            primary_metric=self.config.primary_metric,
            preprocessor=split.preprocessor,
            model=model,
            backend=backend,
            runtime_seconds=runtime_seconds,
            promoted=promoted,
            comparison=comparison,
            decision_reason=decision_reason,
            assessment=assessment.to_dict(),
            diagnosis=diagnosis.to_dict(),
            reason_codes=reason_codes,
            metadata=metadata or {},
        )
        self._log_mlflow_if_available(
            run_id,
            mutation_name,
            backend,
            params,
            metrics,
            promoted,
            artifact_dir,
            metadata or {},
        )

        result = ExperimentResult(
            name=mutation_name,
            backend=backend,
            metric=metrics[self.config.primary_metric],
            promoted=promoted,
            notes=decision_reason,
            run_id=run_id,
            artifact_dir=str(artifact_dir),
            config_path=str(self.config_path),
            hypothesis=hypothesis,
            decision_reason=decision_reason,
            runtime_seconds=runtime_seconds,
            params=params,
            metrics=metrics,
            split_summary=split_summary,
            comparison_to_incumbent=comparison,
            assessment=assessment.to_dict(),
            diagnosis=diagnosis.to_dict(),
            reason_codes=reason_codes,
            metadata=metadata or {},
        )
        self._record_journal(result)
        return result

    def _resolve_model_params(self, overrides: dict[str, Any], base_params: dict[str, Any] | None = None) -> dict[str, Any]:
        params = dict(DEFAULT_MODEL_PARAMS)
        if base_params:
            params.update(base_params)
        else:
            params.update(self.config.model.params)
        params["random_state"] = self.config.seed
        params.update(overrides)
        return params

    def _build_model(self, params: dict[str, Any]) -> tuple[Any, str]:
        if XGBClassifier is not None:
            return XGBClassifier(**params), "xgboost"

        fallback_params = {
            "n_estimators": int(params["n_estimators"]),
            "learning_rate": float(params["learning_rate"]),
            "max_depth": int(params["max_depth"]),
            "subsample": float(params["subsample"]),
            "min_samples_leaf": max(1, int(params.get("min_child_weight", 1))),
            "random_state": int(params["random_state"]),
        }
        return GradientBoostingClassifier(**fallback_params), "sklearn_gradient_boosting"

    def _compute_metrics(self, model: Any, split: DatasetSplit) -> dict[str, float]:
        train_pred = model.predict_proba(split.X_train)[:, 1]
        val_pred = model.predict_proba(split.X_val)[:, 1]
        test_pred = model.predict_proba(split.X_test)[:, 1]

        train_label_pred = (train_pred >= 0.5).astype(int)
        val_label_pred = (val_pred >= 0.5).astype(int)
        test_label_pred = (test_pred >= 0.5).astype(int)

        return {
            "roc_auc": float(roc_auc_score(split.y_val, val_pred)),
            "train_roc_auc": float(roc_auc_score(split.y_train, train_pred)),
            "validation_roc_auc": float(roc_auc_score(split.y_val, val_pred)),
            "test_roc_auc": float(roc_auc_score(split.y_test, test_pred)),
            "train_accuracy": float(accuracy_score(split.y_train, train_label_pred)),
            "validation_accuracy": float(accuracy_score(split.y_val, val_label_pred)),
            "test_accuracy": float(accuracy_score(split.y_test, test_label_pred)),
            "train_log_loss": float(log_loss(split.y_train, train_pred)),
            "validation_log_loss": float(log_loss(split.y_val, val_pred)),
            "test_log_loss": float(log_loss(split.y_test, test_pred)),
        }

    def _promotion_decision(self, metric_value: float) -> tuple[bool, dict[str, Any], str]:
        incumbent = load_incumbent(self.project_root, self.registry_key)
        if incumbent is None:
            comparison = {"incumbent_metric": None, "delta": None, "threshold": self.config.promote_if_delta_at_least}
            return True, comparison, "No incumbent exists yet, so the first successful run becomes the incumbent."

        delta = metric_value - float(incumbent["metric"])
        threshold = self.config.promote_if_delta_at_least
        comparison = {"incumbent_metric": float(incumbent["metric"]), "delta": float(delta), "threshold": threshold}
        if delta >= threshold:
            return True, comparison, f"Validation {self.config.primary_metric} improved by {delta:.4f}, clearing the promotion threshold."
        return False, comparison, f"Validation {self.config.primary_metric} changed by {delta:.4f}, which did not clear the promotion threshold."

    def _build_run_id(self, mutation_name: str) -> str:
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        slug = mutation_name.lower().replace(" ", "-").replace("_", "-")
        return f"{timestamp}-{slug}"

    def _write_artifacts(
        self,
        artifact_dir: Path,
        run_id: str,
        mutation_name: str,
        hypothesis: str,
        params: dict[str, Any],
        metrics: dict[str, float],
        split_summary: dict[str, float | int],
        feature_names: list[str],
        target_name: str,
        primary_metric: str,
        preprocessor: Any,
        model: Any,
        backend: str,
        runtime_seconds: float,
        promoted: bool,
        comparison: dict[str, Any],
        decision_reason: str,
        assessment: dict[str, Any],
        diagnosis: dict[str, Any],
        reason_codes: list[str],
        metadata: dict[str, Any],
    ) -> None:
        normalized_config_path = artifact_dir / "config_snapshot.json"
        metrics_path = artifact_dir / "metrics.json"
        split_path = artifact_dir / "split_summary.json"
        params_path = artifact_dir / "model_params.json"
        context_path = artifact_dir / "run_context.json"
        assessment_path = artifact_dir / "assessment.json"
        diagnosis_path = artifact_dir / "diagnosis.json"
        summary_path = artifact_dir / "summary.md"
        importances_path = artifact_dir / "feature_importances.csv"
        model_bundle_path = artifact_dir / "model_bundle.pkl"
        original_config_path = artifact_dir / self.config_path.name

        shutil.copy2(self.config_path, original_config_path)
        normalized_config_path.write_text(json.dumps(self.config.raw, indent=2, sort_keys=True), encoding="utf-8")
        metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
        split_path.write_text(json.dumps(split_summary, indent=2, sort_keys=True), encoding="utf-8")
        params_path.write_text(json.dumps(params, indent=2, sort_keys=True), encoding="utf-8")
        context_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
        assessment_path.write_text(json.dumps(assessment, indent=2, sort_keys=True), encoding="utf-8")
        diagnosis_path.write_text(json.dumps(diagnosis, indent=2, sort_keys=True), encoding="utf-8")

        importance_frame = pd.DataFrame(
            {
                "feature": feature_names,
                "importance": model.feature_importances_,
            }
        ).sort_values("importance", ascending=False)
        importance_frame.to_csv(importances_path, index=False)
        save_exported_model_bundle(
            ExportedModelBundle(
                run_id=run_id,
                registry_key=self.registry_key,
                config_path=str(self.config_path),
                target_name=target_name,
                primary_metric=primary_metric,
                backend=backend,
                threshold=0.5,
                feature_preprocessor=preprocessor,
                model_params=params,
                metrics=metrics,
                model=model,
            ),
            model_bundle_path,
        )

        summary_path.write_text(
            self._build_summary(
                run_id=run_id,
                mutation_name=mutation_name,
                backend=backend,
                hypothesis=hypothesis,
                metrics=metrics,
                split_summary=split_summary,
                runtime_seconds=runtime_seconds,
                promoted=promoted,
                comparison=comparison,
                decision_reason=decision_reason,
                assessment=assessment,
                diagnosis=diagnosis,
                reason_codes=reason_codes,
                metadata=metadata,
            ),
            encoding="utf-8",
        )

        if promoted:
            save_incumbent(
                self.project_root,
                self.registry_key,
                {
                    "run_id": run_id,
                    "name": mutation_name,
                    "metric": metrics[self.config.primary_metric],
                    "artifact_dir": str(artifact_dir),
                    "config_path": str(self.config_path),
                    "registry_key": self.registry_key,
                    "params": params,
                    "backend": backend,
                    "metrics": metrics,
                    "assessment": assessment,
                    "diagnosis": diagnosis,
                    "reason_codes": reason_codes,
                },
            )

    def _build_summary(
        self,
        run_id: str,
        mutation_name: str,
        backend: str,
        hypothesis: str,
        metrics: dict[str, float],
        split_summary: dict[str, float | int],
        runtime_seconds: float,
        promoted: bool,
        comparison: dict[str, Any],
        decision_reason: str,
        assessment: dict[str, Any],
        diagnosis: dict[str, Any],
        reason_codes: list[str],
        metadata: dict[str, Any],
    ) -> str:
        lines = [
            f"# {mutation_name}",
            "",
            f"- run_id: `{run_id}`",
            f"- benchmark_pack: `{self.config.benchmark.pack}`",
            f"- benchmark_profile: `{self.config.benchmark.profile}`",
            f"- hypothesis: {hypothesis}",
            f"- backend: `{backend}`",
            f"- primary_metric: `{self.config.primary_metric}`",
            f"- validation_{self.config.primary_metric}: `{metrics[self.config.primary_metric]:.4f}`",
            f"- runtime_seconds: `{runtime_seconds:.2f}`",
            f"- decision: `{'promote' if promoted else 'reject'}`",
            f"- explanation: {decision_reason}",
        ]
        if self.config.benchmark.objective:
            lines.append(f"- benchmark_objective: {self.config.benchmark.objective}")
        if metadata:
            lines.extend(
                [
                    "",
                    "## Run context",
                    "",
                    *(f"- {key}: `{value}`" for key, value in metadata.items() if not isinstance(value, (dict, list))),
                ]
            )
        lines.extend(
            [
                "",
                "## Assessment",
                "",
                f"- benchmark_status: `{assessment['benchmark_status']}`",
                f"- benchmark_summary: {assessment['benchmark_summary']}",
                f"- implementation_readiness: `{assessment['implementation_readiness']}`",
            ]
        )
        lines.extend(f"- {check['name']}: `{check['passed']}` ({check['detail']})" for check in assessment["checks"])
        lines.extend(
            [
                "",
                "## Diagnosis",
                "",
                f"- primary_tag: `{diagnosis['primary_tag']}`",
                f"- summary: {diagnosis['summary']}",
                f"- recommended_direction: {diagnosis['recommended_direction']}",
                f"- preferred_mutations: `{', '.join(diagnosis['preferred_mutations']) or 'none'}`",
                f"- avoided_mutations: `{', '.join(diagnosis['avoided_mutations']) or 'none'}`",
                "",
                "## Reason codes",
                "",
                *(f"- `{code}`" for code in reason_codes),
                "",
                "## Split summary",
                "",
                *(f"- {key}: `{value}`" for key, value in split_summary.items()),
                "",
                "## Comparison to incumbent",
                "",
                *(f"- {key}: `{value}`" for key, value in comparison.items()),
            ]
        )
        return "\n".join(lines)

    def _record_journal(self, result: ExperimentResult) -> None:
        append_journal_entry(self.project_root, result.to_dict())

    def _log_mlflow_if_available(
        self,
        run_id: str,
        mutation_name: str,
        backend: str,
        params: dict[str, Any],
        metrics: dict[str, float],
        promoted: bool,
        artifact_dir: Path,
        metadata: dict[str, Any],
    ) -> None:
        if mlflow is None:
            return
        mlflow.set_experiment(self.config.name)
        with mlflow.start_run(run_name=mutation_name):
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            mlflow.set_tag("treehouse_lab.run_id", run_id)
            mlflow.set_tag("treehouse_lab.promoted", str(promoted).lower())
            mlflow.set_tag("treehouse_lab.backend", backend)
            mlflow.set_tag("treehouse_lab.benchmark_profile", self.config.benchmark.profile)
            if "proposal_id" in metadata:
                mlflow.set_tag("treehouse_lab.proposal_id", str(metadata["proposal_id"]))
            if "mutation_type" in metadata:
                mlflow.set_tag("treehouse_lab.mutation_type", str(metadata["mutation_type"]))
            if XGBOOST_IMPORT_ERROR is not None:
                mlflow.set_tag("treehouse_lab.xgboost_fallback", type(XGBOOST_IMPORT_ERROR).__name__)
            mlflow.log_artifacts(str(artifact_dir))
