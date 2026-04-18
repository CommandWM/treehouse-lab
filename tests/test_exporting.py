from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

from treehouse_lab.datasets import fit_feature_preprocessor, transform_feature_frame
from treehouse_lab.exporting import (
    ExportedModelBundle,
    export_model_artifact,
    load_exported_model_bundle,
    save_exported_model_bundle,
)
from treehouse_lab.journal import append_journal_entry, save_incumbent


def make_training_frame() -> tuple[pd.DataFrame, pd.Series]:
    frame = pd.DataFrame(
        {
            "age": [28, 34, 41, 52, 25, 47],
            "segment": ["basic", "pro", "pro", "enterprise", "basic", "enterprise"],
            "visits": [2, 8, 6, 9, 1, 7],
        }
    )
    target = pd.Series([0, 1, 1, 1, 0, 1], name="converted")
    return frame, target


def test_exported_model_bundle_scores_raw_records(tmp_path: Path) -> None:
    frame, target = make_training_frame()
    preprocessor = fit_feature_preprocessor(frame)
    prepared = transform_feature_frame(frame, preprocessor)
    model = GradientBoostingClassifier(random_state=42)
    model.fit(prepared, target)

    bundle = ExportedModelBundle(
        run_id="run-123",
        registry_key="marketing-leads",
        config_path=str(tmp_path / "configs" / "datasets" / "marketing-leads.yaml"),
        target_name="converted",
        task_kind="binary_classification",
        class_labels=["no", "yes"],
        primary_metric="roc_auc",
        backend="sklearn_gradient_boosting",
        threshold=0.5,
        feature_preprocessor=preprocessor,
        model_params={"random_state": 42},
        metrics={"roc_auc": 0.91},
        model=model,
    )
    bundle_path = tmp_path / "model_bundle.pkl"
    save_exported_model_bundle(bundle, bundle_path)

    loaded = load_exported_model_bundle(bundle_path)
    predictions = loaded.predict_records(
        [
            {"age": 31, "segment": "pro", "visits": 7, "ignored_column": "ok"},
            {"age": 23, "segment": "basic", "visits": 1},
        ]
    )

    assert len(predictions) == 2
    assert set(predictions[0]) == {"prediction", "score"}
    assert all(0.0 <= row["score"] <= 1.0 for row in predictions)


def test_export_model_artifact_writes_reusable_package(tmp_path: Path) -> None:
    project_root = tmp_path
    config_dir = project_root / "configs" / "datasets"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "marketing-leads.yaml"
    config_path.write_text("dataset:\n  source:\n    kind: csv\n", encoding="utf-8")

    frame, target = make_training_frame()
    preprocessor = fit_feature_preprocessor(frame)
    prepared = transform_feature_frame(frame, preprocessor)
    model = GradientBoostingClassifier(random_state=42)
    model.fit(prepared, target)

    run_id = "20260418T170000000000Z-baseline"
    artifact_dir = project_root / "runs" / run_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    bundle = ExportedModelBundle(
        run_id=run_id,
        registry_key="marketing-leads",
        config_path=str(config_path),
        target_name="converted",
        task_kind="binary_classification",
        class_labels=["no", "yes"],
        primary_metric="roc_auc",
        backend="sklearn_gradient_boosting",
        threshold=0.5,
        feature_preprocessor=preprocessor,
        model_params={"random_state": 42},
        metrics={"roc_auc": 0.91},
        model=model,
    )
    save_exported_model_bundle(bundle, artifact_dir / "model_bundle.pkl")
    (artifact_dir / "metrics.json").write_text(json.dumps({"roc_auc": 0.91}), encoding="utf-8")
    (artifact_dir / "summary.md").write_text("# baseline\n", encoding="utf-8")
    (artifact_dir / "config_snapshot.json").write_text(json.dumps({"dataset": {"source": {"kind": "csv"}}}), encoding="utf-8")

    append_journal_entry(
        project_root,
        {
            "run_id": run_id,
            "registry_key": "marketing-leads",
            "artifact_dir": str(artifact_dir),
            "config_path": str(config_path),
            "promoted": True,
        },
    )
    save_incumbent(
        project_root,
        "marketing-leads",
        {
            "run_id": run_id,
            "registry_key": "marketing-leads",
            "artifact_dir": str(artifact_dir),
            "config_path": str(config_path),
            "metric": 0.91,
        },
    )

    manifest = export_model_artifact(project_root, "marketing-leads")
    export_dir = Path(manifest["export_dir"])

    assert export_dir.exists()
    assert (export_dir / "model_bundle.pkl").exists()
    assert (export_dir / "app.py").exists()
    assert (export_dir / "Dockerfile").exists()
    assert (export_dir / ".dockerignore").exists()
    assert (export_dir / "requirements.txt").exists()
    assert (export_dir / "manifest.json").exists()
    assert "uvicorn app:app" in manifest["serve_command"]
    assert "docker build" in manifest["docker_build_command"]
    assert manifest["artifact_usage"] == "python_bundle"
