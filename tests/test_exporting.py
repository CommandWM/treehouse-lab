from __future__ import annotations

import importlib.util
import json
import sys
import uuid
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from sklearn.ensemble import GradientBoostingClassifier

from treehouse_lab.datasets import fit_feature_preprocessor, transform_feature_frame
from treehouse_lab.exporting import (
    BUNDLE_FILENAME,
    ExportedModelBundle,
    export_model_artifact,
    load_exported_model_bundle,
    save_exported_model_bundle,
)
from treehouse_lab.features import build_feature_plan
from treehouse_lab.journal import append_journal_entry, save_incumbent


REQUIRED_EXPORT_FILES = {
    BUNDLE_FILENAME,
    "app.py",
    "Dockerfile",
    ".dockerignore",
    "requirements.txt",
    "README.md",
    "manifest.json",
}

REQUIRED_MANIFEST_KEYS = {
    "config_key",
    "run_id",
    "source_artifact_dir",
    "export_dir",
    "bundle_path",
    "bundle_materialization",
    "artifact_usage",
    "containerization",
    "serve_command",
    "docker_build_command",
    "docker_run_command",
    "files",
}


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


def make_exportable_project(project_root: Path) -> tuple[str, Path]:
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
    save_exported_model_bundle(bundle, artifact_dir / BUNDLE_FILENAME)
    (artifact_dir / "metrics.json").write_text(json.dumps({"roc_auc": 0.91}), encoding="utf-8")
    (artifact_dir / "summary.md").write_text("# baseline\n", encoding="utf-8")
    (artifact_dir / "config_snapshot.json").write_text(
        json.dumps({"dataset": {"source": {"kind": "csv"}}}),
        encoding="utf-8",
    )
    (artifact_dir / config_path.name).write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")

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
    return run_id, config_path


def load_generated_app(app_path: Path) -> Any:
    module_name = f"_treehouse_export_app_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, app_path)
    assert spec is not None and spec.loader is not None, f"Could not load generated app module from {app_path}"
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(module_name, None)
    return module.app


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


def test_exported_model_bundle_scores_raw_records_with_generated_features(tmp_path: Path) -> None:
    frame, target = make_training_frame()
    feature_plan = build_feature_plan(
        {
            "feature_generation": {
                "max_new_features": 4,
                "top_k_numeric": 3,
                "operations": ["square", "product"],
            }
        },
        enabled=True,
    )
    preprocessor = fit_feature_preprocessor(frame, target=target, feature_generation_plan=feature_plan)
    prepared = transform_feature_frame(frame, preprocessor)
    model = GradientBoostingClassifier(random_state=42)
    model.fit(prepared, target)

    bundle = ExportedModelBundle(
        run_id="run-feature-123",
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
        metrics={"roc_auc": 0.92},
        model=model,
    )
    bundle_path = tmp_path / "model_bundle_with_features.pkl"
    save_exported_model_bundle(bundle, bundle_path)

    loaded = load_exported_model_bundle(bundle_path)
    predictions = loaded.predict_records(
        [
            {"age": 31, "segment": "pro", "visits": 7},
            {"age": 23, "segment": "basic", "visits": 1},
        ]
    )

    assert loaded.feature_preprocessor.generated_feature_specs
    assert len(predictions) == 2
    assert all(0.0 <= row["score"] <= 1.0 for row in predictions)


def test_export_model_artifact_matches_documented_file_and_manifest_contract(tmp_path: Path) -> None:
    project_root = tmp_path
    run_id, config_path = make_exportable_project(project_root)

    manifest = export_model_artifact(project_root, "marketing-leads")
    export_dir = Path(manifest["export_dir"])
    manifest_path = export_dir / "manifest.json"

    assert export_dir.exists()
    missing_files = sorted(filename for filename in REQUIRED_EXPORT_FILES if not (export_dir / filename).exists())
    assert not missing_files, f"Export is missing documented contract files: {missing_files}"

    copied_run_files = {"metrics.json", "summary.md", "config_snapshot.json", config_path.name}
    missing_copied_files = sorted(filename for filename in copied_run_files if not (export_dir / filename).exists())
    assert not missing_copied_files, f"Export did not copy existing run artifact files: {missing_copied_files}"

    persisted_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert persisted_manifest == manifest, "manifest.json on disk should match the export_model_artifact return value"

    missing_manifest_keys = sorted(REQUIRED_MANIFEST_KEYS.difference(manifest))
    assert not missing_manifest_keys, f"manifest.json is missing required contract keys: {missing_manifest_keys}"

    manifest_files = set(manifest["files"])
    missing_manifest_files = sorted((REQUIRED_EXPORT_FILES - {"manifest.json"} | copied_run_files) - manifest_files)
    assert not missing_manifest_files, f"manifest.json files list omits exported files: {missing_manifest_files}"
    assert "manifest.json" not in manifest_files

    assert manifest["config_key"] == "marketing-leads"
    assert manifest["run_id"] == run_id
    assert manifest["bundle_path"] == str(export_dir / BUNDLE_FILENAME)
    assert manifest["bundle_materialization"] == "existing"
    assert manifest["artifact_usage"] == "python_bundle"
    assert manifest["containerization"] == "dockerfile_included"
    assert "uvicorn app:app" in manifest["serve_command"]
    assert "docker build" in manifest["docker_build_command"]

    bundle = load_exported_model_bundle(export_dir / BUNDLE_FILENAME)
    predictions = bundle.predict_records([{"age": 31, "segment": "pro", "visits": 7}])
    assert bundle.run_id == run_id
    assert hasattr(bundle.model, "predict_proba"), "model_bundle.pkl should contain a trained scoring model"
    assert len(predictions) == 1
    assert set(predictions[0]) == {"prediction", "score"}


def test_generated_export_app_exposes_documented_prediction_interface(tmp_path: Path) -> None:
    testclient_module = pytest.importorskip("fastapi.testclient")
    project_root = tmp_path
    run_id, _ = make_exportable_project(project_root)
    manifest = export_model_artifact(project_root, "marketing-leads")
    export_dir = Path(manifest["export_dir"])

    app = load_generated_app(export_dir / "app.py")
    client = testclient_module.TestClient(app)

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json() == {"status": "ok", "run_id": run_id}

    schema = client.get("/schema")
    assert schema.status_code == 200
    assert schema.json() == {
        "registry_key": "marketing-leads",
        "run_id": run_id,
        "input_columns": ["age", "segment", "visits"],
        "target_name": "converted",
        "task_kind": "binary_classification",
        "class_labels": ["no", "yes"],
        "threshold": 0.5,
    }

    prediction = client.post("/predict", json={"records": [{"age": 31, "segment": "pro", "visits": 7}]})
    assert prediction.status_code == 200
    prediction_payload = prediction.json()
    assert set(prediction_payload) == {"predictions"}
    assert len(prediction_payload["predictions"]) == 1
    assert set(prediction_payload["predictions"][0]) == {"prediction", "score"}

    bad_request = client.post("/predict", json={"records": [{"age": 31}]})
    assert bad_request.status_code == 400
    assert "Missing required feature columns" in bad_request.json()["detail"]


def test_export_model_artifact_reports_missing_incumbent(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="No incumbent exists yet for config 'marketing-leads'"):
        export_model_artifact(tmp_path, "marketing-leads")
