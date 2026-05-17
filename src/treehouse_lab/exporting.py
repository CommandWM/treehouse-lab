from __future__ import annotations

import json
import pickle
import shutil
import textwrap
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from treehouse_lab import __version__
from treehouse_lab.config import load_experiment_config
from treehouse_lab.datasets import FeaturePreprocessor, load_dataset, split_dataset, transform_feature_frame
from treehouse_lab.journal import load_incumbent, load_run_entry

EXPORTS_DIRNAME = "exports"
BUNDLE_FILENAME = "model_bundle.pkl"


@dataclass(slots=True)
class ExportedModelBundle:
    run_id: str
    registry_key: str
    config_path: str
    target_name: str
    task_kind: str
    class_labels: list[str]
    primary_metric: str
    backend: str
    threshold: float | None
    feature_preprocessor: FeaturePreprocessor
    model_params: dict[str, Any]
    metrics: dict[str, float]
    model: Any

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("model", None)
        return payload

    def predict_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        prepared = transform_feature_frame(frame, self.feature_preprocessor)
        probabilities = self.model.predict_proba(prepared)
        predictions = self.model.predict(prepared)

        if self.task_kind == "multiclass_classification":
            result = pd.DataFrame(
                {
                    "score": probabilities.max(axis=1).astype(float),
                    "prediction": pd.Series(predictions).astype(int),
                }
            )
            if self.class_labels:
                result["predicted_label"] = pd.Series(
                    [
                        self.class_labels[int(index)] if int(index) < len(self.class_labels) else str(index)
                        for index in result["prediction"].tolist()
                    ],
                    dtype="string",
                )
            return result

        scores = probabilities[:, 1]
        threshold = 0.5 if self.threshold is None else float(self.threshold)
        binary_predictions = (scores >= threshold).astype(int)
        return pd.DataFrame(
            {
                "score": scores.astype(float),
                "prediction": binary_predictions.astype(int),
            }
        )

    def predict_records(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        frame = pd.DataFrame(records)
        result = self.predict_frame(frame)
        return json.loads(result.to_json(orient="records"))


def save_exported_model_bundle(bundle: ExportedModelBundle, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(bundle, handle)
    return path


def load_exported_model_bundle(path: str | Path) -> ExportedModelBundle:
    with Path(path).expanduser().open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, ExportedModelBundle):
        msg = f"Unsupported exported model bundle payload in {path}"
        raise ValueError(msg)
    return payload


def export_model_artifact(
    project_root: Path,
    config_key: str,
    run_id: str | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    run_entry = _resolve_run_entry(project_root, config_key, run_id)
    artifact_dir = Path(str(run_entry["artifact_dir"]))
    bundle_path = artifact_dir / BUNDLE_FILENAME
    materialization_status = "existing"
    if not bundle_path.exists():
        _rebuild_legacy_bundle(project_root, run_entry, bundle_path)
        materialization_status = "rebuilt_from_legacy_artifacts"

    resolved_output_dir = _resolve_output_dir(project_root, config_key, str(run_entry["run_id"]), output_dir)
    resolved_output_dir.mkdir(parents=True, exist_ok=False)

    copied_files: list[str] = []
    for filename in (
        BUNDLE_FILENAME,
        "assessment.json",
        "config_snapshot.json",
        "diagnosis.json",
        "metrics.json",
        "model_params.json",
        "summary.md",
        "feature_importances.csv",
        Path(str(run_entry["config_path"])).name,
    ):
        source_path = artifact_dir / filename
        if not source_path.exists():
            continue
        target_path = resolved_output_dir / filename
        shutil.copy2(source_path, target_path)
        copied_files.append(target_path.name)

    (resolved_output_dir / "app.py").write_text(_fastapi_app_template(), encoding="utf-8")
    copied_files.append("app.py")
    (resolved_output_dir / "Dockerfile").write_text(_dockerfile_template(), encoding="utf-8")
    copied_files.append("Dockerfile")
    (resolved_output_dir / ".dockerignore").write_text(_dockerignore_template(), encoding="utf-8")
    copied_files.append(".dockerignore")
    (resolved_output_dir / "requirements.txt").write_text(_requirements_template(), encoding="utf-8")
    copied_files.append("requirements.txt")
    (resolved_output_dir / "README.md").write_text(_export_readme_template(run_entry), encoding="utf-8")
    copied_files.append("README.md")

    manifest = {
        "config_key": config_key,
        "run_id": run_entry["run_id"],
        "source_artifact_dir": str(artifact_dir),
        "export_dir": str(resolved_output_dir),
        "bundle_path": str(resolved_output_dir / BUNDLE_FILENAME),
        "bundle_materialization": materialization_status,
        "artifact_usage": "python_bundle",
        "containerization": "dockerfile_included",
        "serve_command": "uvicorn app:app --host 0.0.0.0 --port 8000",
        "docker_build_command": "docker build -t treehouse-lab-export .",
        "docker_run_command": "docker run --rm -p 8000:8000 treehouse-lab-export",
        "files": copied_files,
    }
    (resolved_output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def _resolve_run_entry(project_root: Path, config_key: str, run_id: str | None) -> dict[str, Any]:
    if run_id:
        entry = load_run_entry(project_root, run_id)
        if entry is None:
            msg = f"Unknown run id: {run_id}"
            raise FileNotFoundError(msg)
        return entry

    incumbent = load_incumbent(project_root, config_key)
    if incumbent is None:
        msg = f"No incumbent exists yet for config '{config_key}'."
        raise FileNotFoundError(msg)
    entry = load_run_entry(project_root, str(incumbent["run_id"]))
    if entry is None:
        msg = f"Could not locate incumbent run entry for config '{config_key}'."
        raise FileNotFoundError(msg)
    return entry


def _resolve_output_dir(
    project_root: Path,
    config_key: str,
    run_id: str,
    output_dir: str | Path | None,
) -> Path:
    if output_dir is None:
        return project_root / EXPORTS_DIRNAME / config_key / run_id
    candidate = Path(output_dir).expanduser()
    return candidate if candidate.is_absolute() else (project_root / candidate).resolve()


def _rebuild_legacy_bundle(project_root: Path, run_entry: dict[str, Any], bundle_path: Path) -> Path:
    from treehouse_lab.runner import TreehouseLabRunner

    config_path = Path(str(run_entry.get("config_path", "")))
    if not config_path.exists():
        msg = f"Cannot rebuild a legacy bundle because the config path is missing: {config_path}"
        raise FileNotFoundError(msg)

    runner = TreehouseLabRunner(config_path)
    config = load_experiment_config(config_path)
    bundle = load_dataset(config, project_root)
    split = split_dataset(bundle, config)

    params = dict(run_entry.get("params", {}))
    if not params:
        incumbent = load_incumbent(project_root, config_path.stem)
        params = {} if incumbent is None else dict(incumbent.get("params", {}))
    if not params:
        msg = f"Cannot rebuild a legacy bundle for run {run_entry.get('run_id')} because no saved params were found."
        raise FileNotFoundError(msg)

    model, backend = runner._build_model(params)
    model.fit(split.X_train, split.y_train)
    exported = ExportedModelBundle(
        run_id=str(run_entry["run_id"]),
        registry_key=config_path.stem,
        config_path=str(config_path),
        target_name=bundle.target_name,
        task_kind=str(bundle.target_profile["task_kind"]),
        class_labels=[str(label["raw"]) for label in bundle.target_profile.get("class_labels", [])],
        primary_metric=config.primary_metric,
        backend=str(run_entry.get("backend") or backend),
        threshold=0.5 if str(bundle.target_profile["task_kind"]) == "binary_classification" else None,
        feature_preprocessor=split.preprocessor,
        model_params=params,
        metrics=dict(run_entry.get("metrics", {})),
        model=model,
    )
    return save_exported_model_bundle(exported, bundle_path)


def _fastapi_app_template() -> str:
    template = textwrap.dedent(
        """
        from pathlib import Path

        import pandas as pd
        from fastapi import FastAPI, HTTPException
        from pydantic import BaseModel, Field

        from treehouse_lab.exporting import load_exported_model_bundle


        BUNDLE_PATH = Path(__file__).resolve().parent / "model_bundle.pkl"
        bundle = load_exported_model_bundle(BUNDLE_PATH)

        app = FastAPI(title="Treehouse Lab Exported Model", version="__TREEHOUSE_LAB_VERSION__")


        class PredictRequest(BaseModel):
            records: list[dict[str, object]] = Field(min_length=1)


        @app.get("/health")
        def health() -> dict[str, str]:
            return {"status": "ok", "run_id": bundle.run_id}


        @app.get("/schema")
        def schema() -> dict[str, object]:
            return {
                "registry_key": bundle.registry_key,
                "run_id": bundle.run_id,
                "input_columns": bundle.feature_preprocessor.input_columns,
                "target_name": bundle.target_name,
                "task_kind": bundle.task_kind,
                "class_labels": bundle.class_labels,
                "threshold": bundle.threshold,
            }


        @app.post("/predict")
        def predict(payload: PredictRequest) -> dict[str, object]:
            try:
                predictions = bundle.predict_records(payload.records)
            except Exception as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            return {"predictions": predictions}
        """
    )
    return template.strip().replace("__TREEHOUSE_LAB_VERSION__", __version__) + "\n"


def _requirements_template() -> str:
    return "\n".join(
        [
            "fastapi>=0.115",
            "uvicorn>=0.34",
            "pandas>=2.2",
            "scikit-learn>=1.5",
            "xgboost>=2.1",
            "pyyaml>=6.0",
            "",
        ]
    )


def _dockerfile_template() -> str:
    return textwrap.dedent(
        """
        FROM python:3.13-slim

        WORKDIR /app

        COPY requirements.txt .
        RUN pip install --no-cache-dir -r requirements.txt

        COPY . .

        EXPOSE 8000

        CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
        """
    ).strip() + "\n"


def _dockerignore_template() -> str:
    return "\n".join(
        [
            "__pycache__/",
            "*.pyc",
            ".pytest_cache/",
            "",
        ]
    )


def _export_readme_template(run_entry: dict[str, Any]) -> str:
    run_id = str(run_entry["run_id"])
    config_name = Path(str(run_entry["config_path"])).name
    return textwrap.dedent(
        f"""
        # Treehouse Lab Export

        This package is centered on the exported model artifact in `model_bundle.pkl`.
        The generated FastAPI app and Dockerfile are optional wrappers around that artifact.

        ## Included files

        - `model_bundle.pkl`: trained model plus fitted preprocessing contract
        - `{config_name}`: config snapshot used for the run
        - `metrics.json`, `assessment.json`, `diagnosis.json`, `summary.md`: run context
        - `app.py`: optional minimal scoring API
        - `Dockerfile`: optional containerization path for the scoring API

        ## Option 1: Reuse the artifact directly

        ```python
        from treehouse_lab.exporting import load_exported_model_bundle

        bundle = load_exported_model_bundle("model_bundle.pkl")
        print(bundle.feature_preprocessor.input_columns)
        predictions = bundle.predict_records([...])
        print(predictions)
        ```

        Pass records that include all expected input columns.

        ## Option 2: Run the optional local scoring API

        ```bash
        uvicorn app:app --host 0.0.0.0 --port 8000
        ```

        ## Option 3: Build the optional container image

        ```bash
        docker build -t treehouse-lab-export .
        docker run --rm -p 8000:8000 treehouse-lab-export
        ```

        ## Run metadata

        - `run_id`: `{run_id}`
        """
    ).strip() + "\n"


def _export_readme_template(run_entry: dict[str, Any]) -> str:
    return textwrap.dedent(
        f"""
        # Exported Treehouse Lab Model

        This bundle was exported from run `{run_entry["run_id"]}`.

        Files:
        - `model_bundle.pkl`: serialized model plus preprocessing contract
        - `app.py`: minimal FastAPI scoring app
        - `manifest.json`: export metadata

        Run locally:

        ```bash
        pip install -e '.[web]'
        uvicorn app:app --host 0.0.0.0 --port 8000
        ```
        """
    ).strip() + "\n"
