# Export Contract

Treehouse Lab exports are handoff packages for a trained tabular classification run. The primary artifact is the Python model bundle; the generated FastAPI app is a thin convenience scorer around that bundle.

This is a stable handoff contract for the current implementation. It is not hardened production serving.

## Creating an Export

Export the incumbent for a dataset config:

```bash
treehouse-lab export configs/datasets/<config-key>.yaml
```

By default, the export is written to:

```text
exports/<config-key>/<run-id>/
```

The CLI can also export a specific run with `--run-id` or write to a caller-provided directory with `--output-dir`.

## Included Files

A successful export includes these generated or copied files:

- `model_bundle.pkl`: serialized `ExportedModelBundle`, containing the trained model, fitted preprocessing contract, run metadata, metrics, params, and task metadata.
- `app.py`: generated FastAPI scorer with `/health`, `/schema`, and `/predict`.
- `Dockerfile`: minimal container wrapper for the generated scorer.
- `.dockerignore`: minimal Docker ignore file.
- `requirements.txt`: runtime Python dependencies for the generated scorer.
- `README.md`: export-local quickstart.
- `manifest.json`: export metadata.

The exporter also copies these run artifact files when they exist in the source run directory:

- `assessment.json`
- `config_snapshot.json`
- `diagnosis.json`
- `metrics.json`
- `model_params.json`
- `summary.md`
- `feature_importances.csv`
- the original dataset config file, using its filename from the run metadata

The export does not currently copy every file that a run may have produced. For example, run context, split summary, and feature-generation detail files can exist in `runs/<run-id>/` but are not part of the current export file list.

## Manifest Fields

`manifest.json` is written after the package files are copied or generated. Current fields are:

- `config_key`: dataset config key used for the export.
- `run_id`: exported run id.
- `source_artifact_dir`: source `runs/<run-id>/` artifact directory.
- `export_dir`: destination export directory.
- `bundle_path`: path to the exported `model_bundle.pkl`.
- `bundle_materialization`: `existing` when the run already had a bundle, or `rebuilt_from_legacy_artifacts` when the exporter rebuilt it from older run artifacts.
- `artifact_usage`: currently `python_bundle`.
- `containerization`: currently `dockerfile_included`.
- `serve_command`: generated local `uvicorn` command.
- `docker_build_command`: generated Docker build command.
- `docker_run_command`: generated Docker run command.
- `files`: names of files copied or generated before `manifest.json` is written.

Current detail: `manifest.json` itself exists in the export directory but is not included in the manifest `files` list.

## Model Bundle Contract

`model_bundle.pkl` contains an `ExportedModelBundle`. Downstream Python callers can load it directly:

```python
from treehouse_lab.exporting import load_exported_model_bundle

bundle = load_exported_model_bundle("exports/<config-key>/<run-id>/model_bundle.pkl")
print(bundle.feature_preprocessor.input_columns)

predictions = bundle.predict_records([
    {
        "feature_a": 42,
        "feature_b": "standard",
        "feature_c": 3.14,
    }
])
```

Records must include every column listed in `bundle.feature_preprocessor.input_columns`. Extra columns are ignored by the current preprocessing path. Do not include the target column unless it is also listed as an input feature, which it should not be for normal dataset configs.

Because this is a pickle-based Python artifact, only load bundles from trusted sources in a compatible Python environment with Treehouse Lab available.

## Generated Scorer Endpoints

The generated FastAPI app loads `model_bundle.pkl` from the same directory as `app.py`.

Run it from the export directory:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

Available endpoints:

- `GET /health`: returns `{"status": "ok", "run_id": "<run-id>"}`.
- `GET /schema`: returns scorer metadata and the expected input columns.
- `POST /predict`: scores one or more records.

`GET /schema` returns:

```json
{
  "registry_key": "<config-key>",
  "run_id": "<run-id>",
  "input_columns": ["feature_a", "feature_b", "feature_c"],
  "target_name": "<target-column>",
  "task_kind": "binary_classification",
  "class_labels": ["negative", "positive"],
  "threshold": 0.5
}
```

For multiclass exports, `task_kind` is `multiclass_classification` and `threshold` is `null`.

## Prediction Request Shape

Send a JSON object with a non-empty `records` array:

```json
{
  "records": [
    {
      "feature_a": 42,
      "feature_b": "standard",
      "feature_c": 3.14
    }
  ]
}
```

Each record is one row. Its keys must include every value returned by `/schema` in `input_columns`. Values should be JSON scalars that pandas can coerce through the fitted preprocessing contract:

- numeric training columns are coerced with `pandas.to_numeric`; invalid numeric values become missing and are filled with the training median.
- categorical training columns are string-normalized and one-hot encoded against the training-time categories.
- missing required columns return HTTP 400.
- extra columns are ignored.

## Prediction Response Shape

`POST /predict` returns:

```json
{
  "predictions": [
    {
      "score": 0.82,
      "prediction": 1
    }
  ]
}
```

For binary classification:

- `score` is the model probability for class `1`.
- `prediction` is `1` when `score >= threshold`, otherwise `0`.
- the current generated response does not include `predicted_label` for binary exports.

For multiclass classification:

- `score` is the maximum class probability.
- `prediction` is the integer predicted class index.
- `predicted_label` is included when the bundle has class labels.

Prediction failures are returned as HTTP 400 with the underlying error text in the FastAPI `detail` field.

## Curl Example

Start the scorer, inspect the schema, then send records whose keys match `input_columns` exactly:

```bash
cd exports/<config-key>/<run-id>
uvicorn app:app --host 0.0.0.0 --port 8000
```

```bash
curl -s http://localhost:8000/schema
```

```bash
curl -s -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "records": [
      {
        "feature_a": 42,
        "feature_b": "standard",
        "feature_c": 3.14
      }
    ]
  }'
```

Replace `feature_a`, `feature_b`, and `feature_c` with the exact columns returned by `/schema`.

## How Downstream Systems Should Call It

Downstream systems should treat the export as a run-specific scoring contract:

1. Read `manifest.json` and record `config_key`, `run_id`, `bundle_materialization`, and `bundle_path`.
2. Start the generated FastAPI app only in an environment where the `treehouse_lab` Python package and runtime dependencies are installed.
3. Call `GET /schema` at integration time or process startup.
4. Build each prediction record from `/schema.input_columns`.
5. Send batches to `POST /predict` as `{"records": [...]}`.
6. Interpret scores based on `/schema.task_kind` and `/schema.threshold`.
7. Treat HTTP 400 responses as request/schema/data errors that need caller-side correction.

If the scorer is exposed outside a trusted local environment, put a production service boundary in front of it. Add authentication, TLS, request limits, observability, deployment policy, and model/version routing outside the generated app.

## Current Limitations

- The generated FastAPI app is a minimal scorer, not a hardened production serving layer.
- There is no built-in authentication, authorization, TLS, rate limiting, request size limit, audit logging, metrics, or tracing.
- There is no model registry protocol, canary policy, rollback mechanism, or multi-model routing.
- Request validation only checks that `records` is a non-empty list of objects; feature presence and type issues are handled by the preprocessing path.
- The generated app imports `treehouse_lab.exporting`; the export directory does not currently vendor the Treehouse Lab source package.
- The generated `requirements.txt` lists runtime libraries but not the local Treehouse Lab package itself.
- The Dockerfile is a convenience wrapper around the generated scorer and should be validated in the target environment before use.
- The bundle format is Python and pickle based. It is not a language-neutral model artifact.
- The current task surface is classification. Binary and multiclass responses have slightly different fields.
- No Salesforce-specific handoff, deployment, or API integration is implemented by the repo today.
