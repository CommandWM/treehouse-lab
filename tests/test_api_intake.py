from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from treehouse_lab import api


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    project_root = tmp_path
    config_dir = project_root / "configs" / "datasets"
    config_dir.mkdir(parents=True)

    glossary_path = project_root / "docs" / "glossary.md"
    glossary_path.parent.mkdir(parents=True)
    glossary_path.write_text("# Glossary\n\n## Intake\nDataset intake terms.\n", encoding="utf-8")

    monkeypatch.setattr(api, "PROJECT_ROOT", project_root)
    monkeypatch.setattr(api, "DATASET_CONFIG_DIR", config_dir)
    monkeypatch.setattr(api, "GLOSSARY_PATH", glossary_path)

    return TestClient(api.app)


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(path, index=False)


def test_inspect_dataset_reports_schema_and_binary_target(client: TestClient, tmp_path: Path) -> None:
    dataset_path = tmp_path / "customer_churn.csv"
    write_csv(
        dataset_path,
        pd.DataFrame(
            {
                "age": [34, 28, 51],
                "plan": ["pro", "basic", "basic"],
                "churned": ["yes", "no", "yes"],
            }
        ),
    )

    response = client.post(
        "/api/intake/inspect",
        json={"path": str(dataset_path), "target_column": "churned"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["row_count"] == 3
    assert payload["feature_count"] == 2
    assert payload["target"]["binary_supported"] is True
    assert payload["target"]["class_labels"] == [
        {"raw": "no", "encoded": 0},
        {"raw": "yes", "encoded": 1},
    ]


def test_inspect_dataset_rejects_unknown_target_column(client: TestClient, tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.csv"
    write_csv(dataset_path, pd.DataFrame({"a": [1, 2], "target": [0, 1]}))

    response = client.post(
        "/api/intake/inspect",
        json={"path": str(dataset_path), "target_column": "missing_target"},
    )

    assert response.status_code == 400
    assert "Unknown target column" in response.text


def test_create_dataset_config_writes_explicit_yaml(client: TestClient, tmp_path: Path) -> None:
    dataset_path = tmp_path / "marketing_leads.csv"
    write_csv(
        dataset_path,
        pd.DataFrame(
            {
                "visits": [3, 8, 2, 10],
                "segment": ["paid", "organic", "paid", "partner"],
                "converted": ["yes", "no", "yes", "no"],
            }
        ),
    )

    response = client.post(
        "/api/intake/create",
        json={
            "path": str(dataset_path),
            "target_column": "converted",
            "name": "Marketing Leads",
            "config_key": "marketing-leads",
            "description": "Lead conversion benchmark.",
            "objective": "Establish a baseline before bounded loop steps.",
            "validation_size": 0.25,
            "test_size": 0.25,
            "stratify": True,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["key"] == "marketing-leads"
    assert payload["config"]["task"]["kind"] == "binary_classification"
    assert payload["config"]["source"]["kind"] == "csv"
    assert payload["config"]["source"]["target_column"] == "converted"
    assert payload["config"]["source"]["path"] == "marketing_leads.csv"
    assert payload["inspection"]["target"]["binary_supported"] is True

    config_path = tmp_path / "configs" / "datasets" / "marketing-leads.yaml"
    assert config_path.exists()
    config_text = config_path.read_text(encoding="utf-8")
    assert "kind: csv" in config_text
    assert "kind: binary_classification" in config_text
    assert "target_column: converted" in config_text
    assert "path: marketing_leads.csv" in config_text


def test_create_dataset_config_supports_multiclass_target(client: TestClient, tmp_path: Path) -> None:
    dataset_path = tmp_path / "outcomes.csv"
    write_csv(
        dataset_path,
        pd.DataFrame(
            {
                "score": [1, 2, 3, 4],
                "outcome": ["win", "loss", "draw", "draw"],
            }
        ),
    )

    response = client.post(
        "/api/intake/create",
        json={
            "path": str(dataset_path),
            "target_column": "outcome",
            "name": "Outcomes",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["config"]["task"]["kind"] == "multiclass_classification"
    assert payload["config"]["primary_metric"] == "accuracy"
    assert payload["inspection"]["target"]["multiclass_supported"] is True
