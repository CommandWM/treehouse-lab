from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient
import pytest
import yaml

from treehouse_lab import api, llm
from treehouse_lab.runtime_settings import llm_settings_path


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    project_root = tmp_path
    config_dir = project_root / "configs" / "datasets"
    config_dir.mkdir(parents=True)

    glossary_path = project_root / "docs" / "glossary.md"
    glossary_path.parent.mkdir(parents=True)
    glossary_path.write_text("# Glossary\n\n## Diagnosis\nWhy the loop picked the next move.\n", encoding="utf-8")

    search_space_path = project_root / "configs" / "search_space.yaml"
    search_space_path.parent.mkdir(parents=True, exist_ok=True)
    search_space_path.write_text(
        yaml.safe_dump(
            {
                "xgboost": {
                    "max_depth": [2, 10],
                    "min_child_weight": [1, 10],
                    "subsample": [0.5, 1.0],
                    "colsample_bytree": [0.5, 1.0],
                    "learning_rate": [0.01, 0.3],
                    "n_estimators": [100, 600],
                },
                "policy": {"allow_feature_generation": False},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    config_path = config_dir / "bank-valid-test.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "dataset": {
                    "source": {
                        "kind": "csv",
                        "name": "Bank Valid",
                        "target_column": "y",
                        "path": "custom_datasets/bank-full.csv",
                    },
                    "split": {
                        "validation_size": 0.2,
                        "test_size": 0.2,
                        "stratify": True,
                    },
                },
                "benchmark": {
                    "pack": "user",
                    "profile": "dataset_intake",
                    "objective": "Predict whether a customer subscribes to a term deposit.",
                },
                "evaluation_policy": {
                    "require_promotion_for_readiness": True,
                },
                "experiment": {
                    "name": "bank-valid-baseline",
                    "description": "Bank marketing benchmark.",
                    "primary_metric": "roc_auc",
                    "promote_if_delta_at_least": 0.003,
                    "max_runtime_minutes": 10,
                    "seed": 42,
                    "baseline_hypothesis": "A disciplined XGBoost baseline should establish a credible incumbent.",
                },
                "model": {"params": {}},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(api, "PROJECT_ROOT", project_root)
    monkeypatch.setattr(api, "DATASET_CONFIG_DIR", config_dir)
    monkeypatch.setattr(api, "GLOSSARY_PATH", glossary_path)
    monkeypatch.delenv("TREEHOUSE_LAB_LLM_PROVIDER", raising=False)
    monkeypatch.delenv("TREEHOUSE_LAB_LLM_MODEL", raising=False)
    monkeypatch.delenv("TREEHOUSE_LAB_LOOP_LLM_SELECTION", raising=False)
    monkeypatch.delenv("TREEHOUSE_LAB_OLLAMA_BASE_URL", raising=False)
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("TREEHOUSE_LAB_OPENAI_COMPATIBLE_BASE_URL", raising=False)
    monkeypatch.delenv("TREEHOUSE_LAB_OPENAI_COMPATIBLE_API_KEY", raising=False)
    monkeypatch.delenv("TREEHOUSE_LAB_AGENT_CLI", raising=False)

    return TestClient(api.app)


def test_llm_settings_round_trip(client: TestClient, tmp_path: Path) -> None:
    response = client.post(
        "/api/settings/llm",
        json={
            "provider": "ollama",
            "model": "gpt-oss:20b",
            "loop_llm_selection": True,
            "ollama_base_url": "https://ollama.com",
            "ollama_api_key": "rotated-key",
            "agent_cli": "codex",
            "openai_compatible_base_url": "",
            "openai_compatible_api_key": "",
            "openai_api_key": "",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["provider"] == "ollama"
    assert payload["model"] == "gpt-oss:20b"
    assert payload["loop_llm_selection"] is True
    assert payload["ollama_api_key"] == "rotated-key"
    assert Path(payload["storage_path"]) == llm_settings_path(tmp_path)

    stored = client.get("/api/settings/llm")
    assert stored.status_code == 200
    assert stored.json()["ollama_api_key"] == "rotated-key"


def test_saved_settings_drive_advisor_without_env(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client.post(
        "/api/settings/llm",
        json={
            "provider": "ollama",
            "model": "gpt-oss:20b",
            "loop_llm_selection": False,
            "ollama_base_url": "https://ollama.com",
            "ollama_api_key": "rotated-key",
            "agent_cli": "codex",
            "openai_compatible_base_url": "",
            "openai_compatible_api_key": "",
            "openai_api_key": "",
        },
    )

    captured: dict[str, object] = {}

    class FakeHttpResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {"model": "gpt-oss:20b", "message": {"content": "Current state:\nReady.\n\nNext step:\nTry imbalance adjustment.\n\nWatchouts:\nNone."}}

    class FakeRequests:
        def post(self, url: str, **kwargs: object) -> FakeHttpResponse:
            captured["url"] = url
            captured["headers"] = kwargs["headers"]
            return FakeHttpResponse()

    monkeypatch.setattr(llm, "requests", FakeRequests())

    response = client.post("/api/configs/bank-valid-test/advisor", json={"question": "What next?"})

    assert response.status_code == 200
    assert response.json()["provider"] == "ollama"
    assert captured["url"] == "https://ollama.com/api/chat"
    assert captured["headers"]["Authorization"] == "Bearer rotated-key"


def test_llm_settings_reads_ollama_cloud_key_alias_from_environment(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TREEHOUSE_LAB_OLLAMA_BASE_URL", "https://ollama.com")
    monkeypatch.setenv("VIOLAAMA_CLOUD_KEY", "alias-key")

    response = client.get("/api/settings/llm")

    assert response.status_code == 200
    payload = response.json()
    assert payload["ollama_base_url"] == "https://ollama.com"
    assert payload["ollama_api_key"] == "alias-key"


def test_loop_llm_selection_reads_from_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TREEHOUSE_LAB_LOOP_LLM_SELECTION", "true")
    assert llm.llm_loop_selection_enabled(tmp_path) is True

    monkeypatch.setenv("TREEHOUSE_LAB_LOOP_LLM_SELECTION", "false")
    assert llm.llm_loop_selection_enabled(tmp_path) is False
