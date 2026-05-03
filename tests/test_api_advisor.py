from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
import yaml

from treehouse_lab import api, llm


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

    monkeypatch.setattr(api, "PROJECT_ROOT", project_root)
    monkeypatch.setattr(api, "DATASET_CONFIG_DIR", config_dir)
    monkeypatch.setattr(api, "GLOSSARY_PATH", glossary_path)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    monkeypatch.delenv("TREEHOUSE_LAB_LLM_MODEL", raising=False)
    monkeypatch.delenv("TREEHOUSE_LAB_LLM_PROVIDER", raising=False)
    monkeypatch.delenv("TREEHOUSE_LAB_OLLAMA_BASE_URL", raising=False)
    monkeypatch.delenv("TREEHOUSE_LAB_AGENT_CLI", raising=False)
    monkeypatch.delenv("TREEHOUSE_LAB_OPENAI_COMPATIBLE_BASE_URL", raising=False)
    monkeypatch.delenv("TREEHOUSE_LAB_OPENAI_COMPATIBLE_API_KEY", raising=False)

    return TestClient(api.app)


def write_config(config_dir: Path, key: str = "bank-valid-test") -> Path:
    config_path = config_dir / f"{key}.yaml"
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
                "model": {
                    "params": {
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
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return config_path


def write_bank_history(project_root: Path, key: str = "bank-valid-test") -> None:
    runs_dir = project_root / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    incumbent_dir = runs_dir / "incumbents"
    incumbent_dir.mkdir(parents=True, exist_ok=True)

    baseline_entry = {
        "run_id": "20260418T150216031206Z-baseline",
        "registry_key": key,
        "name": "baseline",
        "metric": 0.943432469854643,
        "promoted": True,
        "comparison_to_incumbent": {"incumbent_metric": None, "delta": None, "threshold": 0.003},
        "params": {
            "n_estimators": 300,
            "max_depth": 6,
            "learning_rate": 0.05,
            "min_child_weight": 1,
            "subsample": 0.9,
            "colsample_bytree": 0.8,
        },
        "metrics": {
            "roc_auc": 0.943432469854643,
            "train_roc_auc": 0.9733589267531466,
            "validation_roc_auc": 0.943432469854643,
            "test_roc_auc": 0.9342592976197099,
        },
        "split_summary": {
            "validation_positive_rate": 0.117,
            "train_positive_rate": 0.117,
        },
        "assessment": {
            "benchmark_status": "baseline_established",
            "implementation_readiness": "implementation_ready",
        },
        "diagnosis": {
            "primary_tag": "class_imbalance",
            "summary": "Validation roc_auc is 0.9434. Positive rate is 0.1170, so imbalance handling is worth considering.",
        },
        "reason_codes": ["baseline_established", "diagnosis_class_imbalance"],
    }

    repeated_entry = {
        "registry_key": key,
        "name": "learning-rate-tradeoff",
        "metric": 0.9440586039754366,
        "promoted": False,
        "comparison_to_incumbent": {
            "incumbent_metric": 0.943432469854643,
            "delta": 0.0006261341207935978,
            "threshold": 0.003,
        },
        "assessment": {
            "benchmark_status": "needs_more_work",
            "implementation_readiness": "needs_more_work",
        },
        "diagnosis": {
            "primary_tag": "class_imbalance",
            "summary": "Validation roc_auc is 0.9434. Positive rate is 0.1170, so imbalance handling is worth considering. Recent deltas are small, so the loop may be plateauing.",
        },
        "proposal": {
            "mutation_type": "learning_rate_tradeoff",
            "mutation_name": "learning-rate-tradeoff",
            "params_override": {"learning_rate": 0.04, "n_estimators": 405},
        },
        "reason_codes": ["diagnosis_class_imbalance", "diagnosis_plateau", "rejected_below_threshold"],
    }

    entries = [baseline_entry]
    for suffix in ("318146", "999824", "678461", "443770"):
        entry = dict(repeated_entry)
        entry["run_id"] = f"20260418T1504{suffix}Z-learning-rate-tradeoff"
        entries.append(entry)

    journal_path = runs_dir / "journal.jsonl"
    with journal_path.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry, sort_keys=True) + "\n")

    incumbent_dir.joinpath(f"{key}.json").write_text(
        json.dumps(
            {
                "artifact_dir": str(project_root / "runs" / baseline_entry["run_id"]),
                "assessment": baseline_entry["assessment"],
                "config_path": str(project_root / "configs" / "datasets" / f"{key}.yaml"),
                "diagnosis": baseline_entry["diagnosis"],
                "metric": baseline_entry["metric"],
                "metrics": baseline_entry["metrics"],
                "params": baseline_entry["params"],
                "registry_key": key,
                "run_id": baseline_entry["run_id"],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def test_advisor_reports_unavailable_without_ollama_cloud_key(
    client: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    write_config(tmp_path / "configs" / "datasets")
    monkeypatch.setenv("TREEHOUSE_LAB_OLLAMA_BASE_URL", "https://ollama.com")

    response = client.post("/api/configs/bank-valid-test/advisor", json={"question": "What should I do next?"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "unavailable"
    assert "OLLAMA_API_KEY" in payload["message"]


def test_advisor_returns_grounded_answer_with_mocked_ollama(
    client: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    write_config(tmp_path / "configs" / "datasets")
    write_bank_history(tmp_path)

    captured: dict[str, object] = {"calls": []}

    class FakeHttpResponse:
        def __init__(self, payload: dict[str, object]) -> None:
            self.payload = payload

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return self.payload

    class FakeRequests:
        def post(self, url: str, **kwargs: object) -> FakeHttpResponse:
            kwargs["url"] = url
            captured["calls"].append(kwargs)
            prompt_text = kwargs["json"]["messages"][1]["content"]
            if "Select the next bounded proposal." in prompt_text:
                selection_payload = json.loads(prompt_text.split("Candidate Context JSON:\n", 1)[1])
                imbalance_candidate = next(
                    candidate
                    for candidate in selection_payload["candidates"]
                    if candidate["mutation_type"] == "imbalance_adjustment"
                )
                return FakeHttpResponse(
                    {
                        "model": "gpt-oss:20b",
                        "message": {
                            "content": json.dumps(
                                {
                                    "selected_proposal_id": imbalance_candidate["proposal_id"],
                                    "rationale": "Bounded class weighting is the next attributable move.",
                                }
                            )
                        },
                    }
                )
            return FakeHttpResponse(
                {
                    "model": "gpt-oss:20b",
                    "message": {
                        "content": (
                            "Current state:\nBaseline is strong and recent learning-rate trades repeated without clearing the promotion bar.\n\n"
                            "Next step:\nTry the imbalance adjustment next because the positive rate is 0.117 and the loop is plateauing.\n\n"
                            "Watchouts:\nDo not treat the held-out test set as a search target."
                        )
                    },
                }
            )

    monkeypatch.setenv("TREEHOUSE_LAB_LLM_PROVIDER", "ollama")
    monkeypatch.setenv("TREEHOUSE_LAB_OLLAMA_BASE_URL", "https://ollama.com")
    monkeypatch.setenv("TREEHOUSE_LAB_LLM_MODEL", "gpt-oss:20b")
    monkeypatch.setenv("OLLAMA_API_KEY", "test-key")
    monkeypatch.setattr(llm, "requests", FakeRequests())

    response = client.post(
        "/api/configs/bank-valid-test/advisor",
        json={"question": "Why is the bank loop stalling, and what should I try next?"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "available"
    assert payload["provider"] == "ollama"
    assert "plateauing" in payload["answer"]
    assert payload["recommended_proposal"]["mutation_type"] == "imbalance_adjustment"
    assert payload["recommended_proposal"]["llm_review"]["status"] == "available"
    assert payload["grounding"]["recent_mutations"] == [
        "baseline",
        "learning-rate-tradeoff",
        "learning-rate-tradeoff",
        "learning-rate-tradeoff",
        "learning-rate-tradeoff",
    ]
    assert payload["grounding"]["bounded_references"][0]["path"] == "configs/search_space.yaml"
    assert payload["grounding"]["proposal_grounding"]["mutation_type"] == "imbalance_adjustment"

    request = captured["calls"][0]
    assert request["headers"]["Authorization"] == "Bearer test-key"
    assert request["url"] == "https://ollama.com/api/chat"
    messages = request["json"]["messages"]
    prompt_text = messages[1]["content"]
    assert "class_imbalance" in prompt_text
    assert "learning-rate-tradeoff" in prompt_text
    assert "Why is the bank loop stalling" in prompt_text
    assert len(captured["calls"]) == 2


def test_advisor_returns_grounded_answer_with_mocked_openai_fallback(
    client: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    write_config(tmp_path / "configs" / "datasets")
    write_bank_history(tmp_path)

    captured: dict[str, object] = {}

    class FakeResponse:
        output_text = (
            "Current state:\nBaseline is strong.\n\n"
            "Next step:\nTry the imbalance adjustment.\n\n"
            "Watchouts:\nDo not touch the held-out test set."
        )

    class FakeResponses:
        def create(self, **kwargs: object) -> FakeResponse:
            captured.update(kwargs)
            return FakeResponse()

    class FakeOpenAI:
        def __init__(self) -> None:
            self.responses = FakeResponses()

    monkeypatch.setenv("TREEHOUSE_LAB_LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(llm, "OpenAI", FakeOpenAI)

    response = client.post(
        "/api/configs/bank-valid-test/advisor",
        json={"question": "What should I do next?"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "available"
    assert payload["provider"] == "openai"
    request_input = captured["input"]
    assert isinstance(request_input, list)


def test_advisor_returns_grounded_answer_with_mocked_openai_compatible(
    client: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    write_config(tmp_path / "configs" / "datasets")
    write_bank_history(tmp_path)

    captured: dict[str, object] = {"calls": []}

    class FakeChatCompletions:
        def create(self, **kwargs: object) -> object:
            captured["calls"].append(kwargs)

            class Response:
                def model_dump(self) -> dict[str, object]:
                    return {
                        "choices": [
                            {
                                "message": {
                                    "content": "Current state:\nPlateau.\n\nNext step:\nTry imbalance adjustment.\n\nWatchouts:\nKeep the test set held out."
                                }
                            }
                        ]
                    }

            return Response()

    class FakeClient:
        def __init__(self, **kwargs: object) -> None:
            captured["client_kwargs"] = kwargs
            self.chat = type("Chat", (), {"completions": FakeChatCompletions()})()

    monkeypatch.setenv("TREEHOUSE_LAB_LLM_PROVIDER", "openai_compatible")
    monkeypatch.setenv("TREEHOUSE_LAB_OPENAI_COMPATIBLE_BASE_URL", "https://compat.example.com/v1")
    monkeypatch.setenv("TREEHOUSE_LAB_OPENAI_COMPATIBLE_API_KEY", "compat-key")
    monkeypatch.setenv("TREEHOUSE_LAB_LLM_MODEL", "provider/model")
    monkeypatch.setattr(llm, "OpenAI", FakeClient)

    response = client.post(
        "/api/configs/bank-valid-test/advisor",
        json={"question": "What should I do next?"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "available"
    assert payload["provider"] == "openai_compatible"
    assert captured["client_kwargs"] == {"base_url": "https://compat.example.com/v1", "api_key": "compat-key"}
    messages = captured["calls"][0]["messages"]
    assert isinstance(messages, list)
    assert "Question: What should I do next?" in messages[1]["content"]


def test_advisor_returns_grounded_answer_with_mocked_codex_cli(
    client: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    write_config(tmp_path / "configs" / "datasets")
    write_bank_history(tmp_path)

    captured: dict[str, object] = {"calls": []}

    def fake_run(cmd: list[str], **kwargs: object) -> object:
        captured["calls"].append(cmd)
        output_index = cmd.index("--output-last-message") + 1
        Path(cmd[output_index]).write_text(
            "Current state:\nPlateau.\n\nNext step:\nTry imbalance adjustment.\n\nWatchouts:\nStay inside the declared templates.",
            encoding="utf-8",
        )

        class Result:
            returncode = 0
            stdout = ""
            stderr = ""

        return Result()

    monkeypatch.setenv("TREEHOUSE_LAB_LLM_PROVIDER", "agent_cli")
    monkeypatch.setenv("TREEHOUSE_LAB_AGENT_CLI", "codex")
    monkeypatch.setenv("TREEHOUSE_LAB_LLM_MODEL", "gpt-5.4-mini")
    monkeypatch.setattr(llm.shutil, "which", lambda name: f"/usr/local/bin/{name}")
    monkeypatch.setattr(llm.subprocess, "run", fake_run)

    response = client.post(
        "/api/configs/bank-valid-test/advisor",
        json={"question": "What should I do next?"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "available"
    assert payload["provider"] == "agent_cli:codex"
    assert captured["calls"][0][:6] == ["codex", "exec", "-C", str(tmp_path), "--sandbox", "read-only"]
    assert "--model" in captured["calls"][0]


def test_advisor_returns_grounded_answer_with_mocked_claude_cli(
    client: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    write_config(tmp_path / "configs" / "datasets")
    write_bank_history(tmp_path)

    captured: dict[str, object] = {"calls": []}

    def fake_run(cmd: list[str], **kwargs: object) -> object:
        captured["calls"].append(cmd)

        class Result:
            returncode = 0
            stdout = "Current state:\nPlateau.\n\nNext step:\nTry imbalance adjustment.\n\nWatchouts:\nKeep the split policy fixed."
            stderr = ""

        return Result()

    monkeypatch.setenv("TREEHOUSE_LAB_LLM_PROVIDER", "agent_cli")
    monkeypatch.setenv("TREEHOUSE_LAB_AGENT_CLI", "claude")
    monkeypatch.setenv("TREEHOUSE_LAB_LLM_MODEL", "sonnet")
    monkeypatch.setattr(llm.shutil, "which", lambda name: f"/usr/local/bin/{name}")
    monkeypatch.setattr(llm.subprocess, "run", fake_run)

    response = client.post(
        "/api/configs/bank-valid-test/advisor",
        json={"question": "What should I do next?"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "available"
    assert payload["provider"] == "agent_cli:claude"
    assert captured["calls"][0][:5] == ["claude", "-p", "--output-format", "text", "--permission-mode"]
    assert "--system-prompt" in captured["calls"][0]


def test_run_coach_recommendation_executes_bounded_proposal(
    client: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    write_config(tmp_path / "configs" / "datasets")
    captured: dict[str, object] = {}

    class FakeController:
        def __init__(self, config_path: Path) -> None:
            captured["config_path"] = str(config_path)

        def proposal_for_mutation_type(self, mutation_type: str) -> object:
            captured["mutation_type"] = mutation_type
            return SimpleNamespace(
                to_dict=lambda: {
                    "proposal_id": "proposal-123",
                    "dataset_key": "bank-valid-test",
                    "mutation_type": mutation_type,
                    "mutation_name": mutation_type.replace("_", "-"),
                    "params_override": {"scale_pos_weight": 7.547},
                }
            )

        def execute_proposal_step(self, proposal: object, preview_follow_up: bool = True) -> object:
            captured["preview_follow_up"] = preview_follow_up
            captured["proposal"] = proposal.to_dict()
            return SimpleNamespace(
                to_dict=lambda: {
                    "step_index": 0,
                    "proposal": proposal.to_dict(),
                    "result": {
                        "run_id": "20260418T160000000000Z-imbalance-adjustment",
                        "name": "imbalance-adjustment",
                    },
                    "narrative_path": str(tmp_path / "runs" / "narrative.md"),
                }
            )

    monkeypatch.setattr(api, "AutonomousLoopController", FakeController)

    response = client.post(
        "/api/configs/bank-valid-test/coach-recommendation/run",
        json={"mutation_type": "imbalance_adjustment"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["proposal"]["mutation_type"] == "imbalance_adjustment"
    assert payload["result"]["run_id"] == "20260418T160000000000Z-imbalance-adjustment"
    assert captured["mutation_type"] == "imbalance_adjustment"
    assert captured["preview_follow_up"] is True


def test_candidate_endpoint_accepts_feature_generation_payload(
    client: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    write_config(tmp_path / "configs" / "datasets")
    captured: dict[str, object] = {}

    class StubRunner:
        def __init__(self, config_path: Path) -> None:
            captured["config_path"] = str(config_path)

        def run_candidate(
            self,
            mutation_name: str,
            overrides: dict[str, object],
            hypothesis: str | None = None,
            feature_generation: dict[str, object] | None = None,
        ) -> SimpleNamespace:
            captured["mutation_name"] = mutation_name
            captured["overrides"] = overrides
            captured["hypothesis"] = hypothesis
            captured["feature_generation"] = feature_generation
            return SimpleNamespace(
                to_dict=lambda: {
                    "name": mutation_name,
                    "hypothesis": hypothesis,
                    "metadata": {"feature_generation": feature_generation},
                }
            )

    monkeypatch.setattr(api, "TreehouseLabRunner", StubRunner)

    response = client.post(
        "/api/configs/bank-valid-test/candidate",
        json={
            "mutation_name": "feature-generation-enable",
            "hypothesis": "Try a capped numeric interaction branch.",
            "overrides": {},
            "feature_generation": {
                "enabled": True,
                "strategy": "train_only_supervised_numeric_interactions",
                "max_new_features": 6,
            },
        },
    )

    assert response.status_code == 200
    assert captured["mutation_name"] == "feature-generation-enable"
    assert captured["feature_generation"] == {
        "enabled": True,
        "strategy": "train_only_supervised_numeric_interactions",
        "max_new_features": 6,
    }
