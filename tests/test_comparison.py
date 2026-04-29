from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

from treehouse_lab.comparison import run_comparison_suite
from treehouse_lab.llm import ComparisonSummaryResponse


def write_fixture_project(tmp_path: Path) -> Path:
    project_root = tmp_path
    config_dir = project_root / "configs" / "datasets"
    config_dir.mkdir(parents=True, exist_ok=True)
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
                "policy": {"allow_feature_generation": True},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    dataset_dir = project_root / "custom_datasets"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = dataset_dir / "compare_bankish.csv"
    frame = pd.DataFrame(
        {
            "age": [25, 31, 42, 36, 29, 55, 47, 52, 38, 41] * 12,
            "job": ["admin", "tech", "mgmt", "admin", "services", "retired", "mgmt", "blue", "tech", "admin"] * 12,
            "balance": [1200, 800, 4200, 1600, 950, 5100, 3300, 2600, 1800, 2100] * 12,
            "housing": ["yes", "yes", "no", "yes", "yes", "no", "no", "yes", "no", "yes"] * 12,
            "duration": [110, 90, 310, 220, 130, 420, 390, 280, 205, 170] * 12,
            "campaign": [1, 2, 1, 3, 2, 1, 1, 2, 2, 3] * 12,
            "y": [0, 0, 1, 0, 0, 1, 1, 1, 0, 0] * 12,
        }
    )
    frame.to_csv(dataset_path, index=False)

    config_path = config_dir / "compare_bankish.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "dataset": {
                    "source": {
                        "kind": "csv",
                        "name": "Compare Bankish",
                        "path": "custom_datasets/compare_bankish.csv",
                        "target_column": "y",
                    },
                    "split": {"validation_size": 0.2, "test_size": 0.2, "stratify": True},
                },
                "benchmark": {
                    "pack": "optional",
                    "profile": "external_probe",
                    "objective": "Exercise the side-by-side comparison harness on a business-style binary classification problem.",
                },
                "evaluation_policy": {
                    "minimum_primary_metric": 0.75,
                    "max_train_validation_gap": 0.08,
                    "max_validation_test_gap": 0.08,
                    "max_runtime_seconds": 30,
                    "max_feature_count": 32,
                    "require_promotion_for_readiness": True,
                },
                "experiment": {
                    "name": "compare-bankish-baseline",
                    "description": "Fixture dataset for comparison harness tests.",
                    "primary_metric": "roc_auc",
                    "promote_if_delta_at_least": 0.002,
                    "max_runtime_minutes": 5,
                    "seed": 42,
                    "baseline_hypothesis": "A disciplined baseline should be reproducible across comparison runners.",
                },
                "model": {
                    "params": {
                        "n_estimators": 180,
                        "max_depth": 4,
                        "learning_rate": 0.06,
                        "min_child_weight": 1,
                        "subsample": 0.9,
                        "colsample_bytree": 0.8,
                    }
                },
                "task": {"kind": "binary_classification"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return config_path


def test_resolve_autogluon_practical_profile_defaults(tmp_path: Path) -> None:
    config_path = write_fixture_project(tmp_path)

    import treehouse_lab.comparison as comparison_module
    from treehouse_lab.runner import TreehouseLabRunner

    runner = TreehouseLabRunner(config_path)
    settings = comparison_module._resolve_autogluon_runner_config(
        runner.config,
        profile="practical",
        presets=None,
        time_limit=None,
    )

    assert settings.profile == "practical"
    assert settings.presets == ["medium_quality", "optimize_for_deployment"]
    assert settings.time_limit == 30
    assert "NN_TORCH" in settings.excluded_model_types
    assert settings.display_name == "AutoGluon Tabular (Practical)"


def test_summarize_loop_llm_guidance_counts_available_reviews() -> None:
    import treehouse_lab.comparison as comparison_module

    guidance = comparison_module._summarize_loop_llm_guidance(
        [
            {"proposal": {"llm_review": {"status": "disabled"}}},
            {
                "proposal": {
                    "llm_review": {
                        "status": "available",
                        "provider": "agent_cli:codex",
                    }
                }
            },
            {
                "proposal": {
                    "llm_review": {
                        "status": "available",
                        "provider": "agent_cli:codex",
                    }
                }
            },
        ]
    )

    assert guidance["llm_guided_step_count"] == 2
    assert guidance["llm_reviewed_step_count"] == 3
    assert guidance["llm_provider"] == "agent_cli:codex"
    assert guidance["llm_guidance_statuses"] == ["available", "disabled"]


def test_build_autogluon_fit_kwargs_uses_holdout_for_bagged_presets() -> None:
    import treehouse_lab.comparison as comparison_module

    config = comparison_module.AutoGluonRunnerConfig(
        profile="practical",
        presets=["medium_quality", "optimize_for_deployment"],
        time_limit=30,
        excluded_model_types=["NN_TORCH"],
    )
    frame = pd.DataFrame({"x": [1, 2], "y": [0, 1]})

    fit_kwargs = comparison_module._build_autogluon_fit_kwargs(frame, frame, config)

    assert fit_kwargs["train_data"].equals(frame)
    assert fit_kwargs["tuning_data"].equals(frame)
    assert fit_kwargs["use_bag_holdout"] is True
    assert fit_kwargs["excluded_model_types"] == ["NN_TORCH"]


def test_run_comparison_suite_without_autogluon(tmp_path: Path) -> None:
    config_path = write_fixture_project(tmp_path)

    result = run_comparison_suite(
        config_path,
        output_dir=tmp_path / "outputs" / "compare",
        include_autogluon=False,
        loop_steps=1,
    )

    assert Path(result.report_path).exists()
    assert Path(result.summary_path).exists()

    runners = {runner["runner_key"]: runner for runner in result.runners}
    assert set(runners) == {"plain_xgboost", "treehouse_lab_baseline", "treehouse_lab_loop"}
    assert all(runner["status"] == "completed" for runner in runners.values())
    assert abs(runners["plain_xgboost"]["validation_metric"] - runners["treehouse_lab_baseline"]["validation_metric"]) < 1e-9

    report_text = Path(result.report_path).read_text(encoding="utf-8")
    assert "## Outcome gates" in report_text
    assert "## Workflow traits" in report_text
    assert "Treehouse Lab 1-Step Loop" in report_text


def test_run_comparison_suite_marks_autogluon_unavailable(tmp_path: Path, monkeypatch) -> None:
    config_path = write_fixture_project(tmp_path)

    import treehouse_lab.comparison as comparison_module

    original_import_module = comparison_module.importlib.import_module

    def fake_import_module(name: str):
        if name == "autogluon.tabular":
            raise ModuleNotFoundError(name)
        return original_import_module(name)

    monkeypatch.setattr(comparison_module.importlib, "import_module", fake_import_module)

    result = run_comparison_suite(
        config_path,
        output_dir=tmp_path / "outputs" / "compare_with_ag",
        include_autogluon=True,
        loop_steps=1,
    )

    runners = {runner["runner_key"]: runner for runner in result.runners}
    assert runners["autogluon_tabular"]["status"] == "unavailable"
    summary_payload = json.loads(Path(result.summary_path).read_text(encoding="utf-8"))
    assert any(runner["runner_key"] == "autogluon_tabular" for runner in summary_payload["runners"])


def test_run_comparison_suite_marks_autogluon_error_without_crashing(tmp_path: Path, monkeypatch) -> None:
    config_path = write_fixture_project(tmp_path)

    import treehouse_lab.comparison as comparison_module

    class BrokenPredictor:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def fit(self, **kwargs):
            raise RuntimeError("synthetic autogluon failure")

    class FakeModule:
        TabularPredictor = BrokenPredictor

    monkeypatch.setattr(comparison_module.importlib, "import_module", lambda name: FakeModule())

    result = run_comparison_suite(
        config_path,
        output_dir=tmp_path / "outputs" / "compare_with_ag_error",
        include_autogluon=True,
        loop_steps=1,
    )

    runners = {runner["runner_key"]: runner for runner in result.runners}
    assert runners["autogluon_tabular"]["status"] == "error"
    assert "synthetic autogluon failure" in runners["autogluon_tabular"]["notes"][-1]


def test_run_comparison_suite_includes_llm_summary_when_requested(tmp_path: Path, monkeypatch) -> None:
    config_path = write_fixture_project(tmp_path)

    import treehouse_lab.comparison as comparison_module

    captured: dict[str, object] = {}

    def fake_generate_comparison_summary(context: dict[str, object], question: str | None = None) -> ComparisonSummaryResponse:
        captured["context"] = context
        captured["question"] = question
        return ComparisonSummaryResponse(
            status="available",
            provider="agent_cli:codex",
            model="gpt-5.4-mini",
            question=question or "default",
            answer=(
                "Current state:\nMetrics are effectively tied.\n\n"
                "Product value:\nTreehouse Lab adds the bounded journal and proposal trail.\n\n"
                "Next step:\nRun another public dataset through the same harness.\n\n"
                "Watchouts:\nDo not oversell a metric tie as a model win."
            ),
        )

    monkeypatch.setattr(comparison_module, "generate_comparison_summary", fake_generate_comparison_summary)

    result = run_comparison_suite(
        config_path,
        output_dir=tmp_path / "outputs" / "compare_with_llm",
        include_autogluon=False,
        include_llm_summary=True,
        llm_question="Where is the real product value here?",
        loop_steps=1,
    )

    assert result.llm_summary is not None
    assert result.llm_summary["status"] == "available"
    assert captured["question"] == "Where is the real product value here?"
    assert captured["context"]["dataset_key"] == "compare_bankish"

    report_text = Path(result.report_path).read_text(encoding="utf-8")
    assert "## LLM synthesis" in report_text
    assert "Product value:" in report_text

    summary_payload = json.loads(Path(result.summary_path).read_text(encoding="utf-8"))
    assert summary_payload["llm_summary"]["provider"] == "agent_cli:codex"


def test_run_comparison_suite_records_unavailable_llm_summary(tmp_path: Path, monkeypatch) -> None:
    config_path = write_fixture_project(tmp_path)

    import treehouse_lab.comparison as comparison_module

    monkeypatch.setattr(
        comparison_module,
        "generate_comparison_summary",
        lambda context, question=None: ComparisonSummaryResponse(
            status="unavailable",
            provider="ollama",
            model="gemma3:4b",
            question=question or "default",
            message="No LLM provider is configured for comparison synthesis.",
        ),
    )

    result = run_comparison_suite(
        config_path,
        output_dir=tmp_path / "outputs" / "compare_with_unavailable_llm",
        include_autogluon=False,
        include_llm_summary=True,
        loop_steps=1,
    )

    assert result.llm_summary is not None
    assert result.llm_summary["status"] == "unavailable"
    report_text = Path(result.report_path).read_text(encoding="utf-8")
    assert "No LLM provider is configured for comparison synthesis." in report_text


def test_render_report_surfaces_feature_generation_decisions() -> None:
    import treehouse_lab.comparison as comparison_module

    class FakeSource:
        name = "Fixture Dataset"

    class FakeConfig:
        source = FakeSource()
        primary_metric = "roc_auc"

    class FakeRunner:
        registry_key = "fixture_dataset"
        config_path = Path("configs/datasets/fixture_dataset.yaml")
        config = FakeConfig()

    class FakeDataset:
        target_name = "target"
        target_profile = {"task_kind": "binary_classification"}

    class FakeSplit:
        def summary(self) -> dict[str, object]:
            return {"train_rows": 80, "validation_rows": 20, "test_rows": 20}

    treehouse_loop = comparison_module.ComparisonRunSummary(
        runner_key="treehouse_lab_loop",
        display_name="Treehouse Lab 1-Step Loop",
        status="completed",
        backend="xgboost",
        validation_metric=0.842,
        test_metric=0.831,
        runtime_seconds=2.4,
        benchmark_status="not_better_than_incumbent",
        implementation_readiness="needs_more_work",
        artifact_path="runs/loops/example",
        workflow_traits={
            "search_style": "bounded_loop",
            "artifact_trail": "full_plus_loop_summary",
            "journal": "yes",
            "bounded_next_step": "yes",
            "llm_guidance": "no",
        },
        details={
            "steps": [
                {
                    "proposal": {
                        "feature_generation": {
                            "enabled": True,
                            "reason": "Recent bounded parameter moves plateaued.",
                            "max_new_features": 4,
                        }
                    },
                    "result": {
                        "promoted": False,
                        "comparison_to_incumbent": {"delta": -0.0004},
                        "assessment": {
                            "benchmark_status": "not_better_than_incumbent",
                            "implementation_readiness": "needs_more_work",
                        },
                        "feature_generation": {
                            "enabled": True,
                            "applied": True,
                            "generated_feature_count": 2,
                            "generated_feature_specs": [
                                {"name": "fg__square__age", "operation": "square", "columns": ["age"]},
                                {"name": "fg__product__age__balance", "operation": "product", "columns": ["age", "balance"]},
                            ],
                        },
                    },
                }
            ]
        },
    )

    report = comparison_module._render_report(
        base_runner=FakeRunner(),
        dataset=FakeDataset(),
        split=FakeSplit(),
        run_summaries=[treehouse_loop],
        loop_steps=1,
        llm_summary=None,
    )

    assert "## Feature-generation decisions" in report
    assert "| Treehouse Lab 1-Step Loop | yes | yes | yes | 2 | not_better_than_incumbent / needs_more_work | Added bounded features, but outcome gates did not justify the added complexity. |" in report
    assert "`fg__square__age` via `square` on `age`" in report
    assert "Recent bounded parameter moves plateaued." in report
