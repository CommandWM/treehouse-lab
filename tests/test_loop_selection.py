from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from treehouse_lab.loop import AutonomousLoopController
from treehouse_lab.proposals import ExperimentProposal, ProposalDecisionContext
from treehouse_lab.runner import ExperimentResult


def make_context() -> ProposalDecisionContext:
    return ProposalDecisionContext(
        dataset_key="bank-valid-test",
        task_kind="binary_classification",
        primary_metric="roc_auc",
        promote_threshold=0.003,
        incumbent_run_id="baseline-run",
        incumbent_metric=0.9434,
        incumbent_params={"n_estimators": 300, "learning_rate": 0.05, "max_depth": 6},
        incumbent_metrics={"roc_auc": 0.9434, "validation_roc_auc": 0.9434, "train_roc_auc": 0.9734},
        split_summary={"validation_positive_rate": 0.117},
        overfit_gap=0.0299,
        positive_rate=0.117,
        search_space={"xgboost": {}},
        journal_entries=[
            {
                "name": "learning-rate-tradeoff",
                "promoted": False,
                "comparison_to_incumbent": {"delta": 0.0006},
                "proposal": {"mutation_type": "learning_rate_tradeoff"},
                "diagnosis": {"summary": "Plateauing after repeated learning-rate tradeoffs."},
            }
        ],
        loop_step_index=0,
        executed_mutation_types=["learning_rate_tradeoff"],
        executed_mutation_names=["learning-rate-tradeoff"],
        allow_feature_generation=False,
        diagnosis={"primary_tag": "class_imbalance", "summary": "Positive rate is 0.117 and recent deltas are small."},
    )


def make_proposal(proposal_id: str, mutation_type: str, score: float) -> ExperimentProposal:
    return ExperimentProposal(
        proposal_id=proposal_id,
        dataset_key="bank-valid-test",
        mutation_type=mutation_type,
        mutation_name=mutation_type.replace("_", "-"),
        diagnosis_summary="Positive rate is 0.117 and recent deltas are small.",
        hypothesis="Test the next bounded move.",
        rationale="Bounded rationale.",
        expected_upside="Some upside.",
        risk_level="medium",
        params_override={"dummy": score},
        score=score,
    )


def make_controller() -> AutonomousLoopController:
    controller = object.__new__(AutonomousLoopController)
    controller.project_root = Path("/tmp/treehouse-lab")
    return controller


def test_select_candidate_uses_deterministic_top_when_llm_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    controller = make_controller()
    context = make_context()
    top = make_proposal("proposal-top", "imbalance_adjustment", 2.1)
    second = make_proposal("proposal-second", "learning_rate_tradeoff", 1.4)
    candidates = [SimpleNamespace(proposal=top), SimpleNamespace(proposal=second)]

    monkeypatch.setattr("treehouse_lab.loop.llm_loop_selection_enabled", lambda *args, **kwargs: False)

    selected = controller._select_candidate(context, candidates)

    assert selected.proposal_id == "proposal-top"
    assert selected.llm_review["status"] == "disabled"


def test_select_candidate_uses_llm_selected_candidate(monkeypatch: pytest.MonkeyPatch) -> None:
    controller = make_controller()
    context = make_context()
    top = make_proposal("proposal-top", "imbalance_adjustment", 2.1)
    second = make_proposal("proposal-second", "learning_rate_tradeoff", 1.4)
    candidates = [SimpleNamespace(proposal=top), SimpleNamespace(proposal=second)]

    monkeypatch.setattr("treehouse_lab.loop.llm_loop_selection_enabled", lambda *args, **kwargs: True)
    monkeypatch.setattr(
        "treehouse_lab.loop.select_bounded_proposal",
        lambda context, candidates: SimpleNamespace(
            selected_proposal_id="proposal-second",
            rationale="The plateaued path still looks most attributable.",
            status="available",
            provider="ollama",
            model="gpt-oss:20b",
            message=None,
            raw_output='{"selected_proposal_id":"proposal-second"}',
            candidate_count=2,
            to_dict=lambda: {
                "status": "available",
                "provider": "ollama",
                "model": "gpt-oss:20b",
                "selected_proposal_id": "proposal-second",
                "rationale": "The plateaued path still looks most attributable.",
                "message": None,
                "raw_output": '{"selected_proposal_id":"proposal-second"}',
                "candidate_count": 2,
            },
        ),
    )

    selected = controller._select_candidate(context, candidates)

    assert selected.proposal_id == "proposal-second"
    assert selected.llm_review["provider"] == "ollama"


def test_select_candidate_falls_back_when_llm_selection_is_invalid(monkeypatch: pytest.MonkeyPatch) -> None:
    controller = make_controller()
    context = make_context()
    top = make_proposal("proposal-top", "imbalance_adjustment", 2.1)
    second = make_proposal("proposal-second", "learning_rate_tradeoff", 1.4)
    candidates = [SimpleNamespace(proposal=top), SimpleNamespace(proposal=second)]

    monkeypatch.setattr("treehouse_lab.loop.llm_loop_selection_enabled", lambda *args, **kwargs: True)
    monkeypatch.setattr(
        "treehouse_lab.loop.select_bounded_proposal",
        lambda context, candidates: SimpleNamespace(
            selected_proposal_id=None,
            to_dict=lambda: {
                "status": "error",
                "provider": "ollama",
                "model": "gpt-oss:20b",
                "selected_proposal_id": None,
                "rationale": None,
                "message": "LLM selection did not return a valid candidate proposal_id.",
                "raw_output": "{}",
                "candidate_count": 2,
            },
        ),
    )

    selected = controller._select_candidate(context, candidates)

    assert selected.proposal_id == "proposal-top"
    assert selected.llm_review["status"] == "fallback"
    assert selected.llm_review["fallback_proposal_id"] == "proposal-top"


def test_select_candidate_force_llm_bypasses_disabled_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    controller = make_controller()
    context = make_context()
    top = make_proposal("proposal-top", "imbalance_adjustment", 2.1)
    second = make_proposal("proposal-second", "learning_rate_tradeoff", 1.4)
    candidates = [SimpleNamespace(proposal=top), SimpleNamespace(proposal=second)]

    monkeypatch.setattr("treehouse_lab.loop.llm_loop_selection_enabled", lambda: False)
    monkeypatch.setattr(
        "treehouse_lab.loop.select_bounded_proposal",
        lambda context, candidates: SimpleNamespace(
            selected_proposal_id="proposal-second",
            rationale="The coach explicitly selected the attributable bounded tradeoff.",
            status="available",
            provider="ollama",
            model="gpt-oss:20b",
            message=None,
            raw_output='{"selected_proposal_id":"proposal-second"}',
            candidate_count=2,
            to_dict=lambda: {
                "status": "available",
                "provider": "ollama",
                "model": "gpt-oss:20b",
                "selected_proposal_id": "proposal-second",
                "rationale": "The coach explicitly selected the attributable bounded tradeoff.",
                "message": None,
                "raw_output": '{"selected_proposal_id":"proposal-second"}',
                "candidate_count": 2,
            },
        ),
    )

    selected = controller._select_candidate(context, candidates, force_llm=True)

    assert selected.proposal_id == "proposal-second"
    assert selected.llm_review["status"] == "available"


def test_plateaued_loop_can_select_and_execute_feature_generation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_dir = tmp_path / "configs" / "datasets"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "feature-loop.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "dataset": {
                    "source": {
                        "kind": "csv",
                        "name": "Feature Loop Fixture",
                        "target_column": "converted",
                        "path": "feature_loop.csv",
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
                    "objective": "Prove a plateau can escalate to bounded feature generation.",
                },
                "evaluation_policy": {
                    "require_promotion_for_readiness": True,
                },
                "experiment": {
                    "name": "feature-loop",
                    "description": "Loop test fixture.",
                    "primary_metric": "roc_auc",
                    "promote_if_delta_at_least": 0.003,
                    "max_runtime_minutes": 10,
                    "seed": 42,
                    "baseline_hypothesis": "A baseline should exist before the plateaued feature branch.",
                },
                "model": {
                    "params": {
                        "n_estimators": 300,
                        "max_depth": 4,
                        "learning_rate": 0.05,
                        "min_child_weight": 1,
                        "subsample": 0.9,
                        "colsample_bytree": 0.8,
                    }
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    search_space_path = tmp_path / "configs" / "search_space.yaml"
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
                "feature_generation": {
                    "max_new_features": 4,
                    "top_k_numeric": 3,
                    "operations": ["square", "product"],
                    "tools": ["openfe"],
                },
                "policy": {
                    "allow_feature_generation": True,
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    runs_dir = tmp_path / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    incumbent_dir = runs_dir / "incumbents"
    incumbent_dir.mkdir(parents=True, exist_ok=True)

    baseline_entry = {
        "run_id": "baseline-run",
        "registry_key": "feature-loop",
        "name": "baseline",
        "metric": 0.91,
        "promoted": True,
        "artifact_dir": str(runs_dir / "baseline-run"),
        "config_path": str(config_path),
        "params": {
            "n_estimators": 300,
            "max_depth": 4,
            "learning_rate": 0.05,
            "min_child_weight": 1,
            "subsample": 0.9,
            "colsample_bytree": 0.8,
        },
        "metrics": {
            "roc_auc": 0.91,
            "train_roc_auc": 0.92,
            "validation_roc_auc": 0.91,
            "test_roc_auc": 0.905,
        },
        "split_summary": {
            "feature_count": 8,
            "raw_numeric_feature_count": 4,
            "validation_positive_rate": 0.5,
        },
        "assessment": {
            "benchmark_status": "baseline_established",
            "implementation_readiness": "implementation_ready",
        },
        "diagnosis": {
            "primary_tag": "healthy",
            "summary": "Baseline established.",
        },
        "reason_codes": ["baseline_established"],
        "proposal": {
            "mutation_type": "baseline",
            "mutation_name": "baseline",
        },
    }
    rejected_entries = [
        {
            "run_id": "reject-1",
            "registry_key": "feature-loop",
            "name": "regularization-tighten",
            "metric": 0.911,
            "promoted": False,
            "artifact_dir": str(runs_dir / "reject-1"),
            "config_path": str(config_path),
            "comparison_to_incumbent": {"incumbent_metric": 0.91, "delta": 0.001, "threshold": 0.003},
            "proposal": {
                "mutation_type": "regularization_tighten",
                "mutation_name": "regularization-tighten",
                "params_override": {"max_depth": 3},
            },
        },
        {
            "run_id": "reject-2",
            "registry_key": "feature-loop",
            "name": "learning-rate-tradeoff",
            "metric": 0.9108,
            "promoted": False,
            "artifact_dir": str(runs_dir / "reject-2"),
            "config_path": str(config_path),
            "comparison_to_incumbent": {"incumbent_metric": 0.91, "delta": 0.0008, "threshold": 0.003},
            "proposal": {
                "mutation_type": "learning_rate_tradeoff",
                "mutation_name": "learning-rate-tradeoff",
                "params_override": {"learning_rate": 0.04, "n_estimators": 405},
            },
        },
    ]

    journal_path = runs_dir / "journal.jsonl"
    with journal_path.open("w", encoding="utf-8") as handle:
        for entry in [baseline_entry, *rejected_entries]:
            handle.write(json.dumps(entry, sort_keys=True) + "\n")
    incumbent_dir.joinpath("feature-loop.json").write_text(
        json.dumps(
            {
                "run_id": "baseline-run",
                "name": "baseline",
                "metric": 0.91,
                "artifact_dir": str(runs_dir / "baseline-run"),
                "config_path": str(config_path),
                "registry_key": "feature-loop",
                "params": baseline_entry["params"],
                "metrics": baseline_entry["metrics"],
                "assessment": baseline_entry["assessment"],
                "diagnosis": baseline_entry["diagnosis"],
                "reason_codes": baseline_entry["reason_codes"],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    controller = AutonomousLoopController(config_path)
    proposal = controller.choose_next_proposal(loop_step_index=0, loop_history=[])

    assert proposal is not None
    assert proposal.mutation_type == "feature_generation_enable"
    assert proposal.feature_generation["enabled"] is True

    def fake_run_proposal(selected_proposal: ExperimentProposal) -> ExperimentResult:
        artifact_dir = runs_dir / "feature-run-123"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        result = ExperimentResult(
            name=selected_proposal.mutation_name,
            backend="xgboost",
            metric=0.916,
            promoted=True,
            notes="Feature branch cleared the bar.",
            run_id="feature-run-123",
            artifact_dir=str(artifact_dir),
            config_path=str(config_path),
            hypothesis=selected_proposal.hypothesis,
            decision_reason="Feature branch improved validation metric.",
            runtime_seconds=0.5,
            params=baseline_entry["params"],
            metrics={
                "roc_auc": 0.916,
                "train_roc_auc": 0.924,
                "validation_roc_auc": 0.916,
                "test_roc_auc": 0.912,
            },
            split_summary={
                "feature_count": 12,
                "generated_feature_count": 4,
                "raw_numeric_feature_count": 4,
                "validation_positive_rate": 0.5,
            },
            comparison_to_incumbent={"incumbent_metric": 0.91, "delta": 0.006, "threshold": 0.003},
            assessment={
                "benchmark_status": "benchmark_better",
                "benchmark_summary": "Validation improved.",
                "implementation_readiness": "implementation_ready",
                "checks": [],
            },
            diagnosis={
                "primary_tag": "healthy",
                "summary": "Feature branch improved validation metric.",
                "recommended_direction": "Keep changes attributable.",
                "preferred_mutations": [],
                "avoided_mutations": [],
            },
            reason_codes=["promoted_metric_gain"],
            feature_generation={
                "enabled": True,
                "plan": selected_proposal.feature_generation,
                "applied": True,
                "generated_feature_count": 4,
                "generated_feature_specs": [
                    {"name": "fg__square__visits", "operation": "square", "columns": ["visits"]},
                ],
            },
            metadata={
                "dataset_key": "feature-loop",
                "proposal": selected_proposal.to_dict(),
                "feature_generation_summary": {
                    "enabled": True,
                    "applied": True,
                    "generated_feature_count": 4,
                    "generated_feature_specs": [
                        {"name": "fg__square__visits", "operation": "square", "columns": ["visits"]},
                    ],
                },
            },
        )
        with journal_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(result.to_dict(), sort_keys=True) + "\n")
        return result

    monkeypatch.setattr(controller, "run_proposal", fake_run_proposal)

    step = controller.execute_proposal_step(proposal, preview_follow_up=False)

    assert step.proposal["mutation_type"] == "feature_generation_enable"
    assert step.result["feature_generation"]["enabled"] is True
    recorded_entry = json.loads(journal_path.read_text(encoding="utf-8").splitlines()[-1])
    assert recorded_entry["proposal"]["mutation_type"] == "feature_generation_enable"
    assert recorded_entry["feature_generation"]["generated_feature_count"] == 4
