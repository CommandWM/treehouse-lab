from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from treehouse_lab.loop import AutonomousLoopController
from treehouse_lab.proposals import ExperimentProposal, ProposalDecisionContext


def make_context() -> ProposalDecisionContext:
    return ProposalDecisionContext(
        dataset_key="bank-valid-test",
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
