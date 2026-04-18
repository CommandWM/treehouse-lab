from __future__ import annotations

from treehouse_lab.mutations import generate_candidates
from treehouse_lab.proposals import ProposalDecisionContext


def test_generate_candidates_penalizes_repeated_rejected_templates() -> None:
    context = ProposalDecisionContext(
        dataset_key="bank-valid-test",
        task_kind="binary_classification",
        primary_metric="roc_auc",
        promote_threshold=0.003,
        incumbent_run_id="baseline-run",
        incumbent_metric=0.9434,
        incumbent_params={
            "n_estimators": 300,
            "max_depth": 6,
            "learning_rate": 0.05,
            "min_child_weight": 1,
            "subsample": 0.9,
            "colsample_bytree": 0.8,
        },
        incumbent_metrics={
            "roc_auc": 0.9434,
            "train_roc_auc": 0.9734,
            "validation_roc_auc": 0.9434,
            "test_roc_auc": 0.9343,
        },
        split_summary={"validation_positive_rate": 0.117},
        overfit_gap=0.0299,
        positive_rate=0.117,
        search_space={
            "xgboost": {
                "max_depth": [2, 10],
                "min_child_weight": [1, 10],
                "subsample": [0.5, 1.0],
                "colsample_bytree": [0.5, 1.0],
                "learning_rate": [0.01, 0.3],
                "n_estimators": [100, 600],
            }
        },
        journal_entries=[
            {
                "name": "learning-rate-tradeoff",
                "promoted": False,
                "comparison_to_incumbent": {"delta": 0.0006},
                "proposal": {"mutation_type": "learning_rate_tradeoff"},
            },
            {
                "name": "learning-rate-tradeoff",
                "promoted": False,
                "comparison_to_incumbent": {"delta": 0.0006},
                "proposal": {"mutation_type": "learning_rate_tradeoff"},
            },
            {
                "name": "learning-rate-tradeoff",
                "promoted": False,
                "comparison_to_incumbent": {"delta": 0.0006},
                "proposal": {"mutation_type": "learning_rate_tradeoff"},
            },
        ],
        loop_step_index=0,
        executed_mutation_types=["learning_rate_tradeoff"] * 3,
        executed_mutation_names=["learning-rate-tradeoff"] * 3,
        allow_feature_generation=False,
        diagnosis={
            "tags": ["class_imbalance", "plateau"],
            "preferred_mutations": ["imbalance_adjustment", "learning_rate_tradeoff"],
            "avoided_mutations": [],
            "summary": "Positive rate is 0.1170 and recent deltas are small.",
        },
    )

    candidates = generate_candidates(context)

    assert candidates[0].proposal.mutation_type == "imbalance_adjustment"
    learning_rate_candidate = next(
        candidate for candidate in candidates if candidate.proposal.mutation_type == "learning_rate_tradeoff"
    )
    assert learning_rate_candidate.proposal.score < candidates[0].proposal.score


def test_generate_candidates_skips_binary_only_imbalance_template_for_multiclass() -> None:
    context = ProposalDecisionContext(
        dataset_key="developer-burnout",
        task_kind="multiclass_classification",
        primary_metric="accuracy",
        promote_threshold=0.003,
        incumbent_run_id="baseline-run",
        incumbent_metric=0.71,
        incumbent_params={
            "n_estimators": 300,
            "max_depth": 6,
            "learning_rate": 0.05,
            "min_child_weight": 1,
            "subsample": 0.9,
            "colsample_bytree": 0.8,
        },
        incumbent_metrics={
            "accuracy": 0.71,
            "train_accuracy": 0.83,
            "validation_accuracy": 0.71,
            "test_accuracy": 0.7,
        },
        split_summary={"class_count": 3},
        overfit_gap=0.12,
        positive_rate=0.5,
        search_space={
            "xgboost": {
                "max_depth": [2, 10],
                "min_child_weight": [1, 10],
                "subsample": [0.5, 1.0],
                "colsample_bytree": [0.5, 1.0],
                "learning_rate": [0.01, 0.3],
                "n_estimators": [100, 600],
            }
        },
        journal_entries=[],
        loop_step_index=0,
        executed_mutation_types=[],
        executed_mutation_names=[],
        allow_feature_generation=False,
        diagnosis={
            "tags": ["overfit"],
            "preferred_mutations": ["regularization_tighten"],
            "avoided_mutations": [],
            "summary": "Validation accuracy is lagging training accuracy.",
        },
    )

    candidates = generate_candidates(context)

    assert all(candidate.proposal.mutation_type != "imbalance_adjustment" for candidate in candidates)
