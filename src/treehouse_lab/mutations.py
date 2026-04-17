from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from treehouse_lab.proposals import ExperimentProposal, ProposalDecisionContext, build_mutation_proposal


@dataclass(slots=True)
class MutationTemplate:
    name: str
    stage: str
    description: str
    base_score: float
    risk_level: str


@dataclass(slots=True)
class MutationCandidate:
    template: MutationTemplate
    proposal: ExperimentProposal


def list_templates(stage: str = "parameter_tuning") -> list[MutationTemplate]:
    templates = [
        MutationTemplate(
            name="regularization_tighten",
            stage="parameter_tuning",
            description="Reduce model capacity to address validation underperformance caused by overfitting.",
            base_score=1.0,
            risk_level="low",
        ),
        MutationTemplate(
            name="learning_rate_tradeoff",
            stage="parameter_tuning",
            description="Trade a slower learning rate for more estimators to improve stability.",
            base_score=0.9,
            risk_level="medium",
        ),
        MutationTemplate(
            name="capacity_increase",
            stage="parameter_tuning",
            description="Increase model capacity when the incumbent appears underfit.",
            base_score=0.8,
            risk_level="medium",
        ),
        MutationTemplate(
            name="imbalance_adjustment",
            stage="parameter_tuning",
            description="Adjust positive-class weighting when the target distribution is skewed.",
            base_score=0.7,
            risk_level="medium",
        ),
    ]
    return [template for template in templates if template.stage == stage]


def generate_candidates(context: ProposalDecisionContext) -> list[MutationCandidate]:
    candidates: list[MutationCandidate] = []
    for template in list_templates():
        proposal = _proposal_for_template(template, context)
        if proposal is not None:
            candidates.append(MutationCandidate(template=template, proposal=proposal))
    return sorted(candidates, key=lambda candidate: candidate.proposal.score, reverse=True)


def apply_template(template: MutationTemplate, incumbent_params: dict[str, Any], search_space: dict[str, Any], positive_rate: float) -> dict[str, Any]:
    if template.name == "regularization_tighten":
        return {
            "max_depth": _bounded_int(int(incumbent_params["max_depth"]) - 1, search_space["max_depth"]),
            "min_child_weight": _bounded_int(int(incumbent_params["min_child_weight"]) + 2, search_space["min_child_weight"]),
            "subsample": _bounded_float(float(incumbent_params["subsample"]) - 0.10, search_space["subsample"]),
            "colsample_bytree": _bounded_float(
                float(incumbent_params["colsample_bytree"]) - 0.10,
                search_space["colsample_bytree"],
            ),
        }
    if template.name == "learning_rate_tradeoff":
        return {
            "learning_rate": _bounded_float(float(incumbent_params["learning_rate"]) * 0.8, search_space["learning_rate"]),
            "n_estimators": _bounded_int(int(round(int(incumbent_params["n_estimators"]) * 1.35)), search_space["n_estimators"]),
        }
    if template.name == "capacity_increase":
        return {
            "max_depth": _bounded_int(int(incumbent_params["max_depth"]) + 1, search_space["max_depth"]),
            "n_estimators": _bounded_int(int(round(int(incumbent_params["n_estimators"]) * 1.2)), search_space["n_estimators"]),
            "min_child_weight": _bounded_int(int(incumbent_params["min_child_weight"]) - 1, search_space["min_child_weight"]),
        }
    if template.name == "imbalance_adjustment":
        positive_rate = min(max(positive_rate, 0.01), 0.99)
        scale_pos_weight = round((1 - positive_rate) / positive_rate, 3)
        return {"scale_pos_weight": scale_pos_weight}
    msg = f"Unsupported template: {template.name}"
    raise ValueError(msg)


def _proposal_for_template(template: MutationTemplate, context: ProposalDecisionContext) -> ExperimentProposal | None:
    params_override = apply_template(template, context.incumbent_params, context.search_space["xgboost"], context.positive_rate)
    if not _is_meaningful_change(context.incumbent_params, params_override):
        return None

    score = _score_template(template, context)
    mutation_name = template.name.replace("_", "-")
    return build_mutation_proposal(
        context=context,
        mutation_type=template.name,
        mutation_name=mutation_name,
        hypothesis=_hypothesis_for_template(template),
        rationale=_rationale_for_template(template, context),
        expected_upside=_expected_upside_for_template(template),
        risk_level=template.risk_level,
        params_override=params_override,
        score=score,
    )


def _score_template(template: MutationTemplate, context: ProposalDecisionContext) -> float:
    score = template.base_score
    overfit_gap = context.overfit_gap
    positive_rate_delta = abs(0.5 - context.positive_rate)

    if template.name == "regularization_tighten":
        score += 0.7 if overfit_gap > 0.03 else -0.2
    if template.name == "learning_rate_tradeoff":
        score += 0.5 if overfit_gap > 0.01 else 0.2
    if template.name == "capacity_increase":
        score += 0.6 if overfit_gap < 0.015 else -0.3
    if template.name == "imbalance_adjustment":
        score += 0.6 if positive_rate_delta > 0.15 else -0.8

    if context.loop_step_index == 0 and template.name == "regularization_tighten":
        score += 1.2
    if context.loop_step_index == 1 and template.name == "learning_rate_tradeoff":
        score += 1.0
    if context.loop_step_index >= 2 and template.name in {"capacity_increase", "imbalance_adjustment"}:
        score += 0.9

    if template.name in context.executed_mutation_types:
        score -= 1.3
    return score


def _hypothesis_for_template(template: MutationTemplate) -> str:
    hypotheses = {
        "regularization_tighten": "A lower-capacity ensemble should reduce overfitting and improve validation performance.",
        "learning_rate_tradeoff": "A slower learner with more boosting rounds may improve validation stability without blowing up runtime.",
        "capacity_increase": "A slightly larger ensemble may recover signal if the incumbent is underfit.",
        "imbalance_adjustment": "Explicit positive-class weighting may improve ranking quality on skewed targets.",
    }
    return hypotheses[template.name]


def _rationale_for_template(template: MutationTemplate, context: ProposalDecisionContext) -> str:
    if template.name == "regularization_tighten":
        return (
            f"The incumbent shows a train-validation ROC AUC gap of {context.overfit_gap:.4f}, "
            "so the next bounded move should reduce tree capacity before trying more complex changes."
        )
    if template.name == "learning_rate_tradeoff":
        return (
            "The incumbent is already reasonably strong, so the next attributable move is to trade a smaller learning rate "
            "for more trees and test whether the gain is more stable."
        )
    if template.name == "capacity_increase":
        return "The incumbent does not look strongly overfit, so a modest capacity increase is justified before feature-generation work."
    return (
        f"The positive-class rate is {context.positive_rate:.3f}, which is far enough from parity to justify "
        "a bounded class-balance adjustment."
    )


def _expected_upside_for_template(template: MutationTemplate) -> str:
    outcomes = {
        "regularization_tighten": "Better validation performance with a simpler, more defensible tree ensemble.",
        "learning_rate_tradeoff": "A modest lift from smoother boosting dynamics.",
        "capacity_increase": "Recovered signal if the current incumbent is leaving performance on the table.",
        "imbalance_adjustment": "Improved ranking on the positive class without changing the dataset split policy.",
    }
    return outcomes[template.name]


def _is_meaningful_change(incumbent_params: dict[str, Any], overrides: dict[str, Any]) -> bool:
    for key, value in overrides.items():
        if incumbent_params.get(key) != value:
            return True
    return False


def _bounded_int(value: int, bounds: list[Any]) -> int:
    return int(min(max(value, int(bounds[0])), int(bounds[1])))


def _bounded_float(value: float, bounds: list[Any]) -> float:
    bounded = min(max(value, float(bounds[0])), float(bounds[1]))
    return round(float(bounded), 3)
