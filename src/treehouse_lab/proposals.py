from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any
from uuid import uuid4


@dataclass(slots=True)
class ExperimentProposal:
    proposal_id: str
    dataset_key: str
    mutation_type: str
    mutation_name: str
    hypothesis: str
    rationale: str
    expected_upside: str
    risk_level: str
    base_params: dict[str, Any] = field(default_factory=dict)
    params_override: dict[str, Any] = field(default_factory=dict)
    feature_generation: dict[str, Any] = field(default_factory=dict)
    depends_on_run_id: str | None = None
    stage: str = "parameter_tuning"
    loop_step_index: int = 0
    score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ProposalDecisionContext:
    dataset_key: str
    primary_metric: str
    promote_threshold: float
    incumbent_run_id: str | None
    incumbent_metric: float | None
    incumbent_params: dict[str, Any]
    incumbent_metrics: dict[str, float]
    split_summary: dict[str, Any]
    overfit_gap: float
    positive_rate: float
    search_space: dict[str, Any]
    journal_entries: list[dict[str, Any]]
    loop_step_index: int
    executed_mutation_types: list[str]
    executed_mutation_names: list[str]
    allow_feature_generation: bool


def build_baseline_proposal(dataset_key: str, hypothesis: str) -> ExperimentProposal:
    return ExperimentProposal(
        proposal_id=_proposal_id(),
        dataset_key=dataset_key,
        mutation_type="baseline",
        mutation_name="baseline",
        hypothesis=hypothesis,
        rationale="No incumbent exists yet, so Treehouse Lab must establish a strong baseline before proposing bounded mutations.",
        expected_upside="Creates the first auditable incumbent and defines the benchmark every later mutation must beat.",
        risk_level="low",
        stage="baseline",
    )


def build_mutation_proposal(
    context: ProposalDecisionContext,
    mutation_type: str,
    mutation_name: str,
    hypothesis: str,
    rationale: str,
    expected_upside: str,
    risk_level: str,
    params_override: dict[str, Any],
    score: float,
) -> ExperimentProposal:
    return ExperimentProposal(
        proposal_id=_proposal_id(),
        dataset_key=context.dataset_key,
        mutation_type=mutation_type,
        mutation_name=mutation_name,
        hypothesis=hypothesis,
        rationale=rationale,
        expected_upside=expected_upside,
        risk_level=risk_level,
        base_params=dict(context.incumbent_params),
        params_override=params_override,
        depends_on_run_id=context.incumbent_run_id,
        stage="parameter_tuning",
        loop_step_index=context.loop_step_index,
        score=score,
    )


def proposal_to_dict(proposal: ExperimentProposal) -> dict[str, Any]:
    return proposal.to_dict()


def _proposal_id() -> str:
    return uuid4().hex[:12]
