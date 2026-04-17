from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from treehouse_lab.proposals import ExperimentProposal
from treehouse_lab.runner import ExperimentResult


@dataclass(slots=True)
class RunNarrative:
    title: str
    summary: str
    decision: str
    markdown: str
    recommended_next_step: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class LoopNarrative:
    title: str
    markdown: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_run_narrative(
    proposal: ExperimentProposal,
    result: ExperimentResult,
    incumbent_before: dict[str, Any] | None,
    recommended_next_step: str | None = None,
) -> RunNarrative:
    delta = result.comparison_to_incumbent.get("delta")
    delta_text = "n/a" if delta is None else f"{delta:.4f}"
    decision = "promote" if result.promoted else "reject"
    incumbent_metric = incumbent_before.get("metric") if incumbent_before else None
    incumbent_metric_text = "n/a" if incumbent_metric is None else f"{float(incumbent_metric):.4f}"
    assessment = result.assessment

    markdown = "\n".join(
        [
            f"# {proposal.mutation_name}",
            "",
            "## Hypothesis",
            "",
            proposal.hypothesis,
            "",
            "## Why this experiment",
            "",
            proposal.rationale,
            "",
            "## Exact mutation",
            "",
            *(f"- `{key}`: `{value}`" for key, value in proposal.params_override.items()),
            "",
            "## Result",
            "",
            f"- incumbent_metric: `{incumbent_metric_text}`",
            f"- candidate_metric: `{result.metric:.4f}`",
            f"- delta: `{delta_text}`",
            f"- runtime_seconds: `{result.runtime_seconds:.2f}`",
            "",
            "## Decision",
            "",
            f"- `{decision}`",
            f"- {result.decision_reason}",
            "",
            "## Assessment",
            "",
            f"- benchmark_status: `{assessment['benchmark_status']}`",
            f"- benchmark_summary: {assessment['benchmark_summary']}",
            f"- implementation_readiness: `{assessment['implementation_readiness']}`",
            *(f"- {check['name']}: `{check['passed']}` ({check['detail']})" for check in assessment["checks"]),
            "",
            "## Next step",
            "",
            recommended_next_step or "No further bounded step has been selected yet.",
        ]
    )
    summary = (
        f"{proposal.mutation_name} {decision} with validation {proposal.dataset_key} "
        f"{result.metric:.4f}, delta {delta_text}, and readiness "
        f"{assessment['implementation_readiness']}."
    )
    return RunNarrative(
        title=proposal.mutation_name,
        summary=summary,
        decision=decision,
        markdown=markdown,
        recommended_next_step=recommended_next_step,
    )


def build_loop_summary(dataset_key: str, history: list[dict[str, Any]], final_incumbent: dict[str, Any] | None) -> LoopNarrative:
    promoted_count = sum(1 for step in history if step["result"]["promoted"])
    total_steps = len(history)
    final_metric = "n/a" if final_incumbent is None else f"{float(final_incumbent['metric']):.4f}"
    final_readiness = "n/a"
    if final_incumbent is not None and "assessment" in final_incumbent:
        final_readiness = str(final_incumbent["assessment"].get("implementation_readiness", "n/a"))
    lines = [
        f"# Autonomous Loop Summary: {dataset_key}",
        "",
        f"- executed_steps: `{total_steps}`",
        f"- promoted_steps: `{promoted_count}`",
        f"- final_incumbent_metric: `{final_metric}`",
        f"- final_implementation_readiness: `{final_readiness}`",
        "",
        "## Step history",
        "",
    ]
    lines.extend(
        (
            f"- step {step['step_index'] + 1}: `{step['proposal']['mutation_name']}` "
            f"-> `{ 'promote' if step['result']['promoted'] else 'reject' }`, "
            f"`{step['result']['assessment']['implementation_readiness']}`"
        )
        for step in history
    )
    return LoopNarrative(
        title=f"{dataset_key} autonomous loop",
        markdown="\n".join(lines),
    )


def render_markdown(narrative: RunNarrative | LoopNarrative) -> str:
    return narrative.markdown
