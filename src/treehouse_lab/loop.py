from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from treehouse_lab.config import load_yaml_file
from treehouse_lab.diagnosis import diagnose_run_state
from treehouse_lab.features import build_feature_plan, run_feature_plan, should_enable_feature_generation
from treehouse_lab.journal import (
    ensure_run_directories,
    load_incumbent,
    load_journal_entries,
    load_run_entry,
    update_journal_entry,
)
from treehouse_lab.llm import llm_loop_selection_enabled, select_bounded_proposal
from treehouse_lab.mutations import MutationCandidate, generate_candidates
from treehouse_lab.narratives import build_loop_summary, build_run_narrative, render_markdown
from treehouse_lab.proposals import ExperimentProposal, ProposalDecisionContext, build_baseline_proposal
from treehouse_lab.runner import ExperimentResult, TreehouseLabRunner


@dataclass(slots=True)
class LoopConfig:
    config_path: str
    max_steps: int = 3


@dataclass(slots=True)
class LoopStepResult:
    step_index: int
    proposal: dict[str, Any]
    result: dict[str, Any]
    narrative_path: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class LoopSummary:
    dataset_key: str
    config_path: str
    baseline_result: dict[str, Any] | None
    steps: list[dict[str, Any]]
    final_incumbent: dict[str, Any] | None
    loop_dir: str
    stop_reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class DiagnosisPreview:
    dataset_key: str
    config_path: str
    diagnosis: dict[str, Any]
    next_proposal: dict[str, Any] | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AutonomousLoopController:
    def __init__(self, config_path: str | Path):
        self.runner = TreehouseLabRunner(config_path)
        self.config = self.runner.config
        self.project_root = self.runner.project_root
        self.registry_key = self.runner.registry_key
        self.search_space = load_yaml_file(self.project_root / "configs" / "search_space.yaml")

    def ensure_incumbent(self) -> ExperimentResult | None:
        incumbent = load_incumbent(self.project_root, self.registry_key)
        if incumbent is not None:
            return None

        proposal = build_baseline_proposal(self.registry_key, self.config.hypothesis)
        result = self.runner.run_baseline(metadata={"dataset_key": self.registry_key, "proposal": proposal.to_dict()})
        narrative = build_run_narrative(proposal, result, incumbent_before=None, recommended_next_step="Run the first bounded mutation.")
        self._write_run_narrative(result, proposal, narrative)
        update_journal_entry(
            self.project_root,
            result.run_id,
            {
                "registry_key": self.registry_key,
                "proposal": proposal.to_dict(),
                "narrative": narrative.to_dict(),
                "recommended_next_step": "Run the first bounded mutation.",
            },
        )
        return result

    def choose_next_proposal(self, loop_step_index: int, loop_history: list[LoopStepResult]) -> ExperimentProposal | None:
        context, candidates = self._candidate_bundle(loop_step_index, loop_history)
        if not candidates:
            return None
        return self._select_candidate(context, candidates)

    def recommend_coach_proposal(self, loop_step_index: int = 0, loop_history: list[LoopStepResult] | None = None) -> ExperimentProposal | None:
        if load_incumbent(self.project_root, self.registry_key) is None:
            proposal = build_baseline_proposal(self.registry_key, self.config.hypothesis)
            proposal.llm_review = {
                "status": "not_applicable",
                "message": "No incumbent exists yet, so the coach recommendation is to establish the baseline first.",
            }
            return proposal

        history = [] if loop_history is None else loop_history
        context, candidates = self._candidate_bundle(loop_step_index, history)
        if not candidates:
            return None
        return self._select_candidate(context, candidates, force_llm=True)

    def proposal_for_mutation_type(
        self,
        mutation_type: str,
        loop_step_index: int = 0,
        loop_history: list[LoopStepResult] | None = None,
    ) -> ExperimentProposal | None:
        normalized_mutation_type = mutation_type.strip()
        if normalized_mutation_type == "baseline":
            if load_incumbent(self.project_root, self.registry_key) is None:
                return build_baseline_proposal(self.registry_key, self.config.hypothesis)
            return None

        history = [] if loop_history is None else loop_history
        _, candidates = self._candidate_bundle(loop_step_index, history)
        for candidate in candidates:
            if candidate.proposal.mutation_type == normalized_mutation_type:
                return candidate.proposal
        return None

    def run_proposal(self, proposal: ExperimentProposal) -> ExperimentResult:
        metadata = {
            "dataset_key": self.registry_key,
            "proposal_id": proposal.proposal_id,
            "mutation_type": proposal.mutation_type,
            "stage": proposal.stage,
            "proposal": proposal.to_dict(),
        }
        return self.runner.run_candidate(
            mutation_name=proposal.mutation_name,
            overrides=proposal.params_override,
            hypothesis=proposal.hypothesis,
            metadata=metadata,
            base_params=proposal.base_params,
        )

    def should_stop(self, history: list[LoopStepResult], max_steps: int) -> tuple[bool, str]:
        if len(history) >= max_steps:
            return True, f"Reached the requested loop length of {max_steps} steps."
        return False, ""

    def preview_next_proposal(self, loop_step_index: int, loop_history: list[LoopStepResult]) -> dict[str, Any] | None:
        proposal = self.choose_next_proposal(loop_step_index, loop_history)
        return proposal.to_dict() if proposal is not None else None

    def execute_proposal_step(
        self,
        proposal: ExperimentProposal,
        loop_step_index: int = 0,
        loop_history: list[LoopStepResult] | None = None,
        preview_follow_up: bool = True,
    ) -> LoopStepResult:
        history = [] if loop_history is None else loop_history
        incumbent_before = load_incumbent(self.project_root, self.registry_key)
        result = self.run_proposal(proposal)

        next_preview = None
        if preview_follow_up:
            preview_step = LoopStepResult(
                step_index=loop_step_index,
                proposal=proposal.to_dict(),
                result=result.to_dict(),
                narrative_path="",
            )
            next_preview = self.preview_next_proposal(loop_step_index + 1, history + [preview_step])

        next_step_text = None if next_preview is None else next_preview["mutation_name"]
        narrative = build_run_narrative(proposal, result, incumbent_before, recommended_next_step=next_step_text)
        narrative_path = self._write_run_narrative(result, proposal, narrative)

        update_journal_entry(
            self.project_root,
            result.run_id,
            {
                "registry_key": self.registry_key,
                "proposal": proposal.to_dict(),
                "narrative": narrative.to_dict(),
                "recommended_next_step": next_preview,
            },
        )

        result.metadata.update(
            {
                "proposal": proposal.to_dict(),
                "narrative": narrative.to_dict(),
                "recommended_next_step": next_preview,
            }
        )

        return LoopStepResult(
            step_index=loop_step_index,
            proposal=proposal.to_dict(),
            result=result.to_dict(),
            narrative_path=str(narrative_path),
        )

    def run_loop(self, max_steps: int = 3) -> LoopSummary:
        loop_dir = self._loop_dir()
        loop_dir.mkdir(parents=True, exist_ok=True)

        baseline_result = self.ensure_incumbent()
        history: list[LoopStepResult] = []
        stop_reason = f"Completed {max_steps} bounded loop steps."

        for loop_step_index in range(max_steps):
            proposal = self.choose_next_proposal(loop_step_index, history)
            if proposal is None:
                stop_reason = "No eligible bounded proposal remained."
                break

            step = self.execute_proposal_step(
                proposal,
                loop_step_index=loop_step_index,
                loop_history=history,
                preview_follow_up=loop_step_index + 1 < max_steps,
            )
            history.append(step)

            stop, reason = self.should_stop(history, max_steps)
            if stop:
                stop_reason = reason
                break

        final_incumbent = self._enriched_incumbent()
        feature_plan = build_feature_plan(self.search_space, should_enable_feature_generation([step.to_dict() for step in history], final_incumbent))
        feature_result = run_feature_plan(feature_plan)
        loop_narrative = build_loop_summary(self.registry_key, [step.to_dict() for step in history], final_incumbent)
        (loop_dir / "summary.md").write_text(render_markdown(loop_narrative), encoding="utf-8")
        (loop_dir / "feature_generation_plan.json").write_text(
            json.dumps(
                {"plan": feature_plan.to_dict(), "result": feature_result.to_dict()},
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

        summary = LoopSummary(
            dataset_key=self.registry_key,
            config_path=str(self.runner.config_path),
            baseline_result=None if baseline_result is None else baseline_result.to_dict(),
            steps=[step.to_dict() for step in history],
            final_incumbent=final_incumbent,
            loop_dir=str(loop_dir),
            stop_reason=stop_reason,
        )
        (loop_dir / "summary.json").write_text(json.dumps(summary.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        return summary

    def next_proposal(self) -> ExperimentProposal:
        incumbent = load_incumbent(self.project_root, self.registry_key)
        if incumbent is None:
            return build_baseline_proposal(self.registry_key, self.config.hypothesis)
        proposal = self.choose_next_proposal(loop_step_index=0, loop_history=[])
        if proposal is None:
            msg = "No eligible bounded proposal is available."
            raise ValueError(msg)
        return proposal

    def diagnose(self) -> DiagnosisPreview:
        context = self._build_context(loop_step_index=0, loop_history=[])
        next_proposal = self.choose_next_proposal(loop_step_index=0, loop_history=[])
        return DiagnosisPreview(
            dataset_key=self.registry_key,
            config_path=str(self.runner.config_path),
            diagnosis=context.diagnosis,
            next_proposal=None if next_proposal is None else next_proposal.to_dict(),
        )

    def _select_candidate(
        self,
        context: ProposalDecisionContext,
        candidates: list[Any],
        force_llm: bool = False,
    ) -> ExperimentProposal:
        top_proposal = candidates[0].proposal
        if not force_llm and not llm_loop_selection_enabled(self.project_root):
            top_proposal.llm_review = {
                "status": "disabled",
                "message": "LLM loop selection is disabled, so Treehouse Lab used deterministic candidate ranking.",
                "candidate_count": len(candidates),
            }
            return top_proposal

        selection_context = {
            "dataset_key": context.dataset_key,
            "project_root": str(self.project_root),
            "diagnosis": context.diagnosis,
            "promote_threshold": context.promote_threshold,
            "incumbent": {
                "run_id": context.incumbent_run_id,
                "metric": context.incumbent_metric,
                "params": context.incumbent_params,
                "metrics": context.incumbent_metrics,
                "split_summary": context.split_summary,
            },
            "recent_entries": [
                {
                    "name": entry.get("name"),
                    "promoted": entry.get("promoted"),
                    "delta": entry.get("comparison_to_incumbent", {}).get("delta"),
                    "mutation_type": entry.get("proposal", {}).get("mutation_type") or entry.get("mutation_type"),
                    "summary": entry.get("diagnosis", {}).get("summary") or entry.get("decision_reason"),
                }
                for entry in context.journal_entries[-5:]
            ],
        }
        selection_candidates = [
            {
                "proposal_id": candidate.proposal.proposal_id,
                "mutation_type": candidate.proposal.mutation_type,
                "mutation_name": candidate.proposal.mutation_name,
                "score": candidate.proposal.score,
                "hypothesis": candidate.proposal.hypothesis,
                "rationale": candidate.proposal.rationale,
                "risk_level": candidate.proposal.risk_level,
                "expected_upside": candidate.proposal.expected_upside,
                "params_override": candidate.proposal.params_override,
            }
            for candidate in candidates
        ]

        selection = select_bounded_proposal(selection_context, selection_candidates)
        selected = next(
            (candidate.proposal for candidate in candidates if candidate.proposal.proposal_id == selection.selected_proposal_id),
            None,
        )
        if selected is None:
            top_proposal.llm_review = {
                **selection.to_dict(),
                "status": "fallback",
                "fallback_proposal_id": top_proposal.proposal_id,
            }
            return top_proposal

        selected.llm_review = selection.to_dict()
        return selected

    def _candidate_bundle(
        self,
        loop_step_index: int,
        loop_history: list[LoopStepResult],
    ) -> tuple[ProposalDecisionContext, list[MutationCandidate]]:
        context = self._build_context(loop_step_index, loop_history)
        candidates = generate_candidates(context)
        return context, candidates

    def _build_context(self, loop_step_index: int, loop_history: list[LoopStepResult]) -> ProposalDecisionContext:
        incumbent = load_incumbent(self.project_root, self.registry_key)
        journal_entries = load_journal_entries(self.project_root, self.registry_key)
        incumbent_entry = None if incumbent is None else load_run_entry(self.project_root, str(incumbent["run_id"]))

        incumbent_params = {}
        incumbent_metrics = {}
        split_summary = {}
        if incumbent is not None:
            incumbent_params = dict(incumbent.get("params", {}))
            incumbent_metrics = dict(incumbent.get("metrics", {}))
            if incumbent_entry is not None:
                incumbent_params.update(incumbent_entry.get("params", {}))
                incumbent_metrics.update(incumbent_entry.get("metrics", {}))
                split_summary = dict(incumbent_entry.get("split_summary", {}))

        if not incumbent_params:
            incumbent_params = self.runner._resolve_model_params({})

        overfit_gap = float(incumbent_metrics.get("train_roc_auc", 0.0)) - float(incumbent_metrics.get("validation_roc_auc", 0.0))
        positive_rate = float(split_summary.get("validation_positive_rate", split_summary.get("train_positive_rate", 0.5)))
        diagnosis = diagnose_run_state(self.config, incumbent_metrics, split_summary, recent_entries=journal_entries).to_dict()

        recent_mutation_types = [
            entry.get("mutation_type")
            or entry.get("metadata", {}).get("mutation_type")
            or entry.get("proposal", {}).get("mutation_type")
            for entry in journal_entries[-5:]
        ]
        recent_mutation_names = [
            entry.get("name") or entry.get("proposal", {}).get("mutation_name")
            for entry in journal_entries[-5:]
        ]

        return ProposalDecisionContext(
            dataset_key=self.registry_key,
            primary_metric=self.config.primary_metric,
            promote_threshold=self.config.promote_if_delta_at_least,
            incumbent_run_id=None if incumbent is None else str(incumbent["run_id"]),
            incumbent_metric=None if incumbent is None else float(incumbent["metric"]),
            incumbent_params=incumbent_params,
            incumbent_metrics=incumbent_metrics,
            split_summary=split_summary,
            overfit_gap=overfit_gap,
            positive_rate=positive_rate,
            search_space=self.search_space,
            journal_entries=journal_entries,
            loop_step_index=loop_step_index,
            executed_mutation_types=[
                mutation_type
                for mutation_type in [*recent_mutation_types, *[step.proposal["mutation_type"] for step in loop_history]]
                if mutation_type
            ],
            executed_mutation_names=[
                mutation_name
                for mutation_name in [*recent_mutation_names, *[step.proposal["mutation_name"] for step in loop_history]]
                if mutation_name
            ],
            allow_feature_generation=bool(self.search_space.get("policy", {}).get("allow_feature_generation", False)),
            diagnosis=diagnosis,
        )

    def _write_run_narrative(self, result: ExperimentResult, proposal: ExperimentProposal, narrative: Any) -> Path:
        artifact_dir = Path(result.artifact_dir)
        narrative_path = artifact_dir / "narrative.md"
        proposal_path = artifact_dir / "proposal.json"
        narrative_path.write_text(render_markdown(narrative), encoding="utf-8")
        proposal_path.write_text(json.dumps(proposal.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        return narrative_path

    def _loop_dir(self) -> Path:
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        loops_dir = ensure_run_directories(self.project_root) / "loops"
        return loops_dir / f"{timestamp}-{self.registry_key}"

    def _enriched_incumbent(self) -> dict[str, Any] | None:
        incumbent = load_incumbent(self.project_root, self.registry_key)
        if incumbent is None:
            return None
        incumbent_entry = load_run_entry(self.project_root, str(incumbent["run_id"]))
        if incumbent_entry is None:
            return incumbent
        enriched = dict(incumbent)
        for key in ("params", "metrics", "split_summary", "metadata", "assessment", "diagnosis", "reason_codes"):
            if key in incumbent_entry:
                enriched[key] = incumbent_entry[key]
        return enriched
