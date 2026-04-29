from __future__ import annotations

import importlib
import json
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from treehouse_lab.config import ExperimentConfig
from treehouse_lab.datasets import DatasetBundle, DatasetSplit, load_dataset, split_dataset
from treehouse_lab.evaluation import RunAssessment, assess_run
from treehouse_lab.llm import generate_comparison_summary
from treehouse_lab.loop import AutonomousLoopController
from treehouse_lab.runner import TreehouseLabRunner

DEFAULT_AUTOGLUON_PROFILE = "practical"
PRACTICAL_AUTOGLUON_PRESETS = ["medium_quality", "optimize_for_deployment"]
PRACTICAL_AUTOGLUON_EXCLUDED_MODEL_TYPES = ["KNN", "NN_TORCH", "FASTAI", "AG_AUTOMM"]
PRACTICAL_AUTOGLUON_TIME_LIMIT_SECONDS = 30


@dataclass(slots=True)
class ComparisonRunSummary:
    runner_key: str
    display_name: str
    status: str
    backend: str
    validation_metric: float | None
    test_metric: float | None
    runtime_seconds: float | None
    benchmark_status: str | None
    implementation_readiness: str | None
    artifact_path: str | None
    notes: list[str] = field(default_factory=list)
    workflow_traits: dict[str, str] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ComparisonSuiteResult:
    dataset_key: str
    config_path: str
    output_dir: str
    primary_metric: str
    split_summary: dict[str, Any]
    runners: list[dict[str, Any]]
    llm_summary: dict[str, Any] | None
    report_path: str
    summary_path: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class AutoGluonRunnerConfig:
    profile: str
    presets: str | list[str]
    time_limit: int
    excluded_model_types: list[str] = field(default_factory=list)
    display_name: str = "AutoGluon Tabular"
    notes: list[str] = field(default_factory=list)


def run_comparison_suite(
    config_path: str | Path,
    *,
    output_dir: str | Path | None = None,
    loop_steps: int = 3,
    include_autogluon: bool = True,
    include_llm_summary: bool = False,
    llm_question: str | None = None,
    autogluon_profile: str = DEFAULT_AUTOGLUON_PROFILE,
    autogluon_presets: str | list[str] | None = None,
    autogluon_time_limit: int | None = None,
) -> ComparisonSuiteResult:
    base_runner = TreehouseLabRunner(config_path)
    config = base_runner.config
    dataset = load_dataset(config, base_runner.project_root)
    split = split_dataset(dataset, config)
    suite_output_dir = _resolve_output_dir(base_runner, output_dir)
    suite_output_dir.mkdir(parents=True, exist_ok=False)
    runners_dir = suite_output_dir / "runners"
    runners_dir.mkdir(parents=True, exist_ok=True)

    run_summaries = [
        _run_plain_xgboost(base_runner, dataset, split, runners_dir / "plain_xgboost"),
        _run_treehouse_baseline(base_runner, runners_dir / "treehouse_baseline"),
        _run_treehouse_loop(base_runner, runners_dir / "treehouse_loop", loop_steps=loop_steps),
    ]

    if include_autogluon:
        autogluon_runner_config = _resolve_autogluon_runner_config(
            base_runner.config,
            profile=autogluon_profile,
            presets=autogluon_presets,
            time_limit=autogluon_time_limit,
        )
        run_summaries.append(
            _run_autogluon(
                base_runner,
                dataset,
                split,
                runners_dir / "autogluon_tabular",
                runner_config=autogluon_runner_config,
            )
        )

    llm_summary = None
    if include_llm_summary:
        llm_summary = generate_comparison_summary(
            _build_comparison_llm_context(
                base_runner=base_runner,
                dataset=dataset,
                split=split,
                run_summaries=run_summaries,
                loop_steps=loop_steps,
            ),
            question=llm_question,
        ).to_dict()

    report_path = suite_output_dir / "report.md"
    summary_path = suite_output_dir / "summary.json"
    report_path.write_text(
        _render_report(
            base_runner=base_runner,
            dataset=dataset,
            split=split,
            run_summaries=run_summaries,
            loop_steps=loop_steps,
            llm_summary=llm_summary,
        ),
        encoding="utf-8",
    )
    suite_result = ComparisonSuiteResult(
        dataset_key=base_runner.registry_key,
        config_path=str(base_runner.config_path),
        output_dir=str(suite_output_dir),
        primary_metric=config.primary_metric,
        split_summary=split.summary(),
        runners=[run_summary.to_dict() for run_summary in run_summaries],
        llm_summary=llm_summary,
        report_path=str(report_path),
        summary_path=str(summary_path),
    )
    summary_path.write_text(json.dumps(suite_result.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    return suite_result


def _run_plain_xgboost(
    base_runner: TreehouseLabRunner,
    dataset: DatasetBundle,
    split: DatasetSplit,
    runner_dir: Path,
) -> ComparisonRunSummary:
    runner_dir.mkdir(parents=True, exist_ok=True)
    run_started = time.perf_counter()
    params = base_runner._resolve_model_params(  # noqa: SLF001
        {},
        task_kind=str(dataset.target_profile["task_kind"]),
        class_count=int(dataset.target_profile["class_count"]),
    )
    model, backend = base_runner._build_model(params)  # noqa: SLF001
    model.fit(split.X_train, split.y_train)
    metrics = base_runner._compute_metrics(model, split, task_kind=str(dataset.target_profile["task_kind"]))  # noqa: SLF001
    runtime_seconds = time.perf_counter() - run_started
    assessment = _baseline_style_assessment(base_runner.config, metrics, split.summary(), runtime_seconds)
    details = {
        "metrics": metrics,
        "assessment": assessment.to_dict(),
        "params": params,
        "target_name": dataset.target_name,
    }
    _write_runner_outputs(
        runner_dir,
        summary=details,
        notes=[
            "Uses the same dataset config, split policy, and baseline params as Treehouse Lab.",
            "Does not create a journal, incumbent registry, or bounded next-step proposal trail.",
        ],
    )
    return ComparisonRunSummary(
        runner_key="plain_xgboost",
        display_name="Plain XGBoost Baseline",
        status="completed",
        backend=backend,
        validation_metric=float(metrics[base_runner.config.primary_metric]),
        test_metric=float(metrics.get(f"test_{base_runner.config.primary_metric}", metrics.get("test_accuracy", 0.0))),
        runtime_seconds=runtime_seconds,
        benchmark_status=assessment.benchmark_status,
        implementation_readiness=assessment.implementation_readiness,
        artifact_path=str(runner_dir),
        notes=[
            "Strong reference baseline on the same split contract.",
            "Useful for metric and runtime comparison, but not for audit trail or guided iteration.",
        ],
        workflow_traits={
            "search_style": "manual_baseline_only",
            "artifact_trail": "minimal",
            "journal": "no",
            "bounded_next_step": "no",
            "llm_guidance": "no",
        },
        details=details,
    )


def _run_treehouse_baseline(base_runner: TreehouseLabRunner, runner_dir: Path) -> ComparisonRunSummary:
    workspace_root = runner_dir / "workspace"
    workspace_config_path = _prepare_isolated_workspace(base_runner, workspace_root)
    runner = TreehouseLabRunner(workspace_config_path)
    run_started = time.perf_counter()
    result = runner.run_baseline()
    runtime_seconds = time.perf_counter() - run_started
    details = result.to_dict()
    details["workspace_root"] = str(workspace_root)
    _write_runner_outputs(runner_dir, summary=details, notes=["Runs Treehouse Lab baseline in an isolated workspace."])
    return ComparisonRunSummary(
        runner_key="treehouse_lab_baseline",
        display_name="Treehouse Lab Baseline",
        status="completed",
        backend=result.backend,
        validation_metric=float(result.metric),
        test_metric=float(result.metrics.get(f"test_{base_runner.config.primary_metric}", result.metrics.get("test_accuracy", 0.0))),
        runtime_seconds=runtime_seconds,
        benchmark_status=result.assessment["benchmark_status"],
        implementation_readiness=result.assessment["implementation_readiness"],
        artifact_path=str(Path(result.artifact_dir)),
        notes=[
            "Same underlying baseline family, but with config snapshot, assessment, and reusable artifacts.",
            "Creates the incumbent and journal foundation that later bounded steps build on.",
        ],
        workflow_traits={
            "search_style": "bounded_baseline",
            "artifact_trail": "full",
            "journal": "yes",
            "bounded_next_step": "not_yet",
            "llm_guidance": "no",
        },
        details=details,
    )


def _run_treehouse_loop(
    base_runner: TreehouseLabRunner,
    runner_dir: Path,
    *,
    loop_steps: int,
) -> ComparisonRunSummary:
    workspace_root = runner_dir / "workspace"
    workspace_config_path = _prepare_isolated_workspace(base_runner, workspace_root)
    controller = AutonomousLoopController(workspace_config_path)
    run_started = time.perf_counter()
    summary = controller.run_loop(max_steps=loop_steps)
    runtime_seconds = time.perf_counter() - run_started
    final_incumbent = summary.final_incumbent or {}
    step_results = summary.steps
    promoted_steps = [step for step in step_results if step["result"].get("promoted")]
    details = summary.to_dict()
    details["workspace_root"] = str(workspace_root)
    details["promotion_count"] = len(promoted_steps)
    llm_guidance = _summarize_loop_llm_guidance(step_results)
    details.update(llm_guidance)
    _write_runner_outputs(
        runner_dir,
        summary=details,
        notes=["Runs the bounded Treehouse loop in an isolated workspace so comparison work does not pollute the main repo journal."],
    )
    final_metrics = final_incumbent.get("metrics", {})
    return ComparisonRunSummary(
        runner_key="treehouse_lab_loop",
        display_name=f"Treehouse Lab {loop_steps}-Step Loop",
        status="completed",
        backend=str(final_incumbent.get("backend", "xgboost")),
        validation_metric=float(final_incumbent.get("metric", 0.0)) if final_incumbent else None,
        test_metric=float(final_metrics.get(f"test_{base_runner.config.primary_metric}", final_metrics.get("test_accuracy", 0.0)))
        if final_metrics
        else None,
        runtime_seconds=runtime_seconds,
        benchmark_status=str(final_incumbent.get("assessment", {}).get("benchmark_status")) if final_incumbent else None,
        implementation_readiness=str(final_incumbent.get("assessment", {}).get("implementation_readiness")) if final_incumbent else None,
        artifact_path=str(summary.loop_dir),
        notes=[
            f"Executes {loop_steps} bounded steps and records why each proposal was selected.",
            f"Promotion count: {len(promoted_steps)}. Stop reason: {summary.stop_reason}",
            _format_loop_llm_guidance_note(llm_guidance),
        ],
        workflow_traits={
            "search_style": "bounded_loop",
            "artifact_trail": "full_plus_loop_summary",
            "journal": "yes",
            "bounded_next_step": "yes",
            "llm_guidance": "yes" if llm_guidance["llm_guided_step_count"] > 0 else "no",
        },
        details=details,
    )


def _run_autogluon(
    base_runner: TreehouseLabRunner,
    dataset: DatasetBundle,
    split: DatasetSplit,
    runner_dir: Path,
    *,
    runner_config: AutoGluonRunnerConfig,
) -> ComparisonRunSummary:
    runner_dir.mkdir(parents=True, exist_ok=True)
    try:
        tabular_module = importlib.import_module("autogluon.tabular")
    except ModuleNotFoundError:
        details = {
            "install_hint": "Install AutoGluon separately and rerun compare to include this external benchmark.",
            "profile": runner_config.profile,
            "requested_presets": runner_config.presets,
            "requested_time_limit": runner_config.time_limit,
            "excluded_model_types": runner_config.excluded_model_types,
        }
        _write_runner_outputs(runner_dir, summary=details, notes=["AutoGluon not installed in the current environment."])
        return ComparisonRunSummary(
            runner_key="autogluon_tabular",
            display_name=runner_config.display_name,
            status="unavailable",
            backend="autogluon",
            validation_metric=None,
            test_metric=None,
            runtime_seconds=None,
            benchmark_status=None,
            implementation_readiness=None,
            artifact_path=str(runner_dir),
            notes=[
                "Optional external benchmark runner.",
                "Unavailable because AutoGluon is not installed in this environment.",
            ],
            workflow_traits={
                "search_style": "opaque_automl",
                "artifact_trail": "model_directory_only",
                "journal": "no",
                "bounded_next_step": "no",
                "llm_guidance": "no",
            },
            details=details,
        )

    TabularPredictor = getattr(tabular_module, "TabularPredictor")
    model_dir = runner_dir / "model"
    run_started = time.perf_counter()
    train_df = _combine_features_and_target(split.X_train, split.y_train, dataset.target_name)
    val_df = _combine_features_and_target(split.X_val, split.y_val, dataset.target_name)
    test_df = _combine_features_and_target(split.X_test, split.y_test, dataset.target_name)
    predictor = TabularPredictor(
        label=dataset.target_name,
        problem_type=_autogluon_problem_type(str(dataset.target_profile["task_kind"])),
        eval_metric=_autogluon_eval_metric(base_runner.config.primary_metric),
        path=str(model_dir),
        verbosity=0,
    )
    fit_kwargs = _build_autogluon_fit_kwargs(train_df, val_df, runner_config)
    try:
        predictor.fit(**fit_kwargs)
    except Exception as exc:  # pragma: no cover - depends on external AutoGluon runtime
        runtime_seconds = time.perf_counter() - run_started
        details = {
            "profile": runner_config.profile,
            "presets": runner_config.presets,
            "time_limit": runner_config.time_limit,
            "excluded_model_types": runner_config.excluded_model_types,
            "model_dir": str(model_dir),
            "error_message": str(exc),
        }
        _write_runner_outputs(
            runner_dir,
            summary=details,
            notes=[
                *runner_config.notes,
                "AutoGluon failed inside the external benchmark harness.",
            ],
        )
        return ComparisonRunSummary(
            runner_key="autogluon_tabular",
            display_name=runner_config.display_name,
            status="error",
            backend="autogluon",
            validation_metric=None,
            test_metric=None,
            runtime_seconds=runtime_seconds,
            benchmark_status=None,
            implementation_readiness=None,
            artifact_path=str(runner_dir),
            notes=[
                *runner_config.notes,
                f"AutoGluon benchmark failed: {exc}",
            ],
            workflow_traits={
                "search_style": "opaque_automl",
                "artifact_trail": "model_directory_only",
                "journal": "no",
                "bounded_next_step": "no",
                "llm_guidance": "no",
            },
            details=details,
        )
    runtime_seconds = time.perf_counter() - run_started
    adapter = _AutoGluonPredictorAdapter(predictor)
    metrics = base_runner._compute_metrics(adapter, split, task_kind=str(dataset.target_profile["task_kind"]))  # noqa: SLF001
    assessment = _baseline_style_assessment(base_runner.config, metrics, split.summary(), runtime_seconds)
    leaderboard = predictor.leaderboard(test_df, display=False)
    details = {
        "metrics": metrics,
        "assessment": assessment.to_dict(),
        "leaderboard": leaderboard.to_dict(orient="records"),
        "profile": runner_config.profile,
        "presets": runner_config.presets,
        "time_limit": runner_config.time_limit,
        "excluded_model_types": runner_config.excluded_model_types,
        "model_dir": str(model_dir),
    }
    _write_runner_outputs(
        runner_dir,
        summary=details,
        notes=[
            "Comparison harness passes the same validation split as tuning_data for a fairer side-by-side.",
            "AutoGluon still chooses its own model families and ensembling strategy internally.",
        ],
    )
    return ComparisonRunSummary(
        runner_key="autogluon_tabular",
        display_name=runner_config.display_name,
        status="completed",
        backend="autogluon",
        validation_metric=float(metrics[base_runner.config.primary_metric]),
        test_metric=float(metrics.get(f"test_{base_runner.config.primary_metric}", metrics.get("test_accuracy", 0.0))),
        runtime_seconds=runtime_seconds,
        benchmark_status=assessment.benchmark_status,
        implementation_readiness=assessment.implementation_readiness,
        artifact_path=str(model_dir),
        notes=[
            *runner_config.notes,
            f"Ran with presets `{runner_config.presets}` and time_limit `{runner_config.time_limit}` seconds.",
            "Useful as an external automation benchmark, but its search path is less explicit than Treehouse Lab.",
        ],
        workflow_traits={
            "search_style": "opaque_automl",
            "artifact_trail": "model_directory_only",
            "journal": "no",
            "bounded_next_step": "no",
            "llm_guidance": "no",
        },
        details=details,
    )


def _render_report(
    *,
    base_runner: TreehouseLabRunner,
    dataset: DatasetBundle,
    split: DatasetSplit,
    run_summaries: list[ComparisonRunSummary],
    loop_steps: int,
    llm_summary: dict[str, Any] | None,
) -> str:
    timestamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%SZ")
    lines = [
        f"# Comparison Report: {base_runner.registry_key}",
        "",
        f"- generated_at: `{timestamp}`",
        f"- config_path: `{base_runner.config_path}`",
        f"- dataset_name: {base_runner.config.source.name or base_runner.registry_key}",
        f"- target: `{dataset.target_name}`",
        f"- task_kind: `{dataset.target_profile['task_kind']}`",
        f"- primary_metric: `{base_runner.config.primary_metric}`",
        f"- requested_treehouse_loop_steps: `{loop_steps}`",
        "",
        "## Shared split contract",
        "",
        *(f"- {key}: `{value}`" for key, value in split.summary().items()),
        "",
        "## Results",
        "",
        "| runner | status | validation | test | runtime_s | readiness | benchmark_status |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for run_summary in run_summaries:
        validation_value = "n/a" if run_summary.validation_metric is None else f"{run_summary.validation_metric:.4f}"
        test_value = "n/a" if run_summary.test_metric is None else f"{run_summary.test_metric:.4f}"
        runtime_value = "n/a" if run_summary.runtime_seconds is None else f"{run_summary.runtime_seconds:.2f}"
        lines.append(
            f"| {run_summary.display_name} | {run_summary.status} | {validation_value} | {test_value} | {runtime_value} | "
            f"{run_summary.implementation_readiness or 'n/a'} | {run_summary.benchmark_status or 'n/a'} |"
        )

    lines.extend(
        [
            "",
            "## Outcome gates",
            "",
            "- `benchmark_status` answers whether a runner actually established or improved the benchmark position.",
            "- `implementation_readiness` answers whether the run cleared the configured runtime, gap, and feature-budget checks.",
            "",
            "| runner | benchmark decision | implementation decision | quick read |",
            "| --- | --- | --- | --- |",
        ]
    )
    for run_summary in run_summaries:
        lines.append(
            f"| {run_summary.display_name} | {run_summary.benchmark_status or 'n/a'} | "
            f"{run_summary.implementation_readiness or 'n/a'} | {_summarize_outcome_split(run_summary)} |"
        )

    lines.extend(
        [
            "",
            "## Workflow traits",
            "",
            "| runner | search_style | artifact_trail | journal | bounded_next_step | llm_guidance |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for run_summary in run_summaries:
        traits = run_summary.workflow_traits
        lines.append(
            f"| {run_summary.display_name} | {traits.get('search_style', 'n/a')} | {traits.get('artifact_trail', 'n/a')} | "
            f"{traits.get('journal', 'n/a')} | {traits.get('bounded_next_step', 'n/a')} | {traits.get('llm_guidance', 'n/a')} |"
        )

    lines.extend(_render_feature_generation_decisions(run_summaries))

    lines.extend(
        [
            "",
            "## Practical takeaway",
            "",
            *_render_practical_takeaway(run_summaries),
            "",
            "## Interpretation",
            "",
            "- `plain_xgboost` tells you whether Treehouse Lab is at least anchored to a strong explicit baseline.",
            "- `treehouse_lab_baseline` shows what you gain immediately from the product: artifacts, policy checks, and a reusable incumbent.",
            f"- `treehouse_lab_loop` shows the product layer: bounded next-step selection, journaling, and narrative attached to execution over {loop_steps} steps.",
            "- `autogluon_tabular` is an external automation benchmark. It is useful for raw automation comparison, not as the center of the Treehouse Lab product shape.",
        ]
    )
    if llm_summary is not None:
        lines.extend(
            [
                "",
                "## LLM synthesis",
                "",
                f"- status: `{llm_summary.get('status', 'unknown')}`",
                f"- provider: `{llm_summary.get('provider', 'unknown')}`",
                f"- model: `{llm_summary.get('model') or 'n/a'}`",
            ]
        )
        if llm_summary.get("message"):
            lines.append(f"- message: {llm_summary['message']}")
        lines.append("")
        answer = str(llm_summary.get("answer") or "").strip()
        if answer:
            lines.append(answer)
            lines.append("")

    lines.extend(["## Runner notes", ""])
    for run_summary in run_summaries:
        lines.append(f"### {run_summary.display_name}")
        lines.append("")
        lines.extend(f"- {note}" for note in run_summary.notes)
        if run_summary.artifact_path:
            lines.append(f"- artifact_path: `{run_summary.artifact_path}`")
        lines.append("")

    return "\n".join(lines)


def _summarize_outcome_split(run_summary: ComparisonRunSummary) -> str:
    benchmark_status = str(run_summary.benchmark_status or "n/a")
    implementation_readiness = str(run_summary.implementation_readiness or "n/a")
    if benchmark_status == "n/a" or implementation_readiness == "n/a":
        return "Decision split unavailable."
    if implementation_readiness == "implementation_ready" and benchmark_status in {"baseline_established", "better_than_incumbent"}:
        return "Strong result: credible benchmark position and ready under current policy."
    if implementation_readiness == "implementation_ready":
        return "Operationally credible, but not enough benchmark lift to change the story."
    if benchmark_status in {"baseline_established", "better_than_incumbent"}:
        return "Benchmark progress exists, but readiness still needs work."
    return "Neither benchmark improvement nor readiness cleared."


def _render_feature_generation_decisions(run_summaries: list[ComparisonRunSummary]) -> list[str]:
    decisions = [_summarize_feature_generation_decision(run_summary) for run_summary in run_summaries]
    lines = [
        "",
        "## Feature-generation decisions",
        "",
        "- `considered` means a runner exposed a bounded feature-generation branch in its executed steps.",
        "- `selected` means that branch was chosen for execution; `applied` means new train-only generated features entered the model matrix.",
        "",
        "| runner | considered | selected | applied | generated features | outcome gates | complexity read |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for decision in decisions:
        lines.append(
            f"| {decision['runner']} | {decision['considered']} | {decision['selected']} | {decision['applied']} | "
            f"{decision['generated_feature_count']} | {decision['outcome_gates']} | {decision['complexity_read']} |"
        )

    detail_lines = _render_feature_generation_detail_lines(decisions)
    if detail_lines:
        lines.extend(["", "Feature-generation detail:", "", *detail_lines])
    return lines


def _summarize_feature_generation_decision(run_summary: ComparisonRunSummary) -> dict[str, Any]:
    considered = False
    selected = False
    applied = False
    generated_feature_count = 0
    generated_feature_specs: list[dict[str, Any]] = []
    reasons: list[str] = []
    benchmark_status = run_summary.benchmark_status or "n/a"
    implementation_readiness = run_summary.implementation_readiness or "n/a"

    for step in run_summary.details.get("steps", []):
        proposal = step.get("proposal", {}) if isinstance(step, dict) else {}
        result = step.get("result", {}) if isinstance(step, dict) else {}
        proposal_feature_generation = proposal.get("feature_generation", {}) if isinstance(proposal, dict) else {}
        result_feature_generation = result.get("feature_generation", {}) if isinstance(result, dict) else {}

        if proposal_feature_generation or result_feature_generation:
            considered = True
        if bool(proposal_feature_generation.get("enabled")) or bool(result_feature_generation.get("enabled")):
            selected = True
        if bool(result_feature_generation.get("applied")):
            applied = True

        generated_feature_count += int(result_feature_generation.get("generated_feature_count") or 0)
        generated_feature_specs.extend(
            spec
            for spec in result_feature_generation.get("generated_feature_specs", [])
            if isinstance(spec, dict)
        )
        reason = proposal_feature_generation.get("reason") or result_feature_generation.get("reason")
        if reason:
            reasons.append(str(reason))

        assessment = result.get("assessment", {}) if isinstance(result, dict) else {}
        benchmark_status = str(assessment.get("benchmark_status") or benchmark_status)
        implementation_readiness = str(assessment.get("implementation_readiness") or implementation_readiness)

    return {
        "runner": run_summary.display_name,
        "considered": _yes_no(considered),
        "selected": _yes_no(selected),
        "applied": _yes_no(applied),
        "generated_feature_count": generated_feature_count,
        "outcome_gates": f"{benchmark_status} / {implementation_readiness}",
        "complexity_read": _summarize_feature_generation_complexity(selected, applied, benchmark_status, implementation_readiness),
        "generated_feature_specs": generated_feature_specs,
        "reasons": reasons,
    }


def _summarize_feature_generation_complexity(
    selected: bool,
    applied: bool,
    benchmark_status: str,
    implementation_readiness: str,
) -> str:
    if not selected:
        return "No bounded feature branch was selected."
    if not applied:
        return "Feature generation was selected, but no generated features were applied."
    if implementation_readiness == "implementation_ready" and benchmark_status in {"baseline_established", "better_than_incumbent"}:
        return "Added bounded features and cleared the outcome gates."
    return "Added bounded features, but outcome gates did not justify the added complexity."


def _render_feature_generation_detail_lines(decisions: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    for decision in decisions:
        reasons = decision["reasons"]
        if reasons:
            lines.append(f"- {decision['runner']} reason: {reasons[-1]}")
        for spec in decision["generated_feature_specs"][:5]:
            columns = ", ".join(str(column) for column in spec.get("columns", []))
            lines.append(
                f"- {decision['runner']} generated feature: `{spec.get('name', 'unknown')}` "
                f"via `{spec.get('operation', 'unknown')}` on `{columns}`"
            )
    return lines


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"


def _resolve_output_dir(base_runner: TreehouseLabRunner, output_dir: str | Path | None) -> Path:
    if output_dir is not None:
        candidate = Path(output_dir).expanduser()
        return candidate if candidate.is_absolute() else (base_runner.project_root / candidate).resolve()
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return base_runner.project_root / "outputs" / "comparisons" / base_runner.registry_key / timestamp


def _resolve_autogluon_runner_config(
    config: ExperimentConfig,
    *,
    profile: str,
    presets: str | list[str] | None,
    time_limit: int | None,
) -> AutoGluonRunnerConfig:
    normalized_profile = profile.strip().lower()
    requested_time_limit = int(config.evaluation_policy.max_runtime_seconds or config.max_runtime_minutes * 60)
    if normalized_profile == "practical":
        return AutoGluonRunnerConfig(
            profile="practical",
            presets=_normalize_autogluon_presets(presets) or PRACTICAL_AUTOGLUON_PRESETS,
            time_limit=time_limit or min(requested_time_limit, PRACTICAL_AUTOGLUON_TIME_LIMIT_SECONDS),
            excluded_model_types=list(PRACTICAL_AUTOGLUON_EXCLUDED_MODEL_TYPES),
            display_name="AutoGluon Tabular (Practical)",
            notes=[
                "Uses a fast rerunnable preset stack for practical benchmark runs.",
                "Keeps AutoGluon in quick-reference mode so Treehouse can be compared against an external baseline instead of a long opaque sweep.",
            ],
        )
    if normalized_profile == "full":
        return AutoGluonRunnerConfig(
            profile="full",
            presets=_normalize_autogluon_presets(presets) or "medium_quality",
            time_limit=time_limit or requested_time_limit,
            display_name="AutoGluon Tabular (Full)",
            notes=["Runs the broader AutoGluon tabular stack for a heavier external automation reference."],
        )
    msg = f"Unsupported AutoGluon profile: {profile}. Supported profiles are `practical` and `full`."
    raise ValueError(msg)


def _normalize_autogluon_presets(presets: str | list[str] | None) -> str | list[str] | None:
    if presets is None:
        return None
    if isinstance(presets, list):
        normalized = [str(item).strip() for item in presets if str(item).strip()]
        return normalized or None
    parts = [part.strip() for part in str(presets).split(",") if part.strip()]
    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]
    return parts


def _build_autogluon_fit_kwargs(
    train_df: pd.DataFrame,
    tuning_df: pd.DataFrame,
    runner_config: AutoGluonRunnerConfig,
) -> dict[str, Any]:
    fit_kwargs: dict[str, Any] = {
        "train_data": train_df,
        "tuning_data": tuning_df,
        "time_limit": runner_config.time_limit,
        "presets": runner_config.presets,
        # Some practical presets enable bagging internally. Keep the explicit
        # validation split usable in that mode instead of letting AutoGluon
        # reject the run after partial setup.
        "use_bag_holdout": True,
    }
    if runner_config.excluded_model_types:
        fit_kwargs["excluded_model_types"] = runner_config.excluded_model_types
    return fit_kwargs


def _build_comparison_llm_context(
    *,
    base_runner: TreehouseLabRunner,
    dataset: DatasetBundle,
    split: DatasetSplit,
    run_summaries: list[ComparisonRunSummary],
    loop_steps: int,
) -> dict[str, Any]:
    runner_payloads = [
        {
            "runner_key": run_summary.runner_key,
            "display_name": run_summary.display_name,
            "status": run_summary.status,
            "backend": run_summary.backend,
            "validation_metric": run_summary.validation_metric,
            "test_metric": run_summary.test_metric,
            "runtime_seconds": run_summary.runtime_seconds,
            "benchmark_status": run_summary.benchmark_status,
            "implementation_readiness": run_summary.implementation_readiness,
            "workflow_traits": run_summary.workflow_traits,
            "notes": run_summary.notes,
            "details": _llm_safe_runner_details(run_summary),
        }
        for run_summary in run_summaries
    ]
    return {
        "project_root": str(base_runner.project_root),
        "dataset_key": base_runner.registry_key,
        "dataset_name": base_runner.config.source.name or base_runner.registry_key,
        "target": dataset.target_name,
        "task_kind": dataset.target_profile["task_kind"],
        "primary_metric": base_runner.config.primary_metric,
        "requested_treehouse_loop_steps": loop_steps,
        "split_summary": split.summary(),
        "runners": runner_payloads,
    }


def _llm_safe_runner_details(run_summary: ComparisonRunSummary) -> dict[str, Any]:
    if run_summary.runner_key == "treehouse_lab_loop":
        return {
            "promotion_count": run_summary.details.get("promotion_count"),
            "stop_reason": run_summary.details.get("stop_reason"),
            "final_metric": run_summary.details.get("final_incumbent", {}).get("metric"),
            "llm_guided_step_count": run_summary.details.get("llm_guided_step_count"),
            "llm_reviewed_step_count": run_summary.details.get("llm_reviewed_step_count"),
            "llm_guidance_statuses": run_summary.details.get("llm_guidance_statuses"),
            "llm_provider": run_summary.details.get("llm_provider"),
        }
    if run_summary.runner_key == "autogluon_tabular":
        return {
            "profile": run_summary.details.get("profile"),
            "requested_presets": run_summary.details.get("requested_presets") or run_summary.details.get("presets"),
            "requested_time_limit": run_summary.details.get("requested_time_limit") or run_summary.details.get("time_limit"),
            "excluded_model_types": run_summary.details.get("excluded_model_types"),
            "install_hint": run_summary.details.get("install_hint"),
        }
    return {
        "target_name": run_summary.details.get("target_name"),
        "assessment": run_summary.details.get("assessment"),
    }


def _summarize_loop_llm_guidance(step_results: list[dict[str, Any]]) -> dict[str, Any]:
    reviews = []
    for step in step_results:
        proposal = step.get("proposal", {})
        review = proposal.get("llm_review", {})
        if isinstance(review, dict) and review.get("status"):
            reviews.append(review)
    guided_reviews = [review for review in reviews if review.get("status") == "available"]
    provider = None
    if guided_reviews:
        provider = guided_reviews[0].get("provider")
    elif reviews:
        provider = reviews[0].get("provider")
    statuses = sorted({str(review.get("status")) for review in reviews if review.get("status")})
    return {
        "llm_guided_step_count": len(guided_reviews),
        "llm_reviewed_step_count": len(reviews),
        "llm_guidance_statuses": statuses,
        "llm_provider": provider,
    }


def _format_loop_llm_guidance_note(llm_guidance: dict[str, Any]) -> str:
    reviewed_step_count = int(llm_guidance.get("llm_reviewed_step_count", 0))
    guided_step_count = int(llm_guidance.get("llm_guided_step_count", 0))
    provider = llm_guidance.get("llm_provider")
    if reviewed_step_count == 0:
        return "LLM-guided candidate selection was not used; Treehouse relied on deterministic candidate ranking."
    provider_suffix = f" via `{provider}`" if provider else ""
    return f"LLM-guided candidate selections: {guided_step_count}/{reviewed_step_count}{provider_suffix}."


def _render_practical_takeaway(run_summaries: list[ComparisonRunSummary]) -> list[str]:
    lines = [
        "- `autogluon_tabular` answers the one-shot automation question: what can a practical external AutoML pass produce on this split?",
        "- Treehouse Lab answers the operating question: what bounded move should come next, why, and what evidence justifies it?",
    ]
    autogluon_summary = next((summary for summary in run_summaries if summary.runner_key == "autogluon_tabular"), None)
    if autogluon_summary is not None:
        if autogluon_summary.status == "completed":
            profile = autogluon_summary.details.get("profile", "unknown")
            lines.append(f"- AutoGluon ran in the `{profile}` profile so the benchmark stays practical enough to rerun.")
        elif autogluon_summary.status == "unavailable":
            lines.append("- AutoGluon was unavailable in this environment, so the practical comparison is Treehouse Lab versus a plain XGBoost anchor.")
    loop_summary = next((summary for summary in run_summaries if summary.runner_key == "treehouse_lab_loop"), None)
    if loop_summary is None:
        return lines
    llm_guided_step_count = int(loop_summary.details.get("llm_guided_step_count", 0) or 0)
    llm_reviewed_step_count = int(loop_summary.details.get("llm_reviewed_step_count", 0) or 0)
    provider = loop_summary.details.get("llm_provider")
    if llm_reviewed_step_count == 0:
        lines.append("- The Treehouse loop stayed deterministic in this run, so the product difference is audit trail and bounded search rather than LLM-guided selection.")
        return lines
    provider_suffix = f" via `{provider}`" if provider else ""
    lines.append(
        f"- The Treehouse loop used LLM guidance on {llm_guided_step_count}/{llm_reviewed_step_count} bounded candidate choices{provider_suffix}, which is the key layer beyond plain AutoGluon."
    )
    return lines


def _prepare_isolated_workspace(base_runner: TreehouseLabRunner, workspace_root: Path) -> Path:
    config = base_runner.config
    raw = json.loads(json.dumps(config.raw))
    source_raw = raw.get("dataset", {}).get("source", {})
    source_path = source_raw.get("path")
    if source_path:
        resolved_source_path = Path(source_path)
        if not resolved_source_path.is_absolute():
            resolved_source_path = (base_runner.project_root / resolved_source_path).resolve()
        source_raw["path"] = str(resolved_source_path)

    workspace_config_dir = workspace_root / "configs" / "datasets"
    workspace_config_dir.mkdir(parents=True, exist_ok=True)
    workspace_search_space_dir = workspace_root / "configs"
    workspace_search_space_dir.mkdir(parents=True, exist_ok=True)
    workspace_config_path = workspace_config_dir / base_runner.config_path.name
    workspace_search_space_path = workspace_search_space_dir / "search_space.yaml"

    workspace_config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    workspace_search_space_path.write_text(
        (base_runner.project_root / "configs" / "search_space.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    return workspace_config_path


def _write_runner_outputs(runner_dir: Path, *, summary: dict[str, Any], notes: list[str]) -> None:
    runner_dir.mkdir(parents=True, exist_ok=True)
    (runner_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    (runner_dir / "notes.md").write_text("\n".join(f"- {note}" for note in notes), encoding="utf-8")


def _combine_features_and_target(X: pd.DataFrame, y: pd.Series, target_name: str) -> pd.DataFrame:
    frame = X.copy()
    frame[target_name] = y.to_numpy()
    return frame


def _baseline_style_assessment(
    config: ExperimentConfig,
    metrics: dict[str, float],
    split_summary: dict[str, Any],
    runtime_seconds: float,
) -> RunAssessment:
    return assess_run(
        config,
        metrics=metrics,
        split_summary=split_summary,
        runtime_seconds=runtime_seconds,
        comparison={"incumbent_metric": None, "delta": None, "threshold": config.promote_if_delta_at_least},
        promoted=True,
    )


def _autogluon_problem_type(task_kind: str) -> str:
    if task_kind == "multiclass_classification":
        return "multiclass"
    return "binary"


def _autogluon_eval_metric(primary_metric: str) -> str | None:
    supported_metrics = {"roc_auc", "accuracy", "log_loss"}
    return primary_metric if primary_metric in supported_metrics else None


class _AutoGluonPredictorAdapter:
    def __init__(self, predictor: Any):
        self.predictor = predictor

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        predictions = self.predictor.predict(frame)
        if hasattr(predictions, "to_numpy"):
            return np.asarray(predictions.to_numpy(), dtype=int)
        return np.asarray(predictions, dtype=int)

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        probabilities = self.predictor.predict_proba(frame)
        if hasattr(probabilities, "to_numpy"):
            values = np.asarray(probabilities.to_numpy())
        else:
            values = np.asarray(probabilities)
        if values.ndim == 1:
            values = np.column_stack([1 - values, values])
        return values.astype(float)
