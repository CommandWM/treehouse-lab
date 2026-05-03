from __future__ import annotations

import argparse
import json
from pathlib import Path

from treehouse_lab.benchmark_suite import run_benchmark_suite
from treehouse_lab.comparison import run_comparison_suite
from treehouse_lab.exporting import export_model_artifact
from treehouse_lab.loop import AutonomousLoopController
from treehouse_lab.runner import TreehouseLabRunner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Treehouse Lab experiments.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    baseline_parser = subparsers.add_parser("baseline", help="Run the baseline experiment for a dataset spec.")
    baseline_parser.add_argument("config", type=Path, help="Path to the dataset config YAML.")

    candidate_parser = subparsers.add_parser("candidate", help="Run one bounded candidate experiment.")
    candidate_parser.add_argument("config", type=Path, help="Path to the dataset config YAML.")
    candidate_parser.add_argument("--name", required=True, help="Human-readable mutation name.")
    candidate_parser.add_argument(
        "--set",
        dest="overrides",
        nargs="+",
        default=[],
        metavar="KEY=VALUE",
        help="Override model params, for example --set max_depth=4 learning_rate=0.08",
    )
    candidate_parser.add_argument("--hypothesis", default="", help="Short experiment hypothesis.")

    propose_parser = subparsers.add_parser("propose", help="Show the next bounded experiment proposal without executing it.")
    propose_parser.add_argument("config", type=Path, help="Path to the dataset config YAML.")

    diagnose_parser = subparsers.add_parser("diagnose", help="Show the current diagnosis and the next bounded proposal.")
    diagnose_parser.add_argument("config", type=Path, help="Path to the dataset config YAML.")

    loop_parser = subparsers.add_parser("loop", help="Run the bounded autonomous research loop.")
    loop_parser.add_argument("config", type=Path, help="Path to the dataset config YAML.")
    loop_parser.add_argument("--steps", type=int, default=3, help="Maximum number of bounded loop steps to run.")

    export_parser = subparsers.add_parser("export", help="Export the incumbent or a specific run as a reusable model bundle.")
    export_parser.add_argument("config", type=Path, help="Path to the dataset config YAML.")
    export_parser.add_argument("--run-id", default="", help="Optional run id to export instead of the current incumbent.")
    export_parser.add_argument("--output-dir", type=Path, default=None, help="Optional destination directory for the exported bundle.")

    compare_parser = subparsers.add_parser("compare", help="Run a side-by-side comparison harness for a dataset config.")
    compare_parser.add_argument("config", type=Path, help="Path to the dataset config YAML.")
    compare_parser.add_argument("--loop-steps", type=int, default=3, help="How many bounded Treehouse loop steps to run in the isolated comparison workspace.")
    compare_parser.add_argument("--output-dir", type=Path, default=None, help="Optional destination directory for comparison outputs.")
    compare_parser.add_argument("--skip-autogluon", action="store_true", help="Skip the optional AutoGluon runner.")
    compare_parser.add_argument("--skip-flaml", action="store_true", help="Skip the optional FLAML runner.")
    compare_parser.add_argument("--llm-summary", action="store_true", help="Ask the configured LLM to synthesize the comparison report.")
    compare_parser.add_argument("--llm-question", default=None, help="Optional question for the comparison synthesis step.")
    compare_parser.add_argument(
        "--autogluon-profile",
        default="practical",
        choices=["practical", "full"],
        help="AutoGluon benchmark profile. `practical` is faster and deployment-oriented; `full` is a heavier reference run.",
    )
    compare_parser.add_argument(
        "--autogluon-presets",
        default=None,
        help="Optional AutoGluon presets override. Use a comma-separated list such as good_quality,optimize_for_deployment.",
    )
    compare_parser.add_argument("--autogluon-time-limit", type=int, default=None, help="Optional AutoGluon time limit in seconds.")
    compare_parser.add_argument("--flaml-time-budget", type=int, default=None, help="Optional FLAML time budget in seconds.")
    compare_parser.add_argument(
        "--flaml-estimators",
        default=None,
        help="Optional FLAML estimator list override. Use a comma-separated list such as xgboost,rf,extra_tree.",
    )

    suite_parser = subparsers.add_parser("benchmark-suite", help="Run a fixed benchmark suite through the comparison harness.")
    suite_parser.add_argument("suite_config", type=Path, help="Path to the benchmark suite YAML.")
    suite_parser.add_argument("--output-dir", type=Path, default=None, help="Optional destination directory for suite outputs.")
    suite_parser.add_argument("--skip-autogluon", action="store_true", help="Skip optional AutoGluon runners for every dataset.")
    suite_parser.add_argument("--skip-flaml", action="store_true", help="Skip optional FLAML runners for every dataset.")
    suite_parser.add_argument("--llm-summary", action="store_true", help="Ask the configured LLM to synthesize each dataset comparison.")

    return parser


def parse_override_items(items: list[str]) -> dict[str, object]:
    overrides: dict[str, object] = {}
    for item in items:
        if "=" not in item:
            msg = f"Invalid override '{item}'. Use KEY=VALUE."
            raise ValueError(msg)
        key, raw_value = item.split("=", 1)
        overrides[key] = parse_value(raw_value)
    return overrides


def parse_value(value: str) -> object:
    lower_value = value.lower()
    if lower_value in {"true", "false"}:
        return lower_value == "true"
    for caster in (int, float):
        try:
            return caster(value)
        except ValueError:
            continue
    return value


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "baseline":
        runner = TreehouseLabRunner(args.config)
        result = runner.run_baseline()
    elif args.command == "candidate":
        runner = TreehouseLabRunner(args.config)
        result = runner.run_candidate(
            mutation_name=args.name,
            overrides=parse_override_items(args.overrides),
            hypothesis=args.hypothesis or None,
        )
    elif args.command == "propose":
        controller = AutonomousLoopController(args.config)
        result = controller.next_proposal().to_dict()
    elif args.command == "diagnose":
        controller = AutonomousLoopController(args.config)
        result = controller.diagnose().to_dict()
    elif args.command == "export":
        config_key = Path(args.config).expanduser().resolve().stem
        project_root = Path(args.config).expanduser().resolve().parents[2]
        result = export_model_artifact(
            project_root=project_root,
            config_key=config_key,
            run_id=args.run_id or None,
            output_dir=args.output_dir,
        )
    elif args.command == "compare":
        result = run_comparison_suite(
            args.config,
            output_dir=args.output_dir,
            loop_steps=args.loop_steps,
            include_autogluon=not args.skip_autogluon,
            include_flaml=not args.skip_flaml,
            include_llm_summary=args.llm_summary,
            llm_question=args.llm_question,
            autogluon_profile=args.autogluon_profile,
            autogluon_presets=args.autogluon_presets,
            autogluon_time_limit=args.autogluon_time_limit,
            flaml_time_budget=args.flaml_time_budget,
            flaml_estimator_list=args.flaml_estimators,
        )
    elif args.command == "benchmark-suite":
        result = run_benchmark_suite(
            args.suite_config,
            output_dir=args.output_dir,
            include_autogluon=not args.skip_autogluon,
            include_flaml=not args.skip_flaml,
            include_llm_summary=args.llm_summary,
        )
    else:
        controller = AutonomousLoopController(args.config)
        result = controller.run_loop(max_steps=args.steps).to_dict()

    if hasattr(result, "to_dict"):
        payload = result.to_dict()
    else:
        payload = result
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
