from __future__ import annotations

import argparse
import json
from pathlib import Path

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
    runner = TreehouseLabRunner(args.config)

    if args.command == "baseline":
        result = runner.run_baseline()
    else:
        result = runner.run_candidate(
            mutation_name=args.name,
            overrides=parse_override_items(args.overrides),
            hypothesis=args.hypothesis or None,
        )

    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
