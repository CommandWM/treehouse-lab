from __future__ import annotations

import json
from pathlib import Path

import yaml

from treehouse_lab.benchmark_suite import load_benchmark_suite_config, run_benchmark_suite
from treehouse_lab.comparison import ComparisonSuiteResult


def write_suite_fixture(tmp_path: Path) -> Path:
    config_dir = tmp_path / "configs" / "datasets"
    config_dir.mkdir(parents=True)
    for key in ("bankish", "adultish"):
        (config_dir / f"{key}.yaml").write_text("experiment:\n  seed: 42\n", encoding="utf-8")

    suite_path = tmp_path / "configs" / "benchmark_suites" / "public_v1_3.yaml"
    suite_path.parent.mkdir(parents=True)
    suite_path.write_text(
        yaml.safe_dump(
            {
                "suite": {
                    "key": "public_v1_3",
                    "name": "Public v1.3",
                    "description": "Fixture public benchmark suite.",
                    "fixed_seed": 42,
                    "loop_steps": 2,
                    "autogluon_profile": "practical",
                    "flaml_time_budget": 20,
                    "flaml_estimator_list": ["xgboost", "rf"],
                },
                "datasets": [
                    {
                        "key": "bankish",
                        "config": "../datasets/bankish.yaml",
                        "fetch_command": "python3 scripts/fetch_bankish.py",
                    },
                    {
                        "key": "adultish",
                        "config": "../datasets/adultish.yaml",
                        "loop_steps": 1,
                        "flaml_time_budget": 12,
                    },
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return suite_path


def test_load_benchmark_suite_config_resolves_defaults_and_paths(tmp_path: Path) -> None:
    suite_path = write_suite_fixture(tmp_path)

    suite = load_benchmark_suite_config(suite_path)

    assert suite.key == "public_v1_3"
    assert suite.fixed_seed == 42
    assert suite.datasets[0].key == "bankish"
    assert suite.datasets[0].config_path == (tmp_path / "configs" / "datasets" / "bankish.yaml").resolve()
    assert suite.datasets[0].loop_steps == 2
    assert suite.datasets[1].loop_steps == 1
    assert suite.datasets[0].autogluon_profile == "practical"
    assert suite.datasets[0].flaml_time_budget == 20
    assert suite.datasets[1].flaml_time_budget == 12
    assert suite.datasets[0].flaml_estimator_list == ["xgboost", "rf"]


def test_run_benchmark_suite_calls_comparison_harness_per_dataset(tmp_path: Path, monkeypatch) -> None:
    suite_path = write_suite_fixture(tmp_path)
    calls: list[dict[str, object]] = []

    def fake_run_comparison_suite(
        config_path: str | Path,
        *,
        output_dir: str | Path | None = None,
        loop_steps: int = 3,
        include_autogluon: bool = True,
        include_flaml: bool = True,
        include_llm_summary: bool = False,
        llm_question: str | None = None,
        autogluon_profile: str = "practical",
        autogluon_presets: str | list[str] | None = None,
        autogluon_time_limit: int | None = None,
        flaml_time_budget: int | None = None,
        flaml_estimator_list: str | list[str] | None = None,
    ) -> ComparisonSuiteResult:
        calls.append(
            {
                "config_path": Path(config_path),
                "output_dir": Path(output_dir or ""),
                "loop_steps": loop_steps,
                "include_autogluon": include_autogluon,
                "include_flaml": include_flaml,
                "include_llm_summary": include_llm_summary,
                "autogluon_profile": autogluon_profile,
                "flaml_time_budget": flaml_time_budget,
                "flaml_estimator_list": flaml_estimator_list,
            }
        )
        output_path = Path(output_dir or tmp_path / "missing")
        output_path.mkdir(parents=True, exist_ok=True)
        return ComparisonSuiteResult(
            dataset_key=Path(config_path).stem,
            config_path=str(config_path),
            output_dir=str(output_path),
            primary_metric="roc_auc",
            split_summary={"seed": 42},
            runners=[],
            llm_summary=None,
            report_path=str(output_path / "report.md"),
            summary_path=str(output_path / "summary.json"),
        )

    monkeypatch.setattr("treehouse_lab.benchmark_suite.run_comparison_suite", fake_run_comparison_suite)

    result = run_benchmark_suite(
        suite_path,
        output_dir=tmp_path / "outputs" / "suite",
        include_autogluon=False,
        include_flaml=True,
        include_llm_summary=True,
    )

    assert [call["loop_steps"] for call in calls] == [2, 1]
    assert all(call["include_autogluon"] is False for call in calls)
    assert all(call["include_flaml"] is True for call in calls)
    assert all(call["include_llm_summary"] is True for call in calls)
    assert [call["flaml_time_budget"] for call in calls] == [20, 12]
    assert all(call["flaml_estimator_list"] == ["xgboost", "rf"] for call in calls)
    assert calls[0]["output_dir"] == tmp_path / "outputs" / "suite" / "bankish"
    assert result.completed_count == 2
    assert result.failed_count == 0
    assert Path(result.summary_path).exists()
    summary = json.loads(Path(result.summary_path).read_text(encoding="utf-8"))
    assert [dataset["key"] for dataset in summary["datasets"]] == ["bankish", "adultish"]


def test_cli_exposes_benchmark_suite_command() -> None:
    from treehouse_lab.cli import build_parser

    args = build_parser().parse_args(
        [
            "benchmark-suite",
            "configs/benchmark_suites/public_v1_3.yaml",
            "--output-dir",
            "outputs/benchmark_suites/manual",
            "--skip-autogluon",
            "--skip-flaml",
            "--llm-summary",
        ]
    )

    assert args.command == "benchmark-suite"
    assert args.suite_config == Path("configs/benchmark_suites/public_v1_3.yaml")
    assert args.output_dir == Path("outputs/benchmark_suites/manual")
    assert args.skip_autogluon is True
    assert args.skip_flaml is True
    assert args.llm_summary is True
