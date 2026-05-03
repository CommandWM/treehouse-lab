from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from treehouse_lab.comparison import ComparisonSuiteResult, run_comparison_suite


@dataclass(slots=True)
class BenchmarkSuiteDataset:
    key: str
    config_path: Path
    loop_steps: int
    autogluon_profile: str
    fetch_command: str = ""
    autogluon_time_limit: int | None = None
    flaml_time_budget: int | None = None
    flaml_estimator_list: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["config_path"] = str(self.config_path)
        return payload


@dataclass(slots=True)
class BenchmarkSuiteConfig:
    key: str
    name: str
    description: str
    fixed_seed: int
    loop_steps: int
    autogluon_profile: str
    flaml_time_budget: int | None
    flaml_estimator_list: list[str]
    datasets: list[BenchmarkSuiteDataset]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["datasets"] = [dataset.to_dict() for dataset in self.datasets]
        return payload


@dataclass(slots=True)
class BenchmarkSuiteDatasetResult:
    key: str
    config_path: str
    status: str
    output_dir: str | None = None
    report_path: str | None = None
    summary_path: str | None = None
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class BenchmarkSuiteRunResult:
    suite_key: str
    suite_name: str
    output_dir: str
    completed_count: int
    failed_count: int
    datasets: list[BenchmarkSuiteDatasetResult]
    summary_path: str

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["datasets"] = [dataset.to_dict() for dataset in self.datasets]
        return payload


def load_benchmark_suite_config(suite_path: str | Path) -> BenchmarkSuiteConfig:
    resolved_suite_path = Path(suite_path).expanduser().resolve()
    raw = yaml.safe_load(resolved_suite_path.read_text(encoding="utf-8")) or {}
    suite_raw = raw.get("suite", {})
    suite_key = str(suite_raw.get("key") or resolved_suite_path.stem)
    loop_steps = int(suite_raw.get("loop_steps", 3))
    autogluon_profile = str(suite_raw.get("autogluon_profile", "practical"))
    flaml_time_budget = _optional_int(suite_raw.get("flaml_time_budget"))
    flaml_estimator_list = _normalize_string_list(suite_raw.get("flaml_estimator_list"))
    fixed_seed = int(suite_raw.get("fixed_seed", 42))
    datasets = [
        _load_suite_dataset(
            item,
            suite_dir=resolved_suite_path.parent,
            default_loop_steps=loop_steps,
            default_autogluon_profile=autogluon_profile,
            default_flaml_time_budget=flaml_time_budget,
            default_flaml_estimator_list=flaml_estimator_list,
        )
        for item in raw.get("datasets", [])
    ]
    if not datasets:
        msg = f"Benchmark suite `{suite_key}` must declare at least one dataset."
        raise ValueError(msg)
    return BenchmarkSuiteConfig(
        key=suite_key,
        name=str(suite_raw.get("name") or suite_key),
        description=str(suite_raw.get("description", "")),
        fixed_seed=fixed_seed,
        loop_steps=loop_steps,
        autogluon_profile=autogluon_profile,
        flaml_time_budget=flaml_time_budget,
        flaml_estimator_list=flaml_estimator_list,
        datasets=datasets,
    )


def run_benchmark_suite(
    suite_path: str | Path,
    *,
    output_dir: str | Path | None = None,
    include_autogluon: bool = True,
    include_flaml: bool = True,
    include_llm_summary: bool = False,
) -> BenchmarkSuiteRunResult:
    suite = load_benchmark_suite_config(suite_path)
    suite_output_dir = _resolve_suite_output_dir(suite, output_dir)
    suite_output_dir.mkdir(parents=True, exist_ok=True)

    dataset_results: list[BenchmarkSuiteDatasetResult] = []
    for dataset in suite.datasets:
        dataset_output_dir = suite_output_dir / dataset.key
        try:
            comparison_result = run_comparison_suite(
                dataset.config_path,
                output_dir=dataset_output_dir,
                loop_steps=dataset.loop_steps,
                include_autogluon=include_autogluon,
                include_flaml=include_flaml,
                include_llm_summary=include_llm_summary,
                autogluon_profile=dataset.autogluon_profile,
                autogluon_time_limit=dataset.autogluon_time_limit,
                flaml_time_budget=dataset.flaml_time_budget,
                flaml_estimator_list=dataset.flaml_estimator_list or None,
            )
        except Exception as exc:  # pragma: no cover - exercised through user data/runtime conditions
            dataset_results.append(
                BenchmarkSuiteDatasetResult(
                    key=dataset.key,
                    config_path=str(dataset.config_path),
                    status="error",
                    output_dir=str(dataset_output_dir),
                    error_message=str(exc),
                )
            )
            continue
        dataset_results.append(_dataset_result_from_comparison(dataset, comparison_result))

    completed_count = sum(1 for result in dataset_results if result.status == "completed")
    failed_count = len(dataset_results) - completed_count
    summary_path = suite_output_dir / "summary.json"
    suite_result = BenchmarkSuiteRunResult(
        suite_key=suite.key,
        suite_name=suite.name,
        output_dir=str(suite_output_dir),
        completed_count=completed_count,
        failed_count=failed_count,
        datasets=dataset_results,
        summary_path=str(summary_path),
    )
    summary_path.write_text(json.dumps(suite_result.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    return suite_result


def _load_suite_dataset(
    raw: dict[str, Any],
    *,
    suite_dir: Path,
    default_loop_steps: int,
    default_autogluon_profile: str,
    default_flaml_time_budget: int | None,
    default_flaml_estimator_list: list[str],
) -> BenchmarkSuiteDataset:
    key = str(raw["key"])
    config_path = Path(str(raw["config"])).expanduser()
    if not config_path.is_absolute():
        config_path = (suite_dir / config_path).resolve()
    flaml_estimator_list = _normalize_string_list(raw.get("flaml_estimator_list", default_flaml_estimator_list))
    return BenchmarkSuiteDataset(
        key=key,
        config_path=config_path,
        loop_steps=int(raw.get("loop_steps", default_loop_steps)),
        autogluon_profile=str(raw.get("autogluon_profile", default_autogluon_profile)),
        fetch_command=str(raw.get("fetch_command", "")),
        autogluon_time_limit=_optional_int(raw.get("autogluon_time_limit")),
        flaml_time_budget=_optional_int(raw.get("flaml_time_budget", default_flaml_time_budget)),
        flaml_estimator_list=flaml_estimator_list,
        notes=[str(note) for note in raw.get("notes", [])],
    )


def _optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    return int(value)


def _normalize_string_list(value: Any) -> list[str]:
    if value in (None, ""):
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [part.strip() for part in str(value).split(",") if part.strip()]


def _resolve_suite_output_dir(suite: BenchmarkSuiteConfig, output_dir: str | Path | None) -> Path:
    if output_dir is not None:
        return Path(output_dir).expanduser().resolve()
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return Path("outputs") / "benchmark_suites" / suite.key / timestamp


def _dataset_result_from_comparison(
    dataset: BenchmarkSuiteDataset,
    comparison_result: ComparisonSuiteResult,
) -> BenchmarkSuiteDatasetResult:
    return BenchmarkSuiteDatasetResult(
        key=dataset.key,
        config_path=str(dataset.config_path),
        status="completed",
        output_dir=comparison_result.output_dir,
        report_path=comparison_result.report_path,
        summary_path=comparison_result.summary_path,
    )
