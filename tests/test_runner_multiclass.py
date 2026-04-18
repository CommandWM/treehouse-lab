from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from treehouse_lab.runner import TreehouseLabRunner


def write_multiclass_fixture(tmp_path: Path) -> Path:
    dataset_path = tmp_path / "burnout_levels.csv"
    frame = pd.DataFrame(
        {
            "hours_per_week": [35, 40, 55, 60, 30, 42, 58, 62, 33, 44, 57, 63],
            "remote_days": [5, 4, 1, 0, 5, 3, 1, 0, 4, 2, 1, 0],
            "team": [
                "platform",
                "platform",
                "infra",
                "infra",
                "product",
                "product",
                "infra",
                "infra",
                "product",
                "platform",
                "infra",
                "infra",
            ],
            "burnout_level": [
                "low",
                "low",
                "high",
                "high",
                "medium",
                "medium",
                "high",
                "high",
                "low",
                "medium",
                "medium",
                "high",
            ],
        }
    )
    frame.to_csv(dataset_path, index=False)

    config_dir = tmp_path / "configs" / "datasets"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "burnout-multiclass.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "dataset": {
                    "source": {
                        "kind": "csv",
                        "name": "Developer Burnout Levels",
                        "target_column": "burnout_level",
                        "path": "burnout_levels.csv",
                    },
                    "split": {
                        "validation_size": 0.25,
                        "test_size": 0.25,
                        "stratify": True,
                    },
                },
                "task": {
                    "kind": "multiclass_classification",
                },
                "benchmark": {
                    "pack": "user",
                    "profile": "dataset_intake",
                    "objective": "Establish a multiclass XGBoost baseline for developer burnout levels.",
                },
                "evaluation_policy": {
                    "require_promotion_for_readiness": True,
                },
                "experiment": {
                    "name": "burnout-multiclass-baseline",
                    "description": "Developer burnout level prediction.",
                    "primary_metric": "accuracy",
                    "promote_if_delta_at_least": 0.003,
                    "max_runtime_minutes": 10,
                    "seed": 42,
                    "baseline_hypothesis": "A disciplined XGBoost baseline should establish a credible multiclass incumbent.",
                },
                "model": {
                    "params": {
                        "n_estimators": 120,
                        "max_depth": 4,
                        "learning_rate": 0.08,
                        "min_child_weight": 1,
                        "subsample": 0.9,
                        "colsample_bytree": 0.8,
                        "reg_alpha": 0.0,
                        "reg_lambda": 1.0,
                        "gamma": 0.0,
                        "tree_method": "hist",
                        "n_jobs": 4,
                    }
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return config_path


def test_runner_supports_multiclass_baselines(tmp_path: Path) -> None:
    config_path = write_multiclass_fixture(tmp_path)

    runner = TreehouseLabRunner(config_path)
    result = runner.run_baseline()

    assert result.promoted is True
    assert result.params["objective"] == "multi:softprob"
    assert result.params["num_class"] == 3
    assert "accuracy" in result.metrics
    assert "macro_f1" in result.metrics
    assert "roc_auc" not in result.metrics
    assert result.split_summary["class_count"] == 3
    assert result.assessment["benchmark_status"] == "baseline_established"
