from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

from treehouse_lab.runner import TreehouseLabRunner


def write_feature_generation_fixture(tmp_path: Path) -> Path:
    dataset_path = tmp_path / "marketing_signups.csv"
    frame = pd.DataFrame(
        {
            "visits": [1, 2, 3, 4, 7, 8, 9, 10, 2, 3, 6, 7, 11, 12, 5, 9],
            "emails_opened": [0, 1, 1, 2, 4, 5, 4, 6, 0, 1, 3, 4, 6, 7, 2, 5],
            "days_since_signup": [21, 18, 15, 12, 8, 6, 5, 3, 20, 17, 10, 9, 4, 2, 11, 7],
            "segment": [
                "organic",
                "organic",
                "organic",
                "paid",
                "paid",
                "paid",
                "partner",
                "partner",
                "organic",
                "organic",
                "paid",
                "paid",
                "partner",
                "partner",
                "paid",
                "partner",
            ],
            "converted": [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1],
        }
    )
    frame.to_csv(dataset_path, index=False)

    config_dir = tmp_path / "configs" / "datasets"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "marketing-feature-audit.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "dataset": {
                    "source": {
                        "kind": "csv",
                        "name": "Marketing Signups",
                        "target_column": "converted",
                        "path": "marketing_signups.csv",
                    },
                    "split": {
                        "validation_size": 0.25,
                        "test_size": 0.25,
                        "stratify": True,
                    },
                },
                "benchmark": {
                    "pack": "user",
                    "profile": "dataset_intake",
                    "objective": "Audit the bounded feature-generation branch.",
                },
                "evaluation_policy": {
                    "require_promotion_for_readiness": True,
                },
                "experiment": {
                    "name": "marketing-feature-audit",
                    "description": "Exercise bounded feature generation audit artifacts.",
                    "primary_metric": "roc_auc",
                    "promote_if_delta_at_least": 0.003,
                    "max_runtime_minutes": 10,
                    "seed": 42,
                    "baseline_hypothesis": "A bounded feature branch should stay auditable.",
                },
                "model": {
                    "params": {
                        "n_estimators": 80,
                        "max_depth": 3,
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


def test_runner_persists_feature_generation_audit_trail(tmp_path: Path) -> None:
    config_path = write_feature_generation_fixture(tmp_path)
    runner = TreehouseLabRunner(config_path)

    result = runner.run_candidate(
        mutation_name="feature-generation-enable",
        overrides={},
        hypothesis="Try a small train-only interaction branch.",
        feature_generation={
            "enabled": True,
            "strategy": "train_only_supervised_numeric_interactions",
            "max_new_features": 4,
            "top_k_numeric": 3,
            "operations": ["square", "product"],
        },
    )

    assert result.feature_generation["enabled"] is True
    assert result.feature_generation["applied"] is True
    assert result.feature_generation["generated_feature_count"] > 0

    artifact_dir = Path(result.artifact_dir)
    feature_generation_path = artifact_dir / "feature_generation.json"
    assert feature_generation_path.exists()

    payload = json.loads(feature_generation_path.read_text(encoding="utf-8"))
    assert payload["generated_feature_count"] == result.feature_generation["generated_feature_count"]
    assert payload["generated_feature_specs"]

    summary_text = (artifact_dir / "summary.md").read_text(encoding="utf-8")
    assert "## Generated features" in summary_text
    assert "feature_generation_strategy" in summary_text

    incumbent_path = tmp_path / "runs" / "incumbents" / "marketing-feature-audit.json"
    incumbent = json.loads(incumbent_path.read_text(encoding="utf-8"))
    assert incumbent["feature_generation"]["generated_feature_count"] == result.feature_generation["generated_feature_count"]
