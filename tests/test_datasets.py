from __future__ import annotations

import pandas as pd
import pytest

from treehouse_lab.datasets import (
    _validate_stratified_split_feasibility,
    inspect_binary_target,
    inspect_classification_target,
    normalize_binary_target,
    normalize_classification_target,
)


def test_normalize_binary_target_supports_semantic_labels() -> None:
    series = pd.Series(["yes", "no", "yes", "no"], name="churned")

    encoded, profile = normalize_binary_target(series, "churned")

    assert encoded.tolist() == [1, 0, 1, 0]
    assert profile["mapping_mode"] == "semantic"
    assert profile["class_labels"] == [
        {"raw": "no", "encoded": 0},
        {"raw": "yes", "encoded": 1},
    ]


def test_normalize_binary_target_supports_numeric_labels() -> None:
    series = pd.Series([2, 1, 2, 1], name="defaulted")

    encoded, profile = normalize_binary_target(series, "defaulted")

    assert encoded.tolist() == [1, 0, 1, 0]
    assert profile["mapping_mode"] == "numeric"
    assert profile["class_labels"] == [
        {"raw": "1", "encoded": 0},
        {"raw": "2", "encoded": 1},
    ]


def test_normalize_binary_target_rejects_non_binary_labels() -> None:
    series = pd.Series(["win", "loss", "draw"], name="outcome")

    with pytest.raises(ValueError, match="must contain exactly 2 distinct labels"):
        normalize_binary_target(series, "outcome")


def test_inspect_binary_target_reports_counts() -> None:
    series = pd.Series([True, False, True, True], name="converted")

    profile = inspect_binary_target(series, "converted")

    assert profile["mapping_mode"] == "boolean"
    assert profile["positive_count"] == 3
    assert profile["negative_count"] == 1
    assert profile["row_count"] == 4


def test_normalize_classification_target_supports_multiclass_labels() -> None:
    series = pd.Series(["low", "medium", "high", "medium"], name="burnout_level")

    encoded, profile = normalize_classification_target(series, "burnout_level")

    assert encoded.tolist() == [1, 2, 0, 2]
    assert profile["task_kind"] == "multiclass_classification"
    assert profile["class_count"] == 3
    assert profile["class_labels"] == [
        {"raw": "high", "encoded": 0},
        {"raw": "low", "encoded": 1},
        {"raw": "medium", "encoded": 2},
    ]


def test_inspect_classification_target_reports_multiclass_shape() -> None:
    series = pd.Series(["low", "medium", "high", "medium"], name="burnout_level")

    profile = inspect_classification_target(series, "burnout_level")

    assert profile["binary_supported"] is False
    assert profile["multiclass_supported"] is True
    assert profile["class_count"] == 3
    assert profile["class_counts"] == {"0": 1, "1": 1, "2": 2}


def test_validate_stratified_split_feasibility_allows_balanced_data() -> None:
    target = pd.Series([0] * 10 + [1] * 10, name="churned")

    _validate_stratified_split_feasibility(target, test_size=0.2, validation_size=0.2)


def test_validate_stratified_split_feasibility_rejects_underrepresented_class() -> None:
    target = pd.Series([0] * 20 + [1], name="churned")

    with pytest.raises(ValueError, match="Stratified split is not feasible"):
        _validate_stratified_split_feasibility(target, test_size=0.2, validation_size=0.2)


def test_validate_stratified_split_feasibility_allows_small_but_feasible_minority() -> None:
    target = pd.Series([0] * 20 + [1] * 4, name="churned")

    _validate_stratified_split_feasibility(target, test_size=0.2, validation_size=0.2)
