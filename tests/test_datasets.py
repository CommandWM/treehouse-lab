from __future__ import annotations

import pandas as pd
import pytest

from treehouse_lab.datasets import inspect_binary_target, normalize_binary_target


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
