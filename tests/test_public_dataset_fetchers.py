from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_script_module(module_name: str, script_name: str):
    script_path = PROJECT_ROOT / "scripts" / script_name
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        msg = f"Unable to load {script_name}"
        raise RuntimeError(msg)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_normalize_adult_combines_train_and_test_and_cleans_labels() -> None:
    module = _load_script_module("fetch_adult", "fetch_adult.py")
    train_text = (
        "39, State-gov, 77516, Bachelors, 13, Never-married, Adm-clerical, Not-in-family, White, Male, 2174, 0, 40, United-States, <=50K\n"
        "50, ?, 83311, Bachelors, 13, Married-civ-spouse, Exec-managerial, Husband, White, Male, 0, 0, 13, United-States, >50K\n"
    )
    test_text = (
        "|1x3 Cross validator\n"
        "38, Private, 215646, HS-grad, 9, Divorced, Handlers-cleaners, Not-in-family, White, Male, 0, 0, 40, United-States, <=50K.\n"
    )

    frame = module.normalize_adult(train_text, test_text)

    assert list(frame.columns) == module.ADULT_COLUMNS
    assert len(frame) == 3
    assert frame.loc[0, "income"] == "<=50K"
    assert frame.loc[1, "income"] == ">50K"
    assert frame.loc[2, "income"] == "<=50K"
    assert pd.isna(frame.loc[1, "workclass"])


def test_normalize_covertype_assigns_expected_column_names() -> None:
    module = _load_script_module("fetch_covertype", "fetch_covertype.py")
    row_one = ",".join(["1"] * len(module.COVERTYPE_COLUMNS))
    row_two = ",".join(["2"] * len(module.COVERTYPE_COLUMNS))

    frame = module.normalize_covertype(f"{row_one}\n{row_two}\n")

    assert list(frame.columns) == module.COVERTYPE_COLUMNS
    assert len(frame) == 2
    assert frame.loc[0, "cover_type"] == 1
    assert frame.loc[1, "cover_type"] == 2
