from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from treehouse_lab.config import ExperimentConfig


@dataclass(slots=True)
class DatasetBundle:
    frame: pd.DataFrame
    target: pd.Series
    feature_frame: pd.DataFrame
    feature_names: list[str]
    target_name: str


@dataclass(slots=True)
class DatasetSplit:
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series

    def summary(self) -> dict[str, float | int]:
        return {
            "train_rows": int(len(self.X_train)),
            "validation_rows": int(len(self.X_val)),
            "test_rows": int(len(self.X_test)),
            "feature_count": int(self.X_train.shape[1]),
            "train_positive_rate": float(self.y_train.mean()),
            "validation_positive_rate": float(self.y_val.mean()),
            "test_positive_rate": float(self.y_test.mean()),
        }


def load_dataset(config: ExperimentConfig, project_root: Path) -> DatasetBundle:
    source = config.source
    if source.kind == "sklearn_breast_cancer":
        data = load_breast_cancer(as_frame=True)
        frame = data.frame.copy()
        target_name = data.target_names[1]
        target = frame.pop("target").astype(int)
        return DatasetBundle(
            frame=frame.copy(),
            target=target,
            feature_frame=frame,
            feature_names=list(frame.columns),
            target_name=target_name,
        )

    if source.kind == "synthetic_churn_demo":
        frame = build_synthetic_churn_demo(rows=source.rows, random_state=source.random_state)
        target_name = source.target_column or "churned"
        target = frame.pop(target_name).astype(int)
        feature_frame = pd.get_dummies(frame, drop_first=False)
        return DatasetBundle(
            frame=frame.copy(),
            target=target,
            feature_frame=feature_frame,
            feature_names=list(feature_frame.columns),
            target_name=target_name,
        )

    if source.kind == "csv":
        if not source.path or not source.target_column:
            msg = "CSV datasets require both path and target_column."
            raise ValueError(msg)
        csv_path = (project_root / source.path).resolve()
        frame = pd.read_csv(csv_path)
        target = frame.pop(source.target_column).astype(int)
        feature_frame = pd.get_dummies(frame, drop_first=False)
        return DatasetBundle(
            frame=frame.copy(),
            target=target,
            feature_frame=feature_frame,
            feature_names=list(feature_frame.columns),
            target_name=source.target_column,
        )

    msg = f"Unsupported dataset source kind: {source.kind}"
    raise ValueError(msg)


def split_dataset(bundle: DatasetBundle, config: ExperimentConfig) -> DatasetSplit:
    test_size = config.split.test_size
    validation_size = config.split.validation_size
    if test_size <= 0 or validation_size <= 0 or test_size + validation_size >= 1:
        msg = "validation_size and test_size must both be positive and sum to less than 1."
        raise ValueError(msg)

    stratify_values = bundle.target if config.split.stratify else None
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        bundle.feature_frame,
        bundle.target,
        test_size=test_size,
        random_state=config.seed,
        stratify=stratify_values,
    )

    validation_share_of_train_val = validation_size / (1 - test_size)
    stratify_train_val = y_train_val if config.split.stratify else None
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=validation_share_of_train_val,
        random_state=config.seed,
        stratify=stratify_train_val,
    )

    return DatasetSplit(
        X_train=X_train.reset_index(drop=True),
        X_val=X_val.reset_index(drop=True),
        X_test=X_test.reset_index(drop=True),
        y_train=y_train.reset_index(drop=True),
        y_val=y_val.reset_index(drop=True),
        y_test=y_test.reset_index(drop=True),
    )


def build_synthetic_churn_demo(rows: int, random_state: int) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)

    monthly_charges = np.round(rng.normal(loc=82, scale=24, size=rows).clip(25, 170), 2)
    tenure_months = rng.integers(1, 72, size=rows)
    support_tickets = rng.poisson(lam=1.8, size=rows)
    contract = rng.choice(["month-to-month", "one-year", "two-year"], size=rows, p=[0.56, 0.25, 0.19])
    internet_service = rng.choice(["fiber", "dsl", "none"], size=rows, p=[0.48, 0.38, 0.14])
    payment_method = rng.choice(
        ["credit-card", "bank-transfer", "mailed-check", "auto-debit"],
        size=rows,
        p=[0.3, 0.23, 0.17, 0.3],
    )
    auto_pay = rng.choice(["yes", "no"], size=rows, p=[0.62, 0.38])
    senior_citizen = rng.choice([0, 1], size=rows, p=[0.78, 0.22])

    risk = (
        -2.5
        + 0.018 * monthly_charges
        - 0.038 * tenure_months
        + 0.42 * support_tickets
        + 1.25 * (contract == "month-to-month")
        + 0.64 * (internet_service == "fiber")
        + 0.48 * (payment_method == "mailed-check")
        + 0.54 * (auto_pay == "no")
        + 0.33 * senior_citizen
    )
    probability = 1 / (1 + np.exp(-risk))
    churned = rng.binomial(1, probability)

    return pd.DataFrame(
        {
            "monthly_charges": monthly_charges,
            "tenure_months": tenure_months,
            "support_tickets_last_90d": support_tickets,
            "contract_type": contract,
            "internet_service": internet_service,
            "payment_method": payment_method,
            "auto_pay": auto_pay,
            "senior_citizen": senior_citizen,
            "churned": churned,
        }
    )
