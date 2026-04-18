from __future__ import annotations

from dataclasses import asdict, dataclass
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
    target_name: str
    target_profile: dict[str, object]


@dataclass(slots=True)
class DatasetSplit:
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series
    preprocessor: FeaturePreprocessor

    def summary(self) -> dict[str, float | int]:
        summary: dict[str, float | int | dict[str, float]] = {
            "train_rows": int(len(self.X_train)),
            "validation_rows": int(len(self.X_val)),
            "test_rows": int(len(self.X_test)),
            "feature_count": int(self.X_train.shape[1]),
            "class_count": int(pd.concat([self.y_train, self.y_val, self.y_test], ignore_index=True).nunique()),
            "train_class_distribution": _class_distribution(self.y_train),
            "validation_class_distribution": _class_distribution(self.y_val),
            "test_class_distribution": _class_distribution(self.y_test),
        }
        if summary["class_count"] == 2:
            summary["train_positive_rate"] = float(self.y_train.mean())
            summary["validation_positive_rate"] = float(self.y_val.mean())
            summary["test_positive_rate"] = float(self.y_test.mean())
        return summary


@dataclass(slots=True)
class FeaturePreprocessor:
    input_columns: list[str]
    numeric_columns: list[str]
    categorical_columns: list[str]
    fill_values: dict[str, float]
    categorical_feature_names: list[str]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def load_dataset(config: ExperimentConfig, project_root: Path) -> DatasetBundle:
    source = config.source
    if source.kind == "sklearn_breast_cancer":
        data = load_breast_cancer(as_frame=True)
        frame = data.frame.copy()
        target_name = data.target_names[1]
        target = frame.pop("target").astype(int)
        target_profile = build_target_profile(
            target,
            target_name="target",
            class_labels=[{"raw": name, "encoded": index} for index, name in enumerate(data.target_names)],
            mapping_mode="dataset",
            task_kind="binary_classification",
        )
        return DatasetBundle(
            frame=frame.copy(),
            target=target,
            target_name=target_name,
            target_profile=target_profile,
        )

    if source.kind == "synthetic_churn_demo":
        frame = build_synthetic_churn_demo(
            rows=source.rows,
            random_state=source.random_state,
            variant=source.variant or "implementation_like",
        )
        target_name = source.target_column or "churned"
        target = frame.pop(target_name).astype(int)
        target_profile = build_target_profile(
            target,
            target_name=target_name,
            class_labels=[
                {"raw": "not_churned", "encoded": 0},
                {"raw": "churned", "encoded": 1},
            ],
            mapping_mode="dataset",
            task_kind="binary_classification",
        )
        return DatasetBundle(
            frame=frame.copy(),
            target=target,
            target_name=target_name,
            target_profile=target_profile,
        )

    if source.kind == "csv":
        if not source.path or not source.target_column:
            msg = "CSV datasets require both path and target_column."
            raise ValueError(msg)
        csv_path = (project_root / source.path).resolve()
        frame = pd.read_csv(csv_path)
        raw_target = frame.pop(source.target_column)
        target, target_profile = normalize_classification_target(raw_target, source.target_column, config.task.kind)
        return DatasetBundle(
            frame=frame.copy(),
            target=target,
            target_name=source.target_column,
            target_profile=target_profile,
        )

    msg = f"Unsupported dataset source kind: {source.kind}"
    raise ValueError(msg)


def split_dataset(bundle: DatasetBundle, config: ExperimentConfig) -> DatasetSplit:
    test_size = config.split.test_size
    validation_size = config.split.validation_size
    if test_size <= 0 or validation_size <= 0 or test_size + validation_size >= 1:
        msg = "validation_size and test_size must both be positive and sum to less than 1."
        raise ValueError(msg)
    if config.split.stratify:
        _validate_stratified_split_feasibility(
            bundle.target,
            test_size=test_size,
            validation_size=validation_size,
        )

    stratify_values = bundle.target if config.split.stratify else None
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        bundle.frame,
        bundle.target,
        test_size=test_size,
        random_state=config.seed,
        stratify=stratify_values,
    )

    validation_share_of_train_val = validation_size / (1 - test_size)
    stratify_train_val = y_train_val if config.split.stratify else None
    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=validation_share_of_train_val,
        random_state=config.seed,
        stratify=stratify_train_val,
    )
    preprocessor = fit_feature_preprocessor(X_train_raw)
    X_train = transform_feature_frame(X_train_raw, preprocessor)
    X_val = transform_feature_frame(X_val_raw, preprocessor)
    X_test = transform_feature_frame(X_test, preprocessor)

    return DatasetSplit(
        X_train=X_train.reset_index(drop=True),
        X_val=X_val.reset_index(drop=True),
        X_test=X_test.reset_index(drop=True),
        y_train=y_train.reset_index(drop=True),
        y_val=y_val.reset_index(drop=True),
        y_test=y_test.reset_index(drop=True),
        preprocessor=preprocessor,
    )


def _validate_stratified_split_feasibility(
    target: pd.Series,
    *,
    test_size: float,
    validation_size: float,
) -> None:
    class_counts = target.value_counts()
    class_distribution = {str(label): int(count) for label, count in class_counts.sort_index().items()}
    formatted_distribution = ", ".join(f"{label}:{count}" for label, count in class_distribution.items())
    row_index = np.arange(len(target))
    try:
        _, _, y_train_val, _ = train_test_split(
            row_index,
            target,
            test_size=test_size,
            random_state=0,
            stratify=target,
        )
        validation_share_of_train_val = validation_size / (1 - test_size)
        train_test_split(
            np.arange(len(y_train_val)),
            y_train_val,
            test_size=validation_share_of_train_val,
            random_state=0,
            stratify=y_train_val,
        )
    except ValueError as exc:
        msg = (
            "Stratified split is not feasible for the requested train/validation/test fractions. "
            f"split.test_size={test_size:.3f}, split.validation_size={validation_size:.3f}. "
            f"Class counts: {formatted_distribution}. "
            f"Original sklearn error: {exc}"
        )
        raise ValueError(msg) from exc


def normalize_classification_target(
    series: pd.Series,
    column_name: str,
    task_kind: str = "auto",
) -> tuple[pd.Series, dict[str, object]]:
    if series.isna().any():
        msg = f"Target column '{column_name}' contains missing values."
        raise ValueError(msg)

    unique_values = list(pd.unique(series))
    if len(unique_values) < 2:
        msg = f"Target column '{column_name}' must contain at least 2 distinct labels for classification."
        raise ValueError(msg)

    resolved_task_kind = _resolve_task_kind(task_kind, unique_values, column_name)
    if resolved_task_kind == "binary_classification":
        mapping, mapping_mode = _build_binary_label_mapping(unique_values)
    else:
        mapping, mapping_mode = _build_multiclass_label_mapping(unique_values)

    encoded = series.map(mapping)
    if encoded.isna().any():
        msg = f"Failed to encode target column '{column_name}' as classification labels."
        raise ValueError(msg)

    normalized = encoded.astype(int).reset_index(drop=True)
    return normalized, build_target_profile(
        normalized,
        target_name=column_name,
        class_labels=[
            {"raw": _stringify_label(raw_value), "encoded": int(encoded_value)}
            for raw_value, encoded_value in sorted(mapping.items(), key=lambda item: item[1])
        ],
        mapping_mode=mapping_mode,
        task_kind=resolved_task_kind,
    )


def inspect_binary_target(series: pd.Series, column_name: str) -> dict[str, object]:
    normalized, profile = normalize_classification_target(series, column_name, task_kind="binary_classification")
    profile["row_count"] = int(len(series))
    profile["positive_count"] = int(normalized.sum())
    profile["negative_count"] = int(len(normalized) - normalized.sum())
    return profile


def normalize_binary_target(series: pd.Series, column_name: str) -> tuple[pd.Series, dict[str, object]]:
    return normalize_classification_target(series, column_name, task_kind="binary_classification")


def inspect_classification_target(series: pd.Series, column_name: str) -> dict[str, object]:
    normalized, profile = normalize_classification_target(series, column_name, task_kind="auto")
    profile["row_count"] = int(len(series))
    if profile["class_count"] == 2:
        profile["positive_count"] = int(normalized.sum())
        profile["negative_count"] = int(len(normalized) - normalized.sum())
        profile["binary_supported"] = True
        profile["multiclass_supported"] = False
    else:
        profile["binary_supported"] = False
        profile["multiclass_supported"] = True
        profile["class_counts"] = {
            str(label): int(count)
            for label, count in normalized.value_counts(sort=False).sort_index().items()
        }
    return profile


def prepare_feature_frames(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit preprocessing on train only to avoid split leakage."""

    preprocessor = fit_feature_preprocessor(X_train)
    return (
        transform_feature_frame(X_train, preprocessor),
        transform_feature_frame(X_val, preprocessor),
        transform_feature_frame(X_test, preprocessor),
    )


def fit_feature_preprocessor(frame: pd.DataFrame) -> FeaturePreprocessor:
    train_frame = frame.reset_index(drop=True).copy()
    categorical_columns = list(train_frame.select_dtypes(include=["object", "category", "bool"]).columns)
    numeric_columns = [column for column in train_frame.columns if column not in categorical_columns]

    train_numeric = _prepare_numeric_frame(train_frame, numeric_columns)
    fill_values = {
        str(column): float(value)
        for column, value in train_numeric.median(numeric_only=True).fillna(0.0).items()
    }
    train_categorical = _prepare_categorical_frame(train_frame, categorical_columns)

    return FeaturePreprocessor(
        input_columns=list(train_frame.columns),
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        fill_values=fill_values,
        categorical_feature_names=list(train_categorical.columns),
    )


def transform_feature_frame(frame: pd.DataFrame, preprocessor: FeaturePreprocessor) -> pd.DataFrame:
    feature_frame = frame.reset_index(drop=True).copy()
    missing_columns = [column for column in preprocessor.input_columns if column not in feature_frame.columns]
    if missing_columns:
        msg = f"Missing required feature columns: {', '.join(missing_columns)}"
        raise ValueError(msg)

    feature_frame = feature_frame.loc[:, preprocessor.input_columns]
    numeric_frame = _prepare_numeric_frame(feature_frame, preprocessor.numeric_columns)
    numeric_frame = numeric_frame.reindex(columns=preprocessor.numeric_columns, fill_value=0.0)
    if preprocessor.numeric_columns:
        fill_values = pd.Series(preprocessor.fill_values)
        numeric_frame = numeric_frame.fillna(fill_values)

    categorical_frame = _prepare_categorical_frame(feature_frame, preprocessor.categorical_columns).reindex(
        columns=preprocessor.categorical_feature_names,
        fill_value=0,
    )

    return pd.concat([numeric_frame, categorical_frame], axis=1)


def _prepare_numeric_frame(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    if not columns:
        return pd.DataFrame(index=frame.index)
    return frame.loc[:, columns].apply(pd.to_numeric, errors="coerce")


def _prepare_categorical_frame(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    if not columns:
        return pd.DataFrame(index=frame.index)
    normalized = frame.loc[:, columns].fillna("__missing__").astype("string")
    return pd.get_dummies(normalized, drop_first=False, dtype=int)


def build_target_profile(
    target: pd.Series,
    target_name: str,
    class_labels: list[dict[str, object]],
    mapping_mode: str,
    task_kind: str,
) -> dict[str, object]:
    profile: dict[str, object] = {
        "column": target_name,
        "task_kind": task_kind,
        "class_count": int(target.nunique()),
        "class_labels": class_labels,
        "mapping_mode": mapping_mode,
    }
    if task_kind == "binary_classification":
        profile["positive_rate"] = float(target.mean())
    return profile


def _class_distribution(series: pd.Series) -> dict[str, float]:
    counts = series.value_counts(normalize=True, sort=False).sort_index()
    return {str(label): float(rate) for label, rate in counts.items()}


def _resolve_task_kind(task_kind: str, unique_values: list[object], column_name: str) -> str:
    normalized_task_kind = task_kind.strip().casefold()
    if normalized_task_kind == "auto":
        return "binary_classification" if len(unique_values) == 2 else "multiclass_classification"
    if normalized_task_kind == "binary_classification":
        if len(unique_values) != 2:
            msg = f"Target column '{column_name}' must contain exactly 2 distinct labels for binary classification."
            raise ValueError(msg)
        return normalized_task_kind
    if normalized_task_kind == "multiclass_classification":
        if len(unique_values) < 3:
            msg = f"Target column '{column_name}' must contain at least 3 distinct labels for multiclass classification."
            raise ValueError(msg)
        return normalized_task_kind
    msg = f"Unsupported task kind: {task_kind}"
    raise ValueError(msg)


def _build_binary_label_mapping(unique_values: list[object]) -> tuple[dict[object, int], str]:
    if all(isinstance(value, (bool, np.bool_)) for value in unique_values):
        return {False: 0, True: 1}, "boolean"

    numeric_values = pd.to_numeric(pd.Series(unique_values), errors="coerce")
    if not numeric_values.isna().any():
        ordered_pairs = sorted(
            zip(unique_values, numeric_values.tolist(), strict=True),
            key=lambda item: float(item[1]),
        )
        return {ordered_pairs[0][0]: 0, ordered_pairs[1][0]: 1}, "numeric"

    positive_tokens = {
        "1",
        "true",
        "yes",
        "y",
        "positive",
        "pos",
        "churn",
        "churned",
        "default",
        "fraud",
        "bad",
        "failure",
        "failed",
        "win",
        "converted",
    }
    negative_tokens = {
        "0",
        "false",
        "no",
        "n",
        "negative",
        "neg",
        "retain",
        "retained",
        "good",
        "pass",
        "passed",
        "loss",
        "lost",
    }
    normalized_labels = {value: _normalize_label_token(value) for value in unique_values}
    positive_candidates = [value for value, token in normalized_labels.items() if token in positive_tokens]
    negative_candidates = [value for value, token in normalized_labels.items() if token in negative_tokens]
    if len(positive_candidates) == 1 and len(negative_candidates) == 1:
        return {
            negative_candidates[0]: 0,
            positive_candidates[0]: 1,
        }, "semantic"

    ordered_values = sorted(unique_values, key=lambda value: _normalize_label_token(value))
    return {ordered_values[0]: 0, ordered_values[1]: 1}, "lexical"


def _build_multiclass_label_mapping(unique_values: list[object]) -> tuple[dict[object, int], str]:
    numeric_values = pd.to_numeric(pd.Series(unique_values), errors="coerce")
    if not numeric_values.isna().any():
        ordered_pairs = sorted(
            zip(unique_values, numeric_values.tolist(), strict=True),
            key=lambda item: float(item[1]),
        )
        return {raw_value: index for index, (raw_value, _) in enumerate(ordered_pairs)}, "numeric"

    ordered_values = sorted(unique_values, key=lambda value: _normalize_label_token(value))
    return {raw_value: index for index, raw_value in enumerate(ordered_values)}, "lexical"


def _normalize_label_token(value: object) -> str:
    return str(value).strip().casefold().replace("_", " ").replace("-", " ")


def _stringify_label(value: object) -> str:
    if pd.isna(value):
        return "null"
    return str(value)


def build_synthetic_churn_demo(rows: int, random_state: int, variant: str = "implementation_like") -> pd.DataFrame:
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

    frame = pd.DataFrame(
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
    if variant == "stress":
        frame = apply_stress_variant(frame, rng)
    elif variant not in {"implementation_like", "default"}:
        msg = f"Unsupported synthetic churn variant: {variant}"
        raise ValueError(msg)
    return frame


def apply_stress_variant(frame: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    stressed = frame.copy()

    # Make the benchmark intentionally messier: more missingness, a few rare categories,
    # and extra weak-signal columns that should not justify broad complexity.
    numeric_missing_columns = ["monthly_charges", "tenure_months", "support_tickets_last_90d"]
    categorical_missing_columns = ["payment_method", "internet_service", "contract_type"]
    for column in numeric_missing_columns:
        mask = rng.random(len(stressed)) < 0.10
        stressed.loc[mask, column] = np.nan
    for column in categorical_missing_columns:
        mask = rng.random(len(stressed)) < 0.08
        stressed.loc[mask, column] = None

    rare_contract_mask = rng.random(len(stressed)) < 0.04
    stressed.loc[rare_contract_mask, "contract_type"] = "prepaid"
    rare_service_mask = rng.random(len(stressed)) < 0.03
    stressed.loc[rare_service_mask, "internet_service"] = "satellite"

    stressed["marketing_segment"] = rng.choice(
        ["organic", "paid-search", "partner", "field-sales", "unknown"],
        size=len(stressed),
        p=[0.34, 0.24, 0.17, 0.09, 0.16],
    )
    stressed["region"] = rng.choice(["north", "south", "east", "west", "central"], size=len(stressed))
    stressed["survey_score"] = np.round(rng.normal(loc=7.1, scale=1.6, size=len(stressed)).clip(1, 10), 1)

    label_noise_mask = rng.random(len(stressed)) < 0.04
    stressed.loc[label_noise_mask, "churned"] = 1 - stressed.loc[label_noise_mask, "churned"]
    return stressed
