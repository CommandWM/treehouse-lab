from __future__ import annotations

import argparse
from io import StringIO
from pathlib import Path
import ssl
from urllib.request import urlopen

import pandas as pd


UCI_ADULT_TRAIN_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
UCI_ADULT_TEST_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"
ADULT_COLUMNS = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education_num",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
    "native_country",
    "income",
]


def _download_text(url: str, *, verify_ssl: bool = True) -> str:
    try:
        import requests

        response = requests.get(url, timeout=60, verify=verify_ssl)
        response.raise_for_status()
        return response.text
    except ImportError:
        context = None
        if verify_ssl:
            try:
                import certifi

                context = ssl.create_default_context(cafile=certifi.where())
            except ImportError:
                context = ssl.create_default_context()
        else:
            context = ssl._create_unverified_context()  # noqa: SLF001

        with urlopen(url, context=context) as response:
            return response.read().decode("utf-8")


def normalize_adult(train_text: str, test_text: str) -> pd.DataFrame:
    train_frame = pd.read_csv(
        StringIO(train_text),
        names=ADULT_COLUMNS,
        header=None,
        skipinitialspace=True,
        na_values=["?"],
    )
    test_lines = [
        line
        for line in test_text.splitlines()
        if line.strip() and not line.lstrip().startswith("|")
    ]
    test_frame = pd.read_csv(
        StringIO("\n".join(test_lines)),
        names=ADULT_COLUMNS,
        header=None,
        skipinitialspace=True,
        na_values=["?"],
    )
    combined = pd.concat([train_frame, test_frame], ignore_index=True)
    combined = combined.dropna(how="all").reset_index(drop=True)

    for column in combined.select_dtypes(include=["object", "string"]).columns:
        cleaned = combined[column].astype("string").str.strip()
        combined[column] = cleaned.replace({"?": pd.NA, "": pd.NA})

    combined["income"] = combined["income"].astype("string").str.rstrip(".")
    return combined


def fetch_adult(output_path: Path, *, verify_ssl: bool = True) -> Path:
    train_text = _download_text(UCI_ADULT_TRAIN_URL, verify_ssl=verify_ssl)
    test_text = _download_text(UCI_ADULT_TEST_URL, verify_ssl=verify_ssl)
    frame = normalize_adult(train_text, test_text)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch the UCI Adult dataset, combine the official train/test files into one CSV, "
            "and normalize it to the CSV shape used by Treehouse Lab."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("custom_datasets/adult.csv"),
        help="Where to write the normalized CSV. Default: custom_datasets/adult.csv",
    )
    parser.add_argument(
        "--insecure",
        action="store_true",
        help="Disable SSL certificate verification as a last resort if the local Python trust store is misconfigured.",
    )
    args = parser.parse_args()

    output_path = fetch_adult(args.output.resolve(), verify_ssl=not args.insecure)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
