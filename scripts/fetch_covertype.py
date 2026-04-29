from __future__ import annotations

import argparse
from gzip import GzipFile
from io import BytesIO, StringIO
from pathlib import Path
import ssl
from urllib.request import urlopen

import pandas as pd


UCI_COVERTYPE_DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz"
COVERTYPE_COLUMNS = [
    "elevation",
    "aspect",
    "slope",
    "horizontal_distance_to_hydrology",
    "vertical_distance_to_hydrology",
    "horizontal_distance_to_roadways",
    "hillshade_9am",
    "hillshade_noon",
    "hillshade_3pm",
    "horizontal_distance_to_fire_points",
    *[f"wilderness_area_{index}" for index in range(1, 5)],
    *[f"soil_type_{index}" for index in range(1, 41)],
    "cover_type",
]


def _download_bytes(url: str, *, verify_ssl: bool = True) -> bytes:
    try:
        import requests

        response = requests.get(url, timeout=120, verify=verify_ssl)
        response.raise_for_status()
        return response.content
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
            return response.read()


def normalize_covertype(raw_text: str) -> pd.DataFrame:
    frame = pd.read_csv(StringIO(raw_text), names=COVERTYPE_COLUMNS, header=None)
    return frame.dropna(how="all").reset_index(drop=True)


def fetch_covertype(output_path: Path, *, verify_ssl: bool = True) -> Path:
    payload = _download_bytes(UCI_COVERTYPE_DATA_URL, verify_ssl=verify_ssl)
    with GzipFile(fileobj=BytesIO(payload)) as handle:
        raw_text = handle.read().decode("utf-8")
    frame = normalize_covertype(raw_text)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch the UCI Covertype dataset and normalize it to the CSV shape used by Treehouse Lab.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("custom_datasets/covertype.csv"),
        help="Where to write the normalized CSV. Default: custom_datasets/covertype.csv",
    )
    parser.add_argument(
        "--insecure",
        action="store_true",
        help="Disable SSL certificate verification as a last resort if the local Python trust store is misconfigured.",
    )
    args = parser.parse_args()

    output_path = fetch_covertype(args.output.resolve(), verify_ssl=not args.insecure)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
