from __future__ import annotations

import argparse
from io import BytesIO
from pathlib import Path
import ssl
from urllib.request import urlopen
from zipfile import ZipFile

import pandas as pd


UCI_BANK_MARKETING_ZIP_URL = "https://archive.ics.uci.edu/static/public/222/bank%2Bmarketing.zip"
OUTER_ZIP_MEMBER_NAME = "bank.zip"
INNER_ZIP_MEMBER_NAME = "bank-full.csv"


def _download_zip_bytes(*, verify_ssl: bool = True) -> bytes:
    try:
        import requests

        response = requests.get(UCI_BANK_MARKETING_ZIP_URL, timeout=60, verify=verify_ssl)
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

        with urlopen(UCI_BANK_MARKETING_ZIP_URL, context=context) as response:
            return response.read()


def fetch_bank_marketing(output_path: Path, *, verify_ssl: bool = True) -> Path:
    payload = _download_zip_bytes(verify_ssl=verify_ssl)

    with ZipFile(BytesIO(payload)) as outer_archive:
        nested_zip_payload = outer_archive.read(OUTER_ZIP_MEMBER_NAME)

    with ZipFile(BytesIO(nested_zip_payload)) as inner_archive:
        with inner_archive.open(INNER_ZIP_MEMBER_NAME) as handle:
            frame = pd.read_csv(handle, sep=";")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch the UCI Bank Marketing dataset and normalize it to the CSV shape used by Treehouse Lab.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("custom_datasets/bank-full.csv"),
        help="Where to write the normalized CSV. Default: custom_datasets/bank-full.csv",
    )
    parser.add_argument(
        "--insecure",
        action="store_true",
        help="Disable SSL certificate verification as a last resort if the local Python trust store is misconfigured.",
    )
    args = parser.parse_args()

    output_path = fetch_bank_marketing(args.output.resolve(), verify_ssl=not args.insecure)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
