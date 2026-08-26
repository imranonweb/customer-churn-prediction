"""Stage 1a: download, load, and validate the raw dataset."""

from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

import pandas as pd

from src import config


def _sha256(path: Path) -> str:
    """Return the SHA-256 hex digest of a file, read in chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_data(force: bool = False) -> Path:
    """Download the Telco CSV to data/raw/ and verify its checksum.

    Skips the download when the file is already present and its checksum
    matches. A mismatch raises instead of proceeding, because silently
    training on a changed upstream file would invalidate every result.
    """
    config.ensure_dirs()
    path = config.RAW_CSV

    if path.exists() and not force:
        actual = _sha256(path)
        if actual == config.DATA_SHA256:
            return path
        print("  ! checksum mismatch on existing file, re-downloading")

    print(f"  downloading {config.DATA_URL}")
    urllib.request.urlretrieve(config.DATA_URL, path)

    actual = _sha256(path)
    if actual != config.DATA_SHA256:
        raise ValueError(
            "Downloaded file failed checksum verification.\n"
            f"  expected: {config.DATA_SHA256}\n"
            f"  actual:   {actual}\n"
            "The upstream dataset may have changed."
        )
    return path


def load_raw(path: Path | None = None) -> pd.DataFrame:
    """Load the raw CSV.

    `TotalCharges` is read as a *string* on purpose: 11 rows hold a blank
    (" ") rather than a number, and letting pandas coerce them silently would
    hide the fact that they need deliberate handling.
    """
    path = path or config.RAW_CSV
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run download_data() first."
        )
    return pd.read_csv(path, dtype={"TotalCharges": "string"})


def validate_raw(df: pd.DataFrame) -> dict:
    """Assert the raw data matches every documented expectation.

    Fails loudly and early: if the input is not what the rest of the pipeline
    was written against, stopping here is much cheaper than debugging strange
    metrics five stages later.
    """
    errors: list[str] = []

    if df.shape != (config.EXPECTED_ROWS, config.EXPECTED_COLS):
        errors.append(
            f"expected shape ({config.EXPECTED_ROWS}, {config.EXPECTED_COLS}), "
            f"got {df.shape}"
        )

    if list(df.columns) != config.RAW_COLUMNS:
        missing = set(config.RAW_COLUMNS) - set(df.columns)
        extra = set(df.columns) - set(config.RAW_COLUMNS)
        errors.append(f"column mismatch (missing={missing}, extra={extra})")

    n_dupe_ids = int(df[config.ID_COL].duplicated().sum())
    if n_dupe_ids:
        errors.append(f"{n_dupe_ids} duplicate {config.ID_COL} values")

    target_values = set(df[config.TARGET].unique())
    if target_values != {"No", "Yes"}:
        errors.append(f"unexpected {config.TARGET} values: {target_values}")

    # The blank TotalCharges rows must all be tenure == 0. This is the
    # assumption that justifies imputing 0 rather than a median, so it is
    # checked rather than trusted.
    blank_mask = df["TotalCharges"].str.strip() == ""
    n_blank = int(blank_mask.sum())
    if n_blank != config.EXPECTED_BLANK_TOTAL_CHARGES:
        errors.append(
            f"expected {config.EXPECTED_BLANK_TOTAL_CHARGES} blank "
            f"TotalCharges, got {n_blank}"
        )
    bad_tenure = df.loc[blank_mask, "tenure"].ne(0).sum()
    if bad_tenure:
        errors.append(
            f"{bad_tenure} blank-TotalCharges rows have tenure != 0 -- "
            "imputing 0 would no longer be justified"
        )

    # Nulls are only expected in TotalCharges, and there they appear as
    # blank strings rather than NaN.
    null_counts = df.isna().sum()
    unexpected_nulls = null_counts[null_counts > 0].to_dict()
    if unexpected_nulls:
        errors.append(f"unexpected nulls: {unexpected_nulls}")

    if errors:
        raise ValueError(
            "Raw data validation failed:\n"
            + "\n".join(f"  - {e}" for e in errors)
        )

    churn_yes = int((df[config.TARGET] == "Yes").sum())
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "duplicate_ids": n_dupe_ids,
        "blank_total_charges": n_blank,
        "blank_all_tenure_zero": True,
        "churn_yes": churn_yes,
        "churn_no": int(df.shape[0] - churn_yes),
        "churn_rate": churn_yes / df.shape[0],
        "majority_class_accuracy": 1 - churn_yes / df.shape[0],
    }
