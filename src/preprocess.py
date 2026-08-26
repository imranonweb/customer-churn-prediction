"""Stage 1b: cleaning, feature building, train/test split, preprocessor.

Leakage discipline for this module
---------------------------------
Every operation in `clean()` is **row-wise stateless**: each row's output
depends only on that row, never on an aggregate over the dataset. That is what
makes it safe to clean before splitting.

The one tempting exception is the blank `TotalCharges`. Filling it with the
*median* would make cleaning data-dependent (a median is an aggregate over all
rows) and computing it before the split would leak test information into
train. All 11 blanks are `tenure == 0` customers who have not been billed yet,
so the correct value is the **constant 0** -- which learns nothing.

Anything that genuinely must learn from data -- `StandardScaler` means and
variances, `OneHotEncoder` category levels -- is deferred: `build_preprocessor`
returns an **unfitted** transformer, fit later inside a Pipeline on training
folds only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted

from src import config


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw frame and build engineered features.

    Row-wise stateless throughout -- see the module docstring.
    """
    d = df.copy()

    # -- TotalCharges: blank -> 0 (a constant, never a learned median) -----
    total_charges = pd.to_numeric(d["TotalCharges"].astype("string").str.strip(),
                                  errors="coerce")
    d["is_new_customer"] = (d["tenure"] == 0).astype(int)
    d["TotalCharges"] = total_charges.fillna(0.0).astype(float)

    # -- Collapse structural placeholders ---------------------------------
    # "No internet service" / "No phone service" duplicate information already
    # held by InternetService / PhoneService, so they become plain "No".
    d[config.COLLAPSE_COLS] = d[config.COLLAPSE_COLS].replace(
        {"No internet service": "No", "No phone service": "No"}
    )

    # -- Engineered features ----------------------------------------------
    # How many services this customer subscribes to.
    d["num_services"] = (
        (d[config.YES_NO_SERVICE_COLS] == "Yes").sum(axis=1)
        + (d["InternetService"] != "No").astype(int)
    ).astype(int)

    # Average spend per month of tenure. clip(lower=1) avoids dividing by zero
    # for brand-new customers; their TotalCharges is 0 so avg_charge is 0.
    d["avg_charge"] = d["TotalCharges"] / d["tenure"].clip(lower=1)

    # -- Target -> binary --------------------------------------------------
    d[config.TARGET] = d[config.TARGET].map({"No": 0, "Yes": 1}).astype(int)

    # -- Drop the identifier ----------------------------------------------
    # A unique ID is the classic route to a leaked, suspiciously good score.
    d = d.drop(columns=[config.ID_COL])

    # Fixed column order keeps the saved CSVs byte-stable across runs.
    missing = set(config.FEATURE_COLS + [config.TARGET]) - set(d.columns)
    if missing:
        raise KeyError(f"clean() did not produce expected columns: {missing}")
    return d[config.FEATURE_COLS + [config.TARGET]]


def split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified train/test split -- 5634 / 1409 rows at the default config.

    Stratifying on the target keeps the ~26.5% churn rate identical in both
    halves, so the test set measures the same problem the model trained on.
    """
    train_df, test_df = train_test_split(
        df,
        test_size=config.TEST_SIZE,
        random_state=config.SEED,
        stratify=df[config.TARGET],
    )
    return train_df, test_df


def build_preprocessor(scale: bool = True) -> ColumnTransformer:
    """Return an **unfitted** ColumnTransformer.

    Contains no `.fit()` call -- fitting happens later, inside a Pipeline, on
    training folds only.

    Parameters
    ----------
    scale
        Standardise numeric columns. Needed by Logistic Regression and SVM;
        pointless for the tree models, which are scale-invariant.
    """
    numeric = StandardScaler() if scale else "passthrough"
    return ColumnTransformer(
        transformers=[
            ("num", numeric, config.NUMERIC_COLS),
            # handle_unknown="ignore" keeps inference from crashing on a
            # category that never appeared in training.
            ("cat", OneHotEncoder(handle_unknown="ignore"),
             config.CATEGORICAL_COLS),
        ],
        remainder="drop",
    )


def save_splits(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """Write the cleaned, *untransformed* splits to data/processed/."""
    config.ensure_dirs()
    train_df.to_csv(config.TRAIN_CSV, index=False)
    test_df.to_csv(config.TEST_CSV, index=False)


def load_splits() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read back the processed splits written by `save_splits`."""
    for path in (config.TRAIN_CSV, config.TEST_CSV):
        if not path.exists():
            raise FileNotFoundError(f"{path} not found. Run run_stage1.py first.")
    return pd.read_csv(config.TRAIN_CSV), pd.read_csv(config.TEST_CSV)


# --------------------------------------------------------------------------
# Leakage checks -- shared by run_stage1.py (display) and tests (enforcement)
# --------------------------------------------------------------------------
def check_preprocessor_unfitted() -> dict:
    """Confirm build_preprocessor() has learned nothing."""
    results = {}
    for scale in (True, False):
        pre = build_preprocessor(scale=scale)
        try:
            check_is_fitted(pre)
            raised = False
        except NotFittedError:
            raised = True
        fitted_attrs = [
            a for a in vars(pre) if a.endswith("_") and not a.startswith("__")
        ]
        results[f"scale={scale}"] = {
            "not_fitted_error_raised": raised,
            "fitted_attributes": fitted_attrs,
            "has_transformers_": hasattr(pre, "transformers_"),
        }
    return results


def check_partition_invariance(raw: pd.DataFrame) -> dict[str, bool]:
    """Confirm `clean()` is row-wise stateless.

    If any step used an aggregate, cleaning a subset would give different
    values than cleaning the whole frame and then selecting those rows. The
    single-row case is the strictest: no aggregate could survive it.
    """
    full = clean(raw)
    rng = np.random.default_rng(config.SEED)
    train_df, test_df = split(raw)

    subsets = {
        "first 100 rows": raw.index[:100],
        "random 500 rows": rng.choice(raw.index, 500, replace=False),
        f"train split ({len(train_df)})": train_df.index,
        f"test split ({len(test_df)})": test_df.index,
        "single row": raw.index[[6]],
    }

    results = {}
    for name, idx in subsets.items():
        try:
            assert_frame_equal(clean(raw.loc[idx]), full.loc[idx])
            results[name] = True
        except AssertionError:
            results[name] = False
    return results


def check_artifacts_untransformed(train_df: pd.DataFrame) -> dict:
    """Confirm no fitted transformation was baked into the saved data.

    Scaled numerics would have a mean near 0; one-hot encoding would have
    replaced the string categoricals with indicator columns.
    """
    return {
        "monthly_charges_mean": float(train_df["MonthlyCharges"].mean()),
        "on_original_scale": bool(train_df["MonthlyCharges"].mean() > 50),
        "contract_is_strings": bool(
            train_df["Contract"].dtype.kind in "OU"
            or str(train_df["Contract"].dtype) == "str"
        ),
        "contract_values": sorted(train_df["Contract"].unique().tolist()),
    }
