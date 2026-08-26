"""Central configuration: paths, constants, and column groups.

Every other module imports its constants from here so there is exactly one
place to change a path, a seed, or a column grouping.
"""

from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------
# Paths (all derived from the project root, so the code runs from anywhere)
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

RAW_CSV = RAW_DIR / "Telco-Customer-Churn.csv"
TRAIN_CSV = PROCESSED_DIR / "train.csv"
TEST_CSV = PROCESSED_DIR / "test.csv"

# --------------------------------------------------------------------------
# Dataset source
# --------------------------------------------------------------------------
# IBM's official public mirror of the Telco Customer Churn dataset.
# NOTE: this repository uses `master`, not `main` -- `main` returns 404.
DATA_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)
# Verified checksum of the file at DATA_URL. Guards against a silently
# changed upstream file, which would invalidate every result downstream.
DATA_SHA256 = "16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91"

# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------
SEED = 42
TEST_SIZE = 0.2

# --------------------------------------------------------------------------
# Expected shape of the raw file (asserted during validation)
# --------------------------------------------------------------------------
EXPECTED_ROWS = 7043
EXPECTED_COLS = 21
# The 11 rows with a blank TotalCharges -- all of them tenure == 0.
EXPECTED_BLANK_TOTAL_CHARGES = 11

TARGET = "Churn"
ID_COL = "customerID"

RAW_COLUMNS = [
    "customerID", "gender", "SeniorCitizen", "Partner", "Dependents",
    "tenure", "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling",
    "PaymentMethod", "MonthlyCharges", "TotalCharges", "Churn",
]

# --------------------------------------------------------------------------
# Column groups
# --------------------------------------------------------------------------
# Columns carrying the structural placeholders "No internet service" /
# "No phone service". These are redundant with InternetService / PhoneService,
# so they collapse to plain "No" -- saving 7 needless one-hot columns.
COLLAPSE_COLS = [
    "MultipleLines", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]

# Yes/No service columns counted by the `num_services` feature.
YES_NO_SERVICE_COLS = ["PhoneService"] + COLLAPSE_COLS

# Feature groups *after* cleaning (i.e. including engineered features).
NUMERIC_COLS = [
    "tenure", "MonthlyCharges", "TotalCharges",
    "avg_charge", "num_services", "is_new_customer", "SeniorCitizen",
]

CATEGORICAL_COLS = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod",
]

FEATURE_COLS = NUMERIC_COLS + CATEGORICAL_COLS


def ensure_dirs() -> None:
    """Create the output directories if they do not already exist."""
    for directory in (
        RAW_DIR, PROCESSED_DIR, ARTIFACTS_DIR, REPORTS_DIR, FIGURES_DIR
    ):
        directory.mkdir(parents=True, exist_ok=True)
