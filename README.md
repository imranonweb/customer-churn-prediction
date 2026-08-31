# Customer Churn Prediction with Explainable AI

Predicts telecom customer churn on the IBM Telco Customer Churn dataset,
compares five classifiers, and explains the best model's predictions with SHAP.
The final application is a Streamlit interactive dashboard.

## Quick start — run the Streamlit app

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd <repo-folder>

# 2. Create and activate a virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the app (from the project root)
python -m streamlit run app/streamlit_app.py
```

The app opens at `http://localhost:8501`. All model artifacts, metrics, and
figures are pre-generated and committed to the repository — no training step
is needed before running the app.

## Streamlit Community Cloud deployment

1. Push this repository to GitHub (all branches are supported).
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**.
3. Set **Main file path** to `app/streamlit_app.py`.
4. In **Advanced settings** set **Python version** to `3.12` or `3.13`
   (recommended for broadest package compatibility).
5. Click **Deploy**.

No secrets or environment variables are required.

## Reproduce the full ML pipeline

These steps are only needed if you want to retrain models or regenerate figures.
The committed artifacts are already the final outputs.

```bash
# Stage 1 — download data, clean, split
python run_stage1.py

# Stage 3 — train all five models (produces artifacts/*.pkl)
# (run_stage3b.py for the SMOTE comparison experiment)
python run_stage3b.py

# Stage 4 — evaluate on the held-out test set
python run_stage4.py

# Stage 5 — SHAP explanations
python run_stage5.py

# Tests
python -m pytest -q
```

## Project structure

```
├── app/
│   ├── streamlit_app.py      main Streamlit entry point
│   ├── ui_components.py      reusable UI helpers
│   └── ui_styles.py          CSS tokens and styling
├── src/
│   ├── config.py             all paths, constants, column groups
│   ├── data_loader.py        download · load · checksum-validate
│   ├── preprocess.py         clean · engineer features · split
│   ├── train.py              train five classifiers
│   ├── evaluate.py           metrics, confusion matrices, curves
│   ├── explain.py            SHAP explanations
│   └── smote_experiment.py   SMOTE vs class-weight comparison
├── tests/
│   ├── test_stage1.py        correctness, reproducibility, no-leakage
│   └── test_stage6_ui.py     UI helpers, risk banding, SHAP regrouping
├── artifacts/                trained models and metrics CSV files
├── reports/figures/          generated PNGs displayed by the app
├── data/raw/                 IBM Telco Customer Churn CSV
├── data/processed/           train.csv / test.csv (stratified 80/20)
├── notebooks/01_eda.ipynb    EDA (Stage 2)
├── docs/proposal.txt         project proposal
├── .streamlit/config.toml    app theme and client settings
├── requirements.txt          Python dependencies
└── README.md                 this file
```

## Dataset

| | |
|---|---|
| Source | IBM's public mirror (`telco-customer-churn-on-icp4d`, `master` branch) |
| Rows / columns | 7,043 × 21, no duplicate `customerID` |
| Target | `Churn` — 1,869 Yes (26.54%) / 5,174 No |
| Integrity | SHA-256 verified on every run; a mismatch raises rather than proceeding |

**Because 73.46% of customers do not churn, accuracy alone is misleading** — a
model that predicts "no churn" for everyone scores 73.46% while catching zero
churners. Recall, F1, and PR-AUC are the metrics that matter here.

## Stage 1 — preprocessing

Cleaning, feature engineering, and the stratified 80/20 split (5,634 train /
1,409 test, churn rate 0.2654 in both).

Engineered features: `is_new_customer`, `num_services`, `avg_charge`.

Three decisions worth knowing about:

1. **The 11 blank `TotalCharges` values are imputed with `0`, not the median.**
   Every one of them has `tenure == 0` — these customers have not been billed
   yet, so `0` is the true value.
2. **`customerID` is dropped immediately** — a unique identifier leaking into
   the features is the usual cause of a suspiciously good churn score.
3. **`No internet service` / `No phone service` collapse to `No`** across 7
   service columns, removing 7 redundant one-hot columns.

### No data leakage in Stage 1

Every Stage 1 operation is **row-wise stateless**: a row's output depends only
on that row, never on an aggregate over the dataset. Anything that genuinely
learns from data — `StandardScaler` means and variances, `OneHotEncoder`
category levels — is deferred, so `build_preprocessor()` returns an **unfitted**
transformer to be fit inside a Pipeline on training folds only.

## Selected model: Random Forest

Chosen on 5-fold cross-validated PR-AUC before the test set was opened.
SHAP (TreeExplainer) provides global feature importance, a beeswarm, and
per-customer waterfall contributions with an additivity check.

## Reproducibility

Fixed seed (42) throughout, dataset pinned by checksum, and re-running
`run_stage1.py` produces byte-identical `train.csv` / `test.csv`.
