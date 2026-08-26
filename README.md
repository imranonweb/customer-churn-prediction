# Customer Churn Prediction with Explainable AI

ML Lab project — Department of Computer Science, Green University of Bangladesh.

Predicts telecom customer churn on the IBM Telco Customer Churn dataset,
compares five classifiers, and explains the best model's predictions with SHAP.

## Setup

```bash
uv venv
uv pip install -r requirements.txt
```

Or with plain pip:

```bash
python -m venv .venv && .venv/Scripts/activate && pip install -r requirements.txt
```

## Running

```bash
python run_stage1.py
```

```bash
python -m pytest -q
```

## Project structure

```
├─ run_stage1.py          one runnable script per stage
├─ src/
│  ├─ config.py           all paths, constants, and column groups
│  ├─ data_loader.py      download (checksum-verified) · load · validate
│  └─ preprocess.py       clean · engineer features · split · preprocessor
├─ tests/test_stage1.py   correctness, reproducibility, no-leakage
├─ data/raw/              downloaded dataset (not committed)
├─ data/processed/        train.csv / test.csv
├─ notebooks/             EDA (Stage 2)
├─ app/                   Streamlit demo (Stage 6)
├─ artifacts/             trained models and metrics
├─ reports/figures/       generated plots
└─ docs/proposal.txt      the project proposal, for the write-up
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

## Stage 1 — preprocessing (complete)

Cleaning, feature engineering, and the stratified 80/20 split (5,634 train /
1,409 test, churn rate 0.2654 in both).

Engineered features: `is_new_customer`, `num_services`, `avg_charge`.

Three decisions worth knowing about:

1. **The 11 blank `TotalCharges` values are imputed with `0`, not the median.**
   Every one of them has `tenure == 0` — these customers have not been billed
   yet, so `0` is the true value. The median would inject about 1,397 of
   invented billing history into precisely the newest-customer segment the
   model most needs to get right. It is also *not row-wise* (see below).
2. **`customerID` is dropped immediately** — a unique identifier leaking into
   the features is the usual cause of a suspiciously good churn score.
3. **`No internet service` / `No phone service` collapse to `No`** across 7
   service columns. `InternetService` and `PhoneService` already carry that
   information, so this removes 7 redundant one-hot columns and loses nothing.

### No data leakage in Stage 1

Every Stage 1 operation is **row-wise stateless**: a row's output depends only
on that row, never on an aggregate over the dataset. That is what makes it safe
to clean before splitting. Anything that genuinely learns from data —
`StandardScaler` means and variances, `OneHotEncoder` category levels — is
deferred, so `build_preprocessor()` returns an **unfitted** transformer to be
fit inside a Pipeline on training folds only.

`run_stage1.py` prints this as a check block, and the test suite enforces it:

- `build_preprocessor()` raises `NotFittedError` and exposes no fitted state.
- `clean(subset)` is row-identical to `clean(full).loc[subset]` for the first
  100 rows, a random 500, both splits, and a **single row** — no aggregate
  could survive the single-row case.
- The saved CSVs stay on the original scale with string categoricals, so no
  fitted transformation reaches disk.
- A deliberately leaky `clean()` is checked to *fail* the invariance test, so
  the test cannot silently become vacuous.

## Remaining stages

| Stage | Deliverable |
|---|---|
| 2 | EDA notebook — distributions, churn by contract/payment, correlations |
| 3 | Train LogReg, Decision Tree, Random Forest, SVM, XGBoost (balanced class weights, 5-fold stratified CV) |
| 4 | Comparison table — Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC + confusion matrices |
| 5 | SHAP explanations for the best model |
| 6 | Streamlit demo |

## Reproducibility

Fixed seed (42) throughout, dataset pinned by checksum, and re-running
`run_stage1.py` produces byte-identical `train.csv` / `test.csv`.
