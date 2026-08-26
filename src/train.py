"""Stage 3: train and cross-validate the five required models.

Run from the project root:

    python -m src.train

Leakage discipline for this module
---------------------------------
This module **never opens the test set**. It reads `data/processed/train.csv`
and nothing else, so the test set stays untouched until Stage 4. (That is also
why it does not call `preprocess.load_splits()`, which would read both files.)

Every model is a `Pipeline` whose first step is the *unfitted* transformer from
`preprocess.build_preprocessor()`. Because the transformer lives inside the
pipeline, `cross_validate` re-fits it on each training fold only -- the scaler's
means and the encoder's category levels are never learned from a fold that is
being scored. Fitting the preprocessor once up front and then cross-validating
would leak fold information; this is the whole reason Stage 1 returned it
unfitted.

Outputs (all under artifacts/):
    logistic_regression.pkl  decision_tree.pkl  random_forest.pkl
    svm.pkl                  xgboost.pkl
    model_cv_results.csv     model_metadata.json
"""

from __future__ import annotations

import ast
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src import config, preprocess

# --------------------------------------------------------------------------
# Stage 3 outputs. Defined here rather than in config.py, which is locked.
# --------------------------------------------------------------------------
CV_RESULTS_CSV = config.ARTIFACTS_DIR / "model_cv_results.csv"
METADATA_JSON = config.ARTIFACTS_DIR / "model_metadata.json"

# --------------------------------------------------------------------------
# Cross-validation setup
# --------------------------------------------------------------------------
N_SPLITS = 5

# Display name -> sklearn scorer. The two ranking metrics are computed from
# predicted probabilities: "roc_auc" calls roc_auc_score and
# "average_precision" calls average_precision_score (PR-AUC).
METRICS: dict[str, str] = {
    "Accuracy": "accuracy",
    "Precision": "precision",
    "Recall": "recall",
    "F1": "f1",
    "ROC-AUC": "roc_auc",
    "PR-AUC": "average_precision",
}

# Model selection metric. Accuracy is deliberately *not* used: at 26.5% churn a
# constant "no churn" prediction already scores 73.5%, so accuracy rewards
# ignoring the minority class. PR-AUC summarises precision/recall trade-offs on
# the positive class across all thresholds, which is what this problem is about.
SELECTION_METRIC = "PR-AUC"


def load_train() -> pd.DataFrame:
    """Read the Stage 1 training split. Deliberately does not read test.csv."""
    if not config.TRAIN_CSV.exists():
        raise FileNotFoundError(
            f"{config.TRAIN_CSV} not found. Run `python run_stage1.py` first."
        )
    return pd.read_csv(config.TRAIN_CSV)


def compute_scale_pos_weight(y: pd.Series) -> float:
    """XGBoost's imbalance knob: negatives / positives.

    Computed from the **training labels only**. It is a label-only ratio, so it
    carries no feature information and nothing from the test set.
    """
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0:
        raise ValueError("no positive class in the training labels")
    return n_neg / n_pos


def build_models(scale_pos_weight: float) -> dict[str, dict]:
    """The five required models, with conservative baseline parameters.

    `scale` selects standardisation: on for the two distance/margin-based
    models, off for the three tree ensembles, which are scale-invariant.
    """
    return {
        "logistic_regression": {
            "name": "Logistic Regression",
            "scale": True,
            "imbalance": "class_weight='balanced'",
            "estimator": LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=config.SEED,
            ),
        },
        "decision_tree": {
            "name": "Decision Tree",
            "scale": False,
            "imbalance": "class_weight='balanced'",
            # Depth-limited on purpose: an unconstrained tree memorises the
            # training set and cross-validates far worse.
            "estimator": DecisionTreeClassifier(
                max_depth=5,
                min_samples_leaf=20,
                class_weight="balanced",
                random_state=config.SEED,
            ),
        },
        "random_forest": {
            "name": "Random Forest",
            "scale": False,
            "imbalance": "class_weight='balanced'",
            "estimator": RandomForestClassifier(
                n_estimators=300,
                # Same leaf floor as the single tree. Fully-grown forests
                # cross-validate slightly worse here (PR-AUC 0.639 vs 0.663)
                # and produce a ~7x larger pickle.
                min_samples_leaf=20,
                class_weight="balanced",
                random_state=config.SEED,
                n_jobs=-1,
            ),
        },
        "svm": {
            "name": "SVM",
            "scale": True,
            "imbalance": "class_weight='balanced'",
            # SVC has no native probabilities. `SVC(probability=True)` is
            # deprecated in scikit-learn 1.9, and its replacement is exactly
            # what it used internally: a sigmoid (Platt) calibration fitted on
            # out-of-fold decision values. Wrapping it explicitly keeps
            # predict_proba available for Stage 4's ROC/PR curves without any
            # deprecation noise.
            #
            # No leakage: CalibratedClassifierCV splits *its own input* -- which
            # is a training fold -- into 5 sub-folds. It never sees data from
            # the fold being scored, let alone the test set.
            "estimator": CalibratedClassifierCV(
                SVC(
                    kernel="rbf",
                    C=1.0,
                    gamma="scale",
                    class_weight="balanced",
                    random_state=config.SEED,
                ),
                method="sigmoid",
                ensemble=False,
                cv=StratifiedKFold(
                    n_splits=N_SPLITS, shuffle=True, random_state=config.SEED
                ),
            ),
        },
        "xgboost": {
            "name": "XGBoost",
            "scale": False,
            "imbalance": f"scale_pos_weight={scale_pos_weight:.6f}",
            "estimator": XGBClassifier(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.1,
                subsample=0.9,
                colsample_bytree=0.9,
                scale_pos_weight=scale_pos_weight,
                eval_metric="logloss",
                tree_method="hist",
                random_state=config.SEED,
                n_jobs=-1,
            ),
        },
    }


def build_pipeline(spec: dict) -> Pipeline:
    """Wrap a model in a pipeline with the Stage 1 preprocessor (unfitted)."""
    return Pipeline(
        [
            ("prep", preprocess.build_preprocessor(scale=spec["scale"])),
            ("model", spec["estimator"]),
        ]
    )


# --------------------------------------------------------------------------
# Console helpers
# --------------------------------------------------------------------------
def config_attributes_used() -> list[str]:
    """Every `config.X` this module actually references, read from its own AST.

    This is how the "test set untouched" claim is checked rather than asserted.
    All file paths in this project come from `config`, so the only way Stage 3
    could reach the test split is `config.TEST_CSV` -- and parsing the source
    finds that regardless of where it appears in the code. Comments and
    docstrings are ignored, so a *mention* of test.csv cannot make the check
    pass or fail; only real attribute access counts.
    """
    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    return sorted(
        {
            node.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "config"
        }
    )


def rule(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def ok(passed: bool) -> str:
    return "PASS" if passed else "*** FAIL ***"


_WIDTHS = {"Accuracy": 8, "Precision": 9, "Recall": 6, "F1": 6,
           "ROC-AUC": 7, "PR-AUC": 6}


def print_results_table(results: pd.DataFrame, best_key: str) -> None:
    """Print the comparison table: Model | Accuracy | ... | PR-AUC."""
    header = "Model".ljust(20) + "".join(
        f" | {m:>{w}}" for m, w in _WIDTHS.items()
    )
    print("  " + header)
    print("  " + "-" * len(header))
    for _, row in results.iterrows():
        cells = "".join(
            f" | {row[m]:>{w}.4f}" for m, w in _WIDTHS.items()
        )
        marker = f"   <-- best {SELECTION_METRIC}" if row["key"] == best_key else ""
        print("  " + row["Model"].ljust(20) + cells + marker)


# --------------------------------------------------------------------------
# Stage steps
# --------------------------------------------------------------------------
def cross_validate_models(
    models: dict[str, dict], X: pd.DataFrame, y: pd.Series, cv: StratifiedKFold
) -> pd.DataFrame:
    """Cross-validate every model on the training set and collect the metrics."""
    rows = []
    for key, spec in models.items():
        print(f"Training {spec['name']}...")
        started = time.perf_counter()
        scores = cross_validate(
            build_pipeline(spec),
            X,
            y,
            cv=cv,
            scoring=list(METRICS.values()),
            n_jobs=-1,
            # Surface a broken fold instead of silently recording NaN.
            error_score="raise",
        )
        elapsed = time.perf_counter() - started

        row = {"key": key, "Model": spec["name"]}
        for display, scorer in METRICS.items():
            fold_scores = scores[f"test_{scorer}"]
            row[display] = float(np.mean(fold_scores))
            row[f"{display}_std"] = float(np.std(fold_scores))
        rows.append(row)

        print(f"  {N_SPLITS}-fold CV done in {elapsed:5.1f}s   "
              + "  ".join(f"{m} {row[m]:.4f}" for m in ("F1", "ROC-AUC", "PR-AUC")))

    return pd.DataFrame(rows)


def fit_and_save(
    models: dict[str, dict], X: pd.DataFrame, y: pd.Series
) -> dict[str, Pipeline]:
    """Fit each full pipeline on the entire training set and persist it."""
    fitted: dict[str, Pipeline] = {}
    for key, spec in models.items():
        started = time.perf_counter()
        pipe = build_pipeline(spec).fit(X, y)
        elapsed = time.perf_counter() - started

        path = config.ARTIFACTS_DIR / f"{key}.pkl"
        joblib.dump(pipe, path)
        fitted[key] = pipe
        print(f"  {spec['name']:<20} fitted in {elapsed:5.1f}s  ->  "
              f"{path.relative_to(config.PROJECT_ROOT)} "
              f"({path.stat().st_size / 1024:,.0f} KB)")
    return fitted


def verify_saved_models(
    models: dict[str, dict], sample: pd.DataFrame, failures: list[str]
) -> None:
    """Reload every pickle and make a prediction on one training row."""
    for key, spec in models.items():
        path = config.ARTIFACTS_DIR / f"{key}.pkl"
        try:
            pipe = joblib.load(path)
            pred = pipe.predict(sample)
            proba = pipe.predict_proba(sample)[:, 1]
            good = (
                pred.shape == (len(sample),)
                and set(np.unique(pred)) <= {0, 1}
                and bool(((proba >= 0.0) & (proba <= 1.0)).all())
                and pipe.named_steps["prep"].n_features_in_ == len(config.FEATURE_COLS)
            )
            print(f"  {spec['name']:<20} reloaded  predict={int(pred[0])}  "
                  f"P(churn)={proba[0]:.4f}  "
                  f"prep.n_features_in_={pipe.named_steps['prep'].n_features_in_}"
                  f"   {ok(good)}")
            if not good:
                failures.append(f"{key}.pkl reloaded but behaved unexpectedly")
        except Exception as exc:  # noqa: BLE001 - report, don't crash the stage
            print(f"  {spec['name']:<20} *** FAIL *** {type(exc).__name__}: {exc}")
            failures.append(f"{key}.pkl could not be reloaded ({exc})")


def write_metadata(
    models: dict[str, dict],
    results: pd.DataFrame,
    fitted: dict[str, Pipeline],
    y: pd.Series,
    scale_pos_weight: float,
    best_key: str,
) -> None:
    """Record everything needed to reproduce or interpret this stage."""
    # Encoded feature names come from a pipeline fitted on the training set.
    encoded = list(
        fitted["logistic_regression"].named_steps["prep"].get_feature_names_out()
    )

    metrics_by_model = {}
    for _, row in results.iterrows():
        metrics_by_model[row["key"]] = {
            m: {"mean": round(row[m], 6), "std": round(row[f"{m}_std"], 6)}
            for m in METRICS
        }

    metadata = {
        "stage": 3,
        "description": "Cross-validated baselines for the five required models",
        "random_seed": config.SEED,
        "training_data": {
            "source": str(config.TRAIN_CSV.relative_to(config.PROJECT_ROOT)),
            "rows": int(len(y)),
            "churn_positive": int((y == 1).sum()),
            "churn_negative": int((y == 0).sum()),
            "churn_rate": round(float(y.mean()), 6),
        },
        "test_set": {
            "used_in_stage_3": False,
            "verified_by": (
                "AST scan of src/train.py: config.TEST_CSV is never referenced"
            ),
            "config_attributes_used": config_attributes_used(),
            "note": "The test split is held untouched for Stage 4.",
        },
        "features": {
            "input_columns": len(config.FEATURE_COLS),
            "numeric": config.NUMERIC_COLS,
            "categorical": config.CATEGORICAL_COLS,
            "engineered_in_stage_1": ["is_new_customer", "num_services", "avg_charge"],
            "columns_after_encoding": len(encoded),
            "names_after_encoding": encoded,
            "encoding": "OneHotEncoder(handle_unknown='ignore')",
            "scaling": (
                "StandardScaler for Logistic Regression and SVM; "
                "passthrough for Decision Tree, Random Forest and XGBoost"
            ),
            "preprocessor_fitted": "inside each Pipeline, on training folds only",
        },
        "cross_validation": {
            "strategy": "StratifiedKFold",
            "n_splits": N_SPLITS,
            "shuffle": True,
            "random_state": config.SEED,
            "applied_to": "training set only",
            "nested_cv": False,
            "metrics": list(METRICS),
            "selection_metric": SELECTION_METRIC,
            "selection_note": (
                "Accuracy is not used for selection: a constant 'no churn' "
                "prediction already scores "
                f"{1 - float(y.mean()):.4f} on this class balance."
            ),
        },
        "class_imbalance_strategy": {
            "approach": "cost re-weighting inside the estimator (no resampling)",
            "resampling": "none - no SMOTE, no under/over-sampling",
            "scale_pos_weight_value": round(scale_pos_weight, 6),
            "scale_pos_weight_source": "negatives / positives, training labels only",
            "per_model": {k: s["imbalance"] for k, s in models.items()},
        },
        "models": {
            key: {
                "display_name": spec["name"],
                "estimator": type(spec["estimator"]).__name__,
                "scaled": spec["scale"],
                "artifact": f"artifacts/{key}.pkl",
                "params": {
                    k: (v if isinstance(v, (int, float, str, bool, type(None)))
                        else str(v))
                    for k, v in spec["estimator"].get_params(deep=False).items()
                },
                "cv_metrics": metrics_by_model[key],
            }
            for key, spec in models.items()
        },
        "best_cv_model_by_pr_auc": {
            "key": best_key,
            "name": models[best_key]["name"],
            "pr_auc": round(
                float(results.loc[results["key"] == best_key, SELECTION_METRIC].iloc[0]),
                6,
            ),
            "note": (
                "Best cross-validated model on the training data. Not the final "
                "winner: Stage 4 evaluates the untouched test set."
            ),
        },
        "outputs": {
            "cv_results": "artifacts/model_cv_results.csv",
            "models": [f"artifacts/{key}.pkl" for key in models],
        },
        "library_versions": {
            "python": ".".join(str(v) for v in sys.version_info[:3]),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit-learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
            "joblib": joblib.__version__,
        },
    }

    with open(METADATA_JSON, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)
        fh.write("\n")


def main() -> int:
    started = time.perf_counter()
    failures: list[str] = []
    config.ensure_dirs()

    # ---------------------------------------------------------------- 1
    rule("1. LOAD TRAINING DATA  (the test set is never opened in Stage 3)")
    train_df = load_train()
    X = train_df[config.FEATURE_COLS]
    y = train_df[config.TARGET]
    spw = compute_scale_pos_weight(y)

    print(f"  file                     : "
          f"{config.TRAIN_CSV.relative_to(config.PROJECT_ROOT)}")
    print(f"  rows                     : {len(train_df):,}")
    print(f"  input features           : {len(config.FEATURE_COLS)} "
          f"({len(config.NUMERIC_COLS)} numeric, "
          f"{len(config.CATEGORICAL_COLS)} categorical)")
    print(f"  churn positives          : {int(y.sum()):,} / {len(y):,} "
          f"= {y.mean():.4f}")
    print(f"  majority-class accuracy  : {1 - y.mean():.4f}"
          "   <-- why PR-AUC selects, not accuracy")
    print(f"  scale_pos_weight         : {spw:.6f}  "
          "(negatives/positives, training labels only)")
    print(f"  NaN in training features : {int(X.isna().sum().sum())}")

    models = build_models(spw)

    # ---------------------------------------------------------------- 2
    rule(f"2. CROSS-VALIDATION  (StratifiedKFold n_splits={N_SPLITS}, "
         f"shuffle=True, random_state={config.SEED})")
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=config.SEED)
    results = cross_validate_models(models, X, y, cv)

    best_idx = results[SELECTION_METRIC].idxmax()
    best_key = results.loc[best_idx, "key"]

    # ---------------------------------------------------------------- 3
    rule("MODEL CROSS-VALIDATION RESULTS")
    print_results_table(results, best_key)
    print()
    print(f"  Best {SELECTION_METRIC} in cross-validation: "
          f"{models[best_key]['name']} "
          f"({results.loc[best_idx, SELECTION_METRIC]:.4f})")
    print("  NOTE: this is the best model on the *training* data only. The final")
    print("        best model is decided in Stage 4 on the untouched test set.")

    # Save the comparison table: means first, then fold standard deviations.
    # Wall-clock timings are deliberately left out so the file is byte-identical
    # across runs -- the same reproducibility promise Stage 1 makes.
    ordered = ["Model"] + list(METRICS) + [f"{m}_std" for m in METRICS]
    results[ordered].to_csv(CV_RESULTS_CSV, index=False)
    print(f"\n  written: {CV_RESULTS_CSV.relative_to(config.PROJECT_ROOT)}")

    # ---------------------------------------------------------------- 3
    rule(f"3. FIT FINAL PIPELINES ON THE FULL TRAINING SET  ({len(X):,} rows)")
    fitted = fit_and_save(models, X, y)

    # ---------------------------------------------------------------- 4
    rule("4. RELOAD SAVED MODELS AND PREDICT ON A TRAINING ROW")
    verify_saved_models(models, X.iloc[[0]], failures)

    # ---------------------------------------------------------------- 5
    rule("5. CHECKS")
    total = time.perf_counter() - started
    write_metadata(models, results, fitted, y, spw, best_key)

    metric_cols = list(METRICS)
    all_metrics = bool(results[metric_cols].notna().all().all())
    in_range = bool(
        ((results[metric_cols] >= 0.0) & (results[metric_cols] <= 1.0)).all().all()
    )
    beats_baseline = bool((results["PR-AUC"] > float(y.mean())).all())
    files = [config.ARTIFACTS_DIR / f"{k}.pkl" for k in models] + [
        CV_RESULTS_CSV, METADATA_JSON
    ]
    missing = [f.name for f in files if not f.exists()]

    used = config_attributes_used()
    test_untouched = "TEST_CSV" not in used

    print(f"  5 models trained                  : {ok(len(results) == 5)}")
    print(f"  6 metrics present for all models  : {ok(all_metrics)}")
    print(f"  all metrics within [0, 1]         : {ok(in_range)}")
    print(f"  every PR-AUC beats the {y.mean():.4f} base rate : {ok(beats_baseline)}")
    print(f"  7 artifacts written               : {ok(not missing)}"
          f"{'  missing: ' + ', '.join(missing) if missing else ''}")
    print(f"  config.TEST_CSV never referenced  : {ok(test_untouched)}")
    print(f"    config attributes used by this module: {', '.join(used)}")

    if len(results) != 5:
        failures.append(f"expected 5 models, got {len(results)}")
    if not all_metrics:
        failures.append("some metrics are missing")
    if not in_range:
        failures.append("a metric fell outside [0, 1]")
    if not beats_baseline:
        failures.append("a model's PR-AUC did not beat the base rate")
    if missing:
        failures.append(f"missing artifacts: {missing}")
    if not test_untouched:
        failures.append("this module references config.TEST_CSV -- possible leakage")

    # ---------------------------------------------------------------- done
    rule("STAGE 3 RESULT")
    print(f"  total runtime: {total:.1f}s")
    if failures:
        print(f"  FAILED -- {len(failures)} problem(s):")
        for f in failures:
            print(f"    - {f}")
        return 1
    print("  All checks passed. Stage 3 complete.")
    print("  Next: Stage 4 evaluates these five models on the held-out test set.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
