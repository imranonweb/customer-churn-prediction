"""Stage 3B: SMOTE against the Stage 3 class-weighting strategy.

A supplementary experiment, not a rebuild. It answers one question -- does
synthetic minority oversampling beat cost re-weighting on this problem? -- and
changes nothing about the project's official results.

Training data only
------------------
The test set was already spent on the Stage 4 final evaluation. Using it again
to choose between two imbalance strategies would turn a held-out score into a
selection score, so this module reads `data/processed/train.csv` and nothing
else. The test split is never read; `run_stage3b.py` checks that with an AST
scan that classifies every use of the path, and with a runtime audit hook that
records every file the process opens.

What is held constant
---------------------
Both arms use Stage 3's own `build_models()` specification, Stage 3's own
`build_pipeline()` for the class-weight arm, and the same
`StratifiedKFold(5, shuffle=True, random_state=42)`. The SMOTE arm is built by
reading each Stage 3 estimator's parameters and overriding exactly one of them
-- the imbalance knob. `imbalance_change()` reports that difference so the
"only the strategy changed" claim is measurable rather than asserted.

Nothing is tuned here. No search, no Optuna, no threshold selection.

Why the class-weight arm is recomputed rather than read from Stage 3
-------------------------------------------------------------------
Reading `artifacts/model_cv_results.csv` would compare numbers produced by two
different code paths. Recomputing puts both arms on one set of folds and one
scorer, which is what makes the comparison controlled. `reproduction_deltas()`
then checks the recomputation against Stage 3's published figures, so if any
hyperparameter, fold, or metric definition had moved, the delta would show it.

Outputs:
    artifacts/smote_cv_results.csv
    reports/figures/smote_comparison.png
"""

from __future__ import annotations

import ast
import hashlib
import time
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.model_selection import StratifiedKFold, cross_validate

from src import config, evaluate, preprocess, train

# Stage 4's figure tokens and helpers, so this figure reads as part of the same
# set. Imported, never modified.
plt = evaluate.plt
SURFACE = evaluate.SURFACE
INK = evaluate.INK
INK_SECONDARY = evaluate.INK_SECONDARY
INK_MUTED = evaluate.INK_MUTED
GRIDLINE = evaluate.GRIDLINE
AXIS = evaluate.AXIS

# --------------------------------------------------------------------------
# Stage 3B outputs. Defined here because config.py is locked.
# --------------------------------------------------------------------------
SMOTE_CV_RESULTS_CSV = config.ARTIFACTS_DIR / "smote_cv_results.csv"
COMPARISON_FIGURE = "smote_comparison.png"

# --------------------------------------------------------------------------
# The experiment
# --------------------------------------------------------------------------
CLASS_WEIGHT = "Class Weight"
SMOTE_LABEL = "SMOTE"
STRATEGIES = (CLASS_WEIGHT, SMOTE_LABEL)

# XGBoost's imbalance knob switched off. 1.0 weights the two classes equally,
# which is the point: applying scale_pos_weight *and* SMOTE would correct the
# imbalance twice and the comparison would no longer be a comparison.
NEUTRAL_SCALE_POS_WEIGHT = 1.0

PRIMARY_METRIC = train.SELECTION_METRIC          # "PR-AUC"
SUPPORTING_METRICS = ["Recall", "F1"]
TABLE_METRICS = [PRIMARY_METRIC] + SUPPORTING_METRICS + ["ROC-AUC"]

# Panel order for the figure. Recall sits next to Precision on purpose: the
# expected SMOTE trade-off is recall up, precision down, and putting them side
# by side lets a reader see both halves of it at once.
PANEL_METRICS = ["PR-AUC", "Recall", "Precision", "F1", "ROC-AUC", "Accuracy"]

# A delta counts as "meaningful" only if it clears the fold-to-fold noise of the
# two arms it came from. This is a descriptive yardstick, not a significance
# test -- 5 folds is far too few to support one, and no p-value is claimed.
MEANINGFUL_SD_MULTIPLE = 1.0

# Two categorical hues for the two strategies. Blue is Stage 4's slot 1 (already
# the incumbent's colour in Stages 4 and 5), green is its slot 6.
# validate_palette.js, light mode on the #fcfcfb surface: lightness band PASS,
# chroma floor PASS, CVD separation PASS (protan dE 26.5), normal-vision floor
# PASS (dE 29.0), contrast PASS (both >= 3:1). The tritan margin is 7.6, inside
# the 6-8 floor band, which is legal only with secondary encoding -- hence the
# distinct marker shapes (circle vs square) and the legend.
STRATEGY_COLORS = {CLASS_WEIGHT: "#2a78d6", SMOTE_LABEL: "#008300"}
STRATEGY_MARKERS = {CLASS_WEIGHT: "o", SMOTE_LABEL: "s"}


# --------------------------------------------------------------------------
# Building the two arms
# --------------------------------------------------------------------------
def _comparable(value):
    """Compare nested estimators by class, everything else by value.

    Rebuilding an estimator produces a new object, so a plain equality test on
    `get_params(deep=True)` would flag the nested SVC as "changed" purely
    because its identity differs. Its own parameters still show up under
    `estimator__*`, which is where a real change would appear.
    """
    if hasattr(value, "get_params"):
        return type(value).__name__
    return repr(value)


def param_differences(left, right) -> list[str]:
    """Parameter names on which two estimators disagree."""
    left_params = left.get_params(deep=True)
    right_params = right.get_params(deep=True)
    return [
        key
        for key in sorted(set(left_params) | set(right_params))
        if _comparable(left_params.get(key)) != _comparable(right_params.get(key))
    ]


def smote_estimator(key: str, spec: dict):
    """Stage 3's estimator with its imbalance correction switched off.

    Built from `get_params()` of the Stage 3 object and rebuilt with exactly one
    key replaced, so every other hyperparameter is carried across by
    construction rather than by being retyped here.
    """
    base = spec["estimator"]
    params = base.get_params(deep=False)

    if key == "xgboost":
        params["scale_pos_weight"] = NEUTRAL_SCALE_POS_WEIGHT
    elif key == "svm":
        # The class weight lives on the SVC inside CalibratedClassifierCV, so
        # the inner estimator is the one that gets rebuilt.
        inner = base.estimator
        inner_params = inner.get_params(deep=False)
        inner_params["class_weight"] = None
        params["estimator"] = type(inner)(**inner_params)
    else:
        params["class_weight"] = None

    return type(base)(**params)


def imbalance_change(key: str, spec: dict) -> list[str]:
    """Which parameters differ between this model's two arms.

    Exactly one entry means the experiment is controlled: the imbalance knob
    moved and nothing else did.
    """
    return param_differences(spec["estimator"], smote_estimator(key, spec))


def build_pipeline(key: str, spec: dict, strategy: str):
    """The pipeline for one model under one strategy.

    Class Weight reuses Stage 3's own `build_pipeline`, so that arm is the Stage
    3 pipeline rather than a lookalike.

    SMOTE inserts a sampler between the preprocessor and the estimator:

        training fold -> preprocess -> SMOTE -> classifier
        validation fold -> preprocess -> classifier

    imbalanced-learn's Pipeline runs samplers during `fit` only and skips them
    for `predict`/`predict_proba`, which is what keeps the validation fold at
    its natural class balance. SMOTE has to sit *after* the preprocessor because
    it interpolates numerically and cannot read raw category strings.
    """
    if strategy == CLASS_WEIGHT:
        return train.build_pipeline(spec)
    return ImbPipeline(
        [
            ("prep", preprocess.build_preprocessor(scale=spec["scale"])),
            ("smote", SMOTE(random_state=config.SEED)),
            ("model", smote_estimator(key, spec)),
        ]
    )


def make_cv() -> StratifiedKFold:
    """The Stage 3 splitter, reused so both arms score on identical folds."""
    return StratifiedKFold(
        n_splits=train.N_SPLITS, shuffle=True, random_state=config.SEED
    )


# --------------------------------------------------------------------------
# Proving SMOTE runs inside the folds
# --------------------------------------------------------------------------
class ResampleLog(list):
    """A list that survives `clone` by identity rather than by copy.

    scikit-learn's `clone` deep-copies any parameter that is not itself an
    estimator, so a plain list handed to the sampler would be duplicated per
    fold and the records would never reach the caller. Returning `self` from
    `__deepcopy__` makes every fold's clone append to this one object.
    """

    def __deepcopy__(self, memo):
        return self


class RecordingSMOTE(SMOTE):
    """SMOTE that appends a record of every resample it performs.

    Used once, by `audit_smote_inside_cv()`, to observe the sampler from inside
    a real `cross_validate` call.
    """

    def __init__(self, *, sampling_strategy="auto", random_state=None,
                 k_neighbors=5, log=None):
        super().__init__(
            sampling_strategy=sampling_strategy,
            random_state=random_state,
            k_neighbors=k_neighbors,
        )
        self.log = log

    def fit_resample(self, X, y):
        resampled_X, resampled_y = super().fit_resample(X, y)
        if self.log is not None:
            counts = np.bincount(np.asarray(resampled_y).astype(int), minlength=2)
            self.log.append(
                {
                    "rows_in": int(np.asarray(X).shape[0]),
                    "rows_out": int(np.asarray(resampled_X).shape[0]),
                    "negatives_out": int(counts[0]),
                    "positives_out": int(counts[1]),
                }
            )
        return resampled_X, resampled_y


def audit_smote_inside_cv(
    X: pd.DataFrame, y: pd.Series, key: str = "decision_tree"
) -> dict:
    """Watch the sampler during a real 5-fold run and check where it fired.

    Runs single-process on purpose: with `n_jobs=-1` the folds execute in worker
    processes and their records would never come back. The cheapest model is
    used because the audit is about the sampler's placement, not about scores.

    The arithmetic is the actual proof. Across 5 folds the sampler should see
    every row exactly `n_splits - 1` times -- each row is in the training part
    of 4 folds out of 5 -- so the input rows must total `4 x len(X)`. If SMOTE
    had been applied to the whole training set before splitting, or to the
    validation folds as well, or to a concatenation involving the test set, that
    total could not come out right.
    """
    log = ResampleLog()
    spec = train.build_models(train.compute_scale_pos_weight(y))[key]
    pipeline = ImbPipeline(
        [
            ("prep", preprocess.build_preprocessor(scale=spec["scale"])),
            ("smote", RecordingSMOTE(random_state=config.SEED, log=log)),
            ("model", smote_estimator(key, spec)),
        ]
    )
    cross_validate(
        pipeline,
        X,
        y,
        cv=make_cv(),
        scoring=["average_precision"],
        n_jobs=1,
        error_score="raise",
    )

    rows_in = [record["rows_in"] for record in log]
    expected_total = (train.N_SPLITS - 1) * len(X)
    # `recorded` gates every check below. Without it an empty log would make
    # each `all(...)` vacuously true and the audit would pass by seeing nothing.
    recorded = len(log) > 0
    return {
        "model": spec["name"],
        "recorded_anything": recorded,
        "resamples": len(log),
        "expected_resamples": train.N_SPLITS,
        "one_per_training_fold": len(log) == train.N_SPLITS,
        "rows_in_total": sum(rows_in),
        "rows_in_expected": expected_total,
        "rows_add_up": sum(rows_in) == expected_total,
        "largest_fold_seen": max(rows_in) if rows_in else 0,
        "training_rows": len(X),
        "never_saw_full_training_set": recorded and max(rows_in) < len(X),
        "grew_every_fold": recorded and all(
            record["rows_out"] > record["rows_in"] for record in log
        ),
        "balanced_output": recorded and all(
            record["negatives_out"] == record["positives_out"] for record in log
        ),
        "log": list(log),
    }


# --------------------------------------------------------------------------
# Cross-validation
# --------------------------------------------------------------------------
def cross_validate_arm(pipeline, X: pd.DataFrame, y: pd.Series) -> dict:
    """Five-fold CV for one pipeline, returning the six metric means and stds.

    Identical `cross_validate` arguments to Stage 3, which is what allows the
    class-weight arm to reproduce Stage 3's published numbers exactly.
    """
    scores = cross_validate(
        pipeline,
        X,
        y,
        cv=make_cv(),
        scoring=list(train.METRICS.values()),
        n_jobs=-1,
        error_score="raise",
    )
    row: dict[str, float] = {}
    for display, scorer in train.METRICS.items():
        fold_scores = scores[f"test_{scorer}"]
        row[display] = float(np.mean(fold_scores))
        row[f"{display}_std"] = float(np.std(fold_scores))
    return row


def run_experiment(X: pd.DataFrame, y: pd.Series, report=print) -> pd.DataFrame:
    """Both strategies, all five models, one row per model/strategy pair."""
    models = train.build_models(train.compute_scale_pos_weight(y))
    rows = []
    for key, spec in models.items():
        for strategy in STRATEGIES:
            started = time.perf_counter()
            metrics = cross_validate_arm(
                build_pipeline(key, spec, strategy), X, y
            )
            elapsed = time.perf_counter() - started
            rows.append(
                {"key": key, "Model": spec["name"], "Strategy": strategy, **metrics}
            )
            report(
                f"  {spec['name']:<20} {strategy:<12} {elapsed:5.1f}s   "
                + "  ".join(
                    f"{m} {metrics[m]:.4f}" for m in (PRIMARY_METRIC, "Recall", "F1")
                )
            )
    return pd.DataFrame(rows)


def comparison_table(results: pd.DataFrame) -> pd.DataFrame:
    """One row per model: each arm's metrics, the delta, and a noise yardstick.

    `pooled_sd` combines the two arms' fold-to-fold standard deviations. A delta
    smaller than that is inside the spread the same experiment produces just by
    changing which rows land in which fold, so it is reported as negligible.
    """
    rows = []
    for key, group in results.groupby("key", sort=False):
        weighted = group[group["Strategy"] == CLASS_WEIGHT].iloc[0]
        smote = group[group["Strategy"] == SMOTE_LABEL].iloc[0]
        row = {"key": key, "Model": weighted["Model"]}
        for metric in train.METRICS:
            delta = float(smote[metric] - weighted[metric])
            pooled_sd = float(
                np.sqrt(
                    (weighted[f"{metric}_std"] ** 2 + smote[f"{metric}_std"] ** 2) / 2
                )
            )
            row[f"{metric}_cw"] = float(weighted[metric])
            row[f"{metric}_smote"] = float(smote[metric])
            row[f"{metric}_delta"] = delta
            row[f"{metric}_pooled_sd"] = pooled_sd
            row[f"{metric}_meaningful"] = bool(
                abs(delta) > MEANINGFUL_SD_MULTIPLE * pooled_sd
            )
        rows.append(row)
    return pd.DataFrame(rows)


def save_results(results: pd.DataFrame) -> Path:
    """Write the required columns first, then the fold standard deviations."""
    ordered = (
        ["Model", "Strategy"]
        + list(train.METRICS)
        + [f"{metric}_std" for metric in train.METRICS]
    )
    config.ensure_dirs()
    results[ordered].to_csv(SMOTE_CV_RESULTS_CSV, index=False)
    return SMOTE_CV_RESULTS_CSV


def reproduction_deltas(results: pd.DataFrame) -> pd.DataFrame:
    """Class-weight arm against Stage 3's stored CV results, model by model.

    A zero delta is the falsifiable evidence that this stage reused Stage 3's
    hyperparameters, folds, and metric definitions untouched. A non-zero one
    would mean something moved, and the comparison against SMOTE would be
    measuring that instead of measuring the strategy.
    """
    stage3 = pd.read_csv(train.CV_RESULTS_CSV)
    arm = results[results["Strategy"] == CLASS_WEIGHT]
    rows = []
    for _, recomputed in arm.iterrows():
        stored = stage3[stage3["Model"] == recomputed["Model"]]
        if stored.empty:
            rows.append({"Model": recomputed["Model"], "max_abs_delta": float("nan"),
                         "found_in_stage_3": False})
            continue
        stored_row = stored.iloc[0]
        deltas = [
            abs(float(recomputed[metric]) - float(stored_row[metric]))
            for metric in train.METRICS
        ]
        rows.append(
            {
                "Model": recomputed["Model"],
                "max_abs_delta": max(deltas),
                "found_in_stage_3": True,
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Figure
# --------------------------------------------------------------------------
def plot_comparison(table: pd.DataFrame, n_rows: int) -> Path:
    """One figure, six panels: Class Weight vs SMOTE for every metric.

    A dumbbell per model, because the question is about movement between two
    states of the same model rather than about six independent bars. Position
    encodes the score, so the axis does not have to start at zero -- but every
    panel shares one x-range, so a small change looks small in every panel
    instead of being magnified by a per-panel zoom.
    """
    n_cols = 3
    n_panel_rows = 2
    fig, axes = plt.subplots(
        n_panel_rows, n_cols, figsize=(14.6, 8.8), dpi=160
    )
    fig.patch.set_facecolor(SURFACE)

    values = np.concatenate(
        [
            table[[f"{metric}_cw", f"{metric}_smote"]].to_numpy().ravel()
            for metric in PANEL_METRICS
        ]
    )
    span = values.max() - values.min()
    # Right-hand pad is wider: the delta label sits outside the rightmost dot.
    low, high = values.min() - span * 0.10, values.max() + span * 0.30

    order = list(table.index)[::-1]  # first model at the top

    for position, metric in enumerate(PANEL_METRICS):
        ax = axes[position // n_cols][position % n_cols]
        evaluate._style_axes(ax, grid_axis="x")
        ax.set_xlim(low, high)
        ax.set_ylim(-0.7, len(table) - 0.3)

        for y_position, index in enumerate(order):
            row = table.loc[index]
            weighted = row[f"{metric}_cw"]
            smote = row[f"{metric}_smote"]

            ax.plot([weighted, smote], [y_position, y_position],
                    color=AXIS, linewidth=2.0, solid_capstyle="round", zorder=2)
            for strategy, value in ((CLASS_WEIGHT, weighted), (SMOTE_LABEL, smote)):
                ax.plot(value, y_position, marker=STRATEGY_MARKERS[strategy],
                        markersize=8, color=STRATEGY_COLORS[strategy],
                        markeredgecolor=SURFACE, markeredgewidth=1.6, zorder=3)

            delta = row[f"{metric}_delta"]
            ax.annotate(
                f"{delta:+.3f}",
                xy=(max(weighted, smote), y_position),
                xytext=(9, 0), textcoords="offset points",
                ha="left", va="center", fontsize=8.5,
                color=INK if row[f"{metric}_meaningful"] else INK_MUTED,
                fontweight="bold" if row[f"{metric}_meaningful"] else "normal",
            )

        ax.set_yticks(range(len(table)))
        ax.set_yticklabels([table.loc[i, "Model"] for i in order],
                           fontsize=9, color=INK_SECONDARY)
        ax.tick_params(axis="y", length=0)
        marker = "  (primary)" if metric == PRIMARY_METRIC else ""
        ax.set_title(f"{metric}{marker}", color=INK, fontsize=11,
                     fontweight="bold", loc="left", pad=8)
        if position % n_cols:
            ax.set_yticklabels([])

    handles = [
        plt.Line2D([], [], marker=STRATEGY_MARKERS[strategy], linestyle="none",
                   markersize=8, color=STRATEGY_COLORS[strategy],
                   markeredgecolor=SURFACE, markeredgewidth=1.6, label=strategy)
        for strategy in STRATEGIES
    ]
    legend = fig.legend(
        handles=handles, loc="upper right", bbox_to_anchor=(0.995, 0.998),
        frameon=False, ncols=2, fontsize=9.5, handletextpad=0.4,
        columnspacing=1.4,
    )
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)

    # The legend sits on the title's row; the subtitle gets its own two lines
    # below it, so neither can run into the other.
    fig.suptitle(
        "Class weighting vs SMOTE, five-fold cross-validation",
        color=INK, fontsize=14, fontweight="bold", x=0.043, ha="left", y=0.988,
    )
    fig.text(
        0.043, 0.918,
        f"training split only ({n_rows:,} rows) · all six panels share one "
        f"x-range, so a small change reads as small\n"
        f"labels are SMOTE minus class weight, bold where the change exceeds "
        f"the folds' own fold-to-fold spread",
        color=INK_SECONDARY, fontsize=9.5, ha="left", va="bottom",
        linespacing=1.5,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.918))
    return evaluate._save(fig, COMPARISON_FIGURE)


# --------------------------------------------------------------------------
# Findings
# --------------------------------------------------------------------------
def _direction(delta: float, meaningful: bool) -> str:
    if not meaningful:
        return "unchanged within fold noise"
    return "improved" if delta > 0 else "reduced"


def per_model_findings(table: pd.DataFrame) -> list[dict]:
    """For each model, what SMOTE did to PR-AUC, Recall, and F1."""
    findings = []
    for _, row in table.iterrows():
        entry = {"Model": row["Model"]}
        for metric in [PRIMARY_METRIC] + SUPPORTING_METRICS:
            entry[metric] = {
                "delta": row[f"{metric}_delta"],
                "pooled_sd": row[f"{metric}_pooled_sd"],
                "meaningful": row[f"{metric}_meaningful"],
                "verdict": _direction(
                    row[f"{metric}_delta"], row[f"{metric}_meaningful"]
                ),
            }
        findings.append(entry)
    return findings


def best_overall(results: pd.DataFrame) -> pd.Series:
    """The single highest-PR-AUC model/strategy pair in the experiment."""
    return results.loc[results[PRIMARY_METRIC].idxmax()]


def findings(results: pd.DataFrame, table: pd.DataFrame) -> list[str]:
    """The five required answers, written from the computed numbers."""
    best = best_overall(results)
    official = results[
        (results["key"] == "random_forest") & (results["Strategy"] == CLASS_WEIGHT)
    ].iloc[0]

    primary_deltas = table[f"{PRIMARY_METRIC}_delta"]
    improved_primary = table[primary_deltas > 0]
    meaningful_primary = table[table[f"{PRIMARY_METRIC}_meaningful"]]
    recall_deltas = table["Recall_delta"]
    f1_deltas = table["F1_delta"]
    precision_deltas = table["Precision_delta"]

    biggest_gain = table.loc[primary_deltas.idxmax()]
    biggest_loss = table.loc[primary_deltas.idxmin()]
    recall_winner = table.loc[recall_deltas.idxmax()]

    out = []

    # 1
    out.append(
        f"Highest PR-AUC in the experiment: {best['Model']} with "
        f"{best['Strategy']} at {best[PRIMARY_METRIC]:.4f}. The official model's "
        f"arm -- Random Forest with class weighting -- scored "
        f"{official[PRIMARY_METRIC]:.4f}, a difference of "
        f"{best[PRIMARY_METRIC] - official[PRIMARY_METRIC]:+.4f}."
    )

    # 2
    out.append(
        f"How much SMOTE moved PR-AUC: from {primary_deltas.min():+.4f} "
        f"({biggest_loss['Model']}) to {primary_deltas.max():+.4f} "
        f"({biggest_gain['Model']}), mean {primary_deltas.mean():+.4f} across the "
        f"five models. It raised PR-AUC for {len(improved_primary)} of 5 and "
        f"lowered it for {5 - len(improved_primary)}. Changes clearing the "
        f"fold-to-fold spread: {len(meaningful_primary)} of 5"
        + (
            ""
            if meaningful_primary.empty
            else " (" + ", ".join(meaningful_primary["Model"]) + ")"
        )
        + "."
    )

    # 3
    out.append(
        f"Did SMOTE improve recall? Recall rose for "
        f"{int((recall_deltas > 0).sum())} of 5 models, range "
        f"{recall_deltas.min():+.4f} to {recall_deltas.max():+.4f}, largest gain "
        f"{recall_winner['Model']} at {recall_winner['Recall_delta']:+.4f}. Over "
        f"the same models precision moved {precision_deltas.min():+.4f} to "
        f"{precision_deltas.max():+.4f} (mean {precision_deltas.mean():+.4f}), so "
        f"recall was not gained for free."
    )

    # 4
    out.append(
        f"Did SMOTE improve F1? F1 rose for {int((f1_deltas > 0).sum())} of 5 "
        f"models, range {f1_deltas.min():+.4f} to {f1_deltas.max():+.4f}, mean "
        f"{f1_deltas.mean():+.4f}. F1 is the balance point between the recall and "
        f"precision movements above, which is why it moves less than either."
    )

    # 5
    if meaningful_primary.empty:
        verdict = (
            "Negligible on the primary metric. Every PR-AUC change is smaller "
            "than the pooled fold-to-fold standard deviation of the two arms it "
            "came from, so the same experiment would produce differences of this "
            "size just by reshuffling which rows land in which fold."
        )
    else:
        verdict = (
            f"{len(meaningful_primary)} of 5 PR-AUC changes exceed the pooled "
            f"fold-to-fold standard deviation of their two arms: "
            + "; ".join(
                f"{row['Model']} {row[f'{PRIMARY_METRIC}_delta']:+.4f} against "
                f"sd {row[f'{PRIMARY_METRIC}_pooled_sd']:.4f}"
                for _, row in meaningful_primary.iterrows()
            )
            + ". The remainder sit inside fold noise."
        )
    out.append(
        "Meaningful or negligible? "
        + verdict
        + " Five folds cannot support a significance test and none is claimed; "
        "this is a descriptive comparison against the spread the folds "
        "themselves show."
    )

    # Interpretation, as required.
    out.append(
        "Reading these numbers: SMOTE synthesises minority examples, so a model "
        "trained on the resampled fold sees a 50/50 balance and predicts churn "
        "more readily. That is why recall can rise while precision falls -- the "
        "extra positives include false ones. Recall alone therefore cannot "
        "decide the comparison. PR-AUC stays the primary metric because it "
        "summarises the precision/recall trade-off across every threshold on the "
        "minority class, which is the quantity this problem actually cares "
        "about at a 26.5% churn rate."
    )
    out.append(
        "These are associations measured on cross-validated training folds. They "
        "describe how each strategy changed the models' scores. They do not show "
        "that resampling causes better or worse churn prediction in general, and "
        "nothing here is evidence about what causes a customer to churn."
    )
    out.append(
        "One methodological caveat worth recording: SMOTE interpolates between "
        "neighbouring rows, and after one-hot encoding it therefore produces "
        "fractional values in columns that can only be 0 or 1 -- a synthetic "
        "customer can come out 0.4 of the way into a month-to-month contract. "
        "Plain SMOTE was specified for this experiment and that is what ran; the "
        "effect is a known limitation of applying it to encoded categoricals, "
        "and it is one plausible reason the resampled arm did not pull ahead."
    )
    return out


def official_model_statement(results: pd.DataFrame, table: pd.DataFrame) -> dict:
    """Whether the experiment justifies reconsidering the Stage 4 choice.

    Deliberately returns a recommendation and not an action. Replacing the final
    model would mean re-running Stage 4 on a test set already spent, so the
    decision belongs to a human.
    """
    best = best_overall(results)
    official = results[
        (results["key"] == "random_forest") & (results["Strategy"] == CLASS_WEIGHT)
    ].iloc[0]
    forest = table[table["key"] == "random_forest"].iloc[0]

    margin = float(best[PRIMARY_METRIC] - official[PRIMARY_METRIC])
    forest_delta = float(forest[f"{PRIMARY_METRIC}_delta"])
    forest_sd = float(forest[f"{PRIMARY_METRIC}_pooled_sd"])
    smote_beats_official = bool(
        (results["Strategy"] == SMOTE_LABEL).any()
        and results[results["Strategy"] == SMOTE_LABEL][PRIMARY_METRIC].max()
        > official[PRIMARY_METRIC] + forest_sd
    )
    return {
        "best_model": best["Model"],
        "best_strategy": best["Strategy"],
        "best_pr_auc": float(best[PRIMARY_METRIC]),
        "official_pr_auc": float(official[PRIMARY_METRIC]),
        "margin": margin,
        "forest_delta": forest_delta,
        "forest_pooled_sd": forest_sd,
        "forest_change_meaningful": bool(forest[f"{PRIMARY_METRIC}_meaningful"]),
        "consider_replacement": smote_beats_official,
    }


# --------------------------------------------------------------------------
# Integrity helpers
# --------------------------------------------------------------------------
def file_checksums(paths) -> dict[str, str]:
    """sha256 for each existing path, keyed by name."""
    checksums = {}
    for path in paths:
        path = Path(path)
        if path.exists():
            checksums[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return checksums


def protected_files() -> list[Path]:
    """Stage 1-5 outputs that Stage 3B must leave byte-identical.

    Stage 3B's own figure is excluded: this stage rewrites it by design, so
    including it would make the check fail for the wrong reason the moment the
    figure stopped being byte-deterministic.

    `config.TEST_CSV` is deliberately absent too. Hashing it would mean opening
    it, which would register in `run_stage3b.py`'s file-access audit and blunt
    the strongest claim available -- that the test split was never opened at all,
    for reading or for writing. It is checked with `test_csv_stat()` instead,
    which needs no read.
    """
    figures = [
        path
        for path in sorted(config.FIGURES_DIR.glob("*.png"))
        if path.name != COMPARISON_FIGURE
    ]
    return (
        [config.TRAIN_CSV]
        + [evaluate.model_path(key) for key in evaluate.MODELS]
        + [train.CV_RESULTS_CSV, train.METADATA_JSON, evaluate.TEST_RESULTS_CSV]
        + figures
    )


def test_csv_stat() -> dict:
    """Size and modification time of the test split, read without opening it.

    `stat()` does not open the file, so this can run alongside the audit hook
    without registering an access. An unchanged mtime is what "never written"
    looks like from the filesystem's side; the audit hook covers reads.
    """
    info = config.TEST_CSV.stat()
    return {"size": info.st_size, "mtime_ns": info.st_mtime_ns}


def stage3b_files() -> list[Path]:
    """The two files this stage created, for the AST scans below."""
    return [
        config.PROJECT_ROOT / "src" / "smote_experiment.py",
        config.PROJECT_ROOT / "run_stage3b.py",
    ]


def called_names(path: Path, names: set[str]) -> list[str]:
    """Call sites in a file matching any of `names`, from its syntax tree.

    Catches both `thing.name(...)` and a bare `name(...)`, because an import
    makes the second form available and Stage 4's `_attribute_calls` only sees
    the first. Reading the AST means a mention in a comment or docstring cannot
    affect the result -- only real calls count.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    hits = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr in names:
            hits.append(f"line {node.lineno}: .{func.attr}()")
        elif isinstance(func, ast.Name) and func.id in names:
            hits.append(f"line {node.lineno}: {func.id}()")
    return hits


# Operations that cannot read a file's contents. Anything else applied to the
# test-split path -- including anything unrecognised -- counts as a read, so
# `test_split_uses()` fails closed rather than passing on a use it does not
# understand.
NON_READING_USES = {
    "stat",         # metadata only; does not open the file
    "exists",
    "resolve",
    "relative_to",
    "name",
    "opened",       # run_stage3b's set-membership test on a normalised path
}

# Anything that would consume the test split as data if handed its path.
READING_CALLS = {
    "open", "read_csv", "read_table", "read_parquet", "read_text", "read_bytes",
    "loadtxt", "genfromtxt", "load", "load_test",
}


def _parent_map(tree) -> dict:
    parents = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    return parents


def test_split_uses(paths=None) -> dict[str, list[dict]]:
    """Every use of the test-split path in Stage 3B, classified read or not.

    The claim under test is not "the name `TEST_CSV` never appears" -- the
    integrity checks themselves have to name the file to prove it was left
    alone. The claim is that the path is never used to *read* the data. So each
    site is reported with the operation performed on it, and any operation
    outside `NON_READING_USES` counts as a read.

    A bare `load_test()` call is reported too, since it reaches the test split
    without mentioning the path.

    `paths` defaults to the two Stage 3B files and exists so the classifier can
    be pointed at a known-bad file to confirm it still fails.
    """
    uses = {}
    for path in paths or stage3b_files():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        parents = _parent_map(tree)
        records = []

        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Attribute)
                and node.attr == "TEST_CSV"
                and isinstance(node.value, ast.Name)
                and node.value.id == "config"
            ):
                continue
            parent = parents.get(node)
            if isinstance(parent, ast.Attribute):
                use, reading = f".{parent.attr}", parent.attr not in NON_READING_USES
            elif isinstance(parent, ast.Call):
                name = getattr(parent.func, "id", getattr(parent.func, "attr", "?"))
                use, reading = f"{name}(...)", name not in NON_READING_USES
            else:
                use, reading = type(parent).__name__, True
            records.append({"line": node.lineno, "use": use, "reading": reading})

        # `load_test()` reaches the test split without naming the path, so it is
        # matched on the call itself.
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = getattr(node.func, "id", getattr(node.func, "attr", None))
                if name == "load_test":
                    records.append(
                        {"line": node.lineno, "use": "load_test()", "reading": True}
                    )

        uses[str(path.relative_to(config.PROJECT_ROOT))] = sorted(
            records, key=lambda record: record["line"]
        )
    return uses


# Search / tuning APIs. None of these may be called: the experiment reuses Stage
# 3's hyperparameters and changes only the imbalance strategy.
TUNING_APIS = {
    "GridSearchCV", "RandomizedSearchCV", "HalvingGridSearchCV",
    "HalvingRandomSearchCV", "BayesSearchCV", "optuna", "create_study",
    "suggest_float", "suggest_int", "suggest_categorical",
}


def check_no_tuning() -> dict[str, list[str]]:
    """No search or tuning API is called anywhere in Stage 3B."""
    return {
        str(path.relative_to(config.PROJECT_ROOT)): called_names(path, TUNING_APIS)
        for path in stage3b_files()
    }
