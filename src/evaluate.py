"""Stage 4: evaluate the Stage 3 pipelines on the held-out test set.

Run from the project root:

    python run_stage4.py

What this module may and may not do
-----------------------------------
This is the **first** stage allowed to read `data/processed/test.csv`. The test
set has never been fitted on, cross-validated on, or used to pick a model, so
the numbers it produces are an honest estimate rather than a tuned one.

To keep that true, this module:

* loads the five saved pipelines with `joblib` and uses them **exactly as
  saved** -- it never calls `.fit()`, never changes a hyperparameter, and never
  re-selects a model after seeing the test scores;
* touches each pipeline **once**, computing `predict` and `predict_proba` in a
  single pass and deriving every metric, curve, and confusion matrix from those
  stored arrays;
* records a SHA-256 of every `.pkl` before and after, so "the models were not
  modified" is measured rather than asserted.

`run_stage4.py` prints those checks, including an AST scan of this file and of
the runner proving no fit-family call exists in Stage 4 code at all.

Figure style
------------
Colours come from a validated categorical palette: five hues taken in their
documented order, checked with a colour-vision-deficiency validator on both the
adjacent and all-pairs pairlists (worst normal-vision Delta E 15.6, worst CVD
6.9). Because the all-pairs CVD margin sits in the 6-8 band, the curve plots
carry a second, non-colour encoding -- a distinct dash pattern per model -- so
the five models stay separable in greyscale, in print, and for colourblind
readers. Identity is never colour-alone.
"""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # headless: write files, never open a window

import matplotlib.pyplot as plt  # noqa: E402  (must follow the backend choice)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline  # noqa: E402

from src import config  # noqa: E402

# --------------------------------------------------------------------------
# Stage 3 artifacts consumed, Stage 4 artifacts produced
# --------------------------------------------------------------------------
CV_RESULTS_CSV = config.ARTIFACTS_DIR / "model_cv_results.csv"
METADATA_JSON = config.ARTIFACTS_DIR / "model_metadata.json"
TEST_RESULTS_CSV = config.ARTIFACTS_DIR / "test_results.csv"

# Artifact stem -> display name. Order is the order the proposal lists them in.
MODELS: dict[str, str] = {
    "logistic_regression": "Logistic Regression",
    "decision_tree": "Decision Tree",
    "random_forest": "Random Forest",
    "svm": "SVM",
    "xgboost": "XGBoost",
}

METRICS = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"]

# PR-AUC decides. On a 26.5% positive class, accuracy rewards predicting the
# majority label, so it cannot be the selection metric.
SELECTION_METRIC = "PR-AUC"

# Methods that would mean a model was refitted or reconfigured here.
FIT_METHODS = {"fit", "fit_transform", "fit_predict", "partial_fit", "set_params"}

# --------------------------------------------------------------------------
# Figure tokens (see the module docstring on how the palette was chosen)
# --------------------------------------------------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
AXIS = "#c3c2b7"

MODEL_COLORS = {
    "logistic_regression": "#2a78d6",  # slot 1, blue
    "decision_tree": "#1baf7a",        # slot 3, aqua
    "random_forest": "#008300",        # slot 6, green
    "svm": "#4a3aa7",                  # slot 7, violet
    "xgboost": "#e34948",              # slot 8, red
}

# Secondary encoding, required because the all-pairs CVD margin is in the 6-8
# band: the curves remain distinguishable with no colour at all.
MODEL_DASHES = {
    "logistic_regression": (0, ()),                       # solid
    "decision_tree": (0, (5, 2)),                         # dashed
    "random_forest": (0, (1, 1.7)),                        # dotted
    "svm": (0, (7, 2, 1.5, 2)),                            # dash-dot
    "xgboost": (0, (3, 1.4, 1.4, 1.4, 1.4, 1.4)),          # dash-dot-dot
}

# Documented single-hue sequential ramp, light -> dark, for the confusion cells.
BLUE_RAMP = LinearSegmentedColormap.from_list(
    "seq_blue",
    ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
     "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281",
     "#0d366b"],
)


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def model_path(key: str) -> Path:
    return config.ARTIFACTS_DIR / f"{key}.pkl"


def model_checksums() -> dict[str, str]:
    """SHA-256 of every saved pipeline, for before/after comparison."""
    return {
        key: hashlib.sha256(model_path(key).read_bytes()).hexdigest()
        for key in MODELS
        if model_path(key).exists()
    }


def load_models() -> dict[str, Pipeline]:
    """Load the five Stage 3 pipelines. No fitting, no reconfiguration."""
    missing = [key for key in MODELS if not model_path(key).exists()]
    if missing:
        raise FileNotFoundError(
            f"missing Stage 3 artifacts: {missing}. Run `python -m src.train` first."
        )
    return {key: joblib.load(model_path(key)) for key in MODELS}


def load_test() -> tuple[pd.DataFrame, pd.Series]:
    """Read the held-out split. Same target and feature definition as Stage 3.

    No manual preprocessing: each saved object is a full Pipeline, so it applies
    the transformer it was fitted with on the training data.
    """
    if not config.TEST_CSV.exists():
        raise FileNotFoundError(
            f"{config.TEST_CSV} not found. Run `python run_stage1.py` first."
        )
    test_df = pd.read_csv(config.TEST_CSV)
    return test_df[config.FEATURE_COLS], test_df[config.TARGET]


# --------------------------------------------------------------------------
# Evaluation -- one pass over the test set per model
# --------------------------------------------------------------------------
def evaluate_models(
    models: dict[str, Pipeline], X: pd.DataFrame
) -> tuple[dict[str, dict], list[str]]:
    """Score every model once and keep the raw outputs.

    Everything downstream (metrics, curves, confusion matrices) is derived from
    the arrays returned here, so the test set is passed through each pipeline
    exactly once. `call_log` records that pass, one entry per model.
    """
    predictions: dict[str, dict] = {}
    call_log: list[str] = []
    for key, pipe in models.items():
        call_log.append(key)
        predictions[key] = {
            "y_pred": pipe.predict(X),
            # Every saved pipeline exposes predict_proba -- including SVM, whose
            # Stage 3 pipeline wraps SVC in a calibrator for exactly this.
            "y_score": pipe.predict_proba(X)[:, 1],
        }
    return predictions, call_log


def compute_metrics(y_true: pd.Series, y_pred: np.ndarray,
                    y_score: np.ndarray) -> dict[str, float]:
    """The six required metrics. Ranking metrics use probabilities."""
    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred)),
        "F1": float(f1_score(y_true, y_pred)),
        "ROC-AUC": float(roc_auc_score(y_true, y_score)),
        "PR-AUC": float(average_precision_score(y_true, y_score)),
    }


def results_frame(predictions: dict[str, dict], y: pd.Series) -> pd.DataFrame:
    """Build the test-results table, one row per model."""
    rows = []
    for key, name in MODELS.items():
        row = {"key": key, "Model": name}
        row.update(compute_metrics(y, predictions[key]["y_pred"],
                                   predictions[key]["y_score"]))
        rows.append(row)
    return pd.DataFrame(rows)


def save_results(results: pd.DataFrame) -> Path:
    """Write artifacts/test_results.csv with exactly the required columns."""
    config.ensure_dirs()
    results[["Model"] + METRICS].to_csv(TEST_RESULTS_CSV, index=False)
    return TEST_RESULTS_CSV


def best_by(results: pd.DataFrame, metric: str) -> tuple[str, str, float]:
    """(key, display name, value) of the top model on `metric`."""
    row = results.loc[results[metric].idxmax()]
    return row["key"], row["Model"], float(row[metric])


def ranking(results: pd.DataFrame, metric: str) -> list[tuple[str, float]]:
    """Models ordered best-first on `metric`."""
    ordered = results.sort_values(metric, ascending=False)
    return list(zip(ordered["Model"], ordered[metric].astype(float)))


def cv_vs_test(results: pd.DataFrame) -> pd.DataFrame:
    """Join the Stage 3 CV PR-AUC to the test PR-AUC. Reads, never writes."""
    if not CV_RESULTS_CSV.exists():
        raise FileNotFoundError(f"{CV_RESULTS_CSV} not found (Stage 3 artifact).")
    cv = pd.read_csv(CV_RESULTS_CSV)[["Model", "PR-AUC"]].rename(
        columns={"PR-AUC": "CV PR-AUC"}
    )
    test = results[["Model", "PR-AUC"]].rename(columns={"PR-AUC": "Test PR-AUC"})
    merged = cv.merge(test, on="Model", how="outer")
    merged["Difference"] = merged["Test PR-AUC"] - merged["CV PR-AUC"]
    merged["CV rank"] = merged["CV PR-AUC"].rank(ascending=False).astype(int)
    merged["Test rank"] = merged["Test PR-AUC"].rank(ascending=False).astype(int)
    return merged


# --------------------------------------------------------------------------
# Integrity checks
# --------------------------------------------------------------------------
def _attribute_calls(path: Path, names: set[str]) -> list[str]:
    """Every `<something>.<name>(...)` call site in a file, from its AST.

    Reading the syntax tree rather than the text means a mention of `.fit()` in
    a comment or docstring cannot affect the result -- only real calls count.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return [
        f"line {node.lineno}: .{node.func.attr}()"
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in names
    ]


def check_no_fit_calls() -> dict[str, list[str]]:
    """No fit-family call may exist anywhere in Stage 4 code."""
    return {
        rel: _attribute_calls(config.PROJECT_ROOT / rel, FIT_METHODS)
        for rel in ("src/evaluate.py", "run_stage4.py")
    }


def count_prediction_call_sites() -> dict[str, int]:
    """How many places in this module can touch a pipeline's outputs.

    One `predict` and one `predict_proba` call site, both inside
    `evaluate_models`, is what "the test set is evaluated once per model" means
    structurally -- there is nowhere else that could score it again.
    """
    counts = {"predict": 0, "predict_proba": 0}
    for site in _attribute_calls(Path(__file__), set(counts)):
        counts[site.rsplit(".", 1)[1].rstrip("()")] += 1
    return counts


def _same_param(recorded, loaded) -> bool:
    """Equality that treats NaN as equal to NaN.

    XGBoost's `missing` parameter defaults to NaN, and `nan != nan`, so plain
    `==` would report a difference where the value is in fact unchanged.
    """
    both_nan = (isinstance(recorded, float) and recorded != recorded
                and isinstance(loaded, float) and loaded != loaded)
    return both_nan or recorded == loaded


def check_hyperparameters_unchanged(
    models: dict[str, Pipeline]
) -> dict[str, list[str]]:
    """Compare each loaded estimator's params to what Stage 3 recorded."""
    if not METADATA_JSON.exists():
        raise FileNotFoundError(f"{METADATA_JSON} not found (Stage 3 artifact).")
    metadata = json.loads(METADATA_JSON.read_text(encoding="utf-8"))

    differences: dict[str, list[str]] = {}
    for key, pipe in models.items():
        recorded = metadata["models"][key]["params"]
        actual = {
            k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v))
            for k, v in pipe.named_steps["model"].get_params(deep=False).items()
        }
        differences[key] = sorted(
            f"{k}: stage3={recorded.get(k)!r} loaded={actual[k]!r}"
            for k in actual
            if k in recorded and not _same_param(recorded[k], actual[k])
        )
    return differences


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
def _style_axes(ax: plt.Axes, grid_axis: str = "both") -> None:
    """Recessive grid and axes so the data carries the ink."""
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(colors=INK_MUTED, labelsize=9, length=3, width=1.0)
    ax.grid(True, axis=grid_axis, color=GRIDLINE, linewidth=1.0, zorder=0)
    ax.set_axisbelow(True)


def _new_figure(figsize: tuple[float, float]) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=figsize, dpi=160)
    fig.patch.set_facecolor(SURFACE)
    return fig, ax


def _titles(ax: plt.Axes, title: str, subtitle: str) -> None:
    """Title above a smaller subtitle, both left-aligned to the plot area.

    Offsets are in points, not axes fractions, so the two never collide
    whatever the figure's aspect ratio happens to be.
    """
    ax.set_title(title, color=INK, fontsize=13, fontweight="bold", loc="left",
                 pad=32)
    ax.annotate(subtitle, xy=(0, 1), xycoords="axes fraction",
                xytext=(0, 9), textcoords="offset points",
                ha="left", va="bottom", color=INK_SECONDARY, fontsize=9.5)


def _save(fig: plt.Figure, filename: str) -> Path:
    config.ensure_dirs()
    path = config.FIGURES_DIR / filename
    fig.savefig(path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_confusion_matrix(
    key: str, y_true: pd.Series, y_pred: np.ndarray, metrics: dict[str, float]
) -> Path:
    """One presentation-ready confusion matrix.

    Cells are shaded by percentage **within the actual class**, not by raw
    count. With 73.5% of the test set in one row, count-shading would leave the
    churn row almost invisible -- the one row this project is about.
    """
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    row_share = cm / cm.sum(axis=1, keepdims=True)
    corner = [["TN", "FP"], ["FN", "TP"]]

    fig, ax = _new_figure((5.5, 4.9))
    ax.imshow(row_share, cmap=BLUE_RAMP, vmin=0.0, vmax=1.0)
    ax.grid(False)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    # A surface-coloured hairline between cells, so the four fills read as four
    # quantities rather than one continuous block.
    ax.axhline(0.5, color=SURFACE, linewidth=3.0)
    ax.axvline(0.5, color=SURFACE, linewidth=3.0)

    for i in range(2):
        for j in range(2):
            dark_cell = row_share[i, j] > 0.55
            text = "#ffffff" if dark_cell else INK
            faint = "#dbe8f7" if dark_cell else INK_MUTED
            ax.text(j, i - 0.13, f"{cm[i, j]:,}", ha="center", va="center",
                    fontsize=23, fontweight="bold", color=text)
            ax.text(j, i + 0.13, f"{row_share[i, j]:.1%} of actual",
                    ha="center", va="center", fontsize=10, color=text)
            ax.text(j - 0.44, i - 0.42, corner[i][j], ha="left", va="center",
                    fontsize=9, color=faint)

    ax.set_xticks([0, 1], ["Predicted\nretained", "Predicted\nchurn"])
    ax.set_yticks([0, 1], ["Actual\nretained", "Actual\nchurn"])
    ax.tick_params(colors=INK_SECONDARY, labelsize=10, length=0)

    _titles(ax, f"{MODELS[key]} — confusion matrix",
            f"held-out test set, n = {len(y_true):,}   ·   "
            f"recall {metrics['Recall']:.3f}   ·   "
            f"precision {metrics['Precision']:.3f}   ·   "
            f"F1 {metrics['F1']:.3f}")
    return _save(fig, f"confusion_matrix_{key}.png")


def plot_roc_curves(predictions: dict[str, dict], y: pd.Series,
                    results: pd.DataFrame) -> Path:
    """All five ROC curves on one axis, ordered in the legend by AUC."""
    fig, ax = _new_figure((7.4, 6.4))
    _style_axes(ax)

    scores = results.set_index("key")["ROC-AUC"].to_dict()
    for key in sorted(MODELS, key=lambda k: -scores[k]):
        fpr, tpr, _ = roc_curve(y, predictions[key]["y_score"])
        ax.plot(fpr, tpr, color=MODEL_COLORS[key], linewidth=2.0,
                linestyle=MODEL_DASHES[key], solid_capstyle="round",
                label=f"{MODELS[key]}   AUC {scores[key]:.4f}")

    ax.plot([0, 1], [0, 1], color=AXIS, linewidth=1.5, linestyle=(0, (2, 3)),
            label="Chance   AUC 0.5000", zorder=1)

    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.set_xlabel("False positive rate", color=INK_SECONDARY, fontsize=11)
    ax.set_ylabel("True positive rate", color=INK_SECONDARY, fontsize=11)
    _titles(ax, "ROC curves — held-out test set",
            f"n = {len(y):,}   ·   {int(y.sum()):,} churners   ·   "
            "higher and further left is better")
    legend = ax.legend(loc="lower right", frameon=True, fontsize=10,
                       facecolor=SURFACE, edgecolor=GRIDLINE, labelcolor=INK)
    legend.get_frame().set_linewidth(1.0)
    return _save(fig, "roc_curves.png")


def plot_pr_curves(predictions: dict[str, dict], y: pd.Series,
                   results: pd.DataFrame) -> Path:
    """All five precision-recall curves. The key plot for an imbalanced target.

    The no-skill reference is the positive-class rate, not 0.5: a model that
    guesses at random sits on that horizontal line, so distance above it is the
    only part of the curve that represents skill.
    """
    base_rate = float(y.mean())
    fig, ax = _new_figure((7.4, 6.4))
    _style_axes(ax)

    scores = results.set_index("key")["PR-AUC"].to_dict()
    for key in sorted(MODELS, key=lambda k: -scores[k]):
        precision, recall, _ = precision_recall_curve(y, predictions[key]["y_score"])
        ax.plot(recall, precision, color=MODEL_COLORS[key], linewidth=2.0,
                linestyle=MODEL_DASHES[key], solid_capstyle="round",
                label=f"{MODELS[key]}   AP {scores[key]:.4f}")

    ax.axhline(base_rate, color=AXIS, linewidth=1.5, linestyle=(0, (2, 3)),
               label=f"No skill   AP {base_rate:.4f}", zorder=1)

    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(0.0, 1.01)
    ax.set_xlabel("Recall  (share of churners found)", color=INK_SECONDARY,
                  fontsize=11)
    ax.set_ylabel("Precision  (share of flagged customers who churn)",
                  color=INK_SECONDARY, fontsize=11)
    _titles(ax, "Precision–recall curves — held-out test set",
            f"n = {len(y):,}   ·   churn base rate {base_rate:.4f}   ·   "
            "the metric that matters under class imbalance")
    legend = ax.legend(loc="upper right", frameon=True, fontsize=10,
                       facecolor=SURFACE, edgecolor=GRIDLINE, labelcolor=INK)
    legend.get_frame().set_linewidth(1.0)
    return _save(fig, "pr_curves.png")


def plot_model_comparison(results: pd.DataFrame, y: pd.Series) -> Path:
    """Small multiples: one panel per metric, five bars each.

    Bars start at zero in every panel. Truncating the axis to make a 0.02 gap
    look decisive is the most common way a comparison chart misleads, and these
    five models genuinely are close.
    """
    fig, axes = plt.subplots(2, 3, figsize=(12.4, 7.0), dpi=160)
    fig.patch.set_facecolor(SURFACE)

    keys = list(MODELS)
    positions = np.arange(len(keys))
    # No-skill reference where one exists for that metric.
    no_skill = {"Accuracy": 1.0 - float(y.mean()), "PR-AUC": float(y.mean())}

    for ax, metric in zip(axes.ravel(), METRICS):
        _style_axes(ax, grid_axis="y")
        values = [float(results.loc[results["key"] == k, metric].iloc[0])
                  for k in keys]
        top = max(values)

        ax.bar(positions, values, width=0.68,
               color=[MODEL_COLORS[k] for k in keys], zorder=2)
        for x, value in zip(positions, values):
            ax.text(x, value + 0.028, f"{value:.3f}", ha="center", va="bottom",
                    fontsize=9,
                    fontweight="bold" if value == top else "normal",
                    color=INK if value == top else INK_SECONDARY)

        if metric in no_skill:
            ax.axhline(no_skill[metric], color=INK_MUTED, linewidth=1.2,
                       linestyle=(0, (3, 2)), zorder=3)
            # Annotated in the empty band above the bars: on the line itself it
            # would collide with the bar value labels.
            ax.annotate(f"dashed line = no skill ({no_skill[metric]:.3f})",
                        xy=(0.015, 0.97), xycoords="axes fraction",
                        ha="left", va="top", fontsize=8, color=INK_MUTED)

        ax.set_ylim(0, 1.12)
        ax.set_xticks(positions, [""] * len(keys))
        ax.set_title(metric, color=INK, fontsize=12, fontweight="bold",
                     loc="left", pad=8)

    handles = [plt.Rectangle((0, 0), 1, 1, color=MODEL_COLORS[k]) for k in keys]
    legend = fig.legend(handles, [MODELS[k] for k in keys], loc="lower center",
                        ncol=5, frameon=False, fontsize=10.5, labelcolor=INK,
                        bbox_to_anchor=(0.5, -0.012))
    del legend

    fig.suptitle("Model comparison — held-out test set", color=INK, fontsize=14,
                 fontweight="bold", x=0.055, ha="left", y=1.0)
    fig.text(0.055, 0.958,
             f"n = {len(y):,}   ·   {int(y.sum()):,} churners   ·   "
             "bars start at zero; the five models are close on every metric",
             color=INK_SECONDARY, fontsize=10)
    fig.tight_layout(rect=(0, 0.035, 1, 0.945))
    return _save(fig, "model_comparison.png")


def make_all_figures(predictions: dict[str, dict], y: pd.Series,
                     results: pd.DataFrame) -> list[Path]:
    """Every Stage 4 figure, from the single evaluation pass."""
    paths = [
        plot_confusion_matrix(
            key, y, predictions[key]["y_pred"],
            results.loc[results["key"] == key, METRICS].iloc[0].to_dict(),
        )
        for key in MODELS
    ]
    paths.append(plot_roc_curves(predictions, y, results))
    paths.append(plot_pr_curves(predictions, y, results))
    paths.append(plot_model_comparison(results, y))
    return paths


# --------------------------------------------------------------------------
# Interpretation
# --------------------------------------------------------------------------
def findings(results: pd.DataFrame, comparison: pd.DataFrame, y: pd.Series,
             predictions: dict[str, dict]) -> list[str]:
    """Observations computed from the test results. No claim without a number."""
    n = len(y)
    n_pos = int(y.sum())
    base_rate = float(y.mean())
    out: list[str] = []

    pr = ranking(results, "PR-AUC")
    out.append(
        f"Highest PR-AUC: {pr[0][0]} at {pr[0][1]:.4f}, ahead of "
        f"{pr[1][0]} at {pr[1][1]:.4f} (a gap of {pr[0][1] - pr[1][1]:.4f}). "
        f"All five sit above the {base_rate:.4f} no-skill rate, spanning "
        f"{pr[-1][1]:.4f} to {pr[0][1]:.4f}."
    )

    roc = ranking(results, "ROC-AUC")
    same_roc = ("the same model as the PR-AUC leader"
                if roc[0][0] == pr[0][0]
                else f"a different model from the PR-AUC leader, {pr[0][0]}")
    out.append(
        f"Highest ROC-AUC: {roc[0][0]} at {roc[0][1]:.4f} -- {same_roc}. The "
        f"ROC-AUC spread across all five is only "
        f"{roc[0][1] - roc[-1][1]:.4f}, so this metric separates them least."
    )

    rec = ranking(results, "Recall")
    rec_key = results.loc[results["Model"] == rec[0][0], "key"].iloc[0]
    tn, fp, fn, tp = confusion_matrix(
        y, predictions[rec_key]["y_pred"], labels=[0, 1]
    ).ravel()
    out.append(
        f"Highest recall: {rec[0][0]} at {rec[0][1]:.4f} -- it finds {tp:,} of "
        f"the {n_pos:,} actual churners and misses {fn:,}, so it is the model "
        f"that identifies the most real churners. The cost is volume: it "
        f"flags {tp + fp:,} customers in total, so {fp:,} of its alerts are "
        f"customers who stayed (precision "
        f"{float(results.loc[results['key'] == rec_key, 'Precision'].iloc[0]):.4f})."
    )

    acc = ranking(results, "Accuracy")
    acc_key = results.loc[results["Model"] == acc[0][0], "key"].iloc[0]
    acc_pr = float(results.loc[results["key"] == acc_key, "PR-AUC"].iloc[0])
    acc_recall = float(results.loc[results["key"] == acc_key, "Recall"].iloc[0])
    if acc[0][0] == pr[0][0]:
        out.append(
            f"The most accurate model is also the PR-AUC leader ({acc[0][0]}, "
            f"accuracy {acc[0][1]:.4f}), so the two criteria agree here."
        )
    else:
        out.append(
            f"Accuracy and PR-AUC disagree: {acc[0][0]} is the most accurate at "
            f"{acc[0][1]:.4f}, but its PR-AUC is {acc_pr:.4f} versus "
            f"{pr[0][1]:.4f} for {pr[0][0]}, and its recall is "
            f"{acc_recall:.4f} versus {dict(rec)[rec[0][0]]:.4f} for the best. "
            f"This is why PR-AUC was fixed as the selection metric before the "
            f"test set was opened."
        )

    majority = 1.0 - base_rate
    beat = [name for name, value in acc if value > majority]
    if len(beat) == len(acc):
        clears = f"All five clear the {majority:.4f} majority-class accuracy"
    elif beat:
        clears = (f"{len(beat)} of 5 clear the {majority:.4f} majority-class "
                  f"accuracy ({', '.join(beat)})")
    else:
        clears = (f"None of the five clears the {majority:.4f} majority-class "
                  f"accuracy")
    out.append(
        f"{clears}, but only by {acc[0][1] - majority:.4f} at best: accuracies "
        f"run {acc[-1][1]:.4f} to {acc[0][1]:.4f} on n = {n:,}. Accuracy "
        f"therefore separates these models very little -- the visible "
        f"differences are in recall ({rec[-1][1]:.4f} to {rec[0][1]:.4f}) and "
        f"precision."
    )

    cv_order = list(comparison.sort_values("CV PR-AUC", ascending=False)["Model"])
    test_order = list(comparison.sort_values("Test PR-AUC", ascending=False)["Model"])
    moved = [m for m in cv_order if cv_order.index(m) != test_order.index(m)]
    worst = comparison.loc[comparison["Difference"].abs().idxmax()]
    if cv_order == test_order:
        out.append(
            "The PR-AUC ranking is identical in cross-validation and on the "
            f"test set. The largest change in value is {worst['Model']} at "
            f"{worst['Difference']:+.4f}."
        )
    else:
        out.append(
            f"The PR-AUC ranking is not identical between CV and test: "
            f"{len(moved)} of 5 models change position "
            f"(CV {' > '.join(cv_order)}; test {' > '.join(test_order)}). "
            f"The largest change in value is {worst['Model']} at "
            f"{worst['Difference']:+.4f}."
        )

    f1 = ranking(results, "F1")
    leaders: dict[str, list[str]] = {}
    for metric in METRICS:
        leaders.setdefault(best_by(results, metric)[1], []).append(metric)
    breakdown = "; ".join(
        f"{model} leads on {', '.join(metrics)}" for model, metrics in leaders.items()
    )
    if len(leaders) == 1:
        out.append(
            f"One model leads on all six metrics: {breakdown}. Best F1 is "
            f"{f1[0][0]} at {f1[0][1]:.4f}. The criteria agree, so the "
            f"selection does not depend on which one is weighted."
        )
    else:
        out.append(
            f"The six metrics do not all point at one model -- {breakdown}. "
            f"Best F1 is {f1[0][0]} at {f1[0][1]:.4f}. {pr[0][0]} is therefore "
            f"reported as the selection under PR-AUC, not as the winner on "
            f"every measure."
        )
    return out
