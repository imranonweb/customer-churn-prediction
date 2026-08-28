"""Stage 5: SHAP explanation of the final selected model (Random Forest).

Run from the project root:

    python run_stage5.py

What this module explains -- and what it does not
------------------------------------------------
SHAP values decompose **this Random Forest's** predicted churn probability into
per-feature contributions. They describe how the model uses each feature. They
are **not** evidence that a feature causes churn: a feature can matter to the
model because it is correlated with something else entirely. Every sentence this
module prints is phrased accordingly ("associated with", "influenced the model
prediction"), and the console output states the distinction outright.

Leakage and mutation discipline
-------------------------------
Stage 4 has already finished final evaluation, so reading the test set here
changes no reported result. Even so:

* the saved Stage 3 pipeline is loaded and used **exactly as saved** -- the
  estimator is never fitted and the preprocessor is never refitted;
* only `prep.transform()` is called, never `fit` or `fit_transform`;
* the preprocessor's fitted state is fingerprinted before and after use, and the
  five `.pkl` files are checksummed before and after, so "nothing was modified"
  is measured rather than asserted.

Correctness of the explanation itself is checked, not assumed:

* **Additivity** -- `base_value + sum(shap_values)` must reproduce
  `pipeline.predict_proba()[:, 1]` for all 1,409 rows. If the class slice, the
  feature ordering, or the transform were wrong, this identity would break.
* **Name alignment** -- the transformed matrix is checked positionally against
  the original columns (a passthrough numeric must equal its source column; each
  one-hot group must be 0/1 and sum to 1 per row), so the readable labels are
  provably attached to the features the model actually saw.

Figure style is inherited from Stage 4 (`src/evaluate.py`) so the two stages'
figures read as one set. Stage 4 is imported, never modified.
"""

from __future__ import annotations

import ast
import hashlib
import pickle
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
from matplotlib.colors import LinearSegmentedColormap
from sklearn.pipeline import Pipeline

from src import config, evaluate

# Stage 4's figure tokens and helpers are reused directly rather than copied, so
# the two stages cannot drift apart visually.
SURFACE = evaluate.SURFACE
INK = evaluate.INK
INK_SECONDARY = evaluate.INK_SECONDARY
INK_MUTED = evaluate.INK_MUTED
GRIDLINE = evaluate.GRIDLINE
AXIS = evaluate.AXIS

plt = evaluate.plt

# --------------------------------------------------------------------------
# The model Stage 4 selected
# --------------------------------------------------------------------------
MODEL_KEY = "random_forest"
MODEL_NAME = "Random Forest"

TOP_N_GLOBAL = 15  # features shown in the bar and beeswarm plots
TOP_N_LOCAL = 12   # features shown in each customer's waterfall

# Positive SHAP pushes toward churn, negative pushes away. Same two hues as the
# Stage 4 palette, validated as a diverging pair (worst CVD Delta E 21.6).
PUSH_TOWARD = "#e34948"
PUSH_AWAY = "#2a78d6"
NEUTRAL_MID = "#f0efec"

# Diverging ramp for the beeswarm's feature-value axis: two hues either side of
# a neutral gray. The neutral band is kept narrow so mid-valued points stay
# visible against the surface rather than washing out.
VALUE_RAMP = LinearSegmentedColormap.from_list(
    "shap_diverging",
    [(0.00, PUSH_AWAY), (0.42, "#9ec5f4"), (0.50, NEUTRAL_MID),
     (0.58, "#f0a2a1"), (1.00, PUSH_TOWARD)],
)


# --------------------------------------------------------------------------
# Human-readable feature names
# --------------------------------------------------------------------------
# The mapping is explicit rather than clever: every transformed name resolves to
# a phrase that keeps the original column and level visible.
NUMERIC_LABELS = {
    "tenure": "Tenure (months)",
    "MonthlyCharges": "Monthly charges",
    "TotalCharges": "Total charges",
    "avg_charge": "Average charge per month",
    "num_services": "Number of services",
    "is_new_customer": "New customer (tenure 0)",
    "SeniorCitizen": "Senior citizen",
}

# One-hot columns whose levels are Yes/No: "<Column>_Yes" -> "Has X".
HAS_NO_LABELS = {
    "Partner": ("Has a partner", "No partner"),
    "Dependents": ("Has dependents", "No dependents"),
    "PhoneService": ("Has phone service", "No phone service"),
    "MultipleLines": ("Has multiple lines", "No multiple lines"),
    "OnlineSecurity": ("Has online security", "No online security"),
    "OnlineBackup": ("Has online backup", "No online backup"),
    "DeviceProtection": ("Has device protection", "No device protection"),
    "TechSupport": ("Has tech support", "No tech support"),
    "StreamingTV": ("Has streaming TV", "No streaming TV"),
    "StreamingMovies": ("Has streaming movies", "No streaming movies"),
    "PaperlessBilling": ("Paperless billing", "No paperless billing"),
}

# One-hot columns whose levels are multi-valued: level goes into a template.
LEVEL_TEMPLATES = {
    "Contract": "{level} contract",
    "PaymentMethod": "{level} payment",
    "InternetService": "{level} internet",
    "gender": "Gender: {level}",
}

# Levels needing a small rewrite before the template is applied.
LEVEL_REWRITES = {
    ("Contract", "One year"): "One-year",
    ("Contract", "Two year"): "Two-year",
    ("InternetService", "No"): "No internet service",
}


def readable_name(transformed: str) -> str:
    """Turn a ColumnTransformer output name into a phrase a reader understands.

    `cat__Contract_Month-to-month` -> `Month-to-month contract`
    `cat__PaymentMethod_Electronic check` -> `Electronic check payment`
    `num__tenure` -> `Tenure (months)`
    """
    if transformed.startswith("num__"):
        column = transformed.removeprefix("num__")
        return NUMERIC_LABELS.get(column, column)

    if transformed.startswith("cat__"):
        body = transformed.removeprefix("cat__")
        # Split on the last underscore that separates a known column from its
        # level; column names here contain no underscores, so the first works.
        column, _, level = body.partition("_")
        if column in HAS_NO_LABELS:
            yes_label, no_label = HAS_NO_LABELS[column]
            return yes_label if level == "Yes" else no_label
        if column in LEVEL_TEMPLATES:
            rewritten = LEVEL_REWRITES.get((column, level), level)
            if rewritten == "No internet service":
                return rewritten
            return LEVEL_TEMPLATES[column].format(level=rewritten)
        return f"{column}: {level}"

    return transformed


def readable_names(transformed: list[str]) -> list[str]:
    return [readable_name(name) for name in transformed]


def check_name_mapping(transformed: list[str]) -> list[str]:
    """Transformed names that fell through to their raw form.

    An empty list means every one of the 41 features got a real label rather
    than a silent passthrough of the encoder's output.
    """
    return [name for name in transformed if readable_name(name) == name]


def is_indicator(transformed: str) -> bool:
    """True for one-hot columns, whose value reads as yes/no rather than a number."""
    return transformed.startswith("cat__")


def format_value(transformed: str, value: float) -> str:
    """How a feature's value is shown next to its name."""
    if is_indicator(transformed):
        return "yes" if value > 0.5 else "no"
    if float(value).is_integer():
        return f"{int(value)}"
    return f"{value:,.2f}"


# --------------------------------------------------------------------------
# Loading and transforming -- no fitting anywhere
# --------------------------------------------------------------------------
def load_model() -> Pipeline:
    """Load the Stage 3 Random Forest pipeline as saved."""
    path = evaluate.model_path(MODEL_KEY)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python -m src.train` first."
        )
    return joblib.load(path)


def load_test() -> tuple[pd.DataFrame, pd.Series]:
    """Same split and target definition Stage 4 evaluated on."""
    return evaluate.load_test()


def preprocessor_fingerprint(prep) -> str:
    """SHA-256 of the fitted preprocessor's pickled state.

    If `transform()` mutated any learned attribute -- category levels, feature
    counts, scaler statistics -- these bytes would differ.
    """
    return hashlib.sha256(pickle.dumps(prep)).hexdigest()


def transform_features(pipe: Pipeline, X: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Apply the already-fitted preprocessor. `transform` only, never `fit`."""
    prep = pipe.named_steps["prep"]
    matrix = prep.transform(X)
    if hasattr(matrix, "toarray"):  # a sparse result would break SHAP indexing
        matrix = matrix.toarray()
    return np.asarray(matrix), list(prep.get_feature_names_out())


def check_feature_alignment(X: pd.DataFrame, matrix: np.ndarray,
                            names: list[str]) -> dict[str, bool]:
    """Prove the names line up positionally with the columns the model saw.

    This is the check that would catch a silently reordered transformer output.
    The Random Forest pipeline uses `scale=False`, so numeric columns pass
    through untouched and must equal their source column exactly.
    """
    index = {name: i for i, name in enumerate(names)}
    checks = {
        "count matches transformed matrix": len(names) == matrix.shape[1],
        "tenure column equals source": bool(np.array_equal(
            matrix[:, index["num__tenure"]], X["tenure"].to_numpy(dtype=float)
        )),
        "MonthlyCharges column equals source": bool(np.array_equal(
            matrix[:, index["num__MonthlyCharges"]],
            X["MonthlyCharges"].to_numpy(dtype=float),
        )),
        "Contract_Month-to-month equals source indicator": bool(np.array_equal(
            matrix[:, index["cat__Contract_Month-to-month"]],
            (X["Contract"] == "Month-to-month").to_numpy(dtype=float),
        )),
    }

    indicator_cols = [i for i, name in enumerate(names) if is_indicator(name)]
    checks["all one-hot columns are 0/1"] = bool(
        np.isin(matrix[:, indicator_cols], (0.0, 1.0)).all()
    )
    # Each original categorical must contribute exactly one active level per row.
    groups_ok = True
    for column in config.CATEGORICAL_COLS:
        cols = [i for i, name in enumerate(names)
                if name.startswith(f"cat__{column}_")]
        groups_ok &= bool(np.array_equal(
            matrix[:, cols].sum(axis=1), np.ones(len(matrix))
        ))
    checks["every one-hot group sums to 1 per row"] = groups_ok
    return checks


# --------------------------------------------------------------------------
# SHAP
# --------------------------------------------------------------------------
def compute_shap(pipe: Pipeline, matrix: np.ndarray,
                 names: list[str]) -> shap.Explanation:
    """TreeExplainer on the fitted Random Forest, sliced to the churn class.

    `TreeExplainer` reads the fitted trees directly -- it does not fit anything.
    For a scikit-learn forest its output is in probability space, so the
    contributions add up to the predicted churn probability.
    """
    rf = pipe.named_steps["model"]
    explainer = shap.TreeExplainer(rf)
    full = explainer(matrix)

    # Binary classifier: values arrive as (rows, features, 2). Class 1 = churn.
    churn_class = list(rf.classes_).index(1)
    return shap.Explanation(
        values=full.values[:, :, churn_class],
        base_values=np.asarray(full.base_values)[:, churn_class],
        data=matrix,
        feature_names=readable_names(names),
    )


def check_shap_validity(explanation: shap.Explanation, matrix: np.ndarray,
                        proba: np.ndarray) -> dict:
    """Additivity, shape agreement, and finiteness of the SHAP output."""
    reconstructed = explanation.base_values + explanation.values.sum(axis=1)
    return {
        "shape": tuple(explanation.values.shape),
        "matches_matrix": explanation.values.shape == matrix.shape,
        "n_names": len(explanation.feature_names),
        "names_match_columns": len(explanation.feature_names) == matrix.shape[1],
        "n_nan": int(np.isnan(explanation.values).sum()),
        "n_inf": int(np.isinf(explanation.values).sum()),
        "all_finite": bool(np.isfinite(explanation.values).all()),
        "max_additivity_error": float(np.abs(reconstructed - proba).max()),
        "base_value": float(explanation.base_values[0]),
    }


def high_value_effect(explanation: shap.Explanation) -> np.ndarray:
    """How much a high (or "yes") value of each feature shifts its contribution.

    Mean *signed* SHAP is not usable as a direction: the forest's baseline is
    0.5004 while the average prediction is far lower, so the contributions must
    sum to a large negative number and almost every feature looks like it
    "pushes away from churn". That offset says nothing about the feature.

    The direction that does mean something is the contrast within the feature:
    mean SHAP among rows where it is high (a one-hot set to 1, or a numeric above
    its median) minus mean SHAP among the rest. Positive means a high value moved
    the prediction toward churn. This is the beeswarm's message as one number.
    NaN where the feature is constant across the test set, so no contrast exists.
    """
    effects = np.full(explanation.values.shape[1], np.nan)
    for column in range(explanation.values.shape[1]):
        raw = explanation.data[:, column]
        cut = 0.5 if set(np.unique(raw)) <= {0.0, 1.0} else np.median(raw)
        high = raw > cut
        if high.any() and (~high).any():
            effects[column] = (explanation.values[high, column].mean()
                               - explanation.values[~high, column].mean())
    return effects


def global_importance(explanation: shap.Explanation,
                      names: list[str]) -> pd.DataFrame:
    """Mean absolute SHAP value per feature, best first.

    `high_value_effect` records which way a high value of the feature pushed the
    prediction -- reported alongside the magnitude in the console, but not
    encoded as a second colour in the magnitude chart.
    """
    return pd.DataFrame({
        "feature": names,
        "readable": readable_names(names),
        "mean_abs_shap": np.abs(explanation.values).mean(axis=0),
        "high_value_effect": high_value_effect(explanation),
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------
# Local explanations
# --------------------------------------------------------------------------
def select_customers(proba: np.ndarray) -> dict[str, int]:
    """Pick the highest- and lowest-probability customers.

    This function receives **only** the probability array -- it has no access to
    `y_test`, so the selection cannot be influenced by the true labels. The
    labels are looked up afterwards, purely to report them.
    """
    return {"high": int(np.argmax(proba)), "low": int(np.argmin(proba))}


def local_contributions(explanation: shap.Explanation, names: list[str],
                        row: int, top_n: int = TOP_N_LOCAL) -> pd.DataFrame:
    """One customer's largest contributions, plus the combined remainder."""
    values = explanation.values[row]
    order = np.argsort(np.abs(values))[::-1]
    top, rest = order[:top_n], order[top_n:]

    frame = pd.DataFrame({
        "feature": [names[i] for i in top],
        "readable": [readable_name(names[i]) for i in top],
        "value": [explanation.data[row, i] for i in top],
        "shap": values[top],
    })
    frame["display_value"] = [
        format_value(f, v) for f, v in zip(frame["feature"], frame["value"])
    ]
    if len(rest):
        remainder = pd.DataFrame([{
            "feature": "__other__",
            "readable": f"Other {len(rest)} features (combined)",
            "value": np.nan,
            "shap": float(values[rest].sum()),
            "display_value": "",
        }])
        frame = pd.concat([frame, remainder], ignore_index=True)
    return frame


def signed_contributions(explanation: shap.Explanation, names: list[str],
                         row: int, k: int = 6
                         ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The k largest contributions pushing each way, for one customer.

    Taking the top k by magnitude and then splitting by sign would leave one
    list empty for a confidently-predicted customer -- every one of their
    strongest contributors points the same way. Ranking each sign separately
    means both directions are reported whenever both exist, which is what makes
    the explanation informative rather than one-sided.
    """
    values = explanation.values[row]

    def side(mask: np.ndarray) -> pd.DataFrame:
        indices = np.where(mask)[0]
        indices = indices[np.argsort(np.abs(values[indices]))[::-1]][:k]
        frame = pd.DataFrame({
            "feature": [names[i] for i in indices],
            "readable": [readable_name(names[i]) for i in indices],
            "value": [explanation.data[row, i] for i in indices],
            "shap": values[indices],
        })
        frame["display_value"] = [
            format_value(f, v) for f, v in zip(frame["feature"], frame["value"])
        ]
        return frame

    return side(values > 0), side(values < 0)


# --------------------------------------------------------------------------
# Figures -- Stage 4's helpers, so the two stages match
# --------------------------------------------------------------------------
def plot_global_importance(importance: pd.DataFrame, n_rows: int,
                           top_n: int = TOP_N_GLOBAL) -> Path:
    """Mean |SHAP| for the top features.

    Magnitude only, so a single hue: which way each feature pushes is the
    beeswarm's job, and colouring this chart by direction would encode two
    different things in one channel.
    """
    top = importance.head(top_n).iloc[::-1]  # largest at the top of the axis
    fig, ax = evaluate._new_figure((8.6, 6.4))
    evaluate._style_axes(ax, grid_axis="x")

    positions = np.arange(len(top))
    ax.barh(positions, top["mean_abs_shap"], height=0.68, color=PUSH_AWAY,
            zorder=2)
    for y, value in zip(positions, top["mean_abs_shap"]):
        ax.text(value + top["mean_abs_shap"].max() * 0.015, y, f"{value:.4f}",
                va="center", ha="left", fontsize=9, color=INK_SECONDARY)

    ax.set_yticks(positions, top["readable"])
    ax.tick_params(axis="y", colors=INK, labelsize=10, length=0)
    ax.set_xlim(0, top["mean_abs_shap"].max() * 1.16)
    ax.set_xlabel("Mean |SHAP value|   (average impact on predicted churn "
                  "probability)", color=INK_SECONDARY, fontsize=10.5)
    evaluate._titles(
        ax, f"What the {MODEL_NAME} relies on — top {top_n} features",
        f"SHAP on the held-out test set, n = {n_rows:,} customers   ·   "
        "magnitude only; direction is in the beeswarm   ·   "
        "model behaviour, not causation",
    )
    return evaluate._save(fig, "shap_feature_importance.png")


def _beeswarm_offsets(values: np.ndarray, order: np.ndarray,
                      n_bins: int = 90, spread: float = 0.36) -> np.ndarray:
    """Deterministic vertical offsets that spread overlapping points.

    Points are binned along the SHAP axis and stacked symmetrically within each
    bin, ordered by feature value. No random jitter, so the figure is identical
    on every run.
    """
    offsets = np.zeros(len(values))
    if values.max() == values.min():
        return offsets
    bins = np.clip(
        ((values - values.min()) / (values.max() - values.min()) * n_bins
         ).astype(int), 0, n_bins - 1
    )
    densest = max(np.bincount(bins).max(), 1)
    for b in np.unique(bins):
        members = np.where(bins == b)[0]
        members = members[np.argsort(order[members], kind="stable")]
        centred = np.arange(len(members)) - (len(members) - 1) / 2
        offsets[members] = centred * (spread / densest) * 2
    return np.clip(offsets, -spread, spread)


def plot_beeswarm(explanation: shap.Explanation, importance: pd.DataFrame,
                  names: list[str], top_n: int = TOP_N_GLOBAL) -> Path:
    """Every test customer's contribution, per feature, coloured by feature value."""
    index = {name: i for i, name in enumerate(names)}
    top = importance.head(top_n).iloc[::-1]

    fig, ax = evaluate._new_figure((9.6, 7.6))
    evaluate._style_axes(ax, grid_axis="x")
    ax.axvline(0.0, color=AXIS, linewidth=1.2, zorder=1)

    for y, (_, feature_row) in enumerate(top.iterrows()):
        column = index[feature_row["feature"]]
        values = explanation.values[:, column]
        raw = explanation.data[:, column]
        # Normalised feature value drives the colour; constant columns sit mid.
        span = raw.max() - raw.min()
        normalised = (raw - raw.min()) / span if span else np.full(len(raw), 0.5)
        ax.scatter(values, y + _beeswarm_offsets(values, normalised),
                   c=normalised, cmap=VALUE_RAMP, vmin=0.0, vmax=1.0,
                   s=11, linewidths=0.25, edgecolors=SURFACE, zorder=2)

    ax.set_yticks(np.arange(len(top)), top["readable"])
    ax.tick_params(axis="y", colors=INK, labelsize=10, length=0)
    ax.set_ylim(-0.7, len(top) - 0.3)
    ax.set_xlabel("SHAP value   (left = pushed away from churn, "
                  "right = pushed toward churn)",
                  color=INK_SECONDARY, fontsize=10.5)
    evaluate._titles(
        ax, f"How feature values moved the {MODEL_NAME}'s prediction",
        f"one point per test customer × feature   ·   "
        f"top {top_n} features by mean |SHAP|   ·   "
        "model behaviour, not causation",
    )

    bar = fig.colorbar(plt.cm.ScalarMappable(cmap=VALUE_RAMP), ax=ax,
                       fraction=0.022, pad=0.015, ticks=[0, 1])
    bar.ax.set_yticklabels(["low / no", "high / yes"], fontsize=9,
                           color=INK_SECONDARY)
    bar.set_label("Feature value", color=INK_SECONDARY, fontsize=9.5)
    bar.outline.set_visible(False)
    return evaluate._save(fig, "shap_beeswarm.png")


def plot_waterfall(contributions: pd.DataFrame, base_value: float,
                   probability: float, title: str, subtitle: str,
                   filename: str) -> Path:
    """One customer: from the model's baseline to its prediction, step by step.

    Bars are drawn end-to-end from the baseline, so the horizontal position of
    each bar is where the running probability had reached -- red segments move
    it toward churn, blue segments move it back.
    """
    frame = contributions.iloc[::-1].reset_index(drop=True)  # first step on top
    starts = base_value + np.concatenate(([0.0], np.cumsum(frame["shap"])[:-1]))

    fig, ax = evaluate._new_figure((9.4, 6.8))
    evaluate._style_axes(ax, grid_axis="x")

    positions = np.arange(len(frame))
    for y, (start, delta, feature) in enumerate(
        zip(starts, frame["shap"], frame["feature"])
    ):
        toward = delta >= 0
        colour = PUSH_TOWARD if toward else PUSH_AWAY
        if feature == "__other__":
            colour = INK_MUTED
        ax.barh(y, delta, left=start, height=0.64, color=colour, zorder=2)
        ax.text(start + delta + (0.006 if toward else -0.006), y,
                f"{delta:+.4f}", va="center",
                ha="left" if toward else "right", fontsize=9,
                color=INK_SECONDARY)

    labels = [
        f"{row.readable}" + (f"  =  {row.display_value}" if row.display_value else "")
        for row in frame.itertuples()
    ]
    ax.set_yticks(positions, labels)
    ax.tick_params(axis="y", colors=INK, labelsize=10, length=0)
    ax.set_ylim(-0.85, len(frame) - 0.25)

    # Autoscaling stops at the bars, which puts the baseline line exactly on the
    # left spine and pushes both annotations outside the axes. Set the extent
    # from the values that must be visible, then pad for the labels.
    ends = np.concatenate([starts, starts + frame["shap"].to_numpy(),
                           [base_value, probability]])
    span = ends.max() - ends.min()
    ax.set_xlim(ends.min() - span * 0.14, ends.max() + span * 0.14)

    ax.axvline(base_value, color=AXIS, linewidth=1.4, linestyle=(0, (3, 2)),
               zorder=1)
    ax.axvline(probability, color=INK, linewidth=1.4, zorder=3)
    # Each marker label goes on whichever side of its line has more room, so
    # neither leaves the plot area -- the two lines swap places between a
    # high-risk customer (prediction on the right) and a low-risk one.
    low, high = ax.get_xlim()
    for x, text, colour, weight in (
        (base_value, f"model baseline {base_value:.4f}", INK_MUTED, "normal"),
        (probability, f"prediction {probability:.4f}", INK, "bold"),
    ):
        leftward = (x - low) > (high - x)
        ax.annotate(text, xy=(x, -0.82),
                    xytext=(-5 if leftward else 5, 0),
                    textcoords="offset points",
                    ha="right" if leftward else "left", va="bottom",
                    fontsize=9, color=colour, fontweight=weight)

    ax.set_xlabel("Predicted probability of churn", color=INK_SECONDARY,
                  fontsize=10.5)
    evaluate._titles(ax, title, subtitle)
    return evaluate._save(fig, filename)


# --------------------------------------------------------------------------
# Stage 2 EDA hypotheses, for the comparison at the end
# --------------------------------------------------------------------------
# Quoted from notebooks/01_eda.ipynb. Stage 2 described associations in the raw
# training data; SHAP describes what the fitted model does with them. The two can
# legitimately disagree, which is the point of comparing them.
#
# `expected` records what Stage 2 predicted -- "strong" or "weak". Without it the
# comparison misreads its own evidence: Stage 2 found `num_services` barely
# separates the classes, so that feature ranking low in SHAP is agreement, not a
# missed driver.
EDA_HYPOTHESES = [
    {
        "label": "Contract type",
        "expected": "strong",
        "features": ["cat__Contract_Month-to-month", "cat__Contract_One year",
                     "cat__Contract_Two year"],
        "eda": "month-to-month customers churned at 42.75% against 2.87% on "
               "two-year contracts -- the strongest categorical separation in "
               "the data, a 14.9x gap",
    },
    {
        "label": "Tenure",
        "expected": "strong",
        "features": ["num__tenure"],
        "eda": "churn fell monotonically from 52.22% in months 0-6 to 6.81% in "
               "months 61-72, the strongest numeric correlation with churn "
               "(r = -0.346)",
    },
    {
        "label": "Payment method",
        "expected": "strong",
        "features": ["cat__PaymentMethod_Electronic check",
                     "cat__PaymentMethod_Mailed check",
                     "cat__PaymentMethod_Bank transfer (automatic)",
                     "cat__PaymentMethod_Credit card (automatic)"],
        "eda": "electronic check churned at 45.74% against 14.92% on credit "
               "card, roughly 3x, with the automatic methods grouped together "
               "at 15.55% against 35.03% for the manual ones",
    },
    {
        "label": "Internet service",
        "expected": "strong",
        "features": ["cat__InternetService_Fiber optic",
                     "cat__InternetService_DSL", "cat__InternetService_No"],
        "eda": "fiber optic churned at 42.09%, DSL at 18.69%, and customers "
               "with no internet service at 7.25%",
    },
    {
        "label": "Monthly charges",
        "expected": "strong",
        "features": ["num__MonthlyCharges"],
        "eda": "churners paid 74.86 a month on average against 61.34, though "
               "the churn rate was not monotonic in price -- it peaked in the "
               "third quartile at 36.98% and fell to 33.29% in the fourth",
    },
    {
        "label": "Number of services",
        "expected": "weak",
        "features": ["num__num_services"],
        "eda": "the service count barely separated the groups -- 4.11 services "
               "for churners against 4.19 for retained customers (r = -0.015)",
    },
]

# EDA finding 7: `avg_charge` correlates 0.995 with `MonthlyCharges` and
# `num_services` 0.852. Stage 2 predicted this would distort linear coefficients
# while leaving tree *accuracy* alone -- it says nothing about attribution, which
# is what SHAP measures, so the split is worth reporting when it appears.
BILLING_FEATURES = ["num__MonthlyCharges", "num__TotalCharges", "num__avg_charge"]


def hypothesis_scores(importance: pd.DataFrame) -> pd.DataFrame:
    """Rank each Stage 2 hypothesis by the SHAP importance it actually earned."""
    lookup = importance.set_index("feature")
    rows = []
    for hypothesis in EDA_HYPOTHESES:
        present = [f for f in hypothesis["features"] if f in lookup.index]
        rows.append({
            "label": hypothesis["label"],
            "expected": hypothesis["expected"],
            "eda": hypothesis["eda"],
            "total_mean_abs_shap": float(lookup.loc[present, "mean_abs_shap"].sum()),
            "best_rank": int(min(lookup.index.get_loc(f) for f in present)) + 1,
            "best_feature": readable_name(
                min(present, key=lambda f: lookup.index.get_loc(f))
            ),
        })
    return pd.DataFrame(rows).sort_values("total_mean_abs_shap", ascending=False)


def _verdict(expected: str, in_top: bool, top_n: int) -> tuple[str, str]:
    """Phrase the agreement between what Stage 2 expected and what SHAP found.

    Returns the verdict clause and the non-causal closing sentence that fits it:
    "the feature influenced the prediction" is the wrong closer for a feature the
    model barely used.
    """
    if expected == "strong" and in_top:
        return ("was important to the model too, agreeing with the EDA pattern",
                "It influenced the model's prediction, which is not the same as "
                "causing churn.")
    if expected == "strong":
        return (f"did not reach the SHAP top {top_n}, so the model leaned on it "
                "less than the raw churn rates suggested",
                "A strong association in the data need not become a strong "
                "model input, and neither result speaks to causation.")
    if in_top:
        return (f"reached the SHAP top {top_n} even so, meaning the model used "
                "it more than the raw churn rates suggested",
                "It influenced the model's prediction, which is not the same as "
                "causing churn.")
    return ("carried little weight either, so the EDA and the model agree",
            "The model made little use of it, and neither result speaks to "
            "causation.")


def xai_findings(importance: pd.DataFrame, scores: pd.DataFrame,
                 top_n: int = TOP_N_GLOBAL) -> list[str]:
    """Compare the SHAP ranking with the Stage 2 observations. No causal claims."""
    top = importance.head(top_n)
    out: list[str] = []

    leader = importance.iloc[0]
    out.append(
        f"The feature the model relies on most is \"{leader['readable']}\" "
        f"(mean |SHAP| {leader['mean_abs_shap']:.4f}), followed by "
        f"\"{importance.iloc[1]['readable']}\" "
        f"({importance.iloc[1]['mean_abs_shap']:.4f}) and "
        f"\"{importance.iloc[2]['readable']}\" "
        f"({importance.iloc[2]['mean_abs_shap']:.4f}). The top "
        f"{top_n} features carry "
        f"{top['mean_abs_shap'].sum() / importance['mean_abs_shap'].sum():.1%} "
        f"of the total attribution across all {len(importance)} features."
    )

    for _, row in scores.iterrows():
        in_top = row["best_rank"] <= top_n
        verdict, closer = _verdict(row["expected"], in_top, top_n)
        out.append(
            f"{row['label']}: Stage 2 found that {row['eda']}. It {verdict} -- "
            f"its highest-ranked feature, \"{row['best_feature']}\", is "
            f"#{row['best_rank']} of {len(importance)}, and the group's "
            f"combined mean |SHAP| is {row['total_mean_abs_shap']:.4f}. {closer}"
        )

    # The billing columns get their own finding below, so they do not also count
    # as surprises here -- Stage 2 discussed all three (findings 4 and 7).
    hypothesised = {f for h in EDA_HYPOTHESES for f in h["features"]}
    hypothesised.update(BILLING_FEATURES)
    surprises = [(rank, row) for rank, (_, row)
                 in enumerate(top.head(8).iterrows(), start=1)
                 if row["feature"] not in hypothesised]
    if surprises:
        listed = ", ".join(
            f"\"{row['readable']}\" (#{rank}, {row['mean_abs_shap']:.4f})"
            for rank, row in surprises
        )
        out.append(
            f"Features in the SHAP top 8 that Stage 2 did not single out as "
            f"churn patterns: {listed}. The model gave them real weight even "
            f"though the raw churn-rate tables did not highlight them."
        )
    else:
        out.append(
            "No unexpected driver appeared at that depth: every feature in the "
            "SHAP top 8 corresponds to a pattern Stage 2 had already recorded, "
            "which is a point in favour of the model having learned the "
            "structure the data actually contains rather than an artefact."
        )

    billing = importance[importance["feature"].isin(BILLING_FEATURES)]
    billing_in_top = billing[billing.index < top_n]
    if len(billing_in_top) >= 2:
        ranked = ", ".join(
            f"\"{row['readable']}\" (#{index + 1}, {row['mean_abs_shap']:.4f})"
            for index, row in billing_in_top.iterrows()
        )
        out.append(
            f"The three correlated billing features all reach the top {top_n}: "
            f"{ranked}, together {billing['mean_abs_shap'].sum():.4f}. Stage 2 "
            f"measured avg_charge at r = 0.995 with MonthlyCharges and expected "
            f"that to distort linear coefficients while leaving tree accuracy "
            f"alone. It does leave accuracy alone, but attribution is a "
            f"different question: SHAP divides the credit for one underlying "
            f"signal across whichever correlated columns the trees happened to "
            f"split on, so each of the three reads lower individually than the "
            f"billing information is worth to the model as a whole."
        )

    directions = [
        f"\"{row['readable']}\" {'toward' if row['high_value_effect'] > 0 else 'away from'} churn"
        for _, row in top.head(4).iterrows()
        if pd.notna(row["high_value_effect"])
    ]
    out.append(
        "Direction, taken from the contrast within each feature rather than its "
        f"mean contribution: {', '.join(directions)}. Read this as the shift a "
        "high or \"yes\" value produced in the model's output, averaged over the "
        "test set -- not as an effect that would follow from changing the "
        "feature."
    )

    out.append(
        "Interpretation limit: SHAP values decompose this Random Forest's "
        "output, so they describe model behaviour. A feature can rank highly "
        "because it is associated with churn, because it correlates with "
        "another predictor, or because of how the trees split -- none of which "
        "establishes that changing it would change a customer's decision."
    )
    return out


# --------------------------------------------------------------------------
# Integrity check shared with the runner
# --------------------------------------------------------------------------
def check_no_fit_calls() -> dict[str, list[str]]:
    """No fit-family call may exist in Stage 5 source.

    Reuses Stage 4's AST helper, so a mention of `.fit()` in a comment or
    docstring cannot satisfy the check -- only real call sites count.
    """
    return {
        rel: evaluate._attribute_calls(config.PROJECT_ROOT / rel,
                                       evaluate.FIT_METHODS)
        for rel in ("src/explain.py", "run_stage5.py")
    }


def count_transform_call_sites() -> dict[str, int]:
    """`transform` is expected; `fit_transform` must be absent."""
    counts = {"transform": 0, "fit_transform": 0}
    for site in evaluate._attribute_calls(Path(__file__), set(counts)):
        counts[site.rsplit(".", 1)[1].rstrip("()")] += 1
    return counts
