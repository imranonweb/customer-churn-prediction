"""Stage 6 tests: the pure UI helpers only.

Deliberately narrow. These cover the two things in the app that would be wrong
silently -- the raw-row contract handed to Stage 1's `clean()`, and the regrouping
of one-hot SHAP values back onto input fields -- plus the risk-band boundaries.

No screenshot tests and no Streamlit runtime: rendering is verified by looking at
the running app, not by asserting on markup.

Run from the project root:

    python -m pytest -q
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from app import ui_components as ui
from src import config, preprocess

# A complete, realistic set of answers in dataset vocabulary.
ANSWERS: dict[str, object] = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "tenure": 3,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 94.40,
    "TotalCharges": 283.20,
}


# ------------------------------------------------------------- risk banding
@pytest.mark.parametrize(
    ("prob", "band"),
    [
        (0.0, "Low"),
        (0.3499, "Low"),
        (0.35, "Medium"),      # lower edge belongs to the higher band
        (0.5, "Medium"),
        (0.6499, "Medium"),
        (0.65, "High"),
        (1.0, "High"),
    ],
)
def test_risk_band_boundaries(prob, band):
    assert ui.risk_band(prob) == band


def test_risk_band_covers_every_probability_exactly_once():
    """No gap and no overlap between the presentation bands."""
    edges = [lo for _, lo, _ in ui.BANDS] + [1.0]
    assert edges == sorted(edges)
    for (name, lo, hi), (_, next_lo, _) in zip(ui.BANDS, ui.BANDS[1:]):
        assert hi == next_lo, f"{name} does not meet the next band"
    assert ui.BANDS[0][1] == 0.0 and ui.BANDS[-1][2] == 1.0


@pytest.mark.parametrize("prob", [-0.01, 1.01, 2.0])
def test_risk_band_rejects_impossible_probability(prob):
    with pytest.raises(ValueError):
        ui.risk_band(prob)


def test_band_verdict_uses_the_argmax_cutoff():
    assert ui.band_verdict(0.4999) == "Likely to stay"
    assert ui.band_verdict(0.5) == "Likely to churn"


# ---------------------------------------------------------------- formatting
def test_fmt_pct_and_signed():
    assert ui.fmt_pct(0.908873) == "90.9"
    assert ui.fmt_pct(0.5, places=2) == "50.00"
    assert ui.fmt_signed(0.0421) == "+0.042"
    assert ui.fmt_signed(-0.0421) == "-0.042"


def test_format_field_value_reads_like_the_form():
    assert ui.format_field_value("SeniorCitizen", 0) == "No"
    assert ui.format_field_value("is_new_customer", 1) == "Yes"
    assert ui.format_field_value("MonthlyCharges", 94.4) == "$94.40"
    assert ui.format_field_value("tenure", 1) == "1 month"
    assert ui.format_field_value("tenure", 3) == "3 months"
    assert ui.format_field_value("num_services", 4.0) == "4"
    assert ui.format_field_value("Contract", "Two year") == "Two year"


# ------------------------------------------------------- model input contract
def test_build_raw_row_matches_the_dataset_schema():
    raw = ui.build_raw_row(ANSWERS)
    assert list(raw.columns) == list(config.RAW_COLUMNS)
    assert len(raw) == 1


def test_build_raw_row_rejects_an_incomplete_form():
    incomplete = {k: v for k, v in ANSWERS.items() if k != "Contract"}
    with pytest.raises(KeyError):
        ui.build_raw_row(incomplete)


def test_raw_row_cleans_into_the_exact_feature_frame():
    """The whole point: Stage 1's own `clean()` accepts the UI row unchanged."""
    features = preprocess.clean(ui.build_raw_row(ANSWERS))[list(config.FEATURE_COLS)]
    assert list(features.columns) == list(config.FEATURE_COLS)
    assert features.shape == (1, 22)
    assert not features.isna().to_numpy().any()
    # Stage 1's three engineered features, computed by Stage 1 rather than here.
    assert features.loc[0, "is_new_customer"] == 0
    assert features.loc[0, "num_services"] == 4  # phone + internet + 2 streaming
    assert features.loc[0, "avg_charge"] == pytest.approx(283.20 / 3)


def test_churn_placeholder_cannot_influence_the_features():
    """`clean()` needs a Churn column; the value we pass must be inert."""
    frames = []
    for placeholder in ("No", "Yes"):
        raw = ui.build_raw_row(ANSWERS)
        raw[config.TARGET] = placeholder
        frames.append(preprocess.clean(raw)[list(config.FEATURE_COLS)])
    assert_frame_equal(frames[0], frames[1])


def test_customer_id_never_reaches_the_features():
    features = preprocess.clean(ui.build_raw_row(ANSWERS))[list(config.FEATURE_COLS)]
    assert config.ID_COL not in features.columns
    assert ui.PLACEHOLDER_ID not in features.to_numpy().tolist()[0]


def test_estimated_total_charges_is_zero_for_a_brand_new_customer():
    """Matches Stage 1, which imputes 0.0 for every tenure-0 row."""
    assert ui.estimate_total_charges(0, 94.40) == 0.0
    assert ui.estimate_total_charges(3, 94.40) == pytest.approx(283.20)


def test_tenure_zero_row_survives_cleaning():
    answers = {**ANSWERS, "tenure": 0, "TotalCharges": 0.0}
    features = preprocess.clean(ui.build_raw_row(answers))[list(config.FEATURE_COLS)]
    assert features.loc[0, "is_new_customer"] == 1
    assert features.loc[0, "avg_charge"] == 0.0  # clip(lower=1), never a divide by zero


# ------------------------------------------------------------ SHAP regrouping
def test_source_column_unwraps_both_transformer_prefixes():
    assert ui.source_column("num__tenure") == "tenure"
    assert ui.source_column("num__is_new_customer") == "is_new_customer"
    assert ui.source_column("cat__Contract_Month-to-month") == "Contract"
    assert ui.source_column("cat__Contract_Two year") == "Contract"
    assert ui.source_column("cat__PaymentMethod_Bank transfer (automatic)") == "PaymentMethod"
    assert ui.source_column("cat__PaperlessBilling_Yes") == "PaperlessBilling"


def test_no_source_column_name_contains_an_underscore():
    """The guarantee that makes `source_column`'s first-underscore split safe."""
    assert all("_" not in column for column in config.CATEGORICAL_COLS)


def test_grouping_preserves_the_total_exactly():
    """Additivity is why summing a field's one-hot levels is legitimate."""
    names = [
        "num__tenure",
        "num__MonthlyCharges",
        "cat__Contract_Month-to-month",
        "cat__Contract_One year",
        "cat__Contract_Two year",
        "cat__PaperlessBilling_No",
        "cat__PaperlessBilling_Yes",
    ]
    values = np.array([0.11, -0.04, 0.07, 0.0, -0.02, 0.0, 0.015])
    grouped = ui.group_shap_by_source(names, values)

    assert list(grouped.index) == ["tenure", "MonthlyCharges", "Contract", "PaperlessBilling"]
    assert grouped.sum() == pytest.approx(values.sum(), abs=1e-12)
    assert grouped["Contract"] == pytest.approx(0.05)
    assert grouped["PaperlessBilling"] == pytest.approx(0.015)


def test_contribution_rows_are_one_per_field_and_ranked():
    features = preprocess.clean(ui.build_raw_row(ANSWERS))[list(config.FEATURE_COLS)]
    names = ["num__tenure", "num__MonthlyCharges", "cat__Contract_Month-to-month"]
    values = np.array([-0.02, 0.31, 0.14])

    rows = ui.contribution_rows(features, names, values)
    assert [r["field"] for r in rows] == ["MonthlyCharges", "Contract", "tenure"]
    assert [r["label"] for r in rows][0] == "Monthly charges"
    assert rows[0]["value"] == "$94.40"
    assert rows[-1]["value"] == "3 months"
    assert [abs(r["shap"]) for r in rows] == sorted(
        (abs(r["shap"]) for r in rows), reverse=True
    )


def test_split_contributions_separates_signs_and_caps_each_side():
    rows = [{"shap": v} for v in (0.5, -0.4, 0.3, -0.2, 0.1, -0.05)]
    up, down = ui.split_contributions(rows, top_n=2)
    assert [r["shap"] for r in up] == [0.5, 0.3]
    assert [r["shap"] for r in down] == [-0.4, -0.2]
    assert all(r["shap"] > 0 for r in up)
    assert all(r["shap"] < 0 for r in down)


def test_split_contributions_handles_a_one_sided_prediction():
    up, down = ui.split_contributions([{"shap": 0.4}, {"shap": 0.1}])
    assert len(up) == 2
    assert down == []


# -------------------------------------------------------------- field labels
def test_every_model_feature_has_a_display_label():
    """A missing label would silently render a raw column name in the UI."""
    assert set(config.FEATURE_COLS) <= set(ui.FIELD_LABELS)


def test_labels_are_unique():
    labels = [ui.FIELD_LABELS[c] for c in config.FEATURE_COLS]
    assert len(set(labels)) == len(labels)


# ------------------------------------------------------- artifacts are intact
def _app_modules() -> list:
    return sorted((config.PROJECT_ROOT / "app").rglob("*.py"))


def test_the_app_reads_a_pipeline_it_did_not_create():
    """Stage 6 must be a reader. Nothing here may fit, dump, or plot."""
    path = config.ARTIFACTS_DIR / "random_forest.pkl"
    assert path.exists(), "Stage 3 artifact missing; the app is a reader, not a trainer"

    forbidden = (".fit(", ".fit_transform(", "joblib.dump", "savefig", ".to_csv(")
    for module in _app_modules():
        text = module.read_text(encoding="utf-8")
        for token in forbidden:
            assert token not in text, f"{module.name} contains {token!r}"


def test_the_app_opens_no_file_for_writing():
    """Stage 4/5 figures and artifacts are inputs; the app must not touch them."""
    for module in _app_modules():
        text = module.read_text(encoding="utf-8")
        for token in ("write_text", "write_bytes", 'open(', "shutil."):
            assert token not in text, f"{module.name} contains {token!r}"


def test_pandas_is_the_only_frame_type_handed_to_the_model():
    """Guards the by-name column selection the ColumnTransformer depends on."""
    features = preprocess.clean(ui.build_raw_row(ANSWERS))[list(config.FEATURE_COLS)]
    assert isinstance(features, pd.DataFrame)
    assert list(features.columns) == list(config.FEATURE_COLS)
