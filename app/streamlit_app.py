"""Stage 6: the Streamlit interface over the finished Stage 1-5 pipeline.

Run from the project root:

    streamlit run app/streamlit_app.py

What this app does and does not do
---------------------------------
It **reads**. Every number on screen comes from one of two places: an artifact
written by an earlier stage (`artifacts/*.csv`, `artifacts/model_metadata.json`,
`reports/figures/*.png`), or a live call into the saved Stage 3 Random Forest
pipeline. Nothing is trained, refitted, recomputed, or hardcoded, and no figure
is written -- the app opens `reports/figures/` read-only.

The one piece of real machinery here is turning a filled-in form into the exact
frame the pipeline expects. That is done by calling Stage 1's own `clean()`
rather than reimplementing its feature engineering: `clean()` is row-wise
stateless by construction (see `run_stage1.py`'s leakage block), so applying it
to a single row is identical to how the training rows were built.

Per-customer SHAP uses Stage 5's own `transform_features` / `compute_shap`. The
41 transformed columns are summed back onto the 22 fields a person actually
filled in -- exact, because TreeExplainer on a scikit-learn forest is additive in
probability space. `explain.plot_waterfall` is deliberately *not* called: it
writes a PNG into `reports/figures/`, and this stage does not touch those files.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
if str(PROJECT_ROOT) not in sys.path:  # so `streamlit run app/...` finds `src/`
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from app import ui_components as ui  # noqa: E402
from app.ui_styles import BRAND_MARK, icon, inject  # noqa: E402
from src import config  # noqa: E402

MODEL_NAME = "Random Forest"
MODEL_FILE = config.ARTIFACTS_DIR / "random_forest.pkl"

PAGES: tuple[tuple[str, str], ...] = (
    ("Predict Churn", "gauge"),
    ("Model Performance", "bars"),
    ("Explainability", "layers"),
    ("About Project", "info"),
)


# ==========================================================================
# Loading -- cached, and every loader fails into a message rather than a trace
# ==========================================================================
@st.cache_resource(show_spinner="Loading the trained Random Forest pipeline...")
def load_pipeline():
    """The Stage 3 pipeline exactly as saved. Cached for the server's lifetime."""
    from src import explain  # imports shap; deferred so the first paint is fast

    return explain.load_model()


@st.cache_data(show_spinner=False)
def load_csv(path_str: str) -> pd.DataFrame | None:
    path = Path(path_str)
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except (OSError, ValueError, pd.errors.ParserError):
        return None


@st.cache_data(show_spinner=False)
def load_metadata() -> dict | None:
    path = config.ARTIFACTS_DIR / "model_metadata.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def test_results() -> pd.DataFrame | None:
    return load_csv(str(config.ARTIFACTS_DIR / "test_results.csv"))


def cv_results() -> pd.DataFrame | None:
    return load_csv(str(config.ARTIFACTS_DIR / "model_cv_results.csv"))


def smote_results() -> pd.DataFrame | None:
    return load_csv(str(config.ARTIFACTS_DIR / "smote_cv_results.csv"))


def rf_test_row(frame: pd.DataFrame | None) -> pd.Series | None:
    if frame is None or "Model" not in frame.columns:
        return None
    match = frame.loc[frame["Model"] == MODEL_NAME]
    return None if match.empty else match.iloc[0]


def missing_artifact(name: str, produced_by: str) -> None:
    ui.callout(
        f"<b>{name} is not available.</b> Expected it in "
        f"<code>artifacts/</code>. Run <code>{produced_by}</code> from the project "
        "root to produce it, then reload this page.",
        kind="warn",
        ic="alert",
    )


# ==========================================================================
# Prediction -- the only place the model is called
# ==========================================================================
def predict_customer(pipe, answers: dict) -> dict:
    """Raw answers -> probability, SHAP contributions, and an additivity check."""
    from src import explain, preprocess

    raw = ui.build_raw_row(answers)
    features = preprocess.clean(raw)[list(config.FEATURE_COLS)]

    proba = float(pipe.predict_proba(features)[0, 1])
    matrix, names = explain.transform_features(pipe, features)
    explanation = explain.compute_shap(pipe, matrix, names)

    values = explanation.values[0]
    base = float(explanation.base_values[0])
    return {
        "proba": proba,
        "base": base,
        "rows": ui.contribution_rows(features, names, values),
        "additivity_error": abs(base + float(values.sum()) - proba),
        "n_transformed": len(names),
    }


# ==========================================================================
# Sidebar
# ==========================================================================
def _select_page(name: str) -> None:
    """Nav click handler.

    This has to be a callback rather than `if st.button(...)`. A callback runs
    *before* the script reruns, so the loop below reads the new page when it
    decides which button is the active one. Assigning inside the loop instead
    would leave the highlight one click behind the content.
    """
    st.session_state["page"] = name


def render_sidebar() -> tuple[str, "st.delta_generator.DeltaGenerator"]:
    """Draw the shell immediately and hand back the slot the status goes in.

    The chrome must not wait on the model: on a cold start the pipeline load takes
    a few seconds, and a blank page for those seconds is worse than a shell with
    an honest "checking" state. The slot is filled once the load has actually
    resolved, so readiness is never claimed before it is true.
    """
    with st.sidebar:
        st.markdown(
            f'<div class="brand">{BRAND_MARK}<div class="txt">'
            '<div class="nm">Churn Intelligence</div>'
            '<div class="rl">Telco retention</div></div></div>',
            unsafe_allow_html=True,
        )
        st.markdown('<div class="nav-label">Sections</div>', unsafe_allow_html=True)

        current = st.session_state.setdefault("page", PAGES[0][0])
        for name, _glyph in PAGES:
            st.button(
                name,
                key=f"nav-{name}",
                type="primary" if name == current else "tertiary",
                width="stretch",
                on_click=_select_page,
                args=(name,),
            )

        st.markdown('<div class="side-foot">', unsafe_allow_html=True)
        status_slot = st.empty()
        status_slot.markdown(
            '<div class="status wait"><span class="dot"></span>Checking model</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="side-meta">'
            f'<div>{icon("doc", 13)}IBM Telco dataset</div>'
            f'<div>{icon("target", 13)}Random Forest</div>'
            f'<div>{icon("layers", 13)}SHAP explainability</div>'
            "</div></div>",
            unsafe_allow_html=True,
        )
    return current, status_slot


def render_status(slot, model_ready: bool) -> None:
    if model_ready:
        slot.markdown(
            '<div class="status ready"><span class="dot"></span>Model Ready</div>',
            unsafe_allow_html=True,
        )
    else:
        slot.markdown(
            '<div class="status down"><span class="dot"></span>Model Unavailable</div>',
            unsafe_allow_html=True,
        )


# ==========================================================================
# Page 1 -- Predict Churn
# ==========================================================================
def customer_form() -> dict:
    """Four grouped panels. Returns raw answers in dataset vocabulary."""
    yn = ("No", "Yes")

    # 1 -- Customer profile ------------------------------------------------
    with st.container(key="panel-profile"):
        ui.group_head(1, "Customer profile", "user", "Demographics")
        left, right = st.columns(2, gap="large")
        with left:
            gender = st.segmented_control(
                "Gender", ("Female", "Male"), default="Female", required=True, key="f_gender"
            )
            partner = st.segmented_control(
                "Has a partner", yn, default="No", required=True, key="f_partner"
            )
        with right:
            senior = st.segmented_control(
                "Senior citizen", yn, default="No", required=True, key="f_senior"
            )
            dependents = st.segmented_control(
                "Has dependents", yn, default="No", required=True, key="f_dependents"
            )

    # 2 -- Account ---------------------------------------------------------
    with st.container(key="panel-account"):
        ui.group_head(2, "Account information", "doc", "Tenure and contract")
        tenure = st.slider(
            "Tenure (months with the company)",
            min_value=0,
            max_value=72,
            value=12,
            step=1,
            key="f_tenure",
            help="0 means the customer has just joined and has not been billed yet.",
        )
        left, right = st.columns([1.7, 1], gap="large")
        with left:
            contract = st.segmented_control(
                "Contract",
                ("Month-to-month", "One year", "Two year"),
                default="Month-to-month",
                required=True,
                key="f_contract",
            )
        with right:
            paperless = st.segmented_control(
                "Paperless billing", yn, default="Yes", required=True, key="f_paperless"
            )

    # 3 -- Services --------------------------------------------------------
    with st.container(key="panel-services"):
        ui.group_head(3, "Services", "grid", "Subscribed products")
        left, right = st.columns([1, 1.6], gap="large")
        with left:
            phone = st.segmented_control(
                "Phone service", yn, default="Yes", required=True, key="f_phone"
            )
        with right:
            internet = st.segmented_control(
                "Internet service",
                ("DSL", "Fiber optic", "No"),
                default="Fiber optic",
                required=True,
                key="f_internet",
            )

        has_phone = phone == "Yes"
        multiple_choice = st.segmented_control(
            "Multiple phone lines",
            yn,
            default="No",
            required=True,
            key="f_multiple",
            disabled=not has_phone,
            help="Requires phone service." if not has_phone else None,
        )
        multiple = multiple_choice if has_phone else "No"

        has_internet = internet != "No"
        st.markdown(
            '<div class="grp" style="margin:1.1rem 0 .3rem">'
            f'{icon("spark", 15)}<span class="t">Internet add-ons</span>'
            + (
                ""
                if has_internet
                else '<span class="hint">Unavailable without internet service</span>'
            )
            + "</div>",
            unsafe_allow_html=True,
        )
        addon_fields = (
            ("OnlineSecurity", "Online security"),
            ("OnlineBackup", "Online backup"),
            ("DeviceProtection", "Device protection"),
            ("TechSupport", "Tech support"),
            ("StreamingTV", "Streaming TV"),
            ("StreamingMovies", "Streaming movies"),
        )
        addons: dict[str, str] = {}
        columns = st.columns(3, gap="large")
        for i, (field, label) in enumerate(addon_fields):
            with columns[i % 3]:
                picked = st.toggle(
                    label, value=False, key=f"f_{field}", disabled=not has_internet
                )
            addons[field] = "Yes" if (picked and has_internet) else "No"

    # 4 -- Billing ---------------------------------------------------------
    with st.container(key="panel-billing"):
        ui.group_head(4, "Billing", "billing", "Charges and payment")
        monthly = st.slider(
            "Monthly charges (USD)",
            min_value=18.0,
            max_value=120.0,
            value=70.0,
            step=0.05,
            format="$%.2f",
            key="f_monthly",
        )
        payment = st.selectbox(
            "Payment method",
            (
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ),
            index=0,
            key="f_payment",
        )

        estimated = ui.estimate_total_charges(tenure, monthly)
        derive = st.toggle(
            "Estimate total charges from tenure × monthly charges",
            value=True,
            key="f_derive_total",
            help=(
                "The model uses TotalCharges, plus an average-charge feature derived "
                "from it. Turn this off to enter a billed total directly."
            ),
        )
        if derive:
            total = estimated
            st.markdown(
                f'<p class="cta-note">Total charges: <b class="num">${total:,.2f}</b> '
                f"({tenure} × ${monthly:,.2f}). A brand-new customer correctly gets "
                "$0.00, matching how Stage 1 treats tenure-0 rows.</p>",
                unsafe_allow_html=True,
            )
        else:
            total = st.number_input(
                "Total charges to date (USD)",
                min_value=0.0,
                max_value=12000.0,
                value=float(estimated),
                step=10.0,
                format="%.2f",
                key="f_total",
            )

    return {
        "gender": gender,
        "SeniorCitizen": 1 if senior == "Yes" else 0,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": int(tenure),
        "PhoneService": phone,
        "MultipleLines": multiple,
        "InternetService": internet,
        **addons,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": round(float(monthly), 2),
        "TotalCharges": round(float(total), 2),
    }


def render_prediction(result: dict) -> None:
    ui.result_card(result["proba"])

    ui.section(
        "Why the model predicted this",
        "SHAP decomposes this single prediction into a contribution per field you "
        "entered. Values are in probability units and sum exactly to the "
        "difference between the model's baseline and this prediction.",
        "layers",
    )
    ui.contribution_bridge(result["base"], result["proba"])
    ui.prose(
        "<p class=\"bridge-note\">The baseline is the forest's own average output, "
        "not the dataset's churn rate. Every estimator was fitted with balanced "
        "class weights, which lifts that average to roughly a half &mdash; so a "
        "prediction above it is above what this model considers typical.</p>"
    )

    rows = result["rows"]
    up, down = ui.split_contributions(rows)
    scale = max((abs(r["shap"]) for r in rows), default=1.0)

    left, right = st.columns(2, gap="large")
    with left:
        ui.contribution_list(
            up, "up", scale, "No field pushed this prediction toward churn."
        )
    with right:
        ui.contribution_list(
            down, "down", scale, "No field pushed this prediction away from churn."
        )

    with st.expander("All 22 fields, and the additivity check"):
        table = pd.DataFrame(
            {
                "Field": [r["label"] for r in rows],
                "Value used": [r["value"] for r in rows],
                "SHAP contribution": [round(r["shap"], 6) for r in rows],
                "Direction": [
                    "toward churn" if r["shap"] > 0 else
                    ("away from churn" if r["shap"] < 0 else "no effect")
                    for r in rows
                ],
            }
        )
        st.dataframe(table, width="stretch", hide_index=True)
        ui.prose(
            f"<p>Baseline <b>{result['base']:.6f}</b> plus the "
            f"{len(rows)} contributions above reproduces the predicted probability "
            f"<b>{result['proba']:.6f}</b> to within "
            f"<b>{result['additivity_error']:.2e}</b>. The model's encoder expands "
            f"these fields into {result['n_transformed']} columns; the contributions "
            "of a field's encoded levels are summed back onto that field, which "
            "leaves the total unchanged.</p>"
        )

    st.markdown('<div style="height:.9rem"></div>', unsafe_allow_html=True)
    ui.disclaimer(
        "These factors explain the model's prediction and do not establish causal "
        "relationships."
    )


def page_predict(pipe) -> None:
    ui.page_head(
        "Customer Churn Intelligence",
        "Predict customer churn risk and understand the factors influencing the "
        "model's decision.",
        [
            ("doc", "IBM Telco Dataset"),
            ("target", "Random Forest"),
            ("layers", "Explainable AI"),
        ],
    )

    if pipe is None:
        ui.callout(
            "<b>Unable to load the trained model. Please verify that the required "
            f"artifact exists.</b> Expected <code>{MODEL_FILE.name}</code> in "
            "<code>artifacts/</code>. Run <code>python run_stage1.py</code> then "
            "<code>python -m src.train</code> from the project root to rebuild it.",
            kind="warn",
            ic="alert",
        )
        return

    answers = customer_form()

    st.markdown('<div style="height:.4rem"></div>', unsafe_allow_html=True)
    clicked = st.button(
        "Analyze Customer", type="primary", width="stretch", key="analyze"
    )
    st.markdown(
        '<p class="cta-note">Runs the saved Random Forest pipeline on this one '
        "customer. Nothing is stored.</p>",
        unsafe_allow_html=True,
    )

    if clicked:
        with st.spinner("Analyzing customer risk...", show_time=False):
            try:
                # Per-session, never a global cache: one user's customer must not
                # become another user's result.
                st.session_state["result"] = predict_customer(pipe, answers)
                st.session_state.pop("result_error", None)
            except Exception as exc:  # noqa: BLE001 - surfaced, never a traceback
                st.session_state.pop("result", None)
                st.session_state["result_error"] = str(exc)

    st.markdown('<div style="height:1.5rem"></div>', unsafe_allow_html=True)

    if st.session_state.get("result_error"):
        ui.callout(
            "<b>The analysis could not be completed.</b> The inputs did not reach "
            "the model in the expected form. Details: <code>"
            f"{st.session_state['result_error']}</code>",
            kind="warn",
            ic="alert",
        )
    elif "result" in st.session_state:
        render_prediction(st.session_state["result"])
    else:
        ui.empty_state(
            "No analysis yet",
            "Enter customer information and run an analysis to estimate churn risk.",
            "search",
        )


# ==========================================================================
# Page 2 -- Model Performance
# ==========================================================================
def page_performance() -> None:
    ui.page_head(
        "Model Performance",
        "Evaluation of five machine learning models on the untouched test set.",
    )

    test = test_results()
    row = rf_test_row(test)
    if row is None:
        missing_artifact("test_results.csv", "python run_stage4.py")
        return

    metadata = load_metadata() or {}
    training = metadata.get("training_data", {})
    churn_rate = training.get("churn_rate")
    baseline = 1 - churn_rate if isinstance(churn_rate, (int, float)) else None

    left, right = st.columns([1, 1.25], gap="large")
    with left:
        ui.kpi_lead(
            "PR-AUC (selection metric)",
            f"{row['PR-AUC']:.4f}",
            f"Highest of the five models on the held-out test set. {MODEL_NAME} was "
            "selected on this metric because it measures performance on the "
            "minority churn class specifically.",
        )
    with right:
        items = [
            ("Recall", "Share of real churners identified", f"{row['Recall']:.4f}"),
            ("F1", "Balance of precision and recall", f"{row['F1']:.4f}"),
            ("ROC-AUC", "Ranking quality across thresholds", f"{row['ROC-AUC']:.4f}"),
            ("Precision", "Share of churn flags that were right", f"{row['Precision']:.4f}"),
        ]
        ui.kpi_row(items)

    st.markdown('<div style="height:.6rem"></div>', unsafe_allow_html=True)
    accuracy_note = (
        f"Accuracy is {row['Accuracy']:.4f}. It is reported but was not used for "
        "selection: always predicting \"no churn\" already scores "
        f"{baseline:.4f} on this data while identifying no churners at all."
        if baseline is not None
        else f"Accuracy is {row['Accuracy']:.4f}, reported but not used for selection."
    )
    ui.callout(f"<b>On accuracy.</b> {accuracy_note}", kind="info", ic="info")

    # -- why RF ------------------------------------------------------------
    ui.section("Why Random Forest?", "", "target")
    cv = cv_results()
    cv_rf = None
    if cv is not None and "Model" in cv.columns:
        match = cv.loc[cv["Model"] == MODEL_NAME]
        cv_rf = None if match.empty else match.iloc[0]

    best_accuracy = test.loc[test["Accuracy"].idxmax()]
    bullets = [
        f"It scored the highest test PR-AUC of the five models, <b>{row['PR-AUC']:.4f}</b>, "
        "and PR-AUC was fixed as the selection metric before evaluation.",
        f"It also recovered the largest share of real churners, recall <b>{row['Recall']:.4f}</b> "
        "&mdash; the error that costs a telecom operator money is the churner it never flags.",
    ]
    if cv_rf is not None:
        bullets.append(
            f"The choice was made on cross-validation, not on the test set: "
            f"{MODEL_NAME} led 5-fold CV PR-AUC at <b>{cv_rf['PR-AUC']:.4f}</b> "
            f"(&plusmn;{cv_rf['PR-AUC_std']:.4f}), and the test result agrees."
        )
    if str(best_accuracy["Model"]) != MODEL_NAME:
        bullets.append(
            f"{best_accuracy['Model']} reached higher accuracy "
            f"(<b>{best_accuracy['Accuracy']:.4f}</b>) but found far fewer churners "
            f"(recall <b>{best_accuracy['Recall']:.4f}</b>) and had the weakest "
            f"PR-AUC (<b>{best_accuracy['PR-AUC']:.4f}</b>). On this problem that is "
            "the worse model, which is exactly why accuracy was not the criterion."
        )
    ui.prose("<ul>" + "".join(f"<li>{b}</li>" for b in bullets) + "</ul>")

    # -- comparison --------------------------------------------------------
    ui.section(
        "Model comparison",
        "All five models, same split, same metrics. Best value in each column is "
        "marked.",
        "bars",
    )
    ui.figure(
        config.FIGURES_DIR / "model_comparison.png",
        "Stage 4: cross-validation and test metrics side by side for the five models.",
        "compare",
    )
    ui.metrics_table(
        test,
        highlight_row=MODEL_NAME,
        note="Source: artifacts/test_results.csv (Stage 4). Test set: 1,409 customers, "
        "held out before any model was fitted.",
    )

    if cv is not None:
        with st.expander("Cross-validation results (Stage 3, mean of 5 stratified folds)"):
            keep = ["Model", *[c for c in cv.columns if c in
                              ("Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC")]]
            ui.metrics_table(
                cv[keep],
                highlight_row=MODEL_NAME,
                note="Source: artifacts/model_cv_results.csv. Model selection used "
                "these numbers; the test set was opened once, afterwards.",
            )

    # -- test performance figures -----------------------------------------
    ui.section(
        "Test set performance",
        f"{MODEL_NAME} on the 1,409 held-out customers.",
        "target",
    )
    ui.figure(
        config.FIGURES_DIR / "confusion_matrix_random_forest.png",
        "Confusion matrix. The bottom-left cell is the costly error: a churner the "
        "model did not flag.",
        "cm",
    )
    # Full width rather than two-up: both figures are ~1040px of matplotlib text,
    # and halving them makes the tick labels unreadable.
    ui.figure(
        config.FIGURES_DIR / "roc_curves.png",
        "ROC curves for all five models.",
        "roc",
    )
    ui.figure(
        config.FIGURES_DIR / "pr_curves.png",
        "Precision-recall curves, the view that matters for the 26.5% minority "
        "class.",
        "pr",
    )

    # -- imbalance ---------------------------------------------------------
    render_imbalance(metadata)


def render_imbalance(metadata: dict) -> None:
    ui.section(
        "Class imbalance",
        "Churn is the minority class, so an untreated model can score well on "
        "accuracy while finding almost no churners.",
        "flat",
    )

    strategy = metadata.get("class_imbalance_strategy", {})
    training = metadata.get("training_data", {})
    facts = []
    if isinstance(training.get("churn_rate"), (int, float)):
        facts.append(
            f"The training set is <b>{training['churn_rate'] * 100:.2f}%</b> churners "
            f"({training.get('churn_positive', '?')} of {training.get('rows', '?')} rows)."
        )
    if strategy.get("approach"):
        facts.append(f"Approach used: <b>{strategy['approach']}</b>.")
    if isinstance(strategy.get("scale_pos_weight_value"), (int, float)):
        facts.append(
            "The four scikit-learn models use <code>class_weight=\"balanced\"</code>; "
            "XGBoost uses the equivalent "
            f"<code>scale_pos_weight={strategy['scale_pos_weight_value']:.4f}</code>."
        )
    facts.append(
        "Re-weighting happens inside the estimator, so it never changes the data and "
        "cannot leak across cross-validation folds."
    )
    ui.prose("<ul>" + "".join(f"<li>{f}</li>" for f in facts) + "</ul>")

    smote = smote_results()
    if smote is None or "Strategy" not in smote.columns:
        ui.prose(
            "<p>A SMOTE comparison was run in Stage 3B and was <b>not adopted</b>. "
            "Its results artifact is not present, so no figures are quoted here.</p>"
        )
        return

    pivot = smote.pivot_table(index="Model", columns="Strategy", values="PR-AUC")
    if not {"Class Weight", "SMOTE"}.issubset(pivot.columns):
        return
    table = pd.DataFrame(
        {
            "Model": pivot.index,
            "Class weighting": pivot["Class Weight"].to_numpy(),
            "SMOTE": pivot["SMOTE"].to_numpy(),
            "Difference": (pivot["Class Weight"] - pivot["SMOTE"]).to_numpy(),
        }
    )
    wins = int((table["Difference"] > 0).sum())
    ui.prose(
        f"<p>Stage 3B re-ran the same 5-fold cross-validation with SMOTE oversampling "
        f"inside the pipeline. Class weighting produced a higher PR-AUC for "
        f"<b>{wins} of {len(table)}</b> models, so <b>SMOTE was not adopted</b> &mdash; "
        "it added synthetic rows without buying accuracy on the metric that was "
        "chosen for selection.</p>"
    )
    ui.metrics_table(
        table.sort_values("Difference", ascending=False),
        highlight_row=MODEL_NAME,
        places=4,
        no_best=("Difference",),
        note="Cross-validated PR-AUC. Source: artifacts/smote_cv_results.csv "
        "(Stage 3B). Positive difference favours class weighting.",
    )
    ui.figure(
        config.FIGURES_DIR / "smote_comparison.png",
        "Stage 3B: class weighting against SMOTE across the five models.",
        "smote",
    )


# ==========================================================================
# Page 3 -- Explainability
# ==========================================================================
def page_explainability() -> None:
    ui.page_head(
        "Explainable AI",
        "Understand which factors influence the model and how individual "
        "predictions are formed.",
    )

    ui.callout(
        "<b>Explaining one customer.</b> The figures below describe the model over "
        "the whole test set. To explain a customer you type in yourself, use "
        "<b>Predict Churn</b> &mdash; it runs the same SHAP method on that single row and "
        "lists the fields that pushed the prediction each way.",
        kind="info",
        ic="spark",
    )

    ui.section(
        "Which features the model relies on",
        "Mean absolute SHAP value per feature: how much each one moves predictions, "
        "regardless of direction.",
        "bars",
    )
    ui.figure(
        config.FIGURES_DIR / "shap_feature_importance.png",
        "Stage 5: the features with the largest average influence on the Random "
        "Forest's predicted churn probability.",
        "shap-global",
    )

    ui.section("Reading the beeswarm plot", "", "layers")
    ui.prose(
        "<p>The beeswarm is the densest view of the model, and it is worth learning "
        "to read:</p>"
        "<ul>"
        "<li><b>Each dot is one customer</b> in the test set, so a row shows all "
        "1,409 of them at once for a single feature.</li>"
        "<li><b>Horizontal position is that customer's SHAP value.</b> Right of the "
        "centre line means the feature pushed <i>that</i> customer's prediction "
        "toward churn; left means it pushed it away.</li>"
        "<li><b>Colour is the feature's own value</b> for that customer, not its "
        "importance. For a yes/no feature, one colour is \"yes\" and the other "
        "\"no\"; for a number, the ramp runs low to high.</li>"
        "<li><b>Where dots pile up</b> tells you how common an effect is, and how "
        "wide the spread is tells you how much the effect depends on the rest of "
        "that customer's profile.</li>"
        "<li><b>Rows are ordered by influence</b>, strongest at the top &mdash; the same "
        "ordering as the chart above.</li>"
        "</ul>"
        "<p>So a row whose high values sit on the right and low values on the left "
        "is a feature where <i>more</i> of it means <i>more</i> predicted churn "
        "risk, according to this model.</p>"
    )
    ui.figure(
        config.FIGURES_DIR / "shap_beeswarm.png",
        "Stage 5: per-customer SHAP values for the most influential features.",
        "shap-beeswarm",
    )

    ui.section(
        "Two example customers",
        "Stage 5 explained the single highest-risk and single lowest-risk customer "
        "in the test set. Both plots start at the model's baseline and add one "
        "feature at a time until they reach that customer's predicted probability.",
        "target",
    )
    # Waterfall plots carry a long feature label per row; two-up would shrink them
    # past reading size.
    ui.figure(
        config.FIGURES_DIR / "shap_customer_high_risk.png",
        "Highest predicted risk in the test set.",
        "shap-high",
    )
    ui.figure(
        config.FIGURES_DIR / "shap_customer_low_risk.png",
        "Lowest predicted risk in the test set.",
        "shap-low",
    )

    st.markdown('<div style="height:.6rem"></div>', unsafe_allow_html=True)
    ui.disclaimer(
        "SHAP explains how the trained model produced its predictions. It shows "
        "associations learned by the model and should not be interpreted as proof "
        "that a feature causes customer churn."
    )


# ==========================================================================
# Page 4 -- About Project
# ==========================================================================
PIPELINE_STEPS: tuple[tuple[str, str, bool], ...] = (
    ("Raw data", "7,043 customers and 21 columns, checksum-verified on download.", False),
    ("Data cleaning", "Blank TotalCharges resolved, structural categories collapsed, "
                      "customerID dropped before anything else.", False),
    ("Feature engineering", "Three row-wise features added: is_new_customer, "
                            "num_services, avg_charge.", False),
    ("Exploratory analysis", "Class balance, tenure and charge distributions, churn by "
                             "contract and payment method, correlations.", False),
    ("Five ML models", "Logistic Regression, Decision Tree, Random Forest, SVM, "
                       "XGBoost — each inside its own pipeline.", False),
    ("Class imbalance handling", "Cost re-weighting inside every estimator; no "
                                 "resampling of the data.", False),
    ("SMOTE comparison", "The same cross-validation re-run with SMOTE, kept as "
                         "evidence and not adopted.", False),
    ("Model evaluation", "Six metrics, confusion matrices, ROC and PR curves on the "
                         "held-out test set.", False),
    ("Random Forest selection", "Chosen on cross-validated PR-AUC before the test set "
                                "was opened.", True),
    ("SHAP explainability", "Global importance, beeswarm, and per-customer "
                            "contributions with an additivity check.", True),
    ("Interactive application", "This interface, reading the saved pipeline and the "
                                "stage artifacts.", True),
)

LIMITATIONS: tuple[str, ...] = (
    "The model was trained on one telecom provider's customers. Contract types, "
    "payment methods, and service mix differ between markets, so the numbers here "
    "should not be transferred to another operator without retraining.",
    "SHAP explains the model, not the world. A feature can matter to the model "
    "because it correlates with something the dataset never recorded, so nothing "
    "here establishes cause.",
    "The dataset is a historical snapshot with no timestamps. It cannot express "
    "seasonality, a tariff change, or a competitor entering the market.",
    "There is no production monitoring. If real customer behaviour drifted away "
    "from this snapshot, this application would keep answering confidently and "
    "would not notice.",
    "The Low / Medium / High bands are presentation categories for reading a "
    "probability. No decision threshold was optimised, and no cost matrix was "
    "supplied to optimise one against.",
)


def tech_stack(metadata: dict) -> list[tuple[str, str]]:
    """Real versions: metadata first (what trained the model), then what is installed."""
    from importlib.metadata import PackageNotFoundError, version

    recorded = (metadata or {}).get("library_versions", {}) or {}
    wanted = (
        ("Python", "python"),
        ("NumPy", "numpy"),
        ("pandas", "pandas"),
        ("scikit-learn", "scikit-learn"),
        ("XGBoost", "xgboost"),
        ("imbalanced-learn", "imbalanced-learn"),
        ("SHAP", "shap"),
        ("Matplotlib", "matplotlib"),
        ("seaborn", "seaborn"),
        ("joblib", "joblib"),
        ("Streamlit", "streamlit"),
    )
    out: list[tuple[str, str]] = []
    for label, key in wanted:
        found = recorded.get(key, "")
        if not found and key != "python":
            try:
                found = version(key)
            except PackageNotFoundError:
                found = ""
        out.append((label, str(found)))
    return out


def page_about() -> None:
    ui.page_head(
        "About This Project",
        "A churn prediction study built end to end: one dataset, five models, an "
        "explained winner, and this interface over it.",
    )

    metadata = load_metadata() or {}

    ui.section("The problem", "", "info")
    ui.prose(
        "<p>Acquiring a telecom subscriber costs far more than keeping one, so an "
        "operator's retention team needs two things from a model, not one. First, "
        "<b>which customers are at risk</b> &mdash; and specifically a model that finds "
        "churners rather than one that looks accurate by agreeing with the majority. "
        "Second, <b>why the model says so</b>, because a risk score with no reason "
        "attached cannot be acted on and will not be trusted.</p>"
        "<p>This project answers both. It compares five classifiers on the metric "
        "that actually reflects minority-class performance, then uses SHAP to "
        "decompose the chosen model's predictions into per-feature contributions "
        "for the whole test set and for any single customer.</p>"
    )

    ui.section("Dataset", "", "doc")
    training = metadata.get("training_data", {})
    features = metadata.get("features", {})
    dataset_facts = [
        "<b>IBM Telco Customer Churn</b> &mdash; 7,043 customers, 21 columns, one row per "
        "customer.",
        "<b>Target:</b> <code>Churn</code> (Yes / No). 1,869 customers churned, "
        "26.54% of the dataset.",
        "<b>Demographics:</b> gender, senior citizen, partner, dependents.",
        "<b>Account:</b> tenure in months, contract type, paperless billing.",
        "<b>Services:</b> phone, multiple lines, internet type, and six add-ons "
        "(security, backup, device protection, tech support, streaming TV, streaming "
        "movies).",
        "<b>Billing:</b> monthly charges, total charges, payment method.",
    ]
    if isinstance(features.get("input_columns"), int):
        dataset_facts.append(
            f"<b>Model input:</b> {features['input_columns']} columns after "
            f"engineering, expanded to {features.get('columns_after_encoding', '?')} "
            "by one-hot encoding."
        )
    if isinstance(training.get("rows"), int):
        dataset_facts.append(
            f"<b>Split:</b> {training['rows']:,} training rows and 1,409 test rows, "
            "stratified, seed 42. The test set was untouched until Stage 4."
        )
    ui.prose("<ul>" + "".join(f"<li>{f}</li>" for f in dataset_facts) + "</ul>")

    ui.section(
        "Project workflow",
        "Each step consumes only what the step above it produced.",
        "arrow-right",
    )
    ui.pipeline_rail(list(PIPELINE_STEPS))

    ui.section("Models evaluated", "", "bars")
    test = test_results()
    if test is not None and {"Model", "PR-AUC"}.issubset(test.columns):
        ranked = test.sort_values("PR-AUC", ascending=False)
        ui.metrics_table(
            ranked[["Model", "PR-AUC", "Recall", "F1"]],
            highlight_row=MODEL_NAME,
            note="Ranked by test PR-AUC, the selection metric. Full table on the "
            "Model Performance page.",
        )
    else:
        ui.prose(
            "<ul><li>Logistic Regression</li><li>Decision Tree</li>"
            "<li>Random Forest (selected)</li><li>SVM</li><li>XGBoost</li></ul>"
        )

    ui.section("Tech stack", "Versions the pipeline was actually built with.", "grid")
    ui.tech_badges(tech_stack(metadata))

    ui.section("Project limitations", "", "alert")
    ui.prose("<ul>" + "".join(f"<li>{text}</li>" for text in LIMITATIONS) + "</ul>")

    st.markdown('<div style="height:1.2rem"></div>', unsafe_allow_html=True)
    ui.prose(
        "<p><b>Customer Churn Prediction using Machine Learning with Explainable "
        "AI</b><br>Md. Al-Imran Emon (232002136) and Abu Sowad (232002191)<br>"
        "Supervised by Ms. Jannathul Moawa Hasi &middot; Department of Computer "
        "Science and Engineering, Green University of Bangladesh</p>"
    )


# ==========================================================================
# Entry point
# ==========================================================================
def main() -> None:
    st.set_page_config(
        page_title="Churn Intelligence",
        page_icon=str(config.FIGURES_DIR / "churn_distribution.png")
        if (config.FIGURES_DIR / "churn_distribution.png").exists()
        else None,
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject()

    page, status_slot = render_sidebar()

    pipe = None
    try:
        pipe = load_pipeline()
    except FileNotFoundError:
        pass
    except Exception as exc:  # noqa: BLE001 - a message, never a traceback
        st.session_state["load_error"] = str(exc)

    render_status(status_slot, model_ready=pipe is not None)

    if st.session_state.get("load_error"):
        ui.callout(
            "<b>Unable to load the trained model. Please verify that the required "
            "artifact exists.</b> It was found on disk but could not be read: "
            f"<code>{st.session_state['load_error']}</code>",
            kind="warn",
            ic="alert",
        )

    if page == "Predict Churn":
        page_predict(pipe)
    elif page == "Model Performance":
        page_performance()
    elif page == "Explainability":
        page_explainability()
    else:
        page_about()


if __name__ == "__main__":
    main()
