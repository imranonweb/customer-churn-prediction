"""Stage 6 UI helpers.

Two kinds of thing live here:

* **Pure functions** (`risk_band`, `build_raw_row`, `group_shap_by_source`,
  `contribution_rows`, ...) -- no Streamlit, no I/O, unit-testable. These carry the
  logic that must be exactly right: the raw-row contract the Stage 1 preprocessor
  expects, and the regrouping of one-hot SHAP values back to the fields a person
  actually filled in.
* **Render functions** that emit authored HTML for a single component.

Nothing here computes a metric or a SHAP value. Numbers arrive from
`artifacts/` or from the live pipeline and are only formatted.
"""

from __future__ import annotations

from html import escape

import pandas as pd
import streamlit as st

from app.ui_styles import PUSH_AWAY, PUSH_TOWARD, RISK_TOKENS, icon
from src import config

# --------------------------------------------------------------------------
# Risk banding
# --------------------------------------------------------------------------
# Presentation bands only. The project never optimised a decision threshold, so
# these are reading aids for a continuous probability -- said plainly in the UI.
BANDS: tuple[tuple[str, float, float], ...] = (
    ("Low", 0.00, 0.35),
    ("Medium", 0.35, 0.65),
    ("High", 0.65, 1.00),
)


def risk_band(prob: float) -> str:
    """Band name for a churn probability. Lower edge inclusive: 0.35 -> Medium."""
    if not 0.0 <= prob <= 1.0:
        raise ValueError(f"probability out of range: {prob!r}")
    if prob < 0.35:
        return "Low"
    if prob < 0.65:
        return "Medium"
    return "High"


def band_verdict(prob: float) -> str:
    """The 0.50 argmax label the model itself would emit."""
    return "Likely to churn" if prob >= 0.5 else "Likely to stay"


def fmt_pct(prob: float, places: int = 1) -> str:
    return f"{prob * 100:.{places}f}"


def fmt_signed(value: float, places: int = 3) -> str:
    """Signed fixed-point, with a real minus sign kept as ASCII for copy/paste."""
    return f"{value:+.{places}f}"


# --------------------------------------------------------------------------
# Model input contract
# --------------------------------------------------------------------------
# `preprocess.clean()` needs the full raw 21-column frame: it drops customerID
# and maps Churn. Both are placeholders here and neither reaches the model --
# customerID is dropped and Churn becomes the target column, which is sliced off
# by FEATURE_COLS. Verified: flipping the placeholder leaves the feature frame and
# the predicted probability bit-identical.
PLACEHOLDER_ID = "ui-input"
PLACEHOLDER_TARGET = "No"


def build_raw_row(answers: dict[str, object]) -> pd.DataFrame:
    """A one-row raw frame in `config.RAW_COLUMNS` order, ready for `clean()`."""
    row: dict[str, object] = {
        config.ID_COL: PLACEHOLDER_ID,
        config.TARGET: PLACEHOLDER_TARGET,
    }
    row.update(answers)
    missing = [c for c in config.RAW_COLUMNS if c not in row]
    if missing:
        raise KeyError(f"missing raw input columns: {missing}")
    return pd.DataFrame([row], columns=list(config.RAW_COLUMNS))


def estimate_total_charges(tenure: int, monthly: float) -> float:
    """tenure x monthly, which is also the correct 0.0 for a tenure-0 customer.

    Stage 1 imputes blank TotalCharges as 0.0 precisely because all 11 blanks in
    the dataset are tenure-0 rows, so this stays consistent with training data.
    """
    return round(float(tenure) * float(monthly), 2)


# --------------------------------------------------------------------------
# SHAP regrouping
# --------------------------------------------------------------------------
def source_column(transformed: str) -> str:
    """`cat__Contract_Two year` -> `Contract`; `num__avg_charge` -> `avg_charge`."""
    if transformed.startswith("num__"):
        return transformed.removeprefix("num__")
    if transformed.startswith("cat__"):
        # No source column name contains "_", so the first "_" is the level split.
        return transformed.removeprefix("cat__").partition("_")[0]
    return transformed


def group_shap_by_source(names: list[str], values) -> "pd.Series":
    """Sum SHAP values of a field's one-hot levels back onto that field.

    Legitimate because TreeExplainer on a scikit-learn RandomForestClassifier is
    exactly additive in probability space: base + sum(all values) == predict_proba.
    Partitioning that sum by source column therefore preserves it exactly, and it
    removes the duplicate rows a one-hot view would show for one answer.
    """
    frame = pd.DataFrame({"source": [source_column(n) for n in names], "shap": list(values)})
    return frame.groupby("source", sort=False)["shap"].sum()


FIELD_LABELS: dict[str, str] = {
    # engineered / numeric (mirrors explain.NUMERIC_LABELS)
    "tenure": "Tenure",
    "MonthlyCharges": "Monthly charges",
    "TotalCharges": "Total charges",
    "avg_charge": "Average charge per month",
    "num_services": "Number of services",
    "is_new_customer": "New customer (tenure 0)",
    "SeniorCitizen": "Senior citizen",
    # categorical
    "gender": "Gender",
    "Partner": "Partner",
    "Dependents": "Dependents",
    "PhoneService": "Phone service",
    "MultipleLines": "Multiple lines",
    "InternetService": "Internet service",
    "OnlineSecurity": "Online security",
    "OnlineBackup": "Online backup",
    "DeviceProtection": "Device protection",
    "TechSupport": "Tech support",
    "StreamingTV": "Streaming TV",
    "StreamingMovies": "Streaming movies",
    "Contract": "Contract",
    "PaperlessBilling": "Paperless billing",
    "PaymentMethod": "Payment method",
}

_YES_NO = {0: "No", 1: "Yes", 0.0: "No", 1.0: "Yes"}
_MONEY = {"MonthlyCharges", "TotalCharges", "avg_charge"}


def format_field_value(field: str, value: object) -> str:
    """How the customer's own answer reads next to its contribution."""
    if field in ("SeniorCitizen", "is_new_customer"):
        return _YES_NO.get(value, str(value))
    if field in _MONEY:
        return f"${float(value):,.2f}"
    if field == "tenure":
        months = int(value)
        return f"{months} month" if months == 1 else f"{months} months"
    if field == "num_services":
        return str(int(value))
    return str(value)


def contribution_rows(features: pd.DataFrame, names: list[str], shap_values) -> list[dict]:
    """One row per input field, largest absolute contribution first.

    `features` is the single-row frame handed to the pipeline, so the value shown
    beside each contribution is the value the model actually saw.
    """
    grouped = group_shap_by_source(names, shap_values)
    row = features.iloc[0]
    out: list[dict] = []
    for field, contribution in grouped.items():
        out.append(
            {
                "field": field,
                "label": FIELD_LABELS.get(field, field),
                "value": format_field_value(field, row[field]),
                "shap": float(contribution),
            }
        )
    out.sort(key=lambda r: abs(r["shap"]), reverse=True)
    return out


def split_contributions(rows: list[dict], top_n: int = 6) -> tuple[list[dict], list[dict]]:
    """(pushes toward churn, pushes away), each already ordered by magnitude."""
    up = [r for r in rows if r["shap"] > 0][:top_n]
    down = [r for r in rows if r["shap"] < 0][:top_n]
    return up, down


# --------------------------------------------------------------------------
# Render helpers
# --------------------------------------------------------------------------
def _html(markup: str) -> None:
    st.markdown(markup, unsafe_allow_html=True)


def spacer(step: int = 4) -> None:
    """Vertical space from the 4/8/12/16/24/32 scale -- never an inline pixel height.

    Stage 6 sprinkled `<div style="height:.9rem">` through the page code, which put
    layout constants in the Python layer where no stylesheet could reconcile them.
    """
    if step not in (3, 4, 5, 6):
        raise ValueError(f"spacer step must be one of 3, 4, 5, 6; got {step!r}")
    _html(f'<div class="sp-{step}"></div>')


def page_head(
    title: str,
    subtitle: str,
    badges: list[tuple[str, str]] | None = None,
    eyebrow: str = "",
    eyebrow_icon: str = "",
    hero: bool = False,
) -> None:
    """Page title block.

    `hero` is for the landing page only: one page gets display type, the rest get
    the page-title step, so opening the app tells you where the product starts.
    """
    cls = "page-head hero" if hero else "page-head"
    parts = [f'<div class="{cls}">']
    if eyebrow:
        glyph = icon(eyebrow_icon, 13) if eyebrow_icon else ""
        parts.append(f'<div class="eyebrow">{glyph}{escape(eyebrow)}</div>')
    parts.append(f"<h1>{escape(title)}</h1>")
    parts.append(f'<p class="sub">{escape(subtitle)}</p>')
    if badges:
        chips = "".join(
            f'<span class="badge">{icon(ic, 13)}{escape(text)}</span>' for ic, text in badges
        )
        parts.append(f'<div class="badge-row">{chips}</div>')
    parts.append("</div>")
    _html("".join(parts))


def section(title: str, note: str = "", ic: str = "") -> None:
    glyph = icon(ic, 17) if ic else ""
    note_html = f'<p class="note">{escape(note)}</p>' if note else ""
    _html(f'<div class="sec"><h2>{glyph}{escape(title)}</h2>{note_html}</div>')


def group_head(number: int, title: str, ic: str, hint: str = "") -> None:
    hint_html = f'<span class="hint">{escape(hint)}</span>' if hint else ""
    _html(
        f'<div class="grp"><span class="n">{number}</span>{icon(ic, 15)}'
        f'<span class="t">{escape(title)}</span>{hint_html}</div>'
    )


def sub_group_head(title: str, ic: str, hint: str = "") -> None:
    """A second-level heading inside a card: no number, no rule, lighter title."""
    hint_html = f'<span class="hint">{escape(hint)}</span>' if hint else ""
    _html(
        f'<div class="grp sub">{icon(ic, 15)}<span class="t">{escape(title)}</span>'
        f"{hint_html}</div>"
    )


def action_head(title: str, detail: str) -> None:
    _html(
        f'<div class="act-head"><span class="t">{escape(title)}</span>'
        f'<span class="d">{escape(detail)}</span></div>'
    )


def prose(markup: str) -> None:
    """Small trusted HTML fragments written in this file, never user input."""
    _html(f'<div class="prose">{markup}</div>')


def callout(text_html: str, kind: str = "", ic: str = "info") -> None:
    cls = f"callout {kind}".strip()
    _html(f'<div class="{cls}">{icon(ic, 16)}<div class="body">{text_html}</div></div>')


def disclaimer(text: str) -> None:
    _html(f'<p class="disclaimer">{escape(text)}</p>')


def empty_state(title: str, body: str, ic: str = "search") -> None:
    _html(
        f'<div class="empty"><div class="glyph">{icon(ic, 20)}</div>'
        f"<h3>{escape(title)}</h3><p>{escape(body)}</p></div>"
    )


def figure(path, caption: str, key: str, label: str = "") -> None:
    """A saved Stage 4/5 PNG in a frame, or an honest note if it is missing.

    The frame matters: the PNG is placed on `--paper`, the surface
    `src/evaluate.py` rendered it against, inside a card that belongs to this
    page. Without that, a saved chart reads as a pasted screenshot.
    """
    with st.container(key=f"fig-{key}"):
        if path.exists():
            st.image(str(path), width="stretch")
            lbl = f'<span class="lbl">{escape(label)}</span>' if label else ""
            _html(f'<p class="fig-cap">{lbl}<span>{escape(caption)}</span></p>')
        else:
            callout(
                f"<b>Figure not available.</b> Expected <code>{escape(path.name)}</code> in "
                "<code>reports/figures/</code>. Run the earlier stage that generates it.",
                kind="warn",
                ic="alert",
            )


# ---- prediction result ---------------------------------------------------
def result_card(prob: float) -> None:
    """The answer, at the top of its own hierarchy.

    Reading order is fixed by the layout: what is being measured, the number, the
    band it falls in, then the model's own label. Below it the meter places the
    number on a scale, and the footer strip names each readout so the probability
    is never mistaken for a calibrated confidence.
    """
    band = risk_band(prob)
    tok = RISK_TOKENS[band]
    band_icon = {"Low": "down", "Medium": "flat", "High": "up"}[band]

    _html(
        f'<div class="result arrive" style="--accent-bar:{tok["mark"]}">'
        '<div class="top">'
        '<div class="prob">'
        '<div class="k">Churn risk</div>'
        f'<div class="v num">{fmt_pct(prob)}<span class="pc">%</span></div>'
        f'<span class="band" style="color:{tok["text"]};background:{tok["wash"]};'
        f'border-color:{tok["line"]}">{icon(band_icon, 17)}{band} risk</span>'
        "</div>"
        '<div class="side">'
        '<div class="k">Model prediction</div>'
        f'<div class="pred">{icon("target", 18)}{escape(band_verdict(prob))}</div>'
        "</div>"
        "</div>"
        f"{_meter(prob, band)}"
        f"{_facts(prob, band)}"
        "</div>"
    )


def _meter(prob: float, band: str) -> str:
    """Labelled 0-100 track. Colour is the fourth cue here, never the only one."""
    tok = RISK_TOKENS[band]
    pos = max(0.0, min(100.0, prob * 100.0))
    chip_pos = min(max(pos, 6.0), 94.0)
    zones = "".join(
        f'<span class="zone" style="width:{(hi - lo) * 100:.0f}%;'
        f'background:{RISK_TOKENS[name]["wash"]}"></span>'
        for name, lo, hi in BANDS
    )
    scale = "".join(
        f'<span class="{"on" if name == band else ""}" '
        f'style="width:{(hi - lo) * 100:.0f}%">{name}<em>'
        f'{lo * 100:.0f}-{hi * 100:.0f}%</em></span>'
        for name, lo, hi in BANDS
    )
    return (
        '<div class="meter">'
        f'<div class="track">{zones}</div>'
        '<div class="marker">'
        f'<span class="needle" style="left:{pos:.2f}%;background:{tok["mark"]}"></span>'
        f'<span class="valchip num" style="left:{chip_pos:.2f}%;background:{tok["mark"]}">'
        f"{fmt_pct(prob)}%</span>"
        "</div>"
        f'<div class="scale num">{scale}</div>'
        "</div>"
    )


def _facts(prob: float, band: str) -> str:
    """Three named readouts. Each label says exactly what the number is."""
    low, high = next((lo, hi) for name, lo, hi in BANDS if name == band)
    cells = (
        (
            "target",
            "Predicted probability",
            f"{fmt_pct(prob)}%",
            "Raw output of the Random Forest. Not a calibrated confidence score.",
        ),
        (
            "gauge",
            "Risk category",
            band,
            f"Presentation band, {low * 100:.0f}-{high * 100:.0f}% probability.",
        ),
        (
            "scale",
            "Decision cut-off",
            "50%",
            "Where the model's own Yes / No label switches over.",
        ),
    )
    body = "".join(
        f'<div class="f"><div class="k">{icon(ic, 12)}{escape(k)}</div>'
        f'<div class="v num">{escape(v)}</div>'
        f'<div class="sub">{escape(sub)}</div></div>'
        for ic, k, v, sub in cells
    )
    return f'<div class="facts">{body}</div>'


def contribution_bridge(base: float, prob: float) -> None:
    """Baseline -> net effect -> prediction.

    `base` is TreeExplainer's expected value: the forest's *own* mean output, not
    the dataset churn rate. Balanced class weights lift it to roughly a half, so
    the label says "average output" rather than anything about base rates.
    """
    net = prob - base
    arrow = "up" if net >= 0 else "down"
    colour = PUSH_TOWARD if net >= 0 else PUSH_AWAY
    _html(
        '<div class="bridge">'
        '<div class="cell">'
        f'<div class="k">{icon("flat", 12)}Model baseline</div>'
        f'<div class="v num">{fmt_pct(base, 1)}%</div>'
        '<div class="sub">The forest\'s average output, before this customer</div>'
        "</div>"
        '<div class="cell net">'
        f'<div class="k">{icon(arrow, 12)}Net effect of this customer</div>'
        f'<div class="v num" style="color:{colour}">{net * 100:+.1f} pts</div>'
        '<div class="sub">Sum of every factor below</div>'
        "</div>"
        '<div class="cell">'
        f'<div class="k">{icon("target", 12)}Predicted probability</div>'
        f'<div class="v num">{fmt_pct(prob, 1)}%</div>'
        '<div class="sub">What the Random Forest returned</div>'
        "</div>"
        "</div>"
    )


def contribution_list(
    rows: list[dict], direction: str, scale: float, empty_text: str
) -> None:
    """`direction` is 'up' (toward churn) or 'down' (away from churn).

    Each row carries four things: the field, the customer's own answer, the signed
    SHAP value, and a bar whose length is that value relative to the largest
    contribution in this prediction. The bar is a reading aid for the number
    beside it, never a replacement for it.
    """
    toward = direction == "up"
    colour = PUSH_TOWARD if toward else PUSH_AWAY
    head = "Factors increasing risk" if toward else "Factors reducing risk"
    sub = (
        "Answers that pushed this prediction toward churn"
        if toward
        else "Answers that pushed this prediction away from churn"
    )
    parts = [
        f'<div class="contrib {direction}">',
        '<div class="hd">',
        f'<div class="contrib-head">{icon(direction, 16)}{escape(head)}</div>',
        f'<p class="contrib-sub">{escape(sub)}</p>',
        "</div>",
        '<div class="bd">',
    ]
    if not rows:
        parts.append(f'<div class="crow none">{escape(empty_text)}</div>')
    else:
        for r in rows:
            width = min(100.0, abs(r["shap"]) / scale * 100.0) if scale > 0 else 0.0
            parts.append(
                '<div class="crow">'
                '<div class="l">'
                f'<span class="nm">{escape(r["label"])} '
                f'<i>&middot; {escape(r["value"])}</i></span>'
                f'<span class="sv num" style="color:{colour}">'
                f'{fmt_signed(r["shap"])}</span>'
                "</div>"
                f'<div class="bar"><i style="width:{width:.1f}%;'
                f'background:{colour}"></i></div>'
                "</div>"
            )
    parts.append("</div></div>")
    _html("".join(parts))


# ---- metrics ------------------------------------------------------------
def kpi_lead(label: str, value: str, detail: str, ic: str = "target") -> None:
    _html(
        f'<div class="kpi-lead"><div class="k">{icon(ic, 13)}{escape(label)}</div>'
        f'<div class="v num">{escape(value)}</div>'
        f'<div class="d">{escape(detail)}</div></div>'
    )


def note_cards(items: list[tuple[str, str, str]], cols: int = 3) -> None:
    """(icon, title, body) as a grid of small explanatory cards.

    For material that would otherwise become a wall of paragraphs -- what SHAP is,
    how to read a beeswarm, what this study cannot tell you. One idea per card,
    titled, so a reader can take them in any order.
    """
    cards = "".join(
        f'<div class="ncard"><div class="h">{icon(ic, 15)}{escape(title)}</div>'
        f"<p>{body}</p></div>"
        for ic, title, body in items
    )
    _html(f'<div class="ncards" style="--nc:{int(cols)}">{cards}</div>')


def kpi_grid(items: list[tuple[str, str, str]], cols: int = 2) -> None:
    """(label, value, detail) as an even grid of small metric cards.

    The detail line is the point: a bare 0.5316 tells a reader nothing, so every
    supporting metric says in one clause what it measures.
    """
    cards = "".join(
        f'<div class="kpi"><div class="k">{escape(label)}</div>'
        f'<div class="v num">{escape(value)}</div>'
        f'<div class="d">{escape(detail)}</div></div>'
        for label, value, detail in items
    )
    _html(f'<div class="kpi-grid" style="--kc:{int(cols)}">{cards}</div>')



def metrics_table(
    frame: pd.DataFrame,
    highlight_row: str = "",
    label_col: str = "Model",
    places: int = 4,
    note: str = "",
    no_best: tuple[str, ...] = (),
) -> None:
    """Authored table: best value per metric marked in text, selected row tinted.

    `no_best` names columns where "highest" carries no meaning -- a difference
    column, for instance, where the largest gap is not the better model.
    """
    metric_cols = [c for c in frame.columns if c != label_col]
    best = {
        c: frame[c].astype(float).max() for c in metric_cols if c not in no_best
    }

    head = "".join(f"<th>{escape(str(c))}</th>" for c in [label_col, *metric_cols])
    body = []
    for _, r in frame.iterrows():
        sel = " class=\"sel\"" if str(r[label_col]) == highlight_row else ""
        cells = [f"<td>{escape(str(r[label_col]))}</td>"]
        for c in metric_cols:
            value = float(r[c])
            cls = (
                ' class="best"'
                if c in best and abs(value - best[c]) < 1e-12
                else ""
            )
            cells.append(f"<td{cls}>{value:.{places}f}</td>")
        body.append(f"<tr{sel}>{''.join(cells)}</tr>")

    note_html = f'<p class="tbl-note">{escape(note)}</p>' if note else ""
    _html(
        f'<div class="tbl-wrap"><table class="tbl"><thead><tr>{head}</tr></thead>'
        f"<tbody>{''.join(body)}</tbody></table></div>{note_html}"
    )


# ---- pipeline rail ------------------------------------------------------
def pipeline_rail(steps: list[tuple[str, str, bool]]) -> None:
    """(title, description, is_key) rendered as a numbered vertical rail.

    Numbered because the order *is* the information: every step depends on the
    one above it, and the report references stages by number.
    """
    rows = []
    for i, (title, desc, key) in enumerate(steps, start=1):
        cls = "step key" if key else "step"
        rows.append(
            f'<div class="{cls}"><div class="gut"><span class="n">{i}</span>'
            '<span class="ln"></span></div>'
            f'<div class="bd"><div class="t">{escape(title)}</div>'
            f'<div class="d">{escape(desc)}</div></div></div>'
        )
    _html(f'<div class="rail">{"".join(rows)}</div>')


def tech_badges(items: list[tuple[str, str]]) -> None:
    chips = "".join(
        f'<span class="badge tech">{escape(name)}'
        + (f' <span class="v num">{escape(version)}</span>' if version else "")
        + "</span>"
        for name, version in items
    )
    _html(f'<div class="badge-row">{chips}</div>')
