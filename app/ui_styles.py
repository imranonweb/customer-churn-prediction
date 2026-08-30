"""Stage 6 design system: tokens, authored icons, and the single stylesheet.

Why the palette is what it is
-----------------------------
The app embeds Stage 4 and Stage 5 PNGs directly, so it inherits their world
rather than inventing a second one. `src/evaluate.py` fixes the figure surface
at `#fcfcfb`, the ink ramp at `#0b0b0b / #52514e / #898781`, rules at `#e1e0d9`,
and a documented single-hue blue ramp whose dark steps include `#184f95`. The UI
primary is that step -- so a chart dropped onto a panel sits on its own paper.

Every colour below was measured, not chosen by eye (WCAG contrast against the
`#fcfcfb` surface, OKLab dE x100 for separation, dichromat simulation for CVD):

    token          hex        contrast   role
    ink            #111110    18.40:1    body
    ink-2          #52514e     7.73:1    secondary
    ink-3          #6e6c66     5.11:1    captions
    primary        #184f95      7.89:1   actions, links, selection
    primary-hover  #123c73     10.66:1
    low-text       #0e6b3d      6.41:1   "Low" label
    med-text       #8f5f00      5.38:1   "Medium" label
    high-text      #a32b29      6.97:1   "High" label
    low-mark       #12824a      4.74:1   meter fill / dot   (needs >= 3:1)
    med-mark       #bd7f12      3.30:1
    high-mark      #c8332f      5.15:1

The three risk marks separate at worst dE 16.4 for normal vision (floor 15).
Under simulated deuteranopia the green/red pair falls to dE 3.3 -- which is the
known, unavoidable cost of traffic-light semantics. That is legal only with a
second, non-colour encoding, so risk is *never* carried by colour here: every
risk readout ships the band name in words, the probability as a number, a drawn
icon, and a marker position on a labelled 0-100 scale. Remove all colour and the
page still reads.

SHAP direction reuses Stage 5's own pair exactly -- `#c8332f` family for "pushed
toward churn", `#184f95` family for "pushed away" -- so the live explanation and
the saved beeswarm agree on which way red means.
"""

from __future__ import annotations

import streamlit as st

# --------------------------------------------------------------------------
# Tokens needed on the Python side (inline styles for data-driven positions).
# Everything else lives in the CSS custom properties below, single source.
# --------------------------------------------------------------------------
RISK_TOKENS = {
    "Low": {"mark": "#12824a", "text": "#0e6b3d", "wash": "#e8f4ec"},
    "Medium": {"mark": "#bd7f12", "text": "#8f5f00", "wash": "#fbf1de"},
    "High": {"mark": "#c8332f", "text": "#a32b29", "wash": "#fbeceb"},
}

PUSH_TOWARD = "#c8332f"   # SHAP > 0, matches explain.PUSH_TOWARD's role
PUSH_AWAY = "#184f95"     # SHAP < 0, matches explain.PUSH_AWAY's role
INK_3 = "#6e6c66"


# --------------------------------------------------------------------------
# Icons -- authored, one grid (16x16), one stroke weight (1.5), currentColor.
# No emoji and no glyph substitutes: an icon set has to be a set.
# --------------------------------------------------------------------------
def icon(name: str, size: int = 16, cls: str = "") -> str:
    """Inline SVG for `name`, inheriting colour from its container."""
    body = _ICON_PATHS.get(name, "")
    return (
        f'<svg class="ic {cls}" width="{size}" height="{size}" viewBox="0 0 16 16" '
        'fill="none" stroke="currentColor" stroke-width="1.5" '
        'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true" '
        f'focusable="false">{body}</svg>'
    )


_ICON_PATHS: dict[str, str] = {
    # navigation
    "gauge": '<path d="M2.4 12.2a5.6 5.6 0 1 1 11.2 0"/><path d="M8 12.2 10.9 8.4"/>',
    "bars": '<path d="M3.2 13.2V8.2"/><path d="M8 13.2V3.4"/><path d="M12.8 13.2V6.6"/>',
    "layers": '<path d="M8 2.3 14 5.2 8 8.1 2 5.2z"/><path d="M2 8.6 8 11.5 14 8.6"/>',
    "info": '<circle cx="8" cy="8" r="5.9"/><path d="M8 7.4v4"/><path d="M8 5.05h.01"/>',
    # form groups
    "user": '<circle cx="8" cy="5.8" r="2.7"/><path d="M2.9 13.7a5.3 5.3 0 0 1 10.2 0"/>',
    "doc": '<path d="M4 2.3h5l3.1 3.1v8.3H4z"/><path d="M9 2.3v3.1h3.1"/>'
           '<path d="M6.2 9.1h3.6"/><path d="M6.2 11.3h2.4"/>',
    "grid": '<rect x="2.6" y="2.6" width="4.6" height="4.6" rx="1"/>'
            '<rect x="8.8" y="2.6" width="4.6" height="4.6" rx="1"/>'
            '<rect x="2.6" y="8.8" width="4.6" height="4.6" rx="1"/>'
            '<rect x="8.8" y="8.8" width="4.6" height="4.6" rx="1"/>',
    "billing": '<rect x="2.2" y="4.2" width="11.6" height="7.6" rx="1.2"/>'
               '<path d="M2.2 7.1h11.6"/><path d="M4.6 9.6h2.6"/>',
    # data / state
    "up": '<path d="M8 12.6V3.9"/><path d="M4.6 7.3 8 3.9l3.4 3.4"/>',
    "down": '<path d="M8 3.4v8.7"/><path d="M11.4 8.7 8 12.1 4.6 8.7"/>',
    "flat": '<path d="M3.6 8h8.8"/>',
    "check": '<path d="M3.3 8.5 6.3 11.5 12.7 4.9"/>',
    "alert": '<path d="M8 2.6 14.2 13H1.8z"/><path d="M8 6.6v3"/><path d="M8 11.4h.01"/>',
    "search": '<circle cx="7.1" cy="7.1" r="4.4"/><path d="M10.4 10.4 13.6 13.6"/>',
    "spark": '<path d="M8 2.2 9.3 6.1 13.2 7.4 9.3 8.7 8 12.6 6.7 8.7 2.8 7.4 6.7 6.1z"/>',
    "lock": '<rect x="3.4" y="7.1" width="9.2" height="6.4" rx="1.2"/>'
            '<path d="M5.8 7.1V5.3a2.2 2.2 0 0 1 4.4 0v1.8"/>',
    "target": '<circle cx="8" cy="8" r="5.9"/><circle cx="8" cy="8" r="2.5"/>',
    "arrow-right": '<path d="M3.2 8h9.6"/><path d="M9.4 4.6 12.8 8l-3.4 3.4"/>',
}

BRAND_MARK = (
    '<svg width="30" height="30" viewBox="0 0 30 30" fill="none" aria-hidden="true">'
    '<rect width="30" height="30" rx="8" fill="#184f95"/>'
    '<path d="M7 20.4 11.2 14.6 14.6 17.4 19.3 9.8" stroke="#ffffff" '
    'stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round"/>'
    '<circle cx="19.3" cy="9.8" r="2.5" fill="#ffffff"/>'
    '<path d="M7 23.6h16" stroke="#7fa9de" stroke-width="1.5" stroke-linecap="round"/>'
    "</svg>"
)


# --------------------------------------------------------------------------
# Stylesheet
# --------------------------------------------------------------------------
_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Archivo:wght@400;500;600;700&display=swap');

:root, .stApp {
  --font: 'Archivo', 'Segoe UI Variable Text', 'Segoe UI', -apple-system,
          BlinkMacSystemFont, Roboto, 'Helvetica Neue', Arial, sans-serif;

  --surface: #fcfcfb;
  --panel: #ffffff;
  --sunken: #f5f4f1;
  --sidebar: #f7f6f3;
  --line: #e5e3dc;
  --line-strong: #cfcdc4;

  --ink: #111110;
  --ink-2: #52514e;
  --ink-3: #6e6c66;

  --primary: #184f95;
  --primary-hover: #123c73;
  --primary-wash: #eef3fa;
  --primary-line: #c5d8ef;

  --low-mark: #12824a;   --low-text: #0e6b3d;   --low-wash: #e8f4ec;
  --med-mark: #bd7f12;   --med-text: #8f5f00;   --med-wash: #fbf1de;
  --high-mark: #c8332f;  --high-text: #a32b29;  --high-wash: #fbeceb;

  --r-sm: 6px; --r-md: 10px; --r-lg: 14px;
  --shadow-sm: 0 1px 2px rgba(17,17,16,.05), 0 1px 1px rgba(17,17,16,.03);
  --shadow-md: 0 2px 4px rgba(17,17,16,.05), 0 8px 20px -8px rgba(17,17,16,.10);
}

/* ---------- base ---------- */
/* Deliberately narrow. A blanket `[class*="st-"]` also hits Streamlit's own
   Material Symbols spans -- they carry an emotion class and no stable testid --
   which replaces every chevron and check with the literal ligature name. Form
   controls are named explicitly because they do not inherit the family. */
html, body, .stApp, button, input, select, textarea,
[data-testid="stMarkdownContainer"] { font-family: var(--font); }

.stApp { background: var(--surface); color: var(--ink); }

body { -webkit-font-smoothing: antialiased; text-rendering: optimizeLegibility; }

/* Browser surfaces are part of the design, not the browser's business. */
::selection { background: #cfe0f4; color: #0b2b52; }
* { scrollbar-width: thin; scrollbar-color: var(--line-strong) transparent; }
::-webkit-scrollbar { width: 11px; height: 11px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb {
  background: var(--line-strong); border-radius: 6px;
  border: 3px solid var(--surface);
}
::-webkit-scrollbar-thumb:hover { background: #b3b1a6; }
:focus-visible { outline: 2px solid var(--primary); outline-offset: 2px; border-radius: 3px; }
input { caret-color: var(--primary); }
a, a:visited { color: var(--primary); text-decoration-thickness: 1px;
               text-underline-offset: 2px; }

/* Streamlit chrome we did not draw */
[data-testid="stDecoration"] { display: none; }
header[data-testid="stHeader"] { background: transparent; height: 0; }
[data-testid="stToolbar"] { right: 8px; top: 6px; }
[data-testid="stMainMenu"] { color: var(--ink-3); }
footer, [data-testid="stStatusWidget"] { visibility: hidden; }

/* ---------- layout ---------- */
[data-testid="stMainBlockContainer"], .block-container {
  max-width: 1128px; padding: 2.1rem 2.4rem 5rem;
}
[data-testid="stVerticalBlock"] { gap: 0; }
[data-testid="stElementContainer"] { margin: 0; }
[data-testid="stHorizontalBlock"] { gap: 1.1rem; }
hr, [data-testid="stDivider"] hr { border-color: var(--line); margin: 1.9rem 0; }

/* ---------- type ---------- */
.stApp p, .stApp li { color: var(--ink-2); font-size: .935rem; line-height: 1.62; }
.stApp h1, .stApp h2, .stApp h3, .stApp h4 {
  color: var(--ink); letter-spacing: -.017em; font-weight: 600; padding: 0;
}
.num { font-variant-numeric: tabular-nums; font-feature-settings: 'tnum' 1; }

/* Page header ------------------------------------------------------------ */
.page-head { margin: .2rem 0 1.7rem; }
.page-head h1 {
  font-size: 2.02rem; line-height: 1.14; font-weight: 700;
  letter-spacing: -.028em; margin: 0 0 .5rem;
}
.page-head .sub {
  color: var(--ink-2); font-size: 1.008rem; line-height: 1.55;
  max-width: 63ch; margin: 0;
}
.badge-row { display: flex; flex-wrap: wrap; gap: .42rem; margin-top: 1.05rem; }
.badge {
  display: inline-flex; align-items: center; gap: .34rem;
  padding: .27rem .58rem .27rem .5rem; border: 1px solid var(--line);
  background: var(--panel); border-radius: 999px;
  font-size: .757rem; font-weight: 500; color: var(--ink-2);
  letter-spacing: .004em; white-space: nowrap;
}
.badge .ic { color: var(--primary); flex: none; }
.badge.tech { border-radius: var(--r-sm); font-weight: 500; }
.badge.tech .v { color: var(--ink-3); font-variant-numeric: tabular-nums; }

/* Section headings ------------------------------------------------------- */
.sec { margin: 2.5rem 0 1rem; }
.sec:first-child { margin-top: 0; }
.sec h2 {
  font-size: 1.235rem; margin: 0 0 .3rem; display: flex;
  align-items: center; gap: .5rem; letter-spacing: -.02em;
}
.sec h2 .ic { color: var(--primary); flex: none; }
.sec .note { color: var(--ink-3); font-size: .86rem; margin: 0; max-width: 74ch; }

/* Prose ------------------------------------------------------------------ */
.prose { max-width: 72ch; }
.prose p { margin: 0 0 .78rem; }
.prose p:last-child { margin-bottom: 0; }
.prose strong { color: var(--ink); font-weight: 600; }
.prose ul { margin: .2rem 0 .8rem; padding-left: 1.05rem; }
.prose li { margin: .2rem 0; }
.prose li::marker { color: var(--line-strong); }

/* ---------- sidebar ---------- */
[data-testid="stSidebar"] {
  background: var(--sidebar); border-right: 1px solid var(--line); width: 268px !important;
}
[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] { padding: 1.35rem 1rem 1.1rem; }
[data-testid="stSidebarCollapseButton"] button { color: var(--ink-3); }

.brand { display: flex; align-items: center; gap: .62rem; padding: 0 .3rem 1.15rem; }
.brand svg { flex: none; border-radius: 8px; box-shadow: var(--shadow-sm); }
.brand .txt { min-width: 0; }
.brand .nm {
  font-size: .906rem; font-weight: 600; color: var(--ink);
  letter-spacing: -.014em; line-height: 1.2;
}
.brand .rl {
  font-size: .717rem; color: var(--ink-3); margin-top: .13rem;
  letter-spacing: .026em; text-transform: uppercase; font-weight: 500;
}
.nav-label {
  font-size: .692rem; text-transform: uppercase; letter-spacing: .085em;
  color: var(--ink-3); font-weight: 600; padding: 0 .3rem .48rem;
}

/* nav rows are real buttons: hover, focus, active, selected */
[data-testid="stSidebar"] .stButton { margin-bottom: 2px; }
[data-testid="stSidebar"] .stButton button {
  width: 100%; justify-content: flex-start; text-align: left;
  padding: .46rem .6rem; border-radius: var(--r-sm); border: 1px solid transparent;
  background: transparent; color: var(--ink-2); font-size: .885rem;
  font-weight: 500; letter-spacing: -.006em; box-shadow: none;
  transition: background .13s ease, color .13s ease;
}
[data-testid="stSidebar"] .stButton button p { font-size: .885rem; font-weight: 500; color: inherit; }
[data-testid="stSidebar"] .stButton button:hover { background: #edece7; color: var(--ink); }
[data-testid="stSidebar"] .stButton button:active { background: #e5e3dc; }
[data-testid="stSidebar"] .stButton button[kind="primary"],
[data-testid="stSidebar"] [data-testid="stBaseButton-primary"] {
  background: var(--primary-wash); color: var(--primary);
  border-color: var(--primary-line); font-weight: 600;
}
[data-testid="stSidebar"] .stButton button[kind="primary"] p,
[data-testid="stSidebar"] [data-testid="stBaseButton-primary"] p { font-weight: 600; }
[data-testid="stSidebar"] .stButton button[kind="primary"]:hover { background: #e4edf8; }

.side-foot { border-top: 1px solid var(--line); margin-top: 1.15rem; padding-top: .95rem; }
.status {
  display: flex; align-items: center; gap: .48rem; font-size: .8rem;
  font-weight: 500; padding: .1rem .3rem .7rem;
}
.status .dot { width: 7px; height: 7px; border-radius: 50%; flex: none; }
.status.ready { color: var(--low-text); }
.status.ready .dot { background: var(--low-mark); box-shadow: 0 0 0 3px rgba(18,130,74,.15); }
.status.down { color: var(--high-text); }
.status.down .dot { background: var(--high-mark); box-shadow: 0 0 0 3px rgba(200,51,47,.15); }
.status.wait { color: var(--ink-3); }
.status.wait .dot { background: var(--line-strong); box-shadow: 0 0 0 3px rgba(207,205,196,.28); }
.side-meta { padding: 0 .3rem; }
.side-meta div {
  font-size: .755rem; color: var(--ink-3); line-height: 1.55;
  display: flex; align-items: center; gap: .4rem;
}
.side-meta div .ic { color: var(--line-strong); flex: none; }

/* ---------- panels ---------- */
[data-testid="stVerticalBlock"][class*="st-key-panel-"] {
  border: 1px solid var(--line); border-radius: var(--r-lg);
  background: var(--panel); box-shadow: var(--shadow-sm);
  padding: 1.15rem 1.25rem 1.3rem; margin-bottom: 1rem;
}
.grp { display: flex; align-items: center; gap: .52rem; margin: 0 0 1rem; }
.grp .n {
  width: 21px; height: 21px; border-radius: var(--r-sm); flex: none;
  background: var(--primary-wash); color: var(--primary);
  font-size: .717rem; font-weight: 600; display: flex;
  align-items: center; justify-content: center; font-variant-numeric: tabular-nums;
}
.grp .t { font-size: .93rem; font-weight: 600; color: var(--ink); letter-spacing: -.012em; }
.grp .hint { font-size: .78rem; color: var(--ink-3); margin-left: auto; font-weight: 400; }

/* ---------- widgets ---------- */
/* Streamlit 1.6x renders react-aria, not BaseWeb: the selectors below match the
   real DOM (verified in the running app), and the theme block in
   .streamlit/config.toml carries border colour and radius for the rest. */
[data-testid="stWidgetLabel"] p {
  font-size: .805rem !important; font-weight: 500 !important;
  color: var(--ink-2) !important; letter-spacing: -.004em;
}
[data-testid="stWidgetLabel"] { margin-bottom: .26rem; }
[data-testid="stSelectbox"], [data-testid="stNumberInput"],
[data-testid="stSlider"], [data-testid="stButtonGroup"] { margin-bottom: .55rem; }

/* text inputs and the combobox */
.react-aria-ComboBox div[role="group"],
[data-testid="stNumberInput"] div[role="group"] {
  background: var(--panel); border: 1px solid var(--line-strong);
  border-radius: var(--r-sm); min-height: 38px;
  transition: border-color .13s ease, box-shadow .13s ease;
}
.react-aria-ComboBox div[role="group"]:hover,
[data-testid="stNumberInput"] div[role="group"]:hover { border-color: #b6b4a9; }
.react-aria-ComboBox div[role="group"]:focus-within,
[data-testid="stNumberInput"] div[role="group"]:focus-within {
  border-color: var(--primary); box-shadow: 0 0 0 3px rgba(24,79,149,.13);
}
.react-aria-ComboBox input, [data-testid="stNumberInput"] input {
  font-size: .885rem; color: var(--ink);
}
[data-testid="stNumberInput"] input { font-variant-numeric: tabular-nums; }
.react-aria-ComboBox input::placeholder { color: var(--ink-3); }

/* segmented control: our own selected state, not the default grey */
[data-testid="stButtonGroup"] button[data-variant="segmented_control"] {
  border-color: var(--line-strong); font-size: .845rem; font-weight: 500;
  color: var(--ink-2); min-height: 34px;
  transition: background .12s ease, color .12s ease, border-color .12s ease;
}
[data-testid="stButtonGroup"] button[data-variant="segmented_control"] p {
  font-size: .845rem; font-weight: 500; color: inherit;
}
[data-testid="stButtonGroup"] button[data-variant="segmented_control"]:hover {
  background: var(--sunken); color: var(--ink);
}
[data-testid="stButtonGroup"] button[data-variant="segmented_control"][aria-checked="true"] {
  background: var(--primary); border-color: var(--primary); color: #fff;
}
[data-testid="stButtonGroup"] button[data-variant="segmented_control"][aria-checked="true"] p {
  color: #fff; font-weight: 600;
}
[data-testid="stButtonGroup"] button[data-variant="segmented_control"][aria-checked="true"]:hover {
  background: var(--primary-hover); border-color: var(--primary-hover);
}
[data-testid="stButtonGroup"] [aria-disabled="true"] button,
[data-testid="stButtonGroup"] button[disabled] { opacity: .45; }
[data-testid="stButtonGroup"] [role="radiogroup"][aria-disabled="true"] { opacity: .45; }

/* toggles: whole row reads as one setting */
[data-testid="stCheckbox"] { margin-bottom: .05rem; }
[data-testid="stCheckbox"] > label {
  padding: .34rem .3rem; border-radius: var(--r-sm); width: 100%;
  transition: background .12s ease;
}
[data-testid="stCheckbox"] > label:hover { background: var(--sunken); }
[data-testid="stCheckbox"] p {
  font-size: .858rem !important; color: var(--ink-2) !important; font-weight: 500 !important;
}
[data-testid="stCheckbox"] > label:has(input:disabled) { opacity: .45; }
[data-testid="stCheckbox"] > label:has(input:disabled):hover { background: transparent; }

[data-testid="stSliderTickBar"] p {
  font-size: .73rem; color: var(--ink-3); font-variant-numeric: tabular-nums;
}
[data-testid="stSliderThumbValue"] {
  color: var(--primary); font-size: .8rem; font-weight: 600;
  font-variant-numeric: tabular-nums;
}

/* primary CTA */
[data-testid="stMain"] .stButton button[kind="primary"],
[data-testid="stMain"] [data-testid="stBaseButton-primary"] {
  background: var(--primary); border: 1px solid var(--primary); color: #fff;
  border-radius: var(--r-sm); padding: .62rem 1.15rem; font-weight: 600;
  font-size: .935rem; letter-spacing: -.006em; box-shadow: var(--shadow-sm);
  transition: background .14s ease, box-shadow .14s ease, transform .08s ease;
}
[data-testid="stMain"] .stButton button[kind="primary"]:hover {
  background: var(--primary-hover); border-color: var(--primary-hover);
  box-shadow: var(--shadow-md);
}
[data-testid="stMain"] .stButton button[kind="primary"]:active { transform: translateY(1px); }
[data-testid="stMain"] .stButton button[kind="secondary"] {
  border-color: var(--line-strong); color: var(--ink-2); background: var(--panel);
  border-radius: var(--r-sm); font-weight: 500; font-size: .87rem;
}
[data-testid="stMain"] .stButton button[kind="secondary"]:hover {
  border-color: #b6b4a9; color: var(--ink); background: var(--sunken);
}
.cta-note { font-size: .8rem; color: var(--ink-3); margin: .5rem 0 0; }

/* ---------- empty / error / callout ---------- */
.empty {
  border: 1px dashed var(--line-strong); border-radius: var(--r-lg);
  background: var(--panel); padding: 2.9rem 2rem; text-align: center;
}
.empty .glyph {
  width: 42px; height: 42px; border-radius: 12px; margin: 0 auto .95rem;
  background: var(--primary-wash); color: var(--primary);
  display: flex; align-items: center; justify-content: center;
}
.empty h3 { font-size: 1.03rem; margin: 0 0 .4rem; font-weight: 600; }
.empty p { color: var(--ink-3); font-size: .885rem; margin: 0 auto; max-width: 44ch; }

.callout {
  display: flex; gap: .68rem; padding: .85rem 1rem; border-radius: var(--r-md);
  border: 1px solid var(--line); background: var(--sunken); align-items: flex-start;
}
.callout .ic { flex: none; margin-top: .12rem; color: var(--ink-3); }
.callout .body { font-size: .858rem; color: var(--ink-2); line-height: 1.56; }
.callout .body b { color: var(--ink); font-weight: 600; }
.callout.warn { background: var(--high-wash); border-color: #f0cfce; }
.callout.warn .ic, .callout.warn .body b { color: var(--high-text); }
.callout.info { background: var(--primary-wash); border-color: var(--primary-line); }
.callout.info .ic, .callout.info .body b { color: var(--primary); }
.disclaimer {
  font-size: .805rem; color: var(--ink-3); line-height: 1.55;
  border-left: 1px solid var(--line-strong); padding-left: .8rem; max-width: 78ch;
}

/* ---------- result card ---------- */
.result {
  border: 1px solid var(--line); border-radius: var(--r-lg); background: var(--panel);
  box-shadow: var(--shadow-md); overflow: hidden;
}
.result .top {
  display: flex; flex-wrap: wrap; align-items: flex-end; gap: 1.5rem 2.4rem;
  padding: 1.45rem 1.55rem 1.3rem;
}
.result .prob { min-width: 0; }
.result .prob .k {
  font-size: .73rem; text-transform: uppercase; letter-spacing: .085em;
  color: var(--ink-3); font-weight: 600; margin-bottom: .18rem;
}
.result .prob .v {
  font-size: 3.32rem; line-height: 1; font-weight: 700; letter-spacing: -.038em;
  font-variant-numeric: tabular-nums; color: var(--ink);
}
.result .prob .v .pc { font-size: 1.6rem; font-weight: 600; margin-left: .06em;
                       letter-spacing: -.02em; color: var(--ink-2); }
.result .side { display: flex; flex-direction: column; gap: .62rem; }
.chip {
  display: inline-flex; align-items: center; gap: .42rem; align-self: flex-start;
  padding: .34rem .68rem .34rem .56rem; border-radius: 999px;
  font-size: .858rem; font-weight: 600; letter-spacing: -.004em; border: 1px solid;
}
.chip .ic { flex: none; }
.verdict { font-size: .885rem; color: var(--ink-2); display: flex;
           align-items: center; gap: .42rem; font-weight: 500; }
.verdict .ic { flex: none; color: var(--ink-3); }

/* risk meter: zones are labelled in text, marker carries the number */
.meter { padding: 0 1.55rem 1.5rem; }
.meter .track {
  position: relative; height: 10px; border-radius: 5px; overflow: hidden;
  display: flex; background: var(--sunken); border: 1px solid var(--line);
}
.meter .zone { height: 100%; }
.meter .zone + .zone { border-left: 2px solid var(--panel); }
.meter .marker { position: relative; height: 26px; margin-top: -1px; }
.meter .needle {
  position: absolute; top: 0; width: 2px; height: 15px; border-radius: 1px;
  transform: translateX(-1px);
}
.meter .valchip {
  position: absolute; top: 14px; transform: translateX(-50%);
  font-size: .73rem; font-weight: 700; padding: .1rem .34rem; border-radius: 4px;
  font-variant-numeric: tabular-nums; white-space: nowrap; color: #fff;
}
.meter .scale {
  display: flex; margin-top: .3rem; font-size: .73rem; color: var(--ink-3);
  font-weight: 500;
}
.meter .scale span { display: flex; flex-direction: column; gap: .07rem; }
.meter .scale span em {
  font-style: normal; font-variant-numeric: tabular-nums; color: var(--ink-3);
  font-size: .705rem; opacity: .85;
}
.meter .scale span.on { color: var(--ink); font-weight: 600; }
.meter .foot { font-size: .78rem; color: var(--ink-3); margin: .85rem 0 0; line-height: 1.5; }

/* ---------- SHAP contributions ---------- */
.bridge {
  display: flex; align-items: stretch; gap: 0; border: 1px solid var(--line);
  border-radius: var(--r-md); background: var(--panel); overflow: hidden;
  margin-bottom: 1.15rem;
}
.bridge .cell { padding: .78rem 1rem; flex: 1 1 0; min-width: 0; }
.bridge .cell + .cell { border-left: 1px solid var(--line); }
.bridge .k {
  font-size: .705rem; text-transform: uppercase; letter-spacing: .075em;
  color: var(--ink-3); font-weight: 600; margin-bottom: .22rem;
  display: flex; align-items: center; gap: .3rem;
}
.bridge .v {
  font-size: 1.16rem; font-weight: 700; font-variant-numeric: tabular-nums;
  letter-spacing: -.02em; color: var(--ink);
}
.bridge .cell.net { background: var(--sunken); }
.bridge .sub { font-size: .73rem; color: var(--ink-3); margin-top: .12rem; }
.prose p.bridge-note {
  font-size: .81rem; color: var(--ink-3); max-width: 68ch;
  margin: -.55rem 0 1.25rem;
}

.contrib-head {
  display: flex; align-items: center; gap: .45rem; font-size: .89rem;
  font-weight: 600; color: var(--ink); margin: 0 0 .2rem;
}
.contrib-head .ic { flex: none; }
.contrib-head.up { color: var(--high-text); }
.contrib-head.down { color: var(--primary); }
.contrib-sub { font-size: .78rem; color: var(--ink-3); margin: 0 0 .7rem; }

.crow { padding: .46rem 0; border-top: 1px solid var(--line); }
.crow:first-of-type { border-top: none; }
.crow .l { display: flex; align-items: baseline; gap: .5rem; justify-content: space-between; }
.crow .nm { font-size: .858rem; color: var(--ink); font-weight: 500; min-width: 0; }
.crow .nm i { font-style: normal; color: var(--ink-3); font-weight: 400; }
.crow .sv {
  font-size: .805rem; font-weight: 600; font-variant-numeric: tabular-nums;
  flex: none; letter-spacing: -.004em;
}
.crow .bar {
  height: 5px; border-radius: 3px; margin-top: .34rem; background: var(--sunken);
  overflow: hidden;
}
.crow .bar i { display: block; height: 100%; border-radius: 3px; }
.crow.none { color: var(--ink-3); font-size: .845rem; padding: .7rem 0; border: none; }

/* ---------- metrics ---------- */
.kpi-lead {
  border: 1px solid var(--primary-line); background: var(--primary-wash);
  border-radius: var(--r-lg); padding: 1.15rem 1.3rem;
}
.kpi-lead .k {
  font-size: .73rem; text-transform: uppercase; letter-spacing: .08em;
  color: var(--primary); font-weight: 600; display: flex; align-items: center;
  gap: .34rem; margin-bottom: .3rem;
}
.kpi-lead .v {
  font-size: 2.62rem; line-height: 1; font-weight: 700; letter-spacing: -.036em;
  font-variant-numeric: tabular-nums; color: #0e3466;
}
.kpi-lead .d { font-size: .8rem; color: #2b5a94; margin-top: .5rem; line-height: 1.5; }

.kpi-row { display: flex; flex-direction: column; gap: .55rem; }
.kpi {
  border: 1px solid var(--line); border-radius: var(--r-md); background: var(--panel);
  padding: .7rem .9rem; display: flex; align-items: baseline;
  justify-content: space-between; gap: .8rem;
}
.kpi .k { font-size: .83rem; color: var(--ink-2); font-weight: 500; }
.kpi .k em { font-style: normal; display: block; font-size: .73rem; color: var(--ink-3);
             margin-top: .08rem; font-weight: 400; }
.kpi .v {
  font-size: 1.31rem; font-weight: 700; font-variant-numeric: tabular-nums;
  letter-spacing: -.022em; color: var(--ink); flex: none;
}

/* metrics table -------------------------------------------------------- */
.tbl-wrap { border: 1px solid var(--line); border-radius: var(--r-md);
            overflow: hidden; background: var(--panel); }
table.tbl { width: 100%; border-collapse: collapse; font-size: .858rem; }
table.tbl th, table.tbl td { padding: .58rem .8rem; text-align: right; }
table.tbl th:first-child, table.tbl td:first-child { text-align: left; }
table.tbl thead th {
  background: var(--sunken); font-size: .73rem; font-weight: 600; color: var(--ink-2);
  text-transform: uppercase; letter-spacing: .05em; border-bottom: 1px solid var(--line);
  white-space: nowrap;
}
table.tbl td { border-top: 1px solid var(--line); color: var(--ink-2);
               font-variant-numeric: tabular-nums; }
table.tbl td:first-child { color: var(--ink); font-weight: 500;
                           font-variant-numeric: normal; }
table.tbl tbody tr:hover td { background: #fbfaf8; }
table.tbl tr.sel td { background: var(--primary-wash); }
table.tbl tr.sel:hover td { background: #e7eff8; }
table.tbl tr.sel td:first-child { color: #0e3466; font-weight: 600; }
table.tbl td.best { color: var(--ink); font-weight: 700; }
table.tbl td.best::after {
  content: 'best'; display: inline-block; margin-left: .38rem; font-size: .63rem;
  font-weight: 600; text-transform: uppercase; letter-spacing: .05em;
  color: var(--primary); vertical-align: .08em;
}
.tbl-note { font-size: .78rem; color: var(--ink-3); margin: .55rem 0 0; }

/* pipeline rail --------------------------------------------------------- */
.rail { display: flex; flex-direction: column; gap: 0; }
.rail .ph {
  font-size: .705rem; text-transform: uppercase; letter-spacing: .085em;
  color: var(--ink-3); font-weight: 600; margin: 1.05rem 0 .5rem;
}
.rail .ph:first-child { margin-top: 0; }
.step { display: flex; gap: .82rem; position: relative; padding-bottom: .1rem; }
.step .gut { display: flex; flex-direction: column; align-items: center; flex: none; width: 24px; }
.step .n {
  width: 24px; height: 24px; border-radius: 7px; flex: none; background: var(--panel);
  border: 1px solid var(--line-strong); color: var(--ink-2); font-size: .717rem;
  font-weight: 600; display: flex; align-items: center; justify-content: center;
  font-variant-numeric: tabular-nums; z-index: 1;
}
.step .ln { width: 1px; flex: 1 1 auto; background: var(--line); min-height: 12px; }
.step:last-child .ln { background: transparent; }
.step .bd { padding-bottom: .82rem; min-width: 0; }
.step .t { font-size: .89rem; font-weight: 600; color: var(--ink); letter-spacing: -.01em; }
.step .d { font-size: .805rem; color: var(--ink-3); margin-top: .1rem; line-height: 1.5; }
.step.key .n { background: var(--primary); border-color: var(--primary); color: #fff; }
.step.key .t { color: var(--primary); }

/* figure frame ---------------------------------------------------------- */
[data-testid="stVerticalBlock"][class*="st-key-fig-"] {
  border: 1px solid var(--line); border-radius: var(--r-md); background: var(--panel);
  padding: .85rem; margin-bottom: 1rem;
}
[data-testid="stImage"] img { border-radius: var(--r-sm); display: block; }
.fig-cap { font-size: .78rem; color: var(--ink-3); margin: .6rem .2rem 0; line-height: 1.5; }

[data-testid="stExpander"] details {
  border: 1px solid var(--line); border-radius: var(--r-md); background: var(--panel);
}
[data-testid="stExpander"] summary { font-size: .858rem; font-weight: 500; color: var(--ink-2); }
[data-testid="stExpander"] summary:hover { color: var(--primary); }

/* Motion: one authored moment -- results arriving. Nothing else animates. */
@keyframes rise { from { opacity: 0; transform: translateY(7px); } to { opacity: 1; transform: none; } }
.arrive { animation: rise .34s cubic-bezier(.16,.84,.44,1) both; }
@media (prefers-reduced-motion: reduce) {
  .arrive { animation: none; }
  * { transition-duration: .01ms !important; }
}

/* ---------- responsive: structural, not fluid type ---------- */
@media (max-width: 1000px) {
  [data-testid="stMainBlockContainer"], .block-container { padding: 1.5rem 1.1rem 3.5rem; }
  .page-head h1 { font-size: 1.72rem; }
  .result .prob .v { font-size: 2.72rem; }
  .bridge { flex-direction: column; }
  .bridge .cell + .cell { border-left: none; border-top: 1px solid var(--line); }
}
@media (max-width: 640px) {
  .result .top { gap: 1rem 1.2rem; padding: 1.15rem 1.15rem 1rem; }
  .meter { padding: 0 1.15rem 1.2rem; }
  .meter .scale { font-size: .69rem; }
  table.tbl th, table.tbl td { padding: .48rem .55rem; font-size: .8rem; }
  table.tbl td.best::after { content: '*'; margin-left: .2rem; }
}
"""


def inject() -> None:
    """Attach the stylesheet once per script run."""
    st.markdown(f"<style>{_CSS}</style>", unsafe_allow_html=True)
