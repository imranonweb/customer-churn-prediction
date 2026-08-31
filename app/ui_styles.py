"""Stage 6A design system: tokens, authored icons, and the single stylesheet.

What changed in 6A, and why
---------------------------
Stage 6 shipped a working interface whose *visual* layer had three structural
faults. They are named here because every rule below is a consequence of one of
them:

1. **Streamlit's negative markdown margin was leaking.** Streamlit hangs
   `margin-bottom: -1rem` on every `stMarkdownContainer` to cancel the 1rem
   bottom margin of a trailing `<p>` it assumes is there. Authored HTML has no
   trailing `<p>`, so the negative margin had nothing to cancel and every block
   spilled 16px past its own box -- absorbed silently where the next block had a
   large top margin, and a visible collision where it did not. Stage 6 patched
   this in the sidebar only, then hand-tuned each main-column margin around the
   leak. So the spacing was not a system; it was compensation. Fixed once, at the
   root, by the rule pair under "base".

2. **No scales.** 33 distinct font sizes, with card titles (.93rem) and body
   text (.935rem) landing on the same step -- the one place hierarchy is most
   needed. Dozens of unrelated padding values. Everything below now comes from
   `--t-*` (9 type steps), `--s*` (the 4/8/12/16/24/32/48/64 spacing scale),
   `--r-*` (5 radii) and `--sh-*` (4 elevations). No magic numbers.

3. **No depth.** Three near-identical warm off-whites and a 5%-alpha shadow read
   as Streamlit's stock light theme. The neutral ramp is now cool and properly
   stepped, the sidebar is a deep navy so the shell reads as chrome rather than
   as more page, and elevation is carried by real layered shadow.

Why the palette is what it is
-----------------------------
The app embeds Stage 4 and Stage 5 PNGs directly, so it must not fight them.
`src/evaluate.py` fixes the figure surface at `#fcfcfb` and a single-hue blue
ramp whose dark steps include `#184f95`. That step stays the interactive primary,
and `--paper` keeps the figure frames at the figures' own surface colour -- so a
chart sits on its own paper while the page around it is cool. Deepening the
neutrals cool and the chrome to navy therefore costs the figures nothing.

Contrast, measured against the surface each colour is used on (WCAG AA needs
4.5:1 for text, 3:1 for a mark or a border that carries meaning):

    token          hex        contrast   role
    ink            #0f1723    17.9:1     body
    ink-2          #47536b     7.6:1     secondary
    ink-3          #6b7789     4.9:1     captions
    primary        #184f95     7.9:1     actions, links, selection
    primary-hover  #123c73    10.7:1
    low-text       #0e6b3d     6.4:1     "Low" label
    med-text       #8f5f00     5.4:1     "Medium" label
    high-text      #a32b29     7.0:1     "High" label
    low-mark       #12824a     4.7:1     meter fill / dot   (needs >= 3:1)
    med-mark       #bd7f12     3.3:1
    high-mark      #c8332f     5.2:1
    sb-ink         #e8edf5    14.6:1     sidebar body on #0d1826
    sb-ink-2       #97a6bd     6.9:1     sidebar secondary
    sb-active      #7fb2f0     8.2:1     active nav item

The three risk marks separate at worst dE 16.4 for normal vision (floor 15).
Under simulated deuteranopia the green/red pair falls to dE 3.3 -- the known,
unavoidable cost of traffic-light semantics. That is legal only with a second,
non-colour encoding, so risk is *never* carried by colour here: every risk
readout ships the band name in words, the probability as a number, a drawn icon,
and a marker position on a labelled 0-100 scale. Remove all colour and the page
still reads.

SHAP direction reuses Stage 5's own pair exactly -- `#c8332f` family for "pushed
toward churn", `#184f95` family for "pushed away" -- so the live explanation and
the saved beeswarm agree on which way red means.

Why so little animates
----------------------
Streamlit re-runs the whole script on every widget change, so a page-load
animation replays every time a segmented control is touched. Entrance motion is
therefore reserved for content that appears only in response to an explicit
action -- the prediction result -- and for pages reached by navigation, which is
also explicit. Continuous form input gets state transitions (hover, focus,
selection) and nothing else. That is the difference between motion that reports
something and motion for its own sake.
"""

from __future__ import annotations

import streamlit as st

# --------------------------------------------------------------------------
# Tokens needed on the Python side (inline styles for data-driven positions).
# Everything else lives in the CSS custom properties below, single source.
# --------------------------------------------------------------------------
RISK_TOKENS = {
    "Low": {"mark": "#12824a", "text": "#0e6b3d", "wash": "#e7f4ed", "line": "#b9e0c9"},
    "Medium": {"mark": "#bd7f12", "text": "#8f5f00", "wash": "#fdf2dd", "line": "#f0d9a6"},
    "High": {"mark": "#c8332f", "text": "#a32b29", "wash": "#fdecea", "line": "#f5cdc9"},
}

PUSH_TOWARD = "#c8332f"   # SHAP > 0, matches explain.PUSH_TOWARD's role
PUSH_AWAY = "#184f95"     # SHAP < 0, matches explain.PUSH_AWAY's role
INK_3 = "#6b7789"


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
    # 6A additions
    "book": '<path d="M2.4 3.4h4.2A1.4 1.4 0 0 1 8 4.8v8.4a1.2 1.2 0 0 0-1.2-1.1H2.4z"/>'
            '<path d="M13.6 3.4H9.4A1.4 1.4 0 0 0 8 4.8v8.4a1.2 1.2 0 0 1 1.2-1.1h4.4z"/>',
    "eye": '<path d="M1.6 8S3.9 4.1 8 4.1 14.4 8 14.4 8 12.1 11.9 8 11.9 1.6 8 1.6 8z"/>'
           '<circle cx="8" cy="8" r="1.9"/>',
    "people": '<circle cx="6.2" cy="5.9" r="2.4"/>'
              '<path d="M1.9 13.4a4.4 4.4 0 0 1 8.6 0"/>'
              '<path d="M10.8 3.9a2.4 2.4 0 0 1 0 4"/>'
              '<path d="M11.9 9.4a4.4 4.4 0 0 1 2.2 3.3"/>',
    "database": '<ellipse cx="8" cy="4" rx="5.2" ry="1.8"/>'
                '<path d="M2.8 4v8c0 1 2.3 1.8 5.2 1.8s5.2-.8 5.2-1.8V4"/>'
                '<path d="M2.8 8c0 1 2.3 1.8 5.2 1.8s5.2-.8 5.2-1.8"/>',
    "scale": '<path d="M8 2.6v10.8"/><path d="M4.2 13.4h7.6"/>'
             '<path d="M2 6.2h12"/><path d="M2 6.2 3.8 9.6h-3.6z" transform="translate(1.8)"/>',
    "reset": '<path d="M13.2 8a5.2 5.2 0 1 1-1.7-3.8"/><path d="M13.4 2.6v3.2h-3.2"/>',
    "shield": '<path d="M8 2.2 13.2 4v4c0 3-2.4 5-5.2 5.8C5.2 13 2.8 11 2.8 8V4z"/>'
              '<path d="M5.9 8 7.5 9.6 10.4 6.6"/>',
}

BRAND_MARK = (
    '<svg width="34" height="34" viewBox="0 0 34 34" fill="none" aria-hidden="true">'
    '<rect width="34" height="34" rx="9" fill="url(#bm)"/>'
    '<defs><linearGradient id="bm" x1="0" y1="0" x2="34" y2="34">'
    '<stop stop-color="#2f6fd0"/><stop offset="1" stop-color="#184f95"/>'
    "</linearGradient></defs>"
    '<path d="M8 23.2 12.6 16.4 16.4 19.6 22 10.8" stroke="#ffffff" '
    'stroke-width="2.1" stroke-linecap="round" stroke-linejoin="round"/>'
    '<circle cx="22" cy="10.8" r="2.8" fill="#ffffff"/>'
    '<path d="M8 26.6h18" stroke="#8fb8ea" stroke-width="1.6" stroke-linecap="round" '
    'opacity=".55"/>'
    "</svg>"
)


# --------------------------------------------------------------------------
# Stylesheet
# --------------------------------------------------------------------------
_CSS = """
/* Archivo is loaded by the `font` key in .streamlit/config.toml (Streamlit's own
   Google Fonts syntax), not by an @import here -- one request, one source. */
:root, .stApp {
  --font: 'Archivo', 'Segoe UI Variable Text', 'Segoe UI', -apple-system,
          BlinkMacSystemFont, Roboto, 'Helvetica Neue', Arial, sans-serif;

  /* surfaces: cool neutral ramp. Elevation is shadow, not tint, so `surface`
     and `raised` share a colour and differ only in --sh-*. */
  --bg: #f5f7fa;
  --surface: #ffffff;
  --raised: #ffffff;
  --sunken: #eef1f6;
  --paper: #fcfcfb;          /* the Stage 4/5 figures' own surface colour */
  --line: #e3e7ee;
  --line-2: #cfd6e1;
  --line-3: #b3bccb;

  --ink: #0f1723;
  --ink-2: #47536b;
  /* Caption/metadata ink. Chosen by measurement, not by eye: this is the lightest
     step that still clears WCAG AA 4.5:1 on all three surfaces this app paints on
     -- page #f5f7fa (4.81), surface white (5.16), sunken #eef1f6 (4.55). The
     obvious #6b7789 fails on two of them (4.23 and 4.01). */
  --ink-3: #636e80;
  --ink-inv: #ffffff;

  /* interactive blue: --primary is the figures' own dark step, kept so a chart
     dropped on a panel agrees with the buttons around it. */
  --primary: #184f95;
  --primary-hover: #123c73;
  --primary-deep: #0e3466;
  --primary-lift: #2f6fd0;
  --primary-wash: #eaf1fa;
  --primary-wash-2: #dce9f7;
  --primary-line: #c3d7ef;

  /* chrome: the sidebar is a deep navy so the shell reads as chrome, not page */
  --sb-bg: #0d1826;
  --sb-bg-2: #16243a;
  --sb-bg-3: #1c2f49;
  --sb-line: #1f3149;
  --sb-ink: #e8edf5;
  --sb-ink-2: #97a6bd;
  --sb-ink-3: #7c8ba3;
  --sb-active: #7fb2f0;

  --low-mark: #12824a; --low-text: #0e6b3d; --low-wash: #e7f4ed; --low-line: #b9e0c9;
  --med-mark: #bd7f12; --med-text: #8f5f00; --med-wash: #fdf2dd; --med-line: #f0d9a6;
  --high-mark: #c8332f; --high-text: #a32b29; --high-wash: #fdecea; --high-line: #f5cdc9;

  /* type scale: 9 steps, and every size in the app is one of them */
  --t-xs: .75rem;      /* 12 - meta, table head, scale ticks */
  --t-sm: .8125rem;    /* 13 - labels, badges, captions */
  --t-md: .875rem;     /* 14 - UI text, table body, secondary */
  --t-base: .9375rem;  /* 15 - body prose */
  --t-lg: 1.0625rem;   /* 17 - card title, lead paragraph */
  --t-xl: 1.3125rem;   /* 21 - section heading */
  --t-2xl: 1.75rem;    /* 28 - page title */
  --t-3xl: 2.25rem;    /* 36 - hero title */
  --t-num: 3.5rem;     /* 56 - the single result number */

  /* spacing scale: 4 / 8 / 12 / 16 / 24 / 32 / 48 / 64 */
  --s1: 4px; --s2: 8px; --s3: 12px; --s4: 16px;
  --s5: 24px; --s6: 32px; --s7: 48px; --s8: 64px;

  --r-xs: 4px; --r-sm: 6px; --r-md: 10px; --r-lg: 14px; --r-xl: 18px;
  --r-full: 999px;

  --sh-1: 0 1px 2px rgba(12,22,38,.06), 0 1px 3px rgba(12,22,38,.05);
  --sh-2: 0 1px 2px rgba(12,22,38,.05), 0 6px 16px -4px rgba(12,22,38,.10);
  --sh-3: 0 2px 6px rgba(12,22,38,.05), 0 16px 36px -12px rgba(12,22,38,.16);
  --sh-cta: 0 1px 2px rgba(12,22,38,.12), 0 8px 18px -6px rgba(24,79,149,.42);
  --sh-cta-h: 0 2px 4px rgba(12,22,38,.14), 0 14px 30px -8px rgba(24,79,149,.50);
  --ring: 0 0 0 3px rgba(24,79,149,.16);

  --e-out: cubic-bezier(.22,.72,.28,1);
  --d1: 140ms; --d2: 200ms; --d3: 280ms;

  --maxw: 1160px;

  /* Figure height cap. The saved PNGs range from 1.06:1 (ROC, PR) to 3.37:1
     (monthly charges), and stretching all of them to the column width meant the
     near-square ones rendered ~620px tall and ate a full screen each. Capping
     height instead of width lets the browser derive the width from the aspect
     ratio, so wide figures are untouched and only the tall ones come down. */
  --fig-max-h: 460px;

  /* Narrowest a card may get before it stops being readable. Card grids ask for a
     column count, but the count is only honourable if the tracks it implies are
     wide enough: measured at 768px, the four supporting metrics sat two-up inside
     a 248px Streamlit column at 116px each, and the two team cards at 194px wrapped
     "Lead Developer and System Architect" onto three lines beside a 42px avatar.
     Below roughly this width a card's description breaks into ribbons of two or
     three words, so the grid drops a column instead. */
  --card-min: 208px;
}

/* ---------- base ---------- */
/* Deliberately narrow. A blanket `[class*="st-"]` also hits Streamlit's own
   Material Symbols spans -- they carry an emotion class and no stable testid --
   which replaces every chevron and check with the literal ligature name. Form
   controls are named explicitly because they do not inherit the family. */
html, body, .stApp, button, input, select, textarea,
[data-testid="stMarkdownContainer"] { font-family: var(--font); }

.stApp { background: var(--bg); color: var(--ink); }
body { -webkit-font-smoothing: antialiased; text-rendering: optimizeLegibility; }

/* THE spacing fix. Streamlit hangs `margin-bottom: -1rem` on every markdown
   container to cancel the 1rem bottom margin of a trailing <p> it assumes is
   there; authored HTML has no trailing <p>, so the negative margin had nothing
   to cancel and every block spilled 16px past its own box. Removing both halves
   of that pair is a no-op for real markdown and stops the leak for authored
   HTML -- so all vertical rhythm below comes from component margins on the
   spacing scale, and none of it is compensation. */
[data-testid="stMarkdownContainer"] { margin-bottom: 0; }
[data-testid="stMarkdownContainer"] p:last-child { margin-bottom: 0; }

/* Browser surfaces are part of the design, not the browser's business. */
::selection { background: #cfe0f4; color: #0b2b52; }
* { scrollbar-width: thin; scrollbar-color: var(--line-2) transparent; }
::-webkit-scrollbar { width: 11px; height: 11px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb {
  background: var(--line-2); border-radius: var(--r-sm);
  border: 3px solid var(--bg);
}
::-webkit-scrollbar-thumb:hover { background: var(--line-3); }
[data-testid="stSidebar"] ::-webkit-scrollbar-thumb {
  background: var(--sb-bg-3); border-color: var(--sb-bg);
}
:focus-visible { outline: 2px solid var(--primary); outline-offset: 2px; border-radius: 3px; }
[data-testid="stSidebar"] :focus-visible { outline-color: var(--sb-active); }
input { caret-color: var(--primary); }
a, a:visited { color: var(--primary); text-decoration-thickness: 1px;
               text-underline-offset: 2px; }

/* Streamlit chrome we did not draw */
[data-testid="stDecoration"] { display: none; }
/* Under toolbarMode "minimal" this header renders empty, but it is still an
   absolutely-positioned 60px strip at z-index 999990, so scrolled content passed
   underneath it. `height` alone is ignored -- the 60px comes from a min-height --
   which is why both are set. Collapsing it unconditionally is not safe: when the
   sidebar is closed Streamlit moves the "expand sidebar" button into this header,
   and a zero-height header would put the only route back to the navigation out of
   reach. So it collapses only while it is genuinely empty, and becomes a real
   opaque bar when it carries that button. */
header[data-testid="stHeader"] { background: transparent; }
header[data-testid="stHeader"]:not(:has(button)) { height: 0; min-height: 0; }
.stApp:has([data-testid="stExpandSidebarButton"]) header[data-testid="stHeader"] {
  background: rgba(245,247,250,.88); backdrop-filter: blur(8px);
  border-bottom: 1px solid var(--line);
}
.stApp:has([data-testid="stExpandSidebarButton"]) [data-testid="stMainBlockContainer"] {
  padding-top: var(--s8);
}
[data-testid="stExpandSidebarButton"] button {
  background: var(--surface); border: 1px solid var(--line-2);
  border-radius: var(--r-sm); box-shadow: var(--sh-1); color: var(--ink-2);
}
[data-testid="stToolbar"] { right: var(--s2); top: var(--s1); }
[data-testid="stMainMenu"] { color: var(--ink-3); }
footer, [data-testid="stStatusWidget"] { visibility: hidden; }

/* ---------- layout ---------- */
[data-testid="stMainBlockContainer"], .block-container {
  max-width: var(--maxw); padding: var(--s6) var(--s6) var(--s8);
}
[data-testid="stVerticalBlock"] { gap: 0; }
[data-testid="stElementContainer"] { margin: 0; }
[data-testid="stHorizontalBlock"] { gap: var(--s5); }
hr, [data-testid="stDivider"] hr { border-color: var(--line); margin: var(--s6) 0; }

/* one spacer utility on the scale, instead of inline pixel heights */
.sp-3 { height: var(--s3); } .sp-4 { height: var(--s4); }
.sp-5 { height: var(--s5); } .sp-6 { height: var(--s6); }

/* Equal-height columns, opt-in. Streamlit columns size to their content, so two
   cards side by side end at different heights and the pair stops reading as one
   comparison. Scoped to containers that ask for it via key= rather than applied
   to every column, because most columns hold widgets that must not stretch. */
[class*="st-key-eqrow"] [data-testid="stColumn"] { display: flex; flex-direction: column; }
[class*="st-key-eqrow"] [data-testid="stColumn"] > [data-testid="stVerticalBlock"],
[class*="st-key-eqrow"] [data-testid="stElementContainer"],
[class*="st-key-eqrow"] [data-testid="stMarkdownContainer"] { flex: 1 1 auto; height: 100%; }

/* ---------- type ---------- */
/* Base paragraph. NOTE for anything added below: this selector is a class plus a
   type, so a component rule written as a bare `.my-note` on a <p> LOSES to it and
   silently keeps body size and colour. Paragraph components below are therefore
   written `.stApp p.my-note`, not `.my-note`. */
.stApp p, .stApp li { color: var(--ink-2); font-size: var(--t-base); line-height: 1.65; }
.stApp h1, .stApp h2, .stApp h3, .stApp h4 {
  color: var(--ink); letter-spacing: -.018em; font-weight: 600; padding: 0;
}
.num { font-variant-numeric: tabular-nums; font-feature-settings: 'tnum' 1; }

/* Page header ------------------------------------------------------------ */
.page-head { margin: 0 0 var(--s6); }
.page-head .eyebrow {
  display: flex; align-items: center; gap: var(--s2); font-size: var(--t-xs);
  font-weight: 600; letter-spacing: .09em; text-transform: uppercase;
  color: var(--primary); margin-bottom: var(--s3);
}
.page-head .eyebrow .ic { flex: none; }
.page-head .eyebrow::after {
  content: ''; flex: 1 1 auto; height: 1px; max-width: 76px;
  background: linear-gradient(to right, var(--primary-line), transparent);
}
.page-head h1 {
  font-size: var(--t-2xl); line-height: 1.18; font-weight: 700;
  letter-spacing: -.028em; margin: 0 0 var(--s3);
}
.page-head.hero h1 { font-size: var(--t-3xl); line-height: 1.12; }
.page-head .sub {
  color: var(--ink-2); font-size: var(--t-lg); line-height: 1.6;
  max-width: 66ch; margin: 0;
}
.badge-row { display: flex; flex-wrap: wrap; gap: var(--s2); margin-top: var(--s5); }
.badge {
  display: inline-flex; align-items: center; gap: var(--s1);
  padding: var(--s1) 10px var(--s1) var(--s2); border: 1px solid var(--line);
  background: var(--surface); border-radius: var(--r-full);
  font-size: var(--t-sm); font-weight: 500; color: var(--ink-2);
  white-space: nowrap; box-shadow: var(--sh-1);
  transition: border-color var(--d1) var(--e-out), box-shadow var(--d1) var(--e-out);
}
.badge:hover { border-color: var(--primary-line); box-shadow: var(--sh-2); }
.badge .ic { color: var(--primary); flex: none; }
.badge.tech { border-radius: var(--r-sm); }
.badge.tech .v { color: var(--ink-3); font-variant-numeric: tabular-nums; }

/* Section headings ------------------------------------------------------- */
/* `.sec:first-child` was the spacing bug the whole app was suffering from, and it
   is worth stating why so it is not reintroduced. Streamlit wraps EVERY markdown
   call in its own container, so a section heading is always the only child of its
   own parent -- which made `:first-child` match all six of them and zero the 48px
   break above each one. Measured: every heading on the About page sat 0px below
   the block before it. A position test has to be made at the element-container
   level, because that is where the siblings actually are. */
.sec { margin: var(--s7) 0 var(--s4); }
[data-testid="stElementContainer"]:first-child .sec { margin-top: 0; }
/* A page head already ends with 32px of its own, so the first section under one
   takes the shorter step instead of stacking 48 on top of it. */
[data-testid="stElementContainer"]:has(.page-head) + [data-testid="stElementContainer"] .sec {
  margin-top: var(--s5);
}
.sec h2 {
  font-size: var(--t-xl); margin: 0 0 var(--s2); display: flex;
  align-items: center; gap: var(--s2); letter-spacing: -.022em;
}
.sec h2 .ic { color: var(--primary); flex: none; }
.sec .note { color: var(--ink-3); font-size: var(--t-md); margin: 0; max-width: 78ch; }

/* Prose ------------------------------------------------------------------ */
.prose { max-width: 74ch; }
.prose p { margin: 0 0 var(--s3); }
.prose p:last-child { margin-bottom: 0; }
.prose strong, .prose b { color: var(--ink); font-weight: 600; }
.prose ul { margin: 0 0 var(--s3); padding-left: var(--s4); list-style: none; }
.prose ul:last-child { margin-bottom: 0; }
.prose li { margin: 0 0 var(--s2); position: relative; padding-left: var(--s3); }
.prose li:last-child { margin-bottom: 0; }
.prose li::before {
  content: ''; position: absolute; left: 0; top: .62em; width: 5px; height: 5px;
  border-radius: 50%; background: var(--primary-line);
}
.prose code, .callout code {
  font-size: var(--t-sm); background: var(--sunken); color: var(--ink);
  padding: 1px var(--s1); border-radius: var(--r-xs);
}

/* ---------- sidebar ---------- */
[data-testid="stSidebar"] {
  background: var(--sb-bg); border-right: 1px solid var(--sb-line);
  width: 272px !important;
}
[data-testid="stSidebar"] [data-testid="stSidebarContent"] {
  padding: var(--s5) var(--s4) var(--s4);
}
/* Streamlit reserves 60px here for a logo we do not set, which pushed the
   wordmark far below the page title. Shrunk to the collapse button's own height
   so the row still centres it. */
[data-testid="stSidebarHeader"] { height: 34px; }
[data-testid="stSidebarCollapseButton"] button { color: var(--sb-ink-2); }
[data-testid="stSidebarCollapseButton"] button:hover {
  color: var(--sb-ink); background: var(--sb-bg-2);
}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] { margin-bottom: 0; }

.brand {
  display: flex; align-items: center; gap: var(--s3);
  padding: 0 var(--s1) var(--s5);
}
.brand svg { flex: none; border-radius: 9px; box-shadow: 0 2px 8px rgba(0,0,0,.32); }
.brand .txt { min-width: 0; }
.brand .nm {
  font-size: var(--t-lg); font-weight: 700; color: var(--sb-ink);
  letter-spacing: -.02em; line-height: 1.15;
}
.brand .rl {
  font-size: var(--t-xs); color: var(--sb-ink-2); margin-top: 3px;
  letter-spacing: .01em; font-weight: 500;
}
.nav-label {
  font-size: var(--t-xs); text-transform: uppercase; letter-spacing: .1em;
  color: var(--sb-ink-3); font-weight: 600; padding: 0 var(--s2) var(--s2);
}

/* nav rows are real buttons: hover, focus, active, selected. The selected row
   carries an accent bar on its leading edge, so the current section is legible
   without relying on the tint. */
[data-testid="stSidebar"] .stButton { margin-bottom: 3px; }
[data-testid="stSidebar"] .stButton button {
  width: 100%; justify-content: flex-start; text-align: left;
  padding: var(--s2) 10px var(--s2) var(--s3); border-radius: var(--r-sm);
  border: 1px solid transparent; background: transparent; color: var(--sb-ink-2);
  font-size: var(--t-md); font-weight: 500; letter-spacing: -.006em;
  box-shadow: none; position: relative; min-height: 38px;
  transition: background var(--d1) var(--e-out), color var(--d1) var(--e-out);
}
[data-testid="stSidebar"] .stButton button p {
  font-size: var(--t-md); font-weight: 500; color: inherit;
}
/* Nav icons are Material Symbols rather than this file's own SVG set: a button
   label cannot carry authored HTML, and Material Symbols is the icon syntax
   Streamlit itself supports for widget labels. Size and weight are matched to
   the 16px stroke-1.5 set used in page content so the two read as one family.
   `justify-content` has to be restated on the two unnamed flex wrappers inside
   the button: they are hard-coded to `center`, so a left-aligned button still
   centred its icon and label as a pair (measured: the icon started 53px in). */
[data-testid="stSidebar"] .stButton button > div,
[data-testid="stSidebar"] .stButton button > div > span {
  justify-content: flex-start; width: 100%; min-width: 0;
}
[data-testid="stSidebar"] .stButton button [data-testid="stIconMaterial"] {
  font-size: 18px; width: 18px; height: 18px; margin-right: var(--s2);
  opacity: .85; font-variation-settings: 'FILL' 0, 'wght' 400;
}
[data-testid="stSidebar"] .stButton button:hover [data-testid="stIconMaterial"],
[data-testid="stSidebar"] .stButton button[kind="primary"] [data-testid="stIconMaterial"] {
  opacity: 1;
}
[data-testid="stSidebar"] .stButton button::before {
  content: ''; position: absolute; left: 0; top: 50%; width: 2px; height: 0;
  border-radius: 0 2px 2px 0; background: var(--sb-active);
  transform: translateY(-50%); transition: height var(--d2) var(--e-out);
}
[data-testid="stSidebar"] .stButton button:hover {
  background: var(--sb-bg-2); color: var(--sb-ink);
}
[data-testid="stSidebar"] .stButton button:active { background: var(--sb-bg-3); }
[data-testid="stSidebar"] .stButton button[kind="primary"],
[data-testid="stSidebar"] [data-testid="stBaseButton-primary"] {
  background: var(--sb-bg-3); color: #ffffff; border-color: transparent;
  font-weight: 600;
}
[data-testid="stSidebar"] .stButton button[kind="primary"] p,
[data-testid="stSidebar"] [data-testid="stBaseButton-primary"] p {
  font-weight: 600; color: #ffffff;
}
[data-testid="stSidebar"] .stButton button[kind="primary"]::before { height: 18px; }
[data-testid="stSidebar"] .stButton button[kind="primary"]:hover {
  background: #24395a;
}

.side-rule {
  height: 0; border-top: 1px solid var(--sb-line);
  margin: var(--s5) 0 var(--s4);
}
.status {
  display: flex; align-items: center; gap: var(--s2); font-size: var(--t-sm);
  font-weight: 500; padding: 0 var(--s2) var(--s3);
}
.status .dot { width: 7px; height: 7px; border-radius: 50%; flex: none; }
.status.ready { color: #6ee7a8; }
.status.ready .dot { background: #34d399; box-shadow: 0 0 0 3px rgba(52,211,153,.20); }
.status.down { color: #fca5a1; }
.status.down .dot { background: #f87171; box-shadow: 0 0 0 3px rgba(248,113,113,.20); }
.status.wait { color: var(--sb-ink-3); }
.status.wait .dot { background: var(--sb-ink-3); box-shadow: 0 0 0 3px rgba(124,139,163,.20); }
.side-meta { padding: 0 var(--s2); }
.side-meta div {
  font-size: var(--t-xs); color: var(--sb-ink-3); line-height: 1.7;
  display: flex; align-items: center; gap: var(--s2);
}
.side-meta div .ic { color: var(--sb-ink-3); flex: none; opacity: .8; }

/* ---------- panels / cards ---------- */
/* One card system: same border, same radius, same elevation, padding from the
   scale. `st-key-panel-*` is Streamlit's own container-key hook, so the class
   is stable rather than an emotion hash. */
[data-testid="stVerticalBlock"][class*="st-key-panel-"] {
  border: 1px solid var(--line); border-radius: var(--r-lg);
  background: var(--surface); box-shadow: var(--sh-1);
  padding: var(--s5); margin-bottom: var(--s4);
  transition: box-shadow var(--d2) var(--e-out), border-color var(--d2) var(--e-out);
}
[data-testid="stVerticalBlock"][class*="st-key-panel-"]:hover {
  box-shadow: var(--sh-2); border-color: var(--line-2);
}
/* group head: numbered, iconed, with a rule that fills the remaining width so
   the card reads as a titled section rather than a floating label */
.grp {
  display: flex; align-items: center; gap: var(--s3);
  margin: 0 0 var(--s5); padding-bottom: var(--s3);
  border-bottom: 1px solid var(--line);
}
.grp .n {
  width: 26px; height: 26px; border-radius: var(--r-sm); flex: none;
  background: var(--primary); color: var(--ink-inv);
  font-size: var(--t-xs); font-weight: 700; display: flex;
  align-items: center; justify-content: center; font-variant-numeric: tabular-nums;
  box-shadow: 0 1px 3px rgba(24,79,149,.35);
}
.grp .ic { color: var(--primary); flex: none; }
.grp .t {
  font-size: var(--t-lg); font-weight: 600; color: var(--ink);
  letter-spacing: -.016em;
}
.grp .hint {
  font-size: var(--t-sm); color: var(--ink-3); margin-left: auto;
  font-weight: 400; text-align: right;
}
/* a sub-group inside a card (internet add-ons): no rule, lighter title */
.grp.sub { border-bottom: none; padding-bottom: 0; margin: var(--s5) 0 var(--s3); }
.grp.sub .t { font-size: var(--t-md); font-weight: 600; }

/* ---------- widgets ---------- */
/* Streamlit 1.6x renders react-aria, not BaseWeb: the selectors below match the
   real DOM (verified in the running app), and the theme block in
   .streamlit/config.toml carries border colour and radius for the rest. */
[data-testid="stWidgetLabel"] p {
  font-size: var(--t-sm) !important; font-weight: 500 !important;
  color: var(--ink-2) !important; letter-spacing: -.002em;
}
[data-testid="stWidgetLabel"] { margin-bottom: var(--s2); align-items: center; }
/* Streamlit gives the help-icon wrapper `flex: 1 1 0; justify-content: flex-end`,
   so in a wide column the icon lands ~600px from the label it belongs to. Pull it
   back beside the text. */
[data-testid="stWidgetLabel"] > div:has(> [data-testid="stTooltipIcon"]) {
  flex: 0 0 auto; justify-content: flex-start; margin-left: var(--s1);
}
[data-testid="stTooltipHoverTarget"] svg { color: var(--ink-3); }
[data-testid="stSelectbox"], [data-testid="stNumberInput"],
[data-testid="stSlider"], [data-testid="stButtonGroup"] { margin-bottom: var(--s4); }

/* text inputs and the combobox */
.react-aria-ComboBox div[role="group"],
[data-testid="stNumberInput"] div[role="group"] {
  background: var(--surface); border: 1px solid var(--line-3);
  border-radius: var(--r-sm); min-height: 40px;
  transition: border-color var(--d1) var(--e-out), box-shadow var(--d1) var(--e-out);
}
.react-aria-ComboBox div[role="group"]:hover,
[data-testid="stNumberInput"] div[role="group"]:hover { border-color: var(--ink-3); }
.react-aria-ComboBox div[role="group"]:focus-within,
[data-testid="stNumberInput"] div[role="group"]:focus-within {
  border-color: var(--primary); box-shadow: var(--ring);
}
.react-aria-ComboBox input, [data-testid="stNumberInput"] input {
  font-size: var(--t-md); color: var(--ink);
}
[data-testid="stNumberInput"] input { font-variant-numeric: tabular-nums; }
.react-aria-ComboBox input::placeholder { color: var(--ink-3); }

/* segmented control: our own selected state, not the default grey */
[data-testid="stButtonGroup"] button[data-variant="segmented_control"] {
  border-color: var(--line-2); font-size: var(--t-md); font-weight: 500;
  color: var(--ink-2); min-height: 36px; background: var(--surface);
  transition: background var(--d1) var(--e-out), color var(--d1) var(--e-out),
              border-color var(--d1) var(--e-out), box-shadow var(--d1) var(--e-out);
}
[data-testid="stButtonGroup"] button[data-variant="segmented_control"] p {
  font-size: var(--t-md); font-weight: 500; color: inherit;
}
[data-testid="stButtonGroup"] button[data-variant="segmented_control"]:hover {
  background: var(--sunken); color: var(--ink); border-color: var(--line-3);
}
[data-testid="stButtonGroup"] button[data-variant="segmented_control"][aria-checked="true"] {
  background: var(--primary); border-color: var(--primary); color: var(--ink-inv);
  box-shadow: 0 1px 3px rgba(24,79,149,.32);
}
[data-testid="stButtonGroup"] button[data-variant="segmented_control"][aria-checked="true"] p {
  color: var(--ink-inv); font-weight: 600;
}
[data-testid="stButtonGroup"] button[data-variant="segmented_control"][aria-checked="true"]:hover {
  background: var(--primary-hover); border-color: var(--primary-hover);
}
[data-testid="stButtonGroup"] [aria-disabled="true"] button,
[data-testid="stButtonGroup"] button[disabled] { opacity: .45; }
[data-testid="stButtonGroup"] [role="radiogroup"][aria-disabled="true"] { opacity: .45; }

/* toggles: whole row reads as one setting */
[data-testid="stCheckbox"] { margin-bottom: 0; }
[data-testid="stCheckbox"] > label {
  padding: var(--s2) 10px; border-radius: var(--r-sm); width: 100%;
  transition: background var(--d1) var(--e-out);
}
[data-testid="stCheckbox"] > label:hover { background: var(--sunken); }
[data-testid="stCheckbox"] p {
  font-size: var(--t-md) !important; color: var(--ink-2) !important;
  font-weight: 500 !important;
}
[data-testid="stCheckbox"] > label:has(input:disabled) { opacity: .45; }
[data-testid="stCheckbox"] > label:has(input:disabled):hover { background: transparent; }

[data-testid="stSliderTickBar"] p {
  font-size: var(--t-xs); color: var(--ink-3); font-variant-numeric: tabular-nums;
}
/* The live value above the slider thumb. Both selectors are needed: the value is
   wrapped in a markdown <p>, so on the wrapper alone the colour and size below lose
   to `.stApp p` and the number rendered in 15px body slate instead of the 13px navy
   accent it is meant to be. The descendant selector is scoped with `.stApp` so it
   outranks `.stApp p` on specificity rather than on source order. */
[data-testid="stSliderThumbValue"],
.stApp [data-testid="stSliderThumbValue"] p {
  color: var(--primary); font-size: var(--t-sm); font-weight: 600;
  font-variant-numeric: tabular-nums;
}

/* ---------- buttons ---------- */
/* Three weights, and they do not look alike: a filled primary that carries a
   tinted shadow so it reads as raised, an outlined secondary, and the sidebar's
   ghost rows above. */
/* Streamlit renders a button's label as a <p>, so `.stApp p` (0,1,1) outranks the
   colour the button itself declares and inheritance never reaches the label. Handing
   it back is what keeps button and label in step: without this the primary CTA painted
   its label in body slate on the navy fill -- measured 1.05:1, invisible -- while its
   icon rendered white at 8.10:1, so the button read as an icon with a smear beside it.
   Stated once for every button here rather than per variant, because stating it per
   variant is how the primary got missed. The sidebar does the same on line 475. It
   also lets the secondary's :hover colour reach its label, which it could not before. */
[data-testid="stMain"] .stButton button p,
[data-testid="stMain"] .stDownloadButton button p,
[data-testid="stMain"] .stFormSubmitButton button p { color: inherit; }
[data-testid="stMain"] .stButton button[kind="primary"],
[data-testid="stMain"] [data-testid="stBaseButton-primary"] {
  background: var(--primary); border: 1px solid var(--primary-deep);
  color: var(--ink-inv); border-radius: var(--r-md);
  padding: var(--s2) var(--s5); font-weight: 600; font-size: var(--t-lg);
  letter-spacing: -.008em; min-height: 52px; box-shadow: var(--sh-cta);
  transition: background var(--d2) var(--e-out), box-shadow var(--d2) var(--e-out),
              transform var(--d1) var(--e-out);
}
[data-testid="stMain"] .stButton button[kind="primary"] p {
  font-size: var(--t-lg); font-weight: 600;
}
[data-testid="stMain"] .stButton button[kind="primary"]:hover {
  background: var(--primary-lift); border-color: var(--primary);
  box-shadow: var(--sh-cta-h); transform: translateY(-1px);
}
[data-testid="stMain"] .stButton button[kind="primary"]:active {
  background: var(--primary-hover); transform: translateY(1px);
  box-shadow: 0 1px 2px rgba(12,22,38,.18);
}
[data-testid="stMain"] .stButton button[kind="secondary"] {
  border: 1px solid var(--line-3); color: var(--ink-2); background: var(--surface);
  border-radius: var(--r-md); font-weight: 500; font-size: var(--t-md);
  /* padding restated so this and the primary resolve to the same 52px box: left to
     Streamlit's own padding the two ended up 2px apart in the same row. */
  padding: var(--s2) var(--s5); min-height: 52px; box-shadow: var(--sh-1);
  transition: border-color var(--d1) var(--e-out), color var(--d1) var(--e-out),
              background var(--d1) var(--e-out);
}
[data-testid="stMain"] .stButton button[kind="secondary"] p { font-size: var(--t-md); }
[data-testid="stMain"] .stButton button[kind="secondary"]:hover {
  border-color: var(--ink-3); color: var(--ink); background: var(--sunken);
}
[data-testid="stMain"] .stButton button[kind="secondary"]:active { transform: translateY(1px); }
.stApp p.cta-note { font-size: var(--t-sm); color: var(--ink-3); margin: var(--s3) 0 0; }

/* the analysis bar: the CTA sits in its own surface so it separates from the
   form above it and reads as the page's one action */
[data-testid="stVerticalBlock"][class*="st-key-actionbar"] {
  border: 1px solid var(--primary-line); border-radius: var(--r-lg);
  background: linear-gradient(180deg, var(--primary-wash), var(--surface) 88%);
  padding: var(--s5); margin: var(--s5) 0 var(--s6); box-shadow: var(--sh-1);
}
.act-head {
  display: flex; align-items: baseline; gap: var(--s3); flex-wrap: wrap;
  margin-bottom: var(--s4);
}
.act-head .t {
  font-size: var(--t-lg); font-weight: 600; color: var(--ink);
  letter-spacing: -.016em;
}
.act-head .d { font-size: var(--t-sm); color: var(--ink-3); }

/* ---------- empty / error / callout ---------- */
.empty {
  border: 1px dashed var(--line-2); border-radius: var(--r-lg);
  background: var(--surface); padding: var(--s8) var(--s5); text-align: center;
}
.empty .glyph {
  width: 48px; height: 48px; border-radius: var(--r-lg); margin: 0 auto var(--s4);
  background: var(--primary-wash); color: var(--primary);
  border: 1px solid var(--primary-line);
  display: flex; align-items: center; justify-content: center;
}
.empty h3 {
  font-size: var(--t-lg); margin: 0 0 var(--s2); font-weight: 600;
  letter-spacing: -.016em;
}
.empty p {
  color: var(--ink-3); font-size: var(--t-md); margin: 0 auto; max-width: 46ch;
}

.callout {
  display: flex; gap: var(--s3); padding: var(--s4); border-radius: var(--r-md);
  border: 1px solid var(--line); background: var(--surface);
  align-items: flex-start; box-shadow: var(--sh-1);
  border-left: 3px solid var(--line-2);
}
.callout .ic { flex: none; margin-top: 2px; color: var(--ink-3); }
.callout .body { font-size: var(--t-md); color: var(--ink-2); line-height: 1.6; }
.callout .body b { color: var(--ink); font-weight: 600; }
.callout.warn {
  background: var(--high-wash); border-color: var(--high-line);
  border-left-color: var(--high-mark);
}
.callout.warn .ic, .callout.warn .body b { color: var(--high-text); }
.callout.info {
  background: var(--primary-wash); border-color: var(--primary-line);
  border-left-color: var(--primary);
}
.callout.info .ic, .callout.info .body b { color: var(--primary); }
.callout.info code { background: var(--primary-wash-2); }
.stApp p.disclaimer {
  font-size: var(--t-sm); color: var(--ink-3); line-height: 1.6;
  border-left: 2px solid var(--line-2); padding: var(--s1) 0 var(--s1) var(--s3);
  max-width: 80ch; margin: 0;
}

/* ---------- result card ---------- */
/* The one place with real display type. Elevation is a step above every other
   card so the answer is unmistakably the answer. */
.result {
  border: 1px solid var(--line); border-radius: var(--r-xl); background: var(--surface);
  box-shadow: var(--sh-3); overflow: hidden; position: relative;
}
.result::before {
  content: ''; position: absolute; inset: 0 0 auto; height: 3px;
  background: var(--accent-bar, var(--primary));
}
.result .top {
  display: flex; flex-wrap: wrap; align-items: center; gap: var(--s5) var(--s7);
  padding: var(--s6) var(--s6) var(--s5);
}
.result .prob { min-width: 0; }
.result .prob .k, .result .side .k {
  font-size: var(--t-xs); text-transform: uppercase; letter-spacing: .1em;
  color: var(--ink-3); font-weight: 600; margin-bottom: var(--s2);
}
.result .prob .v {
  font-size: var(--t-num); line-height: .95; font-weight: 700; letter-spacing: -.04em;
  font-variant-numeric: tabular-nums; color: var(--ink);
}
.result .prob .v .pc {
  font-size: var(--t-2xl); font-weight: 600; margin-left: .04em;
  letter-spacing: -.02em; color: var(--ink-3);
}
.result .prob .band { margin-top: var(--s4); }
.result .side {
  display: flex; flex-direction: column; align-items: flex-start;
  padding-left: var(--s7); border-left: 1px solid var(--line);
}
.result .pred {
  display: flex; align-items: center; gap: var(--s2); font-size: var(--t-xl);
  font-weight: 600; color: var(--ink); letter-spacing: -.022em;
}
.result .pred .ic { color: var(--ink-3); flex: none; }
.band {
  display: inline-flex; align-items: center; gap: var(--s2);
  padding: var(--s2) var(--s4) var(--s2) var(--s3); border-radius: var(--r-md);
  font-size: var(--t-lg); font-weight: 700; letter-spacing: -.014em;
  border: 1px solid; text-transform: uppercase;
}
.band .ic { flex: none; }

/* risk meter: zones are labelled in text, marker carries the number */
.meter { padding: 0 var(--s6) var(--s5); }
.meter .track {
  position: relative; height: 12px; border-radius: var(--r-sm); overflow: hidden;
  display: flex; background: var(--sunken); border: 1px solid var(--line);
}
.meter .zone { height: 100%; }
.meter .zone + .zone { border-left: 2px solid var(--surface); }
/* The marker's children are absolute, so this height is what reserves room for
   them -- it has to be at least the chip's offset plus the chip's own box or the
   chip drops into the scale row underneath. At 30px it overlapped the band label
   by exactly 6px. The chip's line-height is pinned just below so that box is a
   known 20px (17 + 20 = 37) rather than whatever it inherits. */
.meter .marker { position: relative; height: 38px; margin-top: -1px; }
.meter .needle {
  position: absolute; top: 0; width: 3px; height: 18px; border-radius: 2px;
  transform: translateX(-1.5px);
  animation: needle-in var(--d3) var(--e-out) both;
}
.meter .valchip {
  position: absolute; top: 17px; transform: translateX(-50%);
  font-size: var(--t-xs); font-weight: 700; padding: 2px var(--s2);
  line-height: 16px; border-radius: var(--r-xs); font-variant-numeric: tabular-nums;
  white-space: nowrap; color: var(--ink-inv);
  animation: needle-in var(--d3) 40ms var(--e-out) both;
}
.meter .scale {
  display: flex; margin-top: var(--s1); font-size: var(--t-xs); color: var(--ink-3);
  font-weight: 500;
}
.meter .scale span { display: flex; flex-direction: column; gap: 1px; }
.meter .scale span em {
  font-style: normal; font-variant-numeric: tabular-nums; color: var(--ink-3);
  opacity: .8;
}
.meter .scale span.on { color: var(--ink); font-weight: 700; }

/* the three named readouts under the meter */
.facts {
  display: flex; border-top: 1px solid var(--line); background: var(--sunken);
}
.facts .f { flex: 1 1 0; min-width: 0; padding: var(--s4) var(--s5); }
.facts .f + .f { border-left: 1px solid var(--line); }
.facts .k {
  font-size: var(--t-xs); text-transform: uppercase; letter-spacing: .085em;
  color: var(--ink-3); font-weight: 600; margin-bottom: var(--s1);
  display: flex; align-items: center; gap: var(--s1);
}
.facts .v {
  font-size: var(--t-lg); font-weight: 700; color: var(--ink); letter-spacing: -.016em;
}
.facts .v.num { font-variant-numeric: tabular-nums; }
.facts .sub {
  font-size: var(--t-xs); color: var(--ink-3); margin-top: var(--s1); line-height: 1.5;
}

/* ---------- SHAP contributions ---------- */
.bridge {
  display: flex; align-items: stretch; gap: 0; border: 1px solid var(--line);
  border-radius: var(--r-lg); background: var(--surface); overflow: hidden;
  margin-bottom: var(--s4); box-shadow: var(--sh-1);
}
.bridge .cell { padding: var(--s4) var(--s5); flex: 1 1 0; min-width: 0; }
.bridge .cell + .cell { border-left: 1px solid var(--line); }
.bridge .k {
  font-size: var(--t-xs); text-transform: uppercase; letter-spacing: .085em;
  color: var(--ink-3); font-weight: 600; margin-bottom: var(--s2);
  display: flex; align-items: center; gap: var(--s1);
}
.bridge .v {
  font-size: var(--t-xl); font-weight: 700; font-variant-numeric: tabular-nums;
  letter-spacing: -.024em; color: var(--ink); line-height: 1.1;
}
.bridge .cell.net { background: var(--sunken); }
.bridge .sub { font-size: var(--t-xs); color: var(--ink-3); margin-top: var(--s1); }
.stApp p.bridge-note {
  font-size: var(--t-sm); color: var(--ink-3); max-width: 72ch;
  margin: 0 0 var(--s5); line-height: 1.6;
}

/* two contribution columns, each a card so the split reads as a comparison */
.contrib {
  border: 1px solid var(--line); border-radius: var(--r-lg); background: var(--surface);
  box-shadow: var(--sh-1); overflow: hidden; height: 100%;
}
.contrib .hd { padding: var(--s4) var(--s5) var(--s3); border-bottom: 1px solid var(--line); }
.contrib.up .hd { background: var(--high-wash); }
.contrib.down .hd { background: var(--primary-wash); }
.contrib-head {
  display: flex; align-items: center; gap: var(--s2); font-size: var(--t-lg);
  font-weight: 600; margin: 0 0 var(--s1); letter-spacing: -.016em;
}
.contrib-head .ic { flex: none; }
.contrib.up .contrib-head { color: var(--high-text); }
.contrib.down .contrib-head { color: var(--primary-deep); }
.stApp p.contrib-sub { font-size: var(--t-sm); color: var(--ink-3); margin: 0; }
.contrib .bd { padding: var(--s2) var(--s5) var(--s4); }

.crow { padding: var(--s3) 0; border-top: 1px solid var(--line); }
.crow:first-of-type { border-top: none; }
.crow .l {
  display: flex; align-items: baseline; gap: var(--s3); justify-content: space-between;
}
.crow .nm { font-size: var(--t-md); color: var(--ink); font-weight: 500; min-width: 0; }
.crow .nm i { font-style: normal; color: var(--ink-3); font-weight: 400; }
.crow .sv {
  font-size: var(--t-md); font-weight: 700; font-variant-numeric: tabular-nums;
  flex: none; letter-spacing: -.01em;
}
.crow .bar {
  height: 6px; border-radius: 3px; margin-top: var(--s2); background: var(--sunken);
  overflow: hidden;
}
.crow .bar i {
  display: block; height: 100%; border-radius: 3px;
  animation: bar-in 420ms var(--e-out) both;
}
.crow.none {
  color: var(--ink-3); font-size: var(--t-md); padding: var(--s5) 0; border: none;
  text-align: center;
}

/* ---------- metrics ---------- */
.kpi-lead {
  border: 1px solid var(--primary-line); border-radius: var(--r-lg);
  background: linear-gradient(160deg, var(--primary-wash), var(--surface) 92%);
  padding: var(--s5); box-shadow: var(--sh-1); height: 100%;
  display: flex; flex-direction: column;
}
.kpi-lead .k {
  font-size: var(--t-xs); text-transform: uppercase; letter-spacing: .09em;
  color: var(--primary); font-weight: 700; display: flex; align-items: center;
  gap: var(--s2); margin-bottom: var(--s3);
}
.kpi-lead .v {
  font-size: var(--t-3xl); line-height: 1; font-weight: 700; letter-spacing: -.038em;
  font-variant-numeric: tabular-nums; color: var(--primary-deep);
}
.kpi-lead .d {
  font-size: var(--t-md); color: var(--ink-2); margin-top: var(--s3); line-height: 1.6;
}

/* the supporting metrics: an even grid of equal cards, not a stack of rows */
/* Card gutters are one step up from 12px on purpose: at 12 the gap between two
   cards was narrower than the 16px padding inside them, so a pair read as one
   crowded block. Gutter >= card padding is what makes a grid read as separate
   cards. */
/* The column count is a request, not a command, and this is the one place worth
   explaining because the same shape is used by three grids below.

   `--kc` says how many columns the call site wants. The track floor says how
   narrow a card may get. `auto-fit` reconciles them: the minimum of each track is
   whichever is larger of the floor and the exact 1/kc share of the row, so while
   the ideal share is the wider of the two every track takes it and precisely `kc`
   columns fit; once the row is too narrow for that, the floor wins, and auto-fit
   packs as many floor-width columns as fit and drops the rest.

   This replaces two viewport breakpoints that used to hard-set the count at 900px
   and 640px. They were the wrong instrument: a grid inside `st.columns` is limited
   by its container, not by the window, so at 768px the viewport was comfortably
   above the 640px stack point while the container it actually sat in was 248px. */
.kpi-grid {
  --gutter: var(--s4);
  display: grid; gap: var(--gutter);
  grid-template-columns: repeat(auto-fit, minmax(
    max(var(--card-min), (100% - (var(--kc, 2) - 1) * var(--gutter)) / var(--kc, 2)),
    1fr));
}
.kpi {
  border: 1px solid var(--line); border-radius: var(--r-md); background: var(--surface);
  padding: var(--s4); box-shadow: var(--sh-1);
  transition: box-shadow var(--d2) var(--e-out), border-color var(--d2) var(--e-out),
              transform var(--d2) var(--e-out);
}
.kpi:hover {
  box-shadow: var(--sh-2); border-color: var(--line-2); transform: translateY(-1px);
}
.kpi .k {
  font-size: var(--t-xs); text-transform: uppercase; letter-spacing: .08em;
  color: var(--ink-3); font-weight: 600;
}
.kpi .v {
  font-size: var(--t-xl); font-weight: 700; font-variant-numeric: tabular-nums;
  letter-spacing: -.026em; color: var(--ink); margin: var(--s2) 0 var(--s1);
  line-height: 1;
}
.kpi .d { font-size: var(--t-sm); color: var(--ink-3); line-height: 1.5; }

/* explanatory cards: one idea each, for material that would otherwise be a wall
   of paragraphs */
.ncards {
  --gutter: var(--s4);
  display: grid; gap: var(--gutter);
  grid-template-columns: repeat(auto-fit, minmax(
    max(var(--card-min), (100% - (var(--nc, 3) - 1) * var(--gutter)) / var(--nc, 3)),
    1fr));
}
.ncard {
  border: 1px solid var(--line); border-radius: var(--r-md); background: var(--surface);
  padding: var(--s4); box-shadow: var(--sh-1);
  transition: box-shadow var(--d2) var(--e-out), border-color var(--d2) var(--e-out);
}
.ncard:hover { box-shadow: var(--sh-2); border-color: var(--line-2); }
.ncard .h {
  display: flex; align-items: center; gap: var(--s2); font-size: var(--t-md);
  font-weight: 600; color: var(--ink); margin-bottom: var(--s2);
  letter-spacing: -.01em;
}
.ncard .h .ic { color: var(--primary); flex: none; }
.ncard p { font-size: var(--t-sm); color: var(--ink-2); line-height: 1.6; margin: 0; }
.ncard p b { color: var(--ink); font-weight: 600; }
.ncard p + p { margin-top: var(--s2); }

/* metrics table -------------------------------------------------------- */
/* Scrolls sideways rather than clipping: at 375px the six metric columns are
   wider than the viewport, and hidden overflow simply lost the last two. Radius
   still clips the corners. */
.tbl-wrap {
  border: 1px solid var(--line); border-radius: var(--r-lg);
  overflow-x: auto; overflow-y: hidden; background-color: var(--surface);
  box-shadow: var(--sh-1);
  /* Scroll affordance, in CSS rather than a scroll listener. A seven-column metrics
     table cannot fit a tablet column without hiding a column, so it scrolls inside
     this box -- but a table that scrolls with no visible edge just reads as cut off.
     Two edge shadows are pinned to the box (`scroll`), and two surface-coloured
     covers ride with the content (`local`): whichever end you are scrolled to has
     its cover sitting over its shadow, so the shadow shows only on the side that
     still has content, and neither shows on a table that already fits. */
  background-image:
    linear-gradient(to right, var(--surface) 55%, rgba(255, 255, 255, 0)),
    linear-gradient(to left, var(--surface) 55%, rgba(255, 255, 255, 0)),
    linear-gradient(to right, rgba(12, 22, 38, .10), rgba(12, 22, 38, 0)),
    linear-gradient(to left, rgba(12, 22, 38, .10), rgba(12, 22, 38, 0));
  background-repeat: no-repeat;
  background-size: 26px 100%, 26px 100%, 15px 100%, 15px 100%;
  background-position: 0 0, 100% 0, 0 0, 100% 0;
  background-attachment: local, local, scroll, scroll;
}
table.tbl { width: 100%; border-collapse: collapse; font-size: var(--t-md); }
table.tbl th, table.tbl td { padding: var(--s3) var(--s4); text-align: right; }
table.tbl th:first-child, table.tbl td:first-child { text-align: left; }
table.tbl thead th {
  background: var(--sunken); font-size: var(--t-xs); font-weight: 600;
  color: var(--ink-2); text-transform: uppercase; letter-spacing: .06em;
  border-bottom: 1px solid var(--line); white-space: nowrap;
}
table.tbl td {
  border-top: 1px solid var(--line); color: var(--ink-2);
  font-variant-numeric: tabular-nums;
}
table.tbl td:first-child {
  color: var(--ink); font-weight: 500; font-variant-numeric: normal;
}
table.tbl tbody tr { transition: background var(--d1) var(--e-out); }
table.tbl tbody tr:hover td { background: var(--primary-wash); }
table.tbl tr.sel td { background: var(--primary-wash); }
table.tbl tr.sel:hover td { background: var(--primary-wash-2); }
table.tbl tr.sel td:first-child { color: var(--primary-deep); font-weight: 700; }
table.tbl td.best { color: var(--ink); font-weight: 700; }
table.tbl td.best::after {
  content: 'best'; display: inline-block; margin-left: var(--s1);
  font-size: 10px; font-weight: 700; text-transform: uppercase;
  letter-spacing: .06em; color: var(--primary); vertical-align: .1em;
  background: var(--primary-wash); border: 1px solid var(--primary-line);
  border-radius: var(--r-xs); padding: 0 var(--s1);
}
.stApp p.tbl-note { font-size: var(--t-sm); color: var(--ink-3); margin: var(--s3) 0 0; }

/* pipeline rail --------------------------------------------------------- */
.rail { display: flex; flex-direction: column; gap: 0; }
.step { display: flex; gap: var(--s4); position: relative; }
.step .gut {
  display: flex; flex-direction: column; align-items: center; flex: none; width: 28px;
}
.step .n {
  width: 28px; height: 28px; border-radius: var(--r-sm); flex: none;
  background: var(--surface); border: 1px solid var(--line-2); color: var(--ink-2);
  font-size: var(--t-xs); font-weight: 700; display: flex; align-items: center;
  justify-content: center; font-variant-numeric: tabular-nums; z-index: 1;
  box-shadow: var(--sh-1);
}
.step .ln { width: 2px; flex: 1 1 auto; background: var(--line); min-height: var(--s3); }
.step:last-child .ln { background: transparent; }
.step .bd { padding-bottom: var(--s5); min-width: 0; padding-top: 3px; }
.step .t {
  font-size: var(--t-md); font-weight: 600; color: var(--ink); letter-spacing: -.012em;
}
.step .d {
  font-size: var(--t-sm); color: var(--ink-3); margin-top: 2px; line-height: 1.55;
}
.step.key .n {
  background: var(--primary); border-color: var(--primary-deep); color: var(--ink-inv);
  box-shadow: 0 1px 3px rgba(24,79,149,.35);
}
.step.key .t { color: var(--primary-deep); }
.step.key .ln { background: var(--primary-line); }

/* project credits ------------------------------------------------------- */
/* The colophon used to be a three-line paragraph, which is the one thing the rest
   of this page spends its effort not being. Same grammar as every other card here
   -- one surface, one accent edge, a header, a member grid, a supervisor footer --
   so it closes the page as part of the system rather than as a signature. */
.credits {
  border: 1px solid var(--line); border-radius: var(--r-xl); background: var(--surface);
  box-shadow: var(--sh-2); overflow: hidden; position: relative;
}
.credits::before {
  content: ''; position: absolute; inset: 0 0 auto; height: 3px;
  background: linear-gradient(to right, var(--primary), var(--primary-lift));
}
.credits .hd {
  padding: var(--s5) var(--s5) var(--s4); border-bottom: 1px solid var(--line);
}
.credits .hd h3 {
  font-size: var(--t-lg); font-weight: 600; color: var(--ink); margin: 0 0 var(--s2);
  letter-spacing: -.016em; line-height: 1.42; max-width: 58ch;
  display: flex; align-items: flex-start; gap: var(--s2);
}
.credits .hd h3 .ic { color: var(--primary); flex: none; margin-top: 3px; }
.stApp p.cr-org {
  font-size: var(--t-sm); color: var(--ink-3); line-height: 1.5; margin: 0;
  padding-left: 25px; max-width: 62ch;
}
.credits .team {
  --gutter: var(--s4);
  display: grid; gap: var(--gutter); padding: var(--s5);
  grid-template-columns: repeat(auto-fit, minmax(
    max(var(--card-min), (100% - var(--gutter)) / 2), 1fr));
}
.credits .who {
  display: flex; align-items: center; gap: var(--s3); min-width: 0;
  border: 1px solid var(--line); border-radius: var(--r-md);
  padding: var(--s4); background: var(--surface);
  transition: border-color var(--d2) var(--e-out), box-shadow var(--d2) var(--e-out);
}
.credits .who:hover { border-color: var(--primary-line); box-shadow: var(--sh-1); }
.credits .av {
  flex: none; width: 42px; height: 42px; border-radius: var(--r-full);
  display: flex; align-items: center; justify-content: center;
  background: var(--primary); color: var(--ink-inv);
  font-size: var(--t-md); font-weight: 600; letter-spacing: .02em;
  border: 1px solid var(--primary-deep); box-shadow: 0 1px 3px rgba(24,79,149,.32);
}
.credits .tx { min-width: 0; }
.credits .nm {
  display: block; font-size: var(--t-md); font-weight: 600; color: var(--ink);
  letter-spacing: -.012em; line-height: 1.35;
}
.credits .rl {
  display: block; font-size: var(--t-sm); color: var(--ink-3);
  line-height: 1.45; margin-top: 2px;
}
.credits .ft {
  padding: var(--s4) var(--s5); background: var(--sunken);
  border-top: 1px solid var(--line);
}
.credits .ft .who { border: none; background: none; padding: 0; box-shadow: none; }
.credits .ft .av {
  background: var(--surface); color: var(--primary);
  border-color: var(--primary-line); box-shadow: none;
}

/* figure frame ---------------------------------------------------------- */
/* The PNG sits on --paper, the surface it was rendered against, inside a card
   that belongs to this page. That is what stops a saved chart reading as a
   pasted screenshot. */
[data-testid="stVerticalBlock"][class*="st-key-fig-"] {
  border: 1px solid var(--line); border-radius: var(--r-lg); background: var(--surface);
  padding: var(--s3); margin-bottom: var(--s5); box-shadow: var(--sh-1);
  transition: box-shadow var(--d2) var(--e-out);
}
[data-testid="stVerticalBlock"][class*="st-key-fig-"]:hover { box-shadow: var(--sh-2); }
/* The plate is sized to the chart rather than to the column. Once the height cap
   brings a near-square figure down to ~510px, a full-width plate leaves 280px of
   blank paper either side and the border draws a box around nothing, so the plate
   hugs (`fit-content`) and the wrapper Streamlit puts around it -- which already
   shrinks to its content -- is centred from stFullScreenFrame, the nearest element
   above it with a stable testid and the full column width. */
[data-testid="stVerticalBlock"][class*="st-key-fig-"] [data-testid="stImage"] {
  background: var(--paper); border: 1px solid var(--line);
  border-radius: var(--r-md); padding: var(--s3); display: block;
  width: fit-content; max-width: 100%;
}
[data-testid="stVerticalBlock"][class*="st-key-fig-"] [data-testid="stFullScreenFrame"] {
  display: flex; justify-content: center;
}
/* Streamlit writes the stretched width inline (`width: 1070.4px`), and a definite
   width means `max-height` squashes the image instead of scaling it -- measured:
   every figure rendered 1045x460 with its aspect ratio destroyed. An aspect ratio
   is only honoured when both axes are free, so the inline width is handed back
   (the one thing `!important` is actually for) and the size is bounded instead:
   max-width keeps it inside the column, max-height brings the near-square figures
   down, and the browser derives the other axis. A figure narrower than the column
   now renders at its own size rather than being upscaled into softness, and below
   ~700px the width bound takes over and the height cap stops applying. */
[data-testid="stImage"] img {
  border-radius: var(--r-sm); display: block; margin-inline: auto;
  width: auto !important; height: auto;
  max-width: 100%; max-height: var(--fig-max-h);
}
/* Caption centred with the plate above it. A left-aligned caption made sense when
   the plate spanned the column; under a centred plate it detached from the figure
   it belongs to. Width-capped so a centred line does not sprawl. */
.stApp p.fig-cap {
  font-size: var(--t-sm); color: var(--ink-3); margin: var(--s3) auto var(--s1);
  line-height: 1.55; display: flex; gap: var(--s2);
  justify-content: center; text-align: center; max-width: 76ch;
}
.fig-cap .lbl { color: var(--ink-2); font-weight: 600; flex: none; }

[data-testid="stExpander"] { margin-bottom: var(--s4); }
[data-testid="stExpander"] details {
  border: 1px solid var(--line); border-radius: var(--r-md); background: var(--surface);
  box-shadow: var(--sh-1);
}
[data-testid="stExpander"] summary {
  font-size: var(--t-md); font-weight: 500; color: var(--ink-2);
  transition: color var(--d1) var(--e-out);
}
[data-testid="stExpander"] summary:hover { color: var(--primary); }
[data-testid="stDataFrame"] { border-radius: var(--r-md); }

/* ---------- motion ---------- */
/* Reveal animations run on content that appears in response to an explicit
   action -- a prediction, or a page reached from the nav. Form widgets rerun the
   whole script on every keystroke, so nothing on the input side animates in. */
@keyframes rise { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: none; } }
@keyframes fade { from { opacity: 0; } to { opacity: 1; } }
@keyframes bar-in { from { transform: scaleX(0); transform-origin: left; } to { transform: none; } }
@keyframes needle-in { from { opacity: 0; transform: translateY(-6px) translateX(-1.5px); } }
.arrive { animation: rise var(--d3) var(--e-out) both; }
.arrive-1 { animation: rise var(--d3) 60ms var(--e-out) both; }
.arrive-2 { animation: rise var(--d3) 120ms var(--e-out) both; }
.arrive-3 { animation: rise var(--d3) 180ms var(--e-out) both; }
.reveal { animation: fade var(--d3) var(--e-out) both; }

@media (prefers-reduced-motion: reduce) {
  .arrive, .arrive-1, .arrive-2, .arrive-3, .reveal,
  .meter .needle, .meter .valchip, .crow .bar i { animation: none; }
  *, *::before, *::after {
    transition-duration: .01ms !important; animation-duration: .01ms !important;
  }
  [data-testid="stMain"] .stButton button[kind="primary"]:hover,
  .kpi:hover { transform: none; }
}

/* ---------- responsive: structural, not fluid type ---------- */
@media (max-width: 1180px) {
  [data-testid="stMainBlockContainer"], .block-container {
    padding: var(--s5) var(--s5) var(--s7);
  }
}
@media (max-width: 1000px) {
  [data-testid="stMainBlockContainer"], .block-container {
    padding: var(--s5) var(--s4) var(--s7);
  }
  :root, .stApp { --t-3xl: 1.875rem; --t-2xl: 1.5rem; --t-num: 2.75rem; }
  [data-testid="stHorizontalBlock"] { gap: var(--s4); }
  .bridge { flex-direction: column; }
  .bridge .cell + .cell { border-left: none; border-top: 1px solid var(--line); }
  .facts { flex-direction: column; }
  .facts .f + .f { border-left: none; border-top: 1px solid var(--line); }
  .result .top { gap: var(--s4) var(--s5); padding: var(--s5); }
  .result .side {
    padding-left: 0; padding-top: var(--s4); border-left: none;
    border-top: 1px solid var(--line); width: 100%;
  }
  .meter { padding: 0 var(--s5) var(--s5); }
  .kpi-lead .v { font-size: 1.875rem; }
}
@media (max-width: 640px) {
  :root, .stApp { --t-num: 2.5rem; }
  [data-testid="stVerticalBlock"][class*="st-key-panel-"],
  [data-testid="stVerticalBlock"][class*="st-key-actionbar"] { padding: var(--s4); }
  .result .top { padding: var(--s4); }
  .meter { padding: 0 var(--s4) var(--s4); }
  .meter .scale { font-size: 11px; }
  .facts .f { padding: var(--s3) var(--s4); }
  .contrib .hd { padding: var(--s3) var(--s4); }
  .contrib .bd { padding: var(--s1) var(--s4) var(--s3); }
  /* The card grids need no rule here -- their track floor already stacks them at
     this width, and from the container rather than the window. */
  .credits .hd, .credits .team, .credits .ft { padding-left: var(--s4); padding-right: var(--s4); }
  .stApp p.cr-org { padding-left: 0; }
  table.tbl th, table.tbl td { padding: var(--s2) var(--s3); font-size: var(--t-sm); }
  table.tbl td.best::after { content: '*'; border: none; background: none; padding: 0; }
  .empty { padding: var(--s6) var(--s4); }
  /* The group hint is orienting context, not information: at this width it
     steals room from the title it is meant to support. */
  .grp .hint { display: none; }
  .grp { gap: var(--s2); }
}
"""


def inject() -> None:
    """Attach the stylesheet once per script run."""
    st.markdown(f"<style>{_CSS}</style>", unsafe_allow_html=True)
