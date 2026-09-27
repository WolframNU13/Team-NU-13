"""
Design tokens and stylesheet.

Derived from the reference set: a warm paper ground and near-black ink with a single cobalt
accent (editorial reference "The Shape of Intelligence"), terracotta / ochre data accents
(Zirka, Peryton, Mathis Biabiany), hairline structure instead of cards (VanLent), numbered
navigation and edge-anchored meta text (Austensor). Measured contrast on --paper (#F2F2EF):
ink 15.0:1, ink-2 8.0:1 (AAA); muted 5.3:1, accent 6.7:1, terracotta-text 5.0:1 (AA body text).
Data colours used as graphical marks (terracotta 4.4:1, ochre 3.6:1) meet the 3:1 non-text
minimum (WCAG 1.4.11). GHOST / CONTEXT are deliberately de-emphasised inactive states.
"""

from __future__ import annotations

import streamlit as st

PAPER = "#F2F2EF"
PAPER_RAISED = "#F8F8F6"
INK = "#1C1E1B"
INK_2 = "#474A45"
MUTED = "#62645F"
RULE = "#D8D8D3"
RULE_STRONG = "#B9B9B3"
ACCENT = "#3340D1"
ACCENT_SOFT = "#E4E6F7"
TERRACOTTA = "#C24A1E"
TERRACOTTA_TEXT = "#B3431B"
OCHRE = "#A37822"
VIOLET = "#7C4DBF"
GHOST = "#D2D2CC"          # unassembled (N) bins
CONTEXT = "#B4B4AD"        # beads outside the highlighted region
GRID = "#E3E3DE"

SANS = "'Inter Tight', 'Inter', system-ui, -apple-system, 'Segoe UI', sans-serif"
MONO = "'IBM Plex Mono', ui-monospace, 'SFMono-Regular', Consolas, monospace"

# Sequential scales. Every scale starts well away from the paper colour so low values stay
# visible on the page and on lit 3D surfaces.
SCALES: dict[str, list[str]] = {
    "Genomic position": [ACCENT, VIOLET, TERRACOTTA, OCHRE],
    "GC content": ["#AEB4EC", "#6B75DE", ACCENT, "#1A2175"],
    "H3K27ac": ["#E3BBA3", "#D27A4F", TERRACOTTA, "#6E2408"],
    "Monochrome": ["#2B2D2A", "#2B2D2A"],
}
DISTANCE_SCALE = [[0.0, "#1A2175"], [0.2, ACCENT], [0.5, "#9AA2E6"], [0.8, "#DDE0F2"], [1.0, PAPER_RAISED]]
CONTACT_SCALE = [[0.0, PAPER_RAISED], [0.25, "#EBC9B5"], [0.55, "#D9744A"], [0.8, "#A8380F"], [1.0, "#4A1705"]]

BAND_COLORS = {"gneg": "#FBFBF9", "gpos25": "#D9D9D4", "gpos50": "#ABABA5", "gpos75": "#7F7F79",
               "gpos100": "#4F4F4A", "acen": "#D9A58C", "gvar": "#C9CADB", "stalk": "#E6E6E1"}


def config_stops(name: str) -> list[list]:
    cols = SCALES[name]
    return [[i / (len(cols) - 1), c] for i, c in enumerate(cols)]


CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter+Tight:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root {{
  --paper: {PAPER}; --paper-raised: {PAPER_RAISED}; --ink: {INK}; --ink-2: {INK_2}; --muted: {MUTED};
  --rule: {RULE}; --rule-strong: {RULE_STRONG}; --accent: {ACCENT}; --accent-soft: {ACCENT_SOFT};
  --terracotta: {TERRACOTTA_TEXT}; --sans: {SANS}; --mono: {MONO};
}}

/* ---- frame ------------------------------------------------------------------------ */
.stApp {{ background: var(--paper); color: var(--ink); }}
header[data-testid="stHeader"] {{ display: none; }}
.block-container {{ padding: 26px 44px 20px !important; max-width: 1760px; }}
.stApp, .stApp p, .stApp label, .stApp li {{ font-family: var(--sans); }}
*:focus-visible {{ outline: 2px solid var(--accent) !important; outline-offset: 2px; }}

/* ---- type scale ------------------------------------------------------------------- */
.cc-eyebrow {{ font: 500 11px/1.4 var(--sans); letter-spacing: .08em; text-transform: uppercase; color: var(--muted); margin: 0; }}
.cc-meta {{ font: 400 13px/1.45 var(--sans); color: var(--ink-2); margin: 0; }}
.cc-meta.right {{ text-align: right; }}
.cc-num, .cc-locus {{ font-family: var(--mono); font-variant-numeric: tabular-nums; letter-spacing: -0.01em; }}
.cc-title {{ font: 500 44px/1.02 var(--sans); letter-spacing: -0.035em; color: var(--ink); margin: 4px 0 2px; }}
.cc-sub {{ font: 400 14px/1.5 var(--sans); color: var(--muted); margin: 0; }}

/* ---- top bar ---------------------------------------------------------------------- */
.cc-brand {{ display: flex; align-items: center; gap: 12px; height: 38px; }}
.cc-brand svg {{ flex: none; }}
.cc-brand b {{ font: 600 15px/1 var(--sans); letter-spacing: -0.01em; color: var(--ink); }}
.cc-brand span {{ font: 400 13px/1 var(--sans); color: var(--muted); }}
.st-key-nav button {{ background: transparent !important; border: none !important; box-shadow: none !important;
  color: var(--ink) !important; font: 400 14px/1 var(--sans) !important; padding: 8px 2px !important; min-height: 38px; }}
.st-key-nav button:hover {{ color: var(--accent) !important; }}
.st-key-nav button p {{ font-size: 14px !important; }}
.cc-rule {{ border: 0; border-top: 1px solid var(--rule); margin: 10px 0 18px; }}

/* ---- region control (numbered options, as in the reference slider) ---------------- */
.st-key-region [data-testid="stWidgetLabel"] p {{ font: 400 13px/1.4 var(--sans); color: var(--muted); }}
.st-key-region button {{ background: transparent !important; border: none !important; box-shadow: none !important;
  color: var(--ink-2) !important; padding: 4px 14px 4px 0 !important; border-radius: 0 !important; }}
.st-key-region button p {{ font: 400 14px/1.3 var(--sans) !important; }}
.st-key-region button[kind$="Active"], .st-key-region button[data-testid$="Active"] {{ color: var(--accent) !important; }}
.st-key-region button:hover {{ color: var(--accent) !important; }}

/* ---- ideogram --------------------------------------------------------------------- */
.cc-ideo {{ position: relative; height: 14px; display: flex; border: 1px solid var(--rule-strong); border-radius: 7px;
  overflow: hidden; margin: 6px 0 4px; background: var(--paper-raised); }}
.cc-ideo i {{ display: block; height: 100%; border-right: 1px solid rgba(0,0,0,.06); }}
.cc-ideo .win {{ position: absolute; top: -1px; bottom: -1px; border: 2px solid var(--accent); border-radius: 3px;
  background: rgba(51,64,209,.08); }}
.cc-ideo-axis {{ display: flex; justify-content: space-between; font: 400 11px/1.3 var(--mono); color: var(--muted); }}

/* ---- viewport HUD ----------------------------------------------------------------- */
.cc-hud {{ display: flex; justify-content: space-between; align-items: flex-end; gap: 16px; margin: 2px 0 0; }}
.cc-legend {{ display: flex; align-items: center; gap: 8px; font: 400 12px/1 var(--sans); color: var(--ink-2); white-space: nowrap; flex-wrap: nowrap; }}
.cc-legend .bar {{ width: 120px; height: 6px; border-radius: 3px; }}
.cc-legend .sw {{ width: 10px; height: 10px; border-radius: 50%; display: inline-block; }}
.cc-spec {{ margin: 0; padding: 0; list-style: none; font: 400 12px/1.7 var(--sans); color: var(--ink-2); }}
.cc-spec b {{ font-weight: 500; color: var(--ink); }}
.cc-help {{ font: 400 12px/1.6 var(--sans); color: var(--muted); text-align: right; }}
.cc-help kbd {{ font: 500 11px/1 var(--mono); border: 1px solid var(--rule-strong); border-radius: 3px; padding: 1px 4px; color: var(--ink-2); }}
.st-key-stage [data-testid="stPlotlyChart"] {{ border-top: 1px solid var(--rule); border-bottom: 1px solid var(--rule); }}

/* ---- inspector -------------------------------------------------------------------- */
.st-key-inspector {{ border-left: 1px solid var(--rule); padding-left: 22px !important; overflow-x: hidden !important; }}
.st-key-inspector .katex {{ font-size: 0.98em; }}
.st-key-inspector .katex-display {{ margin: 0.35em 0 !important; }}
.st-key-inspector [data-testid="stExpander"] details {{ border: none; border-top: 1px solid var(--rule);
  border-radius: 0; background: transparent; }}
.st-key-inspector [data-testid="stExpander"] summary {{ padding: 14px 0 12px; }}
.st-key-inspector [data-testid="stExpander"] summary p {{ font: 500 15px/1.2 var(--sans); color: var(--ink); }}
.st-key-inspector [data-testid="stExpander"] summary:hover p {{ color: var(--accent); }}
.st-key-inspector [data-testid="stExpanderDetails"] {{ padding: 0 0 14px; }}
.cc-readout {{ margin: 0 0 10px; }}
.cc-readout div {{ display: grid; grid-template-columns: 1fr auto; align-items: baseline; gap: 12px;
  padding: 7px 0; border-bottom: 1px solid var(--rule); }}
.cc-readout dt {{ font: 400 13px/1.35 var(--sans); color: var(--ink-2); margin: 0; }}
.cc-readout dt small {{ display: block; color: var(--muted); font-size: 11.5px; }}
.cc-readout dd {{ margin: 0; font: 500 13.5px/1.35 var(--mono); color: var(--ink); text-align: right; white-space: nowrap; }}
.cc-readout dd em {{ font: 400 12px var(--sans); font-style: normal; color: var(--muted); margin-left: 4px; }}
.cc-tag {{ display: inline-block; font: 500 11px/1 var(--sans); padding: 4px 7px; border-radius: 999px;
  border: 1px solid var(--rule-strong); color: var(--ink-2); }}
.cc-tag.ok {{ border-color: #9DB59A; color: #2F5A2B; background: #EEF3EC; }}
.cc-tag.warn {{ border-color: #D9B08C; color: #7A3E12; background: #F7EDE4; }}
.cc-tag.accent {{ border-color: #AEB4EC; color: var(--accent); background: var(--accent-soft); }}
.cc-note {{ font: 400 12.5px/1.5 var(--sans); color: var(--muted); margin: 6px 0 10px; }}
.cc-note code, .cc-meta code {{ font-family: var(--mono); font-size: 12px; color: var(--ink-2); background: none; padding: 0; }}
.cc-hub {{ display: grid; grid-template-columns: 1fr auto; font: 400 12.5px/1.8 var(--mono); color: var(--ink-2);
  border-bottom: 1px solid var(--rule); }}

/* ---- footer status bar ------------------------------------------------------------ */
.cc-status {{ display: flex; justify-content: space-between; gap: 20px; flex-wrap: wrap; border-top: 1px solid var(--rule);
  padding: 12px 0 0; margin-top: 14px; font: 400 12px/1.5 var(--sans); color: var(--muted); }}
.cc-status .cc-num {{ color: var(--ink-2); }}

/* ---- widgets ---------------------------------------------------------------------- */
.stDownloadButton button, .stButton button {{ font: 500 13px/1 var(--sans) !important; }}
[data-testid="stFileUploaderDropzone"] {{ background: var(--paper-raised); border: 1px dashed var(--rule-strong); }}
[data-testid="stPopoverBody"] {{ background: var(--paper-raised); border: 1px solid var(--rule); box-shadow: 0 12px 32px rgba(28,30,27,.08); }}
.stSlider [data-baseweb="slider"] div[role="slider"] {{ box-shadow: none; }}
</style>
"""


def inject() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def plot_layout(height: int, **kw) -> dict:
    """Base Plotly layout for 2D charts: transparent ground, hairline axes, tabular numerals."""
    axis = dict(gridcolor=GRID, linecolor=RULE_STRONG, zeroline=False, ticks="outside", ticklen=3,
                tickcolor=RULE_STRONG, tickfont=dict(size=11, color=MUTED), title_font=dict(size=11.5, color=INK_2),
                showline=True, mirror=False)
    base = dict(
        height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=SANS, size=12, color=INK_2),
        margin=dict(l=52, r=12, t=10, b=40),
        hoverlabel=dict(bgcolor=PAPER_RAISED, bordercolor=RULE_STRONG, font=dict(family=MONO, size=12, color=INK)),
        showlegend=False, xaxis=dict(axis), yaxis=dict(axis),
    )
    for k, v in kw.items():
        if k in ("xaxis", "yaxis", "xaxis2", "yaxis2") and isinstance(v, dict):
            base[k] = {**axis, **v}
        else:
            base[k] = v
    return base


PLOT_CONFIG = {"displayModeBar": False, "responsive": True}
