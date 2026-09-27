"""
ChronoCell-5D v1.0 — 3D Chromatin Structural Workstation
=========================================================

Interactive visualizer and analytics dashboard for the ChronoCell-5D pipeline:
an SE(3)-Equivariant Graph Neural Network (EGNN) that reconstructs the 3D
spatial coordinates P ∈ R^{N×3} of human Chromosome 22 (GRCh38/hg38) at
10 kb resolution (N = ⌈50,818,468 / 10,000⌉ = 5,082 nodes).

Pipeline hand-off (Teammate 3 — Structural Export & Visualization):
    graph_data.pt  (Teammate 1)  ->  predicted_coords.pt  (Teammate 2)  ->  this app

Run:
    pip install streamlit plotly pandas numpy          # torch optional (needed for .pt uploads)
    streamlit run app.py
"""

from __future__ import annotations

import datetime as dt
import functools
import inspect
import io
import json
import math
import os
from typing import Any

import numpy as np
import pandas as pd
import plotly
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

try:
    import torch

    TORCH_OK = True
except Exception:  # torch is optional: coordinates can also arrive as .pdb / .xyz / .npy / .csv
    torch = None
    TORCH_OK = False


# =============================================================================
# 0. PAGE CONFIG (must be the first Streamlit call)
# =============================================================================
st.set_page_config(
    page_title="ChronoCell-5D · 3D Chromatin Workstation",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================================
# 1. PROJECT CONSTANTS (from the ChronoCell-5D master specification)
# =============================================================================
APP_VERSION = "1.0"
CHROM_NAME = "chr22"
ASSEMBLY = "GRCh38/hg38"
CHROM_SIZE = 50_818_468                          # L, base pairs
RESOLUTION = 10_000                              # B, base pairs per node
N_BINS = math.ceil(CHROM_SIZE / RESOLUTION)      # N = 5,082

# Training hyper-parameters (Teammate 2 code module)
LAMBDA_SMOOTH = 0.1                              # λ1
LAMBDA_STERIC = 0.05                             # λ2
D_MIN = 0.5                                      # minimum hard-sphere diameter
EPS = 1e-3                                       # ε in 1 / (M_ij + ε)
LEARNING_RATE = 0.005                            # AdamW
HIDDEN_DIM = 32
IN_DIM = 2                                       # x_i = [f_GC, f_epi]
NUM_EPOCHS = 150
STERIC_SUBSAMPLE = 1000                          # pairs per step, |i-j| ∈ [3, 10)

# hg38 chr22 landmarks (UCSC cytoBand / assembly gaps)
P_ARM_GAP_END_BP = 10_510_000                    # acrocentric p-arm is unassembled (N) up to ~10.51 Mb
CENTROMERE_BP = (12_200_000, 17_400_000)         # p11.1 + q11.1 'acen' bands
TELOMERE_SPAN_BINS = 100                         # 1 Mb at each chromosome end

# Synthetic polymer bond length. 3.8 Å matches a Cα–Cα virtual bond so PyMOL /
# ChimeraX auto-draw a continuous trace when the exported PDB is opened.
BOND_LENGTH = 3.8

# Palette
C_BG = "#0e1117"
C_PANEL = "#161b22"
C_CARD = "#1f293d"
C_CYAN = "#00f2fe"
C_BLUE = "#4facfe"
C_AMBER = "#ffb454"
C_ROSE = "#ff5f87"
C_TEXT = "#e6edf3"
C_MUTED = "#8b949e"
C_GRID = "#263041"

COLORMAPS = ["Spectral", "Viridis", "Plasma", "Turbo"]
COLOR_BY = [
    "Genomic position (10 kb bin)",
    "GC content · f_GC",
    "H3K27ac signal · f_epi",
    "Radial depth (distance to centroid)",
]

PLOTLY_CONFIG = {
    "displaylogo": False,
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
    "toImageButtonOptions": {"format": "png", "scale": 2, "filename": "chronocell5d_render"},
}


# =============================================================================
# 2. THEME / CSS
# =============================================================================
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap');

:root {
  --bg: #0e1117; --panel: #161b22; --card: #1f293d; --line: #263041;
  --cyan: #00f2fe; --blue: #4facfe; --amber: #ffb454; --rose: #ff5f87;
  --text: #e6edf3; --muted: #8b949e;
  --mono: 'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
}

html, body, [data-testid="stAppViewContainer"], .stApp {
  background: radial-gradient(1200px 600px at 85% -10%, rgba(79,172,254,0.07), transparent 60%),
              radial-gradient(900px 500px at -10% 110%, rgba(0,242,254,0.05), transparent 60%),
              var(--bg) !important;
  color: var(--text);
  font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', sans-serif;
}
[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer, [data-testid="stDecoration"] { visibility: hidden; height: 0; }
.block-container { padding-top: 1.1rem; padding-bottom: 2.5rem; max-width: 1680px; }
h1, h2, h3, h4 { font-family: 'Inter', sans-serif; letter-spacing: -0.01em; color: var(--text); }
p, li, label, .stMarkdown { color: var(--text); }
code, pre, .stCode { font-family: var(--mono) !important; }

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #0a0d13 0%, #11161e 55%, #161b22 100%);
  border-right: 1px solid var(--line);
}
[data-testid="stSidebar"] .block-container { padding-top: 0.6rem; }
.brand { padding: 14px 14px 12px; margin: 0 0 12px; border-radius: 12px;
  background: linear-gradient(135deg, rgba(0,242,254,0.10), rgba(79,172,254,0.04));
  border: 1px solid rgba(0,242,254,0.25); }
.brand-title { font-weight: 800; font-size: 1.28rem; color: var(--text); letter-spacing: -0.02em; }
.brand-title .ver { font-family: var(--mono); font-size: 0.72rem; font-weight: 500; color: var(--cyan);
  border: 1px solid rgba(0,242,254,0.4); border-radius: 6px; padding: 1px 6px; margin-left: 6px; vertical-align: middle; }
.brand-sub { color: var(--muted); font-size: 0.74rem; margin-top: 3px; }
.status { display: inline-flex; align-items: center; gap: 8px; margin-top: 10px; padding: 5px 10px;
  border-radius: 999px; background: #0b0f15; border: 1px solid var(--line);
  font-family: var(--mono); font-size: 0.70rem; color: var(--text); }
.dot { width: 8px; height: 8px; border-radius: 50%; background: var(--cyan);
  box-shadow: 0 0 0 0 rgba(0,242,254,0.7); animation: pulse 1.8s infinite; }
.dot.warn { background: var(--amber); box-shadow: 0 0 0 0 rgba(255,180,84,0.7); animation: pulseAmber 1.8s infinite; }
@keyframes pulse { 0% { box-shadow: 0 0 0 0 rgba(0,242,254,0.65); } 70% { box-shadow: 0 0 0 8px rgba(0,242,254,0); } 100% { box-shadow: 0 0 0 0 rgba(0,242,254,0); } }
@keyframes pulseAmber { 0% { box-shadow: 0 0 0 0 rgba(255,180,84,0.65); } 70% { box-shadow: 0 0 0 8px rgba(255,180,84,0); } 100% { box-shadow: 0 0 0 0 rgba(255,180,84,0); } }
.side-h { font-family: var(--mono); font-size: 0.68rem; letter-spacing: 0.14em; text-transform: uppercase;
  color: var(--cyan); margin: 18px 0 6px; padding-bottom: 5px; border-bottom: 1px solid var(--line); }
.side-card { background: #0b0f15; border: 1px solid var(--line); border-radius: 10px; padding: 10px 12px;
  font-family: var(--mono); font-size: 0.72rem; color: var(--muted); line-height: 1.6; }
.side-card b { color: var(--text); font-weight: 500; }
.side-card .ok { color: var(--cyan); } .side-card .wn { color: var(--amber); }

/* ---------- Header ---------- */
.hero { display: flex; justify-content: space-between; align-items: flex-end; gap: 18px; flex-wrap: wrap;
  padding: 16px 20px; margin-bottom: 14px; border-radius: 14px;
  background: linear-gradient(120deg, rgba(22,27,34,0.95), rgba(31,41,61,0.85));
  border: 1px solid var(--line); position: relative; overflow: hidden; }
.hero:before { content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 3px;
  background: linear-gradient(180deg, var(--cyan), var(--blue)); }
.hero h1 { margin: 0; font-size: 1.55rem; font-weight: 800; }
.hero h1 .grad { background: linear-gradient(90deg, var(--cyan), var(--blue)); -webkit-background-clip: text;
  background-clip: text; color: transparent; }
.hero .tag { color: var(--muted); font-size: 0.86rem; margin-top: 4px; }
.chips { display: flex; gap: 6px; flex-wrap: wrap; }
.chip { font-family: var(--mono); font-size: 0.70rem; padding: 4px 9px; border-radius: 7px;
  background: #0b0f15; border: 1px solid var(--line); color: var(--muted); white-space: nowrap; }
.chip b { color: var(--text); font-weight: 500; }
.chip.hot { border-color: rgba(0,242,254,0.45); color: var(--cyan); }
.chip.warn { border-color: rgba(255,180,84,0.45); color: var(--amber); }

/* ---------- Tabs ---------- */
.stTabs [data-baseweb="tab-list"] { gap: 4px; background: var(--panel); padding: 6px; border-radius: 12px;
  border: 1px solid var(--line); }
.stTabs [data-baseweb="tab"] { height: 40px; border-radius: 8px; padding: 0 18px; color: var(--muted);
  font-weight: 600; font-size: 0.88rem; background: transparent; border: none; }
.stTabs [data-baseweb="tab"]:hover { color: var(--text); background: rgba(255,255,255,0.03); }
.stTabs [aria-selected="true"] { background: linear-gradient(135deg, rgba(0,242,254,0.16), rgba(79,172,254,0.12)) !important;
  color: var(--cyan) !important; box-shadow: inset 0 0 0 1px rgba(0,242,254,0.38); }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display: none; }
.stTabs [data-baseweb="tab-panel"] { padding-top: 14px; }

/* ---------- Panels, cards, charts ---------- */
[data-testid="stPlotlyChart"] { background: var(--panel); border: 1px solid var(--line); border-radius: 12px;
  padding: 6px; overflow: hidden; }
.sec { display: flex; align-items: baseline; gap: 10px; margin: 6px 0 8px; }
.sec .t { font-weight: 700; font-size: 1.02rem; color: var(--text); }
.sec .s { font-family: var(--mono); font-size: 0.72rem; color: var(--muted); }
.sec .n { font-family: var(--mono); font-size: 0.68rem; color: var(--cyan); border: 1px solid rgba(0,242,254,0.35);
  border-radius: 5px; padding: 0 6px; }

.kpi { background: linear-gradient(160deg, var(--card), #182133); border: 1px solid var(--line); border-radius: 12px;
  padding: 14px 16px 12px; position: relative; overflow: hidden; min-height: 118px; }
.kpi:after { content: ""; position: absolute; right: -30px; top: -30px; width: 90px; height: 90px; border-radius: 50%;
  background: radial-gradient(circle, rgba(0,242,254,0.14), transparent 70%); }
.kpi-label { font-family: var(--mono); font-size: 0.68rem; letter-spacing: 0.1em; text-transform: uppercase; color: var(--muted); }
.kpi-value { font-family: var(--mono); font-size: 1.72rem; font-weight: 700; color: var(--text); margin-top: 6px; line-height: 1.1; }
.kpi-unit { font-size: 0.85rem; color: var(--cyan); margin-left: 6px; font-weight: 500; }
.kpi-sub { font-family: var(--mono); font-size: 0.70rem; color: var(--muted); margin-top: 6px; }
.kpi-eq { font-family: var(--mono); font-size: 0.66rem; color: #5d6b82; margin-top: 3px; }

.mini { background: #121822; border: 1px solid var(--line); border-radius: 10px; padding: 9px 12px; }
.mini .l { font-family: var(--mono); font-size: 0.64rem; letter-spacing: 0.08em; text-transform: uppercase; color: var(--muted); }
.mini .v { font-family: var(--mono); font-size: 1.05rem; color: var(--text); margin-top: 2px; white-space: nowrap;
  overflow: hidden; text-overflow: ellipsis; }
.mini .v small { display: block; color: var(--muted); font-size: 0.68rem; margin-top: 1px; overflow: hidden; text-overflow: ellipsis; }

.insp { background: var(--panel); border: 1px solid var(--line); border-radius: 12px; padding: 12px 14px; margin-bottom: 10px; }
.insp .h { font-family: var(--mono); font-size: 0.66rem; letter-spacing: 0.12em; text-transform: uppercase; color: var(--cyan); margin-bottom: 8px; }
.row { display: flex; justify-content: space-between; gap: 12px; font-family: var(--mono); font-size: 0.74rem; padding: 3px 0;
  border-bottom: 1px dashed rgba(38,48,65,0.7); }
.row:last-child { border-bottom: none; }
.row .k { color: var(--muted); white-space: nowrap; } .row .v { color: var(--text); text-align: right; }

.eqhead { font-weight: 700; color: var(--text); font-size: 0.95rem; margin-top: 2px; }
.eqhead span { font-family: var(--mono); color: var(--cyan); font-size: 0.70rem; margin-left: 10px; font-weight: 500;
  border: 1px solid rgba(0,242,254,0.35); border-radius: 5px; padding: 1px 6px; }
.eqtext { color: var(--muted); font-size: 0.84rem; margin: 10px 0 0; line-height: 1.55; }
.katex { color: var(--text); font-size: 1.08em; }
.katex-display { margin: 0.5em 0 !important; overflow-x: auto; overflow-y: hidden; padding-bottom: 2px; }

.flow { display: flex; align-items: stretch; gap: 8px; flex-wrap: wrap; margin: 4px 0 14px; }
.stage { flex: 1 1 170px; background: var(--panel); border: 1px solid var(--line); border-radius: 12px; padding: 12px 13px; position: relative; }
.stage .k { font-family: var(--mono); font-size: 0.64rem; color: var(--cyan); letter-spacing: 0.12em; }
.stage .t { font-weight: 700; margin: 3px 0 5px; font-size: 0.92rem; }
.stage .d { font-family: var(--mono); font-size: 0.70rem; color: var(--muted); line-height: 1.55; }
.arrow { align-self: center; color: var(--cyan); font-size: 1.1rem; opacity: 0.8; }

.banner { display: block; line-height: 1.6; padding: 9px 14px; border-radius: 10px; margin-bottom: 12px;
  background: rgba(255,180,84,0.07); border: 1px solid rgba(255,180,84,0.35); font-size: 0.83rem; color: #f3d9b1; }
.banner.ok { background: rgba(0,242,254,0.06); border-color: rgba(0,242,254,0.35); color: #bff9ff; }
.banner code { background: rgba(0,0,0,0.3); padding: 1px 5px; border-radius: 4px; color: var(--text); }

.dl { background: var(--panel); border: 1px solid var(--line); border-radius: 12px; padding: 14px; margin-bottom: 8px; }
.dl .f { font-family: var(--mono); font-size: 0.86rem; color: var(--text); font-weight: 700; }
.dl .m { font-family: var(--mono); font-size: 0.70rem; color: var(--muted); margin: 4px 0 2px; line-height: 1.5; }

/* ---------- Widgets ---------- */
.stButton > button, .stDownloadButton > button {
  background: var(--card); color: var(--text); border: 1px solid var(--line); border-radius: 9px;
  font-weight: 600; font-size: 0.82rem; transition: all 0.15s ease; }
.stButton > button:hover, .stDownloadButton > button:hover { border-color: var(--cyan); color: var(--cyan);
  box-shadow: 0 0 0 1px rgba(0,242,254,0.25), 0 6px 18px rgba(0,242,254,0.08); }
.stDownloadButton > button { background: linear-gradient(135deg, rgba(0,242,254,0.14), rgba(79,172,254,0.10));
  border-color: rgba(0,242,254,0.4); }
[data-testid="stFileUploaderDropzone"] { background: #0b0f15; border: 1px dashed rgba(0,242,254,0.35); border-radius: 10px; }
[data-testid="stExpander"] { border: 1px solid var(--line); border-radius: 10px; background: var(--panel); }
[data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: 10px; }
hr { border-color: var(--line); }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# =============================================================================
# 3. STREAMLIT VERSION COMPATIBILITY
# =============================================================================
@functools.lru_cache(maxsize=None)
def stretch(element: str) -> dict[str, Any]:
    """Full-width kwarg for st.<element>: `width="stretch"` on new Streamlit, `use_container_width` on older."""
    try:
        params = inspect.signature(getattr(st, element)).parameters
    except (TypeError, ValueError):
        params = {}
    return {"width": "stretch"} if "width" in params else {"use_container_width": True}


def show_plot(fig: go.Figure, key: str) -> None:
    st.plotly_chart(fig, theme=None, config=PLOTLY_CONFIG, key=key, **stretch("plotly_chart"))


# =============================================================================
# 4. COMPUTE BACKEND DETECTION
# =============================================================================
@st.cache_resource(show_spinner=False)
def detect_backend() -> tuple[bool, str]:
    if TORCH_OK:
        try:
            if torch.cuda.is_available():
                return True, torch.cuda.get_device_name(0)
            mps = getattr(torch.backends, "mps", None)
            if mps is not None and mps.is_available():
                return True, "Apple MPS"
        except Exception:
            pass
        return False, f"CPU · PyTorch {torch.__version__}"
    return False, "CPU · NumPy backend"


# =============================================================================
# 5. SYNTHETIC DATA GENERATION (fallback so the UI is live before training ends)
# =============================================================================
def _gaussian_smooth(x: np.ndarray, sigma: float) -> np.ndarray:
    radius = max(1, int(3 * sigma))
    k = np.exp(-0.5 * (np.arange(-radius, radius + 1) / sigma) ** 2)
    k /= k.sum()
    padded = np.pad(x, radius, mode="reflect")
    return np.convolve(padded, k, mode="valid")


def _domain_partition(n: int, rng: np.random.Generator) -> np.ndarray:
    """TAD-like domain sizes in bins (log-normal, median ≈ 550 kb)."""
    sizes: list[int] = []
    total = 0
    while total < n:
        s = int(np.clip(rng.lognormal(np.log(55), 0.45), 15, 180))
        sizes.append(s)
        total += s
    overflow = total - n
    sizes[-1] -= overflow
    if sizes[-1] < 5 and len(sizes) > 1:
        sizes[-2] += sizes.pop()
    return np.asarray(sizes, dtype=int)


@st.cache_data(show_spinner=False)
def synthesize_dataset(n: int = N_BINS, seed: int = 22) -> dict[str, np.ndarray]:
    """
    Realistic polymer random-walk ribbon for chr22 at 10 kb resolution.

    Hierarchical model:
      1. Chain partitioned into TAD-like domains, each tagged A (active) / B (inactive) compartment.
      2. Domain centres follow a confined, persistent random walk (fractal-globule-like packing).
      3. The chain is threaded bead-by-bead with fixed bond length b: a persistent random step
         plus a harmonic pull toward the current domain centre (A domains looser than B).
      4. Light 3-point smoothing for a ribbon-like backbone, then bond lengths re-normalised to b.
    Tracks f_GC and f_epi are generated with compartment-coupled statistics
    (A: GC-rich, H3K27ac-peak-rich), including the unassembled hg38 p-arm gap.
    """
    rng = np.random.default_rng(seed)
    b = BOND_LENGTH
    sizes = _domain_partition(n, rng)
    n_dom = len(sizes)
    is_a = rng.random(n_dom) < 0.5

    # --- 2. domain centres ---------------------------------------------------
    r_glob = b * n ** (1 / 3) * 1.9
    radii = b * sizes ** (1 / 3) * np.where(is_a, 1.55, 1.15)
    centres = np.zeros((n_dom, 3))
    heading = rng.normal(size=3)
    heading /= np.linalg.norm(heading)
    c = rng.normal(size=3) * r_glob * 0.25
    for k in range(n_dom):
        if k > 0:
            step = rng.normal(size=3)
            step /= np.linalg.norm(step)
            direction = 0.6 * step + 0.4 * heading
            norm_c = np.linalg.norm(c)
            if norm_c > 1e-9:
                direction += 1.1 * (norm_c / r_glob) ** 2 * (-c / norm_c)
            direction /= np.linalg.norm(direction)
            heading = direction
            c = c + direction * (radii[k - 1] + radii[k]) * 0.85
        centres[k] = c

    # --- 3. thread the chain -------------------------------------------------
    noise = rng.normal(size=(n, 3))
    noise /= np.linalg.norm(noise, axis=1, keepdims=True)
    coords = np.empty((n, 3))
    pos = centres[0] + rng.normal(size=3) * radii[0] * 0.5
    head = noise[0].copy()
    idx = 0
    for k, size in enumerate(sizes):
        centre, r = centres[k], radii[k]
        kappa = 0.55 if is_a[k] else 0.95
        for _ in range(size):
            to_c = centre - pos
            dist = float(np.sqrt(to_c @ to_c)) + 1e-9
            direction = 0.55 * noise[idx] + 0.45 * head + kappa * (dist / r) ** 2 * (to_c / dist)
            direction /= np.sqrt(direction @ direction)
            pos = pos + b * direction
            head = direction
            coords[idx] = pos
            idx += 1

    # --- 4. ribbon smoothing + bond lengths ≈ b (±6 % thermal fluctuation) ---
    smooth = coords.copy()
    smooth[1:-1] = 0.25 * coords[:-2] + 0.5 * coords[1:-1] + 0.25 * coords[2:]
    bonds = np.diff(smooth, axis=0)
    lengths = b * rng.lognormal(0.0, 0.06, n - 1)
    bonds = bonds / np.linalg.norm(bonds, axis=1, keepdims=True) * lengths[:, None]
    coords = np.vstack([smooth[:1], smooth[:1] + np.cumsum(bonds, axis=0)])
    coords -= coords.mean(axis=0)

    # --- 5. 1D tracks --------------------------------------------------------
    comp_bins = np.repeat(is_a, sizes)[:n]
    gc = np.where(comp_bins, 0.505, 0.448)
    gc = gc + 0.028 * _gaussian_smooth(rng.normal(size=n), 12.0) / 0.12
    gc = gc + rng.normal(0.0, 0.010, n)
    gc = np.clip(gc, 0.30, 0.70)

    epi = rng.gamma(1.3, 0.22, n)
    p_peak = np.where(comp_bins, 0.055, 0.007) * (1.0 + 6.0 * np.clip(gc - 0.46, 0.0, None))
    peaks = rng.random(n) < p_peak
    epi = epi + np.convolve(peaks * rng.lognormal(1.25, 0.55, n), [0.25, 0.6, 1.0, 0.6, 0.25], mode="same")
    a_bins = np.flatnonzero(comp_bins)
    if a_bins.size:
        for centre_bin in rng.choice(a_bins, size=min(7, a_bins.size), replace=False):
            width = rng.uniform(3.0, 7.0)
            epi += rng.uniform(5.0, 9.0) * np.exp(-0.5 * ((np.arange(n) - centre_bin) / width) ** 2)

    if n >= int(0.9 * N_BINS):
        gap = np.arange(n) * RESOLUTION < P_ARM_GAP_END_BP
        gc[gap] = 0.0
        epi[gap] = 0.0

    return {
        "coords": coords.astype(np.float64),
        "gc": gc.astype(np.float64),
        "epi": epi.astype(np.float64),
        "compartment_a": comp_bins.astype(bool),
    }


@st.cache_data(show_spinner=False)
def simulated_training_log(epochs: int = NUM_EPOCHS, seed: int = 5) -> pd.DataFrame:
    """Representative loss trajectory for the EGNN AdamW loop (lr = 0.005, 150 epochs)."""
    rng = np.random.default_rng(seed)
    e = np.arange(1, epochs + 1, dtype=float)
    jitter = lambda s: np.exp(rng.normal(0.0, s, e.size))  # noqa: E731
    contact = (0.92 + 36.0 * np.exp(-e / 17.0) + 5.5 * np.exp(-e / 62.0)) * jitter(0.035)
    # torch.randn initialisation ⇒ E||p_{i+1} − p_i||² = 6 at epoch 0
    smooth = (0.34 + 5.6 * np.exp(-e / 24.0)) * jitter(0.025)
    # transient overlap as the chain collapses onto contact targets, then relaxes
    steric = (0.004 + 0.085 * np.exp(-(((e - 34.0) / 17.0) ** 2)) + 0.022 * np.exp(-e / 7.0)) * jitter(0.06)
    return pd.DataFrame({"epoch": e.astype(int), "L_contact": contact, "L_smooth": smooth, "L_steric": steric})


# =============================================================================
# 6. FILE PARSERS (predicted_coords.pt / .pdb / .xyz / .npy / .csv, graph_data.pt)
# =============================================================================
_COORD_KEYS = ("pos", "coords", "predicted_coords", "pos_out", "positions", "xyz", "P")


def _torch_load(data: bytes) -> Any:
    if not TORCH_OK:
        raise ValueError(
            "PyTorch is not installed in this environment. Run `pip install torch`, "
            "or export the tensor as .xyz / .pdb / .npy."
        )
    buf = io.BytesIO(data)
    try:
        return torch.load(buf, map_location="cpu", weights_only=True)
    except Exception:
        # PyG `Data` objects need full unpickling. Only upload files your own pipeline produced.
        buf.seek(0)
        try:
            return torch.load(buf, map_location="cpu", weights_only=False)
        except TypeError:
            buf.seek(0)
            return torch.load(buf, map_location="cpu")


def _to_numpy(obj: Any) -> np.ndarray:
    if TORCH_OK and isinstance(obj, torch.Tensor):
        return obj.detach().cpu().double().numpy()
    return np.asarray(obj, dtype=np.float64)


def _as_coords(arr: Any) -> np.ndarray:
    a = _to_numpy(arr)
    a = np.squeeze(a)
    if a.ndim == 3 and a.shape[0] == 1:
        a = a[0]
    if a.ndim == 2 and a.shape[1] != 3 and a.shape[0] == 3:
        a = a.T
    if a.ndim != 2 or a.shape[1] < 3:
        raise ValueError(f"Expected an (N, 3) coordinate array, got shape {tuple(a.shape)}.")
    a = a[:, :3].astype(np.float64)
    if a.shape[0] < 4:
        raise ValueError("Structure must contain at least 4 nodes.")
    if not np.all(np.isfinite(a)):
        raise ValueError("Coordinates contain NaN or infinite values.")
    return a


def _extract_coords_from_object(obj: Any) -> np.ndarray:
    if TORCH_OK and isinstance(obj, torch.Tensor) or isinstance(obj, np.ndarray):
        return _as_coords(obj)
    if isinstance(obj, dict):
        for key in _COORD_KEYS:
            if key in obj and obj[key] is not None:
                return _as_coords(obj[key])
        for value in obj.values():
            try:
                return _as_coords(value)
            except Exception:
                continue
        raise ValueError(f"No (N, 3) tensor found in dict keys {list(obj.keys())[:8]}.")
    for key in _COORD_KEYS:
        value = getattr(obj, key, None)
        if value is not None:
            return _as_coords(value)
    if isinstance(obj, (list, tuple)):
        return _as_coords(obj)
    raise ValueError(f"Unsupported object type in .pt file: {type(obj).__name__}.")


def _parse_pdb(text: str) -> np.ndarray:
    pts: list[tuple[float, float, float]] = []
    for line in text.splitlines():
        if line.startswith("ENDMDL") and pts:
            break
        if line.startswith("ATOM") or line.startswith("HETATM"):
            try:
                pts.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
            except ValueError:
                parts = line.split()
                if len(parts) >= 9:
                    pts.append((float(parts[6]), float(parts[7]), float(parts[8])))
    if not pts:
        raise ValueError("No ATOM/HETATM records found in PDB file.")
    return _as_coords(np.asarray(pts))


def _parse_xyz(text: str) -> np.ndarray:
    lines = [ln for ln in text.splitlines() if ln.strip()]
    start = 0
    try:
        int(lines[0].split()[0])
        start = 2
    except (ValueError, IndexError):
        start = 0
    pts: list[list[float]] = []
    for ln in lines[start:]:
        p = ln.split()
        try:
            float(p[0])
            first_numeric = True
        except ValueError:
            first_numeric = False
        try:
            if not first_numeric and len(p) >= 4:
                pts.append([float(p[1]), float(p[2]), float(p[3])])
            elif first_numeric and len(p) == 3:
                pts.append([float(p[0]), float(p[1]), float(p[2])])
            elif first_numeric and len(p) >= 4:
                pts.append([float(p[1]), float(p[2]), float(p[3])])
        except ValueError:
            continue
    if not pts:
        raise ValueError("No coordinate lines found in XYZ file.")
    return _as_coords(np.asarray(pts))


def _parse_csv_numeric(data: bytes, n_cols: int, preferred: tuple[str, ...]) -> np.ndarray:
    df = pd.read_csv(io.BytesIO(data))
    lower = {c.lower().strip(): c for c in df.columns}
    if all(p in lower for p in preferred):
        return df[[lower[p] for p in preferred]].to_numpy(dtype=np.float64)
    numeric = df.select_dtypes(include=[np.number])
    if numeric.shape[1] < n_cols:
        raise ValueError(f"CSV needs at least {n_cols} numeric columns.")
    return numeric.iloc[:, :n_cols].to_numpy(dtype=np.float64)


@st.cache_data(show_spinner=False)
def parse_structure(data: bytes, name: str) -> np.ndarray:
    ext = os.path.splitext(name.lower())[1]
    if ext in (".pt", ".pth"):
        return _extract_coords_from_object(_torch_load(data))
    if ext == ".npy":
        return _as_coords(np.load(io.BytesIO(data), allow_pickle=False))
    if ext == ".pdb":
        return _parse_pdb(data.decode("utf-8", errors="replace"))
    if ext == ".xyz":
        return _parse_xyz(data.decode("utf-8", errors="replace"))
    if ext == ".csv":
        return _as_coords(_parse_csv_numeric(data, 3, ("x", "y", "z")))
    raise ValueError(f"Unsupported file type '{ext}'.")


@st.cache_data(show_spinner=False)
def parse_graph(data: bytes, name: str) -> dict[str, np.ndarray]:
    """Node features x = [f_GC, f_epi] (+ optional Micro-C edges) from graph_data.pt / .npy / .csv."""
    ext = os.path.splitext(name.lower())[1]
    out: dict[str, np.ndarray] = {}
    if ext in (".pt", ".pth"):
        obj = _torch_load(data)
        get = (lambda k: obj.get(k)) if isinstance(obj, dict) else (lambda k: getattr(obj, k, None))
        x = get("x")
        if x is None and (isinstance(obj, np.ndarray) or (TORCH_OK and isinstance(obj, torch.Tensor))):
            x = obj
        if x is None:
            raise ValueError("No node feature matrix `x` found in the uploaded graph file.")
        out["x"] = _to_numpy(x)
        ei, ea = get("edge_index"), get("edge_attr")
        if ei is not None:
            out["edge_index"] = _to_numpy(ei).astype(np.int64)
            if ea is not None:
                out["edge_attr"] = _to_numpy(ea).reshape(-1)
    elif ext == ".npy":
        out["x"] = np.load(io.BytesIO(data), allow_pickle=False).astype(np.float64)
    elif ext == ".csv":
        out["x"] = _parse_csv_numeric(data, 2, ("f_gc", "f_epi"))
    else:
        raise ValueError(f"Unsupported feature file type '{ext}'.")
    x = np.atleast_2d(out["x"])
    if x.shape[1] < 2 and x.shape[0] == 2:
        x = x.T
    if x.shape[1] < 2:
        raise ValueError(f"Node features must be (N, 2) = [f_GC, f_epi], got {x.shape}.")
    out["x"] = x[:, :2].astype(np.float64)
    return out


# =============================================================================
# 7. POLYMER PHYSICS & ANALYTICS
# =============================================================================
def polymer_metrics(p: np.ndarray) -> dict[str, float]:
    centroid = p.mean(axis=0)
    rg = float(np.sqrt(np.mean(np.sum((p - centroid) ** 2, axis=1))))
    re = float(np.linalg.norm(p[-1] - p[0]))
    bonds = np.linalg.norm(np.diff(p, axis=0), axis=1)
    mean_bond = float(bonds.mean())
    return {
        "rg": rg,
        "re": re,
        "mean_bond": mean_bond,
        "bond_cv": float(bonds.std() / mean_bond) if mean_bond > 0 else 0.0,
        "contour": float(bonds.sum()),
        "re2_rg2": (re**2) / (rg**2) if rg > 0 else float("nan"),
        "l_smooth": float(np.mean(bonds**2)),  # (1/(N-1)) Σ ||p_{i+1} − p_i||²
        "extent": float(np.max(np.ptp(p, axis=0))),
    }


@st.cache_data(show_spinner=False)
def steric_analysis(p: np.ndarray, d_min: float, seed: int = 0) -> dict[str, float]:
    """L_steric over Ω = {(i,j) : |i−j| > 2}; exact for ≤ 2,500 nodes, Monte-Carlo estimate above."""
    n = len(p)
    if n < 4:
        return {"l_steric": 0.0, "violations": 0.0, "pairs": 0.0, "exact": True, "min_dist": float("nan")}
    if n <= 2500:
        total_sq, viol, pairs, dmin_obs = 0.0, 0, 0, np.inf
        chunk = 256
        for s in range(0, n, chunk):
            blk = p[s : s + chunk]
            d = np.sqrt(((blk[:, None, :] - p[None, :, :]) ** 2).sum(-1))
            ii = np.arange(s, s + len(blk))[:, None]
            jj = np.arange(n)[None, :]
            mask = (jj - ii) > 2  # upper triangle, |i−j| > 2
            dv = d[mask]
            if dv.size:
                pen = np.clip(d_min - dv, 0.0, None)
                total_sq += float((pen**2).sum())
                viol += int((dv < d_min).sum())
                pairs += int(dv.size)
                dmin_obs = min(dmin_obs, float(dv.min()))
        return {
            "l_steric": total_sq / max(pairs, 1),
            "violations": float(viol),
            "pairs": float(pairs),
            "exact": True,
            "min_dist": dmin_obs,
        }
    rng = np.random.default_rng(seed)
    m = 400_000
    i = rng.integers(0, n, m)
    j = rng.integers(0, n, m)
    keep = np.abs(i - j) > 2
    i, j = i[keep], j[keep]
    dv = np.linalg.norm(p[i] - p[j], axis=1)
    pen = np.clip(d_min - dv, 0.0, None)
    total_pairs = (n - 2) * (n - 3) / 2
    frac = float((dv < d_min).mean())
    return {
        "l_steric": float((pen**2).mean()),
        "violations": frac * total_pairs,
        "pairs": float(total_pairs),
        "exact": False,
        "min_dist": float(dv.min()),
    }


def contact_loss(p: np.ndarray, edge_index: np.ndarray, edge_attr: np.ndarray) -> float:
    """L_contact = (1/|E|) Σ ( ||p_i − p_j||₂ − 1/(M_ij + ε) )²."""
    row, col = edge_index[0], edge_index[1]
    valid = (row < len(p)) & (col < len(p)) & (row >= 0) & (col >= 0)
    row, col, m = row[valid], col[valid], edge_attr[valid]
    if row.size == 0:
        return float("nan")
    d = np.linalg.norm(p[row] - p[col], axis=1)
    return float(np.mean((d - 1.0 / (m + EPS)) ** 2))


@st.cache_data(show_spinner=False)
def coarse_distance_matrix(p: np.ndarray, max_bins: int = 600) -> tuple[np.ndarray, int]:
    """Pairwise Euclidean distance map; windows > max_bins are coarse-grained by block-averaging beads."""
    n = len(p)
    k = max(1, math.ceil(n / max_bins))
    if k > 1:
        m = (n // k) * k
        pooled = p[:m].reshape(-1, k, 3).mean(axis=1)
        if m < n:
            pooled = np.vstack([pooled, p[m:].mean(axis=0, keepdims=True)])
    else:
        pooled = p
    diff = pooled[:, None, :] - pooled[None, :, :]
    return np.sqrt((diff**2).sum(-1)).astype(np.float32), k


@st.cache_data(show_spinner=False)
def distance_scaling(p: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """⟨R(s)⟩ vs genomic separation s and fitted exponent ν in R(s) ∝ s^ν."""
    n = len(p)
    s_max = max(2, n // 2)
    s_vals = np.unique(np.geomspace(1, s_max, 42).astype(int))
    r = np.array([np.linalg.norm(p[s:] - p[:-s], axis=1).mean() for s in s_vals])
    lo, hi = 4, max(10, n // 8)
    sel = (s_vals >= lo) & (s_vals <= hi)
    if sel.sum() >= 3:
        nu = float(np.polyfit(np.log(s_vals[sel]), np.log(r[sel]), 1)[0])
    else:
        nu = float("nan")
    return s_vals, r, nu


@st.cache_data(show_spinner=False)
def enhancer_clusters(epi: np.ndarray, pct: float = 97.0, gap: int = 2, min_len: int = 2) -> list[tuple[int, int, float]]:
    """Contiguous runs of top-percentile H3K27ac bins → (start_bin, end_bin_exclusive, peak_signal)."""
    valid = epi > 0
    if not valid.any():
        return []
    sm = np.convolve(epi, np.ones(5) / 5.0, mode="same")
    thr = np.percentile(sm[valid], pct)
    hot = np.flatnonzero(sm >= thr)
    if hot.size == 0:
        return []
    groups = np.split(hot, np.flatnonzero(np.diff(hot) > gap + 1) + 1)
    out = [(int(g[0]), int(g[-1]) + 1, float(epi[g[0] : g[-1] + 1].max())) for g in groups if len(g) >= min_len]
    return sorted(out, key=lambda t: -t[2])


def catmull_rom(points: np.ndarray, subdiv: int = 4) -> np.ndarray:
    """Centripetal-free uniform Catmull-Rom spline through all beads (for the tubular backbone)."""
    if len(points) < 4 or subdiv <= 1:
        return points
    p = np.vstack([points[:1], points, points[-1:]])
    p0, p1, p2, p3 = p[:-3], p[1:-2], p[2:-1], p[3:]
    t = np.linspace(0.0, 1.0, subdiv, endpoint=False)[None, :, None]
    t2, t3 = t * t, t * t * t
    seg = 0.5 * (
        2 * p1[:, None, :]
        + (-p0 + p2)[:, None, :] * t
        + (2 * p0 - 5 * p1 + 4 * p2 - p3)[:, None, :] * t2
        + (-p0 + 3 * p1 - 3 * p2 + p3)[:, None, :] * t3
    )
    return np.vstack([seg.reshape(-1, 3), points[-1:]])


def egnn_parameter_count(in_dim: int = IN_DIM, hidden: int = HIDDEN_DIM) -> dict[str, int]:
    lin = lambda i, o, bias=True: i * o + (o if bias else 0)  # noqa: E731
    msg = lin(in_dim * 2 + 2, hidden) + lin(hidden, hidden)
    coord = lin(hidden, hidden) + lin(hidden, 1, bias=False)
    node = lin(in_dim + hidden, hidden) + lin(hidden, in_dim)
    return {"MLP_m": msg, "MLP_x": coord, "MLP_h": node, "total": msg + coord + node}


# =============================================================================
# 8. EXPORT BUILDERS
# =============================================================================
@st.cache_data(show_spinner=False)
def build_pdb(
    p: np.ndarray,
    start_bin: int,
    gc: np.ndarray,
    epi: np.ndarray,
    encode_tracks: bool,
    source_label: str,
) -> tuple[str, float]:
    """
    Standard PDB text: HEADER / TITLE / REMARK, one Cα ATOM record per 10 kb bin
    (residue 'GNN', chain A, resSeq = bin index + 1), sequential CONECT backbone bonds, END.
    Returns (pdb_text, scale_factor_applied).
    """
    lo, hi = float(p.min()), float(p.max())
    scale = 1.0
    if hi > 9999.999 or lo < -999.999:
        scale = min(9999.999 / hi if hi > 0 else 1.0, 999.999 / -lo if lo < 0 else 1.0) * 0.999
    q = p * scale
    n = len(q)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%d-%b-%y").upper()
    end_bin = start_bin + n
    g_start = start_bin * RESOLUTION
    g_end = min(end_bin * RESOLUTION, CHROM_SIZE)

    lines = [
        f"{'HEADER    CHROMATIN STRUCTURE':<50}{stamp:<9}   CC5D",
        "TITLE     CHRONOCELL-5D CHROMOSOME 22 MODEL",
        "TITLE    2 SE(3)-EQUIVARIANT GRAPH NEURAL NETWORK RECONSTRUCTION",
        f"REMARK   1 SOURCE: {source_label[:58].upper()}",
        f"REMARK   2 RESOLUTION. {RESOLUTION // 1000} KB PER BEAD (ONE CA ATOM PER GENOMIC BIN).",
        f"REMARK   3 ASSEMBLY {ASSEMBLY}  REGION {CHROM_NAME}:{g_start + 1}-{g_end}",
        f"REMARK   3 BINS {start_bin}-{end_bin - 1}  N = {n}  RESSEQ = BIN INDEX + 1",
        f"REMARK   3 LOSS: L_CONTACT + {LAMBDA_SMOOTH} L_SMOOTH + {LAMBDA_STERIC} L_STERIC (D_MIN {D_MIN})",
    ]
    if encode_tracks:
        lines.append("REMARK   4 OCCUPANCY = F_GC (GC FRACTION); B-FACTOR = F_EPI SCALED TO 0-99.99 (H3K27AC)")
    if scale != 1.0:
        lines.append(f"REMARK   5 COORDINATES RESCALED BY {scale:.6f} TO FIT PDB COLUMN WIDTHS")

    idx = np.arange(start_bin, end_bin)
    if encode_tracks:
        occ = np.clip(gc[idx], 0.0, 1.0)
        e = epi[idx]
        emax = float(e.max()) if e.size and e.max() > 0 else 1.0
        bfac = np.clip(e / emax * 99.99, 0.0, 99.99)
    else:
        occ = np.ones(n)
        bfac = np.full(n, 20.0)

    for k in range(n):
        serial = (k + 1) % 100000
        res_seq = (start_bin + k + 1) % 10000
        x, y, z = q[k]
        lines.append(
            f"ATOM  {serial:5d}  CA  GNN A{res_seq:4d}    {x:8.3f}{y:8.3f}{z:8.3f}{occ[k]:6.2f}{bfac[k]:6.2f}           C"
        )
    lines.append(f"TER   {(n + 1) % 100000:5d}      GNN A{(start_bin + n) % 10000:4d}")
    for k in range(1, n):
        lines.append(f"CONECT{k % 100000:5d}{(k + 1) % 100000:5d}")
    lines.append("END")
    return "\n".join(lines) + "\n", scale


@st.cache_data(show_spinner=False)
def build_xyz(p: np.ndarray, start_bin: int) -> str:
    n = len(p)
    header = [
        f"{n}",
        f"ChronoCell-5D Predicted Spatial Backbone - Chromosome 22 | bins {start_bin}-{start_bin + n - 1} | 10 kb/bead",
    ]
    body = [f"C  {x:10.4f}  {y:10.4f}  {z:10.4f}" for x, y, z in p]
    return "\n".join(header + body) + "\n"


@st.cache_data(show_spinner=False)
def build_csv(p: np.ndarray, start_bin: int, gc: np.ndarray, epi: np.ndarray) -> str:
    idx = np.arange(start_bin, start_bin + len(p))
    df = pd.DataFrame(
        {
            "bin": idx,
            "chrom": CHROM_NAME,
            "start_bp": idx * RESOLUTION,
            "end_bp": np.minimum((idx + 1) * RESOLUTION, CHROM_SIZE),
            "x": p[:, 0],
            "y": p[:, 1],
            "z": p[:, 2],
            "f_gc": gc[idx],
            "f_epi": epi[idx],
        }
    )
    return df.to_csv(index=False, float_format="%.5f")


# =============================================================================
# 9. PLOTLY FIGURE BUILDERS
# =============================================================================
def dark_layout(height: int = 420, **kw: Any) -> dict[str, Any]:
    base = dict(
        height=height,
        paper_bgcolor=C_PANEL,
        plot_bgcolor="#11161e",
        font=dict(family="JetBrains Mono, Consolas, monospace", color=C_TEXT, size=11),
        margin=dict(l=56, r=24, t=46, b=46),
        hoverlabel=dict(bgcolor=C_CARD, bordercolor=C_CYAN, font=dict(family="JetBrains Mono, monospace", color=C_TEXT)),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=C_GRID, font=dict(size=10)),
        title=dict(font=dict(family="Inter, sans-serif", size=14, color=C_TEXT), x=0.01, xanchor="left"),
    )
    for key in ("title", "legend", "hoverlabel"):
        if key in kw:
            base[key] = {**base[key], **kw.pop(key)}
    base.update(kw)
    return base


def axis_style(**kw: Any) -> dict[str, Any]:
    a = dict(gridcolor=C_GRID, zerolinecolor=C_GRID, linecolor=C_GRID, tickfont=dict(color=C_MUTED, size=10),
             title_font=dict(color=C_MUTED, size=11))
    a.update(kw)
    return a


def region_bands(lo: int, hi: int) -> list[tuple[float, float, str, str]]:
    """Annotated genomic landmarks (Mb) intersecting the window."""
    out = []
    lo_mb, hi_mb = lo * RESOLUTION / 1e6, hi * RESOLUTION / 1e6
    for a, b, label, color in (
        (0.0, P_ARM_GAP_END_BP / 1e6, "p-arm N-gap", "rgba(139,148,158,0.10)"),
        (CENTROMERE_BP[0] / 1e6, CENTROMERE_BP[1] / 1e6, "centromere", "rgba(255,95,135,0.10)"),
    ):
        a2, b2 = max(a, lo_mb), min(b, hi_mb)
        if b2 > a2:
            out.append((a2, b2, label, color))
    return out


def build_viewport(
    coords: np.ndarray,
    gc: np.ndarray,
    epi: np.ndarray,
    lo: int,
    hi: int,
    focus: list[tuple[int, int]] | None,
    opts: dict[str, Any],
) -> go.Figure:
    n = len(coords)
    idx = np.arange(lo, hi)
    sub = coords[lo:hi]
    cmap = opts["cmap"] + ("_r" if opts["reverse"] else "")

    color_by = opts["color_by"]
    if color_by.startswith("GC"):
        cvals, ctitle = gc[idx], "f<sub>GC</sub>"
    elif color_by.startswith("H3K27ac"):
        cvals, ctitle = epi[idx], "f<sub>epi</sub><br>H3K27ac"
    elif color_by.startswith("Radial"):
        cvals, ctitle = np.linalg.norm(sub - coords.mean(axis=0), axis=1), "r (Å)"
    else:
        cvals, ctitle = idx.astype(float), "Genomic bin<br>(10 kb)"

    focus_mask = np.zeros(len(idx), dtype=bool)
    if focus:
        for a, b in focus:
            a2, b2 = max(a, lo), min(b, hi)
            if b2 > a2:
                focus_mask[a2 - lo : b2 - lo] = True
    has_focus = bool(focus_mask.any())

    fig = go.Figure()

    if opts["ghost"] and (lo > 0 or hi < n):
        step = max(1, n // 2500)
        g = coords[::step]
        fig.add_trace(go.Scatter3d(
            x=g[:, 0], y=g[:, 1], z=g[:, 2], mode="lines",
            line=dict(color="rgba(139,148,158,0.20)", width=2),
            hoverinfo="skip", name="Chromosome context",
        ))

    if opts["backbone"] and len(sub) > 1:
        if opts["spline"] and len(sub) <= 2500:
            pts = catmull_rom(sub, 4)
            t = np.linspace(0, len(sub) - 1, len(pts))
            lcol = np.interp(t, np.arange(len(sub)), cvals)
        else:
            pts, lcol = sub, cvals
        fig.add_trace(go.Scatter3d(
            x=pts[:, 0], y=pts[:, 1], z=pts[:, 2], mode="lines",
            line=dict(color=lcol, colorscale=cmap, width=opts["tube"], cmin=float(np.min(cvals)), cmax=float(np.max(cvals))),
            opacity=0.45 if has_focus else 0.95,
            hoverinfo="skip", name="Backbone (tubular spline)" if opts["spline"] else "Backbone",
        ))

    start_mb = idx * RESOLUTION / 1e6
    end_mb = np.minimum((idx + 1) * RESOLUTION, CHROM_SIZE) / 1e6
    custom = np.column_stack([idx, start_mb, end_mb, gc[idx], epi[idx]])
    hover = (
        "<b>Bin %{customdata[0]:,.0f}</b>  ·  chr22:%{customdata[1]:.2f}–%{customdata[2]:.2f} Mb<br>"
        "x %{x:.2f}  y %{y:.2f}  z %{z:.2f} Å<br>"
        "f<sub>GC</sub> %{customdata[3]:.3f}   f<sub>epi</sub> %{customdata[4]:.2f}<extra></extra>"
    )
    fig.add_trace(go.Scatter3d(
        x=sub[:, 0], y=sub[:, 1], z=sub[:, 2], mode="markers",
        marker=dict(
            size=opts["size"], color=cvals, colorscale=cmap, opacity=0.35 if has_focus else 0.95,
            line=dict(width=0),
            colorbar=dict(
                title=dict(text=ctitle, font=dict(color=C_MUTED, size=11)), thickness=12, len=0.62, x=1.0,
                tickfont=dict(color=C_MUTED, size=10), outlinewidth=0, bgcolor="rgba(0,0,0,0)",
            ),
        ),
        customdata=custom, hovertemplate=hover, name="10 kb nodes",
    ))

    if has_focus:
        f = np.flatnonzero(focus_mask)
        fp = sub[f]
        fig.add_trace(go.Scatter3d(
            x=fp[:, 0], y=fp[:, 1], z=fp[:, 2], mode="markers",
            marker=dict(size=opts["size"] * 2.0 + 2, color=C_CYAN, opacity=0.95,
                        line=dict(color="#ffffff", width=0.5)),
            customdata=custom[f], hovertemplate=hover, name=opts.get("focus_label", "Highlighted region"),
        ))

    ends = np.vstack([sub[0], sub[-1]])
    fig.add_trace(go.Scatter3d(
        x=ends[:, 0], y=ends[:, 1], z=ends[:, 2], mode="markers+text",
        marker=dict(size=opts["size"] + 5, color=[C_AMBER, C_ROSE], symbol="diamond", line=dict(color="#fff", width=1)),
        text=[f"5′ · bin {lo}", f"3′ · bin {hi - 1}"], textposition="top center",
        textfont=dict(color=C_TEXT, size=11, family="JetBrains Mono, monospace"),
        hoverinfo="text", name="Window ends",
    ))

    scene_axis = dict(
        backgroundcolor="#0b0f15", gridcolor=C_GRID, zerolinecolor=C_GRID, showbackground=True,
        tickfont=dict(color=C_MUTED, size=9), title_font=dict(color=C_MUTED, size=11), showspikes=False,
    )
    fig.update_layout(**dark_layout(
        height=opts["height"],
        margin=dict(l=0, r=0, t=8, b=0),
        paper_bgcolor=C_PANEL,
        uirevision="chronocell-viewport",
        showlegend=True,
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(11,15,21,0.75)", bordercolor=C_GRID, borderwidth=1, font=dict(size=10)),
        scene=dict(
            xaxis=dict(title="X (Å)", **scene_axis),
            yaxis=dict(title="Y (Å)", **scene_axis),
            zaxis=dict(title="Z (Å)", **scene_axis),
            aspectmode="cube" if opts["cube"] else "data",
            bgcolor=C_PANEL,
            camera=dict(eye=dict(x=1.45, y=1.35, z=0.95)),
        ),
    ))
    return fig


def build_distance_heatmap(p: np.ndarray, lo: int, mode: str, cmap: str) -> go.Figure:
    d, k = coarse_distance_matrix(p)
    m = d.shape[0]
    centres_mb = (lo + (np.arange(m) * k + k / 2.0)) * RESOLUTION / 1e6
    if mode.startswith("Inferred"):
        with np.errstate(divide="ignore", invalid="ignore"):
            z = 1.0 / d - EPS
        np.fill_diagonal(z, np.nan)
        z = np.log10(np.clip(z, 1e-6, None))
        title_cb, scale = "log₁₀ M̂", cmap
        hover = "chr22 %{x:.2f} × %{y:.2f} Mb<br>log₁₀ M̂ = %{z:.3f}<extra></extra>"
    else:
        z = d
        title_cb, scale = "d (Å)", cmap
        hover = "chr22 %{x:.2f} × %{y:.2f} Mb<br>d = %{z:.2f} Å<extra></extra>"
    fig = go.Figure(go.Heatmap(
        z=z, x=centres_mb, y=centres_mb, colorscale=scale, hovertemplate=hover,
        colorbar=dict(title=dict(text=title_cb, font=dict(color=C_MUTED, size=11)), thickness=11,
                      tickfont=dict(color=C_MUTED, size=10), outlinewidth=0),
    ))
    res_label = f"{k * RESOLUTION // 1000} kb" if k > 1 else "10 kb (native)"
    fig.update_layout(**dark_layout(
        height=540, margin=dict(l=60, r=24, t=46, b=60),
        title=dict(text=f"Pairwise distance map · {m}×{m} · {res_label} pixels"),
        xaxis=axis_style(title="Genomic position (Mb)", constrain="domain"),
        yaxis=axis_style(title="Genomic position (Mb)", autorange="reversed", scaleanchor="x", constrain="domain"),
    ))
    return fig


def build_feature_overlay(gc: np.ndarray, epi: np.ndarray, lo: int, hi: int, smooth_w: int,
                          clusters: list[tuple[int, int, float]], track_label: str) -> go.Figure:
    idx = np.arange(lo, hi)
    x_mb = idx * RESOLUTION / 1e6
    g, e = gc[idx].copy(), epi[idx].copy()
    if smooth_w > 1 and len(idx) > smooth_w:
        kern = np.ones(smooth_w) / smooth_w
        g = np.convolve(g, kern, mode="same")
        e = np.convolve(e, kern, mode="same")
    no_seq = gc[idx] <= 0  # assembly gap (all-N): no sequence, so break the lines instead of plotting zeros
    g[no_seq] = np.nan
    e[no_seq] = np.nan
    g_valid = g[np.isfinite(g)]
    g_lo = max(0.0, float(g_valid.min()) - 0.04) if g_valid.size else 0.0
    g_hi = min(1.0, float(g_valid.max()) + 0.08) if g_valid.size else 1.0

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(
        x=x_mb, y=e, name="H3K27ac · f<sub>epi</sub>", mode="lines",
        line=dict(color=C_AMBER, width=1.2), fill="tozeroy", fillcolor="rgba(255,180,84,0.16)",
        hovertemplate="%{x:.2f} Mb<br>f<sub>epi</sub> %{y:.3f}<extra></extra>",
    ), secondary_y=True)
    fig.add_trace(go.Scatter(
        x=x_mb, y=g, name="GC fraction · f<sub>GC</sub>", mode="lines",
        line=dict(color=C_CYAN, width=1.6),
        hovertemplate="%{x:.2f} Mb<br>f<sub>GC</sub> %{y:.3f}<extra></extra>",
    ), secondary_y=False)

    for a, b, label, color in region_bands(lo, hi):
        fig.add_vrect(x0=a, x1=b, fillcolor=color, line_width=0, layer="below",
                      annotation_text=label, annotation_position="top left",
                      annotation_font=dict(color=C_MUTED, size=10))
    in_win = [(a, b) for a, b, _ in clusters if b > lo and a < hi]
    if in_win:
        cx = [((a + b) / 2) * RESOLUTION / 1e6 for a, b in in_win]
        fig.add_trace(go.Scatter(
            x=cx, y=[g_hi - 0.015] * len(cx), mode="markers", name="Enhancer cluster",
            marker=dict(symbol="triangle-down", size=9, color=C_ROSE),
            hovertemplate="Enhancer cluster @ %{x:.2f} Mb<extra></extra>",
        ), secondary_y=False)

    fig.update_layout(**dark_layout(
        height=420,
        margin=dict(l=56, r=24, t=46, b=90),
        title=dict(text=f"1D node features x<sub>i</sub> = [f<sub>GC</sub>, f<sub>epi</sub>] · {track_label}"),
        legend=dict(orientation="h", y=-0.24, x=0.0, xanchor="left", yanchor="top", bgcolor="rgba(0,0,0,0)"),
        hovermode="x unified",
    ))
    fig.update_xaxes(**axis_style(title="chr22 position (Mb)"))
    fig.update_yaxes(**axis_style(title="f<sub>GC</sub>", range=[g_lo, g_hi]), secondary_y=False)
    fig.update_yaxes(**axis_style(title="f<sub>epi</sub> (mean signal / bp)", showgrid=False), secondary_y=True)
    return fig


def build_scaling_plot(p: np.ndarray) -> tuple[go.Figure, float]:
    s, r, nu = distance_scaling(p)
    s_kb = s * RESOLUTION / 1000
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=s_kb, y=r, mode="lines+markers", name="⟨R(s)⟩ model",
        line=dict(color=C_CYAN, width=2), marker=dict(size=5, color=C_CYAN),
        hovertemplate="s = %{x:,.0f} kb<br>⟨R⟩ = %{y:.2f} Å<extra></extra>",
    ))
    anchor = int(np.argmin(np.abs(s - 4)))
    for exp_, label, color, dash in ((1 / 3, "ν = 1/3 fractal globule", C_ROSE, "dot"),
                                     (1 / 2, "ν = 1/2 ideal chain", C_AMBER, "dash"),
                                     (3 / 5, "ν = 3/5 swollen coil", C_MUTED, "dashdot")):
        ref = r[anchor] * (s / s[anchor]) ** exp_
        fig.add_trace(go.Scatter(x=s_kb, y=ref, mode="lines", name=label,
                                 line=dict(color=color, width=1.2, dash=dash), hoverinfo="skip"))
    fig.update_layout(**dark_layout(
        height=380,
        title=dict(text=f"Distance scaling ⟨R(s)⟩ ∝ s<sup>ν</sup> · fitted ν = {nu:.3f}"),
        xaxis=axis_style(title="Genomic separation s (kb)", type="log"),
        yaxis=axis_style(title="Mean spatial distance (Å)", type="log"),
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(11,15,21,0.6)"),
    ))
    return fig, nu


def build_bond_histogram(p: np.ndarray) -> go.Figure:
    bonds = np.linalg.norm(np.diff(p, axis=0), axis=1)
    fig = px.histogram(pd.DataFrame({"bond": bonds}), x="bond", nbins=60, color_discrete_sequence=[C_BLUE])
    fig.update_traces(marker_line_width=0, opacity=0.9,
                      hovertemplate="‖p<sub>i+1</sub>−p<sub>i</sub>‖ ∈ %{x}<br>count %{y}<extra></extra>")
    fig.add_vline(x=float(bonds.mean()), line=dict(color=C_CYAN, dash="dash", width=1.5),
                  annotation_text=f"mean {bonds.mean():.3f} Å", annotation_font=dict(color=C_CYAN, size=10))
    fig.update_layout(**dark_layout(
        height=380, bargap=0.04, showlegend=False,
        title=dict(text="Backbone bond-length distribution ‖p<sub>i+1</sub> − p<sub>i</sub>‖"),
        xaxis=axis_style(title="Bond length (Å)", tickformat=".2f"), yaxis=axis_style(title="Count"),
    ))
    return fig


def build_loss_chart(df: pd.DataFrame, lam1: float, lam2: float, log_y: bool, weighted: bool) -> go.Figure:
    e = df["epoch"]
    total = df["L_contact"] + lam1 * df["L_smooth"] + lam2 * df["L_steric"]
    s1 = lam1 if weighted else 1.0
    s2 = lam2 if weighted else 1.0
    fig = go.Figure()
    series = (
        ("L_contact", df["L_contact"], C_CYAN, "L<sub>contact</sub>"),
        ("L_smooth", df["L_smooth"] * s1, C_BLUE, "λ₁·L<sub>smooth</sub>" if weighted else "L<sub>smooth</sub>"),
        ("L_steric", df["L_steric"] * s2, C_AMBER, "λ₂·L<sub>steric</sub>" if weighted else "L<sub>steric</sub>"),
    )
    for _, y, color, name in series:
        fig.add_trace(go.Scatter(x=e, y=y, mode="lines", name=name, line=dict(color=color, width=2),
                                 hovertemplate=f"epoch %{{x}}<br>{name} = %{{y:.4f}}<extra></extra>"))
    fig.add_trace(go.Scatter(x=e, y=total, mode="lines", name="L<sub>total</sub>",
                             line=dict(color=C_TEXT, width=2.4, dash="dash"),
                             hovertemplate="epoch %{x}<br>L<sub>total</sub> = %{y:.4f}<extra></extra>"))
    fig.update_layout(**dark_layout(
        height=440, hovermode="x unified",
        title=dict(text=f"EGNN convergence · AdamW lr={LEARNING_RATE} · λ₁={lam1:g} · λ₂={lam2:g}"),
        xaxis=axis_style(title="Epoch"),
        yaxis=axis_style(title="Loss" + (" (log)" if log_y else ""), type="log" if log_y else "linear"),
        margin=dict(l=56, r=24, t=46, b=84),
        legend=dict(orientation="h", y=-0.2, x=0.0, xanchor="left", yanchor="top"),
    ))
    return fig


# =============================================================================
# 10. HTML COMPONENT HELPERS
# =============================================================================
def html(markup: str) -> None:
    # Flatten to one line: indented lines would otherwise be parsed as Markdown code blocks.
    flat = " ".join(line.strip() for line in markup.splitlines() if line.strip())
    st.markdown(flat, unsafe_allow_html=True)


def section(title: str, sub: str = "", tag: str = "") -> None:
    tag_html = f'<span class="n">{tag}</span>' if tag else ""
    html(f'<div class="sec">{tag_html}<span class="t">{title}</span><span class="s">{sub}</span></div>')


def kpi(label: str, value: str, unit: str, sub: str, eq: str = "") -> str:
    eq_html = f'<div class="kpi-eq">{eq}</div>' if eq else ""
    return (f'<div class="kpi"><div class="kpi-label">{label}</div>'
            f'<div class="kpi-value">{value}<span class="kpi-unit">{unit}</span></div>'
            f'<div class="kpi-sub">{sub}</div>{eq_html}</div>')


def mini(label: str, value: str, unit: str = "") -> str:
    return f'<div class="mini"><div class="l">{label}</div><div class="v">{value}<small>{unit}</small></div></div>'


def rows(pairs: list[tuple[str, str]]) -> str:
    return "".join(f'<div class="row"><span class="k">{k}</span><span class="v">{v}</span></div>' for k, v in pairs)


def mb(bin_idx: int) -> float:
    return min(bin_idx * RESOLUTION, CHROM_SIZE) / 1e6


def fmt_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024  # type: ignore[assignment]
    return f"{n:.1f} TB"


# =============================================================================
# 11. SESSION STATE & PRESET CALLBACKS
# =============================================================================
def _clamp_range(lo: int, hi: int, n: int) -> tuple[int, int]:
    lo, hi = max(0, min(lo, n - 1)), max(1, min(hi, n))
    if hi - lo < 10:
        hi = min(n, lo + 10)
        lo = max(0, hi - 10)
    return lo, hi


def apply_preset(key: str) -> None:
    n = st.session_state["_n_nodes"]
    if key == "full":
        st.session_state["bin_range"] = (0, n)
        st.session_state["focus"] = None
        st.session_state["active_preset"] = "Full Chromosome 22"
    elif key == "centromere":
        a, b = CENTROMERE_BP[0] // RESOLUTION, math.ceil(CENTROMERE_BP[1] / RESOLUTION)
        st.session_state["bin_range"] = _clamp_range(a, b, n)
        st.session_state["focus"] = None
        st.session_state["active_preset"] = "Centromeric Region (12.2–17.4 Mb)"
    elif key == "telomeres":
        span = min(TELOMERE_SPAN_BINS, max(1, n // 10))
        st.session_state["bin_range"] = (0, n)
        st.session_state["focus"] = [(0, span), (n - span, n)]
        st.session_state["focus_label"] = "Telomeric ends (1 Mb each)"
        st.session_state["active_preset"] = "Telomeric Ends"
    elif key == "enhancers":
        clusters = st.session_state.get("_clusters", [])
        st.session_state["bin_range"] = (0, n)
        st.session_state["focus"] = [(a, b) for a, b, _ in clusters]
        st.session_state["focus_label"] = "Active enhancer clusters (H3K27ac ≥ P97)"
        st.session_state["active_preset"] = f"Active Enhancer Clusters ({len(clusters)})"


def jump_to_cluster(a: int, b: int) -> None:
    n = st.session_state["_n_nodes"]
    pad = 40
    st.session_state["bin_range"] = _clamp_range(a - pad, b + pad, n)
    st.session_state["focus"] = [(a, b)]
    st.session_state["focus_label"] = f"Enhancer cluster {mb(a):.2f}–{mb(b):.2f} Mb"
    st.session_state["active_preset"] = f"Cluster @ {mb(a):.2f} Mb"


def clear_focus() -> None:
    st.session_state["focus"] = None


def on_manual_range() -> None:
    st.session_state["active_preset"] = "Custom window"


# =============================================================================
# 12. SIDEBAR — CONTROL PANEL & SYSTEM STATUS
# =============================================================================
gpu_ok, device_name = detect_backend()
with st.sidebar:
    dot_cls = "dot" if gpu_ok else "dot warn"
    gpu_txt = "GPU: Active" if gpu_ok else "GPU: Offline · CPU"
    html(f"""
    <div class="brand">
      <div class="brand-title">🧬 ChronoCell-5D <span class="ver">v{APP_VERSION}</span></div>
      <div class="brand-sub">3D chromatin structural workstation · {CHROM_NAME} · {ASSEMBLY}</div>
      <div class="status"><span class="{dot_cls}"></span>{gpu_txt} | Model: SE(3)-EGNN</div>
      <div class="brand-sub" style="margin-top:6px;font-family:var(--mono)">device → {device_name}</div>
    </div>
    """)

    html('<div class="side-h">01 · Data ingestion</div>')
    coord_file = st.file_uploader(
        "Predicted structure",
        type=["pt", "pth", "pdb", "xyz", "npy", "csv"],
        help="predicted_coords.pt (P ∈ R^{N×3}) from the EGNN, or chromosome22_3d.pdb / .xyz exports.",
    )
    feat_file = st.file_uploader(
        "Node features / graph (optional)",
        type=["pt", "pth", "npy", "csv"],
        help="graph_data.pt (PyG Data: x, edge_index, edge_attr) — or an (N, 2) [f_GC, f_epi] .npy / .csv.",
    )

synthetic = synthesize_dataset(N_BINS, 22)
source_label = "Synthetic polymer preview (seed 22)"
coords = synthetic["coords"]
is_synthetic = True
load_error = None
if coord_file is not None:
    try:
        coords = parse_structure(coord_file.getvalue(), coord_file.name)
        source_label = coord_file.name
        is_synthetic = False
    except Exception as exc:  # show the error, keep the UI alive on the synthetic model
        load_error = f"{coord_file.name}: {exc}"

n_nodes = len(coords)
if n_nodes == N_BINS:
    base_tracks = synthetic
else:
    base_tracks = synthesize_dataset(n_nodes, 22)
gc_track, epi_track = base_tracks["gc"], base_tracks["epi"]
track_label = "synthetic tracks (seeded)"
graph_edges: dict[str, np.ndarray] | None = None
feat_error = None
if feat_file is not None:
    try:
        g = parse_graph(feat_file.getvalue(), feat_file.name)
        if len(g["x"]) != n_nodes:
            feat_error = f"{feat_file.name}: {len(g['x']):,} feature rows ≠ {n_nodes:,} structure nodes — using synthetic tracks."
        else:
            gc_track, epi_track = g["x"][:, 0], g["x"][:, 1]
            track_label = feat_file.name
        if "edge_index" in g and "edge_attr" in g:
            graph_edges = {"edge_index": g["edge_index"], "edge_attr": g["edge_attr"]}
    except Exception as exc:
        feat_error = f"{feat_file.name}: {exc}"

clusters = enhancer_clusters(epi_track)

# Reset navigation state whenever a structure with a different node count is loaded.
if st.session_state.get("_n_nodes") != n_nodes:
    st.session_state["_n_nodes"] = n_nodes
    st.session_state["bin_range"] = (0, n_nodes)
    st.session_state["focus"] = None
    st.session_state["active_preset"] = "Full Chromosome 22"
st.session_state["_clusters"] = clusters
st.session_state.setdefault("focus", None)
st.session_state.setdefault("focus_label", "Highlighted region")
st.session_state.setdefault("active_preset", "Full Chromosome 22")

with st.sidebar:
    status_cls, status_txt = ("wn", "SYNTHETIC PREVIEW") if is_synthetic else ("ok", "PIPELINE OUTPUT")
    ext = coords.max(axis=0) - coords.min(axis=0)
    edges_html = ""
    if graph_edges:
        edges_html = "<br>edges · <b>{:,} Micro-C</b>".format(graph_edges["edge_index"].shape[1])
    html(f"""
    <div class="side-card">
      <span class="{status_cls}">● {status_txt}</span><br>
      source · <b>{source_label}</b><br>
      nodes · <b>{n_nodes:,}</b> × 10 kb = <b>{n_nodes * RESOLUTION / 1e6:.2f} Mb</b><br>
      bbox · <b>{ext[0]:.1f} × {ext[1]:.1f} × {ext[2]:.1f} Å</b><br>
      tracks · <b>{track_label}</b>{edges_html}
    </div>
    """)
    if load_error:
        st.error(load_error)
    if feat_error:
        st.warning(feat_error)

    html('<div class="side-h">02 · Genomic filtering</div>')
    st.slider(
        "Bin window [start, end)",
        min_value=0, max_value=n_nodes, step=1, key="bin_range", on_change=on_manual_range,
        help="10 kb genomic bins. Bin 0 → 0 Mb, Bin 5082 → 50.8 Mb.",
    )
    raw_lo, raw_hi = st.session_state["bin_range"]
    lo, hi = _clamp_range(int(raw_lo), int(raw_hi), n_nodes)
    html(f'<div class="side-card">{CHROM_NAME}:<b>{mb(lo):.2f}</b>–<b>{mb(hi):.2f}</b> Mb · '
         f'<b>{hi - lo:,}</b> bins<br>preset · <span class="ok">{st.session_state["active_preset"]}</span></div>')

    html('<div class="side-h">03 · Display</div>')
    show_backbone = st.checkbox("Show backbone connections", value=True)
    spline = st.checkbox("Smooth tubular spline (≤ 2,500 bins)", value=True, disabled=not show_backbone)
    ghost = st.checkbox("Ghost full-chromosome context", value=True)
    cube = st.checkbox("Lock cubic aspect ratio", value=False, help="Off = true spatial proportions (aspectmode='data').")
    point_size = st.slider("Node point size", 1, 10, 3)
    tube_width = st.slider("Backbone tube width", 1, 14, 5, disabled=not show_backbone)
    cmap = st.selectbox("Colour map", COLORMAPS, index=0)
    reverse_cmap = st.checkbox("Reverse colour map", value=False)
    color_by = st.selectbox("Colour nodes by", COLOR_BY, index=0)
    view_height = st.select_slider("Viewport height", options=[560, 640, 720, 800, 900], value=720)

    html('<div class="side-h">Pipeline</div>')
    html(f"""<div class="side-card">
      T1 · <b>graph_data.pt</b> ← chr22.fa, H3K27ac_signal.bigWig, human_microc.mcool<br>
      T2 · <b>predicted_coords.pt</b> ← EGNN · AdamW · {NUM_EPOCHS} ep<br>
      T3 · <b>chromosome22_3d.pdb / .xyz</b> ← this workstation</div>""")


# =============================================================================
# 13. HEADER
# =============================================================================
sub = coords[lo:hi]
m_sel = polymer_metrics(sub)
m_full = polymer_metrics(coords)

src_chip = ('<span class="chip warn">● synthetic preview</span>' if is_synthetic
            else f'<span class="chip hot">● {source_label}</span>')
html(f"""
<div class="hero">
  <div>
    <h1><span class="grad">ChronoCell-5D</span> · 3D Chromatin Structural Workstation</h1>
    <div class="tag">SE(3)-Equivariant Graph Neural Network reconstruction of human Chromosome 22 from sequence, H3K27ac and Micro-C.</div>
  </div>
  <div class="chips">
    <span class="chip"><b>{CHROM_NAME}</b> · {ASSEMBLY}</span>
    <span class="chip">B = <b>10 kb</b></span>
    <span class="chip">N = <b>{n_nodes:,}</b></span>
    <span class="chip">window <b>{mb(lo):.2f}–{mb(hi):.2f} Mb</b></span>
    {src_chip}
  </div>
</div>
""")
if is_synthetic:
    html('<div class="banner">⚠ Displaying a synthetic polymer preview (5,082 beads, TAD-like domains, A/B compartments). '
         'Upload <code>predicted_coords.pt</code>, <code>chromosome22_3d.pdb</code> or <code>.xyz</code> in the sidebar '
         'to load the trained EGNN structure.</div>')

tab1, tab2, tab3, tab4 = st.tabs([
    "◉  3D Chromatin Viewport",
    "▦  Polymer Physics & Genomic Metrics",
    "∑  Loss Function & EGNN Convergence",
    "⇩  Structural Export & Report",
])


# =============================================================================
# 14. TAB 1 — 3D CHROMATIN VIEWPORT
# =============================================================================
with tab1:
    pc = st.columns(5)
    pc[0].button("⬤  Full Chromosome 22", on_click=apply_preset, args=("full",), key="p_full", **stretch("button"))
    cen_bins_avail = n_nodes > CENTROMERE_BP[0] // RESOLUTION
    pc[1].button("◎  Centromeric Region", on_click=apply_preset, args=("centromere",), key="p_cen",
                 disabled=not cen_bins_avail, **stretch("button"))
    pc[2].button("⇹  Telomeric Ends", on_click=apply_preset, args=("telomeres",), key="p_tel", **stretch("button"))
    pc[3].button("✦  Active Enhancer Clusters", on_click=apply_preset, args=("enhancers",), key="p_enh",
                 disabled=not clusters, **stretch("button"))
    pc[4].button("✕  Clear highlight", on_click=clear_focus, key="p_clear",
                 disabled=not st.session_state.get("focus"), **stretch("button"))

    col_view, col_insp = st.columns([3.3, 1], gap="medium")
    with col_view:
        opts = dict(
            cmap=cmap, reverse=reverse_cmap, color_by=color_by, backbone=show_backbone, spline=spline,
            ghost=ghost, size=point_size, tube=tube_width, cube=cube, height=view_height,
            focus_label=st.session_state.get("focus_label", "Highlighted region"),
        )
        fig3d = build_viewport(coords, gc_track, epi_track, lo, hi, st.session_state.get("focus"), opts)
        show_plot(fig3d, "viewport3d")

    with col_insp:
        focus = st.session_state.get("focus") or []
        focus_bins = sum(max(0, min(b, hi) - max(a, lo)) for a, b in focus)
        comp = base_tracks["compartment_a"][lo:hi] if is_synthetic else None
        region_rows = rows([
            ("preset", st.session_state["active_preset"][:26]),
            ("window", f"{mb(lo):.2f}–{mb(hi):.2f} Mb"),
            ("bins", f"{lo:,} → {hi - 1:,}"),
            ("nodes", f"{hi - lo:,}"),
            ("highlighted", f"{focus_bins:,} bins"),
        ])
        html(f'<div class="insp"><div class="h">Region inspector</div>{region_rows}</div>')
        gc_win = gc_track[lo:hi]
        readout_rows = rows([
            ("R_g", f"{m_sel['rg']:.2f} Å"),
            ("R_e (5′→3′)", f"{m_sel['re']:.2f} Å"),
            ("⟨bond⟩", f"{m_sel['mean_bond']:.3f} Å"),
            ("contour", f"{m_sel['contour']:,.0f} Å"),
            ("max extent", f"{m_sel['extent']:.1f} Å"),
            ("⟨f_GC⟩", f"{gc_win[gc_win > 0].mean():.3f}" if (gc_win > 0).any() else "—"),
            ("A-compartment", f"{comp.mean() * 100:.0f}%" if comp is not None else "n/a"),
        ])
        html(f'<div class="insp"><div class="h">Live polymer readout</div>{readout_rows}</div>')
        in_window = [c for c in clusters if c[1] > lo and c[0] < hi]
        html(f'<div class="insp"><div class="h">Enhancer clusters · {len(in_window)} in window</div>'
             f'<div class="row"><span class="k">threshold</span><span class="v">H3K27ac ≥ P97</span></div></div>')
        for a, b, peak in in_window[:6]:
            st.button(f"↳ {mb(a):.2f}–{mb(b):.2f} Mb · peak {peak:.1f}", key=f"cl_{a}_{b}",
                      on_click=jump_to_cluster, args=(a, b), **stretch("button"))
        if not in_window:
            st.caption("No H3K27ac clusters in this window.")

    st.caption("Drag to rotate · scroll to zoom · right-drag to pan · double-click to reset. "
               "Camera is preserved while you change display settings. Hover a bead for bin, genomic coordinate and features.")


# =============================================================================
# 15. TAB 2 — POLYMER PHYSICS & GENOMIC METRICS
# =============================================================================
with tab2:
    window_note = "full chain" if (lo == 0 and hi == n_nodes) else f"window bins {lo:,}–{hi - 1:,}"
    k1, k2, k3, k4 = st.columns(4)
    total_mb = min(n_nodes * RESOLUTION, CHROM_SIZE) / 1e6
    k1.markdown(kpi("Total length", f"{total_mb:.2f}", "Mb", f"({n_nodes:,} Bins) · window {mb(hi) - mb(lo):.2f} Mb",
                    "N = ⌈L / B⌉, B = 10 kb"), unsafe_allow_html=True)
    k2.markdown(kpi("Radius of gyration R_g", f"{m_sel['rg']:.2f}", "Å",
                    f"{window_note} · full {m_full['rg']:.2f} Å", "√ mean ‖p_i − p̄‖²"), unsafe_allow_html=True)
    k3.markdown(kpi("End-to-end distance R_e", f"{m_sel['re']:.2f}", "Å",
                    f"bin {lo} → bin {hi - 1} · full {m_full['re']:.2f} Å", "‖p_last − p_first‖"), unsafe_allow_html=True)
    k4.markdown(kpi("Polymer smoothness", f"{m_sel['mean_bond']:.3f}", "Å",
                    f"mean bond · CV {m_sel['bond_cv'] * 100:.1f}%", "mean ‖p_{i+1} − p_i‖"), unsafe_allow_html=True)

    st.write("")
    ster = steric_analysis(sub, D_MIN)
    fig_scaling, nu = build_scaling_plot(sub)
    mc = st.columns(6)
    mc[0].markdown(mini("Contour length", f"{m_sel['contour']:,.0f}", "Å"), unsafe_allow_html=True)
    mc[1].markdown(mini("R_e² / R_g²", f"{m_sel['re2_rg2']:.2f}", "ideal = 6"), unsafe_allow_html=True)
    mc[2].markdown(mini("Scaling exponent ν", f"{nu:.3f}", "R ∝ s^ν"), unsafe_allow_html=True)
    mc[3].markdown(mini("L_smooth (live)", f"{m_sel['l_smooth']:.3f}", "Å²"), unsafe_allow_html=True)
    mc[4].markdown(mini(f"Steric clashes &lt; {D_MIN}", f"{ster['violations']:,.0f}",
                        "exact" if ster["exact"] else "MC est."), unsafe_allow_html=True)
    corr_mask = (gc_track[lo:hi] > 0) & (epi_track[lo:hi] > 0)
    corr = float(np.corrcoef(gc_track[lo:hi][corr_mask], epi_track[lo:hi][corr_mask])[0, 1]) if corr_mask.sum() > 3 else float("nan")
    mc[5].markdown(mini("corr(f_GC, f_epi)", f"{corr:+.3f}", "Pearson"), unsafe_allow_html=True)

    st.write("")
    section("Spatial analytics", f"{window_note} · {hi - lo:,} beads", "2D")
    c_left, c_right = st.columns([1.05, 1], gap="medium")
    with c_left:
        hm_mode = st.radio("Heatmap mode", ["Euclidean distance ‖p_i − p_j‖ (Å)", "Inferred contact M̂ = 1/d − ε (log)"],
                           horizontal=True, label_visibility="collapsed")
        hm_cmap = cmap + ("_r" if reverse_cmap else "")
        show_plot(build_distance_heatmap(sub, lo, hm_mode, hm_cmap), "dist_heatmap")
    with c_right:
        smooth_w = st.select_slider("Track smoothing (bins)", options=[1, 3, 5, 9, 15, 25], value=5)
        show_plot(build_feature_overlay(gc_track, epi_track, lo, hi, smooth_w, clusters, track_label), "feature_overlay")
        st.caption("Shaded: hg38 p-arm assembly gap (N, f = 0) and centromere (12.2–17.4 Mb). "
                   "▼ marks H3K27ac enhancer clusters. f_GC = (#G + #C)/B, f_epi = (1/B) Σ Signal_H3K27ac(k).")

    c3, c4 = st.columns(2, gap="medium")
    with c3:
        show_plot(fig_scaling, "scaling")
    with c4:
        show_plot(build_bond_histogram(sub), "bond_hist")


# =============================================================================
# 16. TAB 3 — LOSS FUNCTION & EGNN PHYSICS CONVERGENCE
# =============================================================================
with tab3:
    html("""
    <div class="flow">
      <div class="stage"><div class="k">STAGE 1 · INPUT TRACKS</div><div class="t">1D + 3D genomics</div>
        <div class="d">chr22.fa → f_GC<br>H3K27ac_signal.bigWig → f_epi<br>human_microc.mcool → M_ij</div></div>
      <div class="arrow">➜</div>
      <div class="stage"><div class="k">STAGE 2 · GRAPH (PyG)</div><div class="t">G = (V, E, X, E_attr)</div>
        <div class="d">x_i = [f_GC, f_epi]ᵀ ∈ R²<br>E: |i−j| = 1 ∨ M_ij &gt; 0<br>e_ij = M_ij</div></div>
      <div class="arrow">➜</div>
      <div class="stage"><div class="k">STAGE 3 · SE(3)-EGNN</div><div class="t">Equivariant layers</div>
        <div class="d">d²_ij → m_ij = MLP_m(·)<br>p_i ← p_i + Σ (p_i−p_j)·MLP_x<br>h_i ← MLP_h(h_i ‖ Σ m_ij)</div></div>
      <div class="arrow">➜</div>
      <div class="stage"><div class="k">STAGE 4 · PHYSICS LOSS</div><div class="t">Multi-objective</div>
        <div class="d">L_contact + λ₁ L_smooth<br>+ λ₂ L_steric<br>AdamW · lr 0.005</div></div>
      <div class="arrow">➜</div>
      <div class="stage"><div class="k">STAGE 5 · OUTPUT</div><div class="t">P ∈ R^{N×3}</div>
        <div class="d">predicted_coords.pt<br>chromosome22_3d.pdb / .xyz<br>interactive 3D render</div></div>
    </div>
    """)

    section("Composite physical objective", "minimised end-to-end over network weights and coordinates", "∑")
    with st.container(border=True):
        st.latex(r"\mathcal{L}_{\text{total}} \;=\; \mathcal{L}_{\text{contact}} \;+\; \lambda_1\,\mathcal{L}_{\text{smooth}} \;+\; \lambda_2\,\mathcal{L}_{\text{steric}}"
                 r"\qquad \lambda_1 = 0.1,\;\; \lambda_2 = 0.05")

    loss_terms = (
        ("1 · Micro-C contact reconstruction", "L_contact · weight 1",
         r"\mathcal{L}_{\text{contact}} = \frac{1}{|E|}\sum_{(i,j)\in E}\Big(\lVert p_i - p_j\rVert_2 - \frac{1}{M_{i,j}+\epsilon}\Big)^2",
         "High contact counts M<sub>ij</sub> imply spatial proximity: the target distance is 1/(M<sub>ij</sub>+ε) "
         "with ε = 10⁻³. The squared residual penalises both over-stretching and over-compression of every observed contact."),
        ("2 · Polymer chain smoothness", f"L_smooth · λ₁ = {LAMBDA_SMOOTH}",
         r"\mathcal{L}_{\text{smooth}} = \frac{1}{N-1}\sum_{i=1}^{N-1}\lVert p_{i+1} - p_i\rVert_2^2",
         "DNA is a continuous polymer: adjacent 10 kb bins behave like beads on a wire. Penalising squared bond "
         "lengths prevents chain breaks and unphysical stretching across the N − 1 backbone bonds."),
        ("3 · Steric hindrance overlap", f"L_steric · λ₂ = {LAMBDA_STERIC}",
         r"\mathcal{L}_{\text{steric}} = \frac{1}{|\Omega|}\sum_{(i,j)\in\Omega}\big[\max\!\big(0,\; d_{\min} - \lVert p_i - p_j\rVert_2\big)\big]^2",
         "Volume exclusion over Ω = {(i, j) : |i − j| &gt; 2}: non-adjacent chromatin cannot occupy the same space. "
         "Zero penalty when ‖p<sub>i</sub> − p<sub>j</sub>‖ ≥ d<sub>min</sub> = 0.5; a quadratic repulsive force otherwise."),
    )
    for title, tag, latex, text in loss_terms:
        with st.container(border=True):
            eq_col, txt_col = st.columns([1.45, 1], gap="large")
            with eq_col:
                html(f'<div class="eqhead">{title}<span>{tag}</span></div>')
                st.latex(latex)
            with txt_col:
                html(f'<p class="eqtext">{text}</p>')

    with st.expander("SE(3)-equivariant layer equations & node/edge features", expanded=False):
        e1, e2 = st.columns(2, gap="large")
        with e1:
            st.markdown("**EGNN layer l → l + 1**")
            st.latex(r"d_{i,j}^2 = \lVert p_i^{(l)} - p_j^{(l)}\rVert_2^2 = \sum_{k=1}^{3}\big(p_{i,k}^{(l)} - p_{j,k}^{(l)}\big)^2")
            st.latex(r"m_{i,j} = \mathrm{MLP}_m\big(h_i^{(l)} \,\Vert\, h_j^{(l)} \,\Vert\, d_{i,j}^2 \,\Vert\, e_{i,j}\big)")
            st.latex(r"p_i^{(l+1)} = p_i^{(l)} + \sum_{j\in\mathcal{N}(i)}\big(p_i^{(l)} - p_j^{(l)}\big)\cdot \mathrm{MLP}_x(m_{i,j})")
            st.latex(r"h_i^{(l+1)} = \mathrm{MLP}_h\Big(h_i^{(l)} \,\Vert\, \sum_{j\in\mathcal{N}(i)} m_{i,j}\Big)")
            st.caption("Equivariance: for any R ∈ SO(3), t ∈ R³, rotating/translating the input rotates/translates "
                       "the output identically (R p + t), while d²_ij and h_i stay invariant.")
        with e2:
            st.markdown("**Graph construction (Teammate 1)**")
            st.latex(r"f_{GC}(i) = \frac{1}{E_i - S_i}\sum_{k=S_i}^{E_i-1}\mathbb{I}\big(\mathrm{seq}[k]\in\{G,C\}\big)")
            st.latex(r"f_{epi}(i) = \frac{1}{E_i - S_i}\sum_{k=S_i}^{E_i-1}\mathrm{Signal}_{H3K27ac}(k)")
            st.latex(r"x_i = \begin{bmatrix} f_{GC}(i) \\ f_{epi}(i)\end{bmatrix}\in\mathbb{R}^2,\qquad e_{i,j} = M_{i,j}")
            st.latex(r"E = \{(v_i, v_j) \mid |i-j| = 1 \;\lor\; M_{i,j} > 0\},\qquad N = \Big\lceil \tfrac{50{,}818{,}468}{10{,}000}\Big\rceil = 5{,}082")

    pcount = egnn_parameter_count()
    section("Training configuration", "Teammate 2 · EGNNLayer(in_dim=2, hidden_dim=32)", "cfg")
    cfg = st.columns(6)
    cfg[0].markdown(mini("Optimizer", "AdamW", f"lr {LEARNING_RATE}"), unsafe_allow_html=True)
    cfg[1].markdown(mini("Epochs", f"{NUM_EPOCHS}", "cycles"), unsafe_allow_html=True)
    cfg[2].markdown(mini("λ₁ · λ₂", f"{LAMBDA_SMOOTH} · {LAMBDA_STERIC}", ""), unsafe_allow_html=True)
    cfg[3].markdown(mini("d_min · ε", f"{D_MIN} · 1e-3", ""), unsafe_allow_html=True)
    cfg[4].markdown(mini("EGNN weights", f"{pcount['total']:,}", "params"), unsafe_allow_html=True)
    cfg[5].markdown(mini("Free coords P", f"{n_nodes * 3:,}", "params"), unsafe_allow_html=True)

    st.write("")
    section("Physics convergence", "loss vs. epoch", "fit")
    cc1, cc2 = st.columns([1, 3.2], gap="medium")
    with cc1:
        lam1 = st.slider("λ₁ · smoothness weight", 0.0, 1.0, LAMBDA_SMOOTH, 0.01)
        lam2 = st.slider("λ₂ · steric weight", 0.0, 1.0, LAMBDA_STERIC, 0.01)
        log_y = st.checkbox("Log-scale loss axis", value=True)
        weighted = st.checkbox("Plot λ-weighted contributions", value=False)
        log_file = st.file_uploader("Training log (optional CSV)", type=["csv"],
                                    help="Columns: epoch, and any of contact / smooth / steric (names matched by substring).")
        loss_df = simulated_training_log()
        log_label = "simulated trajectory"
        if log_file is not None:
            try:
                raw = pd.read_csv(log_file)
                cols = {c.lower(): c for c in raw.columns}
                pick = lambda key: next((cols[c] for c in cols if key in c), None)  # noqa: E731
                ep_col = pick("epoch")
                df_new = pd.DataFrame({"epoch": raw[ep_col] if ep_col else np.arange(1, len(raw) + 1)})
                for key, name in (("contact", "L_contact"), ("smooth", "L_smooth"), ("steric", "L_steric")):
                    col = pick(key)
                    df_new[name] = raw[col].astype(float) if col else 0.0
                loss_df = df_new
                log_label = log_file.name
            except Exception as exc:
                st.error(f"Could not read training log: {exc}")
        last = loss_df.iloc[-1]
        final_total = last["L_contact"] + lam1 * last["L_smooth"] + lam2 * last["L_steric"]
        first = loss_df.iloc[0]
        first_total = first["L_contact"] + lam1 * first["L_smooth"] + lam2 * first["L_steric"]
        final_rows = rows([
            ("L_contact", f"{last['L_contact']:.4f}"),
            ("L_smooth", f"{last['L_smooth']:.4f}"),
            ("L_steric", f"{last['L_steric']:.5f}"),
            ("L_total", f"{final_total:.4f}"),
            ("reduction", f"{(1 - final_total / first_total) * 100:.1f}%" if first_total > 0 else "—"),
        ])
        html(f'<div class="insp"><div class="h">Final epoch · {log_label}</div>{final_rows}</div>')
    with cc2:
        show_plot(build_loss_chart(loss_df, lam1, lam2, log_y, weighted), "loss_chart")
        if log_label == "simulated trajectory":
            st.caption("Simulated trajectory for demonstration. Upload the per-epoch CSV logged by the training loop "
                       "to replace it with the real run.")

    section("Live evaluation on the loaded structure", f"{window_note} · uses the exact loss definitions above", "eval")
    lv = st.columns(4)
    d_min_live = lv[0].number_input("d_min for evaluation", min_value=0.01, max_value=50.0, value=float(D_MIN), step=0.1)
    ster_live = steric_analysis(sub, float(d_min_live))
    lv[1].markdown(mini("L_smooth", f"{m_sel['l_smooth']:.4f}", "Å²"), unsafe_allow_html=True)
    lv[2].markdown(mini("L_steric", f"{ster_live['l_steric']:.5f}",
                        f"{'exact' if ster_live['exact'] else 'MC'} · min d {ster_live['min_dist']:.2f} Å"),
                   unsafe_allow_html=True)
    if graph_edges is not None:
        lc = contact_loss(coords, graph_edges["edge_index"], graph_edges["edge_attr"])
        lv[3].markdown(mini("L_contact", f"{lc:.4f}", f"{graph_edges['edge_index'].shape[1]:,} edges"),
                       unsafe_allow_html=True)
    else:
        lv[3].markdown(mini("L_contact", "—", "upload graph_data.pt"), unsafe_allow_html=True)


# =============================================================================
# 17. TAB 4 — STRUCTURAL EXPORT & REPORT GENERATION
# =============================================================================
with tab4:
    section("Export configuration", "PDB / XYZ generated in memory from the active coordinate state", "io")
    x1, x2, x3 = st.columns([1.2, 1.2, 1.6], gap="medium")
    with x1:
        scope = st.radio("Export scope", [f"Active window · bins {lo:,}–{hi - 1:,}", f"Full chromosome · {n_nodes:,} bins"],
                         index=1)
    with x2:
        encode_tracks = st.checkbox("Encode tracks in PDB (occupancy = f_GC, B-factor = f_epi)", value=True,
                                    help="Off = spec defaults (occupancy 1.00, B-factor 20.00).")
        centre = st.checkbox("Centre coordinates at origin", value=False)
    with x3:
        html("""<div class="insp"><div class="h">Open in molecular viewers</div>
          <div class="row"><span class="k">PyMOL</span><span class="v">load chromosome22_3d.pdb; spectrum b</span></div>
          <div class="row"><span class="k">ChimeraX</span><span class="v">open chromosome22_3d.pdb; color bfactor</span></div>
          <div class="row"><span class="k">Mol*</span><span class="v">molstar.org/viewer → Open Files</span></div></div>""")

    if scope.startswith("Active"):
        e_lo, e_hi = lo, hi
    else:
        e_lo, e_hi = 0, n_nodes
    export_coords = coords[e_lo:e_hi].copy()
    if centre:
        export_coords -= export_coords.mean(axis=0)

    pdb_text, pdb_scale = build_pdb(export_coords, e_lo, gc_track, epi_track, encode_tracks, source_label)
    xyz_text = build_xyz(export_coords, e_lo)
    csv_text = build_csv(export_coords, e_lo, gc_track, epi_track)
    m_exp = polymer_metrics(export_coords)
    ster_exp = steric_analysis(export_coords, D_MIN)
    _, _, nu_exp = distance_scaling(export_coords)

    report = {
        "project": "ChronoCell-5D",
        "version": APP_VERSION,
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "model": {
            "architecture": "SE(3)-Equivariant Graph Neural Network (EGNN)",
            "layer": {"in_dim": IN_DIM, "hidden_dim": HIDDEN_DIM, "parameters": pcount},
            "optimizer": {"name": "AdamW", "lr": LEARNING_RATE, "epochs": NUM_EPOCHS},
            "loss": {
                "formula": "L_total = L_contact + lambda_1 * L_smooth + lambda_2 * L_steric",
                "lambda_1": LAMBDA_SMOOTH,
                "lambda_2": LAMBDA_STERIC,
                "d_min": D_MIN,
                "epsilon": EPS,
                "steric_subsample_pairs": STERIC_SUBSAMPLE,
            },
            "compute_device": device_name,
        },
        "genome": {
            "organism": "Homo sapiens",
            "assembly": ASSEMBLY,
            "chromosome": CHROM_NAME,
            "length_bp": CHROM_SIZE,
            "resolution_bp": RESOLUTION,
            "n_bins_spec": N_BINS,
        },
        "inputs": {
            "structure_source": source_label,
            "is_synthetic_preview": is_synthetic,
            "feature_tracks": track_label,
            "micro_c_edges": int(graph_edges["edge_index"].shape[1]) if graph_edges else None,
            "pipeline_files": ["chr22.fa", "H3K27ac_signal.bigWig", "human_microc.mcool", "graph_data.pt", "predicted_coords.pt"],
        },
        "export": {
            "bins": [int(e_lo), int(e_hi - 1)],
            "genomic_region": f"{CHROM_NAME}:{e_lo * RESOLUTION + 1}-{min(e_hi * RESOLUTION, CHROM_SIZE)}",
            "n_nodes": int(e_hi - e_lo),
            "centred": bool(centre),
            "pdb_scale_factor": pdb_scale,
            "pdb_track_encoding": "occupancy=f_GC, bfactor=f_epi(0-99.99)" if encode_tracks else "occupancy=1.00, bfactor=20.00",
        },
        "metrics": {
            "radius_of_gyration_A": round(m_exp["rg"], 4),
            "end_to_end_distance_A": round(m_exp["re"], 4),
            "mean_bond_length_A": round(m_exp["mean_bond"], 5),
            "bond_length_cv": round(m_exp["bond_cv"], 5),
            "contour_length_A": round(m_exp["contour"], 3),
            "re2_over_rg2": round(m_exp["re2_rg2"], 4),
            "scaling_exponent_nu": round(nu_exp, 4),
            "L_smooth": round(m_exp["l_smooth"], 6),
            "L_steric": round(ster_exp["l_steric"], 8),
            "L_steric_exact": bool(ster_exp["exact"]),
            "steric_violations_below_d_min": round(ster_exp["violations"], 1),
            "L_contact": round(contact_loss(coords, graph_edges["edge_index"], graph_edges["edge_attr"]), 6) if graph_edges else None,
        },
        "enhancer_clusters_top10": [
            {"start_bin": a, "end_bin": b, "start_mb": round(mb(a), 3), "end_mb": round(mb(b), 3), "peak_f_epi": round(pk, 3)}
            for a, b, pk in clusters[:10]
        ],
        "deliverables": ["chromosome22_3d.pdb", "chromosome22_3d.xyz", "chronocell_report.json"],
        "software": {
            "streamlit": st.__version__,
            "plotly": plotly.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "torch": torch.__version__ if TORCH_OK else None,
        },
    }
    report_text = json.dumps(report, indent=2)

    st.write("")
    d1, d2, d3, d4 = st.columns(4, gap="medium")
    pdb_bytes, xyz_bytes = pdb_text.encode(), xyz_text.encode()
    json_bytes, csv_bytes = report_text.encode(), csv_text.encode()
    with d1:
        html(f'<div class="dl"><div class="f">chromosome22_3d.pdb</div><div class="m">{e_hi - e_lo:,} ATOM · '
             f'{max(0, e_hi - e_lo - 1):,} CONECT<br>{fmt_bytes(len(pdb_bytes))} · PyMOL / ChimeraX / Mol*</div></div>')
        st.download_button("⇩  Download PDB", pdb_bytes, "chromosome22_3d.pdb", "chemical/x-pdb", key="dl_pdb", **stretch("download_button"))
    with d2:
        html(f'<div class="dl"><div class="f">chromosome22_3d.xyz</div><div class="m">{e_hi - e_lo:,} Cartesian records<br>'
             f'{fmt_bytes(len(xyz_bytes))} · point-cloud renderers</div></div>')
        st.download_button("⇩  Download XYZ", xyz_bytes, "chromosome22_3d.xyz", "chemical/x-xyz", key="dl_xyz", **stretch("download_button"))
    with d3:
        html(f'<div class="dl"><div class="f">chronocell_report.json</div><div class="m">model · loss · metrics · clusters<br>'
             f'{fmt_bytes(len(json_bytes))} · machine-readable</div></div>')
        st.download_button("⇩  Download report", json_bytes, "chronocell_report.json", "application/json", key="dl_json", **stretch("download_button"))
    with d4:
        html(f'<div class="dl"><div class="f">chromosome22_3d_bins.csv</div><div class="m">bin · bp · x y z · f_GC · f_epi<br>'
             f'{fmt_bytes(len(csv_bytes))} · pandas / R</div></div>')
        st.download_button("⇩  Download CSV", csv_bytes, "chromosome22_3d_bins.csv", "text/csv", key="dl_csv", **stretch("download_button"))

    if pdb_scale != 1.0:
        st.warning(f"Coordinates exceeded PDB fixed-width limits and were rescaled by {pdb_scale:.6f} in the PDB file "
                   "(recorded in REMARK 5). XYZ / CSV keep the original values.")

    st.write("")
    section("Preview", "first 40 lines of each generated file", "txt")
    pv1, pv2, pv3 = st.tabs(["PDB", "XYZ", "JSON report"])
    with pv1:
        pdb_lines = pdb_text.splitlines()
        conect_start = next((i for i, ln in enumerate(pdb_lines) if ln.startswith("CONECT")), len(pdb_lines))
        preview = pdb_lines[:32] + ["..."] + pdb_lines[conect_start : conect_start + 5] + ["..."] + pdb_lines[-2:]
        st.code("\n".join(preview), language="text")
    with pv2:
        st.code("\n".join(xyz_text.splitlines()[:40]) + "\n...", language="text")
    with pv3:
        st.code(report_text[:6000], language="json")

html(f"""<div style="margin-top:22px;padding-top:10px;border-top:1px solid {C_GRID};
font-family:'JetBrains Mono',monospace;font-size:0.68rem;color:{C_MUTED};display:flex;justify-content:space-between;flex-wrap:wrap;gap:8px">
<span>ChronoCell-5D v{APP_VERSION} · SE(3)-EGNN · {CHROM_NAME} {ASSEMBLY} @ 10 kb</span>
<span>T1 graph assembly → T2 EGNN training → T3 structural export & visualization</span></div>""")
