"""
ChronoCell-5D — 3D / 4D chromatin structural workstation for human chromosomes (GRCh38).

    streamlit run app.py

Numerics live in the `chronocell` package; `ui/common.py` assembles datasets (coordinate slot ->
reference fallback) and `ui/four_d.py` renders the 4D workspace. This file is the shell and the 3D
workspace. The 3D viewport is an st.fragment: display settings re-render only the figure.
"""

from __future__ import annotations

import datetime as dt
import io
import math
import os
import sys
import threading
import time
import types
from pathlib import Path

import numpy as np
import pandas as pd
import plotly
import streamlit as st

_PROJECT_PACKAGES = ("chronocell", "ui")
# Process-wide guard state, kept outside Streamlit's caches (which a purge clears) and outside the
# project packages (which a purge removes). One lock serialises purge + import across sessions.
_RUNTIME = sys.modules.setdefault("_chronocell_runtime", types.ModuleType("_chronocell_runtime"))
_LOCK = _RUNTIME.__dict__.setdefault("lock", threading.RLock())
_STAMPS: dict[str, float] = _RUNTIME.__dict__.setdefault("stamps", {})


def _purge_project_modules() -> None:
    for name in [n for n in list(sys.modules) if n.partition(".")[0] in _PROJECT_PACKAGES]:
        sys.modules.pop(name, None)
    _STAMPS.clear()
    st.cache_data.clear()
    st.cache_resource.clear()


def _refresh_project_modules() -> None:
    """Keep a long-running `streamlit run` consistent with the code on disk.

    When a file changes, Streamlit's watcher deletes that module from sys.modules, but the parent
    package keeps the old module object as an attribute, so `from chronocell import genome` hands
    back the stale module (the cause of "module 'chronocell.genome' has no attribute
    'MAIN_CHROMOSOMES'" after an upgrade). If any project module is stale — dropped by the watcher,
    or its file changed since import — every project module is purged together so they are
    re-imported consistently, and cached objects built from the old classes are cleared.
    """
    stale = False
    for name, mod in list(sys.modules.items()):
        if name.partition(".")[0] not in _PROJECT_PACKAGES or not getattr(mod, "__file__", None):
            continue
        try:
            mtime = os.path.getmtime(mod.__file__)
        except OSError:
            stale = True
            continue
        if _STAMPS.setdefault(name, mtime) != mtime:
            stale = True
    for pkg_name in _PROJECT_PACKAGES:
        pkg = sys.modules.get(pkg_name)
        for val in (list(vars(pkg).values()) if pkg is not None else ()):
            if (isinstance(val, types.ModuleType) and val.__name__.startswith(pkg_name + ".")
                    and sys.modules.get(val.__name__) is not val):
                stale = True
    if stale:
        _purge_project_modules()


def _stamp_project_modules() -> None:
    for name, mod in list(sys.modules.items()):
        if name.partition(".")[0] in _PROJECT_PACKAGES and getattr(mod, "__file__", None):
            try:
                _STAMPS.setdefault(name, os.path.getmtime(mod.__file__))
            except OSError:
                pass


with _LOCK:
    _refresh_project_modules()
    for _attempt in (1, 2, 3):
        try:
            from chronocell import (agent as A, domains, features, formats, genes as G, genome, pdf_report, physics,
                                    snapshot as SN, states as S, theme as T, viz)
            from ui import agent_panel, compare, drug_lab, four_d, genes_view, guide, states_panel
            from ui.common import (SLOT_ROOT, Dataset, banner, clamp_window, esc, fmt, html, load_dataset, readout,
                                   slot_files, slot_graph, warning_card)
            break
        except (KeyError, ImportError):
            # A file saved while this run was importing: the watcher unloaded a module mid-import.
            # Give the watcher a moment to finish, then re-import everything consistently.
            if _attempt == 3:
                raise
            time.sleep(0.4 * _attempt)
            _purge_project_modules()
    try:
        import torch
        from chronocell import egnn
        TORCH = True
    except ImportError:  # the viewer, physics, 4D scenarios and export work without PyTorch
        torch = egnn = None
        TORCH = False
    _stamp_project_modules()

st.set_page_config(page_title="ChronoCell-5D · chromatin 3D/4D workstation", page_icon="◐", layout="wide",
                   initial_sidebar_state="expanded")
T.inject()

VERSION = "3.2"
MAX_FIT_BEADS = 2000
MIN_FIT_CONTACTS = 20
REGIONS = {"whole": "Whole chromosome", "centromere": "Centromere", "telomeres": "Telomeric ends",
           "hubs": "Enhancer hubs", "custom": "Custom window"}
SHORT = {"whole": "Whole", "centromere": "Centromere", "telomeres": "Telomeres", "hubs": "Enhancer hubs",
         "custom": "Custom"}
ss = st.session_state
ss.setdefault("fits", {})
ss.setdefault("region_choice", "whole")
ss.setdefault("custom_window", None)          # (lo, hi) in local bins; the single source of truth
ss.setdefault("workspace", "3D structure")
ss.setdefault("chrom_choice", "chr22")


# ======================================================================================
# Cached analysis
# ======================================================================================
@st.cache_data(show_spinner=False, max_entries=64)
def read_slot_file(path: str, mtime: float) -> bytes:
    return Path(path).read_bytes()


@st.cache_data(show_spinner=False)
def hover_labels(ds_key: str, _ds: Dataset) -> list[str]:
    ch, n = _ds.chrom, _ds.n
    g = _ds.gbin(np.arange(n))
    start, end = ch.bin_start(g) + 1, ch.bin_end(g)
    band_idx = ch.band_for_bins()[np.clip(g, 0, ch.n_bins - 1)]
    out = []
    for i in range(n):
        head = f"<b>{ch.name}:{start[i]:,}–{end[i]:,}</b><br>bin {int(g[i]):,} · {ch.bands[band_idx[i]].name}"
        if _ds.valid[i] and np.isfinite(_ds.gc[i]):
            out.append(f"{head}<br>GC {_ds.gc[i]:.3f} · H3K27ac {_ds.epi[i]:.2f}")
        else:
            out.append(f"{head}<br>unassembled (N) · no sequence")
    return out


@st.cache_data(show_spinner=False)
def hubs_for(ds_key: str, _ds: Dataset) -> list[tuple[int, int, float]]:
    return features.signal_hubs(_ds.epi, _ds.valid)


@st.cache_resource(show_spinner=False, max_entries=64)
def analyse(coords: np.ndarray, b0: float, d_min_factor: float) -> dict:
    """All structure-level physics for one conformation window (cached on the coordinates)."""
    bonds = physics.bond_lengths(coords)
    fit = physics.distance_scaling(coords)
    ster = physics.loss_steric(coords, b0=b0, d_min=d_min_factor * b0)
    rg, re = physics.radius_of_gyration(coords), physics.end_to_end(coords)
    return dict(rg=rg, re=re, contour=float(bonds.sum()), bond_mean=float(bonds.mean() / b0),
                bond_sd=float(bonds.std() / b0), bonds_b0=bonds / b0, fit=fit, steric=ster,
                ratio=(re / rg) ** 2 if rg > 0 else float("nan"), l_smooth=physics.loss_smooth(coords, b0))


@st.cache_data(show_spinner=False)
def window_contacts(ds_key: str, _ds: Dataset, lo: int, hi: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    m = (_ds.ci >= lo) & (_ds.ci < hi) & (_ds.cj >= lo) & (_ds.cj < hi)
    return _ds.ci[m] - lo, _ds.cj[m] - lo, _ds.cm[m]


@st.cache_data(show_spinner=False)
def equivariance_report() -> dict:
    return egnn.equivariance_check(n=300, seed=0)


def best_fit_window(ds: Dataset, width: int = 800) -> tuple[int, int]:
    """The `width`-bead window with the most contacts (a sensible default for in-app fitting)."""
    if ds.n <= width or not ds.has_contacts:
        return clamp_window(0, min(width, ds.n), ds.n)
    per_bead = np.bincount(ds.ci, minlength=ds.n) + np.bincount(ds.cj, minlength=ds.n)
    csum = np.concatenate([[0], np.cumsum(per_bead)])
    lo = int(np.argmax(csum[width:] - csum[:-width]))
    return clamp_window(lo, lo + width, ds.n)


def full_track(ds: Dataset, arr: np.ndarray) -> np.ndarray:
    """Local (window) track -> chromosome-length array (NaN outside the loaded window)."""
    out = np.full(ds.chrom.n_bins, np.nan)
    out[ds.bin0:ds.bin0 + ds.n] = arr
    return out


# ======================================================================================
# Region logic (every path returns a clamped, non-empty window)
# ======================================================================================
def resolve_region(region: str, ds: Dataset, hubs: list) -> tuple[int, int, list[tuple[int, int]] | None, str]:
    n, ch = ds.n, ds.chrom
    if region == "centromere" and ch.acen:
        a, b = ch.interval_to_bins(*ch.acen)
        a, b = a - ds.bin0, b - ds.bin0
        if b <= 0 or a >= n:
            return 0, n, None, "The centromere lies outside the loaded window; showing the whole window."
        pad = max(10, (b - a) // 2)
        lo, hi = clamp_window(a - pad, b + pad, n)
        return lo, hi, [(max(a, 0), min(b, n))], ""
    if region == "telomeres":
        span = max(1, min(max(10, 1_000_000 // ch.resolution), n // 10))
        return 0, n, [(0, span), (n - span, n)], ""
    if region == "hubs":
        return 0, n, [(a, b) for a, b, _ in hubs] or None, ("" if hubs else "No H3K27ac hubs in the loaded tracks.")
    if region == "custom":
        lo, hi = clamp_window(*(ss.custom_window or best_fit_window(ds)), n)
        return lo, hi, None, ""
    return 0, n, None, ""


def open_window(lo: int, hi: int) -> None:
    """Callback (runs before widgets are created): switch to a custom window."""
    ss.custom_window = (int(lo), int(hi))
    ss.region_choice = "custom"


def _custom_from_slider(n: int) -> None:
    ss.custom_window = clamp_window(*ss.cw_slider, n)


def _custom_from_inputs(ds_n: int, bin0: int, res: int) -> None:
    lo = math.floor(float(ss.cw_start_mb) * 1e6 / res) - bin0
    hi = math.ceil(float(ss.cw_end_mb) * 1e6 / res) - bin0
    ss.custom_window = clamp_window(lo, hi, ds_n)


def ideogram(ds: Dataset, lo: int, hi: int, focus: list[tuple[int, int]] | None) -> str:
    ch = ds.chrom
    total = ch.size
    bands = "".join(f'<i style="width:{(b.end - b.start) / total * 100:.3f}%;background:{T.BAND_COLORS.get(b.stain, "#DDD")}" '
                    f'title="{b.name}"></i>' for b in ch.bands)
    x0 = float(ch.bin_start(ds.bin0 + lo)) / total * 100
    x1 = float(ch.bin_end(ds.bin0 + hi - 1)) / total * 100
    win = f'<span class="win" style="left:{x0:.3f}%;width:{max(x1 - x0, 0.4):.3f}%"></span>'
    marks = ""
    for a, b in (focus or [])[:40]:
        marks += (f'<span style="position:absolute;top:0;bottom:0;left:{float(ch.bin_start(ds.bin0 + a)) / total * 100:.3f}%;'
                  f'width:{max((b - a) * ch.resolution / total * 100, 0.25):.3f}%;background:{T.ACCENT};opacity:.55"></span>')
    cen = ""
    if ch.acen:
        cen = f"cen {ch.acen[0] / 1e6:.1f}–{ch.acen[1] / 1e6:.1f} Mb"
    return (f'<div class="cc-ideo">{bands}{marks}{win}</div>'
            f'<div class="cc-ideo-axis"><span>pter · 0 Mb</span><span>{cen}</span>'
            f'<span>{ch.size / 1e6:.2f} Mb · qter</span></div>')


# ======================================================================================
# Top bar
# ======================================================================================
MARK = ('<svg width="26" height="26" viewBox="0 0 26 26" aria-hidden="true"><path d="M5 20c3-9 5-13 8-13s5 4 8 13" '
        f'fill="none" stroke="{T.INK}" stroke-width="1.6"/><path d="M5 6c3 9 5 13 8 13s5-4 8-13" fill="none" '
        f'stroke="{T.ACCENT}" stroke-width="1.6"/></svg>')

WORKSPACES = {   # key: (number, label, one-line plain-language purpose)
    "3D structure": ("01", "3D structure", "See how one chromosome is folded inside the nucleus, and measure it."),
    "4D dynamics": ("02", "4D dynamics", "Watch the fold change over time, between conditions, or after a DNA rearrangement."),
    "Compare": ("03", "Compare", "Put two biological states side by side; rotating one rotates the other."),
    "Drug lab": ("04", "Drug lab", "Apply a virtual epigenetic drug and see how far it pushes the fold back toward healthy."),
    "Genes": ("05", "Genes", "Find which genes sit in open, active chromatin and which are buried and likely silenced."),
    "Guide": ("06", "Guide", "What everything means, in plain words, with a 2-minute tour."),
}
if ss.get("workspace") not in WORKSPACES:
    ss.workspace = "3D structure"

bar_l, bar_r = st.columns([1, 1.4], vertical_alignment="center")
bar_l.markdown(f'<div class="cc-brand">{MARK}<b>ChronoCell-5D</b><span>chromatin 3D / 4D workstation</span></div>',
               unsafe_allow_html=True)
with bar_r, st.container(key="nav", horizontal=True, horizontal_alignment="right", gap="medium"):
    chrom_choice = st.selectbox("Chromosome", genome.MAIN_CHROMOSOMES, key="chrom_choice",
                                label_visibility="collapsed", width=110)
    slot = slot_files(chrom_choice)
    with st.popover("Data"):
        st.markdown('<p class="cc-eyebrow">Spatial coordinates</p>', unsafe_allow_html=True)
        uploads = st.file_uploader("Coordinates (one or more conditions)",
                                   type=["npz", "pdb", "xyz", "npy", "csv", "pt", "pth"],
                                   accept_multiple_files=True,
                                   help="Colab bundles (.npz, multi-frame), ChronoCell PDB/XYZ exports, "
                                        "predicted_coords.pt or any (N, 3) array. Each file is one condition.")
        html(f'<p class="cc-note">Folder slot: <code>coordinates/{chrom_choice}/</code> · '
             f'{len(slot)} file{"s" if len(slot) != 1 else ""} found. Until coordinates are provided the '
             'reference model is shown.</p>')
        unit = st.selectbox("Coordinate unit", ["Auto", "nm", "Å", "µm", "Model units"],
                            help="Auto keeps files that declare nanometres and calibrates anything else so the "
                                 "median bond equals b₀. Arbitrary network units are never labelled as Å.")
        st.markdown('<p class="cc-eyebrow">Tracks and contacts</p>', unsafe_allow_html=True)
        g_file = st.file_uploader("Graph or contact map", type=["npz", "pt", "pth", "cool", "mcool", "hic", "tsv", "txt"],
                                  help="graph .npz from `python -m chronocell.build_graph` or the Colab notebook "
                                       "(GC + H3K27ac + contacts), or a patient contact map on its own: .cool / .mcool, "
                                       ".hic (needs hic-straw), or a text table (bin_i bin_j count, or chrom pos chrom "
                                       "pos count). A graph*.npz in the folder slot is used automatically.")
        trusted = st.checkbox("Allow unpickling .pt files from my own pipeline",
                              help="PyG Data objects need full unpickling, which can execute code. Only enable for "
                                   "files you produced.")
        seed = st.number_input("Reference model seed", 0, 9999, 7)
    with st.popover("Method"):
        st.markdown('<p class="cc-eyebrow">Physical parameters</p>', unsafe_allow_html=True)
        auto_b0 = st.toggle("b₀ from bead resolution", True,
                            help="b₀ = 50 nm × (resolution / 10 kb)^(1/3): 50 nm at 10 kb, 85 nm at 50 kb.")
        b0_manual = st.number_input("Bond rest length b₀ (nm)", 20.0, 300.0, physics.B0_NM, 5.0, disabled=auto_b0)
        alpha = st.number_input("Contact exponent α", 1.5, 5.0, physics.ALPHA, 0.25,
                                help="M ∝ d^−α; α = 3 is the capture-volume argument and the PASTIS default.")
        dmin_f = st.number_input("Excluded volume d_min / b₀", 0.4, 1.0, physics.D_MIN_FACTOR, 0.05)
    with st.popover("About"):
        html('<p class="cc-meta">ChronoCell-5D reconstructs and analyses the 3D fold of human chromosomes (GRCh38) '
             'from Micro-C contacts, H3K27ac and sequence, and plays 4D trajectories: time courses, disease states '
             'or simulated structural variants. Full guide: <code>APP_GUIDE.md</code>.</p>')
        html('<p class="cc-note">Satorras, Hoogeboom &amp; Welling, ICML 2021 (EGNN) · Lieberman-Aiden et al., Science '
             '2009 (fractal globule) · Varoquaux et al., Bioinformatics 2014 (PASTIS, α = 3) · Lesne et al., Nat Methods '
             '2014 (ShRec3D) · wwPDB Format v3.3 · UCSC hg38 annotation tracks.</p>')
        html(f'<p class="cc-note">Version {VERSION} · PyTorch {"available" if TORCH else "not installed"} · '
             f'Plotly {plotly.__version__} · Streamlit {st.__version__}</p>')

with st.container(key="seg_workspace"):
    workspace = st.segmented_control("Workspace", list(WORKSPACES), required=True, key="workspace",
                                     label_visibility="collapsed",
                                     format_func=lambda w: f"{WORKSPACES[w][0]}  {WORKSPACES[w][1]}")
    workspace = workspace or "3D structure"
html(f'<p class="cc-purpose">{WORKSPACES[workspace][2]}</p>')

# ======================================================================================
# Sidebar: biological state (files found by format) and ChronoAgent settings
# ======================================================================================
bio = states_panel.sidebar(chrom_choice, genome.chrom(chrom_choice).n_bins)
agent_cfg = agent_panel.sidebar_settings()

if workspace == "Guide":            # plain-language guide: needs no data
    guide.render(VERSION)
    st.stop()

# ======================================================================================
# Coordinate slot: uploads > folder > biological-state files > reference model
# ======================================================================================
load_problems: list[str] = []
sources: list[tuple[str, str, bytes] | None] = [None]
for f in uploads or []:
    sources.append((f"Upload · {f.name}", f.name, f.getvalue()))
slot_index: dict[str, int] = {}
for p in slot:
    try:
        sources.append((f"Folder · {p.name}", p.name, read_slot_file(str(p), p.stat().st_mtime)))
        slot_index[os.path.normcase(str(p.resolve()))] = len(sources) - 1
    except OSError as exc:  # vanished / locked / cloud placeholder that cannot be read
        load_problems.append(f"{p.name}: {exc.strerror or exc}")

# Biological-state structures join the same source list, so every view (3D, 4D "Across
# conditions", metrics, ChronoAgent) sees them; each carries its state name and signal track.
state_of: dict[int, str] = {}
track_of: dict[int, tuple] = {}
graph_of: dict[int, tuple] = {}     # a state's own contact map replaces the global graph for its structures
state_src: dict[str, int] = {}
for st_name in S.STATES:
    for sf in bio.plans[st_name].structures:
        try:
            idx = slot_index.get(os.path.normcase(str(Path(sf.path).resolve()))) if sf.path else None
            if idx is None:
                sources.append(states_panel.as_source(sf))
                idx = len(sources) - 1
            state_src[sf.key] = idx
            state_of.setdefault(idx, st_name)
            if (tr := bio.track_for(sf)) is not None:
                track_of[idx] = states_panel.as_source(tr)
            if bio.plans[st_name].contacts:
                graph_of[idx] = states_panel.as_source(bio.plans[st_name].contacts[0])
        except (OSError, KeyError) as exc:
            load_problems.append(f"{sf.name}: {exc}")

graph = None
if g_file is not None:
    graph = (f"Upload · {g_file.name}", g_file.name, g_file.getvalue())
elif (gp := slot_graph(chrom_choice)) is not None:
    try:
        graph = (f"Folder · {gp.name}", gp.name, read_slot_file(str(gp), gp.stat().st_mtime))
    except OSError as exc:
        load_problems.append(f"{gp.name}: {exc.strerror or exc}")

labels_src = [s[0] if s else "Reference model (synthetic)" for s in sources]
default_src = 1 if len(sources) > 1 else 0
src_key = f"source_{chrom_choice}_{len(sources)}"
b0_arg = None if auto_b0 else float(b0_manual)

driven_idx = state_src.get(bio.structure.key) if (bio.drive and bio.structure is not None) else None
ss.setdefault(src_key, default_src)
if driven_idx is not None:
    ss[src_key] = driven_idx          # the Structure source picker follows the state (set before it exists)
src_idx = int(ss.get(src_key, default_src)) if len(sources) > 1 else 0
src_idx = min(max(src_idx, 0), len(sources) - 1)
track_arg = track_of.get(src_idx)
if track_arg is None and bio.drive and bio.track_only:
    try:
        track_arg = states_panel.as_source(bio.track)     # a state with only a track colours the current fold
    except (OSError, KeyError) as exc:
        load_problems.append(f"{bio.track.name}: {exc}")


def load_source(i: int) -> tuple[Dataset, str | None]:
    """Dataset for source i. A graph that cannot be applied is dropped (with a warning) before the
    coordinate file itself is given up on."""
    trk = track_arg if i == src_idx else track_of.get(i)
    cond = state_of.get(i)
    g = graph_of.get(i, graph)
    try:
        return load_dataset(chrom_choice, int(seed), b0_arg, sources[i], unit, g, bool(trusted), trk, cond), None
    except Exception as exc:
        if g is None:
            raise
        g_err = exc
    d = load_dataset(chrom_choice, int(seed), b0_arg, sources[i], unit, None, bool(trusted), trk, cond)
    return d, (f"{g[1]} could not be applied ({g_err}). Tracks and contacts come from the structure file "
               "or the reference instead.")


ds_error = graph_warning = None
try:
    ds, graph_warning = load_source(src_idx)
except Exception as exc:  # keep the workstation usable; report the file problem in place
    ds_error = (labels_src[src_idx], str(exc))
    ds = load_dataset(chrom_choice, int(seed), b0_arg, None, "Auto", None, False)

conditions: list[Dataset] = []
cond_by_idx: dict[int, Dataset] = {}
for i in range(1, len(sources)):
    if i == src_idx and ds_error is None:
        conditions.append(ds)
        cond_by_idx[i] = ds
        continue
    try:
        cond_by_idx[i] = load_source(i)[0]
        conditions.append(cond_by_idx[i])
    except Exception as exc:
        if i != src_idx:
            load_problems.append(f"{labels_src[i]}: {exc}")

# The state shown, and one dataset per state that has a structure (for comparisons)
shown_state = state_of.get(src_idx) if ds_error is None else None
agent_state = shown_state or bio.state
state_datasets: dict[str, Dataset] = {}
for st_name in S.STATES:
    if bio.plans[st_name].structures:
        pick = (bio.structure if st_name == bio.state and bio.structure is not None
                else bio.plans[st_name].structures[0])
        if (d := cond_by_idx.get(state_src.get(pick.key, -1))) is not None:
            state_datasets[st_name] = d
if shown_state is not None:
    state_datasets[shown_state] = ds

ch = ds.chrom
b0 = b0_arg or physics.bond_length_for(ch.resolution)
hubs = hubs_for(ds.key, ds)
html('<hr class="cc-rule">')
if ds_error:
    warning_card(f"Could not load {ds_error[0]}",
                 f"{ds_error[1]} The reference model is shown instead; the sidebar (Biological state) lists how "
                 "each file was recognised.")
if graph_warning:
    warning_card("Graph ignored", graph_warning)
if load_problems:
    warning_card(f"{len(load_problems)} provided file(s) could not be loaded and were skipped", "", load_problems)
if bio.drive and not bio.has_structure and any(not pl.empty for pl in bio.plans.values()):
    banner(f"No structure files for <b>{esc(bio.state)}</b>; showing {esc(labels_src[src_idx])}"
           + (" with this state's signal track." if bio.track_only else "."), "info")

# ======================================================================================
# Meta row
# ======================================================================================
meta_l, meta_r = st.columns([1, 1])
if len(sources) > 1:
    with meta_l:
        st.selectbox("Structure source", range(len(sources)), key=src_key, width=360,
                     format_func=lambda i: labels_src[i], disabled=driven_idx is not None,
                     help=("Set by the Biological state in the sidebar; switch off 'Show this state in the "
                           "workstation' there to choose manually.") if driven_idx is not None else
                          "Each provided file is one condition. The reference model stays available for comparison.")
meta_l.markdown(
    f'<p class="cc-meta">Reconstructing the 3D fold of human {ch.name}<br>from Micro-C contacts, H3K27ac and '
    'sequence composition.</p>', unsafe_allow_html=True)
meta_r.markdown(
    f'<p class="cc-meta right"><span class="cc-locus">GRCh38 · {ch.name} · {ch.resolution / 1000:g} kb</span><br>'
    f'<span class="cc-num">{ds.n:,}</span> beads · <span class="cc-num">{int(ds.valid.sum()):,}</span> assembled · '
    f'<span class="cc-num">{ds.ci.size:,}</span> contacts · <span class="cc-num">{ds.n_frames}</span> '
    f'frame{"s" if ds.n_frames != 1 else ""}</p>', unsafe_allow_html=True)

if ds.is_reference:
    banner(f"<b>Reference model — awaiting spatial coordinates.</b> This is a planted synthetic {ch.name} "
           f"(fractal globule, band-informed tracks, simulated Micro-C). It stays here until coordinates are "
           f"provided: drop Colab bundles into <code>coordinates/{ch.name}/</code>, use Data → Coordinates, or add "
           f"state files (.npy / .pdb) under Biological state in the sidebar.")
elif ds.tracks_are_placeholder and not ds.signal_is_placeholder:
    banner(f"Coordinates and a signal track ({esc(ds.signal_label)}) are provided, but no GC or contacts: GC is a "
           "reference placeholder and reconstruction is disabled. Add a graph file (Data → Graph).", "info")
elif ds.tracks_are_placeholder:
    banner("Coordinates are real, but no tracks/contacts were provided: GC and H3K27ac are reference "
           "placeholders and reconstruction is disabled. Add a graph file (Data → Graph).")
for note in ds.notes[:3]:
    html(f'<p class="cc-note">{note}</p>')

frame_idx = 0
if ds.n_frames > 1 and workspace == "3D structure":
    frame_idx = st.select_slider("Frame", options=list(range(ds.n_frames)), key=f"frame_{ds.key}",
                                 format_func=lambda t: f"{ds.frame_labels[t]}")

# ======================================================================================
# 4D workspace
# ======================================================================================
@st.cache_resource(show_spinner=False, max_entries=32)
def domain_report(key: str, _ds: Dataset, _coords: np.ndarray, lo: int, hi: int, b0: float) -> domains.DomainReport:
    """TADs, compartments, loops and contact decay for one window (measured contacts when present)."""
    n = hi - lo
    ci = cj = cm = None
    if _ds.has_contacts:
        m = (_ds.ci >= lo) & (_ds.ci < hi) & (_ds.cj >= lo) & (_ds.cj < hi)
        if int(m.sum()) >= 50:
            ci, cj, cm = _ds.ci[m] - lo, _ds.cj[m] - lo, _ds.cm[m]
    return domains.analyse(n, _ds.chrom.resolution, b0, _coords, ci, cj, cm, orient=_ds.gc[lo:hi])


@st.cache_data(show_spinner=False, max_entries=32)
def gene_table(key: str, _ds: Dataset, _coords: np.ndarray, lo: int, hi: int, b0: float):
    return G.accessibility_table(_coords, _ds.epi[lo:hi], _ds.valid[lo:hi], b0, _ds.chrom, _ds.bin0 + lo,
                                 not _ds.signal_is_placeholder)


def vkey(scope: str, lo: int, hi: int, reconstruction: bool, coords: np.ndarray) -> str:
    """Cache key of one view (dataset, window, frame or reconstruction)."""
    return f"{ds.key}:{scope}:{lo}:{hi}:{reconstruction}:{hash(coords.tobytes()) if reconstruction else frame_idx}"


def pdb_for(coords: np.ndarray, first_local: int, method: str) -> tuple[str, str] | None:
    """Current-state structure as wwPDB text + a file name, or None if it cannot be written."""
    g0 = ds.bin0 + first_local
    try:
        text, _ = formats.write_pdb(coords, g0, full_track(ds, ds.gc), full_track(ds, ds.epi), ds.epi_ref,
                                    source=ds.structure_label, method=method, chrom=ch)
    except ValueError:
        return None
    tag = S.SHORT_NAMES.get(agent_state, "state") if shown_state else "reference" if ds.is_reference else "input"
    return text, f"ChronoCell_{tag}_{ch.name}_{g0}-{g0 + len(coords) - 1}.pdb"


def analysis_extras(coords: np.ndarray, lo: int, hi: int, key: str) -> tuple[dict, object, object]:
    """Genes, neighbourhoods and the latest drug-lab run for the view (for ChronoAgent and the dossier)."""
    extras: dict = {}
    dom = tab = None
    try:
        dom = domain_report(key, ds, coords, lo, hi, float(b0))
        extras["neighbourhoods"] = dom.summary(ch.resolution)
    except Exception:
        pass
    try:
        tab = gene_table(key, ds, coords, lo, hi, float(b0))
        extras["genes_on_fold"] = G.summary(tab)
        last = ss.get("genes_last")
        if last and last.get("ds_key") == ds.key and last["table"]["expression"].notna().any():
            agree = G.expression_agreement(last["table"])
            if np.isfinite(agree["rho"]):
                extras["genes_on_fold"]["expression_rho"] = round(float(agree["rho"]), 3)
    except Exception:
        pass
    lab = ss.get("drug_lab_last")
    if lab and lab.get("ds_key") == ds.key:
        best = lab.get("ranking")
        extras["drug_lab"] = {"drug": lab["drug"], "mode": lab["mode"], "region": lab["region"],
                              "restoration_pct": lab.get("restoration_pct"),
                              "best_drug": None if best is None or not len(best) else str(best.iloc[0]["drug"])}
    return extras, dom, tab


def dossier_builder(ctx, coords: np.ndarray, lo: int, hi: int, region_name: str, dom, tab):
    """Returns a function (analysis text, engine, question) -> PDF bytes for this view."""
    def build(text: str, engine: str, query: str) -> bytes:
        valid = ds.valid[lo:hi]
        vals = SN.normalise(ds.epi[lo:hi], valid)
        stops = T.SCALES["Epigenomic Signal Heatmap"]
        img = SN.render(coords, vals, stops, valid, size=(1100, 780), title=f"{agent_state} · {region_name}",
                        scale_nm=SN.nice_scale(coords))
        images = []
        healthy = state_datasets.get(S.HEALTHY)
        if (healthy is not None and agent_state != S.HEALTHY and healthy.chrom.name == ch.name
                and healthy.bin0 <= ds.bin0 + lo and healthy.bin0 + healthy.n >= ds.bin0 + hi):
            hx = healthy.frames[0][ds.bin0 + lo - healthy.bin0:ds.bin0 + hi - healthy.bin0]
            _, hx, _ = physics.kabsch_rmsd(coords, hx)
            himg = SN.render(hx, SN.normalise(healthy.epi[ds.bin0 + lo - healthy.bin0:ds.bin0 + hi - healthy.bin0], valid),
                             stops, valid, size=(1100, 780), title=f"Healthy Control · {region_name}",
                             scale_nm=SN.nice_scale(coords))
            images.append(("Healthy Control (left) vs " + agent_state + " (right), signal heatmap",
                           SN.png(SN.side_by_side(himg, img))))
        else:
            images.append((f"{agent_state}, coloured by signal (blue low, magenta high)", SN.png(img)))
        lab = ss.get("drug_lab_last")
        therapy = lab if lab and lab.get("ds_key") == ds.key else None
        return pdf_report.build(ctx, text, engine, query, images, physics.local_density(coords, 1.5 * float(b0)),
                                tab, G.summary(tab) if tab is not None else None,
                                dom.summary(ch.resolution) if dom is not None else None, therapy,
                                software=f"ChronoCell-5D {VERSION}")
    return build


def chrono_agent(coords: np.ndarray, lo: int, hi: int, region: str, reconstruction: bool, scope: str,
                 cards: bool = False) -> None:
    """Metric dashboard (optional) and the ChronoAgent panel for the structure in view."""
    key = vkey(scope, lo, hi, reconstruction, coords)
    try:
        extras, dom, tab = ({}, None, None) if cards else analysis_extras(coords, lo, hi, key)
        ctx = agent_panel.build_context(ds, coords, lo, hi, float(b0), region, agent_state, shown_state is not None,
                                        reconstruction, state_datasets, extras)
    except Exception as exc:  # metrics must never take the workstation down
        warning_card("Structural metrics unavailable for this view", str(exc))
        return
    if cards:
        agent_panel.metric_cards(ctx)
        return
    method = "contact embedding + EGNN" if reconstruction else ("reference model" if ds.is_reference else "input")
    agent_panel.render(ctx, agent_cfg, pdb_for(coords, lo, method), scope,
                       dossier_builder(ctx, coords, lo, hi, region, dom, tab))


def status_bar() -> None:
    html(f'<div class="cc-status"><span>Structure · {ds.structure_label}</span><span>Tracks · {ds.tracks_label}</span>'
         f'<span class="cc-num">{ch.name} · {ch.resolution / 1000:g} kb · b₀ {b0:.0f} nm</span></div>')


if workspace == "4D dynamics":
    four_d.render(ds, conditions or [ds], float(b0), frame_idx)
    chrono_agent(ds.frames[frame_idx], 0, ds.n, "Whole loaded structure", False, "4d")
    status_bar()
    st.stop()

if workspace == "Compare":
    ref_ds = ds if ds.is_reference else load_dataset(chrom_choice, int(seed), b0_arg, None, "Auto", None, False)
    options = [(labels_src[0], ref_ds)] + [(labels_src[i], d) for i, d in sorted(cond_by_idx.items())]
    keys = [d.key for _, d in options]
    cur = keys.index(ds.key) if ds.key in keys else 0
    healthy = state_datasets.get(S.HEALTHY)
    left_i = keys.index(healthy.key) if healthy is not None and healthy.key in keys and healthy.key != ds.key else 0
    right_i = cur if cur != left_i else (1 if len(options) > 1 and left_i == 0 else 0)
    compare.render(options, left_i, right_i, float(b0))
    status_bar()
    st.stop()

if workspace == "Drug lab":
    baseline = state_datasets.get(S.HEALTHY) if agent_state != S.HEALTHY else None
    patient = f"{agent_state} · {ds.structure_label}" if shown_state else ds.structure_label
    if agent_state == S.HEALTHY and shown_state:
        banner("The healthy control is selected. Pick <b>Disease State / Cancer</b> or <b>Senescent State</b> in the "
               "sidebar to treat an abnormal fold.", "info")
    drug_lab.render(ds, baseline, patient, float(b0), frame_idx)
    chrono_agent(ds.frames[frame_idx], 0, ds.n, "Whole loaded structure", False, "lab")
    status_bar()
    st.stop()

if workspace == "Genes":
    state_expr = None
    exp_files = bio.plans[agent_state].expression if shown_state else ()
    if exp_files:
        try:
            state_expr = (G.parse_expression(states_panel.file_bytes(exp_files[0]), exp_files[0].name), exp_files[0].name)
        except Exception as exc:
            warning_card(f"Expression file {exp_files[0].name} could not be read", str(exc))
    genes_view.render(ds, float(b0), frame_idx, state_expr)
    chrono_agent(ds.frames[frame_idx], 0, ds.n, "Whole loaded structure", False, "genes")
    status_bar()
    st.stop()

# ======================================================================================
# 3D workspace: stage header
# ======================================================================================
head_l, head_r = st.columns([1.1, 1], gap="large", vertical_alignment="bottom")
with head_r, st.container(key="seg_region"):
    region = st.segmented_control(
        "Region", list(REGIONS), required=True, key="region_choice",
        format_func=lambda k: f"{list(REGIONS).index(k) + 1:02d}  {SHORT[k]}")
    region = region or "whole"
    if region == "custom":
        # one source of truth (ss.custom_window, clamped); widgets are re-synced to it before creation
        lo_c, hi_c = clamp_window(*(ss.custom_window or best_fit_window(ds)), ds.n)
        ss.custom_window = (lo_c, hi_c)
        ss.cw_slider = (lo_c, hi_c)
        ss.cw_start_mb = round(float(ch.bin_start(ds.bin0 + lo_c)) / 1e6, 3)
        ss.cw_end_mb = round(float(ch.bin_end(ds.bin0 + hi_c - 1)) / 1e6, 3)
        st.slider("Window (bins)", 0, ds.n, key="cw_slider", step=1, label_visibility="collapsed",
                  on_change=_custom_from_slider, args=(ds.n,))
        lo_mb = round(float(ch.bin_start(ds.bin0)) / 1e6, 3)
        hi_mb = round(float(ch.bin_end(ds.bin0 + ds.n - 1)) / 1e6, 3)
        c1, c2 = st.columns(2)
        c1.number_input("Start (Mb)", min_value=lo_mb, max_value=hi_mb, step=0.1, format="%.3f", key="cw_start_mb",
                        on_change=_custom_from_inputs, args=(ds.n, ds.bin0, ch.resolution))
        c2.number_input("End (Mb)", min_value=lo_mb, max_value=hi_mb, step=0.1, format="%.3f", key="cw_end_mb",
                        on_change=_custom_from_inputs, args=(ds.n, ds.bin0, ch.resolution))
    lo, hi, focus, region_note = resolve_region(region, ds, hubs)
    html(ideogram(ds, lo, hi, focus))
    if region_note:
        html(f'<p class="cc-note">{region_note}</p>')

fit_key = f"{ds.key}:{frame_idx}:{lo}:{hi}"
fit = ss.fits.get(fit_key)
g_lo, g_hi = ds.bin0 + lo, ds.bin0 + hi
with head_l:
    html(f'<p class="cc-eyebrow" style="margin-top:22px">Fig. 1 — Reconstructed fold · {ch.name}</p>'
         f'<h1 class="cc-title">{REGIONS[region]}</h1>'
         f'<p class="cc-sub"><span class="cc-locus">{ch.name}:{int(ch.bin_start(g_lo)) + 1:,}–{int(ch.bin_end(g_hi - 1)):,}</span>'
         f' · bins <span class="cc-num">{g_lo:,}–{g_hi - 1:,}</span> · {(hi - lo) * ch.resolution / 1e6:.2f} Mb</p>'
         f'<p class="cc-note">Each bead is {ch.resolution / 1000:g} kb of DNA; the tube follows the DNA from one end of '
         f'the region to the other. Beads that touch in 3D can switch each other’s genes on or off.</p>')
    options = ["Input structure"] + (["EGNN reconstruction"] if fit is not None else [])
    shown = st.segmented_control("Structure", options, default=options[-1] if ss.get("show_fit") else options[0],
                                 required=True, key=f"structure_{fit_key}", label_visibility="collapsed") \
        if len(options) > 1 else options[0]

using_fit = shown == "EGNN reconstruction" and fit is not None and len(fit.coords_nm) == hi - lo
coords_now = ds.frames[frame_idx]
sub = fit.coords_nm if using_fit else coords_now[lo:hi]
phys = analyse(sub, float(b0), float(dmin_f))
labels = hover_labels(ds.key, ds)
focus_mask = None
if focus:
    focus_mask = np.zeros(hi - lo, bool)
    for a, b in focus:
        a2, b2 = max(a, lo) - lo, min(b, hi) - lo
        if b2 > a2:
            focus_mask[a2:b2] = True
    if not focus_mask.any():
        focus_mask = None


# ======================================================================================
# Stage (fragment: display changes re-render only this block)
# ======================================================================================
@st.fragment
def stage(ds: Dataset, lo: int, hi: int, focus_mask: np.ndarray | None, sub: np.ndarray, phys: dict,
          labels: list[str], using_fit: bool, view_key: str, b0: float, coords_now: np.ndarray) -> None:
    ch = ds.chrom
    top_l, top_r = st.columns([4, 1], vertical_alignment="center")
    with top_r, st.container(horizontal=True, horizontal_alignment="right", gap="small"):
        with st.popover("Display"):
            st.segmented_control("Rendering", ["Tube", "Beads", "Line"], default="Tube", required=True, key="disp_style")
            st.selectbox("Colour by", list(T.SCALES), key="disp_colour")
            st.slider("Tube radius (× b₀)", 0.10, 0.45, 0.30, 0.01, key="disp_radius")
            st.slider("Bead size (px)", 2, 12, 5, key="disp_bead")
            st.toggle("Show rest of chromosome", True, key="disp_context")
            st.select_slider("Viewport height", [560, 640, 720, 800, 880], 720, key="disp_height")
    style = ss.get("disp_style") or "Tube"
    colour = ss.get("disp_colour") or "Genomic position"
    idx = ds.gbin(np.arange(lo, hi))
    values = {"Genomic position": idx.astype(float), "GC content": ds.gc[lo:hi],
              "H3K27ac": ds.epi[lo:hi], "Monochrome": np.zeros(hi - lo),
              "Residue Index Spectrum": np.arange(hi - lo, dtype=float),
              "Epigenomic Signal Heatmap": ds.epi[lo:hi]}.get(colour, idx.astype(float))
    if colour in ("A/B compartment", "TAD domains"):
        rep_ = domain_report(vkey("3d", lo, hi, using_fit, sub), ds, sub, lo, hi, float(b0))
        values = rep_.compartment if colour == "A/B compartment" else rep_.tad_labels()
    intensity = viz.encode(values, ds.valid[lo:hi], focus_mask)
    focus_color = T.ACCENT if (colour == "Monochrome" and focus_mask is not None) else None
    ctx = None
    if ss.get("disp_context", True) and not using_fit and (lo > 0 or hi < ds.n):
        ctx = coords_now[::max(1, ds.n // 2500)]
    extent = float(np.max(np.ptp(sub, axis=0))) if len(sub) > 1 else 1.0
    bar = float(10 ** np.floor(np.log10(max(extent / 4, 1.0))))
    fig = viz.viewport(sub, idx, intensity, labels[lo:hi], scale=colour, focus_color=focus_color, style=style,
                       radius=float(ss.get("disp_radius", 0.30)) * b0, bead_px=int(ss.get("disp_bead", 5)),
                       height=int(ss.get("disp_height", 720)), context=ctx, uirevision=view_key,
                       scale_bar_nm=bar, gc=ds.gc[lo:hi], epi=ds.epi[lo:hi], valid=ds.valid[lo:hi], chrom=ch)

    if colour == "Monochrome":
        legend = f'<span class="sw" style="background:{T.INK}"></span> chromatin fibre'
        if focus_mask is not None:
            legend += f'&nbsp;&nbsp;<span class="sw" style="background:{T.ACCENT}"></span> highlighted'
    else:
        grad = ", ".join(T.SCALES[colour])
        ends = {"Genomic position": (f"{float(ch.bin_start(idx[0])) / 1e6:.1f} Mb", f"{float(ch.bin_end(idx[-1])) / 1e6:.1f} Mb"),
                "GC content": ("AT-rich", "GC-rich"), "H3K27ac": ("low", "high"),
                "Residue Index Spectrum": ("bead 1", f"bead {hi - lo:,}"),
                "Epigenomic Signal Heatmap": ("low signal", "high signal"),
                "A/B compartment": ("B · inactive", "A · active"),
                "TAD domains": ("domain", "next domain")}[colour]
        legend = (f'{colour}&nbsp; <span class="cc-num">{ends[0]}</span>'
                  f'<span class="bar" style="background:linear-gradient(90deg,{grad})"></span>'
                  f'<span class="cc-num">{ends[1]}</span>')
    legend += f'&nbsp;&nbsp;<span class="sw" style="background:{T.GHOST}"></span> unassembled'
    top_l.markdown(f'<div class="cc-legend">{legend}</div>', unsafe_allow_html=True)

    with st.container(key="stage"):
        st.plotly_chart(fig, theme=None, key="viewport", width="stretch",
                        config={"displayModeBar": True, "displaylogo": False, "scrollZoom": True, "responsive": True,
                                "modeBarButtonsToRemove": ["zoom3d", "pan3d", "orbitRotation", "tableRotation",
                                                           "handleDrag3d", "resetCameraLastSave3d", "hoverClosest3d",
                                                           "resetCameraDefault3d"],
                                "toImageButtonOptions": {"format": "png", "scale": 3,
                                                         "filename": f"chronocell_{ch.name}_{lo}-{hi}"}})

    fitv = phys["fit"]
    spec_l, spec_r = st.columns([1, 1])
    spec_l.markdown(
        f'<ul class="cc-spec"><li><b>{"EGNN reconstruction" if using_fit else ds.structure_label}</b></li>'
        f'<li>R<sub>g</sub> <span class="cc-num">{fmt(phys["rg"], 0)}</span> nm · '
        f'ν <span class="cc-num">{fmt(fitv.nu, 3)}</span> ({fitv.regime}) · '
        f'<span class="cc-num">{phys["steric"].overlaps}</span> overlaps</li></ul>', unsafe_allow_html=True)
    spec_r.markdown('<p class="cc-help"><kbd>drag</kbd> rotate · <kbd>scroll</kbd> zoom · <kbd>right-drag</kbd> pan · '
                    '<kbd>double-click</kbd> reset<br>hover a bead for its locus · camera icon (top right of the view) '
                    'saves a PNG</p>', unsafe_allow_html=True)


main_l, main_r = st.columns([2.2, 1], gap="large")
with main_l:
    chrono_agent(sub, lo, hi, REGIONS[region], using_fit, "3d", cards=True)
    stage(ds, lo, hi, focus_mask, sub, phys, labels, using_fit, f"{fit_key}:{shown}", float(b0), coords_now)

# ======================================================================================
# Inspector
# ======================================================================================
with main_r, st.container(height=int(ss.get("disp_height", 720)) + 120, key="inspector", border=False):

    # ---- 01 Polymer physics ------------------------------------------------------------
    with st.expander("01   Polymer physics", expanded=True):
        f = phys["fit"]
        ster = phys["steric"]
        regime_tag = "ok" if f.regime in ("fractal globule", "ideal chain", "self-avoiding walk") else "warn"
        kb = ch.resolution / 1000
        readout([
            ("Radius of gyration R<sub>g</sub><small>√((1/N) Σ‖x<sub>i</sub> − x<sub>cm</sub>‖²)</small>", fmt(phys["rg"]), "nm"),
            ("End-to-end distance R<sub>e</sub>", fmt(phys["re"]), "nm"),
            ("Contour length", fmt(phys["contour"] / 1000, 2), "µm"),
            ("Mean bond ⟨b⟩ / b₀", f'{phys["bond_mean"]:.3f} ± {phys["bond_sd"]:.3f}', f"b₀ = {b0:.0f} nm"),
            (f"Scaling exponent ν<small>R(s) ∝ s<sup>ν</sup>, s = {f.fit_range[0] * kb:,.0f}–{f.fit_range[1] * kb:,.0f} kb, "
             f"R² {fmt(f.r2, 3)}</small>", f"{fmt(f.nu, 3)} ± {fmt(f.nu_se, 3)}", ""),
            (f"Excluded-volume overlaps<small>|i−j| &gt; 1, depth &gt; {0.02 * b0:.0f} nm below d<sub>min</sub> = "
             f"{dmin_f * b0:.0f} nm</small>", f'{ster.overlaps:,}', f'max {ster.max_penetration:.2f} nm'),
            ("R<sub>e</sub>² / R<sub>g</sub>²<small>6 for an ideal chain; lower for compact globules</small>",
             fmt(phys["ratio"], 2), ""),
        ])
        html(f'<span class="cc-tag {regime_tag}">{f.regime}</span>')
        if len(f.s):
            st.plotly_chart(viz.scaling_chart(f.s, f.r_rms, f.fit_range, f.nu, resolution=ch.resolution), theme=None,
                            width="stretch", config=T.PLOT_CONFIG, key="scaling")
        html('<p class="cc-note">Accent segment = fit window. Dashed guides: fractal globule (1/3), ideal chain '
             '(1/2), self-avoiding walk (0.59). RMS distance is used, as predicted by polymer theory. Windows shorter '
             'than 40 beads cannot support a fit.</p>')
        st.plotly_chart(viz.bond_histogram(phys["bonds_b0"]), theme=None, width="stretch", config=T.PLOT_CONFIG,
                        key="bonds")

    # ---- 02 Genomic features -------------------------------------------------------------
    with st.expander("02   Genomic features", expanded=False):
        v = ds.valid[lo:hi]
        gcw = ds.gc[lo:hi][v & np.isfinite(ds.gc[lo:hi])]
        epw = ds.epi[lo:hi][v & np.isfinite(ds.epi[lo:hi])]
        empty = np.empty(0, np.int64)
        wci, wcj, wcm = window_contacts(ds.key, ds, lo, hi) if ds.has_contacts else (empty, empty, np.empty(0))
        s_p, p_s, gamma = physics.contact_decay(wci, wcj, wcm, hi - lo) if wci.size else (None, None, float("nan"))
        in_hubs = [h for h in hubs if h[1] > lo and h[0] < hi]
        readout([
            ("Assembled bins", f"{int(v.sum()):,} / {hi - lo:,}", f"{v.mean() * 100:.0f}%"),
            ("Median f<sub>GC</sub>", fmt(np.median(gcw), 3) if gcw.size else "—", ""),
            ("Median H3K27ac", fmt(np.median(epw), 2) if epw.size else "—", ds.tracks_label[:26]),
            ("Contacts in window", f"{wci.size:,}", ""),
            ("Contact decay γ<small>P(s) ∝ s<sup>−γ</sup>; ≈ 3ν for M ∝ d<sup>−3</sup></small>", fmt(gamma, 2), ""),
            ("Enhancer hubs<small>H3K27ac ≥ 97th percentile, assembled bins</small>", f"{len(in_hubs)}", ""),
        ])
        focus_runs = [(max(a, lo), min(b, hi)) for a, b in (focus or []) if b > lo and a < hi]
        st.plotly_chart(viz.tracks_chart(ds.gbin(np.arange(lo, hi)), ds.gc[lo:hi], ds.epi[lo:hi],
                                         [(ds.bin0 + a, ds.bin0 + b) for a, b in focus_runs], chrom=ch),
                        theme=None, width="stretch", config=T.PLOT_CONFIG, key="tracks")
        html('<p class="cc-note">Aligned small multiples, not a dual-axis chart: GC fraction and ChIP signal have '
             'unrelated units. Gaps are unassembled (N) bins; shaded bands are the highlighted region.</p>')
        kinds = ["Contact map"] if wci.size else []
        kind = st.segmented_control("Matrix", kinds + ["Distance map"], default=(kinds or ["Distance map"])[0],
                                    required=True, key=f"matrix_kind_{bool(wci.size)}", label_visibility="collapsed")
        if kind == "Contact map":
            mat, k = physics.coarse_contact_map(wci + lo, wcj + lo, wcm, lo, hi)
            st.plotly_chart(viz.matrix_chart(mat, g_lo, k, "contacts", resolution=ch.resolution), theme=None,
                            width="stretch", config=T.PLOT_CONFIG, key="matrix")
        else:
            mat, k = physics.coarse_distance_map(sub)
            st.plotly_chart(viz.matrix_chart(mat, g_lo, k, "distance", resolution=ch.resolution), theme=None,
                            width="stretch", config=T.PLOT_CONFIG, key="matrix")
        html(f'<p class="cc-note">{"Native" if k == 1 else f"{k * kb:g} kb"} pixels'
             f'{"" if k == 1 else " (window block-averaged)"}.</p>')
        if s_p is not None:
            st.plotly_chart(viz.decay_chart(s_p, p_s, gamma, resolution=ch.resolution), theme=None, width="stretch",
                            config=T.PLOT_CONFIG, key="decay")
        if in_hubs:
            html('<p class="cc-eyebrow" style="margin-top:8px">Strongest hubs in window</p>')
            shown_hubs = in_hubs[:12]
            table = pd.DataFrame([{"Locus": f"{ch.name}:{int(ch.bin_start(ds.bin0 + a)) + 1:,}–{int(ch.bin_end(ds.bin0 + b - 1)):,}",
                                   "Bins": b - a, "Peak H3K27ac": round(p, 2)} for a, b, p in shown_hubs])
            # Keyed by window: a row selected in one window must never index the hub list of another.
            pick = st.dataframe(table, hide_index=True, width="stretch", on_select="rerun", selection_mode="single-row",
                                key=f"hub_table_{ds.key}_{lo}_{hi}", height=min(38 + 35 * len(table), 330))
            rows = list(pick.selection.rows) if pick and pick.selection else []
            if rows and 0 <= rows[0] < len(shown_hubs):
                a, b, _ = shown_hubs[rows[0]]
                wlo, whi = clamp_window(a - 60, b + 60, ds.n)
                st.button("Open this hub as a window", key="open_hub", on_click=open_window, args=(wlo, whi))

    # ---- 03 Model & convergence ----------------------------------------------------------
    with st.expander("03   Model & convergence", expanded=fit is not None):
        for eq in (
            r"\mathcal{L} = \mathcal{L}_{\mathrm{contact}} + \lambda_1\mathcal{L}_{\mathrm{smooth}} + \lambda_2\mathcal{L}_{\mathrm{steric}}",
            r"d^{*}_{ij} = b_0\,(M_{ij}/M_{\mathrm{ref}})^{-1/\alpha}",
            r"\mathcal{L}_{\mathrm{contact}} = \tfrac{1}{|C|}\textstyle\sum_{C}\big((d_{ij}-d^{*}_{ij})/d^{*}_{ij}\big)^{2}",
            r"\mathcal{L}_{\mathrm{smooth}} = \tfrac{1}{N-1}\textstyle\sum_i\big((\lVert x_{i+1}-x_i\rVert - b_0)/b_0\big)^{2}",
            r"\mathcal{L}_{\mathrm{steric}} = \tfrac{1}{N}\textstyle\sum_{|i-j|>1}\big([d_{\min}-d_{ij}]_{+}/b_0\big)^{2}",
            r"x_i' = x_i + \tfrac{1}{|\mathcal{N}(i)|}\textstyle\sum_{j}\hat{u}_{ij}\, s\tanh\phi_x(m_{ij})",
            r"\hat{u}_{ij} = \tfrac{x_i-x_j}{d_{ij}+\epsilon},\;\; m_{ij} = \phi_e(h_i, h_j, \mathrm{RBF}(d_{ij}/b_0), a_{ij})",
        ):
            st.latex(eq)
        cfg = egnn.FitConfig() if TORCH else None
        m_ref = physics.reference_count(ds.ci, ds.cj, ds.cm) if ds.has_contacts else float("nan")
        readout([
            ("Bond rest length b₀", fmt(b0, 0), "nm"),
            ("Contact exponent α · reference count M<sub>ref</sub>", f"{alpha:g} · {fmt(m_ref, 0)}", ""),
            ("Target clip", f"[{dmin_f:g}, {physics.TARGET_CLIP[1]:g}]", "× b₀"),
            ("λ₁ smooth · λ₂ steric", f"{cfg.lambda_smooth:g} · {cfg.lambda_steric:g}" if cfg else "1 · 20", ""),
            ("Compute device", str(egnn.resolve_device()) if TORCH else "—", "CUDA on Colab T4"),
        ])
        n_win = hi - lo
        n_win_contacts = int(((ds.ci >= lo) & (ds.ci < hi) & (ds.cj >= lo) & (ds.cj < hi)).sum()) if ds.has_contacts else 0
        if not TORCH:
            html('<p class="cc-note">PyTorch is not installed, so reconstruction and the equivariance test are '
                 'unavailable. <code>pip install torch</code> and restart.</p>')
        elif not ds.has_contacts:
            html('<p class="cc-note">Reconstruction needs contacts. Provide a graph (Data → Graph, or '
                 f'<code>coordinates/{ch.name}/graph.npz</code>).</p>')
        elif n_win > MAX_FIT_BEADS:
            s_lo, s_hi = best_fit_window(ds)
            html(f'<p class="cc-note">This window has {n_win:,} beads; in-app reconstruction is limited to '
                 f'{MAX_FIT_BEADS:,} (about 20 s per 800 beads on CPU). Use a smaller window, the Colab notebook '
                 f'(T4 GPU), or <code>python -m chronocell.train</code> for the whole chromosome.</p>')
            st.button(f"Use the most contact-dense {s_hi - s_lo} beads "
                      f"({float(ch.bin_start(ds.bin0 + s_lo)) / 1e6:.1f}–{float(ch.bin_end(ds.bin0 + s_hi - 1)) / 1e6:.1f} Mb)",
                      key="suggest_window", on_click=open_window, args=(s_lo, s_hi))
        elif n_win_contacts < MIN_FIT_CONTACTS or int(ds.valid[lo:hi].sum()) < MIN_FIT_CONTACTS:
            html(f'<p class="cc-note">This window has {n_win_contacts:,} contacts and {int(ds.valid[lo:hi].sum()):,} '
                 'assembled bins — too little data to reconstruct (unassembled regions carry no reads). '
                 'Move the window onto assembled sequence.</p>')
        else:
            with st.form("fit_form", border=False):
                c1, c2 = st.columns(2)
                pre = c1.number_input("Embedding epochs", 200, 3000, 800, 100)
                ref = c2.number_input("EGNN refinement epochs", 0, 1000, 100, 50)
                go_fit = st.form_submit_button("Reconstruct this window", type="primary", width="stretch")
            if go_fit:
                wci, wcj, wcm = window_contacts(ds.key, ds, lo, hi)
                bar_ = st.progress(0.0, text="Contact embedding…")
                total = int(pre + ref)

                def on_epoch(stage_name: str, ep: int, tot: int, row: dict) -> None:
                    done = ep if stage_name == "embed" else int(pre) + ep
                    if done % 10 == 0 or done == total:
                        label = "Contact embedding" if stage_name == "embed" else "EGNN refinement"
                        bar_.progress(min(done / total, 1.0), text=f"{label} · epoch {ep}/{tot} · L_contact {row['contact']:.4f}")

                cfg = egnn.FitConfig(prefit_epochs=int(pre), refine_epochs=int(ref), alpha=float(alpha),
                                     d_min=float(dmin_f))
                feats = egnn.node_features(ds.gc[lo:hi], ds.epi[lo:hi], ds.valid[lo:hi])
                try:
                    ss.fits[fit_key] = egnn.fit_structure(n_win, feats, wci, wcj, wcm, cfg, b0=float(b0),
                                                          progress=on_epoch)
                    ss.show_fit = True
                    st.rerun()
                except (ValueError, FloatingPointError) as exc:
                    st.error(f"Reconstruction stopped: {exc}")
        if fit is not None:
            h = fit.history
            rows = [("Runtime", f"{fit.seconds:.1f}", f"s on {fit.config.get('device_used', 'cpu')}"),
                    ("Network parameters", f"{fit.n_params:,}", ""),
                    ("Final L_contact", f"{h['contact'][-1]:.4f}", "")]
            if ds.truth is not None and frame_idx == 0 and len(fit.coords_nm) == hi - lo:
                truth = ds.truth[lo:hi]
                rmsd, _, mirrored = physics.kabsch_rmsd(truth, fit.coords_nm)
                rg_t = physics.radius_of_gyration(truth)
                rows += [("RMSD to planted structure<small>optimal superposition over O(3)</small>", fmt(rmsd), "nm"),
                         ("RMSD / R<sub>g</sub>", f"{rmsd / rg_t:.3f}", "mirror" if mirrored else ""),
                         ("Distance correlation", f"{physics.distance_correlation(truth, fit.coords_nm):.3f}", "Pearson")]
            readout(rows)
            st.plotly_chart(viz.loss_chart(h, fit.config["lambda_smooth"], fit.config["lambda_steric"]), theme=None,
                            width="stretch", config=T.PLOT_CONFIG, key="loss")
            html('<p class="cc-note">Contact data fix a structure only up to reflection, so accuracy is measured over '
                 'O(3). On planted structures, EGNN refinement matches plain coordinate refinement (AUDIT.md §5).</p>')
        if TORCH:
            if st.button("Verify E(3) equivariance", key="equiv"):
                ss.equiv_result = equivariance_report()
            if "equiv_result" in ss:
                e = ss.equiv_result
                readout([("Rotation · max coordinate error", f"{e['rotation_coord_err']:.1e}", ""),
                         ("Reflection · max coordinate error", f"{e['reflection_coord_err']:.1e}", ""),
                         ("Invariant features · max error", f"{max(e['rotation_feat_err'], e['reflection_feat_err']):.1e}", ""),
                         ("Mean displacement (non-trivial map)", f"{e['displacement_scale']:.2f}", "b₀")])

    # ---- 04 Export ------------------------------------------------------------------------
    with st.expander("04   Export", expanded=False):
        scope_opts = ["This window", "Whole loaded structure"] if not using_fit else ["This window"]
        scope = st.segmented_control("Scope", scope_opts, default=scope_opts[0], required=True,
                                     key=f"export_scope_{using_fit}")
        if scope == "Whole loaded structure":
            e_lo, e_coords = 0, coords_now
        else:
            e_lo, e_coords = lo, sub
        start_bin = ds.bin0 + e_lo
        gc_full, epi_full = full_track(ds, ds.gc), full_track(ds, ds.epi)
        method = "contact embedding + EGNN" if using_fit else ("reference model" if ds.is_reference else "input")
        try:
            pdb_text, frame = formats.write_pdb(e_coords, start_bin, gc_full, epi_full, ds.epi_ref,
                                                source=ds.structure_label, method=method, chrom=ch)
            check = formats.validate_pdb(pdb_text)
        except ValueError as exc:
            pdb_text, frame, check = None, None, None
            st.warning(f"PDB export unavailable: {exc}")
        xyz_text = formats.write_xyz(e_coords, start_bin, chrom=ch)
        idx_e = np.arange(start_bin, start_bin + len(e_coords))
        loc_e = np.arange(e_lo, e_lo + len(e_coords))
        table = pd.DataFrame({"bin": idx_e, "chrom": ch.name, "start": ch.bin_start(idx_e), "end": ch.bin_end(idx_e),
                              "x_nm": e_coords[:, 0], "y_nm": e_coords[:, 1], "z_nm": e_coords[:, 2],
                              "f_gc": ds.gc[loc_e], "h3k27ac": ds.epi[loc_e], "assembled": ds.valid[loc_e]})
        e_phys = analyse(e_coords, float(b0), float(dmin_f))
        buf_bundle = io.BytesIO()
        formats.write_bundle(buf_bundle, formats.StructureBundle(
            chrom=ch.name, resolution=ch.resolution, frames=e_coords[None], times=np.zeros(1), labels=[method],
            condition=ds.condition, source=ds.structure_label, start_bin=start_bin))
        report = {
            "software": {"name": "ChronoCell-5D", "version": VERSION, "numpy": np.__version__,
                         "torch": torch.__version__ if TORCH else None},
            "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "genome": {"assembly": genome.ASSEMBLY, "chrom": ch.name, "length_bp": ch.size,
                       "resolution_bp": ch.resolution},
            "export": {"first_bin": int(start_bin), "n_beads": len(e_coords), "units": "nm", "frame": frame_idx,
                       "pdb_unit_nm": frame.unit_nm if frame else None,
                       "pdb_origin_offset_nm": frame.offset_nm.round(3).tolist() if frame else None,
                       "pdb_check": ({"passed": check.ok, "atoms": check.n_atoms, "conect": check.n_conect,
                                      "issues": check.issues} if check else None)},
            "provenance": {"structure": ds.structure_label, "reference_model": ds.is_reference, "method": method,
                           "tracks": ds.tracks_label, "notes": list(ds.notes)},
            "parameters": {"b0_nm": float(b0), "alpha": float(alpha), "d_min_nm": float(dmin_f * b0)},
            "metrics": {"rg_nm": e_phys["rg"], "re_nm": e_phys["re"], "contour_nm": e_phys["contour"],
                        "nu": e_phys["fit"].nu if np.isfinite(e_phys["fit"].nu) else None,
                        "nu_se": e_phys["fit"].nu_se if np.isfinite(e_phys["fit"].nu_se) else None,
                        "regime": e_phys["fit"].regime, "overlaps": e_phys["steric"].overlaps,
                        "l_smooth": e_phys["l_smooth"]},
        }
        if using_fit:
            report["fit"] = {"seconds": fit.seconds, "config": fit.config,
                             "final": {k: fit.history[k][-1] for k in ("contact", "smooth", "steric", "total")}}
        report_text = formats.report_json(report)
        if check is not None:
            tag = ('<span class="cc-tag ok">wwPDB columns · pass</span>' if check.ok
                   else f'<span class="cc-tag warn">wwPDB · {len(check.issues)} issues</span>')
            html(f'{tag} <span class="cc-note">{check.n_atoms:,} ATOM · {check.n_conect:,} CONECT · '
                 f'1 file unit = {frame.unit_nm:g} nm</span>')
        stem = f"{ch.name}_{start_bin}-{start_bin + len(e_coords) - 1}"
        c1, c2 = st.columns(2)
        if pdb_text:
            c1.download_button("PDB", pdb_text, f"{stem}.pdb", "chemical/x-pdb", width="stretch", icon=":material/download:")
        c2.download_button("XYZ", xyz_text, f"{stem}.xyz", "chemical/x-xyz", width="stretch", icon=":material/download:")
        c1.download_button("Bundle (.npz)", buf_bundle.getvalue(), f"{stem}_bundle.npz", "application/octet-stream",
                           width="stretch", icon=":material/download:")
        c2.download_button("Report (JSON)", report_text, f"{stem}_report.json", "application/json", width="stretch",
                           icon=":material/download:")
        c1.download_button("Bins (CSV)", table.to_csv(index=False, float_format="%.4f"), f"{stem}_bins.csv",
                           "text/csv", width="stretch", icon=":material/download:")
        html('<p class="cc-note">PDB coordinates are nanometres shifted into the positive octant so a whole '
             'chromosome fits the %8.3f columns; REMARK 250 records chromosome, window, unit and offset, and '
             'ChronoCell reads them back onto the right loci. Occupancy = f<sub>GC</sub>; B-factor = log-scaled H3K27ac.</p>')
        if check is not None and check.issues:
            st.code("\n".join(check.issues), language="text")
        if pdb_text:
            st.code("\n".join(pdb_text.splitlines()[:22]), language="text")

    # ---- 05 Neighbourhoods ---------------------------------------------------------------
    with st.expander("05   Neighbourhoods (TADs & compartments)", expanded=False):
        html('<p class="cc-note">DNA is organised into self-contained <b>neighbourhoods</b> (TADs, like rooms in a '
             'house) and two <b>compartments</b>: A, the busy city centre of active genes, and B, the quiet suburbs.</p>')
        try:
            rep_nb = domain_report(vkey("3d", lo, hi, using_fit, sub), ds, sub, lo, hi, float(b0))
            sm = rep_nb.summary(ch.resolution)
            readout([
                ("Built from", sm["source"], ""),
                ("TAD-like neighbourhoods<small>insulation-score minima</small>", f"{sm['tads']:,}",
                 f"median {sm['median_tad_mb']} Mb" if sm["median_tad_mb"] else ""),
                ("Active (A) compartment", fmt(100 * sm["a_compartment_fraction"], 0)
                 if sm["a_compartment_fraction"] is not None else "—", "% of region"),
                ("Candidate loops<small>strongest long-range enrichments</small>", f"{sm['loops']:,}", ""),
                ("Contact decay γ<small>P(s) ∝ s<sup>−γ</sup>; ≈ 1 crumpled, ≈ 1.5 loose</small>",
                 fmt(sm["contact_decay_gamma"], 3) if sm["contact_decay_gamma"] is not None else "—", ""),
            ])
            gbins = ds.gbin(np.arange(lo, hi))
            st.plotly_chart(viz.domains_chart(ch.bin_start(gbins) / 1e6, rep_nb.insulation,
                                              [float(ch.bin_start(ds.bin0 + lo + b)) / 1e6 for b in rep_nb.boundaries],
                                              rep_nb.compartment),
                            theme=None, width="stretch", config=T.PLOT_CONFIG, key="domains")
            for note in rep_nb.notes:
                html(f'<p class="cc-note">{esc(note)}</p>')
            html('<p class="cc-note">Colour the fold by <b>A/B compartment</b> or <b>TAD domains</b> under Display.</p>')
        except Exception as exc:
            warning_card("Neighbourhoods could not be computed for this window", str(exc))

# ======================================================================================
# ChronoAgent (fragment: questions and analyses re-render only the panel)
# ======================================================================================
chrono_agent(sub, lo, hi, REGIONS[region], using_fit, "3d")

# ======================================================================================
# Status bar
# ======================================================================================
html(f'<div class="cc-status"><span>Structure · {ds.structure_label}{" (EGNN reconstruction shown)" if using_fit else ""}'
     f'</span><span>Tracks · {ds.tracks_label}</span><span class="cc-num">{ch.name} · {ch.resolution / 1000:g} kb · '
     f'b₀ {b0:.0f} nm · α {alpha:g} · d_min {dmin_f * b0:.0f} nm</span></div>')
