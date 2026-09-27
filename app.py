"""
ChronoCell-5D — 3D chromatin structural workstation for human chr22 (GRCh38) at 10 kb.

    streamlit run app.py

Numerics live in the `chronocell` package (physics, egnn, formats, synthetic); this file only
orchestrates state and layout. The viewport is an st.fragment: display settings re-render the
figure without re-running analysis, inference or any other part of the page.
"""

from __future__ import annotations

import datetime as dt
import hashlib
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import plotly
import streamlit as st

from chronocell import features, formats, genome, physics, synthetic, theme as T, viz

try:
    import torch
    from chronocell import egnn
    TORCH = True
except ImportError:  # the viewer, physics and export work without PyTorch
    torch = egnn = None
    TORCH = False

st.set_page_config(page_title="ChronoCell-5D · chr22 3D structure", page_icon="◐", layout="wide",
                   initial_sidebar_state="collapsed")
T.inject()

VERSION = "2.0"
MAX_FIT_BEADS = 2000
REGIONS = {"whole": "Whole chromosome", "centromere": "Centromere", "telomeres": "Telomeric ends",
           "hubs": "Enhancer hubs", "custom": "Custom window"}
SHORT = {"whole": "Whole", "centromere": "Centromere", "telomeres": "Telomeres", "hubs": "Enhancer hubs",
         "custom": "Custom"}
ss = st.session_state
ss.setdefault("fits", {})
ss.setdefault("custom_slider", (2600, 3400))
ss.setdefault("region_choice", "whole")


# ======================================================================================
# Data
# ======================================================================================
@dataclass(frozen=True)
class Dataset:
    key: str
    coords: np.ndarray
    gc: np.ndarray
    epi: np.ndarray
    valid: np.ndarray
    ci: np.ndarray
    cj: np.ndarray
    cm: np.ndarray
    truth: np.ndarray | None
    structure_label: str
    tracks_label: str
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def n(self) -> int:
        return len(self.coords)

    @property
    def has_contacts(self) -> bool:
        return self.ci.size > 0

    @property
    def epi_ref(self) -> float:
        e = self.epi[np.isfinite(self.epi)]
        return float(np.percentile(e, 99.5)) if e.size else 1.0


def _digest(*parts) -> str:
    h = hashlib.sha1()
    for p in parts:
        h.update(p if isinstance(p, bytes) else repr(p).encode())
    return h.hexdigest()[:16]


@st.cache_resource(show_spinner="Generating the synthetic fractal globule…")
def synthetic_chromosome(seed: int, b0: float) -> synthetic.SyntheticChromosome:
    return synthetic.build(seed=seed, b0=b0)


@st.cache_resource(show_spinner=False)
def load_dataset(seed: int, b0: float, s_bytes: bytes | None, s_name: str | None, unit: str,
                 g_bytes: bytes | None, g_name: str | None, trusted: bool) -> Dataset:
    syn = synthetic_chromosome(seed, b0)
    notes: list[str] = []
    coords, truth, s_label = syn.coords, syn.coords, f"Synthetic fractal globule · seed {seed}"
    if s_bytes is not None:
        raw, declared = formats.read_structure(s_bytes, s_name, trusted=trusted)
        if unit == "nm" or (unit == "Auto" and declared == "nm"):
            coords = raw
        elif unit == "Å":
            coords = raw / 10.0
        elif unit == "µm":
            coords = raw * 1000.0
        else:
            coords, factor = physics.calibrate_to_bond_length(raw, b0)
            notes.append(f"Unknown coordinate unit: rescaled ×{factor:.4g} so the median bond equals b₀ = {b0:g} nm.")
        truth, s_label = None, s_name

    gc, epi, valid, ci, cj, cm, t_label = syn.gc, syn.epi, syn.valid, syn.ci, syn.cj, syn.cm, "Synthetic tracks & contacts"
    if g_bytes is not None:
        g = formats.read_graph(g_bytes, g_name, trusted=trusted)
        if len(g.gc) != len(coords):
            notes.append(f"Graph has {len(g.gc):,} bins but the structure has {len(coords):,}; graph ignored.")
        else:
            gc, epi, valid, ci, cj, cm, t_label = g.gc, g.epi, g.valid, g.ci, g.cj, g.cm, g_name
            notes += g.notes
    elif s_bytes is not None:
        ci = cj = np.empty(0, np.int64)
        cm = np.empty(0)
        notes.append("No graph uploaded: contacts unavailable; tracks are synthetic placeholders.")
    if len(gc) != len(coords):   # uploaded structure without a matching genome grid
        gc = np.full(len(coords), np.nan)
        epi = np.full(len(coords), np.nan)
        valid = np.ones(len(coords), bool)
        t_label = "no tracks"
    if s_bytes is None and g_bytes is not None:
        truth = None             # contacts no longer come from the planted structure
    key = _digest(seed, b0, s_bytes or b"", unit, g_bytes or b"", trusted)
    return Dataset(key, coords, gc, epi, valid, ci, cj, cm, truth, s_label, t_label, tuple(notes))


@st.cache_data(show_spinner=False)
def hover_labels(ds_key: str, _ds: Dataset) -> list[str]:
    n = _ds.n
    idx = np.arange(n)
    start, end = genome.bin_start(idx) + 1, genome.bin_end(idx)
    band = [genome.CYTOBANDS[b].name for b in genome.band_for_bins(n)] if n == genome.N_BINS else [""] * n
    out = []
    for i in range(n):
        head = f"<b>chr22:{start[i]:,}–{end[i]:,}</b><br>bin {i:,} · {band[i]}"
        if _ds.valid[i] and np.isfinite(_ds.gc[i]):
            out.append(f"{head}<br>GC {_ds.gc[i]:.3f} · H3K27ac {_ds.epi[i]:.2f}")
        else:
            out.append(f"{head}<br>unassembled (N) · no sequence")
    return out


@st.cache_data(show_spinner=False)
def hubs_for(ds_key: str, _ds: Dataset) -> list[tuple[int, int, float]]:
    return features.signal_hubs(_ds.epi, _ds.valid)


@st.cache_data(show_spinner=False)
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


# ======================================================================================
# Helpers
# ======================================================================================
def html(markup: str) -> None:
    st.markdown(markup, unsafe_allow_html=True)


def readout(rows: list[tuple[str, str, str]]) -> None:
    """Definition list: label (sans), value (mono), unit (sans)."""
    body = "".join(f"<div><dt>{label}</dt><dd>{value}<em>{unit}</em></dd></div>" for label, value, unit in rows)
    html(f'<dl class="cc-readout">{body}</dl>')


def fmt(x: float, nd: int = 1) -> str:
    return "—" if x is None or not np.isfinite(x) else f"{x:,.{nd}f}"


def resolve_region(region: str, n: int, hubs: list) -> tuple[int, int, list[tuple[int, int]] | None]:
    if region == "centromere":
        a, b = genome.interval_to_bins(*genome.ACEN)
        return max(0, a - 300), min(n, b + 300), [(a, min(b, n))]
    if region == "telomeres":
        span = min(100, n // 10)
        return 0, n, [(0, span), (n - span, n)]
    if region == "hubs":
        return 0, n, [(a, b) for a, b, _ in hubs] or None
    if region == "custom":
        lo, hi = ss.custom_slider
        lo, hi = max(0, min(lo, n - 10)), min(n, max(hi, lo + 10))
        return lo, hi, None
    return 0, n, None


def open_window(a: int, b: int) -> None:
    """Callback: switch to a custom window (runs before widgets are instantiated)."""
    ss.custom_slider = (a, b)
    ss.region_choice = "custom"


def ideogram(lo: int, hi: int, focus: list[tuple[int, int]] | None) -> str:
    total = genome.CHROM_SIZE
    bands = "".join(f'<i style="width:{(b.end - b.start) / total * 100:.3f}%;background:{T.BAND_COLORS[b.stain]}" '
                    f'title="{b.name}"></i>' for b in genome.CYTOBANDS)
    x0 = genome.bin_start(lo) / total * 100
    x1 = genome.bin_end(hi - 1) / total * 100
    win = f'<span class="win" style="left:{x0:.3f}%;width:{max(x1 - x0, 0.4):.3f}%"></span>'
    marks = ""
    for a, b in (focus or [])[:40]:
        marks += (f'<span style="position:absolute;top:0;bottom:0;left:{genome.bin_start(a) / total * 100:.3f}%;'
                  f'width:{max((b - a) * genome.RESOLUTION / total * 100, 0.25):.3f}%;background:{T.ACCENT};opacity:.55"></span>')
    return (f'<div class="cc-ideo">{bands}{marks}{win}</div>'
            f'<div class="cc-ideo-axis"><span>pter · 0 Mb</span><span>p11.1 ◆ q11.1</span>'
            f'<span>{genome.CHROM_SIZE / 1e6:.2f} Mb · qter</span></div>')


# ======================================================================================
# Top bar
# ======================================================================================
MARK = ('<svg width="26" height="26" viewBox="0 0 26 26" aria-hidden="true"><path d="M5 20c3-9 5-13 8-13s5 4 8 13" '
        f'fill="none" stroke="{T.INK}" stroke-width="1.6"/><path d="M5 6c3 9 5 13 8 13s5-4 8-13" fill="none" '
        f'stroke="{T.ACCENT}" stroke-width="1.6"/></svg>')

bar_l, bar_r = st.columns([1, 1], vertical_alignment="center")
bar_l.markdown(f'<div class="cc-brand">{MARK}<b>ChronoCell-5D</b><span>3D chromatin workstation</span></div>',
               unsafe_allow_html=True)
with bar_r, st.container(key="nav", horizontal=True, horizontal_alignment="right", gap="medium"):
    with st.popover("Data"):
        st.markdown('<p class="cc-eyebrow">Structure</p>', unsafe_allow_html=True)
        s_file = st.file_uploader("Coordinates", type=["pt", "pth", "pdb", "xyz", "npy", "npz", "csv"],
                                  help="predicted_coords.pt, a ChronoCell PDB/XYZ export, or an (N, 3) array.")
        unit = st.selectbox("Coordinate unit", ["Auto", "nm", "Å", "µm", "Model units"],
                            help="Auto keeps files that declare nanometres and calibrates anything else so the "
                                 "median bond equals b₀. Arbitrary network units are never labelled as Å.")
        st.markdown('<p class="cc-eyebrow">Tracks and contacts</p>', unsafe_allow_html=True)
        g_file = st.file_uploader("Graph", type=["npz", "pt", "pth"],
                                  help="graph .npz from `python -m chronocell.build_graph`, or the legacy graph_data.pt.")
        trusted = st.checkbox("Allow unpickling .pt files from my own pipeline",
                              help="PyG Data objects need full unpickling, which can execute code. Only enable for "
                                   "files you produced.")
        seed = st.number_input("Synthetic seed", 0, 9999, 7, help="Used when no structure is uploaded.")
    with st.popover("Method"):
        st.markdown('<p class="cc-eyebrow">Physical parameters</p>', unsafe_allow_html=True)
        b0 = st.number_input("Bond rest length b₀ (nm)", 20.0, 150.0, physics.B0_NM, 5.0,
                             help="Spatial distance between adjacent 10 kb beads. 50 nm follows 30 nm-fibre coarse "
                                  "graining (≈3 kb per 30 nm monomer) scaled to 10 kb.")
        alpha = st.number_input("Contact exponent α", 1.5, 5.0, physics.ALPHA, 0.25,
                                help="M ∝ d^−α; α = 3 is the capture-volume argument and the PASTIS default.")
        dmin_f = st.number_input("Excluded volume d_min / b₀", 0.4, 1.0, physics.D_MIN_FACTOR, 0.05)
    with st.popover("About"):
        html('<p class="cc-meta">ChronoCell-5D reconstructs the 3D fold of human chromosome 22 (GRCh38) at 10 kb from '
             'Micro-C contacts, H3K27ac and sequence composition, using a contact embedding refined by an '
             'E(3)-equivariant graph network.</p>')
        html('<p class="cc-note">Satorras, Hoogeboom &amp; Welling, ICML 2021 (EGNN) · Lieberman-Aiden et al., Science '
             '2009 (fractal globule) · Varoquaux et al., Bioinformatics 2014 (PASTIS, α = 3) · Rosa &amp; Everaers, '
             'PLoS Comput Biol 2008 (bead scale) · wwPDB Format v3.3 · UCSC hg38 cytoBand / gap / centromeres.</p>')
        html(f'<p class="cc-note">Version {VERSION} · PyTorch {"available" if TORCH else "not installed"} · '
             f'Plotly {plotly.__version__} · Streamlit {st.__version__}</p>')

try:
    ds = load_dataset(int(seed), float(b0), s_file.getvalue() if s_file else None, s_file.name if s_file else None,
                      unit, g_file.getvalue() if g_file else None, g_file.name if g_file else None, bool(trusted))
except Exception as exc:  # keep the workstation usable; report the file problem in place
    st.error(f"Could not load the uploaded files: {exc}")
    ds = load_dataset(int(seed), float(b0), None, None, "Auto", None, None, False)

hubs = hubs_for(ds.key, ds)
html('<hr class="cc-rule">')

# ======================================================================================
# Meta row + stage header
# ======================================================================================
meta_l, meta_r = st.columns([1, 1])
meta_l.markdown(
    '<p class="cc-meta">Reconstructing the 3D fold of human chromosome 22<br>from Micro-C contacts, H3K27ac and '
    'sequence composition.</p>', unsafe_allow_html=True)
assembled = int(ds.valid.sum())
meta_r.markdown(
    f'<p class="cc-meta right"><span class="cc-locus">GRCh38 · chr22 · 10 kb</span><br>'
    f'<span class="cc-num">{ds.n:,}</span> beads · <span class="cc-num">{assembled:,}</span> assembled · '
    f'<span class="cc-num">{ds.ci.size:,}</span> contacts</p>', unsafe_allow_html=True)

head_l, head_r = st.columns([1.1, 1], gap="large", vertical_alignment="bottom")
with head_r, st.container(key="region"):
    region = st.segmented_control(
        "Region", list(REGIONS), required=True, key="region_choice",
        format_func=lambda k: f"{list(REGIONS).index(k) + 1:02d}  {SHORT[k]}")
    a0, b0_ = ss.custom_slider
    ss.custom_slider = (max(0, min(a0, ds.n - 10)), min(ds.n, max(b0_, a0 + 10)))
    if region == "custom":
        st.slider("Window (bins)", 0, ds.n, step=10, key="custom_slider", label_visibility="collapsed")
    lo, hi, focus = resolve_region(region, ds.n, hubs)
    html(ideogram(lo, hi, focus))

fit_key = f"{ds.key}:{lo}:{hi}"
fit = ss.fits.get(fit_key)
with head_l:
    html(f'<p class="cc-eyebrow" style="margin-top:22px">Fig. 1 — Reconstructed fold</p>'
         f'<h1 class="cc-title">{REGIONS[region]}</h1>'
         f'<p class="cc-sub"><span class="cc-locus">{genome.locus(lo).split("-")[0]}–{int(genome.bin_end(hi - 1)):,}</span>'
         f' · bins <span class="cc-num">{lo:,}–{hi - 1:,}</span> · {(hi - lo) * genome.RESOLUTION / 1e6:.2f} Mb</p>')
    options = ["Input structure"] + (["EGNN reconstruction"] if fit is not None else [])
    shown = st.segmented_control("Structure", options, default=options[-1] if ss.get("show_fit") else options[0],
                                 required=True, key=f"structure_{fit_key}", label_visibility="collapsed") \
        if len(options) > 1 else options[0]

using_fit = shown == "EGNN reconstruction" and fit is not None
sub = fit.coords_nm if using_fit else ds.coords[lo:hi]
phys = analyse(sub, float(b0), float(dmin_f))
labels = hover_labels(ds.key, ds)
focus_mask = None
if focus:
    focus_mask = np.zeros(hi - lo, bool)
    for a, b in focus:
        focus_mask[max(a, lo) - lo:max(min(b, hi) - lo, 0)] = True


# ======================================================================================
# Stage (fragment: display changes re-render only this block)
# ======================================================================================
@st.fragment
def stage(ds: Dataset, lo: int, hi: int, focus_mask: np.ndarray | None, sub: np.ndarray, phys: dict,
          labels: list[str], using_fit: bool, view_key: str, b0: float) -> None:
    top_l, top_r = st.columns([4, 1], vertical_alignment="center")
    with top_r, st.container(horizontal=True, horizontal_alignment="right", gap="small"):
        with st.popover("Display"):
            style = st.segmented_control("Rendering", ["Tube", "Beads", "Line"], default="Tube", required=True,
                                         key="disp_style")
            colour = st.selectbox("Colour by", list(T.SCALES), key="disp_colour")
            radius = st.slider("Tube radius (× b₀)", 0.10, 0.45, 0.30, 0.01, key="disp_radius")
            bead_px = st.slider("Bead size (px)", 2, 12, 5, key="disp_bead")
            context = st.toggle("Show rest of chromosome", True, key="disp_context")
            height = st.select_slider("Viewport height", [560, 640, 720, 800, 880], 720, key="disp_height")
    style = ss.get("disp_style", "Tube")
    colour = ss.get("disp_colour", "Genomic position")
    idx = np.arange(lo, hi)
    values = {"Genomic position": idx.astype(float), "GC content": ds.gc[lo:hi],
              "H3K27ac": ds.epi[lo:hi], "Monochrome": np.zeros(hi - lo)}[colour]
    intensity = viz.encode(values, ds.valid[lo:hi], focus_mask)
    focus_color = T.ACCENT if (colour == "Monochrome" and focus_mask is not None) else None
    ctx = None
    if ss.get("disp_context", True) and not using_fit and (lo > 0 or hi < ds.n):
        step = max(1, ds.n // 2500)
        ctx = ds.coords[::step]
    extent = float(np.max(np.ptp(sub, axis=0)))
    bar = float(10 ** np.floor(np.log10(max(extent / 4, 1.0))))
    fig = viz.viewport(sub, idx, intensity, labels[lo:hi], scale=colour, focus_color=focus_color, style=style,
                       radius=ss.get("disp_radius", 0.30) * b0, bead_px=ss.get("disp_bead", 5),
                       height=ss.get("disp_height", 720), context=ctx, uirevision=view_key,
                       scale_bar_nm=bar, gc=ds.gc[lo:hi], epi=ds.epi[lo:hi], valid=ds.valid[lo:hi])

    # legend for the active encoding
    if colour == "Monochrome":
        legend = f'<span class="sw" style="background:{T.INK}"></span> chromatin fibre'
        if focus_mask is not None:
            legend += f'&nbsp;&nbsp;<span class="sw" style="background:{T.ACCENT}"></span> highlighted'
    else:
        grad = ", ".join(T.SCALES[colour])
        ends = {"Genomic position": (f"{lo * genome.RESOLUTION / 1e6:.1f} Mb", f"{hi * genome.RESOLUTION / 1e6:.1f} Mb"),
                "GC content": ("AT-rich", "GC-rich"), "H3K27ac": ("low", "high")}[colour]
        legend = (f'{colour}&nbsp; <span class="cc-num">{ends[0]}</span>'
                  f'<span class="bar" style="background:linear-gradient(90deg,{grad})"></span>'
                  f'<span class="cc-num">{ends[1]}</span>')
    legend += f'&nbsp;&nbsp;<span class="sw" style="background:{T.GHOST}"></span> unassembled'
    top_l.markdown(f'<div class="cc-legend">{legend}</div>', unsafe_allow_html=True)

    with st.container(key="stage"):
        st.plotly_chart(fig, theme=None, key="viewport", width="stretch",
                        config={"displayModeBar": False, "scrollZoom": True, "responsive": True})

    fitv = phys["fit"]
    spec_l, spec_r = st.columns([1, 1])
    spec_l.markdown(
        f'<ul class="cc-spec"><li><b>{"EGNN reconstruction" if using_fit else ds.structure_label}</b></li>'
        f'<li>R<sub>g</sub> <span class="cc-num">{fmt(phys["rg"], 0)}</span> nm · '
        f'ν <span class="cc-num">{fmt(fitv.nu, 3)}</span> ({fitv.regime}) · '
        f'<span class="cc-num">{phys["steric"].overlaps}</span> overlaps</li></ul>', unsafe_allow_html=True)
    spec_r.markdown('<p class="cc-help"><kbd>drag</kbd> rotate · <kbd>scroll</kbd> zoom · <kbd>right-drag</kbd> pan · '
                    '<kbd>double-click</kbd> reset<br>hover a bead for its locus</p>', unsafe_allow_html=True)


main_l, main_r = st.columns([2.2, 1], gap="large")
with main_l:
    stage(ds, lo, hi, focus_mask, sub, phys, labels, using_fit, f"{fit_key}:{shown}", float(b0))

# ======================================================================================
# Inspector
# ======================================================================================
with main_r, st.container(height=ss.get("disp_height", 720) + 120, key="inspector", border=False):

    # ---- 01 Polymer physics ------------------------------------------------------------
    with st.expander("01   Polymer physics", expanded=True):
        f = phys["fit"]
        ster = phys["steric"]
        regime_tag = "ok" if f.regime in ("fractal globule", "ideal chain", "self-avoiding walk") else "warn"
        readout([
            ("Radius of gyration R<sub>g</sub><small>√((1/N) Σ‖x<sub>i</sub> − x<sub>cm</sub>‖²)</small>", fmt(phys["rg"]), "nm"),
            ("End-to-end distance R<sub>e</sub>", fmt(phys["re"]), "nm"),
            ("Contour length", fmt(phys["contour"] / 1000, 2), "µm"),
            ("Mean bond ⟨b⟩ / b₀", f'{phys["bond_mean"]:.3f} ± {phys["bond_sd"]:.3f}', ""),
            (f"Scaling exponent ν<small>R(s) ∝ s<sup>ν</sup>, s = {f.fit_range[0] * 10:,}–{f.fit_range[1] * 10:,} kb, "
             f"R² {fmt(f.r2, 3)}</small>", f"{fmt(f.nu, 3)} ± {fmt(f.nu_se, 3)}", ""),
            (f"Excluded-volume overlaps<small>|i−j| &gt; 1, depth &gt; 1 nm below d<sub>min</sub> = "
             f"{dmin_f * b0:.0f} nm</small>", f'{ster.overlaps:,}', f'max {ster.max_penetration:.2f} nm'),
            ("R<sub>e</sub>² / R<sub>g</sub>²<small>6 for an ideal chain; lower for compact globules</small>",
             fmt(phys["ratio"], 2), ""),
        ])
        html(f'<span class="cc-tag {regime_tag}">{f.regime}</span>')
        st.plotly_chart(viz.scaling_chart(f.s, f.r_rms, f.fit_range, f.nu), theme=None, width="stretch",
                        config=T.PLOT_CONFIG, key="scaling")
        html('<p class="cc-note">Accent segment = fit window. Dashed guides: fractal globule (1/3), ideal chain '
             '(1/2), self-avoiding walk (0.59), anchored at the window start. RMS distance is used, as predicted '
             'by polymer theory.</p>')
        st.plotly_chart(viz.bond_histogram(phys["bonds_b0"]), theme=None, width="stretch", config=T.PLOT_CONFIG,
                        key="bonds")

    # ---- 02 Genomic features -------------------------------------------------------------
    with st.expander("02   Genomic features", expanded=False):
        v = ds.valid[lo:hi]
        gcw = ds.gc[lo:hi][v & np.isfinite(ds.gc[lo:hi])]
        epw = ds.epi[lo:hi][v & np.isfinite(ds.epi[lo:hi])]
        wci, wcj, wcm = window_contacts(ds.key, ds, lo, hi) if ds.has_contacts else (np.array([], int),) * 3
        s_p, p_s, gamma = physics.contact_decay(wci, wcj, wcm, hi - lo) if wci.size else (None, None, float("nan"))
        in_hubs = [h for h in hubs if h[1] > lo and h[0] < hi]
        readout([
            ("Assembled bins", f"{int(v.sum()):,} / {hi - lo:,}", f"{v.mean() * 100:.0f}%"),
            ("Median f<sub>GC</sub>", fmt(np.median(gcw), 3) if gcw.size else "—", ""),
            ("Median H3K27ac", fmt(np.median(epw), 2) if epw.size else "—", ds.tracks_label[:24]),
            ("Contacts in window", f"{wci.size:,}", ""),
            ("Contact decay γ<small>P(s) ∝ s<sup>−γ</sup>; ≈ 3ν for M ∝ d<sup>−3</sup></small>", fmt(gamma, 2), ""),
            ("Enhancer hubs<small>H3K27ac ≥ 97th percentile, assembled bins</small>", f"{len(in_hubs)}", ""),
        ])
        focus_runs = [(max(a, lo), min(b, hi)) for a, b in (focus or []) if b > lo and a < hi]
        st.plotly_chart(viz.tracks_chart(np.arange(lo, hi), ds.gc[lo:hi], ds.epi[lo:hi], focus_runs), theme=None,
                        width="stretch", config=T.PLOT_CONFIG, key="tracks")
        html('<p class="cc-note">Aligned small multiples, not a dual-axis chart: GC fraction and ChIP signal have '
             'unrelated units. Gaps are unassembled (N) bins; shaded bands are the highlighted region.</p>')
        kinds = ["Contact map"] if wci.size else []
        kind = st.segmented_control("Matrix", kinds + ["Distance map"], default=(kinds or ["Distance map"])[0],
                                    required=True, key="matrix_kind", label_visibility="collapsed")
        if kind == "Contact map":
            mat, k = physics.coarse_contact_map(wci + lo, wcj + lo, wcm, lo, hi)
            st.plotly_chart(viz.matrix_chart(mat, lo, k, "contacts"), theme=None, width="stretch",
                            config=T.PLOT_CONFIG, key="matrix")
        else:
            mat, k = physics.coarse_distance_map(sub)
            st.plotly_chart(viz.matrix_chart(mat, lo, k, "distance"), theme=None, width="stretch",
                            config=T.PLOT_CONFIG, key="matrix")
        html(f'<p class="cc-note">{"Native 10 kb pixels" if k == 1 else f"{k * 10} kb pixels (window block-averaged)"}.'
             '</p>')
        if s_p is not None:
            st.plotly_chart(viz.decay_chart(s_p, p_s, gamma), theme=None, width="stretch", config=T.PLOT_CONFIG,
                            key="decay")
        if in_hubs:
            html('<p class="cc-eyebrow" style="margin-top:8px">Strongest hubs in window</p>')
            table = pd.DataFrame([{"Locus": f"chr22:{genome.bin_start(a) + 1:,}–{genome.bin_end(b - 1):,}",
                                   "Bins": b - a, "Peak H3K27ac": round(p, 2)} for a, b, p in in_hubs[:12]])
            pick = st.dataframe(table, hide_index=True, width="stretch", on_select="rerun", selection_mode="single-row",
                                key="hub_table", height=min(38 + 35 * len(table), 330))
            rows = pick.selection.rows if pick and pick.selection else []
            if rows:
                a, b, _ = in_hubs[rows[0]]
                st.button("Open this hub as a window", key="open_hub", on_click=open_window,
                          args=(max(0, a - 60), min(ds.n, b + 60)))

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
        ])
        n_win = hi - lo
        if not TORCH:
            html('<p class="cc-note">PyTorch is not installed, so reconstruction and the equivariance test are '
                 'unavailable. <code>pip install torch</code> and restart.</p>')
        elif not ds.has_contacts:
            html('<p class="cc-note">Reconstruction needs contacts. Upload a graph (.npz or graph_data.pt) under Data.</p>')
        elif n_win > MAX_FIT_BEADS:
            html(f'<p class="cc-note">This window has {n_win:,} beads; in-app reconstruction is limited to '
                 f'{MAX_FIT_BEADS:,} (about 20 s per 800 beads on CPU). Pick a smaller window or run '
                 f'<code>python -m chronocell.train</code> for the whole chromosome.</p>')
            st.button("Use chr22:26.0–34.0 Mb (800 beads)", key="suggest_window", on_click=open_window,
                      args=(2600, 3400))
        else:
            with st.form("fit_form", border=False):
                c1, c2 = st.columns(2)
                pre = c1.number_input("Embedding epochs", 200, 3000, 800, 100)
                ref = c2.number_input("EGNN refinement epochs", 0, 1000, 100, 50)
                go_fit = st.form_submit_button("Reconstruct this window", type="primary", width="stretch")
            if go_fit:
                wci, wcj, wcm = window_contacts(ds.key, ds, lo, hi)
                bar_ = st.progress(0.0, text="Contact embedding…")
                total = pre + ref

                def on_epoch(stage_name: str, ep: int, tot: int, row: dict) -> None:
                    done = ep if stage_name == "embed" else pre + ep
                    if done % 10 == 0 or done == total:
                        label = "Contact embedding" if stage_name == "embed" else "EGNN refinement"
                        bar_.progress(done / total, text=f"{label} · epoch {ep}/{tot} · L_contact {row['contact']:.4f}")

                cfg = egnn.FitConfig(prefit_epochs=int(pre), refine_epochs=int(ref), alpha=float(alpha),
                                     d_min=float(dmin_f))
                feats = egnn.node_features(ds.gc[lo:hi], ds.epi[lo:hi], ds.valid[lo:hi])
                ss.fits[fit_key] = egnn.fit_structure(n_win, feats, wci, wcj, wcm, cfg, b0=float(b0), progress=on_epoch)
                ss.show_fit = True
                st.rerun()
        if fit is not None:
            h = fit.history
            rows = [("Runtime", f"{fit.seconds:.1f}", "s"), ("Network parameters", f"{fit.n_params:,}", ""),
                    ("Final L_contact", f"{h['contact'][-1]:.4f}", "")]
            if ds.truth is not None:
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
                 'O(3). On planted structures, EGNN refinement matches plain coordinate refinement (AUDIT.md §5): '
                 'for a single-structure fit the network is a reparameterisation of the coordinates.</p>')
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
        scope_opts = ["This window", "Whole chromosome"] if not using_fit else ["This window"]
        scope = st.segmented_control("Scope", scope_opts, default=scope_opts[0], required=True, key="export_scope")
        if scope == "Whole chromosome":
            e_lo, e_coords = 0, ds.coords
        else:
            e_lo, e_coords = lo, sub
        method = "contact embedding + EGNN" if using_fit else ("synthetic ground truth" if ds.truth is not None else "input")
        pdb_text, frame = formats.write_pdb(e_coords, e_lo, ds.gc, ds.epi, ds.epi_ref,
                                            source=ds.structure_label, method=method)
        check = formats.validate_pdb(pdb_text)
        xyz_text = formats.write_xyz(e_coords, e_lo)
        idx_e = np.arange(e_lo, e_lo + len(e_coords))
        table = pd.DataFrame({"bin": idx_e, "chrom": genome.CHROM, "start": genome.bin_start(idx_e),
                              "end": genome.bin_end(idx_e), "x_nm": e_coords[:, 0], "y_nm": e_coords[:, 1],
                              "z_nm": e_coords[:, 2], "f_gc": ds.gc[idx_e], "h3k27ac": ds.epi[idx_e],
                              "assembled": ds.valid[idx_e]})
        e_phys = analyse(e_coords, float(b0), float(dmin_f))
        report = {
            "software": {"name": "ChronoCell-5D", "version": VERSION, "numpy": np.__version__,
                         "torch": torch.__version__ if TORCH else None},
            "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "genome": {"assembly": genome.ASSEMBLY, "chrom": genome.CHROM, "length_bp": genome.CHROM_SIZE,
                       "resolution_bp": genome.RESOLUTION},
            "export": {"first_bin": int(e_lo), "n_beads": len(e_coords), "units": "nm",
                       "pdb_unit_nm": frame.unit_nm, "pdb_origin_offset_nm": frame.offset_nm.round(3).tolist(),
                       "pdb_check": {"passed": check.ok, "atoms": check.n_atoms, "conect": check.n_conect,
                                     "issues": check.issues}},
            "provenance": {"structure": ds.structure_label, "method": method, "tracks": ds.tracks_label,
                           "notes": list(ds.notes)},
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
        tag = ('<span class="cc-tag ok">wwPDB columns · pass</span>' if check.ok
               else f'<span class="cc-tag warn">wwPDB · {len(check.issues)} issues</span>')
        html(f'{tag} <span class="cc-note">{check.n_atoms:,} ATOM · {check.n_conect:,} CONECT · '
             f'1 file unit = {frame.unit_nm:g} nm</span>')
        stem = f"chr22_{e_lo}-{e_lo + len(e_coords) - 1}"
        c1, c2 = st.columns(2)
        c1.download_button("PDB", pdb_text, f"{stem}.pdb", "chemical/x-pdb", width="stretch", icon=":material/download:")
        c2.download_button("XYZ", xyz_text, f"{stem}.xyz", "chemical/x-xyz", width="stretch", icon=":material/download:")
        c1.download_button("Report (JSON)", report_text, f"{stem}_report.json", "application/json", width="stretch",
                           icon=":material/download:")
        c2.download_button("Bins (CSV)", table.to_csv(index=False, float_format="%.4f"), f"{stem}_bins.csv",
                           "text/csv", width="stretch", icon=":material/download:")
        html('<p class="cc-note">PDB coordinates are nanometres shifted into the positive octant so a whole '
             'chromosome fits the %8.3f columns; REMARK 250 records unit and offset, and ChronoCell reads them back '
             'exactly. Occupancy = f<sub>GC</sub>; B-factor = log-scaled H3K27ac against the chromosome-wide 99.5th '
             'percentile, so values compare across exports.</p>')
        if check.issues:
            st.code("\n".join(check.issues), language="text")
        st.code("\n".join(pdb_text.splitlines()[:22]), language="text")

# ======================================================================================
# Status bar
# ======================================================================================
note = f" · {ds.notes[0]}" if ds.notes else ""
html(f'<div class="cc-status"><span>Structure · {ds.structure_label}{" (EGNN reconstruction shown)" if using_fit else ""}'
     f'</span><span>Tracks · {ds.tracks_label}{note}</span><span class="cc-num">nm · b₀ {b0:g} · α {alpha:g} · '
     f'd_min {dmin_f * b0:.0f} nm</span></div>')
