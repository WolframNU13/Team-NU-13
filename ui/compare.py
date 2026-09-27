"""
Compare workspace: two biological states side by side, cameras linked.

The right structure is superimposed on the left one (optimal rotation, reflection allowed because
contact data fix a fold only up to its mirror image), so the same camera shows the same region of
both. "Difference" colours every bead by how far it sits from its counterpart after superposition:
the pathology lights up.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from chronocell import domains, genes as G, physics, snapshot as SN, theme as T, viz
from ui import agent_panel, sync_view
from ui.common import Dataset, banner, clamp_window, esc, fmt, html, warning_card

COLOURS = ("Difference between the two", "Genomic position", "Epigenomic signal (each its own)")


@st.cache_data(show_spinner="Superimposing the two structures…", max_entries=16)
def _pair(key_l: str, _dl: Dataset, key_r: str, _dr: Dataset, g0: int, g1: int) -> dict:
    xl = _dl.frames[0][g0 - _dl.bin0:g1 - _dl.bin0]
    xr = _dr.frames[0][g0 - _dr.bin0:g1 - _dr.bin0]
    rmsd, xr_al, mirrored = physics.kabsch_rmsd(xl, xr)
    diff = np.linalg.norm(xr_al - xl, axis=1)
    k = 5
    smooth = np.convolve(np.pad(diff, k, mode="edge"), np.ones(2 * k + 1) / (2 * k + 1), mode="valid")
    return {"xl": xl, "xr": xr_al, "diff": diff, "smooth": smooth, "rmsd": rmsd, "mirrored": mirrored}


@st.cache_data(show_spinner=False, max_entries=32)
def _domains(key: str, _x: np.ndarray, b0: float, resolution: int, lo: int, hi: int, _gc: np.ndarray) -> dict:
    return domains.analyse(len(_x), resolution, b0, _x, orient=_gc).summary(resolution)


def _peaks(v: np.ndarray, k: int = 5, spacing: int = 30) -> list[int]:
    out: list[int] = []
    for i in np.argsort(v)[::-1]:
        if all(abs(int(i) - p) >= spacing for p in out):
            out.append(int(i))
        if len(out) == k:
            break
    return out


def render(options: list[tuple[str, Dataset]], default_left: int, default_right: int, b0: float) -> None:
    if len(options) < 2:
        warning_card("Only one structure is available",
                     "Compare needs two: add files for another biological state in the sidebar, switch on "
                     "'Load demo patients', or upload a second coordinate file under Data.")
        return
    labels = [o[0] for o in options]
    c1, c2, c3 = st.columns([1, 1, 1])
    li = c1.selectbox("Left", range(len(options)), index=default_left, format_func=lambda i: labels[i], key="cmp_left",
                      help="Usually the healthy control.")
    ri = c2.selectbox("Right", range(len(options)), index=default_right, format_func=lambda i: labels[i], key="cmp_right",
                      help="Usually the disease or senescent state.")
    colour = c3.selectbox("Colour", COLOURS, key="cmp_colour",
                          help="Difference: how far each piece of DNA moved between the two states (after the two "
                               "structures are overlaid as well as possible).")
    (lab_l, dl), (lab_r, dr) = options[li], options[ri]
    if dl.chrom.name != dr.chrom.name or dl.chrom.resolution != dr.chrom.resolution:
        warning_card("These two structures cannot be compared bead by bead",
                     f"Left is {dl.chrom.name} at {dl.chrom.resolution:,} bp, right is {dr.chrom.name} at "
                     f"{dr.chrom.resolution:,} bp. Pick two structures of the same chromosome and resolution.")
        return
    g0, g1 = max(dl.bin0, dr.bin0), min(dl.bin0 + dl.n, dr.bin0 + dr.n)
    if g1 - g0 < 20:
        warning_card("The two structures do not cover the same part of the chromosome",
                     "They share fewer than 20 beads, so there is nothing to overlay.")
        return
    ch = dl.chrom
    pair = _pair(dl.key, dl, dr.key, dr, g0, g1)
    n_all = g1 - g0

    r1, r2 = st.columns([1.2, 1])
    region = r1.segmented_control("Region", ["Whole shared region", "Most different part", "Custom (Mb)"],
                                  default="Most different part" if n_all > 1200 else "Whole shared region",
                                  required=True, key="cmp_region")
    if region == "Most different part":
        width = min(800, n_all)
        csum = np.concatenate([[0.0], np.cumsum(pair["diff"])])
        a = int(np.argmax(csum[width:] - csum[:-width])) if n_all > width else 0
        lo, hi = a, a + width
    elif region == "Custom (Mb)":
        mb0, mb1 = float(ch.bin_start(g0)) / 1e6, float(ch.bin_end(g1 - 1)) / 1e6
        a_mb = r2.number_input("From (Mb)", min_value=round(mb0, 2), max_value=round(mb1, 2), value=round(mb0, 2),
                               step=0.5, key="cmp_a")
        b_mb = r2.number_input("To (Mb)", min_value=round(mb0, 2), max_value=round(mb1, 2), value=round(mb1, 2),
                               step=0.5, key="cmp_b")
        lo, hi = clamp_window(int(a_mb * 1e6 / ch.resolution) - g0, int(np.ceil(b_mb * 1e6 / ch.resolution)) - g0, n_all)
    else:
        lo, hi = 0, n_all
    xl, xr = pair["xl"][lo:hi], pair["xr"][lo:hi]
    diff = pair["diff"][lo:hi]
    gb = np.arange(g0 + lo, g0 + hi)
    pos_mb = ch.bin_start(gb) / 1e6
    hover = [f"{ch.name}:{p:.2f} Mb<br>{d:,.0f} nm apart" for p, d in zip(pos_mb, diff)]
    vl_valid = dl.valid[g0 - dl.bin0 + lo:g0 - dl.bin0 + hi]

    if colour == COLOURS[0]:
        cmax = float(np.nanpercentile(pair["diff"], 99))
        vl = vr = diff
        scale, cmin, key_hint = viz.DIFF_SCALE, 0.0, "Colour: light = unchanged, dark red = moved the most."
    elif colour == COLOURS[1]:
        vl = vr = pos_mb
        scale, cmin, cmax = T.config_stops("Genomic position"), float(pos_mb.min()), float(pos_mb.max())
        key_hint = "Colour: position along the chromosome, the same on both sides."
    else:
        el = dl.epi[g0 - dl.bin0 + lo:g0 - dl.bin0 + hi]
        er = dr.epi[g0 - dr.bin0 + lo:g0 - dr.bin0 + hi]
        both = np.concatenate([el[np.isfinite(el)], er[np.isfinite(er)]])
        cmin, cmax = (float(np.percentile(both, 1)), float(np.percentile(both, 99))) if both.size else (0.0, 1.0)
        vl, vr = np.clip(el, cmin, cmax), np.clip(er, cmin, cmax)
        scale = T.config_stops("Epigenomic Signal Heatmap")
        key_hint = "Colour: each side's own signal track on a shared scale (blue low, magenta high)."
    peaks = _peaks(pair["smooth"][lo:hi]) if hi - lo > 60 else []
    markers = [{"x": xr[p], "label": f"{pos_mb[p]:.1f} Mb", "color": T.INK, "size": 5, "font": 10} for p in peaks[:3]]
    fl = viz.fibre_figure(xl, vl, scale, hover, height=540, cmin=cmin, cmax=cmax, uirevision="cmp", valid=vl_valid)
    fr = viz.fibre_figure(xr, vr, scale, hover, height=540, cmin=cmin, cmax=cmax, uirevision="cmp", markers=markers,
                          valid=vl_valid)
    sync_view.render_pair(fl, fr, lab_l, lab_r, height=540, key_hint=key_hint)
    html(f'<p class="cc-note">Overlay error (RMSD) over the shared region: <span class="cc-num">{fmt(pair["rmsd"], 0)} nm</span>'
         f'{" · the right structure is the mirror image of the best overlay (contact data cannot tell mirror images apart)" if pair["mirrored"] else ""}. '
         f'Showing {ch.name}:{float(pos_mb[0]):.2f}–{float(pos_mb[-1]):.2f} Mb ({hi - lo:,} beads).</p>')

    # ---- numbers side by side -----------------------------------------------------------
    left_col, right_col = st.columns([1.1, 1], gap="large")
    with left_col:
        html('<p class="cc-eyebrow">The two folds in numbers</p>')
        ml = agent_panel.dataset_metrics(dl, xl, g0 - dl.bin0 + lo, g0 - dl.bin0 + hi, b0)
        mr = agent_panel.dataset_metrics(dr, xr, g0 - dr.bin0 + lo, g0 - dr.bin0 + hi, b0)
        dml = _domains(f"{dl.key}:{g0 + lo}:{g0 + hi}", xl, b0, ch.resolution, lo, hi, dl.gc[g0 - dl.bin0 + lo:g0 - dl.bin0 + hi])
        dmr = _domains(f"{dr.key}:{g0 + lo}:{g0 + hi}:r", xr, b0, ch.resolution, lo, hi, dr.gc[g0 - dr.bin0 + lo:g0 - dr.bin0 + hi])

        def row(name, a, b, nd=0, pct=True, plain=""):
            chg = "—"
            if np.isfinite(a) and np.isfinite(b) and a != 0 and pct:
                chg = f"{(b - a) / abs(a) * 100:+.1f}%"
            elif np.isfinite(a) and np.isfinite(b):
                chg = f"{b - a:+.{max(nd, 2)}f}"
            return {"Measure": name, "Left": fmt(a, nd), "Right": fmt(b, nd), "Change": chg, "What it means": plain}

        f = lambda v: float("nan") if v is None else float(v)  # noqa: E731
        table = pd.DataFrame([
            row("Size, R_g (nm)", ml.rg_nm, mr.rg_nm, 0, True, "bigger = more open, swollen"),
            row("Max 3D span (nm)", ml.span_nm, mr.span_nm, 0, True, "widest distance across the fold"),
            row("Compaction exponent ν", ml.nu, mr.nu, 3, False, "≈ 0.33 compact globule, 0.5 loose coil"),
            row("Packing fraction", ml.packing, mr.packing, 3, True, "how much of its space the DNA fills"),
            row("Crowded beads (%)", 100 * ml.dense_fraction, 100 * mr.dense_fraction, 1, False, "dense, heterochromatin-like spots"),
            row("Mean signal", ml.mean_signal, mr.mean_signal, 3, True, "average activity mark (H3K27ac or own track)"),
            row("Contact decay γ", f(dml.get("contact_decay_gamma")), f(dmr.get("contact_decay_gamma")), 3, False,
                "how fast contacts fade with distance"),
            row("TAD neighbourhoods", f(dml.get("tads")), f(dmr.get("tads")), 0, False, "self-contained DNA 'rooms'"),
        ])
        st.dataframe(table, hide_index=True, width="stretch")
    with right_col:
        html('<p class="cc-eyebrow">Where the two differ most</p>')
        st.plotly_chart(viz.difference_profile(pos_mb, pair["smooth"][lo:hi], [float(pos_mb[p]) for p in peaks]),
                        theme=None, width="stretch", config=T.PLOT_CONFIG, key="cmp_diff")
        rows = []
        for p in peaks:
            start = int(ch.bin_start(gb[max(0, p - 5)]))
            end = int(ch.bin_end(gb[min(len(gb) - 1, p + 5)]))
            names = G.in_region(ch.name, start, end)["name"].head(5).tolist()
            rows.append({"Locus": f"{ch.name}:{pos_mb[p]:.2f} Mb", "Moved (nm)": round(float(pair["smooth"][lo:hi][p])),
                         "Genes there": ", ".join(names) or "—"})
        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        img_l = SN.render(xl, SN.normalise(vl, vl_valid), scale, vl_valid, size=(700, 560), title=f"Left · {lab_l[:48]}",
                          scale_nm=SN.nice_scale(xl))
        img_r = SN.render(xr, SN.normalise(vr, vl_valid), scale, vl_valid, size=(700, 560), title=f"Right · {lab_r[:48]}",
                          scale_nm=SN.nice_scale(xl))
        st.download_button("Side-by-side image (PNG)", SN.png(SN.side_by_side(img_l, img_r)), "chronocell_compare.png",
                           "image/png", icon=":material/download:", key="cmp_png")
    if dl.is_reference or dr.is_reference:
        banner("One side is the synthetic reference model: differences illustrate the method, not biology.", "info")
