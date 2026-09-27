"""
4D workspace: chromatin conformations over time or disease state.

Provided coordinates (multi-frame bundles, or several conditions of one chromosome) are played
back as measured/computed. Until such coordinates exist, a simulated structural-variant scenario
on the current structure is shown and labelled as simulated.
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import streamlit as st

from chronocell import formats, genome, scenarios as SC, snapshot as SN, theme as T, viz
from ui.common import Dataset, banner, clamp_window, fmt, html, readout

OPERATIONS = ("deletion", "duplication", "inversion", "translocation")
KIND_NAMES = ("native", "duplicated copy", "translocation partner", "inverted")
KIND_SCALE = [[0.0, T.INK_2], [0.249, T.INK_2], [0.25, T.OCHRE], [0.499, T.OCHRE],
              [0.5, T.TERRACOTTA], [0.749, T.TERRACOTTA], [0.75, T.VIOLET], [1.0, T.VIOLET]]
DISP_SCALE = [[0.0, "#D9DCF2"], [0.35, "#9AA2E6"], [0.7, T.TERRACOTTA], [1.0, "#6E2408"]]


@st.cache_resource(show_spinner="Simulating the structural variant…", max_entries=16)
def _simulate(ds_key: str, _ds: Dataset, frame: int, operation: str, params: tuple, b0: float,
              title: str, description: str, n_frames: int, sweeps: int) -> SC.Trajectory:
    x = _ds.frames[frame]
    return SC.simulate(x, _ds.chrom, operation, dict(params), b0, title, description, n_frames, sweeps)


def _mb_to_local(ds: Dataset, mb: float) -> int:
    return int(round(mb * 1e6 / ds.chrom.resolution)) - ds.bin0


def _hover(traj: SC.Trajectory, res: int) -> list[str]:
    out = []
    for b, c, k in zip(traj.bead_bin, traj.bead_chrom, traj.bead_kind):
        s = int(b) * res
        out.append(f"<b>{c}:{s + 1:,}–{s + res:,}</b><br>bin {int(b):,} · {KIND_NAMES[int(k)]}")
    return out


def _scenario_controls(ds: Dataset, b0: float) -> tuple[str, dict, str, str] | None:
    """Preset or custom structural variant; every numeric input is clamped to the loaded data."""
    ch = ds.chrom
    here = [p for p in SC.PRESETS.values() if p.chrom == ch.name]
    elsewhere = [p for p in SC.PRESETS.values() if p.chrom != ch.name]
    options = [p.key for p in here] + ["custom"]
    names = {p.key: p.title for p in here} | {"custom": "Custom structural variant"}
    choice = st.selectbox("Scenario", options, format_func=lambda k: names[k], key=f"sc_choice_{ch.name}")
    if elsewhere:
        html('<p class="cc-note">Other presets: ' + " · ".join(f"{p.title} ({p.chrom})" for p in elsewhere) +
             ' — switch chromosome to use them.</p>')
    size_mb = ch.size / 1e6
    lo_mb, hi_mb = float(ch.bin_start(ds.bin0)) / 1e6, float(ch.bin_end(ds.bin0 + ds.n - 1)) / 1e6

    if choice != "custom":
        p = SC.PRESETS[choice]
        region = SC.preset_region(p, ch)
        params = dict(region)
        for key in ("a", "b", "breakpoint"):
            if key in params:
                params[key] = int(params[key]) - ds.bin0
        span_ok = all(0 <= params[k] <= ds.n for k in ("a", "b", "breakpoint") if k in params)
        if not span_ok:
            banner(f"The {p.title} locus lies outside the loaded window "
                   f"({lo_mb:.2f}–{hi_mb:.2f} Mb). Load the whole chromosome to run this preset.")
            return None
        html(f'<p class="cc-meta"><b>{p.disease}</b><br>{p.summary}</p><p class="cc-note">{p.reference}</p>')
        return p.operation, params, p.title, f"{p.disease}. {p.summary}"

    op = st.selectbox("Operation", OPERATIONS, key="sc_op")
    params: dict = {}
    if op == "translocation":
        bp_mb = st.number_input("Breakpoint on this chromosome (Mb)", min_value=round(lo_mb, 2),
                                max_value=round(hi_mb, 2), value=round((lo_mb + hi_mb) / 2, 2), step=0.1,
                                key="sc_bp")
        partners = [c for c in genome.MAIN_CHROMOSOMES if c != ch.name]
        partner = st.selectbox("Partner chromosome", partners, index=partners.index("chr9") if "chr9" in partners else 0,
                               key="sc_partner")
        p_size = genome.chromosome_size(partner) / 1e6
        p_mb = st.number_input(f"Partner segment starts at (Mb, fused through to {partner} qter)", min_value=0.0,
                               max_value=round(p_size - 0.5, 2), value=round(max(p_size - 8.0, 0.0), 2), step=0.5,
                               key="sc_pstart")
        bp = int(np.clip(_mb_to_local(ds, bp_mb), 2, ds.n - 1))
        params = {"breakpoint": bp, "partner": partner, "partner_start_bp": int(p_mb * 1e6)}
        where = f"{ch.name}:{bp_mb:.2f} Mb fused to {partner}:{p_mb:.2f} Mb→qter"
    else:
        c1, c2 = st.columns(2)
        a_mb = c1.number_input("Start (Mb)", min_value=round(lo_mb, 2), max_value=round(hi_mb, 2),
                               value=round(lo_mb + (hi_mb - lo_mb) * 0.45, 2), step=0.1, key="sc_a")
        b_mb = c2.number_input("End (Mb)", min_value=round(lo_mb, 2), max_value=round(hi_mb, 2),
                               value=round(lo_mb + (hi_mb - lo_mb) * 0.5, 2), step=0.1, key="sc_b")
        a, b = clamp_window(_mb_to_local(ds, a_mb), _mb_to_local(ds, b_mb), ds.n, min_len=2)
        params = {"a": a, "b": b}
        if op == "duplication":
            params["copies"] = int(st.number_input("Extra copies", min_value=1, max_value=4, value=1, key="sc_copies"))
        where = f"{ch.name}:{float(ch.bin_start(ds.bin0 + a)) / 1e6:.2f}–{float(ch.bin_end(ds.bin0 + b - 1)) / 1e6:.2f} Mb"
    return op, params, f"Custom {op}", f"User-defined {op} at {where}."


@st.fragment
def render(ds: Dataset, conditions: list[Dataset], b0: float, frame: int) -> None:
    """The whole 4D workspace is one fragment: mode, scenario, colour and export changes re-run
    only this block (the animation itself plays client-side)."""
    ch = ds.chrom
    provided_multi = ds.n_frames > 1
    comparable = [d for d in conditions if d.n == ds.n and d.bin0 == ds.bin0 and d.chrom.name == ch.name]
    modes = []
    if provided_multi:
        modes.append("Provided frames")
    if len(comparable) > 1:
        modes.append("Across conditions")
    modes.append("Simulated scenario")

    head_l, head_r = st.columns([1.1, 1], gap="large", vertical_alignment="bottom")
    with head_r, st.container(key="seg_fourd"):
        mode = st.segmented_control("Source of the 4th dimension", modes, default=modes[0], required=True,
                                    key=f"fourd_mode_{ds.key}")

    col_view, col_insp = st.columns([2.2, 1], gap="large")
    with col_insp, st.container(height=900, key="inspector", border=False):
        with st.expander("01   Scenario & frames", expanded=True):
            if mode == "Simulated scenario":
                spec = _scenario_controls(ds, b0)
                n_frames = st.slider("Frames", 8, 48, 24, 4, key="sc_frames")
                sweeps = st.slider("Relaxation sweeps per frame", 5, 40, 15, 5, key="sc_sweeps")
                if spec is None:
                    return
                op, params, title, description = spec
                traj = _simulate(ds.key, ds, frame, op, tuple(sorted(params.items())), float(b0), title, description,
                                 int(n_frames), int(sweeps))
            elif mode == "Across conditions":
                frames = np.stack([d.frames[0] for d in comparable])
                traj = SC.from_bundle(frames, np.arange(len(comparable)), [d.condition or d.structure_label[:24]
                                                                           for d in comparable],
                                      ch, ds.bin0, "Across conditions",
                                      "First frame of each provided condition, aligned onto the first.", "condition")
                html('<p class="cc-note">Conditions: ' + " → ".join(traj.labels) + '</p>')
            else:
                traj = SC.from_bundle(ds.frames, ds.frame_times, list(ds.frame_labels), ch, ds.bin0,
                                      f"{ds.condition} · {ds.n_frames} frames", ds.structure_label, ds.time_unit)
                html(f'<p class="cc-note">{ds.n_frames} frames from <code>{ds.structure_label}</code>.</p>')

        m = SC.frame_metrics(traj)
        colour = st.segmented_control("Colour", ["Genomic position", "Displacement", "Segment"],
                                      default="Displacement" if traj.simulated else "Genomic position",
                                      required=True, key="fourd_colour")

        with st.expander("02   Dynamics", expanded=True):
            last = traj.n_frames - 1
            k_max = int(np.argmax(m["disp"][last]))
            readout([
                ("Frames · beads", f"{traj.n_frames} · {traj.frames.shape[1]:,}", ""),
                ("R<sub>g</sub> first → last", f"{fmt(m['rg'][0], 0)} → {fmt(m['rg'][last], 0)}", "nm"),
                ("RMSD last vs first<small>proper-rotation superposition</small>", fmt(m["rmsd"][last]), "nm"),
                ("Largest displacement", fmt(m["disp"][last][k_max], 0),
                 f"nm at {traj.bead_chrom[k_max]}:{int(traj.bead_bin[k_max]) * ch.resolution / 1e6:.2f} Mb"),
                ("Median bond, last frame", fmt(m["median_bond"][last] / b0, 3), "× b₀"),
            ])
            for e in traj.events:
                html(f'<p class="cc-note">{e}</p>')
            st.plotly_chart(viz.timeseries_chart(traj.times, [("R_g (nm)", m["rg"], T.INK), ("RMSD (nm)", m["rmsd"], T.ACCENT),
                                                              ("median bond (nm)", m["median_bond"], T.TERRACOTTA)],
                                                 traj.time_unit), theme=None, width="stretch", config=T.PLOT_CONFIG,
                            key="fourd_ts")
            pos_mb = traj.bead_bin * ch.resolution / 1e6
            st.plotly_chart(viz.displacement_profile(pos_mb, m["disp"][last], traj.bead_kind), theme=None,
                            width="stretch", config=T.PLOT_CONFIG, key="fourd_disp")
            html('<p class="cc-note">Displacement of every bead between the first and last frame. Colours: ink = native, '
                 'ochre = duplicated copy, terracotta = translocation partner, violet = inverted.</p>')

        with st.expander("03   Export trajectory", expanded=False):
            res_seq = traj.bead_bin + 1
            segs = np.array([formats.segment_id(c) for c in traj.bead_chrom])
            pdb = formats.write_pdb_trajectory(traj.frames, res_seq, segs, ch, traj.title,
                                               "simulated relaxation" if traj.simulated else "provided frames")
            chk = formats.validate_pdb(pdb)
            html(('<span class="cc-tag ok">wwPDB multi-model · pass</span>' if chk.ok
                  else f'<span class="cc-tag warn">{len(chk.issues)} issues</span>') +
                 f' <span class="cc-note">{traj.n_frames} MODEL records — plays as a movie in PyMOL / ChimeraX</span>')
            buf = io.BytesIO()
            np.savez_compressed(buf, frames=traj.frames.astype(np.float32), times=traj.times, labels=np.array(traj.labels),
                                bead_bin=traj.bead_bin, bead_chrom=np.array(traj.bead_chrom, dtype=str),
                                bead_kind=traj.bead_kind, chrom=np.array(ch.name), resolution=np.array(ch.resolution),
                                simulated=np.array(traj.simulated), title=np.array(traj.title),
                                format=np.array("chronocell-trajectory-1"))
            table = pd.DataFrame({"frame": np.arange(traj.n_frames), "label": traj.labels, "time": traj.times,
                                  "rg_nm": m["rg"], "re_nm": m["re"], "rmsd_nm": m["rmsd"],
                                  "median_bond_nm": m["median_bond"]})
            stem = f"{ch.name}_{traj.title.split()[0].lower()}_4d"
            c1, c2 = st.columns(2)
            c1.download_button("Multi-model PDB", pdb, f"{stem}.pdb", "chemical/x-pdb", width="stretch",
                               icon=":material/download:")
            c2.download_button("Trajectory (.npz)", buf.getvalue(), f"{stem}.npz", "application/octet-stream",
                               width="stretch", icon=":material/download:")
            c1.download_button("Metrics (CSV)", table.to_csv(index=False, float_format="%.3f"), f"{stem}_metrics.csv",
                               "text/csv", width="stretch", icon=":material/download:")
            gif_key = f"gif:{ds.key}:{traj.title}:{traj.n_frames}"
            have = st.session_state.get("fourd_gif", (None,))[0] == gif_key
            if not have and c2.button("Build animated GIF", width="stretch", icon=":material/animation:",
                                      key="fourd_gif_build", help="A slowly spinning animation of all frames, for slides."):
                with st.spinner("Rendering frames…"):
                    if traj.simulated:
                        vals = np.stack([SN.normalise(d) for d in m["disp"]])
                        stops = DISP_SCALE
                    else:
                        vals = SN.normalise(np.arange(traj.frames.shape[1], dtype=float))
                        stops = T.SCALES["Genomic position"]
                    st.session_state.fourd_gif = (gif_key, SN.gif(traj.frames, vals, stops, list(traj.labels), spin=3.0))
                have = True
            if have:
                c2.download_button("Animated GIF", st.session_state.fourd_gif[1], f"{stem}.gif", "image/gif",
                                   width="stretch", icon=":material/animation:", key="fourd_gif_dl")

    # ---- title + stage ----------------------------------------------------------------------
    with head_l:
        html(f'<p class="cc-eyebrow" style="margin-top:22px">Fig. 2 — 4D dynamics · {ch.name}</p>'
             f'<h1 class="cc-title">{traj.title}</h1>'
             f'<p class="cc-sub">{traj.n_frames} frames · time axis: {traj.time_unit}</p>')
    with col_view:
        if traj.simulated:
            banner("<b>Simulated scenario, not measured data.</b> The rearrangement is applied to the current structure "
                   "and the polymer is relaxed (bond lengths + excluded volume); frames are relaxation sweeps, not "
                   "biological time. It shows the geometric consequence of the karyotype under the polymer model. "
                   "Provide time-resolved or disease-state coordinates (Colab bundles in <code>coordinates/"
                   f"{ch.name}/</code> or Data → Coordinates) to replace it.")
        elif ds.is_reference:
            banner("Frames are derived from the reference model. Provide coordinates to analyse real data.", "info")
        hover = _hover(traj, ch.resolution)
        if colour == "Displacement":
            values, scale = m["disp"], DISP_SCALE
        elif colour == "Segment":
            values, scale = traj.bead_kind.astype(float) / 3.0, KIND_SCALE
        else:
            values, scale = np.arange(traj.frames.shape[1], dtype=float), T.config_stops("Genomic position")
        fig = viz.trajectory_figure(traj.frames, values, scale, traj.labels, hover, height=720,
                                    uirevision=f"4d:{ds.key}:{traj.title}")
        with st.container(key="stage"):
            st.plotly_chart(fig, theme=None, key="fourd_view", width="stretch",
                            config={"displayModeBar": False, "scrollZoom": True, "responsive": True})
        html('<p class="cc-help"><kbd>Play</kbd> animates in the browser (no reloads) · drag the slider to scrub · '
             '<kbd>drag</kbd> rotate · <kbd>scroll</kbd> zoom</p>')
