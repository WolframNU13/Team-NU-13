"""
Genes workspace: structure + epigenome + (optional) transcriptome on one fold.

Every gene's promoter is placed on its bead. Its 3D accessibility (crowding) and signal give a
predicted status (active / intermediate / silenced). With an RNA-seq table the measured values
sit next to the prediction and their rank agreement is reported. Picking a gene lists the genes
that touch it in 3D (promoters within 2 b0), candidate co-regulated partners that can be far
apart along the DNA.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from chronocell import genes as G, physics, theme as T, viz
from ui.common import Dataset, banner, clamp_window, esc, fmt, html, warning_card

ss = st.session_state
STATUS_COLOR = {G.OPEN: T.TERRACOTTA, G.MIDDLE: T.INK_2, G.BURIED: "#2438C9"}
FILTERS = ("All genes", "Cancer genes", "Neuro-disease genes", "Predicted active", "Predicted silenced")


@st.cache_data(show_spinner="Placing genes on the fold…", max_entries=24)
def _table(key: str, _x: np.ndarray, _sig: np.ndarray, _valid: np.ndarray, b0: float, chrom_name: str, resolution: int,
           first_bin: int, signal_real: bool, _expr: pd.Series | None, expr_key: str):
    from chronocell import genome
    ch = genome.chrom(chrom_name, resolution)
    sc = G.bead_scores(_x, _sig, _valid, b0, signal_real)
    tab = G.accessibility_table(_x, _sig, _valid, b0, ch, first_bin, signal_real, _expr, scores=sc)
    return tab, sc["score"], bool(sc["uses_signal"])


def expression_from_upload() -> tuple[pd.Series | None, str]:
    f = st.file_uploader("RNA-seq table (optional)", type=["csv", "tsv", "txt"], key="genes_expr",
                         help="Two columns are enough: gene symbol and a value (TPM, FPKM or counts). Header optional.")
    if f is None:
        return None, ""
    try:
        return G.parse_expression(f.getvalue(), f.name), f.name
    except Exception as exc:
        warning_card("That expression table could not be read", str(exc))
        return None, ""


def render(ds: Dataset, b0: float, frame: int, state_expression: tuple[pd.Series, str] | None) -> None:
    ch = ds.chrom
    all_genes = G.on_chromosome(ch.name)
    g_first, g_last = ds.bin0, ds.bin0 + ds.n
    in_data = all_genes[(all_genes["tss"] // ch.resolution >= g_first) & (all_genes["tss"] // ch.resolution < g_last)]
    names = in_data["name"].tolist()

    c1, c2, c3 = st.columns([1.1, 1, 1])
    pick = c1.selectbox("Find a gene", names, index=None, placeholder="Type a gene name, e.g. BCR, NF2, CHEK2",
                        key=f"genes_pick_{ch.name}", help=f"{len(names):,} genes of {ch.name} lie on the loaded structure.")
    where = c2.segmented_control("Show", ["Around the gene (±1 Mb)", "Whole structure"],
                                 default="Around the gene (±1 Mb)" if pick else "Whole structure", required=True,
                                 key="genes_where")
    flt = c3.selectbox("List", FILTERS, key="genes_filter")
    if pick and where.startswith("Around"):
        tss = int(in_data.loc[in_data["name"] == pick, "tss"].iloc[0])
        centre = tss // ch.resolution - ds.bin0
        half = max(10, int(1_000_000 / ch.resolution))
        lo, hi = clamp_window(centre - half, centre + half, ds.n, min_len=40)
    else:
        lo, hi = 0, ds.n

    expr, expr_label = None, ""
    with st.expander("Add measured expression (RNA-seq) to test the prediction", expanded=False):
        up, up_name = expression_from_upload()
        if up is not None:
            expr, expr_label = up, up_name
        elif state_expression is not None:
            expr, expr_label = state_expression
            html(f'<p class="cc-note">Using the expression table found for this biological state: '
                 f'<code>{esc(expr_label)}</code>.</p>')

    x = ds.frames[frame][lo:hi]
    sig, valid = ds.epi[lo:hi], ds.valid[lo:hi]
    expr_key = f"{expr_label}:{len(expr) if expr is not None else 0}"
    tab, score, uses_signal = _table(f"{ds.key}:{frame}:{lo}:{hi}:{expr_key}", x, sig, valid, float(b0), ch.name,
                                     ch.resolution, ds.bin0 + lo, not ds.signal_is_placeholder, expr, expr_key)
    s = G.summary(tab)
    ss.genes_last = {"ds_key": ds.key, "lo": lo, "hi": hi, "table": tab, "summary": s}

    flagged = tab[tab["category"] != ""]
    cards = [
        ("Genes in view", f"{s['genes_in_view']:,}", f"{ch.name}:{float(ch.bin_start(ds.bin0 + lo)) / 1e6:.1f}–"
                                                      f"{float(ch.bin_end(ds.bin0 + hi - 1)) / 1e6:.1f} Mb"),
        ("Predicted active", f"{s['open']:,}", "open + marked promoters"),
        ("Predicted silenced", f"{s['buried']:,}", "buried in crowded chromatin"),
        ("Disease genes here", f"{len(flagged):,}", ", ".join(flagged["gene"].head(4)) or "none"),
    ]
    html('<div class="cc-cards">' + "".join(f'<div class="cc-card"><span class="k">{k}</span><span class="v">{v}</span>'
                                            f'<span class="d">{esc(d)}</span></div>' for k, v, d in cards) + "</div>")
    if not uses_signal:
        banner("No real signal track for this structure: the prediction uses 3D crowding only.", "info")

    view_l, view_r = st.columns([1.5, 1], gap="large")
    with view_l:
        markers = []
        show = pd.concat([flagged.head(12), tab.head(4), tab.tail(4)]).drop_duplicates("gene")
        for _, r in show.iterrows():
            markers.append({"x": x[int(r["bead"])], "label": r["gene"], "color": STATUS_COLOR[r["status"]], "size": 6,
                            "font": 10, "hover": f"<b>{r['gene']}</b><br>{r['locus']}<br>{r['status']}"})
        if pick and pick in set(tab["gene"]):
            r = tab[tab["gene"] == pick].iloc[0]
            markers.append({"x": x[int(r["bead"])], "label": f"▶ {pick}", "color": T.INK, "size": 11, "font": 13,
                            "symbol": "diamond", "hover": f"<b>{pick}</b><br>{r['locus']}<br>{r['status']}"})
        gb = np.arange(ds.bin0 + lo, ds.bin0 + hi)
        hover = [f"{ch.name}:{float(ch.bin_start(g)) / 1e6:.3f} Mb<br>accessibility {v:+.2f}" if np.isfinite(v)
                 else f"{ch.name}:{float(ch.bin_start(g)) / 1e6:.3f} Mb<br>unassembled" for g, v in zip(gb, score)]
        fig = viz.fibre_figure(x, np.clip(score, -2, 2), viz.ACCESS_SCALE, hover, height=560, cmin=-2, cmax=2,
                               markers=markers, uirevision=f"genes:{ds.key}:{lo}:{hi}", valid=valid)
        with st.container(key="stage_genes"):
            st.plotly_chart(fig, theme=None, key="genes_view", width="stretch",
                            config={"displayModeBar": True, "displaylogo": False, "scrollZoom": True,
                                    "modeBarButtonsToRemove": ["zoom3d", "pan3d", "orbitRotation", "tableRotation",
                                                               "handleDrag3d", "resetCameraLastSave3d", "hoverClosest3d"],
                                    "toImageButtonOptions": {"format": "png", "scale": 3, "filename": "chronocell_genes"}})
        html('<div class="cc-legend">Accessibility&nbsp; <span class="cc-num">buried</span>'
             f'<span class="bar" style="background:linear-gradient(90deg,#1F35C8,#B9B9B3,{T.TERRACOTTA})"></span>'
             '<span class="cc-num">open</span>&nbsp;&nbsp;labels: disease genes and the most open / most buried genes</div>')
    with view_r:
        if pick and pick in set(tab["gene"]):
            r = tab[tab["gene"] == pick].iloc[0]
            bead = int(r["bead"])
            d = np.linalg.norm(x[tab["bead"].to_numpy()] - x[bead], axis=1)
            near = tab.assign(dist=d)[(d < 2 * b0) & (tab["gene"] != pick)]
            near = near.assign(apart_mb=(near["bead"] - bead).abs() * ch.resolution / 1e6).sort_values("dist")
            html(f'<p class="cc-eyebrow">{esc(pick)}</p>'
                 f'<p class="cc-meta"><b>{esc(r["status"])}</b><br>{esc(r["locus"])} · accessibility score '
                 f'<span class="cc-num">{r["score"]:+.2f}</span> · crowding {r["crowding"]:.1f} · signal '
                 f'{fmt(r["signal"], 2)}' + (f' · expression {r["expression"]:.2f}' if np.isfinite(r["expression"]) else "")
                 + (f' · <span class="cc-tag warn">{esc(r["category"])} gene</span>' if r["category"] else "") + "</p>")
            if len(near):
                html('<p class="cc-note"><b>Touches in 3D</b> (promoters within 2 b₀): candidate partners that may be '
                     'co-regulated even when far apart along the DNA.</p>')
                st.dataframe(near[["gene", "status", "apart_mb"]].rename(columns={"apart_mb": "Mb apart along DNA"})
                             .head(12).round(2), hide_index=True, width="stretch")
            else:
                html('<p class="cc-note">No other promoter touches it in 3D.</p>')
        elif pick:
            html('<p class="cc-note">This gene\'s promoter is outside the region shown.</p>')
        else:
            html('<p class="cc-note">Pick a gene above to see its status and which genes it touches in 3D.</p>')
        if expr is not None:
            agree = G.expression_agreement(tab)
            html('<p class="cc-eyebrow" style="margin-top:10px">Prediction vs measured expression</p>')
            if np.isfinite(agree["rho"]):
                verdict = ("the 3D prediction tracks expression" if agree["rho"] >= 0.2 else
                           "weak agreement" if agree["rho"] > 0.05 else "no agreement in this region")
                html(f'<p class="cc-meta">Spearman ρ = <span class="cc-num">{agree["rho"]:+.2f}</span> over '
                     f'{agree["n"]:,} genes: {verdict}.</p>')
                st.plotly_chart(viz.expression_scatter(tab), theme=None, width="stretch", config=T.PLOT_CONFIG,
                                key="genes_scatter")
            else:
                html(f'<p class="cc-note">Only {agree["n"]} genes in view have expression values; at least 20 are needed.</p>')

    view = tab
    if flt == "Cancer genes":
        view = tab[tab["category"] == "cancer"]
    elif flt == "Neuro-disease genes":
        view = tab[tab["category"] == "neuro"]
    elif flt == "Predicted active":
        view = tab[tab["status"] == G.OPEN]
    elif flt == "Predicted silenced":
        view = tab[tab["status"] == G.BURIED]
    html(f'<p class="cc-eyebrow" style="margin-top:8px">{esc(flt)} · {len(view):,}</p>')
    show_cols = ["gene", "locus", "status", "score", "crowding", "signal", "category", "biotype"] + \
                (["expression"] if expr is not None else [])
    st.dataframe(view[show_cols], hide_index=True, width="stretch", height=320,
                 column_config={"score": st.column_config.NumberColumn("accessibility", format="%+.2f",
                                                                        help="> +0.5 open / active; < -0.5 buried / silenced"),
                                "category": st.column_config.TextColumn("flag")})
    st.download_button("Gene table (CSV)", view[show_cols].to_csv(index=False), f"chronocell_genes_{ch.name}.csv",
                       "text/csv", icon=":material/download:", key="genes_csv")
    html(f'<p class="cc-note">Gene annotation: {esc(G.source_note())}. Status is a <b>prediction</b> from how crowded each '
         'promoter is in 3D and how strong its activity mark is, relative to the region shown; flags mark curated '
         'cancer and neuro-disease genes.</p>')
