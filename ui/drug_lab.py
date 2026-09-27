"""
Drug lab: a virtual treatment sandbox on the fold (see chronocell/therapy.py for the model).

The dose slider is part of the 3D figure: all doses (0-100 %) are pre-computed, so dragging it
animates the fold in the browser instantly. Everything is labelled as a mechanism simulation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from chronocell import therapy as TH, theme as T, viz
from chronocell.states import HEALTHY
from ui.common import Dataset, banner, clamp_window, esc, fmt, html, warning_card

ss = st.session_state
WEIGHT_SCALE = [[0.0, "#C9CBD9"], [0.35, "#9AA2E6"], [0.7, T.ACCENT], [1.0, "#1A2175"]]


@st.cache_resource(show_spinner="Simulating the treatment at 11 doses…", max_entries=24)
def _treat(key: str, _x: np.ndarray, _sig: np.ndarray, _valid: np.ndarray, b0: float, drug: str,
           _healthy: np.ndarray | None, efficacy: float, _sig_ref: np.ndarray | None) -> TH.TreatmentResult:
    return TH.simulate_treatment(_x, _sig, _valid, b0, drug, _healthy, efficacy, signal_ref=_sig_ref)


@st.cache_data(show_spinner="Testing every drug class at full dose…", max_entries=12)
def _ranking(key: str, _x: np.ndarray, _sig: np.ndarray, _valid: np.ndarray, b0: float, _healthy: np.ndarray,
             efficacy: float, _sig_ref: np.ndarray | None) -> pd.DataFrame:
    return TH.compare_drugs(_x, _sig, _valid, b0, _healthy, efficacy, signal_ref=_sig_ref)


def _card(k: str, v: str, sub: str) -> str:
    return f'<div class="cc-card"><span class="k">{k}</span><span class="v">{v}</span><span class="d">{sub}</span></div>'


def render(ds: Dataset, baseline: Dataset | None, patient_label: str, b0: float, frame: int) -> None:
    ch = ds.chrom
    banner("<b>Simulation, not a prediction of how well a drug works in patients.</b> Each drug is reduced to "
           "<i>where</i> it acts on chromatin and <i>which way</i> it pushes (open or compact); the dose sets how far. "
           "It shows whether a mechanism <i>could</i> move this fold back toward healthy.", "info")
    same = (baseline is not None and baseline.chrom.name == ch.name and baseline.chrom.resolution == ch.resolution
            and baseline.key != ds.key)
    g0 = max(ds.bin0, baseline.bin0) if same else ds.bin0
    g1 = min(ds.bin0 + ds.n, baseline.bin0 + baseline.n) if same else ds.bin0 + ds.n
    if same and g1 - g0 < 40:
        same = False
        g0, g1 = ds.bin0, ds.bin0 + ds.n
    xp_all = ds.frames[frame][g0 - ds.bin0:g1 - ds.bin0]
    xh_all = baseline.frames[0][g0 - baseline.bin0:g1 - baseline.bin0] if same else None
    sig_all = ds.epi[g0 - ds.bin0:g1 - ds.bin0]
    sig_ref = baseline.epi if same else ds.epi

    left, right = st.columns([1, 2.3], gap="large")
    with left:
        html(f'<p class="cc-eyebrow">Patient fold</p><p class="cc-meta"><b>{esc(patient_label)}</b></p>')
        if same:
            html(f'<p class="cc-note">Healthy baseline: <b>{esc(baseline.structure_label)}</b>. Restoration is measured '
                 'against it.</p>')
        else:
            html('<p class="cc-note">No healthy baseline covering the same DNA: the drug acts by mechanism only and '
                 'restoration cannot be measured. Add a Healthy Control state (or switch on demo patients) to measure it.</p>')
        n_all = g1 - g0
        default = "Where the fold is most abnormal" if n_all > 1000 else "Whole region"
        where = st.radio("1 · Region to treat", ["Where the fold is most abnormal", "Whole region", "Custom (Mb)"],
                         index=["Where the fold is most abnormal", "Whole region", "Custom (Mb)"].index(default),
                         key="lab_where", help="The simulation takes a few seconds for up to ~1,000 beads.")
        if where == "Where the fold is most abnormal":
            lo, hi = TH.suggest_window(xp_all, xh_all, sig_all, 800)
        elif where == "Custom (Mb)":
            mb0, mb1 = float(ch.bin_start(g0)) / 1e6, float(ch.bin_end(g1 - 1)) / 1e6
            a = st.number_input("From (Mb)", min_value=round(mb0, 2), max_value=round(mb1, 2),
                                value=round(mb0 + (mb1 - mb0) * 0.4, 2), step=0.5, key="lab_a")
            b = st.number_input("To (Mb)", min_value=round(mb0, 2), max_value=round(mb1, 2),
                                value=round(min(mb1, mb0 + (mb1 - mb0) * 0.4 + 8), 2), step=0.5, key="lab_b")
            lo, hi = clamp_window(int(a * 1e6 / ch.resolution) - g0, int(np.ceil(b * 1e6 / ch.resolution)) - g0, n_all,
                                  min_len=40)
        else:
            lo, hi = 0, n_all
        if hi - lo > 3000:
            lo, hi = TH.suggest_window(xp_all, xh_all, sig_all, 3000)
            html('<p class="cc-note">Limited to the 3,000 most relevant beads to keep the simulation interactive.</p>')
        html(f'<p class="cc-note">Region: {ch.name}:{float(ch.bin_start(g0 + lo)) / 1e6:.2f}–'
             f'{float(ch.bin_end(g0 + hi - 1)) / 1e6:.2f} Mb · {hi - lo:,} beads.</p>')

        x = xp_all[lo:hi]
        xh = xh_all[lo:hi] if xh_all is not None else None
        sig = sig_all[lo:hi]
        valid = ds.valid[g0 - ds.bin0 + lo:g0 - ds.bin0 + hi]
        key = f"{ds.key}:{baseline.key if same else '-'}:{frame}:{g0 + lo}:{g0 + hi}"
        eff_now = float(ss.get("lab_eff", 0.8))
        rank = _ranking(key, x, sig, valid, float(b0), xh, eff_now, sig_ref) if same and xh is not None else None
        best_key = str(rank.iloc[0]["key"]) if rank is not None and rank.iloc[0]["restoration_pct"] > 5 else None
        if best_key and ss.get("lab_best_for") != key:
            ss.lab_drug = best_key                        # pre-select the best match for this fold (set before the widget)
            ss.lab_best_for = key
        keys = list(TH.DRUGS)
        drug = st.radio("2 · Drug class", keys, key="lab_drug",
                        format_func=lambda k: TH.DRUGS[k].name + ("  ★ best match for this fold" if k == best_key else ""),
                        captions=[TH.DRUGS[k].plain for k in keys])
        d = TH.DRUGS[drug]
        html(f'<p class="cc-note"><b>Examples:</b> {esc(d.examples)}.<br><b>Evidence:</b> {esc(d.evidence)}.</p>')
        eff = st.slider("3 · Maximum effect at full dose", 0.2, 1.0, 0.8, 0.05, key="lab_eff",
                        help="How far (at most) the targeted chromatin can be moved toward its healthy position. "
                             "A scenario parameter, not a measured drug property.")

    try:
        res = _treat(key, x, sig, valid, float(b0), drug, xh, float(eff), sig_ref)
    except ValueError as exc:
        warning_card("The simulation could not run on this region", str(exc))
        return
    m = res.metrics
    last = m.iloc[-1]
    ss.drug_lab_last = {"ds_key": ds.key, "drug": d.name, "mode": res.mode, "efficacy": res.efficacy, "plain": d.plain,
                        "table": m, "region": f"{ch.name}:{float(ch.bin_start(g0 + lo)) / 1e6:.2f}-"
                                              f"{float(ch.bin_end(g0 + hi - 1)) / 1e6:.2f} Mb",
                        "restoration_pct": float(last.get("restoration_pct", np.nan))}

    with right:
        cards = []
        if "restoration_pct" in m.columns:
            cards.append(_card("Fold restored at full dose", f"{last['restoration_pct']:.0f}<em>%</em>",
                               "how much closer to the healthy fold"))
        base = res.baseline
        cards.append(_card("Size R<sub>g</sub>", f"{m.rg_nm.iloc[0]:,.0f} → {last['rg_nm']:,.0f}<em>nm</em>",
                           f"healthy {base['rg_nm']:,.0f} nm" if base else "untreated → full dose"))
        cards.append(_card("P(s) slope (−γ)", f"{-m.gamma.iloc[0]:.2f} → {-last['gamma']:.2f}",
                           f"healthy {-base['gamma']:.2f}" if base else "how fast contacts fade with distance"))
        cards.append(_card("Beads the drug reaches", f"{100 * np.mean(res.weights > 0.2):.0f}<em>%</em>",
                           "dark blue in the view"))
        html('<div class="cc-cards">' + "".join(cards) + "</div>")
        for note in res.notes:
            html(f'<p class="cc-note">{esc(note)}</p>')
        if best_key and best_key != drug and float(last.get("restoration_pct", 0) or 0) < 5:
            html(f'<div class="cc-banner info">{esc(d.name)} pushes chromatin {"open" if d.direction > 0 else "closed" if d.direction < 0 else "into loops"}, '
                 f'which is not what this fold needs here, so it barely changes it. The best match is '
                 f'<b>{esc(TH.DRUGS[best_key].name)}</b>.</div>')
        labels = [f"Dose {int(p)}%" for p in m["dose_pct"]]
        hover = [f"{ch.name}:{float(ch.bin_start(g0 + lo + i)) / 1e6:.2f} Mb · targeted {w:.0%}"
                 for i, w in enumerate(res.weights)]
        fig = viz.trajectory_figure(res.frames, res.weights, WEIGHT_SCALE, labels, hover, height=560,
                                    uirevision=f"lab:{key}:{drug}", frame_ms=180)
        with st.container(key="stage_lab"):
            st.plotly_chart(fig, theme=None, key="lab_view", width="stretch",
                            config={"displayModeBar": False, "scrollZoom": True, "responsive": True})
        html('<p class="cc-help">Drag the <b>dose slider</b> under the fold (or press <kbd>Play</kbd>) to raise the dose '
             'in real time · dark blue = chromatin the drug acts on · <kbd>drag</kbd> rotate · <kbd>scroll</kbd> zoom</p>')

    c1, c2 = st.columns([1.2, 1], gap="large")
    with c1:
        html('<p class="cc-eyebrow">Dose–response</p>')
        st.plotly_chart(viz.dose_response_chart(m, res.baseline), theme=None, width="stretch", config=T.PLOT_CONFIG,
                        key="lab_dose")
        html('<p class="cc-note">Dashed lines = the healthy fold. P(s) slope: how quickly the chance of two pieces of DNA '
             'touching falls with their distance along the chromosome (steeper = more open). It is estimated from 3D '
             'proximity and is noisy on short regions.</p>')
    with c2:
        if same and xh is not None:
            html('<p class="cc-eyebrow">Which mechanism fits this fold?</p>')
            rank = _ranking(key, x, sig, valid, float(b0), xh, float(eff), sig_ref)
            st.plotly_chart(viz.drug_bar_chart(rank), theme=None, width="stretch", config=T.PLOT_CONFIG, key="lab_rank")
            best = rank.iloc[0]
            if best["restoration_pct"] > 5:
                html(f'<p class="cc-note">In this simulation <b>{esc(best["drug"])}</b> moves the fold furthest back toward '
                     f'healthy ({best["restoration_pct"]:.0f}% at full dose), because its targets overlap the abnormal '
                     'chromatin. Drugs that push the wrong way for this defect do nothing, by design.</p>')
            else:
                html('<p class="cc-note">No drug class restores this region much: its abnormality does not match any '
                     'mechanism here.</p>')
            ss.drug_lab_last["ranking"] = rank
        tab = m.copy()
        tab["P(s) slope"] = -tab.pop("gamma")
        st.dataframe(tab.round(3), hide_index=True, width="stretch", height=250)
        st.download_button("Dose table (CSV)", tab.to_csv(index=False), f"chronocell_druglab_{drug}.csv", "text/csv",
                           icon=":material/download:", key="lab_csv")
