"""
ChronoAgent UI: sidebar settings, the metric dashboard and the interpreter panel.

The panel is an st.fragment, so typing a question or running an analysis re-renders only the
panel. Without an API key the deterministic heuristic engine answers instantly; with a key the
selected free LLM (Gemini or OpenRouter) is asked on demand and any failure falls back to the
heuristic answer with a warning card. The key lives only in this browser session's widget state.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

import numpy as np
import streamlit as st

from chronocell import agent as A, genome
from chronocell.states import HEALTHY, SHORT_NAMES
from ui.common import Dataset, esc, html, warning_card

ss = st.session_state
AUTO = "Auto-detect from key"
TITLE = "🤖 ChronoAgent: Structural Genomics Interpreter"


@dataclass(frozen=True)
class AgentSettings:
    api_key: str
    provider: str | None          # resolved provider, None = offline
    model: str
    note: str

    @property
    def online(self) -> bool:
        return bool(self.api_key) and self.provider is not None


# ======================================================================================
# Sidebar
# ======================================================================================
def sidebar_settings() -> AgentSettings:
    with st.sidebar:
        html('<p class="cc-side-title">ChronoAgent</p>')
        key = st.sidebar.text_input("AI API Key", type="password", key="agent_api_key",
                                    help="Optional. A free Google AI Studio key (Gemini, starts with AIza) or an "
                                         "OpenRouter key (sk-or-…). Kept only in this browser session and sent only "
                                         "to that provider, in a request header. Without a key the offline "
                                         "heuristic engine answers.")
        choice = st.selectbox("Provider", [AUTO, A.GEMINI, A.OPENROUTER], key="agent_provider")
        model = st.text_input("Model", key="agent_model", placeholder="blank = free default models",
                              help="Gemini: " + ", ".join(A.DEFAULT_MODELS[A.GEMINI]) + ". OpenRouter: "
                                   + ", ".join(A.DEFAULT_MODELS[A.OPENROUTER]) + ". Defaults are tried in turn.")
        key = (key or "").strip()
        provider = A.detect_provider(key) if choice == AUTO else choice
        if not key:
            note = "Offline heuristic engine (no key)."
        elif provider is None:
            note = "Key format not recognised: choose the provider above. Using the offline engine until then."
        else:
            note = f"{provider} · {model.strip() or 'default free models'}"
        html(f'<p class="cc-note">{esc(note)}</p>')
    return AgentSettings(key, provider if key else None, (model or "").strip(), note)


# ======================================================================================
# Metrics (cached on the coordinates and signal)
# ======================================================================================
@st.cache_data(show_spinner=False, max_entries=96)
def metrics_for(coords: np.ndarray, signal: np.ndarray, valid: np.ndarray, b0: float, chrom_name: str,
                resolution: int, first_bin: int, label: str, placeholder: bool) -> A.Metrics:
    return A.compute_metrics(coords, signal, valid, b0, genome.chrom(chrom_name, resolution), first_bin, label,
                             placeholder)


def dataset_metrics(ds: Dataset, coords: np.ndarray, lo: int, hi: int, b0: float) -> A.Metrics:
    return metrics_for(np.ascontiguousarray(coords, dtype=np.float64), ds.epi[lo:hi], ds.valid[lo:hi], float(b0),
                       ds.chrom.name, ds.chrom.resolution, ds.bin0 + lo, ds.signal_label, ds.signal_is_placeholder)


def comparisons_for(ds: Dataset, lo: int, hi: int, b0: float, others: dict[str, Dataset]) -> tuple[A.Comparison, ...]:
    """Metrics of the other biological states: same beads when the bead sets match, else whole structure."""
    out = []
    for state, d in others.items():
        try:
            same = d.chrom.name == ds.chrom.name and d.bin0 == ds.bin0 and d.n == ds.n
            if same:
                m = dataset_metrics(d, d.frames[0][lo:hi], lo, hi, b0)
            else:
                m = dataset_metrics(d, d.frames[0], 0, d.n, b0)
            out.append(A.Comparison(state, m, same, d.structure_label))
        except Exception:  # a comparison that cannot be computed is simply left out
            continue
    return tuple(sorted(out, key=lambda c: A.state_order(c.state)))


def build_context(ds: Dataset, coords: np.ndarray, lo: int, hi: int, b0: float, region: str, state: str,
                  state_has_data: bool, reconstruction: bool, others: dict[str, Dataset]) -> A.AgentContext:
    ch = ds.chrom
    g_lo, g_hi = ds.bin0 + lo, ds.bin0 + hi
    start, end = int(ch.bin_start(g_lo)), int(ch.bin_end(g_hi - 1))
    return A.AgentContext(
        state=state, state_has_data=state_has_data, chrom=ch.name, resolution=ch.resolution, region=region,
        locus=f"{ch.name}:{start + 1:,}-{end:,}", structure=ds.structure_label, tracks=ds.tracks_label,
        is_reference=ds.is_reference, reconstruction=reconstruction, b0=float(b0),
        metrics=dataset_metrics(ds, coords, lo, hi, b0),
        genes=tuple(g.name for g in genome.genes_in(ch.name, start, end)),
        comparisons=comparisons_for(ds, lo, hi, b0, {s: d for s, d in others.items() if s != state}))


# ======================================================================================
# Metric dashboard (above the 3D viewport)
# ======================================================================================
def _delta(new: float, old: float) -> str:
    if not (math.isfinite(new) and math.isfinite(old)) or old == 0:
        return ""
    pct = (new - old) / abs(old) * 100
    if abs(pct) < 0.05:
        return f'<span class="d">±0.0% vs {SHORT_NAMES[HEALTHY]}</span>'
    cls = "up" if pct > 0 else "down"
    return f'<span class="d {cls}">{pct:+.1f}% vs {SHORT_NAMES[HEALTHY]}</span>'


def metric_cards(ctx: A.AgentContext) -> None:
    m = ctx.metrics
    base = next((c.metrics for c in ctx.comparisons if c.state == HEALTHY and c.same_window), None)

    def card(k: str, v: str, unit: str, delta: str = "", sub: str = "") -> str:
        sub_html = '<span class="d">' + sub + "</span>" if sub else ""
        return (f'<div class="cc-card"><span class="k">{k}</span><span class="v">{v}<em>{unit}</em></span>'
                f'{sub_html}{delta}</div>')

    fmt = lambda v, nd=0: "—" if not math.isfinite(v) else f"{v:,.{nd}f}"
    sig_unit = "placeholder" if m.signal_is_placeholder else ""
    state_sub = esc((ctx.structure if ctx.state_has_data else "no files for this state")[:64])
    cards = [
        card("Radius of gyration R<sub>g</sub>", fmt(m.rg_nm), "nm",
             _delta(m.rg_nm, base.rg_nm) if base else "", f"ν {fmt(m.nu, 3)} · {esc(m.regime)}"),
        card("Max 3D span", fmt(m.span_nm), "nm", _delta(m.span_nm, base.span_nm) if base else "",
             f"{m.n_beads:,} beads"),
        card("Mean signal intensity", fmt(m.mean_signal, 3), sig_unit,
             _delta(m.mean_signal, base.mean_signal) if base and not base.signal_is_placeholder else "",
             esc(m.signal_label[:40])),
        f'<div class="cc-card state"><span class="k">Biological state</span><span class="v s">{esc(ctx.state)}</span>'
        f'<span class="d">{state_sub}</span></div>',
    ]
    html('<div class="cc-cards">' + "".join(cards) + "</div>")


# ======================================================================================
# Interpreter panel
# ======================================================================================
_IMG = re.compile(r"!\[[^\]]*\]\([^)]*\)")


@st.cache_data(show_spinner=False, max_entries=64)
def _heuristic(_ctx: A.AgentContext, fingerprint: str, query: str) -> str:
    return A.heuristic_analysis(_ctx, query)


def _safe_markdown(text: str) -> str:
    """LLM output is rendered as Markdown only (no HTML); remote images are stripped."""
    return _IMG.sub("", text)


@st.fragment
def render(ctx: A.AgentContext, settings: AgentSettings, pdb: tuple[str, str] | None, scope: str) -> None:
    ss.setdefault("agent_results", {})
    with st.expander(TITLE, expanded=True), st.container(key=f"agent_{scope}"):
        fp = ctx.fingerprint()
        engine_tag = (f'<span class="cc-tag accent">{esc(settings.provider)}</span>' if settings.online
                      else '<span class="cc-tag">offline heuristic engine</span>')
        html(f'<p class="cc-meta">{engine_tag} &nbsp;State <b>{esc(ctx.state)}</b> · {esc(ctx.region)} · '
             f'<span class="cc-locus">{esc(ctx.locus)}</span></p>')
        with st.form(f"agent_form_{scope}", border=False):
            query = st.text_area("Ask ChronoAgent (optional)", key=f"agent_query_{scope}", height=78,
                                 placeholder="e.g. How does this state differ from the healthy control, and which "
                                             "interventions could restore the fold?")
            label = f"Analyse with {settings.provider}" if settings.online else "Analyse (offline engine)"
            go = st.form_submit_button(label, type="primary", icon=":material/psychology:")
        query = (query or "").strip()
        heur = _heuristic(ctx, fp, query)
        rkey = (fp, query, settings.provider, settings.model)
        if go and settings.online:
            with st.spinner(f"Asking {settings.provider}…"):
                try:
                    res = A.ask_llm(ctx, query, heur, settings.provider, settings.api_key, settings.model or None)
                    ss.agent_results[rkey] = (_safe_markdown(res.text), f"{res.provider} · {res.model}")
                except A.AgentError as exc:
                    ss.agent_results.pop(rkey, None)
                    warning_card("The AI provider did not answer; showing the offline heuristic analysis.", str(exc))
        text, engine = ss.agent_results.get(rkey, (heur, "offline heuristic engine"))
        if settings.online and rkey not in ss.agent_results:
            html('<p class="cc-note">Showing the offline analysis for this view. Press <b>Analyse</b> to ask '
                 f'{esc(settings.provider)} (one request per press).</p>')
        with st.container(key=f"agent_out_{scope}"):
            st.markdown(text)
        report = A.report_markdown(ctx, text, engine, query)
        c1, c2, _ = st.columns([1, 1, 1.4])
        c1.download_button("Report (ChronoCell_Analysis_Report.md)", report, "ChronoCell_Analysis_Report.md",
                           "text/markdown", width="stretch", icon=":material/download:", key=f"agent_md_{scope}")
        if pdb is not None:
            c2.download_button("Structure (.pdb) · current state", pdb[0], pdb[1], "chemical/x-pdb", width="stretch",
                               icon=":material/download:", key=f"agent_pdb_{scope}")
        html(f'<p class="cc-note">Engine: {esc(engine)}. {esc(A.DISCLAIMER)}</p>')
