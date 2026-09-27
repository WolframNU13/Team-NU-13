"""
ChronoAgent — structural-genomics interpreter for one chromatin conformation.

Two engines share one measured context (`AgentContext`):

* Heuristic engine (offline, deterministic, instant). Rules over polymer metrics computed here:
  R_g, maximum span, scaling exponent ν, packing fraction, local crowding, signal statistics and
  rank correlations between signal and structure, plus comparisons with the other biological
  states that have data. Thresholds are anchored to polymer theory (ν = 1/3, 1/2, 0.59) and to the
  bundled reference fractal globule (packing ≈ 0.19), not tuned to any disease.
* LLM engine (optional). Google Gemini or OpenRouter (free tiers) over their REST APIs. The model
  receives the numeric context plus the heuristic findings and is instructed to stay grounded in
  them. The API key travels only in a request header, is never logged and never written to the
  report. Any failure falls back to the heuristic engine.

Everything therapeutic is framed as a research hypothesis for experimental follow-up. The agent
does not give medical advice, and the report says so.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, field
from typing import Callable

import numpy as np

from . import features, genome, physics
from .states import DISEASE, HEALTHY, SENESCENT, STATES

DENSITY_RADIUS_B0 = 1.5          # crowding is counted within 1.5 b0
DENSE_NEIGHBOURS = 12            # >= 12 non-bonded beads within 1.5 b0 ~ local packing 0.44
REFERENCE_PACKING = 0.19         # whole chr22 reference fractal globule (seed 7): calibration anchor
MIN_CORR_BEADS = 20
DISCLAIMER = ("Research-use interpretation of a computational model. Therapeutic items are hypotheses for "
              "experimental follow-up, not medical advice or treatment recommendations.")


# ======================================================================================
# Metrics
# ======================================================================================
@dataclass(frozen=True)
class Locus:
    label: str
    bin: int
    signal: float
    density: float


@dataclass(frozen=True)
class Metrics:
    n_beads: int
    n_assembled: int
    rg_nm: float
    span_nm: float
    re_nm: float
    nu: float
    nu_se: float
    regime: str
    packing: float                 # N b0^3 / (8 R^3), R = sqrt(5/3) R_g (sphere of equal R_g)
    asphericity: float
    density_mean: float            # mean non-bonded neighbours within 1.5 b0
    dense_fraction: float          # fraction of beads with >= DENSE_NEIGHBOURS neighbours
    mean_signal: float
    median_signal: float
    signal_sd: float
    signal_label: str
    signal_is_placeholder: bool
    rho_signal_density: float      # Spearman
    rho_signal_radial: float       # Spearman, radial distance from the centroid
    n_hubs: int
    accessible: tuple[Locus, ...] = ()
    compact: tuple[Locus, ...] = ()

    def summary(self) -> dict:
        def r(v, nd=3):
            return None if v is None or (isinstance(v, float) and not math.isfinite(v)) else round(float(v), nd)
        return {"beads": self.n_beads, "assembled_beads": self.n_assembled, "rg_nm": r(self.rg_nm, 1),
                "max_span_nm": r(self.span_nm, 1), "end_to_end_nm": r(self.re_nm, 1), "nu": r(self.nu),
                "nu_se": r(self.nu_se), "regime": self.regime, "packing_fraction": r(self.packing),
                "asphericity": r(self.asphericity), "mean_neighbours_1p5b0": r(self.density_mean, 2),
                "dense_fraction": r(self.dense_fraction), "signal": self.signal_label,
                "signal_placeholder": self.signal_is_placeholder, "mean_signal": r(self.mean_signal),
                "median_signal": r(self.median_signal), "signal_sd": r(self.signal_sd),
                "spearman_signal_vs_crowding": r(self.rho_signal_density),
                "spearman_signal_vs_radial_position": r(self.rho_signal_radial), "signal_hubs": self.n_hubs,
                "most_accessible_loci": [l.label for l in self.accessible],
                "most_compact_low_signal_loci": [l.label for l in self.compact]}


def _rank(a: np.ndarray) -> np.ndarray:
    """Average ranks (ties share the mean rank), without SciPy."""
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=np.float64)
    ranks[order] = np.arange(len(a), dtype=np.float64)
    _, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
    return (np.bincount(inv, weights=ranks) / counts)[inv]


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    if len(a) < MIN_CORR_BEADS or len(a) != len(b):
        return float("nan")
    ra, rb = _rank(a), _rank(b)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def _z(v: np.ndarray) -> np.ndarray:
    sd = v.std()
    return (v - v.mean()) / sd if sd > 0 else np.zeros_like(v)


def _top_loci(score: np.ndarray, beads: np.ndarray, signal: np.ndarray, density: np.ndarray, chrom: genome.Chrom,
              first_bin: int, k: int = 3, spacing: int = 10) -> tuple[Locus, ...]:
    picked: list[int] = []
    for idx in np.argsort(score)[::-1]:
        b = int(beads[idx])
        if all(abs(b - p) >= spacing for p in picked):
            picked.append(b)
        if len(picked) == k:
            break
    out = []
    for b in picked:
        g = first_bin + b
        out.append(Locus(f"{chrom.name}:{float(chrom.bin_start(g)) / 1e6:.2f} Mb", int(g),
                         float(signal[b]) if np.isfinite(signal[b]) else float("nan"), float(density[b])))
    return tuple(out)


def compute_metrics(coords: np.ndarray, signal: np.ndarray, valid: np.ndarray, b0: float, chrom: genome.Chrom,
                    first_bin: int = 0, signal_label: str = "H3K27ac", placeholder: bool = False) -> Metrics:
    """All agent metrics for one conformation (window) and its per-bead signal."""
    x = np.asarray(coords, dtype=np.float64)
    n = len(x)
    if n < 4 or not np.all(np.isfinite(x)):
        raise ValueError("Metrics need at least 4 finite bead coordinates.")
    signal = np.asarray(signal, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    if len(signal) != n or len(valid) != n:
        raise ValueError(f"Signal ({len(signal)}) and mask ({len(valid)}) must have one value per bead ({n}).")
    rg = physics.radius_of_gyration(x)
    fit = physics.distance_scaling(x)
    shape = physics.gyration_shape(x)
    packing = n * b0 ** 3 / (8.0 * (math.sqrt(5.0 / 3.0) * rg) ** 3) if rg > 0 else float("nan")
    density = physics.local_density(x, DENSITY_RADIUS_B0 * b0)

    ok = valid & np.isfinite(signal)
    s_ok = signal[ok]
    radial = np.linalg.norm(x - x.mean(axis=0), axis=1)
    beads = np.arange(n)
    if ok.sum() >= MIN_CORR_BEADS and not placeholder:
        rho_d, rho_r = spearman(s_ok, density[ok]), spearman(s_ok, radial[ok])
        zs, zd = _z(s_ok), _z(density[ok])
        accessible = _top_loci(zs - zd, beads[ok], signal, density, chrom, first_bin)
        compact = _top_loci(zd - zs, beads[ok], signal, density, chrom, first_bin)
    else:
        rho_d = rho_r = float("nan")
        pool = beads[valid] if valid.any() else beads
        accessible = _top_loci(-density[pool], pool, signal, density, chrom, first_bin)
        compact = _top_loci(density[pool], pool, signal, density, chrom, first_bin)
    hubs = features.signal_hubs(signal, valid) if ok.any() and not placeholder else []
    return Metrics(
        n_beads=n, n_assembled=int(valid.sum()), rg_nm=rg, span_nm=physics.max_span(x),
        re_nm=physics.end_to_end(x), nu=fit.nu, nu_se=fit.nu_se, regime=fit.regime, packing=float(packing),
        asphericity=shape.asphericity, density_mean=float(density.mean()),
        dense_fraction=float(np.mean(density >= DENSE_NEIGHBOURS)),
        mean_signal=float(s_ok.mean()) if s_ok.size else float("nan"),
        median_signal=float(np.median(s_ok)) if s_ok.size else float("nan"),
        signal_sd=float(s_ok.std()) if s_ok.size else float("nan"),
        signal_label=signal_label, signal_is_placeholder=bool(placeholder),
        rho_signal_density=rho_d, rho_signal_radial=rho_r, n_hubs=len(hubs),
        accessible=accessible, compact=compact)


# ======================================================================================
# Context
# ======================================================================================
@dataclass(frozen=True)
class Comparison:
    state: str
    metrics: Metrics
    same_window: bool              # same beads as the current view (R_g directly comparable)
    source: str


@dataclass(frozen=True)
class AgentContext:
    state: str                     # selected biological state
    state_has_data: bool           # the displayed structure comes from that state's files
    chrom: str
    resolution: int
    region: str
    locus: str
    structure: str
    tracks: str
    is_reference: bool
    reconstruction: bool
    b0: float
    metrics: Metrics
    genes: tuple[str, ...] = ()
    comparisons: tuple[Comparison, ...] = field(default_factory=tuple)
    extras: dict = field(default_factory=dict)   # genes_on_fold / neighbourhoods / drug_lab summaries (v3.2)

    def payload(self) -> dict:
        return {
            "selected_state": self.state, "selected_state_has_data": self.state_has_data,
            "genome": {"assembly": genome.ASSEMBLY, "chromosome": self.chrom, "bead_resolution_bp": self.resolution,
                       "region": self.region, "locus": self.locus, "anchor_genes_in_view": list(self.genes)},
            "data_provenance": {"structure": self.structure, "tracks": self.tracks,
                                "synthetic_reference_model": self.is_reference,
                                "egnn_reconstruction_shown": self.reconstruction, "b0_nm": round(self.b0, 1)},
            "metrics": self.metrics.summary(),
            "other_states": [{"state": c.state, "same_beads_as_view": c.same_window, "source": c.source,
                              "metrics": c.metrics.summary()} for c in self.comparisons],
            "reference_points": {"nu_fractal_globule": 0.333, "nu_ideal_chain": 0.5, "nu_self_avoiding": 0.588,
                                 "packing_reference_globule": REFERENCE_PACKING,
                                 "dense_neighbourhood": f">= {DENSE_NEIGHBOURS} beads within {DENSITY_RADIUS_B0} b0"},
            **({"additional_analyses": self.extras} if self.extras else {}),
        }

    def fingerprint(self) -> str:
        return hashlib.sha1(json.dumps(self.payload(), sort_keys=True, default=str).encode()).hexdigest()[:16]


# ======================================================================================
# Knowledge used by the heuristic engine (anchor genes shipped in chronocell/data/hg38.json)
# ======================================================================================
GENE_NOTES = {
    "22q11": ("22q11.2 deletion region (DGCR6, TBX1, COMT, LZTR1)",
              "Hemizygous 22q11.2 deletion is associated with a markedly increased risk of early-onset Parkinson's "
              "disease (Butcher et al., JAMA Neurol 2013); COMT encodes catechol-O-methyltransferase (dopamine "
              "catabolism) and TBX1 haploinsufficiency underlies the cardiac/craniofacial phenotype.",
              "No structure-directed therapy exists; parkinsonism in 22q11.2 deletion carriers is managed with "
              "standard dopaminergic therapy. Research angle: COMT dosage and dopamine turnover."),
    "BCR": ("BCR (22q11.23), Philadelphia-chromosome partner",
            "t(9;22) fuses BCR to ABL1, producing the constitutively active BCR-ABL1 kinase of chronic myeloid "
            "leukaemia and a subset of ALL.",
            "BCR-ABL1 tyrosine-kinase inhibitors (imatinib and later ATP-competitive TKIs; asciminib, an allosteric "
            "inhibitor) are established therapy - the fold changes nothing about that; structure can at most "
            "explain breakpoint-proximal regulatory rewiring."),
    "ABL1": ("ABL1 (9q34.12), Philadelphia-chromosome partner",
             "ABL1 kinase domain retained in the BCR-ABL1 fusion.",
             "BCR-ABL1 tyrosine-kinase inhibitors are established therapy."),
    "EWSR1": ("EWSR1 (22q12.2), Ewing-sarcoma fusion partner",
              "t(11;22) creates EWSR1-FLI1, which turns GGAA microsatellites into de novo enhancers "
              "(Riggi et al., Cancer Cell 2014) - a direct link between the fusion and 3D enhancer activity.",
              "No approved direct EWSR1-FLI1 inhibitor. Investigational: LSD1 inhibition (e.g. seclidemstat), "
              "trabectedin/lurbinectedin (interfere with EWSR1-FLI1-driven transcription), and targeting its "
              "transcriptional co-dependencies."),
    "FLI1": ("FLI1 (11q24.3), Ewing-sarcoma fusion partner",
             "Contributes the ETS DNA-binding domain of EWSR1-FLI1.",
             "As for EWSR1-FLI1 (investigational approaches only)."),
    "SNCA": ("SNCA (4q22.1), alpha-synuclein",
             "SNCA locus triplication or duplication causes autosomal-dominant Parkinson's disease "
             "(Singleton et al., Science 2003): gene dosage, not sequence, is pathogenic.",
             "Research: alpha-synuclein-lowering antisense oligonucleotides and anti-alpha-synuclein antibodies "
             "(early-phase clinical trials); structural readout = whether extra copies sit in an open, "
             "transcriptionally permissive neighbourhood."),
}


def _gene_topics(genes: tuple[str, ...]) -> list[tuple[str, str, str]]:
    out, seen = [], set()
    for g in genes:
        key = "22q11" if g in ("DGCR6", "TBX1", "COMT", "LZTR1") else g
        if key in GENE_NOTES and key not in seen:
            seen.add(key)
            out.append(GENE_NOTES[key])
    return out


# ======================================================================================
# Heuristic engine
# ======================================================================================
def _f(v: float, nd: int = 0, unit: str = "") -> str:
    if v is None or not math.isfinite(v):
        return "n/a"
    return f"{v:,.{nd}f}{unit}"


def _signed(pct: float) -> str:
    return f"{pct:+.1f}%" if math.isfinite(pct) else "n/a"


def _pct(new: float, old: float) -> float:
    return (new - old) / old * 100.0 if old and math.isfinite(old) and math.isfinite(new) else float("nan")


def _compaction_call(m: Metrics) -> tuple[str, int]:
    """(description, direction) with direction -1 = compact, 0 = typical, +1 = open."""
    nu, phi = m.nu, m.packing
    if math.isfinite(nu):
        if nu < 0.28:
            return "over-compacted / confined (ν below the fractal-globule value)", -1
        if nu < 0.36:
            if math.isfinite(phi) and phi > 1.4 * REFERENCE_PACKING:
                return "densely packed globule (heterochromatin-like)", -1
            return "compact fractal-globule-like fold (typical interphase chromatin)", 0
        if nu < 0.45:
            return "intermediate compaction between a globule and an ideal chain", 0
        if nu < 0.56:
            return "relaxed, ideal-chain-like fold (decompacted relative to a globule)", 1
        return "swollen / extended fold (strong decompaction or stretching)", 1
    if math.isfinite(phi):
        if phi > 1.4 * REFERENCE_PACKING:
            return "densely packed (ν not measurable in this window)", -1
        if phi < 0.5 * REFERENCE_PACKING:
            return "loosely packed (ν not measurable in this window)", 1
    return "undetermined (window too short for a scaling fit)", 0


def _rho_words(rho: float) -> str:
    a = abs(rho)
    return "no" if a < 0.1 else ("weak" if a < 0.3 else ("moderate" if a < 0.5 else "strong"))


def _state_shift(ctx: AgentContext) -> tuple[int, list[str]]:
    """Direction of compaction vs the healthy control (if available) and comparison bullets."""
    lines, direction = [], 0
    m = ctx.metrics
    for c in ctx.comparisons:
        o = c.metrics
        parts = []
        if c.same_window:
            parts.append(f"R_g {_signed(_pct(m.rg_nm, o.rg_nm))}")
            parts.append(f"span {_signed(_pct(m.span_nm, o.span_nm))}")
        if math.isfinite(m.nu) and math.isfinite(o.nu):
            parts.append(f"Δν {m.nu - o.nu:+.3f}")
        d_phi = _pct(m.packing, o.packing) if c.same_window or (math.isfinite(m.nu) and m.nu < 0.4) else float("nan")
        if math.isfinite(d_phi):
            parts.append(f"packing {d_phi:+.1f}%")
        parts.append(f"dense neighbourhoods {100 * (m.dense_fraction - o.dense_fraction):+.1f} pts")
        if not m.signal_is_placeholder and not o.signal_is_placeholder:
            parts.append(f"mean signal {_signed(_pct(m.mean_signal, o.mean_signal))}")
        scope = "same beads" if c.same_window else "whole structures (different bead sets: R_g not compared)"
        lines.append(f"vs **{c.state}** ({scope}): " + " · ".join(parts))
        if c.state == HEALTHY and ctx.state != HEALTHY:
            score = 0.0
            if c.same_window and math.isfinite(m.rg_nm) and o.rg_nm > 0:
                score += (m.rg_nm - o.rg_nm) / o.rg_nm / 0.05          # ±5 % R_g = one unit
            if math.isfinite(m.nu) and math.isfinite(o.nu):
                score += (m.nu - o.nu) / 0.03                           # ±0.03 in ν = one unit
            score -= 100 * (m.dense_fraction - o.dense_fraction) / 5.0  # ±5 pts dense = one unit
            direction = 1 if score >= 1 else (-1 if score <= -1 else 0)
    return direction, lines


def heuristic_analysis(ctx: AgentContext, query: str = "") -> str:
    m = ctx.metrics
    call, own_dir = _compaction_call(m)
    shift, comp_lines = _state_shift(ctx)
    direction = shift if any(c.state == HEALTHY for c in ctx.comparisons) and ctx.state != HEALTHY else own_dir
    signal_ok = not m.signal_is_placeholder and math.isfinite(m.mean_signal)
    out: list[str] = []

    # ---- data basis ------------------------------------------------------------------
    basis = []
    if not ctx.state_has_data:
        basis.append(f"No structure files were found for **{ctx.state}**; the view shows "
                     f"{'the synthetic reference model' if ctx.is_reference else 'another source'}, so state-specific "
                     "conclusions are withheld.")
    if ctx.is_reference:
        basis.append("The structure is the planted synthetic reference (fractal globule with band-derived tracks): "
                     "findings illustrate the method, not biology.")
    if m.signal_is_placeholder:
        basis.append("Signal tracks are reference placeholders: signal-based insights are withheld.")
    if ctx.reconstruction:
        basis.append("Metrics describe the EGNN reconstruction of this window.")
    out.append("**Data basis.** " + (" ".join(basis) if basis else
                                     f"{ctx.structure}; signal: {m.signal_label}."))

    # ---- biophysical diagnosis ---------------------------------------------------------
    out.append("### Biophysical diagnosis")
    out.append(f"- **Fold class:** {call}. ν = {_f(m.nu, 3)} ({m.regime}); packing fraction {_f(m.packing, 3)} "
               f"(reference globule ≈ {REFERENCE_PACKING}).")
    out.append(f"- **Size:** R_g {_f(m.rg_nm, 0)} nm, maximum span {_f(m.span_nm, 0)} nm "
               f"(span/R_g {_f(m.span_nm / m.rg_nm if m.rg_nm else float('nan'), 2)}; "
               f"≈ 3.2–3.5 for globules and random coils), asphericity {_f(m.asphericity, 2)} "
               f"({'elongated' if math.isfinite(m.asphericity) and m.asphericity > 0.4 else 'near-isotropic'}).")
    out.append(f"- **Crowding:** {_f(100 * m.dense_fraction, 1)}% of beads sit in dense neighbourhoods "
               f"(≥ {DENSE_NEIGHBOURS} non-bonded beads within {DENSITY_RADIUS_B0} b₀); mean "
               f"{_f(m.density_mean, 1)} neighbours. "
               + ("Foci this frequent are consistent with heterochromatin clustering."
                  if m.dense_fraction > 0.3 else
                  "Few crowded foci: chromatin is evenly or loosely packed." if m.dense_fraction < 0.1 else ""))
    for ln in comp_lines:
        out.append(f"- {ln}")
    dom = ctx.extras.get("neighbourhoods") or {}
    if dom:
        parts = [f"{dom.get('tads')} TAD-like neighbourhoods" + (f" (median {dom['median_tad_mb']} Mb)" if dom.get("median_tad_mb") else "")]
        if dom.get("a_compartment_fraction") is not None:
            parts.append(f"{100 * dom['a_compartment_fraction']:.0f}% of the region in the active A compartment")
        if dom.get("contact_decay_gamma") is not None:
            parts.append(f"contact decay γ = {dom['contact_decay_gamma']} (≈ 1 crumpled globule, ≈ 1.5 loose coil)")
        out.append(f"- **Neighbourhoods** (from {dom.get('source', 'contacts')}): " + "; ".join(parts) + ".")
    if ctx.state != HEALTHY and any(c.state == HEALTHY for c in ctx.comparisons):
        out.append(f"- **Net shift vs Healthy Control:** "
                   + {1: "decompaction (open, expanded domains).", -1: "compaction (condensed, crowded foci).",
                      0: "no consistent compaction shift."}[direction])

    # ---- therapeutic strategy ----------------------------------------------------------
    out.append("### Therapeutic strategy (research hypotheses)")
    topics = _gene_topics(ctx.genes)
    if not ctx.state_has_data:
        out.append(f"- Withheld: there is no {ctx.state} data to interpret. Add files for this state to "
                   "compare it with the others.")
    elif ctx.state == HEALTHY:
        out.append("- Healthy control: no intervention is indicated. Use this fold as the baseline that disease or "
                   "senescent states are measured against.")
    elif ctx.state == DISEASE:
        if direction > 0:
            out.append("- **Open, expanded domains** suggest enhancer activation or super-enhancer-driven "
                       "transcription. Experimental probes: BET-bromodomain inhibitors (JQ1-class; BRD4 displacement "
                       "preferentially collapses super-enhancer output), CDK7 inhibition, or p300/CBP catalytic "
                       "inhibition. Readout: R_g/ν and hub contacts should move back toward the control.")
        elif direction < 0:
            out.append("- **Condensed, crowded foci** suggest epigenetic silencing (e.g. of tumour-suppressor loci). "
                       "Experimental probes: HDAC inhibitors (vorinostat, romidepsin), DNMT inhibitors "
                       "(azacitidine, decitabine) and, if H3K27me3 is enriched, EZH2 inhibition (tazemetostat). "
                       "Readout: decompaction with signal gain at the compact loci listed below.")
        else:
            out.append("- No dominant compaction shift: prioritise locus-level differences (hubs and breakpoints) "
                       "over global chromatin-state drugs.")
    else:  # senescent
        if direction < 0 or m.dense_fraction > 0.3:
            out.append("- **Crowded foci** resemble senescence-associated heterochromatin foci (SAHF; Narita et al., "
                       "Cell 2003), typical of oncogene-induced senescence. Research strategies: senolytics "
                       "(dasatinib + quercetin; BCL-2/BCL-xL inhibitors such as navitoclax) to clear senescent cells, "
                       "or senomorphics (mTOR inhibition with rapamycin, JAK inhibition) to damp the SASP.")
        else:
            out.append("- **Decompaction** fits senescence-associated distension of satellites (SADS; Swanson et al., "
                       "J Cell Biol 2013) and lamin B1 loss. The same senolytic/senomorphic classes apply; the "
                       "structural readout is re-compaction or loss of the senescent population.")
    for title, _, therapy in topics:
        out.append(f"- **{title}:** {therapy}")
    lab = ctx.extras.get("drug_lab") or {}
    if lab:
        line = f"- **Virtual drug lab** ({lab.get('region', 'this region')}): {lab.get('drug')} ({lab.get('mode')})"
        if lab.get("restoration_pct") is not None and math.isfinite(lab["restoration_pct"]):
            line += f" restored {lab['restoration_pct']:.0f}% of the fold toward healthy at full dose"
        if lab.get("best_drug"):
            line += f"; the best-matching mechanism in the simulation was {lab['best_drug']}"
        out.append(line + ". This is a mechanism simulation, not an efficacy prediction.")

    # ---- expression & accessibility ----------------------------------------------------
    out.append("### Expression & accessibility insights")
    if signal_ok:
        rd, rr = m.rho_signal_density, m.rho_signal_radial
        if math.isfinite(rd):
            if rd <= -0.1:
                txt = "signal is higher where chromatin is less crowded, as expected for accessible, active chromatin"
            elif rd >= 0.1:
                txt = ("signal concentrates in crowded neighbourhoods, consistent with transcription hubs or "
                       "condensates that cluster active elements")
            else:
                txt = "signal and crowding are uncoupled at this scale"
            out.append(f"- {m.signal_label} vs crowding: Spearman ρ = {rd:+.2f} ({_rho_words(rd)}): {txt}.")
        if math.isfinite(rr):
            where = ("toward the surface of the fold (active chromatin looping out of the territory core)" if rr >= 0.1
                     else "toward the interior of the fold" if rr <= -0.1 else "without a radial preference")
            out.append(f"- Radial position: ρ = {rr:+.2f}; signal sits {where}.")
        out.append(f"- Mean {m.signal_label} {_f(m.mean_signal, 2)} (median {_f(m.median_signal, 2)}, "
                   f"SD {_f(m.signal_sd, 2)}); {m.n_hubs} top-3% signal hubs in view.")
        if m.accessible:
            out.append("- **Predicted most accessible / transcriptionally permissive** (high signal, low crowding): "
                       + ", ".join(l.label for l in m.accessible) + ".")
        if m.compact:
            out.append("- **Predicted compact / repressed** (low signal, high crowding): "
                       + ", ".join(l.label for l in m.compact) + ".")
    else:
        out.append("- Without a real signal track, accessibility is inferred from structure alone.")
        if m.accessible:
            out.append("- Least crowded (most accessible) loci: " + ", ".join(l.label for l in m.accessible) + ".")
        if m.compact:
            out.append("- Most crowded loci: " + ", ".join(l.label for l in m.compact) + ".")
    gs = ctx.extras.get("genes_on_fold") or {}
    if gs and gs.get("genes_in_view"):
        out.append(f"- **Genes on the fold:** {gs['genes_in_view']} promoters in view; {gs.get('open', 0)} predicted "
                   f"active, {gs.get('buried', 0)} predicted silenced."
                   + (f" Most open: {', '.join(gs['top_open'][:5])}." if gs.get("top_open") else "")
                   + (f" Most buried: {', '.join(gs['top_buried'][:5])}." if gs.get("top_buried") else ""))
        flagged = [f"{g['gene']} ({g['status'].split(' (')[0].lower()})" for g in gs.get("flagged_disease_genes", [])][:8]
        if flagged:
            out.append("- **Disease-relevant genes here:** " + ", ".join(flagged) + ".")
        if gs.get("expression_rho") is not None:
            out.append(f"- Measured expression vs predicted accessibility: Spearman ρ = {gs['expression_rho']:+.2f}.")
    out.append("- Hypothesis: transcription scales with accessibility; loci that are both open and signal-rich "
               "are the first candidates for expression changes between states (test with RNA-seq or ATAC-seq).")
    for title, biology, _ in topics:
        out.append(f"- **{title}:** {biology}")

    # ---- question ------------------------------------------------------------------------
    q = (query or "").strip()
    if q:
        out.append("### Answer to your question")
        ql = q.lower()
        routes = []
        if re.search(r"therap|drug|treat|inhibit|target|intervent|rescue", ql):
            routes.append("therapeutic strategy")
        if re.search(r"compar|differ|versus|\bvs\b|healthy|control|cancer|senesc|disease", ql):
            routes.append("state comparison in the biophysical diagnosis")
        if re.search(r"express|transcri|access|open|enhancer|acetyl|signal|hub", ql):
            routes.append("expression & accessibility insights")
        if re.search(r"compact|condens|dens|fold|shape|size|span|radius|gyration|nu\b|ν", ql):
            routes.append("biophysical diagnosis")
        named = [g for g in GENE_NOTES if g.lower() in ql and g != "22q11"]
        for g in named:
            routes.append(f"notes on {g}")
        if routes:
            out.append(f"The offline engine matched your question to: {', '.join(dict.fromkeys(routes))} (above). "
                       "Add an AI API key in the sidebar for a free-text answer grounded in the same metrics.")
        else:
            out.append("The offline heuristic engine answers through the sections above only. Add an AI API key in "
                       "the sidebar for a free-text answer grounded in the same metrics.")
    out.append(f"\n_{DISCLAIMER}_")
    return "\n".join(out)


# ======================================================================================
# LLM engine (free tiers: Google Gemini, OpenRouter)
# ======================================================================================
GEMINI, OPENROUTER = "Google Gemini", "OpenRouter"
PROVIDERS = (GEMINI, OPENROUTER)
DEFAULT_MODELS = {
    GEMINI: ("gemini-flash-latest", "gemini-2.5-flash", "gemini-2.0-flash"),
    OPENROUTER: ("meta-llama/llama-3.3-70b-instruct:free", "deepseek/deepseek-chat-v3-0324:free",
                 "google/gemma-3-27b-it:free"),
}
_GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
_OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
_MODEL_RE = re.compile(r"^[A-Za-z0-9._\-/:]{1,120}$")

SYSTEM_PROMPT = (
    "You are ChronoAgent, a structural-genomics interpreter embedded in ChronoCell-5D, a 3D/4D chromatin "
    "workstation. You receive measured polymer metrics of one chromatin conformation (and of other biological "
    "states when available) plus findings from a deterministic heuristic engine.\n"
    "Rules:\n"
    "1. Ground every claim in the numbers provided; quote them. When you go beyond them, label it a hypothesis.\n"
    "2. If the context says the structure is a synthetic reference model, that tracks are placeholders, or that "
    "the selected state has no data, say so first and keep biology generic.\n"
    "3. Therapeutic content = research hypotheses for experimental follow-up: name mechanisms and drug classes "
    "with their evidence level (approved / clinical trial / preclinical). Never give dosing or patient advice.\n"
    "4. Do not invent measurements, genes or loci that are not in the context.\n"
    "5. Reply in GitHub Markdown (no HTML) with exactly these level-3 headings, in order: "
    "'### Biophysical diagnosis', '### Therapeutic strategy (research hypotheses)', "
    "'### Expression & accessibility insights', and '### Answer to your question' only if a question is given. "
    "At most 450 words.")

Transport = Callable[[str, dict, dict, float], tuple[int, dict]]


class AgentError(RuntimeError):
    pass


@dataclass(frozen=True)
class LlmResult:
    text: str
    provider: str
    model: str


def detect_provider(api_key: str) -> str | None:
    k = (api_key or "").strip()
    if k.startswith("AIza"):
        return GEMINI
    if k.startswith("sk-or-"):
        return OPENROUTER
    return None


def build_prompt(ctx: AgentContext, query: str, heuristics: str) -> str:
    return ("CONTEXT (JSON):\n" + json.dumps(ctx.payload(), indent=1, default=str) +
            "\n\nHEURISTIC ENGINE FINDINGS (deterministic, from the same metrics):\n" + heuristics +
            "\n\nUSER QUESTION: " + ((query or "").strip() or "(none - give the three standard sections)"))


def _default_transport(url: str, headers: dict, payload: dict, timeout: float) -> tuple[int, dict]:
    import requests  # a Streamlit dependency; imported lazily so the numerics never need it

    try:
        r = requests.post(url, headers=headers, json=payload, timeout=(10, timeout))
    except requests.RequestException as exc:
        raise AgentError(f"network error: {type(exc).__name__}") from None
    try:
        body = r.json()
    except ValueError:
        body = {"error": {"message": r.text[:300]}}
    return r.status_code, body if isinstance(body, dict) else {"data": body}


def _error_message(body: dict) -> str:
    err = body.get("error") if isinstance(body, dict) else None
    if isinstance(err, dict):
        return str(err.get("message") or err.get("status") or err)[:300]
    if isinstance(err, str):
        return err[:300]
    return json.dumps(body)[:300] if body else "empty response"


def _parse_gemini(body: dict) -> str:
    cands = body.get("candidates") or []
    if not cands:
        reason = (body.get("promptFeedback") or {}).get("blockReason")
        raise AgentError(f"Gemini returned no answer{f' (blocked: {reason})' if reason else ''}.")
    parts = (cands[0].get("content") or {}).get("parts") or []
    text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
    if not text.strip():
        raise AgentError(f"Gemini returned an empty answer (finish reason: {cands[0].get('finishReason', '?')}).")
    return text


def _parse_openrouter(body: dict) -> str:
    choices = body.get("choices") or []
    if not choices:
        raise AgentError(f"OpenRouter returned no answer: {_error_message(body)}")
    text = ((choices[0].get("message") or {}).get("content") or "")
    if not text.strip():
        raise AgentError("OpenRouter returned an empty answer.")
    return text


def ask_llm(ctx: AgentContext, query: str, heuristics: str, provider: str, api_key: str, model: str | None = None,
            transport: Transport | None = None, timeout: float = 90.0) -> LlmResult:
    """Send the grounded prompt to Gemini or OpenRouter. Default models are tried in turn when one is
    unavailable or rate-limited; a rejected key stops immediately. Raises AgentError (key-free message)."""
    key = (api_key or "").strip()
    if not key:
        raise AgentError("No API key.")
    if provider not in PROVIDERS:
        raise AgentError(f"Unknown provider '{provider}'.")
    chain = [model.strip()] if model and model.strip() else list(DEFAULT_MODELS[provider])
    for m in chain:
        if not _MODEL_RE.match(m):
            raise AgentError(f"Invalid model name '{m[:60]}'.")
    send = transport or _default_transport
    prompt = build_prompt(ctx, query, heuristics)
    last = "no model tried"
    for m in chain:
        if provider == GEMINI:
            url = _GEMINI_URL.format(model=m)
            headers = {"x-goog-api-key": key, "Content-Type": "application/json"}
            payload = {"systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
                       "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                       "generationConfig": {"temperature": 0.35, "maxOutputTokens": 8192}}
        else:
            url = _OPENROUTER_URL
            headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json", "X-Title": "ChronoCell-5D"}
            payload = {"model": m, "temperature": 0.35, "max_tokens": 2048,
                       "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]}
        try:
            status, body = send(url, headers, payload, timeout)
        except AgentError as exc:
            raise AgentError(str(exc).replace(key, "•••")) from None
        if status == 200 and not (isinstance(body, dict) and body.get("error")):
            try:
                text = _parse_gemini(body) if provider == GEMINI else _parse_openrouter(body)
            except AgentError as exc:
                last = str(exc)
                continue
            return LlmResult(text.strip(), provider, m)
        msg = _error_message(body).replace(key, "•••")
        if status in (401, 403) or re.search(r"api[ _]?key.{0,20}(not valid|invalid)|API_KEY_INVALID", msg, re.I):
            raise AgentError(f"{provider} rejected the API key ({status}): {msg}")
        last = f"{m}: HTTP {status} - {msg}"
        if status in (404, 429, 500, 502, 503, 504) or (status == 400 and "model" in msg.lower()):
            continue                                   # try the next default model
        raise AgentError(f"{provider} error: {last}")
    raise AgentError(f"{provider}: no model answered ({last}).")


# ======================================================================================
# Report
# ======================================================================================
def _table(rows: list[tuple[str, str]]) -> str:
    return "| Metric | Value |\n|---|---|\n" + "\n".join(f"| {a} | {b} |" for a, b in rows)


def metric_rows(m: Metrics) -> list[tuple[str, str]]:
    return [("Beads (assembled)", f"{m.n_beads:,} ({m.n_assembled:,})"),
            ("Radius of gyration R_g", f"{_f(m.rg_nm, 1)} nm"),
            ("Maximum 3D span", f"{_f(m.span_nm, 1)} nm"),
            ("End-to-end distance", f"{_f(m.re_nm, 1)} nm"),
            ("Scaling exponent ν", f"{_f(m.nu, 3)} ± {_f(m.nu_se, 3)} ({m.regime})"),
            ("Packing fraction", _f(m.packing, 3)),
            ("Asphericity", _f(m.asphericity, 3)),
            (f"Dense-neighbourhood fraction (≥ {DENSE_NEIGHBOURS} within {DENSITY_RADIUS_B0} b₀)",
             f"{_f(100 * m.dense_fraction, 1)}%"),
            (f"Mean signal ({m.signal_label})", _f(m.mean_signal, 3) + (" (placeholder)" if m.signal_is_placeholder else "")),
            ("Spearman ρ signal vs crowding", _f(m.rho_signal_density, 3)),
            ("Spearman ρ signal vs radial position", _f(m.rho_signal_radial, 3)),
            ("Signal hubs (top 3 %)", str(m.n_hubs))]


def report_markdown(ctx: AgentContext, analysis: str, engine: str, query: str = "",
                    generated_utc: dt.datetime | None = None, software: str = "ChronoCell-5D") -> str:
    when = (generated_utc or dt.datetime.now(dt.timezone.utc)).strftime("%Y-%m-%d %H:%M UTC")
    lines = [f"# ChronoCell-5D Analysis Report", "",
             f"*Generated {when} by {software} · ChronoAgent engine: {engine}*", "",
             "## Sample", "",
             "| Field | Value |", "|---|---|",
             f"| Biological state | {ctx.state}{'' if ctx.state_has_data else ' (no files for this state; view shows another source)'} |",
             f"| Chromosome · resolution | {ctx.chrom} ({genome.ASSEMBLY}) · {ctx.resolution / 1000:g} kb beads |",
             f"| Region | {ctx.region} · {ctx.locus} |",
             f"| Structure | {ctx.structure}{' (EGNN reconstruction)' if ctx.reconstruction else ''} |",
             f"| Tracks | {ctx.tracks} |",
             f"| Synthetic reference model | {'yes' if ctx.is_reference else 'no'} |",
             f"| Anchor genes in view | {', '.join(ctx.genes) or 'none'} |",
             f"| Bond length b₀ | {ctx.b0:.0f} nm |", "",
             "## Structural metrics", "", _table(metric_rows(ctx.metrics)), ""]
    if ctx.comparisons:
        lines += ["## Other biological states", "",
                  "| State | Same beads | R_g (nm) | Span (nm) | ν | Packing | Dense % | Mean signal |",
                  "|---|---|---|---|---|---|---|---|"]
        for c in ctx.comparisons:
            o = c.metrics
            lines.append(f"| {c.state} | {'yes' if c.same_window else 'no'} | {_f(o.rg_nm, 0)} | {_f(o.span_nm, 0)} | "
                         f"{_f(o.nu, 3)} | {_f(o.packing, 3)} | {_f(100 * o.dense_fraction, 1)} | "
                         f"{_f(o.mean_signal, 3)} |")
        lines.append("")
    if query.strip():
        lines += ["## Question", "", "> " + query.strip().replace("\n", "\n> "), ""]
    lines += ["## ChronoAgent analysis", "", analysis.strip(), "",
              "## Method notes", "",
              "- R_g = sqrt(mean ||x_i - x_cm||²); span = maximum pairwise bead distance; ν from RMS R(s) ∝ s^ν "
              "(log-log fit over s = 4 … N/10 beads).",
              f"- Packing fraction = N b₀³ / (8 R³) with R = √(5/3) R_g; crowding = non-bonded beads within "
              f"{DENSITY_RADIUS_B0} b₀ (|i − j| ≥ 2).",
              "- Correlations are Spearman rank correlations over assembled beads.", "",
              f"*{DISCLAIMER}*", ""]
    return "\n".join(lines)


def state_order(state: str) -> int:
    return STATES.index(state) if state in STATES else len(STATES)


def as_dict(m: Metrics) -> dict:
    return asdict(m)
