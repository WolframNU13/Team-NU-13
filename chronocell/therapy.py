"""
Virtual drug lab: "what would the fold look like if a drug's mechanism acted on it?"

This is a mechanism sandbox, not a pharmacology model. Each drug class is reduced to two
facts that are well established qualitatively:

    WHERE it acts   (target weights per bead, from the signal track and 3D crowding)
    WHICH WAY       (+1 opens chromatin, -1 compacts it, 0 re-draws loops)

and the dose scales how far the targeted beads move. Two modes:

* Toward a healthy baseline (a healthy structure of the same beads is loaded): targeted beads are
  moved a fraction dose x max_effect of the way to their healthy positions, but only where that
  movement agrees with the drug's direction (an opening drug never compacts a region). Restoration
  is then measurable: 100 % x (1 - RMSD(treated, healthy) / RMSD(untreated, healthy)).
* Mechanism only (no baseline): targeted neighbourhoods are expanded or contracted about their
  local centre; loop stabilisation pulls loop anchors together.

Every treated conformation is relaxed (bond lengths -> b0, excluded volume) so it stays a valid
polymer. Outputs per dose: R_g, nu, contact-decay exponent gamma (from 3D proximity contacts; the
P(s) slope is -gamma), packing, and restoration when a baseline exists.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import domains, physics, synthetic


@dataclass(frozen=True)
class Drug:
    key: str
    name: str
    examples: str
    plain: str            # one-sentence, jargon-free mechanism
    target: str           # compact_low_signal | low_signal | hubs | loops
    direction: int        # +1 open, -1 compact, 0 loops
    evidence: str


DRUGS: dict[str, Drug] = {d.key: d for d in (
    Drug("ezh2", "EZH2 / EED inhibitor",
         "tazemetostat (EZH2; approved for epithelioid sarcoma and follicular lymphoma); EED inhibitors (clinical trials)",
         "Stops Polycomb from writing its 'keep this closed' mark, so tightly packed, silent regions can loosen.",
         "compact_low_signal", +1, "approved (EZH2) / clinical trials (EED)"),
    Drug("hdac", "HDAC inhibitor",
         "vorinostat, romidepsin (approved for cutaneous T-cell lymphoma)",
         "Stops cells from erasing 'open' (acetyl) marks, so chromatin loosens broadly, most where it was least open.",
         "low_signal", +1, "approved (haematological cancers)"),
    Drug("bet", "BET bromodomain inhibitor",
         "JQ1 (research tool); several clinical-stage BET inhibitors",
         "Pulls the reader protein BRD4 off over-active enhancer hubs, so those swollen, hyper-active hubs settle down.",
         "hubs", -1, "clinical trials"),
    Drug("ctcf", "CTCF / cohesin loop stabiliser",
         "no approved drug (hypothetical; mimics stronger cohesin retention, e.g. WAPL loss in research)",
         "Strengthens the loops that fence the genome into neighbourhoods, pulling each loop's two anchors together.",
         "loops", 0, "hypothetical / research"),
)}


@dataclass
class TreatmentResult:
    drug: Drug
    doses: np.ndarray                 # 0 ... 1
    frames: np.ndarray                # (D, N, 3), aligned onto the untreated structure
    weights: np.ndarray               # (N,) how strongly each bead is targeted, 0 ... 1
    metrics: pd.DataFrame             # one row per dose
    baseline: dict | None             # healthy metrics (same beads) or None
    mode: str
    efficacy: float
    notes: list[str] = field(default_factory=list)

    def row(self, dose: float) -> pd.Series:
        return self.metrics.iloc[int(np.argmin(np.abs(self.doses - dose)))]


# ======================================================================================
# Targeting
# ======================================================================================
def _smooth(v: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0 or len(v) < 3:
        return v
    r = int(max(1, round(3 * sigma)))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    return np.convolve(np.pad(v, r, mode="edge"), k, mode="valid")


def _pct(v: np.ndarray, ok: np.ndarray, reference: np.ndarray | None = None) -> np.ndarray:
    """Percentile rank in [0, 1] of each ok bead, against `reference` values when given (e.g. the whole
    chromosome, so a region-wide gain is visible), else among the ok beads themselves (0.5 elsewhere)."""
    out = np.full(len(v), 0.5)
    if reference is not None:
        ref = np.sort(np.asarray(reference, float)[np.isfinite(reference)])
        if ref.size >= 2:
            out[ok] = np.searchsorted(ref, v[ok], side="right") / ref.size
            return out
    if ok.sum() >= 2:
        order = np.argsort(np.argsort(v[ok]))
        out[ok] = order / (ok.sum() - 1)
    return out


def target_weights(drug: Drug, x: np.ndarray, signal: np.ndarray, valid: np.ndarray, b0: float,
                   loops: list[tuple[int, int, float]] | None = None, signal_ref: np.ndarray | None = None) -> np.ndarray:
    n = len(x)
    valid = np.asarray(valid, bool)
    sig = np.asarray(signal, float)
    sig_ok = valid & np.isfinite(sig)
    ps = _pct(np.nan_to_num(sig), sig_ok, signal_ref)
    crowd = _smooth(physics.local_density(x, domains.CAPTURE_RADIUS_B0 * b0), 2.0)
    pc = _pct(crowd, valid)
    if drug.target == "compact_low_signal":
        w = np.clip(((pc + (1 - ps)) / 2 - 0.5) * 2, 0, 1)
    elif drug.target == "low_signal":
        w = np.clip(((1 - ps) - 0.3) / 0.7, 0, 1)
    elif drug.target == "hubs":
        w = np.clip((ps - 0.75) / 0.25, 0, 1)
        w = np.clip(_smooth(w, 3.0) * 2.0, 0, 1)
    else:  # loops: anchors +- 2 beads
        w = np.zeros(n)
        for i, j, _ in (loops or []):
            for a in (i, j):
                w[max(0, a - 2):min(n, a + 3)] = 1.0
    w = np.where(valid, _smooth(w, 1.5) if drug.target != "hubs" else w, 0.0)
    return np.clip(w, 0, 1)


# ======================================================================================
# Moves
# ======================================================================================
def _local_centres(x: np.ndarray, half: int = 30) -> np.ndarray:
    n = len(x)
    c = np.cumsum(np.vstack([np.zeros((1, 3)), x]), axis=0)
    lo = np.clip(np.arange(n) - half, 0, n)
    hi = np.clip(np.arange(n) + half + 1, 0, n)
    return (c[hi] - c[lo]) / (hi - lo)[:, None]


def _pull_loops(x: np.ndarray, loops: list[tuple[int, int, float]], strength: float, b0: float) -> np.ndarray:
    y = x.copy()
    n = len(x)
    for i, j, _ in loops:
        d = y[j] - y[i]
        dist = float(np.linalg.norm(d))
        if dist <= 1.2 * b0:
            continue
        step = d * (1 - 1.2 * b0 / dist) * 0.5 * strength
        for off, f in ((0, 1.0), (1, 0.6), (2, 0.3)):
            for a, sgn in ((i, 1.0), (j, -1.0)):
                for k in {a - off, a + off}:
                    if 0 <= k < n:
                        y[k] = y[k] + sgn * f * step
    return y


def _metrics(x: np.ndarray, b0: float) -> dict:
    rg = physics.radius_of_gyration(x)
    ci, cj, cm = domains.proximity_contacts(x, b0)
    gamma = domains.decay_exponent(ci, cj, cm, len(x))
    n = len(x)
    packing = n * b0 ** 3 / (8.0 * (math.sqrt(5.0 / 3.0) * rg) ** 3) if rg > 0 else float("nan")
    return {"rg_nm": rg, "nu": physics.distance_scaling(x).nu, "gamma": gamma, "packing": packing}


def suggest_window(x_disease: np.ndarray, x_healthy: np.ndarray | None, signal: np.ndarray, width: int = 800
                   ) -> tuple[int, int]:
    """The `width`-bead window where the disease fold differs most from the healthy one (or, without a
    baseline, the window with the strongest signal hubs)."""
    n = len(x_disease)
    if n <= width:
        return 0, n
    if x_healthy is not None and len(x_healthy) == n:
        _, h, _ = physics.kabsch_rmsd(x_disease, x_healthy)
        score = np.linalg.norm(h - x_disease, axis=1)
    else:
        score = np.nan_to_num(np.asarray(signal, float))
    csum = np.concatenate([[0.0], np.cumsum(score)])
    lo = int(np.argmax(csum[width:] - csum[:-width]))
    return lo, lo + width


def simulate_treatment(x_disease: np.ndarray, signal: np.ndarray, valid: np.ndarray, b0: float, drug_key: str,
                       x_healthy: np.ndarray | None = None, efficacy: float = 0.8,
                       doses: np.ndarray | None = None, relax_iters: int = 15,
                       loops: list[tuple[int, int, float]] | None = None,
                       signal_ref: np.ndarray | None = None) -> TreatmentResult:
    drug = DRUGS[drug_key]
    x0 = np.asarray(x_disease, dtype=np.float64)
    n = len(x0)
    if n < 20:
        raise ValueError("The drug lab needs at least 20 beads.")
    doses = np.linspace(0.0, 1.0, 11) if doses is None else np.asarray(doses, float)
    notes: list[str] = []
    if drug.target == "loops" and not loops:
        ci, cj, cm = domains.proximity_contacts(x0, b0)
        # Without measured loops, use the strongest long-range 3D contacts of the untreated fold as anchors.
        s = cj - ci
        far = s >= 15
        order = np.argsort(-s[far])[:25]
        loops = [(int(ci[far][k]), int(cj[far][k]), 1.0) for k in order]
        notes.append("No measured loops: the longest-range 3D contacts of the untreated fold are used as loop anchors.")
    w = target_weights(drug, x0, signal, valid, b0, loops, signal_ref)

    healthy = None
    if x_healthy is not None and len(x_healthy) == n:
        _, healthy, _ = physics.kabsch_rmsd(x0, np.asarray(x_healthy, float))
        mode = "toward healthy baseline"
    else:
        mode = "mechanism only"
        if x_healthy is not None:
            notes.append("The healthy structure covers different beads, so no restoration can be measured.")

    gate = np.ones(n)
    if healthy is not None and drug.direction != 0:
        cd = _smooth(physics.local_density(x0, domains.CAPTURE_RADIUS_B0 * b0), 2.0)
        ch = _smooth(physics.local_density(healthy, domains.CAPTURE_RADIUS_B0 * b0), 2.0)
        diff = (cd - ch) * drug.direction          # > 0 where the drug's direction moves toward healthy
        sd = float(np.std(cd - ch)) or 1.0
        gate = np.clip(diff / sd, 0, 1)
    eff_w = w * gate

    d_min = physics.D_MIN_FACTOR * b0
    frames = []
    for d in doses:
        s = float(d) * float(efficacy)
        if s <= 0:
            frames.append(x0.copy())
            continue
        if drug.target == "loops":
            y = _pull_loops(x0, loops or [], s, b0)
            if healthy is not None:
                y = y + (s * w)[:, None] * (healthy - y) * 0.5
        elif healthy is not None:
            y = x0 + (s * eff_w)[:, None] * (healthy - x0)
        else:
            # Fold-scale moves (about +-30-bead centres) in rounds, relaxing in between, so bond
            # constraints don't simply undo them: opening swells targeted domains, compaction condenses them.
            y = x0.copy()
            per_round = (1.0 + (0.9 if drug.direction > 0 else -0.5) * s * w) ** 0.25
            for _ in range(4):
                c = _local_centres(y)
                y = synthetic.relax(c + (y - c) * per_round[:, None], b0, d_min, iters=max(4, relax_iters // 3))
        frames.append(synthetic.relax(y, b0, d_min, iters=relax_iters))
    frames_arr = np.stack(frames)
    for t in range(1, len(frames_arr)):
        _, frames_arr[t], _ = physics.kabsch_rmsd(frames_arr[0], frames_arr[t], allow_reflection=False)

    rows = []
    base = _metrics(healthy, b0) if healthy is not None else None
    rmsd0 = physics.kabsch_rmsd(healthy, frames_arr[0])[0] if healthy is not None else float("nan")
    for d, fx in zip(doses, frames_arr):
        m = _metrics(fx, b0)
        m["dose_pct"] = round(100 * float(d))
        if healthy is not None:
            r = physics.kabsch_rmsd(healthy, fx)[0]
            m["rmsd_to_healthy_nm"] = r
            m["restoration_pct"] = float(np.clip(100 * (1 - r / rmsd0), -100, 100)) if rmsd0 > 0 else float("nan")
        rows.append(m)
    cols = ["dose_pct", "rg_nm", "nu", "gamma", "packing"] + (["rmsd_to_healthy_nm", "restoration_pct"] if healthy is not None else [])
    table = pd.DataFrame(rows)[cols]
    if float(np.mean(eff_w > 0.2)) < 0.02:
        notes.append("This drug's targets do not overlap the changes in this region, so it has almost no effect here.")
    return TreatmentResult(drug, doses, frames_arr, eff_w, table, base, mode, float(efficacy), notes)


def compare_drugs(x_disease: np.ndarray, signal: np.ndarray, valid: np.ndarray, b0: float, x_healthy: np.ndarray,
                  efficacy: float = 0.8, relax_iters: int = 15, signal_ref: np.ndarray | None = None) -> pd.DataFrame:
    """Full-dose restoration for every drug class (needs a healthy baseline of the same beads)."""
    out = []
    for key, drug in DRUGS.items():
        r = simulate_treatment(x_disease, signal, valid, b0, key, x_healthy, efficacy, np.array([0.0, 1.0]), relax_iters,
                               signal_ref=signal_ref)
        last = r.metrics.iloc[-1]
        out.append({"drug": drug.name, "key": key, "restoration_pct": last.get("restoration_pct", float("nan")),
                    "targeted_beads_pct": 100 * float(np.mean(r.weights > 0.2)), "evidence": drug.evidence})
    return pd.DataFrame(out).sort_values("restoration_pct", ascending=False).reset_index(drop=True)
