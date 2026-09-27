"""
Genome neighbourhoods: TADs, A/B compartments, loops and the contact-decay exponent.

Everything works from a sparse contact list (ci < cj, counts cm, local bead indices). When a
dataset has no measured contacts, contacts are derived from the 3D structure itself: every pair
of beads closer than a capture radius counts once ("proximity contacts", the in-silico
equivalent of a Hi-C ligation).

* Insulation score (Crane et al., Nature 2015): contacts crossing each bead within a window w.
  Domain boundaries are local minima of the log2 insulation score with sufficient depth.
* A/B compartments (Lieberman-Aiden et al., Science 2009): first eigenvector of the Pearson
  correlation of the distance-normalised (observed / expected) contact map, computed on a
  coarsened map; the sign is chosen so that "A" correlates with GC / active signal.
* Loops: long-range pixels with the highest observed / expected enrichment, one per
  neighbourhood (non-maximum suppression). Requires measured contacts: in proximity contacts
  every touching pair has the same weight, so no enrichment can be read from them.
* Contact decay P(s) ~ s^-gamma (physics.contact_decay): gamma ~ 1 for a fractal globule,
  ~ 1.5 for an ideal chain; the "slope" reported in the UI is -gamma.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import physics

CAPTURE_RADIUS_B0 = 1.5


def proximity_contacts(x: np.ndarray, b0: float, radius_b0: float = CAPTURE_RADIUS_B0
                       ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pairs of beads (|i-j| >= 1) closer than radius_b0 * b0, each counted once."""
    i, j, _ = physics.neighbor_pairs(np.asarray(x, float), radius_b0 * b0, min_sep=1)
    return i.astype(np.int64), j.astype(np.int64), np.ones(len(i))


def expected_by_distance(ci: np.ndarray, cj: np.ndarray, cm: np.ndarray, n: int) -> np.ndarray:
    """Mean contact count per bead pair at each separation s (zeros included)."""
    s = np.abs(np.asarray(cj) - np.asarray(ci))
    sums = np.bincount(s, weights=cm, minlength=n)[:n]
    return sums / np.maximum(n - np.arange(n), 1)


def insulation(ci: np.ndarray, cj: np.ndarray, cm: np.ndarray, n: int, w: int) -> np.ndarray:
    """log2 insulation score per bead: contacts between [i-w, i) and (i, i+w], normalised to the mean.

    A contact (a, b) is counted for every bead i with a < i < b, i - a <= w and b - i <= w, i.e.
    i in [max(a+1, b-w), min(a+w, b-1)]; accumulated with a difference array in O(C + N).
    """
    ci, cj, cm = np.asarray(ci, np.int64), np.asarray(cj, np.int64), np.asarray(cm, float)
    a, b = np.minimum(ci, cj), np.maximum(ci, cj)
    lo, hi = np.maximum(a + 1, b - w), np.minimum(a + w, b - 1)
    ok = lo <= hi
    diff = np.zeros(n + 1)
    np.add.at(diff, lo[ok], cm[ok])
    np.add.at(diff, hi[ok] + 1, -cm[ok])
    score = np.cumsum(diff)[:n]
    out = np.full(n, np.nan)
    inner = np.arange(n)
    inner = (inner >= w) & (inner < n - w)
    if inner.any():
        mean = float(np.mean(score[inner]))
        if mean > 0:
            out[inner] = np.log2(np.maximum(score[inner], 1e-9) / mean)
    return out


def boundaries(ins: np.ndarray, w: int, min_depth: float = 0.15) -> list[int]:
    """Local minima of the insulation score deeper than min_depth (log2) below the local maxima
    on both sides within +-w; boundaries closer than w/2 keep the deeper one."""
    n = len(ins)
    v = np.where(np.isfinite(ins), ins, np.nan)
    cand = []
    half = max(2, w // 2)
    for i in range(1, n - 1):
        if not np.isfinite(v[i]):
            continue
        lo, hi = max(0, i - half), min(n, i + half + 1)
        win = v[lo:hi]
        if v[i] > np.nanmin(win):
            continue
        left, right = v[max(0, i - w):i], v[i + 1:min(n, i + w + 1)]
        if not (np.isfinite(left).any() and np.isfinite(right).any()):
            continue
        depth = min(np.nanmax(left), np.nanmax(right)) - v[i]
        if depth >= min_depth:
            cand.append((i, depth))
    kept: list[tuple[int, float]] = []
    for i, d in sorted(cand, key=lambda t: -t[1]):
        if all(abs(i - k) >= half for k, _ in kept):
            kept.append((i, d))
    return sorted(i for i, _ in kept)


def tads_from_boundaries(bounds: list[int], n: int, min_size: int = 3) -> list[tuple[int, int]]:
    edges = [0] + [b for b in bounds if 0 < b < n] + [n]
    return [(a, b) for a, b in zip(edges[:-1], edges[1:]) if b - a >= min_size]


def compartments(ci: np.ndarray, cj: np.ndarray, cm: np.ndarray, n: int, orient: np.ndarray | None,
                 max_bins: int = 500) -> tuple[np.ndarray, int]:
    """First eigenvector of the O/E correlation map, per bead (NaN where undefined), and the coarsening k."""
    k = max(1, int(np.ceil(n / max_bins)))
    m = int(np.ceil(n / k))
    a = np.asarray(ci, np.int64) // k
    b = np.asarray(cj, np.int64) // k
    mat = np.zeros((m, m))
    np.add.at(mat, (a, b), cm)
    off = a != b
    np.add.at(mat, (b[off], a[off]), np.asarray(cm, float)[off])
    cover = mat.sum(axis=1) > 0
    ev_beads = np.full(n, np.nan)
    if cover.sum() < 10:
        return ev_beads, k
    sub = mat[np.ix_(cover, cover)]
    s = sub.shape[0]
    oe = np.zeros_like(sub)
    for d in range(s):
        diag = np.diagonal(sub, d)
        mu = diag.mean()
        if mu > 0:
            idx = np.arange(s - d)
            oe[idx, idx + d] = diag / mu
            oe[idx + d, idx] = diag / mu
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.corrcoef(oe)
    corr = np.nan_to_num(corr, nan=0.0)
    vals, vecs = np.linalg.eigh(corr)
    ev = vecs[:, -1]
    full = np.full(m, np.nan)
    full[cover] = ev
    if orient is not None:
        o = np.asarray(orient, float)
        o_coarse = np.array([np.nanmean(o[i * k:(i + 1) * k]) if np.isfinite(o[i * k:(i + 1) * k]).any() else np.nan
                             for i in range(m)])
        ok = np.isfinite(full) & np.isfinite(o_coarse)
        if ok.sum() >= 3 and np.corrcoef(full[ok], o_coarse[ok])[0, 1] < 0:
            full = -full
    ev_beads[:] = np.repeat(full, k)[:n]
    return ev_beads, k


def loops(ci: np.ndarray, cj: np.ndarray, cm: np.ndarray, n: int, min_sep: int = 5, max_sep: int | None = None,
          top: int = 30, min_oe: float = 3.0, min_count: float = 5.0, suppress: int = 4) -> list[tuple[int, int, float]]:
    """Candidate loops: the strongest enrichments (i, j, observed/expected) between min_sep and max_sep
    beads apart with at least min_count reads, one per neighbourhood."""
    ci, cj, cm = np.asarray(ci, np.int64), np.asarray(cj, np.int64), np.asarray(cm, float)
    if ci.size == 0 or np.all(cm == cm[0]):
        return []
    max_sep = max_sep or max(min_sep + 1, n // 4)
    exp = expected_by_distance(ci, cj, cm, n)
    s = cj - ci
    sel = (s >= min_sep) & (s <= max_sep) & (exp[np.clip(s, 0, n - 1)] > 0)
    if not sel.any():
        return []
    oe = cm[sel] / exp[s[sel]]
    count_floor = max(float(np.percentile(cm[sel], 90)), min_count)
    strong = (oe >= min_oe) & (cm[sel] >= count_floor)
    order = np.argsort(-oe[strong])
    ii, jj, vv = ci[sel][strong][order], cj[sel][strong][order], oe[strong][order]
    kept: list[tuple[int, int, float]] = []
    for i, j, v in zip(ii, jj, vv):
        if all(abs(i - a) > suppress or abs(j - b) > suppress for a, b, _ in kept):
            kept.append((int(i), int(j), float(v)))
        if len(kept) >= top:
            break
    return kept


def decay_exponent(ci: np.ndarray, cj: np.ndarray, cm: np.ndarray, n: int) -> float:
    """gamma in P(s) ~ s^-gamma (fit over s = 2 ... n/10)."""
    if len(ci) == 0 or n < 30:
        return float("nan")
    return physics.contact_decay(np.asarray(ci), np.asarray(cj), np.asarray(cm, float), n)[2]


@dataclass
class DomainReport:
    n: int
    window: int                              # insulation window (beads)
    insulation: np.ndarray
    boundaries: list[int]
    tads: list[tuple[int, int]]
    compartment: np.ndarray                  # eigenvector per bead (NaN = undefined)
    coarsening: int
    loops: list[tuple[int, int, float]]
    gamma: float
    source: str                              # "measured contacts" | "3D proximity"
    notes: list[str] = field(default_factory=list)

    @property
    def a_fraction(self) -> float:
        ok = np.isfinite(self.compartment)
        return float(np.mean(self.compartment[ok] > 0)) if ok.any() else float("nan")

    def tad_labels(self) -> np.ndarray:
        lab = np.zeros(self.n)
        for k, (a, b) in enumerate(self.tads):
            lab[a:b] = k % 2
        return lab

    def summary(self, resolution: int) -> dict:
        sizes = [(b - a) * resolution / 1e6 for a, b in self.tads]
        return {"source": self.source, "tads": len(self.tads),
                "median_tad_mb": round(float(np.median(sizes)), 3) if sizes else None,
                "a_compartment_fraction": None if not np.isfinite(self.a_fraction) else round(self.a_fraction, 3),
                "loops": len(self.loops), "contact_decay_gamma": None if not np.isfinite(self.gamma) else round(self.gamma, 3)}


def window_for(resolution: int, n: int) -> int:
    """Insulation window ~ 500 kb, at least 5 beads, at most n/8."""
    return int(max(5, min(round(500_000 / resolution), max(5, n // 8))))


def analyse(n: int, resolution: int, b0: float, coords: np.ndarray | None = None,
            ci: np.ndarray | None = None, cj: np.ndarray | None = None, cm: np.ndarray | None = None,
            orient: np.ndarray | None = None) -> DomainReport:
    """Domains from measured contacts when given, else from the structure's own proximity contacts."""
    notes = []
    if ci is not None and len(ci) >= 50:
        source = "measured contacts"
        ci, cj, cm = np.asarray(ci, np.int64), np.asarray(cj, np.int64), np.asarray(cm, float)
    elif coords is not None:
        source = "3D proximity"
        ci, cj, cm = proximity_contacts(coords, b0)
        notes.append(f"No measured contacts: beads closer than {CAPTURE_RADIUS_B0} b0 in 3D are treated as contacts.")
    else:
        raise ValueError("Domain analysis needs contacts or coordinates.")
    w = window_for(resolution, n)
    ins = insulation(ci, cj, cm, n, w)
    bnd = boundaries(ins, w)
    ev, k = compartments(ci, cj, cm, n, orient)
    lp = loops(ci, cj, cm, n, max_sep=max(10, round(2_000_000 / resolution))) if source == "measured contacts" else []
    if source != "measured contacts":
        notes.append("Loops need measured contact counts and are not called from 3D proximity.")
    return DomainReport(n, w, ins, bnd, tads_from_boundaries(bnd, n), ev, k, lp, decay_exponent(ci, cj, cm, n), source,
                        notes)
