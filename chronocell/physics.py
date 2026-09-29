"""
Polymer physics for 10 kb coarse-grained chromatin (NumPy, no autograd).

Units: every length is in nanometres. Losses are dimensionless (lengths divided by the
bond rest length b0), so their magnitudes are comparable and independent of the
coordinate scale an upstream model happens to emit.

Physical scale. Coarse-grained 30 nm-fibre models map ~3 kb onto a 30 nm monomer (e.g. Rosa &
Everaers, PLoS Comput Biol 2008). A 10 kb bead is ~3.3 such monomers; scaling 30 nm by
(10/3)^nu for nu in [1/3, 1/2] gives 45–55 nm, so the default bond rest length is b0 = 50 nm.
This is an order-of-magnitude anchor, exposed as a parameter, not a measured constant.

Contact to distance. Contact frequency falls with spatial distance as M ~ d^-alpha. alpha = 3
is the capture-volume argument (probability of being within a fixed capture radius ~ d^-3)
and is the PASTIS default (Varoquaux et al., Bioinformatics 2014). Inverting and anchoring
the median nearest-neighbour count M_ref to the bond length:
    d*_ij = b0 * (M_ij / M_ref)^(-1/alpha)
which is finite for every observed contact (M_ij > 0) and fixes the length scale from data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

B0_NM = 50.0            # bond rest length for a 10 kb bead (nm)


def bond_length_for(resolution_bp: int, nu: float = 1.0 / 3.0) -> float:
    """b0 for coarser beads: b0(r) = 50 nm * (r / 10 kb)^nu, compact (fractal-globule) scaling.

    chr22 uses 10 kb -> 50 nm; a 40 kb bead (chr4 default) -> 79 nm; 50 kb (chr1) -> 85 nm.
    """
    return B0_NM * (resolution_bp / 10_000) ** nu
ALPHA = 3.0             # contact-frequency / distance power law exponent
D_MIN_FACTOR = 0.8      # excluded-volume diameter as a fraction of b0
# Contact target distances are clipped to [d_min, 8 b0]. The lower bound equals the
# excluded-volume diameter: a contact cannot imply a separation the steric term forbids,
# otherwise Poisson-high counts (d* < d_min) set the two terms against each other.
TARGET_CLIP = (D_MIN_FACTOR, 8.0)

# Reference 3D scaling exponents for R(s) ~ s^nu
NU_FRACTAL_GLOBULE = 1.0 / 3.0   # compact, space-filling (Grosberg 1988; Lieberman-Aiden 2009)
NU_IDEAL = 0.5                   # ideal / theta-solvent Gaussian chain
NU_SAW = 0.5876                  # self-avoiding walk (RG value; Flory mean field gives 3/5)


# ----------------------------------------------------------------------------------------
# Global shape descriptors
# ----------------------------------------------------------------------------------------
def radius_of_gyration(x: np.ndarray) -> float:
    """R_g = sqrt( (1/N) sum_i ||x_i - x_cm||^2 ), equal-mass beads.

    Identical to sqrt( (1/(2 N^2)) sum_{i,j} ||x_i - x_j||^2 ); the pairwise form is O(N^2)
    and is only used in the test-suite to verify this O(N) centre-of-mass form.
    """
    x = np.asarray(x, dtype=np.float64)
    return float(np.sqrt(np.mean(np.sum((x - x.mean(axis=0)) ** 2, axis=1))))


def end_to_end(x: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(x[-1], float) - np.asarray(x[0], float)))


def bond_lengths(x: np.ndarray) -> np.ndarray:
    return np.linalg.norm(np.diff(np.asarray(x, dtype=np.float64), axis=0), axis=1)


def calibrate_to_bond_length(x: np.ndarray, b0: float = B0_NM) -> tuple[np.ndarray, float]:
    """Rescale coordinates in arbitrary model units so the median bond equals b0 (nm).

    A median is used so a few stretched bonds (e.g. across unassembled gaps) do not set the scale.
    Returns (scaled coordinates, scale factor applied).
    """
    med = float(np.median(bond_lengths(x)))
    if not np.isfinite(med) or med <= 0:
        raise ValueError("Cannot calibrate units: median bond length is zero or undefined.")
    s = b0 / med
    return np.asarray(x, dtype=np.float64) * s, s


# ----------------------------------------------------------------------------------------
# Distance scaling R(s) ~ s^nu
# ----------------------------------------------------------------------------------------
@dataclass(frozen=True)
class ScalingFit:
    s: np.ndarray               # genomic separations (bins), log-spaced, unique
    r_rms: np.ndarray           # sqrt(<||x_{i+s} - x_i||^2>) in nm
    local_slope: np.ndarray     # d log R / d log s
    nu: float
    nu_se: float                # OLS standard error (ignores correlation between s values)
    r2: float
    fit_range: tuple[int, int]
    regime: str


def classify_regime(nu: float) -> str:
    if not np.isfinite(nu):
        return "undetermined"
    if nu < 0.28:
        return "saturated (confined / territory-limited)"
    candidates = (
        (NU_FRACTAL_GLOBULE, "fractal globule"),
        (NU_IDEAL, "ideal chain"),
        (NU_SAW, "self-avoiding walk"),
    )
    ref, name = min(candidates, key=lambda c: abs(nu - c[0]))
    if abs(nu - ref) <= 0.05:
        return name
    if nu > NU_SAW:
        return "stiff (persistence-dominated)"
    return "intermediate"


def distance_scaling(x: np.ndarray, n_points: int = 40, fit_range: tuple[int, int] | None = None) -> ScalingFit:
    """RMS spatial distance vs genomic separation and a log-log OLS fit of nu.

    * RMS distance sqrt(<R^2(s)>) is the quantity polymer theory predicts (not <R>).
    * Separations are log-spaced so every decade of s carries equal weight in the fit;
      linear spacing would let the (end-effect-dominated) large-s tail dominate.
    * The default fit window [4, N/10] avoids the bond-scale regime (s < 4, set by
      smoothing / bond geometry) and finite-chain end effects (s > N/10).
    """
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if n < 8:
        empty = np.array([], dtype=float)
        return ScalingFit(empty, empty, empty, float("nan"), float("nan"), float("nan"), (0, 0), "undetermined")
    s = np.unique(np.round(np.geomspace(1, n - 1, n_points)).astype(np.int64))
    r = np.array([np.sqrt(np.mean(np.sum((x[k:] - x[:-k]) ** 2, axis=1))) for k in s])
    logs, logr = np.log(s), np.log(np.maximum(r, 1e-12))
    local = np.gradient(logr, logs) if len(s) > 2 else np.full_like(logr, np.nan)

    lo, hi = fit_range if fit_range else (4, max(4, n // 10))
    sel = (s >= lo) & (s <= hi)
    nu = se = r2 = float("nan")
    if sel.sum() >= 4 and hi / max(lo, 1) >= 5:
        A = np.vstack([logs[sel], np.ones(sel.sum())]).T
        coef, res, *_ = np.linalg.lstsq(A, logr[sel], rcond=None)
        nu = float(coef[0])
        pred = A @ coef
        ss_res = float(np.sum((logr[sel] - pred) ** 2))
        ss_tot = float(np.sum((logr[sel] - logr[sel].mean()) ** 2))
        dof = sel.sum() - 2
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        sxx = float(np.sum((logs[sel] - logs[sel].mean()) ** 2))
        se = float(np.sqrt(ss_res / dof / sxx)) if dof > 0 and sxx > 0 else float("nan")
    return ScalingFit(s, r, local, nu, se, r2, (int(lo), int(hi)), classify_regime(nu))


# ----------------------------------------------------------------------------------------
# Neighbour search: cell list, O(N) expected
# ----------------------------------------------------------------------------------------
_OFFSETS = np.array([(a, b, c) for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1)], dtype=np.int64)


def neighbor_pairs(x: np.ndarray, r: float, min_sep: int = 2) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """All pairs (i < j) with j - i >= min_sep and ||x_i - x_j|| < r.

    Beads are hashed into cubic cells of edge r; each bead is compared only with beads in the
    27 surrounding cells. Memory and time scale with N x (beads per cell), never N^2.
    Each unordered pair is visited exactly once (j > i is enforced before the distance test).
    """
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    empty = (np.empty(0, np.int64), np.empty(0, np.int64), np.empty(0, np.float64))
    if n < 2 or r <= 0:
        return empty
    cell = np.floor((x - x.min(axis=0)) / r).astype(np.int64) + 1   # +1 pad: neighbours never wrap
    dims = cell.max(axis=0) + 2
    stride = np.array([dims[1] * dims[2], dims[2], 1], dtype=np.int64)
    key = cell @ stride
    order = np.argsort(key, kind="stable")
    ukeys, ustart, ucount = np.unique(key[order], return_index=True, return_counts=True)

    out_i, out_j, out_d = [], [], []
    for off in _OFFSETS:
        nkey = key + off @ stride
        pos = np.minimum(np.searchsorted(ukeys, nkey), len(ukeys) - 1)
        src = np.flatnonzero(ukeys[pos] == nkey)
        if src.size == 0:
            continue
        cnt = ucount[pos[src]]
        total = int(cnt.sum())
        ii = np.repeat(src, cnt)
        within = np.arange(total) - np.repeat(np.cumsum(cnt) - cnt, cnt)
        jj = order[np.repeat(ustart[pos[src]], cnt) + within]
        keep = (jj - ii) >= min_sep
        ii, jj = ii[keep], jj[keep]
        d = np.linalg.norm(x[ii] - x[jj], axis=1)
        close = d < r
        out_i.append(ii[close])
        out_j.append(jj[close])
        out_d.append(d[close])
    if not out_i:
        return empty
    return np.concatenate(out_i), np.concatenate(out_j), np.concatenate(out_d)


# ----------------------------------------------------------------------------------------
# Contact <-> distance mapping
# ----------------------------------------------------------------------------------------
def reference_count(ci: np.ndarray, cj: np.ndarray, cm: np.ndarray) -> float:
    """M_ref: median count of nearest-neighbour (|i-j| = 1) contacts, the b0 anchor.

    Falls back to the 95th percentile of all counts if no backbone contacts exist.
    """
    adj = np.abs(np.asarray(cj) - np.asarray(ci)) == 1
    pool = cm[adj] if adj.any() else cm
    if pool.size == 0:
        raise ValueError("Contact list is empty.")
    return float(np.median(pool) if adj.any() else np.percentile(pool, 95))


def contact_target_distance(cm: np.ndarray, m_ref: float, b0: float = B0_NM, alpha: float = ALPHA,
                            clip: tuple[float, float] = TARGET_CLIP) -> np.ndarray:
    """d*_ij = b0 * (M_ij / M_ref)^(-1/alpha), clipped to [clip_lo, clip_hi] * b0.

    Only observed contacts (M_ij > 0) are passed in, so the power is always finite.
    Clipping bounds targets from single-read pairs (huge d*) and saturated pairs (d* -> 0).
    """
    cm = np.asarray(cm, dtype=np.float64)
    if np.any(cm <= 0):
        raise ValueError("Contact counts must be strictly positive; drop zero pixels upstream.")
    d = b0 * (cm / m_ref) ** (-1.0 / alpha)
    return np.clip(d, clip[0] * b0, clip[1] * b0)


# ----------------------------------------------------------------------------------------
# Loss terms (NumPy evaluation; the differentiable twins live in chronocell.egnn)
# ----------------------------------------------------------------------------------------
def loss_contact(x: np.ndarray, ci: np.ndarray, cj: np.ndarray, target: np.ndarray) -> float:
    """Relative stress: L_contact = (1/|C|) sum_C ((d_ij - d*_ij) / d*_ij)^2.

    Relative (not absolute) error because weak contacts - large d* - carry the most Poisson
    noise; an absolute error would let them dominate the objective.
    """
    if len(ci) == 0:
        return float("nan")
    d = np.linalg.norm(x[ci] - x[cj], axis=1)
    return float(np.mean(((d - target) / target) ** 2))


def loss_smooth(x: np.ndarray, b0: float = B0_NM) -> float:
    """Harmonic backbone: L_smooth = (1/(N-1)) sum_i ((||x_{i+1} - x_i|| - b0) / b0)^2.

    The rest length b0 is essential: without it (sum ||dx||^2) the minimum is a collapsed chain.
    """
    return float(np.mean(((bond_lengths(x) - b0) / b0) ** 2))


def loss_bend(x: np.ndarray, cos0: float = 0.0) -> float:
    """Bending stiffness: L_bend = mean_i (cos theta_i - cos0)^2, theta_i the angle between bonds i and i+1.

    A quadratic (harmonic-like) penalty on the bond-bond correlation. cos0 = 0 is the uncorrelated
    (freely-jointed) average; cos0 > 0 favours straighter, stiffer chains.
    """
    v = np.diff(np.asarray(x, dtype=np.float64), axis=0)
    v /= np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)
    if len(v) < 2:
        return 0.0
    return float(np.mean((np.sum(v[1:] * v[:-1], axis=1) - cos0) ** 2))


def loss_confinement(x: np.ndarray, radius: float, b0: float = B0_NM) -> float:
    """Nuclear-envelope confinement: L_conf = mean_i (max(0, |x_i - x_cm| - R) / b0)^2 (spherical wall)."""
    x = np.asarray(x, dtype=np.float64)
    r = np.linalg.norm(x - x.mean(axis=0), axis=1)
    return float(np.mean((np.maximum(r - radius, 0.0) / b0) ** 2))


@dataclass(frozen=True)
class StericReport:
    loss: float             # (1/N) sum_{|i-j|>1, d<d_min} ((d_min - d)/b0)^2
    overlaps: int           # pairs penetrating deeper than `tolerance`
    max_penetration: float  # nm; 0 if no pair is inside d_min
    min_distance: float     # closest non-bonded pair within d_min (nm), inf if none


def loss_steric(x: np.ndarray, b0: float = B0_NM, d_min: float | None = None,
                tolerance: float | None = None) -> StericReport:
    """Soft-core excluded volume for non-bonded pairs |i - j| > 1.

    A quadratic barrier max(0, d_min - d)^2: zero force outside d_min, a linear restoring force
    inside - a truncated, softened hard sphere that stays finite (unlike Lennard-Jones) when
    beads overlap during early optimisation. Summed over violating pairs and divided by N so
    the term is intensive; dividing by |Omega| ~ N^2/2 would dilute it to nothing.

    A soft barrier in equilibrium with attractive terms always leaves pairs a fraction of a nm
    inside d_min, so an overlap is only counted beyond `tolerance` (default 2 % of b0 = 1 nm).
    """
    d_min = D_MIN_FACTOR * b0 if d_min is None else d_min
    tolerance = 0.02 * b0 if tolerance is None else tolerance
    _, _, d = neighbor_pairs(x, d_min, min_sep=2)
    pen = d_min - d
    loss = float(np.sum((pen / b0) ** 2) / len(x)) if len(x) else 0.0
    return StericReport(loss, int(np.sum(pen > tolerance)), float(pen.max()) if pen.size else 0.0,
                        float(d.min()) if d.size else float("inf"))


# ----------------------------------------------------------------------------------------
# Structure comparison
# ----------------------------------------------------------------------------------------
def kabsch_rmsd(ref: np.ndarray, mobile: np.ndarray, allow_reflection: bool = True) -> tuple[float, np.ndarray, bool]:
    """Optimal-superposition RMSD of `mobile` onto `ref`.

    Contact maps (and any distance-based model, including an E(3)-equivariant EGNN) cannot
    distinguish a structure from its mirror image, so reconstruction accuracy must be measured
    over all orthogonal transforms, O(3), unless chirality is known from another source.
    Returns (rmsd_nm, mobile_aligned, reflected).
    """
    p = np.asarray(ref, dtype=np.float64)
    q = np.asarray(mobile, dtype=np.float64)
    pc, qc = p - p.mean(axis=0), q - q.mean(axis=0)
    u, _, vt = np.linalg.svd(qc.T @ pc)
    rot = u @ vt
    reflected = bool(np.linalg.det(rot) < 0)
    if reflected and not allow_reflection:
        u[:, -1] *= -1
        rot = u @ vt
        reflected = False
    aligned = qc @ rot + p.mean(axis=0)
    rmsd = float(np.sqrt(np.mean(np.sum((aligned - p) ** 2, axis=1))))
    return rmsd, aligned, reflected


def distance_correlation(a: np.ndarray, b: np.ndarray, n_pairs: int = 50_000, seed: int = 0) -> float:
    """Pearson correlation of pairwise distances between two conformations (rotation-free)."""
    n = len(a)
    rng = np.random.default_rng(seed)
    i = rng.integers(0, n, n_pairs)
    j = rng.integers(0, n, n_pairs)
    keep = i != j
    da = np.linalg.norm(a[i[keep]] - a[j[keep]], axis=1)
    db = np.linalg.norm(b[i[keep]] - b[j[keep]], axis=1)
    return float(np.corrcoef(da, db)[0, 1])


# ----------------------------------------------------------------------------------------
# Contact-map statistics and 2D maps
# ----------------------------------------------------------------------------------------
def contact_decay(ci: np.ndarray, cj: np.ndarray, cm: np.ndarray, n: int, n_points: int = 30,
                  fit_range: tuple[int, int] | None = None) -> tuple[np.ndarray, np.ndarray, float]:
    """P(s): mean contact count per pair at separation s (zeros included), and gamma in P ~ s^-gamma."""
    s = np.abs(cj - ci)
    sums = np.bincount(s, weights=cm, minlength=n)
    pairs = np.maximum(n - np.arange(n), 1)
    p = sums / pairs
    grid = np.unique(np.round(np.geomspace(1, max(2, n - 1), n_points)).astype(np.int64))
    grid = grid[grid < n]
    ps = p[grid]
    lo, hi = fit_range if fit_range else (2, max(10, n // 10))
    sel = (grid >= lo) & (grid <= hi) & (ps > 0)
    gamma = float(-np.polyfit(np.log(grid[sel]), np.log(ps[sel]), 1)[0]) if sel.sum() >= 4 else float("nan")
    return grid, ps, gamma


def coarse_distance_map(x: np.ndarray, max_px: int = 400) -> tuple[np.ndarray, int]:
    """Pairwise distances between block centroids; windows > max_px beads are block-averaged."""
    n = len(x)
    k = max(1, int(np.ceil(n / max_px)))
    m = n // k
    pooled = x[: m * k].reshape(m, k, 3).mean(axis=1)
    if m * k < n:
        pooled = np.vstack([pooled, x[m * k:].mean(axis=0, keepdims=True)])
    diff = pooled[:, None, :] - pooled[None, :, :]
    return np.sqrt(np.sum(diff ** 2, axis=-1)).astype(np.float32), k


def coarse_contact_map(ci: np.ndarray, cj: np.ndarray, cm: np.ndarray, lo: int, hi: int,
                       max_px: int = 400) -> tuple[np.ndarray, int]:
    """Summed contact counts in k x k bin blocks for the window [lo, hi), symmetric."""
    n = hi - lo
    k = max(1, int(np.ceil(n / max_px)))
    m = int(np.ceil(n / k))
    inside = (ci >= lo) & (ci < hi) & (cj >= lo) & (cj < hi)
    a = (ci[inside] - lo) // k
    b = (cj[inside] - lo) // k
    w = cm[inside]
    mat = np.zeros((m, m), dtype=np.float64)
    np.add.at(mat, (a, b), w)
    off = a != b                       # mirror off-diagonal blocks only; never double the diagonal
    np.add.at(mat, (b[off], a[off]), w[off])
    return mat, k


# ----------------------------------------------------------------------------------------
# Shape and packing descriptors (metric dashboard and ChronoAgent)
# ----------------------------------------------------------------------------------------
def max_span(x: np.ndarray, chunk: int = 256) -> float:
    """Maximum pairwise distance (the structure's 3D diameter), exact.

    The farthest pair always lies on the convex hull; to stay dependency-free the search is
    pruned instead: beads within the inscribed sphere of the bounding box that cannot beat the
    current best are dropped, then the remaining pairs are scanned in chunks (O(N * M) memory).
    """
    x = np.asarray(x, dtype=np.float64)
    if len(x) < 2:
        return 0.0
    c = x.mean(axis=0)
    r = np.linalg.norm(x - c, axis=1)
    order = np.argsort(r)[::-1]
    best = float(np.max(np.linalg.norm(x - x[order[0]], axis=1)))   # lower bound from the outermost bead
    # A pair (i, j) can only exceed `best` if r_i + r_j > best (triangle inequality through c).
    cand = x[r + r.max() > best] if best > 0 else x
    for s in range(0, len(cand), chunk):
        blk = cand[s:s + chunk]
        d2 = np.sum((blk[:, None, :] - cand[None, :, :]) ** 2, axis=-1)
        best = max(best, float(np.sqrt(d2.max())))
    return best


@dataclass(frozen=True)
class GyrationShape:
    eigenvalues: tuple[float, float, float]   # λ1 >= λ2 >= λ3 of the gyration tensor (nm²); Σλ = R_g²
    asphericity: float                         # (λ1 - (λ2+λ3)/2) / R_g², 0 = sphere, 1 = rod
    anisotropy: float                          # relative shape anisotropy κ² in [0, 1]


def gyration_shape(x: np.ndarray) -> GyrationShape:
    """Principal moments of the gyration tensor S = (1/N) Σ (x_i - x_cm)(x_i - x_cm)^T."""
    x = np.asarray(x, dtype=np.float64)
    d = x - x.mean(axis=0)
    lam = np.sort(np.linalg.eigvalsh(d.T @ d / max(len(x), 1)))[::-1]
    tr = float(lam.sum())
    if tr <= 0:
        return GyrationShape((0.0, 0.0, 0.0), float("nan"), float("nan"))
    b = float(lam[0] - 0.5 * (lam[1] + lam[2]))
    kappa2 = 1.0 - 3.0 * float(lam[0] * lam[1] + lam[1] * lam[2] + lam[0] * lam[2]) / tr ** 2
    return GyrationShape(tuple(float(v) for v in lam), b / tr, kappa2)


def local_density(x: np.ndarray, r: float, min_sep: int = 2) -> np.ndarray:
    """Per-bead count of non-bonded beads (|i-j| >= min_sep) within distance r (cell list, O(N))."""
    i, j, _ = neighbor_pairs(x, r, min_sep=min_sep)
    n = len(x)
    return (np.bincount(i, minlength=n) + np.bincount(j, minlength=n)).astype(np.float64)
