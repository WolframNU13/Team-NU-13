"""
Synthetic ground truth for chr22 at 10 kb: a fractal-globule conformation, band-informed 1D
tracks and Micro-C-like contacts sampled from that conformation.

Why a Hilbert curve. The fractal (crumpled) globule of Grosberg et al. (1988) and
Lieberman-Aiden et al. (Science 2009) is realised exactly by a 3D Hilbert curve: a
space-filling, unknotted path with R(s) ~ s^(1/3) and P(s) ~ s^-1. It is generated here in
closed form by bit manipulation (Skilling, AIP Conf Proc 2004), vectorised over all beads.

Everything is deterministic given `seed`. Contacts are drawn with the same power law used to
invert them (M ~ d^-alpha), so a reconstruction can be scored against the known structure.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import genome, physics


# ----------------------------------------------------------------------------------------
# Hilbert curve (Skilling's transpose algorithm, vectorised over indices)
# ----------------------------------------------------------------------------------------
def hilbert_curve_3d(n_points: int, order: int | None = None) -> np.ndarray:
    """Integer lattice coordinates of the first `n_points` vertices of a 3D Hilbert curve.

    Consecutive vertices differ by exactly one lattice unit along one axis.
    """
    if order is None:
        order = max(1, int(np.ceil(np.log2(n_points) / 3)))
    if n_points > 8 ** order:
        raise ValueError("n_points exceeds the curve length for this order.")
    h = np.arange(n_points, dtype=np.int64)
    # Transposed index: bit k of axis d is bit (3k + 2 - d) of h (MSB of h -> MSB of X[0]).
    X = [np.zeros(n_points, dtype=np.int64) for _ in range(3)]
    for k in range(order):
        for d in range(3):
            X[d] |= ((h >> (3 * k + 2 - d)) & 1) << k
    # Gray decode
    t = X[2] >> 1
    X[2] ^= X[1]
    X[1] ^= X[0]
    X[0] ^= t
    # Undo excess work
    q = 2
    top = 1 << order
    while q != top:
        p = q - 1
        for i in (2, 1, 0):
            bit = (X[i] & q) != 0
            if i == 0:
                X[0] = np.where(bit, X[0] ^ p, X[0])
            else:
                t = (X[0] ^ X[i]) & p
                X[0], X[i] = np.where(bit, X[0] ^ p, X[0] ^ t), np.where(bit, X[i], X[i] ^ t)
        q <<= 1
    return np.stack(X, axis=1)


def _gaussian_smooth_rows(x: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian filter along the chain (axis 0), reflect-padded; vectorised per column."""
    if sigma <= 0:
        return x.copy()
    radius = max(1, int(3 * sigma))
    k = np.exp(-0.5 * (np.arange(-radius, radius + 1) / sigma) ** 2)
    k /= k.sum()
    padded = np.pad(x, ((radius, radius), (0, 0)), mode="reflect")
    windows = np.lib.stride_tricks.sliding_window_view(padded, 2 * radius + 1, axis=0)
    return windows @ k


def relax(x: np.ndarray, b0: float, d_min: float, iters: int = 40, rebuild: int = 10) -> np.ndarray:
    """Position-based relaxation: separate non-bonded overlaps and restore bond lengths.

    Each sweep is vectorised (np.add.at scatter); the loop is over sweeps. Candidate pairs come
    from a Verlet list (cell list with cutoff 1.5 d_min) rebuilt every `rebuild` sweeps, since
    beads move far less than 0.5 d_min between rebuilds.
    """
    x = x.copy()
    ci = cj = np.empty(0, np.int64)
    for sweep in range(iters):
        if sweep % rebuild == 0:
            ci, cj, _ = physics.neighbor_pairs(x, 1.5 * d_min, min_sep=2)
        d = np.linalg.norm(x[cj] - x[ci], axis=1)
        close = d < d_min
        i, j, d = ci[close], cj[close], d[close]
        if i.size:
            u = (x[j] - x[i]) / np.maximum(d, 1e-9)[:, None]
            push = 0.5 * (d_min - d)[:, None] * u
            np.add.at(x, i, -push)
            np.add.at(x, j, push)
        bond = x[1:] - x[:-1]
        length = np.linalg.norm(bond, axis=1, keepdims=True)
        corr = 0.5 * (length - b0) / np.maximum(length, 1e-9) * bond
        delta = np.zeros_like(x)
        delta[:-1] += 0.5 * corr
        delta[1:] -= 0.5 * corr
        x += delta
        if i.size == 0 and np.max(np.abs(length - b0)) < 0.02 * b0 and sweep % rebuild == rebuild - 1:
            break
    return x


def fractal_globule(n: int = genome.N_BINS, b0: float = physics.B0_NM, seed: int = 7) -> np.ndarray:
    """A crumpled-globule conformation in nm with median bond b0 and no steric overlaps.

    The complete Hilbert curve of the smallest sufficient order is sampled at uniform arc
    length, so the globule fills an isotropic volume. Taking only the first n vertices instead
    fills one sub-cube and spills the remainder into a half-empty neighbour - a lopsided
    'tower' that is an artefact of the curve, not of chromatin.
    """
    rng = np.random.default_rng(seed)
    order = max(1, int(np.ceil(np.log2(n) / 3)))
    total = 8 ** order
    lattice = hilbert_curve_3d(total, order)[np.round(np.linspace(0, total - 1, n)).astype(np.int64)]
    lattice = lattice.astype(np.float64)

    # Round the cube into a territory-like body: partial radial squash towards a sphere.
    mid = (lattice.max(axis=0) + lattice.min(axis=0)) / 2
    half = np.maximum((lattice.max(axis=0) - lattice.min(axis=0)) / 2, 1.0)
    u = (lattice - mid) / half
    r_inf = np.max(np.abs(u), axis=1)
    r_2 = np.linalg.norm(u, axis=1)
    squash = np.where(r_2 > 0, (r_inf / np.maximum(r_2, 1e-12)) ** 0.5, 1.0)
    x = u * squash[:, None] * half

    # Low-frequency bending: a few random plane waves (amplitude ~2 lattice units).
    extent = float(np.max(half)) * 2
    for _ in range(6):
        kvec = rng.normal(size=3)
        kvec *= (2 * np.pi / extent) * rng.uniform(0.6, 1.6) / np.linalg.norm(kvec)
        amp = rng.normal(size=3) * 1.5
        x += np.sin(x @ kvec + rng.uniform(0, 2 * np.pi))[:, None] * amp

    # Light smoothing rounds lattice corners; noise breaks exact lattice degeneracy. Bond lengths
    # are then set *locally* by relaxation rather than by a global rescale.
    # Lattice spacing 0.85 b0 leaves free volume: at spacing ~0.7 b0 the chain is jammed and
    # bond lengths and excluded volume cannot both be satisfied.
    x = _gaussian_smooth_rows(x, sigma=0.35)
    x += rng.normal(scale=0.1, size=x.shape)
    x = relax(x * 0.85 * b0, b0=b0, d_min=1.03 * physics.D_MIN_FACTOR * b0, iters=2000)
    x, _ = physics.calibrate_to_bond_length(x, b0)        # residual factor ~1.00
    return x - x.mean(axis=0)


# ----------------------------------------------------------------------------------------
# 1D tracks
# ----------------------------------------------------------------------------------------
# Giemsa-negative (R) bands are GC-rich and gene-dense; G-positive bands are AT-rich.
_STAIN_GC = {"gneg": 0.49, "gpos25": 0.445, "gpos50": 0.415, "gpos75": 0.40, "gpos100": 0.385,
             "acen": 0.39, "gvar": 0.42, "stalk": 0.55}


def synthetic_tracks(n: int = genome.N_BINS, seed: int = 7) -> dict[str, np.ndarray]:
    """GC fraction and H3K27ac mean signal per bin; NaN in unassembled (N) bins."""
    rng = np.random.default_rng(seed + 1)
    valid = genome.assembled_mask(n)
    band = genome.band_for_bins(n)
    base = np.array([_STAIN_GC[genome.CYTOBANDS[b].stain] for b in band])
    iso = _gaussian_smooth_rows(rng.normal(size=(n, 1)), 15.0)[:, 0]
    gc = np.clip(base + 0.02 * iso / max(iso.std(), 1e-9) + rng.normal(0, 0.008, n), 0.30, 0.68)

    # H3K27ac: gamma background, peak rate rising with GC (gene density), log-normal heights.
    epi = rng.gamma(1.4, 0.35, n)
    rate = 0.004 + 0.09 * np.clip((gc - 0.43) / 0.08, 0, None) ** 1.5
    peaks = rng.random(n) < rate
    kernel = np.array([0.25, 0.6, 1.0, 0.6, 0.25])
    epi += np.convolve(peaks * rng.lognormal(1.3, 0.55, n), kernel, mode="same")
    rich = np.flatnonzero(valid & (gc > np.nanpercentile(gc[valid], 85)))
    for c in rng.choice(rich, size=min(8, rich.size), replace=False):   # super-enhancer-like hubs
        epi += rng.uniform(5, 9) * np.exp(-0.5 * ((np.arange(n) - c) / rng.uniform(2.5, 6)) ** 2)

    gc[~valid] = np.nan
    epi[~valid] = np.nan
    return {"gc": gc, "epi": epi, "valid": valid}


# ----------------------------------------------------------------------------------------
# Contacts
# ----------------------------------------------------------------------------------------
def simulate_contacts(x: np.ndarray, valid: np.ndarray, b0: float = physics.B0_NM,
                      alpha: float = physics.ALPHA, depth: float = 60.0, r_cut: float = 2.5,
                      seed: int = 7) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Poisson counts with mean depth * (d / b0)^-alpha for pairs closer than r_cut * b0.

    Bins without sequence (unassembled) receive no reads, as in a real Micro-C experiment.
    Returns upper-triangle (i < j) contacts with count > 0.
    """
    rng = np.random.default_rng(seed + 2)
    i, j, d = physics.neighbor_pairs(x, r_cut * b0, min_sep=1)
    keep = valid[i] & valid[j]
    i, j, d = i[keep], j[keep], d[keep]
    lam = depth * (np.maximum(d, 0.2 * b0) / b0) ** (-alpha)
    m = rng.poisson(lam)
    nz = m > 0
    order = np.lexsort((j[nz], i[nz]))
    return i[nz][order], j[nz][order], m[nz][order].astype(np.float64)


@dataclass
class SyntheticChromosome:
    coords: np.ndarray
    gc: np.ndarray
    epi: np.ndarray
    valid: np.ndarray
    ci: np.ndarray
    cj: np.ndarray
    cm: np.ndarray
    meta: dict = field(default_factory=dict)


def build(n: int = genome.N_BINS, seed: int = 7, b0: float = physics.B0_NM) -> SyntheticChromosome:
    x = fractal_globule(n, b0=b0, seed=seed)
    tr = synthetic_tracks(n, seed=seed)
    ci, cj, cm = simulate_contacts(x, tr["valid"], b0=b0, seed=seed)
    meta = {"generator": "Hilbert fractal globule + band-informed tracks + Poisson contacts",
            "seed": seed, "b0_nm": b0, "alpha": physics.ALPHA, "depth": 60.0, "r_cut_b0": 2.5}
    return SyntheticChromosome(x, tr["gc"], tr["epi"], tr["valid"], ci, cj, cm, meta)
