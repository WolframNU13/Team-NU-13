"""
Per-bin node features and contact extraction (replaces the Teammate 1 loop).

Corrections relative to the original pipeline
* f_GC divides by the number of *called* bases (A/C/G/T), not by B. Dividing by B made the
  final partial bin (8,468 bp) and every N-rich bin look AT-rich; an all-N bin is reported as
  missing (NaN), not as 0 % GC.
* f_epi is a NaN-aware mean. bigWig returns NaN where there is no coverage; nan_to_num(...)
  turned "no data" into "no signal".
* Contacts keep i < j, drop the diagonal and merge mirrored pixels (see formats.canonical_contacts).
Everything is vectorised: one reshape per track instead of a Python loop over 5,082 bins.
"""

from __future__ import annotations

import warnings

import numpy as np

from . import genome

_A, _C, _G, _T = (ord(c) for c in "ACGT")


def _pad_to_bins(a: np.ndarray, n_bins: int, resolution: int, fill) -> np.ndarray:
    need = n_bins * resolution
    if a.size < need:
        a = np.concatenate([a, np.full(need - a.size, fill, dtype=a.dtype)])
    return a[:need].reshape(n_bins, resolution)


def gc_fraction(sequence: bytes | str, n_bins: int = genome.N_BINS, resolution: int = genome.RESOLUTION,
                min_called: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """f_GC(i) = #{G, C} / #{A, C, G, T} over bin i (case-insensitive; soft-masked bases count).

    Returns (f_gc, valid) where valid means at least `min_called` of the bin's bases are called;
    invalid bins get NaN.
    """
    raw = sequence.encode() if isinstance(sequence, str) else sequence
    seq = np.frombuffer(raw, dtype=np.uint8) & 0xDF              # ASCII upper-case
    seq = _pad_to_bins(seq, n_bins, resolution, ord("N"))
    gc = ((seq == _G) | (seq == _C)).sum(axis=1)
    called = gc + ((seq == _A) | (seq == _T)).sum(axis=1)
    lengths = genome.bin_lengths(n_bins) if n_bins == genome.N_BINS else np.full(n_bins, resolution)
    valid = called >= min_called * lengths
    with np.errstate(invalid="ignore", divide="ignore"):
        f = np.where(valid, gc / np.maximum(called, 1), np.nan)
    return f, valid


def binned_mean(values: np.ndarray, n_bins: int = genome.N_BINS, resolution: int = genome.RESOLUTION) -> np.ndarray:
    """NaN-aware per-bin mean of a base-resolution signal (e.g. pyBigWig .values output)."""
    v = _pad_to_bins(np.asarray(values, dtype=np.float64), n_bins, resolution, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)   # all-NaN bins -> NaN
        return np.nanmean(v, axis=1)


def signal_hubs(epi: np.ndarray, valid: np.ndarray, pct: float = 97.0, gap: int = 2,
                min_len: int = 2) -> list[tuple[int, int, float]]:
    """Runs of bins whose 5-bin smoothed H3K27ac exceeds the `pct` percentile of assembled bins.

    Returns (start_bin, end_bin_exclusive, peak signal), strongest first. Bins without sequence
    never qualify, so assembly gaps cannot create or split a hub.
    """
    e = np.where(valid & np.isfinite(epi), epi, 0.0)
    if not valid.any():
        return []
    smooth = np.convolve(e, np.ones(5) / 5.0, mode="same")
    thr = np.percentile(smooth[valid], pct)
    hot = np.flatnonzero((smooth >= thr) & valid)
    if hot.size == 0:
        return []
    runs = np.split(hot, np.flatnonzero(np.diff(hot) > gap + 1) + 1)
    hubs = [(int(r[0]), int(r[-1]) + 1, float(e[r[0]:r[-1] + 1].max())) for r in runs if len(r) >= min_len]
    return sorted(hubs, key=lambda h: -h[2])


def contacts_from_pixels(bin1: np.ndarray, bin2: np.ndarray, count: np.ndarray, n_bins: int,
                         min_separation: int = 1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cooler pixel table (bin1_id, bin2_id, count) -> canonical upper-triangle contacts.

    Using the sparse pixel table avoids the dense 5,082 x 5,082 float64 matrix (207 MB).
    """
    from .formats import canonical_contacts

    ci, cj, cm, _ = canonical_contacts(bin1, bin2, count, n_bins)
    keep = (cj - ci) >= min_separation
    return ci[keep], cj[keep], cm[keep]
