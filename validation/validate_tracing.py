"""
Accuracy of ChronoCell-5D's 3D reconstruction against real microscopy (chromatin tracing).

Data: Bintu et al., Science 2018 (github.com/BogdanBintu/ChromatinImaging), multiplexed DNA FISH:
the 3D position (nm) of every 30 kb segment of a 2 Mb region, in thousands of single cells.

Test (no information leaks from answer to model):
  1. Split the cells at random into two halves, A and B.
  2. From half A keep ONLY contact frequencies: how often two segments are closer than 150 nm
     (the quantity Hi-C / Micro-C estimates). All distances of half A are thrown away.
  3. ChronoCell-5D rebuilds the 3D fold from those contact frequencies alone (same pipeline as the
     app: contact -> distance mapping, MDS start, gradient + EGNN refinement).
  4. Compare every model distance with the median distance MEASURED in half B (cells the model
     never saw).
Scores: Spearman / Pearson correlation of the distance matrices; the same after removing the
trivial "further along the DNA = further in space" trend (distance-corrected); Lin's concordance;
and two references: a baseline that only knows genomic separation, and the ceiling = how well the
two halves of the experiment agree with each other.

    python validation/validate_tracing.py            # writes validation/results.json (data downloads on first run)
"""

from __future__ import annotations

import json
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from chronocell import agent, egnn, physics  # noqa: E402

BASE = "https://raw.githubusercontent.com/BogdanBintu/ChromatinImaging/master/Data/"
DATASETS = [("IMR90", "chr21:28-30 Mb", "IMR90_chr21-28-30Mb.csv"),
            ("IMR90", "chr21:18-20 Mb", "IMR90_chr21-18-20Mb.csv"),
            ("A549", "chr21:28-30 Mb", "A549_chr21-28-30Mb.csv")]
CONTACT_NM = 150.0          # Bintu et al. contact threshold
SEGMENT_BP = 30_000
SPLITS = 3                  # independent random half/half splits per dataset


def load(name: str) -> np.ndarray:
    path = ROOT / "data" / name
    if not path.exists():
        path.parent.mkdir(exist_ok=True)
        urllib.request.urlretrieve(BASE + urllib.request.quote(name), path)
    rows = np.genfromtxt(path, delimiter=",", skip_header=2)
    cell = rows[:, 0].astype(int)
    seg = rows[:, 1].astype(int) - 1
    n_cells, n_seg = cell.max(), seg.max() + 1
    xyz = np.full((n_cells, n_seg, 3), np.nan)
    xyz[cell - 1, seg] = rows[:, 2:5]
    ok = np.isfinite(xyz[..., 0]).mean(axis=1) >= 0.5         # keep cells with >= half the segments detected
    return xyz[ok]


def pair_dist(xyz: np.ndarray) -> np.ndarray:
    return np.linalg.norm(xyz[:, :, None, :] - xyz[:, None, :, :], axis=-1)   # (cells, n, n), NaN if missing


def contact_frequency(d: np.ndarray) -> np.ndarray:
    seen = np.isfinite(d).sum(axis=0)
    close = (d < CONTACT_NM).sum(axis=0)
    return close / np.maximum(seen, 1)


def reconstruct(freq: np.ndarray, seed: int) -> np.ndarray:
    n = len(freq)
    i, j = np.triu_indices(n, 1)
    m = freq[i, j]
    keep = m > 0
    b0 = physics.bond_length_for(SEGMENT_BP)
    feats = egnn.node_features(np.full(n, 0.42), np.zeros(n), np.ones(n, bool))
    cfg = egnn.FitConfig(seed=seed)
    res = egnn.fit_structure(n, feats, i[keep], j[keep], m[keep] * 1000.0, cfg, b0=b0)
    return res.coords_nm


def lin_ccc(a: np.ndarray, b: np.ndarray) -> float:
    cov = np.mean((a - a.mean()) * (b - b.mean()))
    return float(2 * cov / (a.var() + b.var() + (a.mean() - b.mean()) ** 2))


def scores(pred: np.ndarray, truth: np.ndarray) -> dict:
    n = len(truth)
    i, j = np.triu_indices(n, 1)
    p, t, s = pred[i, j], truth[i, j], j - i
    ok = np.isfinite(p) & np.isfinite(t)
    p, t, s = p[ok], t[ok], s[ok]
    # distance-corrected: divide each value by the mean at its genomic separation (observed / expected)
    pe = np.array([p[s == k].mean() for k in range(n)])[s] if len(p) else p
    te = np.array([t[s == k].mean() if (s == k).any() else np.nan for k in range(n)])[s]
    return {"spearman": agent.spearman(p, t), "pearson": float(np.corrcoef(p, t)[0, 1]),
            "spearman_distance_corrected": agent.spearman(p / pe, t / te),
            "lin_ccc_nm": lin_ccc(p, t), "median_scale_model_over_real": float(np.median(p / t))}


def genomic_baseline(freq_a_dist: np.ndarray) -> np.ndarray:
    """Distance predicted from genomic separation alone (power law fitted on half A)."""
    n = len(freq_a_dist)
    i, j = np.triu_indices(n, 1)
    s = (j - i).astype(float)
    ok = np.isfinite(freq_a_dist[i, j])
    k, logc = np.polyfit(np.log(s[ok]), np.log(freq_a_dist[i, j][ok]), 1)
    out = np.exp(logc) * np.abs(np.subtract.outer(np.arange(n), np.arange(n))).astype(float) ** k
    return out


def main() -> None:
    t0 = time.time()
    results = []
    for cell_line, region, fname in DATASETS:
        xyz = load(fname)
        d_all = pair_dist(xyz)
        for split in range(SPLITS):
            rng = np.random.default_rng(split)
            idx = rng.permutation(len(xyz))
            a, b = idx[: len(idx) // 2], idx[len(idx) // 2:]
            freq_a = contact_frequency(d_all[a])
            truth_b = np.nanmedian(d_all[b], axis=0)
            median_a = np.nanmedian(d_all[a], axis=0)
            model = reconstruct(freq_a, seed=split)
            d_model = np.linalg.norm(model[:, None] - model[None], axis=-1)
            results.append({
                "cell_line": cell_line, "region": region, "split": split, "cells_total": int(len(xyz)),
                "segments": int(xyz.shape[1]),
                "model": scores(d_model, truth_b),
                "baseline_genomic_distance_only": scores(genomic_baseline(median_a), truth_b),
                "ceiling_half_A_vs_half_B": scores(median_a, truth_b),
            })
            r = results[-1]
            print(f"{cell_line} {region} split {split}: model rho {r['model']['spearman']:.3f} "
                  f"(corrected {r['model']['spearman_distance_corrected']:.3f}) | baseline {r['baseline_genomic_distance_only']['spearman']:.3f} "
                  f"(corrected {r['baseline_genomic_distance_only']['spearman_distance_corrected']:.3f}) | ceiling {r['ceiling_half_A_vs_half_B']['spearman']:.3f} "
                  f"(corrected {r['ceiling_half_A_vs_half_B']['spearman_distance_corrected']:.3f})", flush=True)
    (ROOT / "results.json").write_text(json.dumps(results, indent=1, default=float))
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
