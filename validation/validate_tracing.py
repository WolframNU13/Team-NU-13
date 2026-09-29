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
Two reconstructions are scored:
  * v3.2 single structure (chronocell.egnn);
  * v3.3 population model (chronocell.ensemble): a maximum-entropy Gaussian ensemble of chains,
    sampled by 100 exact Langevin trajectories. Its prediction is the ensemble median distance,
    reported both exactly and from the 100 x 50 sampled structures.
A reference with no 3D model at all (each contact frequency inverted on its own) shows what the
3D population adds.

Two scores, never mixed:
  * Contact-map fit: Spearman between the model's contacts and the INPUT frequencies (half A).
    It shows the optimisation converged; it is not evidence of accuracy.
  * Microscopy accuracy: agreement with half B's measured distances, which the model never saw.
    Headline = trend-removed Spearman as a fraction of the ceiling; raw Spearman, Pearson and Lin's
    concordance (nm) are reported too, with two references: a baseline that only knows genomic
    separation, and the ceiling = how well the two halves of the experiment agree with each other.

Held-out protocol: settings were tuned only on PRACTICE datasets (validation/TUNING.md); the three
TEST datasets were not used for any choice.

    python validation/validate_tracing.py              # test datasets -> validation/results.json
    python validation/validate_tracing.py --practice   # practice datasets -> validation/results_practice.json
(data download on first run into validation/data/)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from chronocell import agent, egnn, ensemble, physics  # noqa: E402

BASE = "https://raw.githubusercontent.com/BogdanBintu/ChromatinImaging/master/Data/"
DATASETS = [("IMR90", "chr21:28-30 Mb", "IMR90_chr21-28-30Mb.csv"),
            ("IMR90", "chr21:18-20 Mb", "IMR90_chr21-18-20Mb.csv"),
            ("A549", "chr21:28-30 Mb", "A549_chr21-28-30Mb.csv")]
PRACTICE = [("K562", "chr21:28-30 Mb", "K562_chr21-28-30Mb.csv"),
            ("HCT116", "chr21:28-30 Mb", "HCT116_chr21-28-30Mb_untreated.csv"),
            ("HCT116 + auxin (cohesin depleted)", "chr21:28-30 Mb", "HCT116_chr21-28-30Mb_6h auxin.csv"),
            ("HCT116", "chr21:34-37 Mb", "HCT116_chr21-34-37Mb_untreated.csv")]
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
    """v3.2 single structure (chronocell.egnn), exactly as the app builds it."""
    n = len(freq)
    i, j = np.triu_indices(n, 1)
    m = freq[i, j]
    keep = m > 0
    b0 = physics.bond_length_for(SEGMENT_BP)
    feats = egnn.node_features(np.full(n, 0.42), np.zeros(n), np.ones(n, bool))
    cfg = egnn.FitConfig(seed=seed)
    res = egnn.fit_structure(n, feats, i[keep], j[keep], m[keep] * 1000.0, cfg, b0=b0)
    return res.coords_nm


def reconstruct_ensemble(freq: np.ndarray, seen: np.ndarray, seed: int) -> ensemble.EnsembleResult:
    """v3.3 population model with the app's default settings (chronocell.ensemble.EnsembleConfig,
    frozen after tuning on the practice datasets)."""
    return ensemble.fit_ensemble(freq, seen, r_c_nm=CONTACT_NM, cfg=ensemble.EnsembleConfig(seed=seed))


def contact_fit(model_contacts: np.ndarray, freq: np.ndarray) -> float:
    """Spearman between the model's contact score and the input frequencies (all pairs i < j)."""
    i, j = np.triu_indices(len(freq), 1)
    ok = np.isfinite(model_contacts[i, j]) & np.isfinite(freq[i, j])
    return agent.spearman(model_contacts[i, j][ok], freq[i, j][ok])


def spread_cv(d: np.ndarray) -> float:
    """Cell-to-cell variability: mean over pairs of std/mean distance (NaN-aware)."""
    i, j = np.triu_indices(d.shape[1], 1)
    x = d[:, i, j]
    return float(np.nanmean(np.nanstd(x, axis=0) / np.nanmean(x, axis=0)))


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


def summarise(results: list[dict]) -> dict:
    """Per dataset: mean over splits. Overall: sum(model) / sum(ceiling) of the trend-removed Spearman
    (the ceiling-weighted mean of per-dataset ratios, fixed before tuning; see UPDATES.md)."""
    out: dict = {"datasets": {}, "overall": {}}
    names = list(dict.fromkeys(f"{r['cell_line']} {r['region']}" for r in results))
    for model in ("single_structure_v3_2", "ensemble_v3_3", "ensemble_v3_3_100_trajectories",
                  "reference_direct_inversion_no_3d"):
        tot_m = tot_c = 0.0
        for name in names:
            rs = [r for r in results if f"{r['cell_line']} {r['region']}" == name]
            m = float(np.mean([r[model]["microscopy"]["spearman_distance_corrected"] for r in rs]))
            c = float(np.mean([r["ceiling_half_A_vs_half_B"]["spearman_distance_corrected"] for r in rs]))
            tot_m, tot_c = tot_m + m, tot_c + c
            out["datasets"].setdefault(name, {})[model] = {
                "microscopy_trend_removed": m, "ceiling_trend_removed": c, "percent_of_ceiling": 100 * m / c,
                "microscopy_raw_spearman": float(np.mean([r[model]["microscopy"]["spearman"] for r in rs])),
                "lin_ccc_nm": float(np.mean([r[model]["microscopy"]["lin_ccc_nm"] for r in rs])),
                "contact_map_fit": float(np.mean([r[model].get("contact_map_fit", np.nan) for r in rs])),
                "splits": len(rs)}
        out["overall"][model] = {"percent_of_ceiling": 100 * tot_m / tot_c,
                                 "definition": "sum over datasets of trend-removed Spearman / sum of ceilings"}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--practice", action="store_true", help="run the practice (tuning) datasets instead of the test set")
    ap.add_argument("--splits", type=int, default=SPLITS)
    args = ap.parse_args()
    datasets, out_name = (PRACTICE, "results_practice.json") if args.practice else (DATASETS, "results.json")
    t0 = time.time()
    results = []
    for cell_line, region, fname in datasets:
        xyz = load(fname)
        d_all = pair_dist(xyz)
        for split in range(args.splits):
            rng = np.random.default_rng(split)
            idx = rng.permutation(len(xyz))
            a, b = idx[: len(idx) // 2], idx[len(idx) // 2:]
            freq_a = contact_frequency(d_all[a])
            seen_a = np.isfinite(d_all[a]).sum(axis=0)
            truth_b = np.nanmedian(d_all[b], axis=0)
            median_a = np.nanmedian(d_all[a], axis=0)
            model = reconstruct(freq_a, seed=split)
            d_model = np.linalg.norm(model[:, None] - model[None], axis=-1)
            ens = reconstruct_ensemble(freq_a, seen_a, seed=split)
            results.append({
                "cell_line": cell_line, "region": region, "split": split, "cells_total": int(len(xyz)),
                "segments": int(xyz.shape[1]),
                "single_structure_v3_2": {"microscopy": scores(d_model, truth_b),
                                          "contact_map_fit": contact_fit(-d_model, freq_a)},
                "ensemble_v3_3": {"microscopy": scores(ens.median_distance_nm, truth_b),
                                  "contact_map_fit": ens.contact_fit,
                                  "spread_cv_model": ens.spread_cv, "spread_cv_measured_half_B": spread_cv(d_all[b]),
                                  "seconds": ens.seconds, "device": ens.config["device_used"]},
                "ensemble_v3_3_100_trajectories": {"microscopy": scores(ens.sampled_median_distance_nm, truth_b),
                                                   "contact_map_fit": ens.contact_fit},
                "reference_direct_inversion_no_3d": {
                    "microscopy": scores(ensemble.gaussian_median_distance(
                        np.clip(freq_a, 0.5 / np.maximum(seen_a, 1), 1 - 0.5 / np.maximum(seen_a, 1)), CONTACT_NM), truth_b)},
                "baseline_genomic_distance_only": scores(genomic_baseline(median_a), truth_b),
                "ceiling_half_A_vs_half_B": scores(median_a, truth_b),
            })
            r = results[-1]
            ceil = r["ceiling_half_A_vs_half_B"]["spearman_distance_corrected"]
            s32 = r["single_structure_v3_2"]["microscopy"]["spearman_distance_corrected"]
            s33 = r["ensemble_v3_3"]["microscopy"]["spearman_distance_corrected"]
            s100 = r["ensemble_v3_3_100_trajectories"]["microscopy"]["spearman_distance_corrected"]
            print(f"{cell_line} {region} split {split}: trend-removed rho  v3.2 {s32:.3f} ({100 * s32 / ceil:.0f}%) | "
                  f"v3.3 ensemble {s33:.3f} ({100 * s33 / ceil:.0f}%), 100 trajectories {s100:.3f} "
                  f"({100 * s100 / ceil:.0f}%) | ceiling {ceil:.3f} | "
                  f"raw rho v3.3 {r['ensemble_v3_3']['microscopy']['spearman']:.3f} vs baseline "
                  f"{r['baseline_genomic_distance_only']['spearman']:.3f} | contact-map fit {ens.contact_fit:.3f}", flush=True)
    summary = summarise(results)
    (ROOT / out_name).write_text(json.dumps({"summary": summary, "runs": results}, indent=1, default=float))
    for model, o in summary["overall"].items():
        print(f"OVERALL {model}: {o['percent_of_ceiling']:.1f}% of ceiling")
    print(f"done in {time.time() - t0:.0f}s -> {out_name}")


if __name__ == "__main__":
    main()
