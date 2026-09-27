"""Reconstruction benchmark on planted fractal globules (AUDIT.md section 5).

    python -m chronocell.benchmark
"""
import time
import numpy as np
from chronocell import egnn as E, physics as P, synthetic as S

syn = S.build(seed=7)
windows = [(1400, 800), (2600, 800), (3800, 800), (2000, 1600), (3000, 400)]
arms = {
    "random walk, no refine": E.FitConfig(init="random_walk", lr_prefit=0.2, phantom_fraction=0.6, refine_epochs=0),
    "MDS, no refine": E.FitConfig(refine_epochs=0),
    "MDS + coordinate refine": E.FitConfig(refine_epochs=100, refine_with_egnn=False),
    "MDS + EGNN refine": E.FitConfig(refine_epochs=100),
}
print("| window (bins) | beads | arm | RMSD/Rg | distance r | overlaps | L_contact (truth) | time (s) |")
print("|---|---|---|---|---|---|---|---|")
for lo, w in windows:
    hi = lo + w
    m = (syn.ci >= lo) & (syn.ci < hi) & (syn.cj >= lo) & (syn.cj < hi)
    ci, cj, cm = syn.ci[m] - lo, syn.cj[m] - lo, syn.cm[m]
    truth = syn.coords[lo:hi]
    tgt = P.contact_target_distance(cm, P.reference_count(ci, cj, cm))
    floor = P.loss_contact(truth, ci, cj, tgt)
    feats = E.node_features(syn.gc[lo:hi], syn.epi[lo:hi], syn.valid[lo:hi])
    for name, cfg in arms.items():
        t = time.time()
        res = E.fit_structure(w, feats, ci, cj, cm, cfg)
        r, _, _ = P.kabsch_rmsd(truth, res.coords_nm)
        print(f"| {lo}-{hi - 1} | {w} | {name} | {r / P.radius_of_gyration(truth):.3f} | "
              f"{P.distance_correlation(truth, res.coords_nm):.3f} | {P.loss_steric(res.coords_nm).overlaps} | "
              f"{res.history['contact'][-1]:.4f} ({floor:.4f}) | {time.time() - t:.1f} |", flush=True)
