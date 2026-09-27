"""
Reconstruct 3D coordinates from a graph .npz (or legacy graph_data.pt).

    python -m chronocell.train --graph graph_chr22.npz --out predicted_coords.npz \
        [--start 2600 --end 3400] [--refine-epochs 100] [--truth graph_chr22_truth.npz]

Writes coords (nm), the per-epoch loss history (CSV) and a wwPDB file next to --out.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from . import egnn, formats, genome, physics


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--graph", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--prefit-epochs", type=int, default=800)
    ap.add_argument("--refine-epochs", type=int, default=100)
    ap.add_argument("--b0", type=float, default=None, help="default: b0 for the graph's resolution")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--truth", help="npz with ground-truth coords (synthetic runs) for RMSD")
    ap.add_argument("--trust-pickle", action="store_true", help="allow unpickling PyG .pt graphs")
    a = ap.parse_args()

    with open(a.graph, "rb") as fh:
        raw = fh.read()
    g = formats.read_graph(raw, a.graph, trusted=a.trust_pickle)
    chrom = genome.DEFAULT
    if a.graph.endswith(".npz"):
        z = np.load(a.graph, allow_pickle=False)
        if "chrom" in z and "resolution" in z:
            chrom = genome.chrom(str(z["chrom"]), int(z["resolution"]))
    lo, hi = a.start, a.end or len(g.gc)
    m = (g.ci >= lo) & (g.ci < hi) & (g.cj >= lo) & (g.cj < hi)
    feats = egnn.node_features(g.gc[lo:hi], g.epi[lo:hi], g.valid[lo:hi])
    cfg = egnn.FitConfig(prefit_epochs=a.prefit_epochs, refine_epochs=a.refine_epochs, seed=a.seed)

    def progress(stage: str, ep: int, total: int, row: dict) -> None:
        if ep == total or ep % 100 == 0:
            print(f"{stage:6s} {ep:4d}/{total}  L_contact {row['contact']:.4f}  L_smooth {row['smooth']:.4f}"
                  f"  L_steric {row['steric']:.5f}")

    b0 = a.b0 or physics.bond_length_for(chrom.resolution)
    res = egnn.fit_structure(hi - lo, feats, g.ci[m] - lo, g.cj[m] - lo, g.cm[m], cfg, b0=b0, progress=progress)
    stem = a.out.removesuffix(".npz")
    np.savez_compressed(a.out, coords=res.coords_nm, units_nm=np.array(1.0), start_bin=np.array(lo))
    pd.DataFrame(res.history).to_csv(stem + "_history.csv", index=False)
    epi_ref = float(np.nanpercentile(g.epi, 99.5))
    pdb, _ = formats.write_pdb(res.coords_nm, lo, g.gc, g.epi, epi_ref, source=a.graph, method="contact embedding + EGNN",
                               chrom=chrom)
    formats.write_bundle(stem + "_bundle.npz", formats.StructureBundle(
        chrom=chrom.name, resolution=chrom.resolution, frames=res.coords_nm[None], times=np.zeros(1),
        labels=["fit"], condition="fit", source=a.graph, start_bin=lo))
    with open(stem + ".pdb", "w") as fh:
        fh.write(pdb)
    fit = physics.distance_scaling(res.coords_nm)
    print(f"done in {res.seconds:.1f}s | Rg {physics.radius_of_gyration(res.coords_nm):.1f} nm | nu {fit.nu:.3f} ({fit.regime})")
    if a.truth:
        truth = np.load(a.truth)["coords"][lo:hi]
        rmsd, _, refl = physics.kabsch_rmsd(truth, res.coords_nm)
        print(f"RMSD {rmsd:.1f} nm = {rmsd / physics.radius_of_gyration(truth):.3f} Rg (mirror: {refl})")


if __name__ == "__main__":
    main()
