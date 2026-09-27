"""
Write a synthetic demo set of biological-state files for trying the state engine.

    python -m chronocell.demo_states demo_states            # writes demo_states/chr22/...
    CHRONOCELL_COORDINATES=demo_states streamlit run app.py  # point the workstation at it

Everything here is SYNTHETIC (every file name says so): a fractal-globule chr22 as the healthy
control, the same fold with 18-26 Mb swollen and relaxed plus a 2.2x signal gain there as the
"disease" state, and a globally compacted fold with 0.7x signal as the "senescent" state. The
files exercise every format the engine sniffs: (N, 3) .npy coordinates, (N,) .npy tracks and a
wwPDB structure. It is a demonstration of the software, not of biology, so it is never written
into coordinates/ by default.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from . import formats, genome, physics, synthetic

SWOLLEN = (1800, 2600)       # beads (18-26 Mb at 10 kb) swollen in the demo "disease" state


def write_demo(root: str | Path, seed: int = 11, relax_iters: int = 40) -> list[Path]:
    root = Path(root)
    ch = genome.chrom("chr22")
    b0 = physics.bond_length_for(ch.resolution)
    base = synthetic.build(ch, seed=seed, b0=b0)
    healthy = base.coords.copy()

    a, b = SWOLLEN
    disease = healthy.copy()
    c = disease[a:b].mean(axis=0)
    disease[a:b] = c + (disease[a:b] - c) * 1.45
    disease = synthetic.relax(disease, b0, physics.D_MIN_FACTOR * b0, iters=relax_iters)

    senescent = healthy.mean(axis=0) + (healthy - healthy.mean(axis=0)) * 0.82
    senescent = synthetic.relax(senescent, b0, physics.D_MIN_FACTOR * b0, iters=relax_iters)

    epi = np.nan_to_num(base.epi, nan=0.0)
    epi_disease = epi.copy()
    epi_disease[a:b] *= 2.2

    out = {
        "chr22/healthy/synthetic_demo_coords.npy": healthy.astype(np.float32),
        "chr22/healthy/synthetic_demo_h3k27ac.npy": epi.astype(np.float32),
        "chr22/cancer/synthetic_demo_tumour_coords.npy": disease.astype(np.float32),
        "chr22/cancer/synthetic_demo_tumour_h3k27ac.npy": epi_disease.astype(np.float32),
        "chr22/senescent/synthetic_demo_signal.npy": (epi * 0.7).astype(np.float32),
    }
    written = []
    for rel, arr in out.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        np.save(p, arr)
        written.append(p)
    pdb, _ = formats.write_pdb(senescent, 0, base.gc, base.epi, float(np.nanpercentile(base.epi, 99.5)),
                               source="chronocell.demo_states (synthetic)", method="synthetic demo", chrom=ch)
    p = root / "chr22/senescent/synthetic_demo_senescent.pdb"
    p.write_text(pdb)
    written.append(p)
    return written


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", nargs="?", default="demo_states", help="output folder (default: demo_states)")
    ap.add_argument("--seed", type=int, default=11)
    a = ap.parse_args()
    for p in write_demo(a.out, a.seed):
        print(p)


if __name__ == "__main__":
    main()
