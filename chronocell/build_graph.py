"""
Build the chr22 graph from the three raw inputs.

    python -m chronocell.build_graph --fasta chr22.fa --bigwig H3K27ac_signal.bigWig \
        --mcool human_microc.mcool --out graph_chr22.npz

    python -m chronocell.build_graph --synthetic --out graph_synthetic.npz   # no external files

Requires `pyfaidx`, `pyBigWig` and `cooler` for real inputs (imported only when used).
Output keys: gc, epi, valid (N,), ci, cj, cm (upper-triangle contacts), chrom, resolution.
"""

from __future__ import annotations

import argparse

import numpy as np

from . import features, formats, genome, synthetic


def from_files(fasta: str, bigwig: str, mcool: str, chrom: str = genome.CHROM,
               resolution: int = genome.RESOLUTION) -> tuple[np.ndarray, ...]:
    import cooler
    import pyBigWig
    from pyfaidx import Fasta

    fa = Fasta(fasta)
    seq = str(fa[chrom if chrom in fa else chrom.removeprefix("chr")][:]).encode()
    n_bins = int(np.ceil(len(seq) / resolution))
    gc, valid = features.gc_fraction(seq, n_bins, resolution)

    bw = pyBigWig.open(bigwig)
    key = chrom if chrom in bw.chroms() else chrom.removeprefix("chr")
    if key not in bw.chroms():
        raise ValueError(f"{bigwig} has no {chrom} track (chromosomes: {list(bw.chroms())[:5]}...).")
    values = np.asarray(bw.values(key, 0, bw.chroms()[key]), dtype=np.float64)   # NaN = no data
    bw.close()
    epi = features.binned_mean(values, n_bins, resolution)
    epi = np.where(valid, epi, np.nan)

    uri = f"{mcool}::resolutions/{resolution}" if cooler.fileops.is_multires_file(mcool) else mcool
    clr = cooler.Cooler(uri)
    if clr.binsize != resolution:
        raise ValueError(f"{mcool} has {clr.binsize:,} bp bins; requested {resolution:,} bp.")
    cname = chrom if chrom in clr.chromnames else chrom.removeprefix("chr")
    pixels = clr.matrix(balance=False, as_pixels=True).fetch(cname)
    offset = int(clr.offset(cname))
    ci, cj, cm = features.contacts_from_pixels(pixels["bin1_id"].to_numpy() - offset,
                                               pixels["bin2_id"].to_numpy() - offset,
                                               pixels["count"].to_numpy(), n_bins)
    inside = valid[ci] & valid[cj]
    return gc, epi, valid, ci[inside], cj[inside], cm[inside]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fasta")
    ap.add_argument("--bigwig")
    ap.add_argument("--mcool")
    ap.add_argument("--synthetic", action="store_true", help="planted fractal globule instead of files")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.synthetic:
        s = synthetic.build(seed=a.seed)
        arrays = (s.gc, s.epi, s.valid, s.ci, s.cj, s.cm)
        np.savez_compressed(a.out.replace(".npz", "_truth.npz"), coords=s.coords, units_nm=np.array(1.0))
    else:
        if not (a.fasta and a.bigwig and a.mcool):
            ap.error("--fasta, --bigwig and --mcool are required unless --synthetic is given")
        arrays = from_files(a.fasta, a.bigwig, a.mcool)
    formats.write_graph_npz(a.out, *arrays)
    gc, _, valid, ci, *_ = arrays
    print(f"wrote {a.out}: {len(gc):,} bins ({int(valid.sum()):,} assembled), {len(ci):,} contacts")


if __name__ == "__main__":
    main()
