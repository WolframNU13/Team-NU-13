"""
Reference genome model (GRCh38/hg38) for any main human chromosome.

All coordinates are 0-based, half-open [start, end) in base pairs, as served by the UCSC
Genome Browser REST API (chromosomes, cytoBand, gap, centromeres, ncbiRefSeqSelect).
The records are stored in `chronocell/data/hg38.json`.

`Chrom` bundles one chromosome with a bin resolution. Resolution defaults to the finest of the
standard Hi-C resolutions that keeps the chromosome at <= 6,000 beads (chr22 -> 10 kb,
chr1 -> 50 kb); it can also be inferred from the bead count of an uploaded structure.
The module-level names (CHROM, N_BINS, bin_start, ...) describe chr22 at 10 kb and are kept
for backwards compatibility.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

ASSEMBLY = "GRCh38/hg38"
RESOLUTIONS = (5_000, 10_000, 20_000, 25_000, 40_000, 50_000, 100_000, 250_000, 500_000, 1_000_000)
MAX_DEFAULT_BEADS = 6_000
MAIN_CHROMOSOMES = tuple([f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"])


@dataclass(frozen=True)
class Band:
    start: int
    end: int
    name: str
    stain: str


@dataclass(frozen=True)
class Gene:
    name: str
    chrom: str
    start: int
    end: int
    strand: str


@lru_cache(maxsize=1)
def _data() -> dict:
    with open(Path(__file__).with_name("data") / "hg38.json", encoding="utf-8") as fh:
        return json.load(fh)


def gene(name: str) -> Gene:
    g = _data()["genes"][name]
    return Gene(name, g["chrom"], g["start"], g["end"], g["strand"])


def genes_in(chrom_name: str, start: int, end: int) -> list[Gene]:
    """Annotated anchor genes overlapping [start, end) on chrom_name (sorted by position)."""
    out = [gene(n) for n, g in _data()["genes"].items()
           if g["chrom"] == chrom_name and g["start"] < end and g["end"] > start]
    return sorted(out, key=lambda g: g.start)


def chromosome_size(name: str) -> int:
    return int(_data()["chromosomes"][name]["size"])


def default_resolution(name: str) -> int:
    size = chromosome_size(name)
    for r in RESOLUTIONS:
        if math.ceil(size / r) <= MAX_DEFAULT_BEADS:
            return r
    return RESOLUTIONS[-1]


def resolution_for_beads(name: str, n_beads: int) -> int:
    """Resolution whose bin count equals `n_beads` (standard values first, else size / n)."""
    size = chromosome_size(name)
    for r in RESOLUTIONS:
        if math.ceil(size / r) == n_beads:
            return r
    return max(1, math.ceil(size / max(n_beads, 1)))


@dataclass(frozen=True)
class Chrom:
    name: str
    size: int
    resolution: int
    bands: tuple[Band, ...]
    gaps: tuple[tuple[int, int], ...]
    centromere_model: tuple[int, int] | None

    # ---- bins ---------------------------------------------------------------------------
    @property
    def n_bins(self) -> int:
        return math.ceil(self.size / self.resolution)

    @property
    def short(self) -> str:
        return self.name.removeprefix("chr")

    def bin_start(self, i) -> np.ndarray:
        return np.asarray(i, dtype=np.int64) * self.resolution

    def bin_end(self, i) -> np.ndarray:
        return np.minimum((np.asarray(i, dtype=np.int64) + 1) * self.resolution, self.size)

    def bin_lengths(self) -> np.ndarray:
        idx = np.arange(self.n_bins)
        return (self.bin_end(idx) - self.bin_start(idx)).astype(np.int64)

    def interval_to_bins(self, start: int, end: int) -> tuple[int, int]:
        """Half-open bp interval -> half-open bin interval covering it (clipped to the chromosome)."""
        return max(0, start // self.resolution), min(self.n_bins, math.ceil(end / self.resolution))

    def locus(self, i: int) -> str:
        return f"{self.name}:{int(self.bin_start(i)) + 1:,}-{int(self.bin_end(i)):,}"

    def mb(self, i) -> np.ndarray:
        return self.bin_start(i) / 1e6

    # ---- annotation ---------------------------------------------------------------------
    def gap_fraction(self) -> np.ndarray:
        """Fraction of each bin covered by assembly gaps (N), by exact interval overlap."""
        n = self.n_bins
        idx = np.arange(n)
        starts, ends = self.bin_start(idx), self.bin_end(idx)
        covered = np.zeros(n, dtype=np.int64)
        for g0, g1 in self.gaps:
            b0, b1 = self.interval_to_bins(g0, g1)
            if b0 >= b1:
                continue
            sl = slice(b0, b1)
            covered[sl] += np.clip(np.minimum(ends[sl], g1) - np.maximum(starts[sl], g0), 0, None)
        return covered / np.maximum(ends - starts, 1)

    def assembled_mask(self, max_gap: float = 0.5) -> np.ndarray:
        return self.gap_fraction() <= max_gap

    def band_for_bins(self) -> np.ndarray:
        idx = np.arange(self.n_bins)
        mids = (self.bin_start(idx) + self.bin_end(idx)) / 2
        edges = np.array([b.end for b in self.bands])
        return np.searchsorted(edges, mids, side="right").clip(0, len(self.bands) - 1)

    @property
    def acen(self) -> tuple[int, int] | None:
        spans = [(b.start, b.end) for b in self.bands if b.stain == "acen"]
        return (min(s for s, _ in spans), max(e for _, e in spans)) if spans else None

    def with_resolution(self, resolution: int) -> "Chrom":
        return Chrom(self.name, self.size, int(resolution), self.bands, self.gaps, self.centromere_model)


@lru_cache(maxsize=64)
def chrom(name: str = "chr22", resolution: int | None = None) -> Chrom:
    rec = _data()["chromosomes"][name]
    bands = tuple(Band(int(s), int(e), n, st) for s, e, n, st in rec["bands"])
    gaps = tuple((int(s), int(e)) for s, e in rec["gaps"])
    cen = tuple(rec["centromere_model"]) if rec["centromere_model"] else None
    return Chrom(name, int(rec["size"]), int(resolution or default_resolution(name)), bands, gaps, cen)


def chrom_for_beads(name: str, n_beads: int) -> Chrom:
    return chrom(name, resolution_for_beads(name, n_beads))


# ---- chr22 at 10 kb: backwards-compatible module API --------------------------------------
DEFAULT = chrom("chr22", 10_000)
CHROM = DEFAULT.name
CHROM_SIZE = DEFAULT.size
RESOLUTION = DEFAULT.resolution
N_BINS = DEFAULT.n_bins
CYTOBANDS = DEFAULT.bands
GAPS = DEFAULT.gaps
CENTROMERE_MODEL = DEFAULT.centromere_model
ACEN = DEFAULT.acen


def bin_start(i) -> np.ndarray:
    return DEFAULT.bin_start(i)


def bin_end(i) -> np.ndarray:
    return DEFAULT.bin_end(i)


def bin_lengths(n: int = N_BINS) -> np.ndarray:
    return DEFAULT.bin_lengths()[:n]


def interval_to_bins(start: int, end: int) -> tuple[int, int]:
    return DEFAULT.interval_to_bins(start, end)


def gap_fraction(n: int = N_BINS) -> np.ndarray:
    return DEFAULT.gap_fraction()[:n]


def assembled_mask(n: int = N_BINS, max_gap: float = 0.5) -> np.ndarray:
    return DEFAULT.assembled_mask(max_gap)[:n]


def band_for_bins(n: int = N_BINS) -> np.ndarray:
    return DEFAULT.band_for_bins()[:n]


def locus(i: int) -> str:
    return DEFAULT.locus(i)
