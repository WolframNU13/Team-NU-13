"""
Reference genome constants for human chr22 (GRCh38/hg38) at 10 kb resolution.

All coordinates are 0-based, half-open [start, end) in base pairs, exactly as served by
the UCSC Genome Browser REST API (tracks `cytoBand`, `gap`, `centromeres`, hg38, chr22).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

ASSEMBLY = "GRCh38/hg38"
CHROM = "chr22"
CHROM_SIZE = 50_818_468          # bp (UCSC chromInfo)
RESOLUTION = 10_000              # bp per bin / bead
N_BINS = math.ceil(CHROM_SIZE / RESOLUTION)   # 5,082; the last bin spans 8,468 bp


@dataclass(frozen=True)
class Band:
    start: int
    end: int
    name: str
    stain: str


# UCSC hg38 cytoBand, chr22
CYTOBANDS: tuple[Band, ...] = (
    Band(0, 4_300_000, "p13", "gvar"),
    Band(4_300_000, 9_400_000, "p12", "stalk"),
    Band(9_400_000, 13_700_000, "p11.2", "gvar"),
    Band(13_700_000, 15_000_000, "p11.1", "acen"),
    Band(15_000_000, 17_400_000, "q11.1", "acen"),
    Band(17_400_000, 21_700_000, "q11.21", "gneg"),
    Band(21_700_000, 23_100_000, "q11.22", "gpos25"),
    Band(23_100_000, 25_500_000, "q11.23", "gneg"),
    Band(25_500_000, 29_200_000, "q12.1", "gpos50"),
    Band(29_200_000, 31_800_000, "q12.2", "gneg"),
    Band(31_800_000, 37_200_000, "q12.3", "gpos50"),
    Band(37_200_000, 40_600_000, "q13.1", "gneg"),
    Band(40_600_000, 43_800_000, "q13.2", "gpos50"),
    Band(43_800_000, 48_100_000, "q13.31", "gneg"),
    Band(48_100_000, 49_100_000, "q13.32", "gpos50"),
    Band(49_100_000, 50_818_468, "q13.33", "gneg"),
)

# UCSC hg38 `gap` track, chr22 (all 45 N runs).
GAPS: tuple[tuple[int, int], ...] = (
    (0, 10_000), (10_000, 10_510_000),
    (10_784_643, 10_834_643), (10_874_572, 10_924_572), (10_966_724, 11_016_724),
    (11_068_987, 11_118_987), (11_160_921, 11_210_921), (11_378_056, 11_428_056),
    (11_497_337, 11_547_337), (11_631_288, 11_681_288), (11_724_629, 11_774_629),
    (11_977_555, 12_027_555), (12_225_588, 12_275_588), (12_438_690, 12_488_690),
    (12_641_730, 12_691_730), (12_726_204, 12_776_204), (12_818_137, 12_868_137),
    (12_904_788, 12_954_788), (12_977_325, 12_977_425), (12_986_171, 12_994_027),
    (13_011_653, 13_014_130), (13_021_322, 13_021_422), (13_109_444, 13_109_544),
    (13_163_677, 13_163_777), (13_227_312, 13_227_412), (13_248_082, 13_248_182),
    (13_254_852, 13_254_952), (13_258_197, 13_258_297), (13_280_858, 13_280_958),
    (13_285_143, 13_285_243), (14_419_454, 14_419_554), (14_419_894, 14_419_994),
    (14_420_334, 14_420_434), (14_421_632, 14_421_732), (15_054_318, 15_154_318),
    (16_279_672, 16_302_843), (16_304_296, 16_305_427), (16_307_048, 16_307_605),
    (16_310_302, 16_310_402), (16_313_516, 16_314_010), (18_239_129, 18_339_129),
    (18_433_513, 18_483_513), (18_659_564, 18_709_564), (49_973_865, 49_975_365),
    (50_808_468, 50_818_468),
)

# UCSC hg38 `centromeres` track: modelled alpha-satellite sequence (not N).
CENTROMERE_MODEL = (12_954_788, 15_054_318)
ACEN = (13_700_000, 17_400_000)  # p11.1 + q11.1 cytogenetic 'acen' bands


def bin_start(i: np.ndarray | int) -> np.ndarray:
    return np.asarray(i, dtype=np.int64) * RESOLUTION


def bin_end(i: np.ndarray | int) -> np.ndarray:
    return np.minimum((np.asarray(i, dtype=np.int64) + 1) * RESOLUTION, CHROM_SIZE)


def bin_lengths(n: int = N_BINS) -> np.ndarray:
    """Base pairs per bin; every bin is 10 kb except the final partial bin (8,468 bp)."""
    idx = np.arange(n)
    return (bin_end(idx) - bin_start(idx)).astype(np.int64)


def interval_to_bins(start: int, end: int) -> tuple[int, int]:
    """Half-open bp interval -> half-open bin interval covering it."""
    return start // RESOLUTION, math.ceil(end / RESOLUTION)


def gap_fraction(n: int = N_BINS) -> np.ndarray:
    """Fraction of each bin covered by assembly gaps (N), computed exactly by interval overlap."""
    starts = bin_start(np.arange(n)).astype(np.int64)
    ends = bin_end(np.arange(n)).astype(np.int64)
    covered = np.zeros(n, dtype=np.int64)
    for g0, g1 in GAPS:
        b0, b1 = interval_to_bins(g0, g1)
        b1 = min(b1, n)
        if b0 >= b1:
            continue
        sl = slice(b0, b1)
        covered[sl] += np.clip(np.minimum(ends[sl], g1) - np.maximum(starts[sl], g0), 0, None)
    return covered / (ends - starts)


def assembled_mask(n: int = N_BINS, max_gap: float = 0.5) -> np.ndarray:
    """True for bins with sequence; bins more than `max_gap` N are unassembled (no data)."""
    return gap_fraction(n) <= max_gap


def band_for_bins(n: int = N_BINS) -> np.ndarray:
    """Index into CYTOBANDS for the band containing each bin's midpoint."""
    mids = (bin_start(np.arange(n)) + bin_end(np.arange(n))) / 2
    edges = np.array([b.end for b in CYTOBANDS])
    return np.searchsorted(edges, mids, side="right").clip(0, len(CYTOBANDS) - 1)


def locus(i: int) -> str:
    """UCSC-style 1-based locus string for bin i."""
    return f"{CHROM}:{int(bin_start(i)) + 1:,}-{int(bin_end(i)):,}"
