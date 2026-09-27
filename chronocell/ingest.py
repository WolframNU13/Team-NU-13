"""
Patient / laboratory data ingestion beyond coordinates: signal tracks, contact maps, expression.

Signal tracks (-> one value per bin of the chromosome at the structure's resolution)
    .npy                1-D array, one value per bead (or per chromosome bin)
    .bedGraph / .bdg    chrom  start  end  value      -> length-weighted mean per bin
    .bed                chrom  start  end  [name [score]] -> score-weighted coverage per bin
    .bigWig / .bw       mean per bin (needs the optional pyBigWig package)

Contact maps (-> sparse cis contacts of one chromosome at the structure's resolution)
    .cool / .mcool      read directly with h5py (cooler schema v2/v3); raw counts; an .mcool
                        uses the matching resolution (or a finer one, aggregated)
    .hic                through the optional hic-straw package (else: convert with hic2cool)
    text (.tsv/.txt/.csv/.pairs)
                        3 numeric columns          bin_i  bin_j  count      (chromosome-local bins)
                        5 columns                  chrom1 pos1 chrom2 pos2 count
                        6-7 columns (BEDPE-like)   chrom1 start1 end1 chrom2 start2 end2 [count]

Expression tables (gene, value) are parsed in chronocell.genes.parse_expression.

`classify_text` tells these text formats apart by their column structure, not by file name.
"""

from __future__ import annotations

import io
import os
import tempfile

import numpy as np
import pandas as pd

from . import genome
from .formats import canonical_contacts

TRACK_TEXT = (".bedgraph", ".bdg", ".bed")
CONTACT_BINARY = (".cool", ".mcool", ".hic")
TEXT_TABLE = (".tsv", ".txt", ".csv", ".pairs", ".gz")


def _norm_chrom(v: str) -> str:
    v = str(v)
    return v if v.startswith("chr") else f"chr{v}"


def _read_table(data: bytes) -> pd.DataFrame:
    text = data.decode("utf-8", "replace")
    lines = [ln for ln in text.splitlines() if ln.strip() and not ln.startswith(("#", "track", "browser"))]
    if not lines:
        raise ValueError("The file has no data lines.")
    sep = "\t" if "\t" in lines[0] else ("," if "," in lines[0] else r"\s+")
    df = pd.read_csv(io.StringIO("\n".join(lines)), sep=sep, header=None, engine="python")
    # drop a header row if the first line is not numeric where numbers are expected
    first = df.iloc[0].astype(str).tolist()
    if any(c.lower() in ("chrom", "chr", "chrom1", "bin1", "bin_i", "gene", "start") for c in first):
        df = df.iloc[1:].reset_index(drop=True)
    return df


def _is_num(col: pd.Series) -> bool:
    return pd.to_numeric(col, errors="coerce").notna().mean() > 0.95


def classify_text(data: bytes) -> str:
    """'track' (BED / bedGraph), 'contacts', 'expression' or 'unknown', from the column structure."""
    try:
        df = _read_table(data[:200_000])
    except Exception:
        return "unknown"
    k = df.shape[1]
    num = [_is_num(df[c]) for c in df.columns]
    chromish = lambda c: df[c].astype(str).str.match(r"^(chr)?([0-9]{1,2}|[XYM])$").mean() > 0.9  # noqa: E731
    if k == 2 and not num[0] and num[1]:
        return "expression"
    if k == 3 and all(num):
        a, b = (pd.to_numeric(df[c], errors="coerce") for c in (0, 1))
        bins = (a % 1 == 0).all() and (b % 1 == 0).all() and (a >= 0).all() and (b >= 0).all()
        return "contacts" if bins else "unknown"            # float triples are coordinates, not bins
    if k >= 3 and chromish(0) and num[1] and num[2]:           # BED3+ / bedGraph / contact pairs
        if k >= 5 and chromish(3 if k >= 6 else 2):
            return "contacts"
        return "track"
    if k >= 2 and not num[0] and any(num[1:]):
        return "expression"
    return "unknown"


# ======================================================================================
# Tracks
# ======================================================================================
def _bin_intervals(chrom: genome.Chrom, starts: np.ndarray, ends: np.ndarray, values: np.ndarray, mean: bool) -> np.ndarray:
    """Length-weighted sum (or mean) of interval values per bin of `chrom`."""
    n, res = chrom.n_bins, chrom.resolution
    acc = np.zeros(n)
    cover = np.zeros(n)
    ok = np.isfinite(starts) & np.isfinite(ends) & np.isfinite(values)
    starts = np.clip(starts[ok].astype(np.int64), 0, chrom.size)
    ends = np.clip(ends[ok].astype(np.int64), 0, chrom.size)
    values = np.asarray(values, float)[ok]
    keep = ends > starts
    starts, ends, values = starts[keep], ends[keep], values[keep]
    first, last = starts // res, np.minimum((ends - 1) // res, n - 1)
    single = first == last                      # the common case for fine-grained tracks: vectorised
    length = (ends - starts).astype(float)
    np.add.at(acc, first[single], values[single] * length[single])
    np.add.at(cover, first[single], length[single])
    for s, e, v, b0, b1 in zip(starts[~single], ends[~single], values[~single], first[~single], last[~single]):
        for b in range(int(b0), int(b1) + 1):
            ov = min(e, (b + 1) * res) - max(s, b * res)
            if ov > 0:
                acc[b] += v * ov
                cover[b] += ov
    if mean:
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(cover > 0, acc / cover, np.nan)
    return acc / res


def read_track(data: bytes, name: str, chrom: genome.Chrom) -> tuple[np.ndarray, str]:
    """A per-bin (or per-bead for .npy) signal array and a description."""
    ext = os.path.splitext(name.lower())[1]
    if ext == ".npy":
        from .states import load_track
        a = load_track(data)
        return a, f"{len(a):,} values"
    if ext in (".bw", ".bigwig"):
        try:
            import pyBigWig  # optional
        except ImportError as exc:
            raise ValueError("Reading bigWig needs the optional package pyBigWig (pip install pyBigWig), or convert "
                             "it to bedGraph (bigWigToBedGraph).") from exc
        with tempfile.NamedTemporaryFile(suffix=".bw", delete=False) as fh:
            fh.write(data)
            path = fh.name
        try:
            bw = pyBigWig.open(path)
            names = bw.chroms()
            key = chrom.name if chrom.name in names else chrom.name.removeprefix("chr")
            if key not in names:
                raise ValueError(f"{chrom.name} not in bigWig ({', '.join(list(names)[:6])}...).")
            vals = np.array(bw.stats(key, 0, chrom.size, type="mean", nBins=chrom.n_bins), dtype=float)
            bw.close()
        finally:
            os.unlink(path)
        return vals, f"bigWig means over {chrom.n_bins:,} bins"
    df = _read_table(data)
    if df.shape[1] < 3:
        raise ValueError("BED/bedGraph needs at least chrom, start, end columns.")
    chroms = df[0].map(_norm_chrom)
    sub = df[chroms == chrom.name]
    if sub.empty:
        raise ValueError(f"No intervals on {chrom.name} (found {', '.join(sorted(chroms.unique())[:6])}).")
    starts = pd.to_numeric(sub[1], errors="coerce").to_numpy()
    ends = pd.to_numeric(sub[2], errors="coerce").to_numpy()
    if ext in (".bedgraph", ".bdg") or (df.shape[1] == 4 and _is_num(df[3])):
        vals = pd.to_numeric(sub[3], errors="coerce").fillna(0).to_numpy(float)
        return _bin_intervals(chrom, starts, ends, vals, mean=True), f"bedGraph · {len(sub):,} intervals on {chrom.name}"
    score = pd.to_numeric(sub[4], errors="coerce").fillna(1).to_numpy(float) if df.shape[1] >= 5 else np.ones(len(sub))
    return _bin_intervals(chrom, starts, ends, score, mean=False), f"BED coverage · {len(sub):,} features on {chrom.name}"


# ======================================================================================
# Contacts
# ======================================================================================
def _cool_group(f, resolution: int):
    if "resolutions" in f:
        avail = sorted(int(r) for r in f["resolutions"].keys())
        if resolution in avail:
            return f["resolutions"][str(resolution)], resolution
        finer = [r for r in avail if resolution % r == 0]
        if not finer:
            raise ValueError(f"The .mcool has resolutions {avail}; none divides {resolution:,} bp.")
        return f["resolutions"][str(max(finer))], max(finer)
    binsize = int(f.attrs.get("bin-size", 0)) or int(np.diff(f["bins"]["start"][:2])[0])
    if resolution % binsize:
        raise ValueError(f"The .cool is at {binsize:,} bp, which does not divide {resolution:,} bp.")
    return f, binsize


def read_cool(data: bytes, chrom: genome.Chrom) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    try:
        import h5py
    except ImportError as exc:
        raise ValueError("Reading .cool/.mcool needs h5py (pip install h5py).") from exc
    with h5py.File(io.BytesIO(data), "r") as f:
        g, binsize = _cool_group(f, chrom.resolution)
        names = [v.decode() if isinstance(v, bytes) else str(v) for v in g["chroms"]["name"][:]]
        norm = [_norm_chrom(v) for v in names]
        if chrom.name not in norm:
            raise ValueError(f"{chrom.name} not in the cooler ({', '.join(names[:6])}...).")
        cid = norm.index(chrom.name)
        offsets = g["indexes"]["chrom_offset"][:]
        lo, hi = int(offsets[cid]), int(offsets[cid + 1])
        bin1_offset = g["indexes"]["bin1_offset"][:]
        p0, p1 = int(bin1_offset[lo]), int(bin1_offset[hi])
        b1 = g["pixels"]["bin1_id"][p0:p1].astype(np.int64)
        b2 = g["pixels"]["bin2_id"][p0:p1].astype(np.int64)
        cnt = g["pixels"]["count"][p0:p1].astype(np.float64)
    cis = (b2 >= lo) & (b2 < hi)
    i, j, c = b1[cis] - lo, b2[cis] - lo, cnt[cis]
    factor = chrom.resolution // binsize
    notes = [f"cooler at {binsize:,} bp" + (f", aggregated x{factor} to {chrom.resolution:,} bp" if factor > 1 else "")]
    if factor > 1:
        i, j = i // factor, j // factor
    ci, cj, cm, cn = canonical_contacts(i, j, c, chrom.n_bins) if factor == 1 else _aggregate(i, j, c, chrom.n_bins)
    return ci, cj, cm, notes + cn


def _aggregate(i: np.ndarray, j: np.ndarray, c: np.ndarray, n: int):
    a, b = np.minimum(i, j), np.maximum(i, j)
    keep = (a != b) & (a >= 0) & (b < n) & (c > 0)
    key = a[keep] * n + b[keep]
    uniq, inv = np.unique(key, return_inverse=True)
    sums = np.bincount(inv, weights=c[keep])
    return (uniq // n).astype(np.int64), (uniq % n).astype(np.int64), sums, []


def read_hic(data: bytes, chrom: genome.Chrom):
    try:
        import hicstraw  # optional
    except ImportError as exc:
        raise ValueError(".hic needs the optional package hic-straw (pip install hic-straw); or convert it with "
                         "hic2cool and upload the .cool / .mcool.") from exc
    with tempfile.NamedTemporaryFile(suffix=".hic", delete=False) as fh:
        fh.write(data)
        path = fh.name
    try:
        key = chrom.name.removeprefix("chr")
        recs = hicstraw.straw("observed", "NONE", path, key, key, "BP", chrom.resolution)
    finally:
        os.unlink(path)
    i = np.array([r.binX for r in recs], np.int64) // chrom.resolution
    j = np.array([r.binY for r in recs], np.int64) // chrom.resolution
    c = np.array([r.counts for r in recs], float)
    ci, cj, cm, notes = canonical_contacts(i, j, c, chrom.n_bins)
    return ci, cj, cm, ["hic observed counts"] + notes


def read_contact_table(data: bytes, chrom: genome.Chrom):
    df = _read_table(data)
    k = df.shape[1]
    res = chrom.resolution
    if k == 3:
        i, j, c = (pd.to_numeric(df[x], errors="coerce") for x in (0, 1, 2))
        ok = i.notna() & j.notna() & c.notna()
        i, j, c = i[ok].to_numpy(np.int64), j[ok].to_numpy(np.int64), c[ok].to_numpy(float)
        note = "bin_i bin_j count (chromosome-local bins)"
    else:
        if k == 5:
            c1, p1, c2, p2, cn = 0, 1, 2, 3, 4
        elif k >= 6:
            c1, p1, c2, p2, cn = 0, 1, 3, 4, (6 if k >= 7 else None)
        else:
            raise ValueError(f"Unrecognised contact table with {k} columns.")
        same = (df[c1].map(_norm_chrom) == chrom.name) & (df[c2].map(_norm_chrom) == chrom.name)
        sub = df[same]
        if sub.empty:
            raise ValueError(f"No {chrom.name} cis contacts in the table.")
        i = (pd.to_numeric(sub[p1], errors="coerce") // res).to_numpy(np.int64)
        j = (pd.to_numeric(sub[p2], errors="coerce") // res).to_numpy(np.int64)
        c = pd.to_numeric(sub[cn], errors="coerce").fillna(1).to_numpy(float) if cn is not None else np.ones(len(sub))
        note = f"positions binned at {res:,} bp"
    a, b, m, _ = _aggregate(i, j, c, chrom.n_bins)
    return a, b, m, [note]


def is_contact_file(name: str, data: bytes | None = None) -> bool:
    ext = os.path.splitext(name.lower())[1]
    if ext in CONTACT_BINARY:
        return True
    if ext in TEXT_TABLE and data is not None:
        return classify_text(data) == "contacts"
    return False


def read_contacts(data: bytes, name: str, chrom: genome.Chrom) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Cis contacts of `chrom` at chrom.resolution: (ci < cj, counts, notes); indices are chromosome bins."""
    ext = os.path.splitext(name.lower())[1]
    if ext in (".cool", ".mcool"):
        return read_cool(data, chrom)
    if ext == ".hic":
        return read_hic(data, chrom)
    return read_contact_table(data, chrom)
