"""
Genes on the 3D fold: annotation, accessibility and (optional) expression.

Annotation: RefSeq Select / MANE (one transcript per gene, 19,386 genes on chr1-22, X, Y),
downloaded from the UCSC Genome Browser REST API (hg38 track `ncbiRefSeqSelect`) into
`chronocell/data/genes_hg38.json.gz`. Coordinates are 0-based, half-open.

Accessibility of a gene is read at its promoter bead (the bead holding the transcription start
site), relative to the region being viewed:

    openness  = -z(crowding)   crowding = non-bonded beads within 1.5 b0 (smoothed over +-2 beads)
    activity  = +z(signal)     H3K27ac or the state's own track (skipped when it is a placeholder)
    score     = mean of the available terms

    score >= +0.5  -> "Hyper-accessible (predicted active)"
    score <= -0.5  -> "Buried (predicted silenced)"
    otherwise      -> "Intermediate"

These are predictions from structure (and signal), not measured expression. When an RNA-seq
table is supplied the measured values are shown next to them and their rank correlation with
the score is reported, which is the honest test of the prediction.
"""

from __future__ import annotations

import gzip
import io
import json
import math
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from . import genome, physics

OPEN, MIDDLE, BURIED = "Hyper-accessible (predicted active)", "Intermediate", "Buried (predicted silenced)"
STATUS_ORDER = (OPEN, MIDDLE, BURIED)
THRESHOLD = 0.5

# Curated, well-established gene sets (symbols as in RefSeq). Used only to flag genes of interest.
CANCER_GENES = frozenset("""
ABL1 AKT1 ALK APC AR ARID1A ASXL1 ATM ATR BAP1 BCL2 BCL6 BCR BRAF BRCA1 BRCA2 CALR CCND1 CDH1 CDK4 CDKN2A
CDKN2B CHEK2 CIC CREBBP CTNNB1 DNMT3A EGFR EP300 ERBB2 ESR1 ETV6 EWSR1 EZH2 FGFR1 FGFR2 FGFR3 FLI1 FLT3
FOXA1 GATA3 HRAS IDH1 IDH2 JAK2 KDM6A KEAP1 KIT KMT2A KRAS LZTR1 MAP2K1 MDM2 MET MLH1 MN1 MPL MSH2 MSH6
MYC MYCN NF1 NF2 NFE2L2 NOTCH1 NPM1 NRAS PAX3 PAX7 PDGFB PDGFRA PIK3CA PMS2 POLE PTCH1 PTEN RB1 RET RUNX1
SF3B1 SMAD4 SMARCA4 SMARCB1 SMO SOX2 SRSF2 STK11 SUFU TERT TET2 TP53 TSC1 TSC2 U2AF1 VHL WT1
""".split())
NEURO_GENES = frozenset("""
APOE APP ATP13A2 C9orf72 COMT DNAJC6 FUS GBA1 HTT LRRK2 MAPT PARK7 PINK1 PRKN PSEN1 PSEN2 SNCA SOD1
SYNJ1 TARDBP TBX1 VPS35
""".split())


def category(name: str) -> str:
    if name in CANCER_GENES:
        return "cancer"
    if name in NEURO_GENES:
        return "neuro"
    return ""


@lru_cache(maxsize=1)
def table() -> pd.DataFrame:
    """All annotated genes: name, chrom, start, end, strand, biotype, tss."""
    with gzip.open(Path(__file__).with_name("data") / "genes_hg38.json.gz", "rt", encoding="utf-8") as fh:
        meta = json.load(fh)
    df = pd.DataFrame(meta["genes"], columns=meta["fields"])
    df["tss"] = np.where(df["strand"] == "+", df["start"], df["end"] - 1)
    return df.sort_values(["chrom", "start"]).reset_index(drop=True)


def source_note() -> str:
    return "RefSeq Select / MANE (UCSC hg38 ncbiRefSeqSelect)"


def on_chromosome(chrom_name: str) -> pd.DataFrame:
    df = table()
    return df[df["chrom"] == chrom_name].reset_index(drop=True)


def in_region(chrom_name: str, start: int, end: int) -> pd.DataFrame:
    df = on_chromosome(chrom_name)
    return df[(df["start"] < end) & (df["end"] > start)].reset_index(drop=True)


def find(name: str) -> pd.Series | None:
    """Gene by symbol (case-insensitive exact match, then unique prefix)."""
    q = (name or "").strip().upper()
    if not q:
        return None
    df = table()
    hit = df[df["name"].str.upper() == q]
    if hit.empty:
        hit = df[df["name"].str.upper().str.startswith(q)]
        if len(hit) != 1:
            return None
    return hit.iloc[0]


# ======================================================================================
# Accessibility
# ======================================================================================
def _z(v: np.ndarray, ok: np.ndarray) -> np.ndarray:
    out = np.zeros(len(v))
    if ok.sum() >= 3:
        mu, sd = float(np.mean(v[ok])), float(np.std(v[ok]))
        out[ok] = (v[ok] - mu) / sd if sd > 0 else 0.0
    return out


def bead_scores(coords: np.ndarray, signal: np.ndarray, valid: np.ndarray, b0: float,
                signal_real: bool = True) -> dict[str, np.ndarray]:
    """Per-bead crowding, openness, activity and combined accessibility score for one region."""
    x = np.asarray(coords, dtype=np.float64)
    n = len(x)
    crowd = physics.local_density(x, 1.5 * b0)
    crowd = np.convolve(np.pad(crowd, 2, mode="edge"), np.ones(5) / 5, mode="valid")
    ok = np.asarray(valid, bool)
    openness = -_z(crowd, ok)
    sig = np.asarray(signal, dtype=np.float64)
    sig_ok = ok & np.isfinite(sig)
    use_signal = bool(signal_real and sig_ok.sum() >= 10)
    activity = _z(np.log1p(np.clip(np.nan_to_num(sig), 0, None)), sig_ok) if use_signal else np.zeros(n)
    score = (openness + activity) / 2.0 if use_signal else openness
    score = np.where(ok, score, np.nan)
    return {"crowding": crowd, "openness": openness, "activity": activity, "score": score,
            "uses_signal": np.array(use_signal)}


def status_of(score: float) -> str:
    if not math.isfinite(score):
        return MIDDLE
    return OPEN if score >= THRESHOLD else (BURIED if score <= -THRESHOLD else MIDDLE)


def accessibility_table(coords: np.ndarray, signal: np.ndarray, valid: np.ndarray, b0: float, chrom: genome.Chrom,
                        first_bin: int, signal_real: bool = True, expression: pd.Series | None = None,
                        scores: dict | None = None) -> pd.DataFrame:
    """One row per gene whose promoter lies in the region: locus, bead, scores, status, category."""
    n = len(coords)
    sc = scores or bead_scores(coords, signal, valid, b0, signal_real)
    start_bp = int(chrom.bin_start(first_bin))
    end_bp = int(chrom.bin_end(first_bin + n - 1))
    g = in_region(chrom.name, start_bp, end_bp)
    g = g[(g["tss"] >= start_bp) & (g["tss"] < end_bp)].copy()
    if g.empty:
        return pd.DataFrame(columns=["gene", "locus", "bead", "biotype", "category", "crowding", "signal", "score",
                                     "status", "expression"])
    bead = (g["tss"].to_numpy() // chrom.resolution - first_bin).astype(np.int64)
    bead = np.clip(bead, 0, n - 1)
    sig = np.asarray(signal, dtype=np.float64)
    out = pd.DataFrame({
        "gene": g["name"].to_numpy(),
        "locus": [f"{chrom.name}:{int(s) + 1:,}" for s in g["tss"]],
        "bead": bead,
        "biotype": g["biotype"].to_numpy(),
        "category": [category(v) for v in g["name"]],
        "crowding": np.round(sc["crowding"][bead], 2),
        "signal": np.round(sig[bead], 3),
        "score": np.round(sc["score"][bead], 3),
    })
    out["status"] = [status_of(v) for v in out["score"]]
    if expression is not None and len(expression):
        ex = expression.copy()
        ex.index = ex.index.astype(str).str.upper()
        out["expression"] = out["gene"].str.upper().map(ex)
    else:
        out["expression"] = np.nan
    return out.sort_values("score", ascending=False, na_position="last").reset_index(drop=True)


def summary(tab: pd.DataFrame) -> dict:
    counts = tab["status"].value_counts() if len(tab) else pd.Series(dtype=int)
    top_open = tab[tab["status"] == OPEN]["gene"].head(8).tolist()
    top_buried = tab.sort_values("score")[lambda d: d["status"] == BURIED]["gene"].head(8).tolist() if len(tab) else []
    flagged = tab[tab["category"] != ""][["gene", "category", "status"]].head(20).to_dict("records") if len(tab) else []
    return {"genes_in_view": int(len(tab)), "open": int(counts.get(OPEN, 0)), "intermediate": int(counts.get(MIDDLE, 0)),
            "buried": int(counts.get(BURIED, 0)), "top_open": top_open, "top_buried": top_buried,
            "flagged_disease_genes": flagged}


# ======================================================================================
# Expression (RNA-seq) tables
# ======================================================================================
def parse_expression(data: bytes, name: str = "") -> pd.Series:
    """gene -> value from a CSV/TSV/TXT table. The gene column is the first text column whose values
    look like symbols; the value column is the first numeric column (TPM, FPKM or counts)."""
    text = data.decode("utf-8", "replace")
    sep = "\t" if text.count("\t") >= text.count(",") else ","
    df = pd.read_csv(io.StringIO(text), sep=sep, comment="#")
    if df.shape[1] < 2:
        raise ValueError("An expression table needs a gene column and a numeric value column.")
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    gene_col = next((c for c in df.columns if c not in num_cols), None)     # object or string dtype
    if gene_col is None or not num_cols:
        raise ValueError("Could not find a gene-name column and a numeric value column.")
    s = pd.Series(df[num_cols[0]].to_numpy(float), index=df[gene_col].astype(str).str.strip())
    s = s[~s.index.duplicated(keep="first") & np.isfinite(s.to_numpy())]
    s.name = str(num_cols[0])
    if s.empty:
        raise ValueError("The expression table has no numeric values.")
    return s


def expression_agreement(tab: pd.DataFrame) -> dict:
    """How well the structural prediction agrees with measured expression (Spearman rank)."""
    from .agent import spearman  # local import: agent imports this module's users, not vice versa
    d = tab.dropna(subset=["expression", "score"])
    if len(d) < 20:
        return {"n": int(len(d)), "rho": float("nan")}
    rho = spearman(d["score"].to_numpy(float), np.log1p(np.clip(d["expression"].to_numpy(float), 0, None)))
    return {"n": int(len(d)), "rho": rho}
