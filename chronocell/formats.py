"""
Structure and graph I/O: wwPDB v3.3 fixed-column PDB, XYZ, JSON report, and loaders for the
pipeline's own artefacts (predicted_coords.pt / graph_data.pt / .npz / .npy / .csv).

PDB unit policy. The PDB coordinate field is %8.3f in Angstrom, i.e. -999.999 .. 9999.999.
A 10 kb chromatin model spans ~2 um = 20,000 A, so writing Angstrom would overflow the columns.
Coordinates are therefore written in nanometres, translated into the positive octant (which
restores the full 0 .. 9999.999 range), and scaled by a power of ten only if still too large.
The unit, scale and origin offset are recorded in REMARK 250 so the file is exactly invertible.
"""

from __future__ import annotations

import datetime as dt
import io
import json
import os
import re
from dataclasses import dataclass, field

import numpy as np

from . import genome

PDB_WIDTH = 80
RES_NAME = "GNN"          # residue name agreed in the team specification
ATOM_NAME = " CA "        # one bead = one Calpha-like pseudo-atom; lets viewers draw a trace
CHAIN = "A"
ELEMENT = " C"


def segment_id(chrom_name: str) -> str:
    """4-character segID (columns 73-76) for a chromosome, e.g. chr22 -> 'CH22', chrX -> 'CHX '."""
    return ("CH" + chrom_name.removeprefix("chr"))[:4]


def _pad(line: str) -> str:
    if len(line) > PDB_WIDTH:
        raise ValueError(f"PDB record exceeds 80 columns: {line!r}")
    return line.ljust(PDB_WIDTH)


def atom_record(serial: int, res_seq: int, x: float, y: float, z: float, occ: float, bfac: float,
                segment: str = "CH22") -> str:
    """One ATOM record with every field in its wwPDB column (1-based, inclusive):

    1-6 'ATOM  ' | 7-11 serial | 13-16 name | 17 altLoc | 18-20 resName | 22 chainID
    23-26 resSeq | 27 iCode | 31-38 x | 39-46 y | 47-54 z (%8.3f) | 55-60 occupancy
    61-66 tempFactor (%6.2f) | 73-76 segID | 77-78 element | 79-80 charge
    """
    if not 0 < serial <= 99_999:
        raise ValueError(f"Atom serial {serial} does not fit columns 7-11.")
    if not -999 <= res_seq <= 9_999:
        raise ValueError(f"Residue number {res_seq} does not fit columns 23-26.")
    for v in (x, y, z):
        if not -999.999 <= v <= 9999.999:
            raise ValueError(f"Coordinate {v} does not fit an %8.3f field.")
    line = (f"ATOM  {serial:5d} {ATOM_NAME}{' '}{RES_NAME} {CHAIN}{res_seq:4d}{' '}   "
            f"{x:8.3f}{y:8.3f}{z:8.3f}{occ:6.2f}{bfac:6.2f}      {segment:<4s}{ELEMENT:>2s}  ")
    return _pad(line)


@dataclass(frozen=True)
class PdbFrame:
    """How nm coordinates were mapped into the file: file = (nm + offset_nm) / unit_nm."""
    unit_nm: float
    offset_nm: np.ndarray


def pdb_frame(coords_nm: np.ndarray, margin_nm: float = 1.0) -> PdbFrame:
    offset = -coords_nm.min(axis=0) + margin_nm
    span = float((coords_nm + offset).max())
    unit = 1.0
    while span / unit > 9999.999:
        unit *= 10.0
    return PdbFrame(unit, offset)


def epi_to_bfactor(epi: np.ndarray, epi_ref: float) -> np.ndarray:
    """Fixed, dataset-wide map of H3K27ac signal to 0..99.99 (log1p-compressed, clipped).

    Using a chromosome-wide reference (not the export window's maximum) keeps B-factors
    comparable between exports of different regions.
    """
    e = np.nan_to_num(np.clip(epi, 0, None), nan=0.0)
    ref = np.log1p(max(epi_ref, 1e-9))
    return np.clip(99.99 * np.log1p(e) / ref, 0.0, 99.99)


def write_pdb(coords_nm: np.ndarray, start_bin: int, gc: np.ndarray, epi: np.ndarray, epi_ref: float,
              source: str, method: str, chrom: genome.Chrom | None = None, bead_bins: np.ndarray | None = None,
              segments: np.ndarray | None = None) -> tuple[str, PdbFrame]:
    """PDB text: HEADER, TITLE, REMARK 2/250, CRYST1, ATOM, TER, CONECT (each bead lists its sequence
    neighbours i-1 and i+1 only), END.

    By default the beads are bins [start_bin, start_bin + N) of `chrom`. Derived chromosomes
    (deletions, duplications, fusions) pass `bead_bins` (bin of each bead) and `segments`
    (segID per bead, e.g. 'CH22' for native beads and 'CH9 ' for a translocation partner);
    tracks are only read for beads whose segID is the chromosome's own.
    """
    chrom = chrom or genome.DEFAULT
    coords_nm = np.asarray(coords_nm, dtype=np.float64)
    n = len(coords_nm)
    if n > 99_998:
        raise ValueError("More than 99,998 beads cannot be numbered in PDB columns 7-11.")
    own = segment_id(chrom.name)
    idx = np.arange(start_bin, start_bin + n) if bead_bins is None else np.asarray(bead_bins, dtype=np.int64)
    seg = np.full(n, own) if segments is None else np.asarray(segments)
    native = (seg == own) & (idx >= 0) & (idx < len(gc))
    occ = np.zeros(n)
    bfac = np.zeros(n)
    occ[native] = np.nan_to_num(np.clip(gc[idx[native]], 0.0, 1.0), nan=0.0)
    bfac[native] = epi_to_bfactor(epi[idx[native]], epi_ref)
    if idx.max() + 1 <= 9_999:
        res_seq, res_note = idx + 1, "RESSEQ = BIN + 1"
    elif n <= 9_999:
        res_seq, res_note = np.arange(1, n + 1), "RESSEQ = BEAD ORDER (BIN > 9999)"
    else:
        raise ValueError(f"{n:,} beads exceed PDB residue numbering; export a window or a coarser resolution.")
    frame = pdb_frame(coords_nm)
    q = (coords_nm + frame.offset_nm) / frame.unit_nm
    lo_bin, hi_bin = int(idx[native].min()) if native.any() else 0, int(idx[native].max()) if native.any() else 0
    region = f"{chrom.name.upper()}:{int(chrom.bin_start(lo_bin)) + 1}-{int(chrom.bin_end(hi_bin))}"
    kb = f"{chrom.resolution / 1000:g}"
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%d-%b-%y").upper()

    def remark250(key: str, value: str) -> str:
        return _pad(f"REMARK 250  {key:<31s}: {value}"[:PDB_WIDTH])

    lines = [
        _pad(f"HEADER    {'CHROMATIN STRUCTURE':<40s}{stamp:>9s}   CC5D"),
        _pad(f"TITLE     CHRONOCELL-5D MODEL OF HUMAN {chrom.name.upper()} (GRCH38) AT {kb} KB"),
        _pad("REMARK   2"),
        _pad("REMARK   2 RESOLUTION. NOT APPLICABLE."),
        _pad("REMARK 250"),
        _pad("REMARK 250 EXPERIMENTAL DETAILS"),
        remark250("EXPERIMENT TYPE", "COMPUTATIONAL MODEL"),
        remark250("METHOD", method.upper()[:36]),
        remark250("SOURCE", source.upper()[:36]),
        remark250("REGION", region),
        remark250("BEAD", f"ONE CA PER {kb} KB BIN, {res_note}"[:46]),
        remark250("COORDINATE UNIT", f"{frame.unit_nm:g} NM PER FILE UNIT (NOT A)"),
        remark250("ORIGIN OFFSET (NM)", " ".join(f"{v:.3f}" for v in frame.offset_nm)),
        remark250("OCCUPANCY", "F_GC; 0.00 = UNASSEMBLED OR PARTNER BIN"),
        remark250("B-FACTOR", f"99.99*LN(1+F_EPI)/LN(1+{epi_ref:.3f})"),
        _pad(f"CRYST1{1.0:9.3f}{1.0:9.3f}{1.0:9.3f}{90.0:7.2f}{90.0:7.2f}{90.0:7.2f} {'P 1':<11s}{1:4d}"),
    ]
    for k in range(n):
        lines.append(atom_record(k + 1, int(res_seq[k]), *q[k], float(occ[k]), float(bfac[k]), str(seg[k])))
    lines.append(_pad(f"TER   {n + 1:5d}      {RES_NAME} {CHAIN}{int(res_seq[-1]):4d}"))
    for k in range(n):
        partners = [s for s in (k, k + 2) if 1 <= s <= n]        # serials of beads i-1 and i+1
        lines.append(_pad("CONECT" + f"{k + 1:5d}" + "".join(f"{s:5d}" for s in partners)))
    lines.append(_pad("END"))
    return "\n".join(lines) + "\n", frame


def write_pdb_trajectory(frames_nm: np.ndarray, res_seq: np.ndarray, segments: np.ndarray, chrom: genome.Chrom,
                         title: str, method: str) -> str:
    """Multi-model PDB (MODEL / ENDMDL per frame, one shared unit/offset), CONECT once after the last model.

    Molecular viewers play the models as a movie, which is the standard way to ship a trajectory
    in PDB format. Residue numbers that exceed 4 digits fall back to bead order.
    """
    frames_nm = np.asarray(frames_nm, dtype=np.float64)
    t_n, n = frames_nm.shape[:2]
    if n > 99_998:
        raise ValueError("More than 99,998 beads cannot be numbered in PDB columns 7-11.")
    res = np.asarray(res_seq, dtype=np.int64)
    if res.max() > 9_999:
        res = np.arange(1, n + 1)
        if n > 9_999:
            raise ValueError(f"{n:,} beads exceed PDB residue numbering; export a window or a coarser resolution.")
    frame = pdb_frame(frames_nm.reshape(-1, 3))
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%d-%b-%y").upper()
    lines = [
        _pad(f"HEADER    {'CHROMATIN STRUCTURE':<40s}{stamp:>9s}   CC5D"),
        _pad(f"TITLE     CHRONOCELL-5D {chrom.name.upper()} 4D: {title.upper()}"[:PDB_WIDTH]),
        _pad(f"NUMMDL    {t_n:4d}"),
        _pad("REMARK   2"),
        _pad("REMARK   2 RESOLUTION. NOT APPLICABLE."),
        _pad("REMARK 250"),
        _pad(f"REMARK 250  {'METHOD':<31s}: {method.upper()}"[:PDB_WIDTH]),
        _pad(f"REMARK 250  {'COORDINATE UNIT':<31s}: {frame.unit_nm:g} NM PER FILE UNIT (NOT A)"),
        _pad(f"REMARK 250  {'ORIGIN OFFSET (NM)':<31s}: " + " ".join(f"{v:.3f}" for v in frame.offset_nm)),
        _pad(f"CRYST1{1.0:9.3f}{1.0:9.3f}{1.0:9.3f}{90.0:7.2f}{90.0:7.2f}{90.0:7.2f} {'P 1':<11s}{1:4d}"),
    ]
    for t in range(t_n):
        q = (frames_nm[t] + frame.offset_nm) / frame.unit_nm
        lines.append(_pad(f"MODEL     {t + 1:4d}"))
        for k in range(n):
            lines.append(atom_record(k + 1, int(res[k]), *q[k], 1.0, 0.0, str(segments[k])))
        lines.append(_pad(f"TER   {n + 1:5d}      {RES_NAME} {CHAIN}{int(res[-1]):4d}"))
        lines.append(_pad("ENDMDL"))
    for k in range(n):
        partners = [s for s in (k, k + 2) if 1 <= s <= n]
        lines.append(_pad("CONECT" + f"{k + 1:5d}" + "".join(f"{s:5d}" for s in partners)))
    lines.append(_pad("END"))
    return "\n".join(lines) + "\n"


@dataclass
class PdbCheck:
    n_atoms: int = 0
    n_conect: int = 0
    n_models: int = 0
    issues: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.issues


def validate_pdb(text: str, max_issues: int = 25) -> PdbCheck:
    """Column-level conformance check of ATOM / CONECT records against wwPDB v3.3."""
    chk = PdbCheck()

    def issue(msg: str) -> None:
        if len(chk.issues) < max_issues:
            chk.issues.append(msg)

    serials: list[int] = []
    model_serials: list[int] = []
    bonds: dict[int, set[int]] = {}
    lines = text.splitlines()
    for ln, line in enumerate(lines, 1):
        if len(line) > PDB_WIDTH:
            issue(f"line {ln}: {len(line)} columns (> 80)")
        rec = line[:6]
        if rec == "ATOM  ":
            chk.n_atoms += 1
            try:
                serial = int(line[6:11])
                int(line[22:26])
                [float(line[a:b]) for a, b in ((30, 38), (38, 46), (46, 54), (54, 60), (60, 66))]
            except ValueError:
                issue(f"line {ln}: numeric field not parseable at its fixed columns")
                continue
            for pos, name in ((34, "x"), (42, "y"), (50, "z"), (57, "occupancy"), (63, "B-factor")):
                if line[pos] != ".":
                    issue(f"line {ln}: {name} decimal point not in column {pos + 1}")
            if line[11] != " " or line[26] != " " or line[20] != " ":
                issue(f"line {ln}: separator columns 12/21/27 not blank")
            if not line[12:16].strip() or not line[17:20].strip() or not line[21].strip():
                issue(f"line {ln}: atom name / residue name / chain ID empty")
            if line[76:78].strip() == "" or line[76:78] != line[76:78].rjust(2):
                issue(f"line {ln}: element symbol not right-justified in columns 77-78")
            if model_serials and serial != model_serials[-1] + 1:
                issue(f"line {ln}: atom serial {serial} not sequential")
            model_serials.append(serial)
            serials.append(serial)
        elif rec == "MODEL ":
            chk.n_models += 1
            model_serials = []                        # serial numbering restarts in every model
        elif rec == "CONECT":
            chk.n_conect += 1
            fields = [line[a:a + 5] for a in range(6, 31, 5)]
            nums = [int(f) for f in fields if f.strip()]
            if not nums:
                issue(f"line {ln}: empty CONECT")
                continue
            bonds.setdefault(nums[0], set()).update(nums[1:])
    known = set(serials)
    for a, partners in bonds.items():
        for b in partners:
            if a not in known or b not in known:
                issue(f"CONECT {a}-{b} references a missing atom")
            elif abs(a - b) != 1:
                issue(f"CONECT {a}-{b} is not a sequential backbone bond")
            elif a not in bonds.get(b, set()):
                issue(f"CONECT {a}-{b} is not reciprocated")
    if lines and lines[-1].strip() != "END":
        issue("file does not terminate with END")
    return chk


def read_pdb(text: str) -> tuple[np.ndarray, str | None]:
    """First-model ATOM/HETATM coordinates by fixed columns.

    Returns (coords, unit) where unit is 'nm' (with scale and offset undone) for files that
    carry our REMARK 250 unit record, or None when the unit is unknown (Angstrom by convention).
    """
    pts = []
    unit_nm = offset = None
    for line in text.splitlines():
        if line.startswith("REMARK 250  COORDINATE UNIT"):
            m = re.search(r":\s*([0-9.eE+-]+)\s*NM", line)
            unit_nm = float(m.group(1)) if m else None
        elif line.startswith("REMARK 250  ORIGIN OFFSET"):
            offset = np.array([float(v) for v in line.split(":", 1)[1].split()])
        elif line.startswith("ENDMDL") and pts:
            break
        elif line.startswith(("ATOM  ", "HETATM")):
            pts.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
    if not pts:
        raise ValueError("No ATOM/HETATM records found.")
    x = np.asarray(pts, dtype=np.float64)
    if unit_nm is not None and offset is not None and offset.size == 3:
        return x * unit_nm - offset, "nm"
    return x, None


def write_xyz(coords_nm: np.ndarray, start_bin: int, chrom: genome.Chrom | None = None) -> str:
    chrom = chrom or genome.DEFAULT
    n = len(coords_nm)
    last = min(start_bin + n - 1, chrom.n_bins - 1)
    region = f"{chrom.name}:{int(chrom.bin_start(start_bin)) + 1}-{int(chrom.bin_end(last))}"
    head = [str(n), f"ChronoCell-5D {region} {chrom.resolution / 1000:g}kb beads units=nm first_bin={start_bin}"]
    return "\n".join(head + [f"C {x:12.4f} {y:12.4f} {z:12.4f}" for x, y, z in coords_nm]) + "\n"


def read_xyz(text: str) -> tuple[np.ndarray, str | None]:
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        raise ValueError("Empty XYZ file.")
    try:
        count = int(lines[0].split()[0])
        comment, body = lines[1] if len(lines) > 1 else "", lines[2:2 + count]
    except ValueError:
        comment, body = "", lines
    pts = []
    for ln in body:
        f = ln.split()
        vals = f[1:4] if len(f) >= 4 else f[:3]
        try:
            pts.append([float(v) for v in vals])
        except ValueError:
            continue
    if not pts:
        raise ValueError("No coordinate lines in XYZ file.")
    return np.asarray(pts, dtype=np.float64), ("nm" if "units=nm" in comment else None)


# ----------------------------------------------------------------------------------------
# Pipeline artefact loaders
# ----------------------------------------------------------------------------------------
_COORD_KEYS = ("coords", "pos", "predicted_coords", "pos_out", "positions", "xyz", "P")


def _torch_load(data: bytes, trusted: bool):
    import torch  # optional dependency

    buf = io.BytesIO(data)
    try:
        return torch.load(buf, map_location="cpu", weights_only=True)
    except Exception:
        if not trusted:
            raise ValueError("This .pt file contains Python objects (e.g. a PyG Data). Unpickling can run "
                             "arbitrary code; confirm it came from your own pipeline to load it.")
        buf.seek(0)
        return torch.load(buf, map_location="cpu", weights_only=False)


def _as_array(obj) -> np.ndarray:
    if hasattr(obj, "detach"):
        return obj.detach().cpu().double().numpy()
    return np.asarray(obj, dtype=np.float64)


def as_coords(obj) -> np.ndarray:
    a = np.squeeze(_as_array(obj))
    if a.ndim == 2 and a.shape[1] != 3 and a.shape[0] == 3:
        a = a.T
    if a.ndim != 2 or a.shape[1] != 3 or len(a) < 4:
        raise ValueError(f"Expected an (N, 3) coordinate array, got shape {a.shape}.")
    if not np.all(np.isfinite(a)):
        raise ValueError("Coordinates contain NaN or infinite values.")
    return a


def read_structure(data: bytes, name: str, trusted: bool = False) -> tuple[np.ndarray, str | None]:
    """(coords, unit or None) from .pdb / .xyz / .npy / .npz / .csv / .pt."""
    ext = os.path.splitext(name.lower())[1]
    if ext == ".pdb":
        return read_pdb(data.decode("utf-8", "replace"))
    if ext == ".xyz":
        return read_xyz(data.decode("utf-8", "replace"))
    if ext == ".npy":
        return as_coords(np.load(io.BytesIO(data), allow_pickle=False)), None
    if ext == ".npz":
        z = np.load(io.BytesIO(data), allow_pickle=False)
        for k in _COORD_KEYS:
            if k in z:
                return as_coords(z[k]), ("nm" if "units_nm" in z else None)
        raise ValueError(f"No coordinate array in npz (keys: {list(z.keys())}).")
    if ext == ".csv":
        import pandas as pd
        df = pd.read_csv(io.BytesIO(data))
        cols = {c.lower(): c for c in df.columns}
        if all(c in cols for c in "xyz"):
            return as_coords(df[[cols["x"], cols["y"], cols["z"]]].to_numpy(float)), None
        return as_coords(df.select_dtypes("number").iloc[:, :3].to_numpy(float)), None
    if ext in (".pt", ".pth"):
        obj = _torch_load(data, trusted)
        if isinstance(obj, dict):
            for k in _COORD_KEYS:
                if k in obj:
                    return as_coords(obj[k]), None
            raise ValueError(f"No coordinate tensor in dict (keys: {list(obj)[:8]}).")
        for k in _COORD_KEYS:
            if getattr(obj, k, None) is not None:
                return as_coords(getattr(obj, k)), None
        return as_coords(obj), None
    raise ValueError(f"Unsupported structure format '{ext}'.")


@dataclass
class GraphData:
    gc: np.ndarray
    epi: np.ndarray
    valid: np.ndarray
    ci: np.ndarray
    cj: np.ndarray
    cm: np.ndarray
    notes: list[str] = field(default_factory=list)


def canonical_contacts(i: np.ndarray, j: np.ndarray, m: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Upper-triangle, off-diagonal, de-duplicated, strictly positive contacts.

    The original pipeline took np.where(matrix > 0) on the full symmetric matrix: every contact
    appears twice and the diagonal (self-contacts, target distance 0) is included.
    """
    notes = []
    i, j, m = np.asarray(i, np.int64), np.asarray(j, np.int64), np.asarray(m, np.float64).reshape(-1)
    diag = i == j
    if diag.any():
        notes.append(f"dropped {int(diag.sum()):,} diagonal self-contacts")
    keep = ~diag & (m > 0) & (i >= 0) & (j >= 0) & (i < n) & (j < n)
    a, b, w = np.minimum(i[keep], j[keep]), np.maximum(i[keep], j[keep]), m[keep]
    key = a * n + b
    uniq, first, inv = np.unique(key, return_index=True, return_inverse=True)
    if uniq.size < key.size:
        notes.append(f"merged {key.size - uniq.size:,} mirrored/duplicate pixels (kept max count)")
    mx = np.zeros(uniq.size)
    np.maximum.at(mx, inv, w)
    return (uniq // n).astype(np.int64), (uniq % n).astype(np.int64), mx, notes


def read_graph(data: bytes, name: str, trusted: bool = False) -> GraphData:
    """Node tracks + contacts from graph .npz (our format) or graph_data.pt (PyG Data)."""
    ext = os.path.splitext(name.lower())[1]
    if ext == ".npz":
        z = dict(np.load(io.BytesIO(data), allow_pickle=False))
    elif ext in (".pt", ".pth"):
        obj = _torch_load(data, trusted)
        get = obj.get if isinstance(obj, dict) else (lambda k, d=None: getattr(obj, k, d))
        z = {k: _as_array(get(k)) for k in ("x", "edge_index", "edge_attr", "gc", "epi", "valid")
             if get(k) is not None}
    else:
        raise ValueError(f"Unsupported graph format '{ext}'.")
    notes: list[str] = []
    if "gc" in z and "epi" in z:
        gc, epi = np.asarray(z["gc"], float), np.asarray(z["epi"], float)
    elif "x" in z:
        x = np.asarray(z["x"], float)
        gc, epi = x[:, 0].copy(), x[:, 1].copy()
        notes.append("node features read from x[:, 0:2] as [f_GC, f_epi]")
    else:
        raise ValueError("Graph has neither gc/epi arrays nor a node feature matrix x.")
    n = len(gc)
    valid = np.asarray(z["valid"], bool) if "valid" in z else np.isfinite(gc) & (gc > 0)
    if "valid" not in z:
        notes.append("assembled mask inferred from f_GC > 0 (legacy files encode N-bins as 0)")
    gc = np.where(valid, gc, np.nan)
    epi = np.where(valid, epi, np.nan)
    if {"ci", "cj", "cm"} <= z.keys():
        i, j, m = z["ci"], z["cj"], z["cm"]
    elif "edge_index" in z:
        ei = np.asarray(z["edge_index"], np.int64)
        i, j = ei[0], ei[1]
        m = np.asarray(z.get("edge_attr", np.ones(ei.shape[1])), float).reshape(-1)
    else:
        raise ValueError("Graph has no contacts (ci/cj/cm or edge_index/edge_attr).")
    ci, cj, cm, cnotes = canonical_contacts(i, j, m, n)
    return GraphData(gc, epi, valid, ci, cj, cm, notes + cnotes)


def write_graph_npz(path: str, gc: np.ndarray, epi: np.ndarray, valid: np.ndarray,
                    ci: np.ndarray, cj: np.ndarray, cm: np.ndarray, chrom: str = genome.CHROM,
                    resolution: int = genome.RESOLUTION) -> None:
    np.savez_compressed(path, gc=gc, epi=epi, valid=valid, ci=ci, cj=cj, cm=cm,
                        chrom=np.array(chrom), resolution=np.array(resolution))


# ----------------------------------------------------------------------------------------
# Structure bundle: the hand-off format between Colab (GPU) and the workstation
# ----------------------------------------------------------------------------------------
@dataclass
class StructureBundle:
    """One chromosome, one condition, T >= 1 frames (the 4th dimension: time or state).

    frames      (T, N, 3) float, nanometres
    times       (T,) float, in `time_unit` (hours, days, pseudotime, relaxation steps, ...)
    labels      T frame labels, e.g. ["G1", "S", "G2/M"] or ["0 h", "24 h"]
    Optional per-bin tracks (gc, epi, valid) and contacts (ci, cj, cm) of the same chromosome.
    """
    chrom: str
    resolution: int
    frames: np.ndarray
    times: np.ndarray
    labels: list[str]
    condition: str = "reference"
    source: str = ""
    time_unit: str = "frame"
    start_bin: int = 0
    gc: np.ndarray | None = None
    epi: np.ndarray | None = None
    valid: np.ndarray | None = None
    ci: np.ndarray | None = None
    cj: np.ndarray | None = None
    cm: np.ndarray | None = None

    @property
    def n_frames(self) -> int:
        return self.frames.shape[0]

    @property
    def n_beads(self) -> int:
        return self.frames.shape[1]

    def validate(self) -> None:
        if self.frames.ndim != 3 or self.frames.shape[2] != 3:
            raise ValueError(f"frames must be (T, N, 3); got {self.frames.shape}.")
        if self.n_beads < 4:
            raise ValueError("A structure needs at least 4 beads.")
        if not np.all(np.isfinite(self.frames)):
            raise ValueError("frames contain NaN or infinite values.")
        if len(self.times) != self.n_frames or len(self.labels) != self.n_frames:
            raise ValueError("times and labels must have one entry per frame.")
        if self.chrom not in genome.MAIN_CHROMOSOMES:
            raise ValueError(f"Unknown chromosome '{self.chrom}'; expected one of chr1..chr22, chrX, chrY.")
        n_bins = genome.chrom(self.chrom, self.resolution).n_bins
        if self.start_bin < 0 or self.start_bin + self.n_beads > n_bins:
            raise ValueError(f"Bins {self.start_bin}..{self.start_bin + self.n_beads - 1} fall outside "
                             f"{self.chrom} at {self.resolution:,} bp ({n_bins:,} bins).")


def write_bundle(path_or_buffer, b: StructureBundle) -> None:
    b.validate()
    arrays = dict(frames=b.frames.astype(np.float32), times=np.asarray(b.times, float),
                  labels=np.array(b.labels), chrom=np.array(b.chrom), resolution=np.array(b.resolution),
                  condition=np.array(b.condition), source=np.array(b.source), time_unit=np.array(b.time_unit),
                  units_nm=np.array(1.0), start_bin=np.array(b.start_bin), format=np.array("chronocell-bundle-1"))
    for k in ("gc", "epi", "valid", "ci", "cj", "cm"):
        v = getattr(b, k)
        if v is not None:
            arrays[k] = v
    np.savez_compressed(path_or_buffer, **arrays)


def region_hint(data: bytes, ext: str) -> tuple[str, int, int] | None:
    """(chrom, start_bin, resolution) from a ChronoCell PDB REMARK 250 or XYZ comment line."""
    head = data[:4000].decode("utf-8", "replace")
    if ext == ".pdb":
        reg = re.search(r"REMARK 250  REGION\s*:\s*CHR(\w+):(\d+)-(\d+)", head)
        kb = re.search(r"ONE CA PER ([0-9.]+) KB", head)
        if reg and kb:
            res = int(round(float(kb.group(1)) * 1000))
            return f"chr{reg.group(1)}", (int(reg.group(2)) - 1) // res, res
    if ext == ".xyz":
        m = re.search(r"(chr\w+):(\d+)-(\d+) ([0-9.]+)kb .*first_bin=(\d+)", head)
        if m:
            return m.group(1), int(m.group(5)), int(round(float(m.group(4)) * 1000))
    return None


def _scalar(z, key, default):
    return z[key].item() if key in z else default


def read_bundle(data: bytes, name: str, chrom_hint: str = genome.CHROM, trusted: bool = False,
                unit: str = "Auto", b0: float | None = None) -> tuple[StructureBundle, list[str]]:
    """Any supported coordinate file -> StructureBundle (+ notes about conversions applied).

    Bundle .npz files carry chromosome, resolution and units. Single-structure files (.pdb,
    .xyz, .npy, .csv, .pt, plain .npz) become a one-frame bundle for `chrom_hint`; the
    resolution is inferred from the bead count, and unknown units are calibrated to b0.
    """
    from . import physics

    notes: list[str] = []
    ext = os.path.splitext(name.lower())[1]
    z = dict(np.load(io.BytesIO(data), allow_pickle=False)) if ext == ".npz" else {}
    if "frames" in z:
        frames = np.asarray(z["frames"], dtype=np.float64)
        if frames.ndim == 2:
            frames = frames[None]
        chrom_name = str(_scalar(z, "chrom", chrom_hint))
        t = frames.shape[0]
        b = StructureBundle(
            chrom=chrom_name,
            resolution=int(_scalar(z, "resolution", genome.resolution_for_beads(chrom_name, frames.shape[1]))),
            frames=frames * float(_scalar(z, "units_nm", 1.0)),
            times=np.asarray(z.get("times", np.arange(t)), float),
            labels=[str(v) for v in z.get("labels", [f"t{k}" for k in range(t)])],
            condition=str(_scalar(z, "condition", "uploaded")), source=str(_scalar(z, "source", name)),
            time_unit=str(_scalar(z, "time_unit", "frame")), start_bin=int(_scalar(z, "start_bin", 0)),
            **{k: z[k] for k in ("gc", "epi", "valid", "ci", "cj", "cm") if k in z})
        declared = "nm"
    else:
        coords, declared = read_structure(data, name, trusted=trusted)
        hint = region_hint(data, ext)
        if hint:
            chrom_name, start_bin, res = hint
            notes.append(f"Region read from file header: {chrom_name}, bin {start_bin:,}, {res:,} bp.")
        else:
            chrom_name, start_bin, res = chrom_hint, 0, genome.resolution_for_beads(chrom_hint, len(coords))
        b = StructureBundle(chrom=chrom_name, resolution=res, frames=coords[None].astype(np.float64),
                            times=np.zeros(1), labels=["t0"], condition="uploaded", source=name, start_bin=start_bin)
    b.validate()
    if not (unit == "nm" or (unit == "Auto" and declared == "nm")):
        if unit == "Å":
            b.frames = b.frames / 10.0
        elif unit == "µm":
            b.frames = b.frames * 1000.0
        else:
            b0 = b0 or physics.bond_length_for(b.resolution)
            med = float(np.median(np.linalg.norm(np.diff(b.frames[0], axis=0), axis=1)))
            if med <= 0:
                raise ValueError("Cannot calibrate units: median bond length is zero.")
            b.frames = b.frames * (b0 / med)
            notes.append(f"Unknown coordinate unit: rescaled ×{b0 / med:.4g} so the median bond equals b₀ = {b0:.0f} nm.")
    n_expected = genome.chrom(b.chrom, b.resolution).n_bins
    if b.n_beads != n_expected:
        notes.append(f"Window of {b.n_beads:,} beads: bins {b.start_bin:,}–{b.start_bin + b.n_beads - 1:,} of "
                     f"{n_expected:,} ({b.chrom} at {b.resolution:,} bp).")
    return b, notes


def report_json(payload: dict) -> str:
    def default(o):
        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o).__name__)
    return json.dumps(payload, indent=2, default=default, allow_nan=False)
