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
SEGMENT = "CH22"
ELEMENT = " C"


def _pad(line: str) -> str:
    if len(line) > PDB_WIDTH:
        raise ValueError(f"PDB record exceeds 80 columns: {line!r}")
    return line.ljust(PDB_WIDTH)


def atom_record(serial: int, res_seq: int, x: float, y: float, z: float, occ: float, bfac: float) -> str:
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
            f"{x:8.3f}{y:8.3f}{z:8.3f}{occ:6.2f}{bfac:6.2f}      {SEGMENT:<4s}{ELEMENT:>2s}  ")
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
              source: str, method: str) -> tuple[str, PdbFrame]:
    """PDB text for beads [start_bin, start_bin + N): HEADER, TITLE, REMARK 2/250, CRYST1,
    ATOM, TER, CONECT (each bead lists its sequence neighbours i-1 and i+1 only), END."""
    coords_nm = np.asarray(coords_nm, dtype=np.float64)
    n = len(coords_nm)
    frame = pdb_frame(coords_nm)
    q = (coords_nm + frame.offset_nm) / frame.unit_nm
    idx = np.arange(start_bin, start_bin + n)
    occ = np.nan_to_num(np.clip(gc[idx], 0.0, 1.0), nan=0.0)
    bfac = epi_to_bfactor(epi[idx], epi_ref)
    region = f"{genome.CHROM.upper()}:{int(genome.bin_start(start_bin)) + 1}-{int(genome.bin_end(start_bin + n - 1))}"
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%d-%b-%y").upper()

    def remark250(key: str, value: str) -> str:
        return _pad(f"REMARK 250  {key:<31s}: {value}"[:PDB_WIDTH])

    lines = [
        _pad(f"HEADER    {'CHROMATIN STRUCTURE':<40s}{stamp:>9s}   CC5D"),
        _pad("TITLE     CHRONOCELL-5D MODEL OF HUMAN CHROMOSOME 22 (GRCH38) AT 10 KB"),
        _pad("REMARK   2"),
        _pad("REMARK   2 RESOLUTION. NOT APPLICABLE."),
        _pad("REMARK 250"),
        _pad("REMARK 250 EXPERIMENTAL DETAILS"),
        remark250("EXPERIMENT TYPE", "COMPUTATIONAL MODEL"),
        remark250("METHOD", method.upper()[:36]),
        remark250("SOURCE", source.upper()[:36]),
        remark250("REGION", region),
        remark250("BEAD", "ONE CA PER 10 KB BIN, RESSEQ = BIN + 1"),
        remark250("COORDINATE UNIT", f"{frame.unit_nm:g} NM PER FILE UNIT (NOT A)"),
        remark250("ORIGIN OFFSET (NM)", " ".join(f"{v:.3f}" for v in frame.offset_nm)),
        remark250("OCCUPANCY", "F_GC; 0.00 = UNASSEMBLED (N) BIN"),
        remark250("B-FACTOR", f"99.99*LN(1+F_EPI)/LN(1+{epi_ref:.3f})"),
        _pad(f"CRYST1{1.0:9.3f}{1.0:9.3f}{1.0:9.3f}{90.0:7.2f}{90.0:7.2f}{90.0:7.2f} {'P 1':<11s}{1:4d}"),
    ]
    for k in range(n):
        lines.append(atom_record(k + 1, int(idx[k]) + 1, *q[k], float(occ[k]), float(bfac[k])))
    lines.append(_pad(f"TER   {n + 1:5d}      {RES_NAME} {CHAIN}{int(idx[-1]) + 1:4d}"))
    for k in range(n):
        partners = [s for s in (k, k + 2) if 1 <= s <= n]        # serials of beads i-1 and i+1
        lines.append(_pad("CONECT" + f"{k + 1:5d}" + "".join(f"{s:5d}" for s in partners)))
    lines.append(_pad("END"))
    return "\n".join(lines) + "\n", frame


@dataclass
class PdbCheck:
    n_atoms: int = 0
    n_conect: int = 0
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
            if serials and serial != serials[-1] + 1:
                issue(f"line {ln}: atom serial {serial} not sequential")
            serials.append(serial)
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


def write_xyz(coords_nm: np.ndarray, start_bin: int) -> str:
    n = len(coords_nm)
    region = f"{genome.CHROM}:{int(genome.bin_start(start_bin)) + 1}-{int(genome.bin_end(start_bin + n - 1))}"
    head = [str(n), f"ChronoCell-5D {region} 10kb beads units=nm first_bin={start_bin}"]
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
                    ci: np.ndarray, cj: np.ndarray, cm: np.ndarray) -> None:
    np.savez_compressed(path, gc=gc, epi=epi, valid=valid, ci=ci, cj=cj, cm=cm,
                        chrom=np.array(genome.CHROM), resolution=np.array(genome.RESOLUTION))


def report_json(payload: dict) -> str:
    def default(o):
        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o).__name__)
    return json.dumps(payload, indent=2, default=default, allow_nan=False)
