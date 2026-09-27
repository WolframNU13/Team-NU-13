"""
Biological-state data engine: find structure and signal files by *format*, not by name.

Every candidate file is sniffed from its content:

    .npy  (N, 3) or (3, N)   -> coordinates (one conformation)
    .npy  (T, N, 3)          -> coordinate frames (time course / several conformations)
    .npy  (N,) or (N, 1)     -> 1-D epigenomic track (H3K27ac or any per-bead signal)
    .pdb  ATOM/HETATM lines  -> coordinates (first MODEL)
    .npz  frames / coords    -> ChronoCell bundle or coordinate archive
    .xyz / .csv              -> coordinates

Only the .npy header is read to classify arrays (no data load, never unpickling). A file is
assigned to one of three biological states from words in its path (file name first, then the
enclosing folders), e.g. `tumour_K562.npy`, `Senescent/IMR90.pdb`, `ctrl/h3k27ac.npy`; files
uploaded into a state in the UI are assigned explicitly. A path component that names a
chromosome (`chr9`, `chrX`) restricts the file to that chromosome; otherwise it applies to the
chromosome being viewed.
"""

from __future__ import annotations

import io
import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HEALTHY, DISEASE, SENESCENT = "Healthy Control", "Disease State / Cancer", "Senescent State"
STATES = (HEALTHY, DISEASE, SENESCENT)
SHORT_NAMES = {HEALTHY: "Healthy", DISEASE: "Disease", SENESCENT: "Senescent"}

COORDS, FRAMES, TRACK, GRAPH, UNSUPPORTED = "coords", "frames", "track", "graph", "unsupported"
EXTENSIONS = (".npy", ".pdb", ".npz", ".xyz", ".csv")
MIN_BEADS = 4
MAX_SCAN_FILES = 400
MAX_DEPTH = 4
_COORD_KEYS = ("coords", "pos", "predicted_coords", "pos_out", "positions", "xyz", "P")

# Long words match anywhere in a path component (handles CamelCase / suffixes); short tokens
# must stand alone between separators. The longest match in the deepest component wins, so
# "abnormal" (disease) beats "normal" (healthy).
_WORDS = {
    HEALTHY: ("healthy", "control", "normal", "wildtype", "wild-type", "baseline", "untreated", "unaffected"),
    DISEASE: ("disease", "diseased", "cancer", "tumor", "tumour", "malignant", "carcinoma", "leukemia", "leukaemia",
              "lymphoma", "sarcoma", "myeloma", "glioma", "melanoma", "neoplasm", "metasta", "patient", "parkinson",
              "alzheimer", "mutant", "affected", "abnormal", "deletion", "translocation"),
    SENESCENT: ("senescent", "senescence", "sasp", "ageing", "aging", "replicative", "presenescent"),
}
_TOKENS = {
    HEALTHY: ("ctrl", "wt", "hc", "mock"),
    DISEASE: ("cml", "aml", "pd", "mut", "tum", "ph"),
    SENESCENT: ("sen", "ois", "ris", "aged"),
}
# Words that contain a state word but do not name a state ("normalized counts", "uncontrolled").
_NOISE = ("normaliz", "normalis", "uncontrol")
_CHROM_RE = re.compile(r"(?<![A-Za-z0-9])chr([0-9]{1,2}|[XYxy])(?![0-9A-Za-z])")


@dataclass(frozen=True)
class StateFile:
    key: str                      # "folder:<relative path>" or "upload:<state>:<name>"
    name: str                     # display name (relative path or file name)
    origin: str                   # "folder" | "upload"
    kind: str                     # COORDS | FRAMES | TRACK | GRAPH | UNSUPPORTED
    n: int                        # beads (structures) or length (tracks)
    shape: tuple[int, ...]
    state: str | None
    chrom: str | None             # chromosome named in the path, if any
    detail: str                   # human-readable format summary or the reason it is unsupported
    path: str | None = None       # absolute path (folder files)

    @property
    def is_structure(self) -> bool:
        return self.kind in (COORDS, FRAMES)

    @property
    def is_track(self) -> bool:
        return self.kind == TRACK


# ----------------------------------------------------------------------------------------
# Format sniffing
# ----------------------------------------------------------------------------------------
def _classify_array(shape: tuple[int, ...], dtype: np.dtype) -> tuple[str, int, str]:
    if dtype.kind not in "iuf":
        return UNSUPPORTED, 0, f"dtype {dtype} is not numeric"
    s = tuple(int(v) for v in shape)
    if len(s) == 1 or (len(s) == 2 and s[1] == 1):
        n = s[0]
        return (TRACK, n, f"1-D track · {n:,} values") if n >= MIN_BEADS else (UNSUPPORTED, n, "track too short")
    if len(s) == 2 and s[1] == 3 and s[0] >= MIN_BEADS:
        return COORDS, s[0], f"(N, 3) coordinates · {s[0]:,} beads"
    if len(s) == 2 and s[0] == 3 and s[1] >= MIN_BEADS:
        return COORDS, s[1], f"(3, N) coordinates · {s[1]:,} beads (transposed on load)"
    if len(s) == 3 and s[2] == 3 and s[1] >= MIN_BEADS and s[0] >= 1:
        return FRAMES, s[1], f"(T, N, 3) frames · {s[0]} × {s[1]:,} beads"
    return UNSUPPORTED, 0, f"shape {s} is neither (N, 3) coordinates nor an (N,) track"


def _npy_header(fh) -> tuple[tuple[int, ...], np.dtype]:
    version = np.lib.format.read_magic(fh)
    if version == (1, 0):
        shape, _, dtype = np.lib.format.read_array_header_1_0(fh)
    elif version == (2, 0):
        shape, _, dtype = np.lib.format.read_array_header_2_0(fh)
    else:  # format 3.0 (utf-8 header): fall back to a pickle-free load of the header via np.load
        fh.seek(0)
        arr = np.load(fh, allow_pickle=False, mmap_mode=None)
        shape, dtype = arr.shape, arr.dtype
    return tuple(shape), np.dtype(dtype)


def _count_pdb(text: str) -> tuple[int, int]:
    atoms = models = 0
    in_first = True
    for line in text.splitlines():
        if line.startswith("MODEL"):
            models += 1
        elif line.startswith("ENDMDL"):
            in_first = False
        elif in_first and line.startswith(("ATOM  ", "HETATM")):
            atoms += 1
    return atoms, max(models, 1)


def sniff(name: str, data: bytes | None = None, path: str | os.PathLike | None = None) -> tuple[str, int, tuple, str]:
    """(kind, n, shape, detail) from the file's content. Never raises."""
    ext = os.path.splitext(str(name).lower())[1]
    try:
        if ext == ".npy":
            if data is not None:
                shape, dtype = _npy_header(io.BytesIO(data))
            else:
                with open(path, "rb") as fh:
                    shape, dtype = _npy_header(fh)
            kind, n, detail = _classify_array(shape, dtype)
            return kind, n, shape, detail
        if data is None:
            data = Path(path).read_bytes()
        if ext == ".pdb":
            atoms, models = _count_pdb(data.decode("utf-8", "replace"))
            if atoms < MIN_BEADS:
                return UNSUPPORTED, atoms, (), "no ATOM/HETATM records"
            kind = FRAMES if models > 1 else COORDS
            extra = f" · {models} models" if models > 1 else ""
            return kind, atoms, (atoms, 3), f"wwPDB · {atoms:,} atoms{extra}"
        if ext == ".npz":
            z = np.load(io.BytesIO(data), allow_pickle=False)
            keys = set(z.files)
            if "frames" in keys:
                f = z["frames"]
                shape = tuple(f.shape) if f.ndim == 3 else (1, *f.shape)
                return FRAMES if shape[0] > 1 else COORDS, shape[1], shape, f"bundle · {shape[0]} frame(s) × {shape[1]:,} beads"
            for k in _COORD_KEYS:
                if k in keys:
                    kind, n, detail = _classify_array(z[k].shape, z[k].dtype)
                    if kind in (COORDS, FRAMES):
                        return kind, n, tuple(z[k].shape), f"npz[{k}] · {detail}"
            if {"gc", "epi"} <= keys or "edge_index" in keys:
                return GRAPH, int(len(z["gc"])) if "gc" in keys else 0, (), "graph (tracks + contacts)"
            return UNSUPPORTED, 0, (), f"npz without coordinates (keys: {', '.join(sorted(keys)[:6])})"
        if ext in (".xyz", ".csv"):
            from .formats import read_structure
            x, _ = read_structure(data, name)
            return COORDS, len(x), x.shape, f"{ext[1:].upper()} · {len(x):,} beads"
    except Exception as exc:  # corrupted / truncated files are reported, not raised
        return UNSUPPORTED, 0, (), f"unreadable: {str(exc)[:120]}"
    return UNSUPPORTED, 0, (), f"extension {ext or '(none)'} not supported"


def load_track(data: bytes) -> np.ndarray:
    """1-D float track from .npy bytes (pickle disabled)."""
    a = np.load(io.BytesIO(data), allow_pickle=False)
    a = np.asarray(a, dtype=np.float64).reshape(-1) if a.ndim == 2 and a.shape[1] == 1 else np.asarray(a, np.float64)
    if a.ndim != 1:
        raise ValueError(f"Expected a 1-D track, got shape {a.shape}.")
    return a


# ----------------------------------------------------------------------------------------
# State and chromosome inference from the path
# ----------------------------------------------------------------------------------------
def _component_state(component: str) -> tuple[str | None, int]:
    low = component.lower()
    for w in _NOISE:
        low = low.replace(w, " ")
    best: tuple[str | None, int] = (None, 0)
    tie = False
    tokens = set(re.split(r"[^a-z0-9]+", low))
    for state in STATES:
        hits = [w for w in _WORDS[state] if w in low] + [t for t in _TOKENS[state] if t in tokens]
        if not hits:
            continue
        length = max(len(h) for h in hits)
        if length > best[1]:
            best, tie = (state, length), False
        elif length == best[1] and state != best[0]:
            tie = True
    return (None, 0) if tie else best


def infer_state(rel_path: str) -> str | None:
    """State from the deepest path component that names one (file name, then folders)."""
    parts = [p for p in re.split(r"[\\/]", rel_path) if p]
    if parts:
        parts[-1] = os.path.splitext(parts[-1])[0]
    for comp in reversed(parts):
        state, _ = _component_state(comp)
        if state:
            return state
    return None


def infer_chrom(rel_path: str) -> str | None:
    for comp in reversed([p for p in re.split(r"[\\/]", rel_path) if p]):
        m = _CHROM_RE.search(comp)
        if m:
            v = m.group(1)
            return f"chr{v.upper() if v.isalpha() else int(v)}"
    return None


# ----------------------------------------------------------------------------------------
# Scanning
# ----------------------------------------------------------------------------------------
def folder_signature(root: str | os.PathLike) -> tuple[tuple[str, float, int], ...]:
    """Cheap fingerprint (path, mtime, size) of every candidate file under root, for caching."""
    root = Path(root)
    out = []
    if not root.is_dir():
        return ()
    for dirpath, dirnames, filenames in os.walk(root):
        depth = len(Path(dirpath).relative_to(root).parts)
        dirnames[:] = sorted(d for d in dirnames if not d.startswith((".", "_")) and depth < MAX_DEPTH)
        for fn in sorted(filenames):
            if os.path.splitext(fn.lower())[1] in EXTENSIONS:
                p = os.path.join(dirpath, fn)
                try:
                    st = os.stat(p)
                except OSError:
                    continue
                out.append((p, st.st_mtime, st.st_size))
                if len(out) >= MAX_SCAN_FILES:
                    return tuple(out)
    return tuple(out)


def scan_folder(root: str | os.PathLike, signature=None) -> list[StateFile]:
    """Every structure / track file under root with its sniffed format, state and chromosome."""
    root = Path(root)
    files = []
    for p, _, _ in (signature if signature is not None else folder_signature(root)):
        rel = os.path.relpath(p, root).replace("\\", "/")
        kind, n, shape, detail = sniff(p, path=p)
        files.append(StateFile(key=f"folder:{rel}", name=rel, origin="folder", kind=kind, n=int(n),
                               shape=tuple(int(v) for v in shape), state=infer_state(rel), chrom=infer_chrom(rel),
                               detail=detail, path=p))
    return files


def sniff_upload(state: str, name: str, data: bytes) -> StateFile:
    kind, n, shape, detail = sniff(name, data=data)
    return StateFile(key=f"upload:{state}:{name}", name=name, origin="upload", kind=kind, n=int(n),
                     shape=tuple(int(v) for v in shape), state=state, chrom=infer_chrom(name), detail=detail)


def for_chromosome(files: list[StateFile], chrom_name: str) -> list[StateFile]:
    """Files that apply to chrom_name: those naming it, plus chromosome-agnostic ones."""
    return [f for f in files if f.chrom in (None, chrom_name)]


# ----------------------------------------------------------------------------------------
# Pairing structures with tracks
# ----------------------------------------------------------------------------------------
@dataclass(frozen=True)
class StatePlan:
    state: str
    structures: tuple[StateFile, ...]
    tracks: tuple[StateFile, ...]
    other: tuple[StateFile, ...]          # graphs and unsupported files (reported, not used)

    @property
    def empty(self) -> bool:
        return not self.structures and not self.tracks


def plan(files: list[StateFile], state: str) -> StatePlan:
    mine = [f for f in files if f.state == state]
    order = lambda f: (0 if f.origin == "upload" else 1, f.name.lower())  # uploads first, then by path
    return StatePlan(state,
                     tuple(sorted((f for f in mine if f.is_structure), key=order)),
                     tuple(sorted((f for f in mine if f.is_track), key=order)),
                     tuple(sorted((f for f in mine if not f.is_structure and not f.is_track), key=order)))


def _stem_tokens(name: str) -> set[str]:
    return {t for t in re.split(r"[^a-z0-9]+", os.path.splitext(name.lower())[0]) if len(t) > 1}


def best_track(structure: StateFile | None, tracks: tuple[StateFile, ...] | list[StateFile],
               n_full: int | None = None) -> StateFile | None:
    """The track most likely to belong to `structure`: equal length (one value per bead) or the
    full chromosome length, then same folder, then most shared name tokens."""
    if not tracks:
        return None
    if structure is None:
        return tracks[0]

    def score(t: StateFile) -> tuple:
        length = 2 if t.n == structure.n else (1 if n_full and t.n == n_full else 0)
        same_dir = int(os.path.dirname(t.name) == os.path.dirname(structure.name))
        shared = len(_stem_tokens(t.name) & _stem_tokens(structure.name))
        return (length, same_dir, shared)

    best = max(tracks, key=score)
    return best if score(best)[0] > 0 else None
