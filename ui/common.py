"""
Shared UI layer: dataset assembly (coordinate slot -> reference fallback), window clamping and
small HTML helpers. Numerics stay in `chronocell`; this module only wires them to Streamlit.

Coordinate slot
---------------
Spatial coordinates enter the workstation from three places, in priority order:

1. Files uploaded in the Data menu (any number; each is one condition).
2. Files dropped into `coordinates/<chrom>/` (e.g. Colab outputs), picked up automatically.
3. The reference model: a planted synthetic chromosome, clearly labelled, which stays in place
   until real coordinates are provided.

A `graph*.npz` file in the same folder (or uploaded) supplies GC, H3K27ac and Micro-C contacts.
Biological-state files (see `chronocell/states.py`) add a per-bead signal track and a condition
label on top of the same loader.
"""

from __future__ import annotations

import hashlib
import math
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import streamlit as st

from chronocell import formats, genome, physics, states, synthetic

# The coordinate slot; CHRONOCELL_COORDINATES points it at another folder (e.g. a Colab download).
SLOT_ROOT = Path(os.environ.get("CHRONOCELL_COORDINATES") or Path(__file__).resolve().parent.parent / "coordinates")
COORD_TYPES = ("npz", "pdb", "xyz", "npy", "csv", "pt", "pth")
MIN_WINDOW = 10


# ======================================================================================
# Window arithmetic (the single place where user-supplied ranges are made safe)
# ======================================================================================
def clamp_window(lo, hi, n: int, min_len: int = MIN_WINDOW) -> tuple[int, int]:
    """Any pair of numbers -> a valid half-open bin window inside [0, n).

    Handles reversed bounds, negatives, values past the end, NaN/None and windows shorter than
    `min_len` (expanded, and shifted back inside the data if they would overrun the end).
    """
    n = int(n)
    if n <= 0:
        raise ValueError("Dataset has no beads.")
    min_len = max(1, min(int(min_len), n))

    def _int(v, default):
        try:
            v = float(v)
        except (TypeError, ValueError):
            return default
        return default if not math.isfinite(v) else int(round(v))

    lo, hi = _int(lo, 0), _int(hi, n)
    if hi < lo:
        lo, hi = hi, lo
    lo = min(max(lo, 0), n - min_len)
    hi = min(max(hi, lo + min_len), n)
    return lo, hi


# ======================================================================================
# Dataset
# ======================================================================================
@dataclass(frozen=True)
class Dataset:
    key: str
    chrom: genome.Chrom
    bin0: int                       # genomic bin of local bead 0 (windowed files)
    frames: np.ndarray              # (T, N, 3) nm
    frame_labels: tuple[str, ...]
    frame_times: np.ndarray
    time_unit: str
    condition: str
    gc: np.ndarray
    epi: np.ndarray
    valid: np.ndarray
    ci: np.ndarray                  # local bead indices, i < j
    cj: np.ndarray
    cm: np.ndarray
    truth: np.ndarray | None        # planted structure (reference model only)
    structure_label: str
    tracks_label: str
    is_reference: bool
    tracks_are_placeholder: bool
    notes: tuple[str, ...] = field(default_factory=tuple)
    signal_label: str = "H3K27ac"             # what `epi` holds (a state track may replace H3K27ac)
    signal_is_placeholder: bool = False        # epi is the reference placeholder, not measured

    @property
    def n(self) -> int:
        return self.frames.shape[1]

    @property
    def n_frames(self) -> int:
        return self.frames.shape[0]

    @property
    def has_contacts(self) -> bool:
        return self.ci.size > 0

    @property
    def epi_ref(self) -> float:
        e = self.epi[np.isfinite(self.epi)]
        return float(np.percentile(e, 99.5)) if e.size else 1.0

    def gbin(self, local) -> np.ndarray:
        return np.asarray(local, dtype=np.int64) + self.bin0


def digest(*parts) -> str:
    h = hashlib.sha1()
    for p in parts:
        h.update(p if isinstance(p, bytes) else repr(p).encode())
    return h.hexdigest()[:16]


@st.cache_resource(show_spinner="Building the reference chromosome…")
def reference_chromosome(chrom_name: str, resolution: int, seed: int, b0: float) -> synthetic.SyntheticChromosome:
    return synthetic.build(genome.chrom(chrom_name, resolution), seed=seed, b0=b0)


def _is_track_file(p: Path) -> bool:
    """1-D .npy arrays are signal tracks, not coordinates (read from the header only)."""
    return p.suffix.lower() == ".npy" and states.sniff(p.name, path=p)[0] == states.TRACK


def slot_files(chrom_name: str) -> list[Path]:
    """Coordinate files waiting in coordinates/<chrom>/ (graph*.npz and 1-D signal tracks excluded)."""
    d = SLOT_ROOT / chrom_name
    if not d.is_dir():
        return []
    return sorted(p for p in d.iterdir()
                  if p.is_file() and p.suffix.lstrip(".").lower() in COORD_TYPES
                  and not p.name.lower().startswith("graph") and not _is_track_file(p))


def slot_graph(chrom_name: str) -> Path | None:
    d = SLOT_ROOT / chrom_name
    hits = sorted(d.glob("graph*.npz")) if d.is_dir() else []
    return hits[0] if hits else None


def _window_tracks(arr: np.ndarray | None, n_full: int, bin0: int, n: int) -> np.ndarray | None:
    if arr is None:
        return None
    arr = np.asarray(arr, dtype=np.float64)
    if len(arr) == n:
        return arr
    if len(arr) == n_full:
        return arr[bin0:bin0 + n]
    return None


def _window_contacts(ci, cj, cm, bin0: int, n: int, full: bool):
    ci, cj, cm = np.asarray(ci, np.int64), np.asarray(cj, np.int64), np.asarray(cm, np.float64)
    if full:
        ci, cj = ci - bin0, cj - bin0
    keep = (ci >= 0) & (cj >= 0) & (ci < n) & (cj < n) & (ci != cj) & (cm > 0)
    a, b = np.minimum(ci[keep], cj[keep]), np.maximum(ci[keep], cj[keep])
    return a, b, cm[keep]


@st.cache_resource(show_spinner=False)
def load_dataset(chrom_name: str, seed: int, b0: float | None, source: tuple, unit: str,
                 graph: tuple | None, trusted: bool, track: tuple | None = None,
                 condition: str | None = None) -> Dataset:
    """source / graph / track = (label, name, bytes) or None (source None = the reference model).

    `track` is a 1-D per-bead signal (.npy) that replaces H3K27ac, e.g. a biological state's own
    track; `condition` overrides the condition label (the biological state's name).
    """
    notes: list[str] = []
    if source is None:
        ch = genome.chrom(chrom_name)
        b0 = b0 or physics.bond_length_for(ch.resolution)
        ref = reference_chromosome(ch.name, ch.resolution, seed, b0)
        frames = ref.coords[None]
        bin0, labels, times, t_unit, cond = 0, ("reference",), np.zeros(1), "frame", "reference"
        s_label, is_ref, truth = f"Reference model · synthetic fractal globule · seed {seed}", True, ref.coords
        bundle = None
    else:
        s_label, s_name, s_bytes = source
        bundle, bnotes = formats.read_bundle(s_bytes, s_name, chrom_hint=chrom_name, trusted=trusted, unit=unit, b0=b0)
        notes += bnotes
        ch = genome.chrom(bundle.chrom, bundle.resolution)
        frames, bin0 = bundle.frames, bundle.start_bin
        labels, times, t_unit, cond = tuple(bundle.labels), bundle.times, bundle.time_unit, bundle.condition
        is_ref, truth = False, None
        if bundle.chrom != chrom_name:
            notes.append(f"{s_name} describes {bundle.chrom}; the chromosome selector was overridden.")
    n = frames.shape[1]
    n_full = ch.n_bins

    # ---- tracks and contacts: uploaded/slot graph > bundle's own > reference > placeholder ----
    gc = epi = valid = None
    ci = cj = cm = np.empty(0)
    t_label, placeholder = "", False
    if graph is not None:
        g_label, g_name, g_bytes = graph
        gd = formats.read_graph(g_bytes, g_name, trusted=trusted)
        gc, epi = _window_tracks(gd.gc, n_full, bin0, n), _window_tracks(gd.epi, n_full, bin0, n)
        valid = _window_tracks(gd.valid.astype(float), n_full, bin0, n)
        if gc is None:
            notes.append(f"{g_name}: {len(gd.gc):,} bins do not match {ch.name} at {ch.resolution:,} bp; graph ignored.")
        else:
            ci, cj, cm = _window_contacts(gd.ci, gd.cj, gd.cm, bin0, n, full=len(gd.gc) == n_full)
            t_label = g_label
            notes += gd.notes
    if gc is None and bundle is not None and bundle.gc is not None:
        gc, epi = _window_tracks(bundle.gc, n_full, bin0, n), _window_tracks(bundle.epi, n_full, bin0, n)
        valid = _window_tracks(bundle.valid.astype(float), n_full, bin0, n) if bundle.valid is not None else None
        if bundle.ci is not None:
            ci, cj, cm = _window_contacts(bundle.ci, bundle.cj, bundle.cm, bin0, n,
                                          full=bundle.gc is not None and len(bundle.gc) == n_full)
        t_label = "tracks embedded in the coordinate file"
    if gc is None:
        ref = reference_chromosome(ch.name, ch.resolution, seed, b0 or physics.bond_length_for(ch.resolution))
        gc, epi, valid = ref.gc[bin0:bin0 + n], ref.epi[bin0:bin0 + n], ref.valid[bin0:bin0 + n].astype(float)
        if source is None:
            ci, cj, cm, t_label = ref.ci, ref.cj, ref.cm, "Reference tracks & simulated Micro-C contacts"
        else:
            placeholder = True
            t_label = "Placeholder tracks (reference) · no contacts — provide a graph"
    if valid is None:
        valid = np.isfinite(gc).astype(float)
    valid = np.asarray(valid) > 0.5

    # ---- per-bead signal track (biological state): replaces H3K27ac, keeps GC / mask / contacts ----
    sig_label, sig_placeholder = "H3K27ac", placeholder
    if track is not None:
        tr_label, tr_name, tr_bytes = track
        try:
            arr = states.load_track(tr_bytes)
            w = _window_tracks(arr, n_full, bin0, n)
            if w is None:
                notes.append(f"{tr_name}: {len(arr):,} values match neither the {n:,} beads nor {ch.name} at "
                             f"{ch.resolution:,} bp ({n_full:,} bins); signal track ignored.")
            else:
                epi = np.where(np.isfinite(w), w, np.nan)
                sig_label, sig_placeholder = f"signal ({tr_name})", False
                t_label = f"{t_label} · signal from {tr_label}"
        except Exception as exc:  # a broken track never takes the structure down with it
            notes.append(f"{tr_name}: signal track unreadable ({exc}); ignored.")
    gc = np.where(valid, gc, np.nan)
    epi = np.where(valid, epi, np.nan)
    cond = condition or cond

    key = digest(chrom_name, seed, b0, source[2] if source else b"ref", unit, graph[2] if graph else b"", trusted,
                 track[2] if track else b"", condition)
    return Dataset(key, ch, int(bin0), np.asarray(frames, float), labels, np.asarray(times, float), t_unit, cond,
                   np.asarray(gc, float), np.asarray(epi, float), valid,
                   np.asarray(ci, np.int64), np.asarray(cj, np.int64), np.asarray(cm, float),
                   truth, s_label, t_label, is_ref, placeholder, tuple(notes), sig_label, sig_placeholder)


# ======================================================================================
# HTML helpers
# ======================================================================================
def html(markup: str) -> None:
    st.markdown(markup, unsafe_allow_html=True)


def readout(rows: list[tuple[str, str, str]]) -> None:
    """Definition list: label (sans), value (mono), unit (sans)."""
    body = "".join(f"<div><dt>{label}</dt><dd>{value}<em>{unit}</em></dd></div>" for label, value, unit in rows)
    html(f'<dl class="cc-readout">{body}</dl>')


def fmt(x, nd: int = 1) -> str:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return "—"
    return "—" if not math.isfinite(x) else f"{x:,.{nd}f}"


def banner(text: str, kind: str = "warn") -> None:
    html(f'<div class="cc-banner {kind}">{text}</div>')


def esc(text) -> str:
    """HTML-escape user- or file-derived text before it goes into unsafe_allow_html markup."""
    import html as _html
    return _html.escape(str(text), quote=True)


def warning_card(title: str, detail: str = "", items: list[str] | None = None, kind: str = "warn") -> None:
    """A friendly, non-blocking problem card (file missing, corrupted, mismatched). Text is escaped."""
    body = f"<b>{esc(title)}</b>"
    if detail:
        body += f"<br>{esc(detail)}"
    if items:
        body += "<ul>" + "".join(f"<li>{esc(i)}</li>" for i in items[:8]) + "</ul>"
        if len(items) > 8:
            body += f"<span>… and {len(items) - 8} more</span>"
    html(f'<div class="cc-banner cc-card-warn {kind}">{body}</div>')
