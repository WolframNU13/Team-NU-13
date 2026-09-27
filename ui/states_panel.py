"""
Sidebar: biological-state selector backed by the format-sniffing data engine.

Files are discovered in `coordinates/` (recursively; any folder or file name that mentions a
state, see `chronocell/states.py`) and from uploads made here into the selected state. Each
file is classified by content: (N, 3) coordinates, (T, N, 3) frames, (N,) signal tracks,
wwPDB structures. The selected state's structure (and its best-matching track) then drives the
whole workstation: 3D model, metrics, signal colouring, genomic-feature charts, 4D comparisons
and ChronoAgent.
"""

from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass, field

import streamlit as st

from chronocell import demo_states, states as S
from ui.common import CACHE_DIR, SLOT_ROOT, esc, html

UPLOAD_TYPES = ["npy", "pdb", "npz", "xyz", "csv", "tsv", "txt", "bed", "bedgraph", "bdg", "bw", "bigwig", "cool",
                "mcool", "hic"]
KIND_TAG = {S.COORDS: "coords", S.FRAMES: "frames", S.TRACK: "track", S.GRAPH: "graph", S.UNSUPPORTED: "skipped",
            S.CONTACTS: "contacts", S.EXPRESSION: "RNA-seq"}
DEMO_ROOT = CACHE_DIR / "demo_states"


@st.cache_resource(show_spinner="Creating the demo patients (synthetic; first time only)…")
def _ensure_demo() -> str:
    """Synthetic Healthy / Disease / Senescent chr22 files in a hidden cache folder (never in coordinates/)."""
    marker = DEMO_ROOT / "chr22" / "senescent" / "synthetic_demo_senescent.pdb"
    if not marker.exists():
        demo_states.write_demo(DEMO_ROOT)
    return str(DEMO_ROOT)
ss = st.session_state


@dataclass
class BioSelection:
    state: str
    drive: bool
    files: tuple[S.StateFile, ...]
    plans: dict[str, S.StatePlan]
    structure: S.StateFile | None
    track: S.StateFile | None
    n_full: int | None = None
    unassigned: tuple[S.StateFile, ...] = field(default_factory=tuple)

    @property
    def has_structure(self) -> bool:
        return self.structure is not None

    @property
    def track_only(self) -> bool:
        return self.structure is None and self.track is not None

    def track_for(self, sf: S.StateFile) -> S.StateFile | None:
        """The selected track for the selected structure; the best-matching track for any other."""
        if self.structure is not None and sf.key == self.structure.key:
            return self.track
        return S.best_track(sf, self.plans[sf.state].tracks, self.n_full) if sf.state in self.plans else None


# ======================================================================================
# File access
# ======================================================================================
@st.cache_resource(show_spinner=False, max_entries=16)
def _scan(root: str, signature: tuple) -> list[S.StateFile]:
    return S.scan_folder(root, signature)


@st.cache_data(show_spinner=False, max_entries=64)
def _read(path: str, mtime: float, size: int) -> bytes:
    with open(path, "rb") as fh:
        return fh.read()


def file_bytes(sf: S.StateFile) -> bytes:
    """Raw bytes of a state file (folder files re-read only when they change)."""
    if sf.origin == "upload":
        return ss.state_uploads[sf.state][sf.name][0]
    st_ = os.stat(sf.path)
    return _read(sf.path, st_.st_mtime, st_.st_size)


def as_source(sf: S.StateFile) -> tuple[str, str, bytes]:
    """(label, name, bytes) in the form ui.common.load_dataset expects."""
    return f"{sf.state} · {sf.name}", os.path.basename(sf.name), file_bytes(sf)


# ======================================================================================
# Uploads (kept per state across reruns; the uploader itself is cleared after each drop)
# ======================================================================================
def _ingest(state: str, widget_key: str) -> None:
    for f in ss.get(widget_key) or []:
        data = f.getvalue()
        ss.state_uploads[state][f.name] = (data, S.sniff_upload(state, f.name, data))
    ss.state_up_nonce += 1


def _clear(state: str) -> None:
    ss.state_uploads[state] = {}


# ======================================================================================
# Sidebar
# ======================================================================================
def _file_line(sf: S.StateFile, used: set[str]) -> str:
    tag = KIND_TAG[sf.kind]
    cls = "ok" if sf.key in used else ("warn" if sf.kind == S.UNSUPPORTED else "")
    origin = "upload" if sf.origin == "upload" else "folder"
    return (f'<li><span class="cc-tag {cls}">{tag}</span> <code>{esc(sf.name)}</code>'
            f'<small>{esc(sf.detail)} · {origin}</small></li>')


def sidebar(chrom_name: str, n_full: int | None) -> BioSelection:
    ss.setdefault("state_uploads", {s: {} for s in S.STATES})
    ss.setdefault("state_up_nonce", 0)
    with st.sidebar:
        html('<p class="cc-side-title">Biological state</p>')
        demo = st.toggle("Load demo patients (synthetic)", key="demo_patients",
                         help="Adds a synthetic healthy, cancer and senescent chr22 so every page can be explored "
                              "without data. They are labelled 'demo' everywhere and are not real patients.")
        state = st.selectbox("Biological state", S.STATES, key="bio_state", label_visibility="collapsed",
                             help="Files are matched to a state by content (format) plus the words in their path, "
                                  "e.g. coordinates/chr22/healthy/…, tumour_K562.npy, Senescent/IMR90.pdb.")
        try:
            folder = S.for_chromosome(_scan(str(SLOT_ROOT), S.folder_signature(SLOT_ROOT)), chrom_name)
        except OSError as exc:  # unreadable folder: keep going with uploads only
            folder = []
            html(f'<div class="cc-banner">Could not scan <code>coordinates/</code>: {esc(exc)}</div>')
        if demo:
            try:
                root = _ensure_demo()
                demo_files = [dataclasses.replace(f, key=f"demo:{f.name}", name=f"demo/{f.name}")
                              for f in S.scan_folder(root)]
                folder += S.for_chromosome(demo_files, chrom_name)
                if chrom_name != "chr22":
                    html('<p class="cc-note">The demo patients are chr22: switch the chromosome to chr22 to use them.</p>')
            except Exception as exc:  # demo generation must never break the app
                html(f'<div class="cc-banner">Demo patients unavailable: {esc(exc)}</div>')
        uploads = [meta for s in S.STATES for _, meta in ss.state_uploads[s].values()]
        files = tuple(S.for_chromosome(uploads, chrom_name) + folder)
        plans = {s: S.plan(list(files), s) for s in S.STATES}
        p = plans[state]

        structure = track = None
        if p.structures:
            opts = list(p.structures)
            i = st.selectbox("Structure file", range(len(opts)), key=f"bio_struct_{state}_{chrom_name}",
                             format_func=lambda k: f"{opts[k].name} · {opts[k].n:,} beads",
                             disabled=len(opts) == 1)
            structure = opts[min(int(i or 0), len(opts) - 1)]
        if p.tracks:
            default = S.best_track(structure, p.tracks, n_full)
            opts_t = [None] + list(p.tracks)
            j = st.selectbox("Epigenomic track", range(len(opts_t)),
                             index=opts_t.index(default) if default in opts_t else 0,
                             key=f"bio_track_{state}_{chrom_name}_{structure.key if structure else ''}",
                             format_func=lambda k: "None" if opts_t[k] is None else f"{opts_t[k].name} · {opts_t[k].n:,}")
            track = opts_t[min(int(j or 0), len(opts_t) - 1)]
            if structure is not None and track is not None and track.n not in (structure.n, n_full):
                html(f'<p class="cc-note">This track has {track.n:,} values; the structure has {structure.n:,} beads. '
                     'It will be ignored unless it covers the whole chromosome.</p>')

        drive = st.toggle("Show this state in the workstation", value=True, key="bio_drive",
                          help="On: the selected state's structure and track replace the Structure source. "
                               "Off: pick any source manually; states remain available for comparison.")
        if p.empty:
            html(f'<div class="cc-banner">No files for <b>{esc(state)}</b> on {esc(chrom_name)} yet. '
                 f'The workstation keeps its current source. Add .npy (N×3 coordinates or N-value track) or .pdb '
                 f'files below, or into <code>coordinates/{esc(chrom_name)}/{esc(S.SHORT_NAMES[state].lower())}/</code>.</div>')
        elif not p.structures:
            html('<p class="cc-note">Only a signal track is available for this state: it is applied to the current '
                 'structure when the lengths match.</p>')

        used = {f.key for f in (structure, track) if f is not None}
        mine = list(p.structures) + list(p.tracks) + list(p.contacts) + list(p.expression) + list(p.other)
        if mine:
            html('<ul class="cc-files">' + "".join(_file_line(f, used) for f in mine) + "</ul>")

        wkey = f"state_up_{ss.state_up_nonce}"
        st.file_uploader(f"Add files to {S.SHORT_NAMES[state]}", type=UPLOAD_TYPES, accept_multiple_files=True,
                         key=wkey, on_change=_ingest, args=(state, wkey),
                         help="Classified by content: (N, 3) or (T, N, 3) arrays and .pdb are structures; (N,) arrays, "
                              ".bedGraph, .bed and .bigWig are signal tracks; .cool / .mcool / .hic or a bin-bin-count "
                              "table are contact maps; a gene + value table is RNA-seq. Pickled arrays are never loaded.")
        if ss.state_uploads[state]:
            st.button(f"Remove {len(ss.state_uploads[state])} uploaded file(s)", key=f"bio_clear_{state}",
                      on_click=_clear, args=(state,), width="stretch")
        counts = " · ".join(f"{S.SHORT_NAMES[s]} {len(plans[s].structures)}/{len(plans[s].tracks)}" for s in S.STATES)
        unassigned = tuple(f for f in folder if f.state is None and (f.is_structure or f.is_track)
                           and os.path.dirname(f.name) not in ("", chrom_name))
        html(f'<p class="cc-note">Structures / tracks per state: {counts}.'
             + (f' {len(unassigned)} file(s) in sub-folders name no state and are not assigned.' if unassigned else "")
             + '</p>')
    return BioSelection(state, bool(drive), files, plans, structure, track, n_full, unassigned)
