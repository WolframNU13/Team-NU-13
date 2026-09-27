"""
GPU pipeline for Google Colab (T4): raw data -> graph -> 3D/4D coordinates -> workstation bundle.

Used by colab/ChronoCell5D_Colab.ipynb; every function also runs on CPU.

A *condition* is one biological state (healthy, tumour, patient-derived line, ...). Each condition
may have several *frames* (time points, cell-cycle phases, treatment doses): one Micro-C graph per
frame. Frames of a condition are fitted in order with a warm start from the previous frame, then
superimposed, so the 4th dimension is continuous rather than a set of arbitrarily rotated fits.
Output layout (unzip into the repository):  coordinates/<chrom>/<condition>.npz  (+ graph.npz)
"""

from __future__ import annotations

import os
import time
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import egnn, formats, genome, physics, scenarios, synthetic

UCSC_FASTA = "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/chromosomes/{chrom}.fa.gz"


def environment() -> dict:
    import torch
    info = {"torch": torch.__version__, "cuda": torch.cuda.is_available()}
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        info |= {"gpu": p.name, "memory_gb": round(p.total_memory / 1e9, 1)}
    return info


def download_fasta(chrom: str, dest_dir: str) -> str:
    """UCSC hg38 per-chromosome FASTA (gzipped); returns the decompressed path."""
    import gzip
    import shutil
    os.makedirs(dest_dir, exist_ok=True)
    gz = os.path.join(dest_dir, f"{chrom}.fa.gz")
    out = gz[:-3]
    if not os.path.exists(out):
        urllib.request.urlretrieve(UCSC_FASTA.format(chrom=chrom), gz)
        with gzip.open(gz, "rb") as src, open(out, "wb") as dst:
            shutil.copyfileobj(src, dst)
    return out


@dataclass
class Frame:
    """One time point of one condition: its input files (or a prepared graph)."""
    label: str
    time: float
    mcool: str | None = None
    bigwig: str | None = None
    graph_npz: str | None = None


@dataclass
class Condition:
    name: str
    frames: list[Frame]
    time_unit: str = "hours"


@dataclass
class GraphArrays:
    gc: np.ndarray
    epi: np.ndarray
    valid: np.ndarray
    ci: np.ndarray
    cj: np.ndarray
    cm: np.ndarray


def build_graph(chrom: str, resolution: int, fasta: str, frame: Frame) -> GraphArrays:
    if frame.graph_npz:
        with open(frame.graph_npz, "rb") as fh:
            g = formats.read_graph(fh.read(), frame.graph_npz)
        return GraphArrays(g.gc, g.epi, g.valid, g.ci, g.cj, g.cm)
    from .build_graph import from_files
    return GraphArrays(*from_files(fasta, frame.bigwig, frame.mcool, chrom=chrom, resolution=resolution))


@dataclass
class ConditionResult:
    condition: str
    frames_nm: np.ndarray
    times: np.ndarray
    labels: list[str]
    seconds: list[float]
    final_contact_loss: list[float]
    graph: GraphArrays
    notes: list[str] = field(default_factory=list)


def reconstruct_condition(chrom: genome.Chrom, graphs: list[GraphArrays], cond: Condition,
                          cfg: egnn.FitConfig | None = None, warm_cfg: egnn.FitConfig | None = None,
                          log=print) -> ConditionResult:
    """Fit every frame of a condition on the GPU (warm-started), then superimpose the frames."""
    cfg = cfg or egnn.FitConfig()
    warm_cfg = warm_cfg or egnn.FitConfig(prefit_epochs=300, phantom_fraction=0.0, refine_epochs=cfg.refine_epochs,
                                          lr_prefit=0.02, device=cfg.device)
    b0 = physics.bond_length_for(chrom.resolution)
    out, secs, losses, prev = [], [], [], None
    for g, fr in zip(graphs, cond.frames):
        if len(g.gc) != chrom.n_bins:
            raise ValueError(f"{cond.name}/{fr.label}: graph has {len(g.gc):,} bins, {chrom.name} at "
                             f"{chrom.resolution:,} bp has {chrom.n_bins:,}.")
        feats = egnn.node_features(g.gc, g.epi, g.valid)
        use = warm_cfg if prev is not None else cfg
        t0 = time.time()
        res = egnn.fit_structure(chrom.n_bins, feats, g.ci, g.cj, g.cm, use, b0=b0, init_nm=prev)
        secs.append(time.time() - t0)
        losses.append(res.history["contact"][-1])
        log(f"  {cond.name} · {fr.label}: {secs[-1]:.1f}s on {res.config['device_used']}, "
            f"L_contact {losses[-1]:.4f}")
        out.append(res.coords_nm)
        prev = res.coords_nm
    frames = scenarios.align_frames(np.stack(out))
    return ConditionResult(cond.name, frames, np.array([f.time for f in cond.frames], float),
                           [f.label for f in cond.frames], secs, losses, graphs[-1])


def write_outputs(chrom: genome.Chrom, results: list[ConditionResult], out_root: str, source: str) -> str:
    """coordinates/<chrom>/<condition>.npz (+ graph.npz from the first condition); returns a zip path."""
    d = Path(out_root) / "coordinates" / chrom.name
    d.mkdir(parents=True, exist_ok=True)
    for r in results:
        formats.write_bundle(str(d / f"{r.condition}.npz"), formats.StructureBundle(
            chrom=chrom.name, resolution=chrom.resolution, frames=r.frames_nm, times=r.times, labels=r.labels,
            condition=r.condition, source=source, time_unit="hours",
            gc=r.graph.gc, epi=r.graph.epi, valid=r.graph.valid, ci=r.graph.ci, cj=r.graph.cj, cm=r.graph.cm))
    g = results[0].graph
    formats.write_graph_npz(str(d / "graph.npz"), g.gc, g.epi, g.valid, g.ci, g.cj, g.cm,
                            chrom=chrom.name, resolution=chrom.resolution)
    zpath = str(Path(out_root) / f"chronocell_{chrom.name}_coordinates.zip")
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for p in d.iterdir():
            z.write(p, arcname=str(p.relative_to(out_root)))
    return zpath


# ----------------------------------------------------------------------------------------
# Demo inputs (no downloads): a reference state and a rearranged state with simulated Micro-C
# ----------------------------------------------------------------------------------------
def demo_graphs(chrom: genome.Chrom, n_frames: int = 3, seed: int = 7) -> dict[str, list[GraphArrays]]:
    """Synthetic 'healthy' and 'rearranged' conditions, each a short time course, for dry runs.

    The rearranged condition applies the chromosome's first disease preset (or a 2 Mb inversion)
    and re-simulates contacts from the relaxed structure. Labelled synthetic everywhere.
    """
    b0 = physics.bond_length_for(chrom.resolution)
    ref = synthetic.build(chrom, seed=seed, b0=b0)
    presets = [p for p in scenarios.PRESETS.values() if p.chrom == chrom.name and p.operation != "translocation"]
    if presets:
        p = presets[0]
        op, params = p.operation, scenarios.preset_region(p, chrom)
    else:
        mid = chrom.n_bins // 2
        op, params = "inversion", {"a": mid, "b": mid + 2_000_000 // chrom.resolution}
    graphs: dict[str, list[GraphArrays]] = {"healthy": [], "rearranged": []}
    rng = np.random.default_rng(seed)
    for t in range(n_frames):
        wobble = ref.coords + rng.normal(scale=0.15 * b0 * (t + 1), size=ref.coords.shape)
        x_h = synthetic.relax(wobble, b0, physics.D_MIN_FACTOR * b0, iters=60)
        ci, cj, cm = synthetic.simulate_contacts(x_h, ref.valid, b0=b0, seed=seed + t)
        graphs["healthy"].append(GraphArrays(ref.gc, ref.epi, ref.valid, ci, cj, cm))
        if op in ("deletion", "duplication"):
            # bin count changes; keep the reference grid and remove / keep contacts accordingly
            a, b = params["a"], params["b"]
            drop = ((ci >= a) & (ci < b)) | ((cj >= a) & (cj < b)) if op == "deletion" else np.zeros(ci.size, bool)
            cm2 = cm.copy()
            if op == "duplication":
                both = (ci >= a) & (ci < b) & (cj >= a) & (cj < b)
                cm2[both] = cm2[both] * (1 + params.get("copies", 1))      # dosage: more copies, more reads
            valid2 = ref.valid.copy()
            if op == "deletion":
                valid2[a:b] = False
            graphs["rearranged"].append(GraphArrays(ref.gc, ref.epi, valid2, ci[~drop], cj[~drop], cm2[~drop]))
        else:
            tr = scenarios.simulate(x_h, chrom, op, params, b0, "demo", "demo", n_frames=6, sweeps=20)
            x_r = tr.frames[-1][np.argsort(tr.bead_bin)]
            ci2, cj2, cm2 = synthetic.simulate_contacts(x_r, ref.valid, b0=b0, seed=seed + 100 + t)
            graphs["rearranged"].append(GraphArrays(ref.gc, ref.epi, ref.valid, ci2, cj2, cm2))
    return graphs
