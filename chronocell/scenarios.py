"""
4D structures: frames of 3D conformations over time or disease state.

Two sources feed the 4D workspace:

1. Measured / computed trajectories: StructureBundle files with T frames (e.g. Colab GPU fits of
   healthy vs tumour Micro-C, or time courses). These are shown as provided.
2. Simulated structural-variant scenarios, used as a *reference until real coordinates are
   provided*. A karyotypic change (deletion, duplication, inversion, translocation) is applied to
   the reference conformation and the polymer is relaxed (bond lengths + excluded volume). The
   frames show the geometric consequence of the rearrangement under the polymer model.
   They are hypotheses, not measured disease conformations; the time axis is relaxation sweeps.

Disease presets are anchored on UCSC hg38 gene coordinates (chronocell/data/hg38.json).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import genome, physics, synthetic


@dataclass
class Trajectory:
    """T frames of one (possibly rearranged) chromosome.

    bead_bin[k]    bin of bead k in its source chromosome
    bead_chrom[k]  source chromosome of bead k (partner chromosome for translocated segments)
    bead_kind[k]   0 native, 1 duplicated copy, 2 translocation partner, 3 inverted
    """
    frames: np.ndarray
    times: np.ndarray
    labels: list[str]
    bead_bin: np.ndarray
    bead_chrom: np.ndarray
    bead_kind: np.ndarray
    title: str
    description: str
    simulated: bool
    time_unit: str = "relaxation sweeps"
    events: list[str] = field(default_factory=list)

    @property
    def n_frames(self) -> int:
        return self.frames.shape[0]


# ----------------------------------------------------------------------------------------
# Geometry helpers
# ----------------------------------------------------------------------------------------
def clamp_region(a: int, b: int, n: int, min_len: int = 2) -> tuple[int, int]:
    """Clamp a half-open bead interval into [0, n) with at least `min_len` beads."""
    n = int(n)
    min_len = max(1, min(int(min_len), n))
    a, b = int(a), int(b)
    if b < a:
        a, b = b, a
    a = max(0, min(a, n - min_len))
    b = max(a + min_len, min(b, n))
    return a, b


def _outward(x: np.ndarray, at: int) -> np.ndarray:
    v = x[at] - x.mean(axis=0)
    norm = np.linalg.norm(v)
    return v / norm if norm > 1e-9 else np.array([1.0, 0.0, 0.0])


def relax_frames(x0: np.ndarray, b0: float, n_frames: int = 24, sweeps: int = 15) -> np.ndarray:
    """Frames of position-based relaxation (bond lengths -> b0, excluded volume d_min)."""
    frames = [x0.copy()]
    x = x0.copy()
    d_min = physics.D_MIN_FACTOR * b0
    for _ in range(n_frames - 1):
        x = synthetic.relax(x, b0=b0, d_min=d_min, iters=sweeps)
        frames.append(x.copy())
    return np.stack(frames)


def align_frames(frames: np.ndarray) -> np.ndarray:
    """Kabsch-align every frame onto frame 0 (proper rotations only: frames share handedness)."""
    out = frames.copy()
    for t in range(1, len(frames)):
        _, out[t], _ = physics.kabsch_rmsd(frames[0], frames[t], allow_reflection=False)
    return out


# ----------------------------------------------------------------------------------------
# Structural-variant operators: reference (N,3) -> rearranged starting conformation
# ----------------------------------------------------------------------------------------
def _base(n: int, chrom: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return np.arange(n), np.full(n, chrom, dtype=object), np.zeros(n, dtype=np.int8)


def deletion(x: np.ndarray, chrom: str, a: int, b: int):
    n = len(x)
    a, b = clamp_region(a, b, n, 1)
    keep = np.r_[0:a, b:n]
    bins, chroms, kinds = _base(n, chrom)
    return x[keep].copy(), bins[keep], chroms[keep], kinds[keep], f"deleted bins {a:,}–{b - 1:,} ({b - a:,} beads)"


def duplication(x: np.ndarray, chrom: str, a: int, b: int, copies: int, b0: float):
    """Tandem copies of [a, b) inserted after b-1, first placed beside the original segment."""
    n = len(x)
    a, b = clamp_region(a, b, n, 1)
    seg = x[a:b]
    bins, chroms, kinds = _base(n, chrom)
    parts, pb, pc, pk = [x[:b]], [bins[:b]], [chroms[:b]], [kinds[:b]]
    out = _outward(x, (a + b) // 2)
    for c in range(1, copies + 1):
        parts.append(seg + out * (1.5 * b0 * c))
        pb.append(bins[a:b])
        pc.append(chroms[a:b])
        pk.append(np.ones(b - a, dtype=np.int8))
    parts.append(x[b:])
    pb.append(bins[b:])
    pc.append(chroms[b:])
    pk.append(kinds[b:])
    return (np.vstack(parts), np.concatenate(pb), np.concatenate(pc), np.concatenate(pk),
            f"{copies + 1} copies of bins {a:,}–{b - 1:,} ({(b - a) * copies:,} beads inserted)")


def inversion(x: np.ndarray, chrom: str, a: int, b: int):
    n = len(x)
    a, b = clamp_region(a, b, n, 2)
    order = np.r_[0:a, np.arange(b - 1, a - 1, -1), b:n]
    bins, chroms, kinds = _base(n, chrom)
    kinds = kinds.copy()
    kinds[a:b] = 3
    # beads keep their positions; only chain connectivity changes, so the two junction bonds stretch
    return x[order].copy(), bins[order], chroms[order], kinds[order], f"inverted bins {a:,}–{b - 1:,}"


def translocation(x: np.ndarray, chrom: str, breakpoint: int, partner: genome.Chrom, partner_start_bin: int,
                  b0: float, seed: int = 11):
    """Derivative chromosome: this chromosome pter -> breakpoint, fused to partner bin -> qter.

    The partner segment's conformation is a synthetic fractal globule (its own measured
    coordinates are not part of this chromosome's data), placed outside the territory at the
    junction; relaxation then docks it through the fusion bond.
    """
    n = len(x)
    bp = int(np.clip(breakpoint, 2, n - 1))
    ps = int(np.clip(partner_start_bin, 0, partner.n_bins - 2))
    n_p = partner.n_bins - ps
    seg = synthetic.fractal_globule(n_p, b0=b0, seed=seed)
    out = _outward(x, bp - 1)
    seg = seg - seg[0] + x[bp - 1] + out * (b0 * (4.0 + 0.6 * n_p ** (1 / 3)))
    bins, chroms, kinds = _base(bp, chrom)
    return (np.vstack([x[:bp], seg]),
            np.concatenate([bins, np.arange(ps, partner.n_bins)]),
            np.concatenate([chroms, np.full(n_p, partner.name, dtype=object)]),
            np.concatenate([kinds, np.full(n_p, 2, dtype=np.int8)]),
            f"der({chrom.removeprefix('chr')}): {chrom} bins 0–{bp - 1:,} fused to {partner.name} bins "
            f"{ps:,}–{partner.n_bins - 1:,} ({n_p:,} partner beads)")


# ----------------------------------------------------------------------------------------
# Disease presets (hg38 gene anchors from UCSC ncbiRefSeqSelect)
# ----------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Preset:
    key: str
    chrom: str
    title: str
    disease: str
    operation: str
    summary: str
    reference: str


PRESETS: dict[str, Preset] = {p.key: p for p in (
    Preset("22q11del", "chr22", "22q11.2 deletion (LCR22A–D)",
           "22q11.2 deletion syndrome · risk factor for early-onset Parkinson's disease and schizophrenia",
           "deletion",
           "Heterozygous ~2.1–2.5 Mb deletion spanning DGCR6…LZTR1 (LCR22A–D, approximate).",
           "Butcher et al., JAMA Neurol 2013; McDonald-McGinn et al., Nat Rev Dis Primers 2015"),
    Preset("ph", "chr22", "Philadelphia chromosome der(22) t(9;22)(q34;q11)",
           "Chronic myeloid leukaemia · Ph+ acute lymphoblastic leukaemia",
           "translocation",
           "chr22 pter→BCR fused to chr9 ABL1→qter (BCR–ABL1). Breakpoint placed mid-BCR (M-bcr approx.).",
           "Nowell & Hungerford, Science 1960; Rowley, Nature 1973"),
    Preset("ewing", "chr22", "Ewing sarcoma der(22) t(11;22)(q24;q12)",
           "Ewing sarcoma",
           "translocation",
           "chr22 pter→EWSR1 fused to chr11 FLI1→qter (EWSR1–FLI1). Breakpoints placed mid-gene (introns 7–10 / 5–9 approx.).",
           "Delattre et al., Nature 1992"),
    Preset("snca3", "chr4", "SNCA locus triplication (PARK4)",
           "Familial Parkinson's disease (autosomal dominant)",
           "duplication",
           "Tandem triplication of ~1.1 Mb around SNCA (illustrative span; reported events are 0.4–4 Mb).",
           "Singleton et al., Science 2003"),
)}


def preset_region(p: Preset, chrom: genome.Chrom) -> dict:
    """Bin coordinates of a preset on `chrom` (its resolution), from gene anchors."""
    if p.key == "22q11del":
        a = genome.gene("DGCR6").start
        b = genome.gene("LZTR1").end
        return {"a": a // chrom.resolution, "b": -(-b // chrom.resolution)}
    if p.key == "ph":
        g, partner = genome.gene("BCR"), genome.gene("ABL1")
        return {"breakpoint": ((g.start + g.end) // 2) // chrom.resolution, "partner": "chr9",
                "partner_start_bp": partner.start}
    if p.key == "ewing":
        g, partner = genome.gene("EWSR1"), genome.gene("FLI1")
        return {"breakpoint": ((g.start + g.end) // 2) // chrom.resolution, "partner": "chr11",
                "partner_start_bp": (partner.start + partner.end) // 2}
    if p.key == "snca3":
        g = genome.gene("SNCA")
        return {"a": (g.start - 500_000) // chrom.resolution, "b": -(-(g.end + 500_000) // chrom.resolution),
                "copies": 2}
    raise KeyError(p.key)


def simulate(x: np.ndarray, chrom: genome.Chrom, operation: str, params: dict, b0: float,
             title: str, description: str, n_frames: int = 24, sweeps: int = 15) -> Trajectory:
    """Apply one structural variant to the reference conformation and relax it into frames."""
    if operation == "deletion":
        x1, bins, chroms, kinds, event = deletion(x, chrom.name, params["a"], params["b"])
    elif operation == "duplication":
        x1, bins, chroms, kinds, event = duplication(x, chrom.name, params["a"], params["b"],
                                                     int(params.get("copies", 1)), b0)
    elif operation == "inversion":
        x1, bins, chroms, kinds, event = inversion(x, chrom.name, params["a"], params["b"])
    elif operation == "translocation":
        partner = genome.chrom(params["partner"], chrom.resolution)
        x1, bins, chroms, kinds, event = translocation(
            x, chrom.name, params["breakpoint"], partner, int(params["partner_start_bp"]) // chrom.resolution, b0)
    else:
        raise ValueError(f"Unknown operation '{operation}'.")
    frames = align_frames(relax_frames(x1, b0, n_frames, sweeps))
    return Trajectory(frames, np.arange(n_frames) * sweeps, [f"{k * sweeps} sweeps" for k in range(n_frames)],
                      bins, chroms, kinds, title, description, simulated=True, events=[event])


def from_bundle(frames: np.ndarray, times: np.ndarray, labels: list[str], chrom: genome.Chrom, start_bin: int,
                title: str, description: str, time_unit: str) -> Trajectory:
    n = frames.shape[1]
    return Trajectory(align_frames(frames), np.asarray(times, float), list(labels),
                      np.arange(start_bin, start_bin + n), np.full(n, chrom.name, dtype=object),
                      np.zeros(n, dtype=np.int8), title, description, simulated=False, time_unit=time_unit)


def frame_metrics(traj: Trajectory) -> dict[str, np.ndarray]:
    """Per-frame Rg, end-to-end distance, RMSD to frame 0 and per-bead displacement from frame 0."""
    f = traj.frames
    rg = np.array([physics.radius_of_gyration(x) for x in f])
    re = np.array([physics.end_to_end(x) for x in f])
    disp = np.linalg.norm(f - f[0], axis=2)
    rmsd = np.sqrt(np.mean(disp ** 2, axis=1))
    bonds = np.array([np.median(physics.bond_lengths(x)) for x in f])
    return {"rg": rg, "re": re, "rmsd": rmsd, "disp": disp, "median_bond": bonds}
