"""
E(3)-equivariant graph neural network for chromatin reconstruction (PyTorch).

Layer (Satorras, Hoogeboom & Welling, ICML 2021), with the stabilised coordinate update:

    d_ij   = sqrt(||x_i - x_j||^2 + eps^2)                        smooth at d -> 0
    m_ij   = phi_e( h_i, h_j, RBF(d_ij / b0), a_ij )               invariant message
    x_i'   = x_i + (1/|N(i)|) sum_j  (x_i - x_j)/(d_ij + eps) * s * tanh(phi_x(m_ij))
    h_i'   = LayerNorm( h_i + phi_h( h_i, (1/|N(i)|) sum_j m_ij ) )

Why each change matters
* Unit direction (x_i - x_j)/(d_ij + eps): the raw difference makes the step proportional to
  distance, so distant pairs (hundreds of nm) produce huge jumps. The unit vector has norm < 1.
* Mean aggregation 1/|N(i)|: Micro-C hubs have hundreds of contacts; a plain sum scales the
  update with degree (Satorras et al. use C = 1/(M-1) for the same reason).
* s * tanh(.): bounds each layer's displacement to s bead diameters.
* RBF(d/b0) instead of raw d^2: d^2 spans ~10^0-10^3 in b0^2 units and saturates the MLP.
* sqrt(. + eps^2): the gradient of ||v|| is undefined at v = 0 (coincident beads).

Symmetry. Only relative vectors and distances enter, so the layer is equivariant to all of
E(3): translations, rotations *and reflections*. Contact data cannot fix chirality, so a
reconstruction is determined only up to a mirror image; see physics.kabsch_rmsd.

All coordinates inside this module are in units of b0 (dimensionless).
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Callable

import numpy as np
import torch
import torch.nn as nn

from . import physics

EPS = 1e-6


# ----------------------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------------------
class RadialBasis(nn.Module):
    """Gaussian radial basis on d (in b0 units): bounded, smooth, invariant edge embedding."""

    def __init__(self, n: int = 16, cutoff: float = 8.0):
        super().__init__()
        centers = torch.linspace(0.0, cutoff, n)
        self.register_buffer("centers", centers)
        self.gamma = float(1.0 / (centers[1] - centers[0]) ** 2)

    def forward(self, d: torch.Tensor) -> torch.Tensor:
        return torch.exp(-self.gamma * (d - self.centers) ** 2)


class EGNNLayer(nn.Module):
    def __init__(self, hidden: int, edge_dim: int, n_rbf: int = 16, max_step: float = 0.25):
        super().__init__()
        self.rbf = RadialBasis(n_rbf)
        self.phi_e = nn.Sequential(nn.Linear(2 * hidden + n_rbf + edge_dim, hidden), nn.SiLU(),
                                   nn.Linear(hidden, hidden), nn.SiLU())
        self.phi_x = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1))
        self.phi_h = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.norm = nn.LayerNorm(hidden)
        self.max_step = max_step
        nn.init.uniform_(self.phi_x[-1].weight, -1e-3, 1e-3)   # start near the identity map
        nn.init.zeros_(self.phi_x[-1].bias)

    def forward(self, h: torch.Tensor, x: torch.Tensor, g: "MessageGraph") -> tuple[torch.Tensor, torch.Tensor]:
        diff = x[g.dst] - x[g.src]                                            # x_i - x_j
        d = torch.sqrt((diff * diff).sum(-1, keepdim=True) + EPS ** 2)
        m = self.phi_e(torch.cat([h[g.dst], h[g.src], self.rbf(d), g.edge_attr], dim=-1))
        w = self.max_step * torch.tanh(self.phi_x(m))
        dx = torch.zeros_like(x).index_add(0, g.dst, diff / (d + EPS) * w) * g.inv_deg
        agg = torch.zeros_like(h).index_add(0, g.dst, m) * g.inv_deg
        h = self.norm(h + self.phi_h(torch.cat([h, agg], dim=-1)))
        return h, x + dx


class ChromatinEGNN(nn.Module):
    def __init__(self, node_dim: int = 3, edge_dim: int = 3, hidden: int = 32, n_layers: int = 3,
                 max_step: float = 0.25):
        super().__init__()
        self.embed = nn.Linear(node_dim, hidden)
        self.layers = nn.ModuleList(EGNNLayer(hidden, edge_dim, max_step=max_step) for _ in range(n_layers))

    def forward(self, x: torch.Tensor, node_feat: torch.Tensor, g: "MessageGraph") -> tuple[torch.Tensor, torch.Tensor]:
        h = self.embed(node_feat)
        for layer in self.layers:
            h, x = layer(h, x, g)
        return x, h


# ----------------------------------------------------------------------------------------
# Graphs
# ----------------------------------------------------------------------------------------
@dataclass
class MessageGraph:
    src: torch.Tensor
    dst: torch.Tensor
    edge_attr: torch.Tensor
    inv_deg: torch.Tensor

    def to(self, dtype: torch.dtype) -> "MessageGraph":
        return MessageGraph(self.src, self.dst, self.edge_attr.to(dtype), self.inv_deg.to(dtype))

    def to_device(self, device: torch.device) -> "MessageGraph":
        return MessageGraph(self.src.to(device), self.dst.to(device), self.edge_attr.to(device), self.inv_deg.to(device))


def build_message_graph(n: int, ci: np.ndarray, cj: np.ndarray, cm: np.ndarray, k_top: int = 8,
                        dtype: torch.dtype = torch.float32) -> MessageGraph:
    """Sparse, symmetric message-passing graph.

    Backbone (|i-j| in {1, 2}) plus each node's k strongest contacts. Passing messages over all
    Micro-C pixels would cost O(|contacts| x hidden) memory per layer; supervision still uses
    every contact in the loss, which only needs a distance per pair.
    Edge attributes: [log1p(M)/log1p(M_max), log1p(|i-j|)/log1p(N), backbone flag].
    """
    ci, cj, cm = np.asarray(ci, np.int64), np.asarray(cj, np.int64), np.asarray(cm, np.float64)
    long = np.abs(cj - ci) > 2
    a = np.concatenate([ci[long], cj[long]])
    b = np.concatenate([cj[long], ci[long]])
    w = np.concatenate([cm[long], cm[long]])
    order = np.lexsort((-w, a))                               # group by a, strongest first
    a, b, w = a[order], b[order], w[order]
    first = np.r_[0, np.flatnonzero(np.diff(a)) + 1] if a.size else np.empty(0, np.int64)
    group_start = np.repeat(first, np.diff(np.r_[first, a.size])) if a.size else first
    rank = np.arange(a.size) - group_start
    keep = rank < k_top
    kn_i, kn_j, kn_w = a[keep], b[keep], w[keep]

    idx = np.arange(n)
    bb_i = np.concatenate([idx[:-1], idx[:-2]]) if n > 2 else idx[:-1]
    bb_j = np.concatenate([idx[1:], idx[2:]]) if n > 2 else idx[1:]
    # contact counts for backbone pairs (0 if unobserved), via sorted-key lookup
    bb_w = np.zeros(bb_i.size)
    if ci.size:
        lin = np.minimum(ci, cj) * n + np.maximum(ci, cj)
        order_lin = np.argsort(lin)
        sorted_lin = lin[order_lin]
        pos = np.minimum(np.searchsorted(sorted_lin, bb_i * n + bb_j), sorted_lin.size - 1)
        hit = sorted_lin[pos] == bb_i * n + bb_j
        bb_w[hit] = cm[order_lin[pos[hit]]]

    src = np.concatenate([bb_i, bb_j, kn_j])
    dst = np.concatenate([bb_j, bb_i, kn_i])
    wt = np.concatenate([bb_w, bb_w, kn_w])
    is_bb = np.concatenate([np.ones(2 * bb_i.size), np.zeros(kn_i.size)])
    # de-duplicate (a pair can be both top-k for i and for j)
    key = dst * n + src
    _, uniq = np.unique(key, return_index=True)
    src, dst, wt, is_bb = src[uniq], dst[uniq], wt[uniq], is_bb[uniq]

    m_max = max(float(cm.max()) if cm.size else 1.0, 1.0)
    attr = np.stack([np.log1p(wt) / np.log1p(m_max),
                     np.log1p(np.abs(dst - src)) / np.log1p(max(n, 2)),
                     is_bb], axis=1)
    deg = np.bincount(dst, minlength=n).astype(np.float64)
    return MessageGraph(
        src=torch.as_tensor(src, dtype=torch.long),
        dst=torch.as_tensor(dst, dtype=torch.long),
        edge_attr=torch.as_tensor(attr, dtype=dtype),
        inv_deg=torch.as_tensor(1.0 / np.maximum(deg, 1.0), dtype=dtype).unsqueeze(-1),
    )


def node_features(gc: np.ndarray, epi: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """x_i = [z(f_GC), z(log1p f_epi), assembled]; missing bins get 0 and flag 0.

    Standardising removes the scale mismatch between a fraction in [0, 1] and an unbounded
    ChIP signal; log1p tames the heavy tail of H3K27ac peaks.
    """
    valid = np.asarray(valid, bool) & np.isfinite(gc) & np.isfinite(epi)
    out = np.zeros((len(gc), 3), dtype=np.float32)
    if valid.any():
        g = gc[valid]
        e = np.log1p(np.clip(epi[valid], 0, None))
        out[valid, 0] = (g - g.mean()) / (g.std() + 1e-8)
        out[valid, 1] = (e - e.mean()) / (e.std() + 1e-8)
    out[:, 2] = valid
    return out


# ----------------------------------------------------------------------------------------
# Losses (differentiable twins of physics.loss_*; coordinates in b0 units)
# ----------------------------------------------------------------------------------------
def _dist(x: torch.Tensor, i: torch.Tensor, j: torch.Tensor) -> torch.Tensor:
    v = x[i] - x[j]
    return torch.sqrt((v * v).sum(-1) + EPS ** 2)


def contact_loss(x: torch.Tensor, ci: torch.Tensor, cj: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean(((_dist(x, ci, cj) - target) / target) ** 2)


def smooth_loss(x: torch.Tensor) -> torch.Tensor:
    v = x[1:] - x[:-1]
    return torch.mean((torch.sqrt((v * v).sum(-1) + EPS ** 2) - 1.0) ** 2)


def steric_loss(x: torch.Tensor, pi: torch.Tensor, pj: torch.Tensor, d_min: float) -> torch.Tensor:
    if pi.numel() == 0:
        return x.sum() * 0.0
    return torch.sum(torch.relu(d_min - _dist(x, pi, pj)) ** 2) / x.shape[0]


# ----------------------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------------------
@dataclass
class FitConfig:
    """Two-stage schedule validated on planted fractal globules (see AUDIT.md, section 5).

    Stage 1 - contact embedding. Free coordinates P, no network, started from the
    shortest-path MDS fold (`init="mds"`) or a random walk (`init="random_walk"`). The first
    `phantom_fraction` of epochs run without excluded volume (a phantom chain can pass through
    itself to fix the topology); the steric term then switches on.
    Stage 2 - EGNN refinement. X = EGNN(P | GC, H3K27ac) optimised jointly with P on the full
    objective. Set `refine_epochs = 0` to skip it.
    """
    init: str = "mds"
    prefit_epochs: int = 800
    phantom_fraction: float = 0.3
    refine_epochs: int = 100
    refine_with_egnn: bool = True    # False: same schedule on free coordinates (ablation control)
    lr_prefit: float = 0.05          # coordinates are in b0 units (use 0.2 with a random-walk start)
    lr_refine: float = 0.02
    lr_model: float = 2e-3
    weight_decay: float = 1e-4       # network weights only, never coordinates (would shrink P)
    lambda_smooth: float = 1.0
    lambda_steric: float = 20.0
    alpha: float = physics.ALPHA
    d_min: float = physics.D_MIN_FACTOR   # b0 units
    hidden: int = 32
    n_layers: int = 3
    k_top: int = 8
    neighbor_every: int = 10
    skin: float = 0.5
    grad_clip: float = 5.0
    seed: int = 0
    device: str = "auto"             # "auto" -> CUDA (e.g. Colab T4) when available, else CPU


@dataclass
class FitResult:
    coords_nm: np.ndarray
    history: dict[str, list[float]]
    n_params: int
    seconds: float
    config: dict = field(default_factory=dict)


def resolve_device(name: str = "auto") -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def random_walk(n: int, rng: np.random.Generator) -> np.ndarray:
    """Freely-jointed chain with unit bonds: a polymer-shaped, symmetry-neutral starting point."""
    steps = rng.normal(size=(n, 3))
    steps /= np.linalg.norm(steps, axis=1, keepdims=True)
    x = np.cumsum(steps, axis=0)
    return x - x.mean(axis=0)


def shortest_path_mds(n: int, ci: np.ndarray, cj: np.ndarray, target: np.ndarray, max_nodes: int = 1000,
                      device: torch.device | str = "cpu") -> np.ndarray:
    """Global-fold initialisation from graph distances (ShRec3D; Lesne et al., Nat Methods 2014).

    Contact targets (b0 units) and unit backbone bonds define a weighted graph; all-pairs shortest
    paths (Floyd-Warshall, in-place min-plus updates) approximate spatial distances, and classical
    MDS embeds them in 3D. Windows above `max_nodes` beads are coarse-grained into blocks of k
    consecutive beads (block distance = closest member contact) so the O(m^3) step stays bounded;
    bead positions are then interpolated along the chain between block centres.
    Random-walk starts fold only locally and trap the optimiser in misfolded minima.
    """
    k = max(1, int(np.ceil(n / max_nodes)))
    m = int(np.ceil(n / k))
    a, b = np.asarray(ci) // k, np.asarray(cj) // k
    keep = a != b
    dev = torch.device(device)
    dist = torch.full((m * m,), float("inf"), dtype=torch.float64, device=dev)
    w = torch.as_tensor(np.asarray(target, dtype=np.float64)[keep], device=dev)
    for u, v in ((a[keep], b[keep]), (b[keep], a[keep])):
        dist.scatter_reduce_(0, torch.as_tensor(u * m + v, device=dev), w, reduce="amin")
    # Backbone edges, plus a polymer prior only where data are missing: i <-> i+S (S = 2, 4, 8, ...)
    # at 1.5x the compact-globule distance b0 (S k)^(1/3), for pairs touching a block without any
    # contact. Without it, contact-free stretches (assembly gaps, unmappable repeats) have shortest
    # paths that grow linearly along the chain and MDS lays them out as micrometre-long rods; applied
    # everywhere it would override real contact geometry, so data-rich pairs never get prior edges.
    degree = np.bincount(np.concatenate([a[keep], b[keep]]), minlength=m)
    empty_block = degree == 0
    u = np.arange(m - 1)
    bb = torch.full((u.size,), float(k), dtype=torch.float64, device=dev)
    for p_, q_ in ((u, u + 1), (u + 1, u)):
        dist.scatter_reduce_(0, torch.as_tensor(p_ * m + q_, device=dev), bb, reduce="amin")
    s_step = 2
    while s_step < m and empty_block.any():
        u = np.arange(m - s_step)
        u = u[empty_block[u] | empty_block[u + s_step]]
        if u.size:
            prior = torch.full((u.size,), 1.5 * (s_step * k) ** (1.0 / 3.0), dtype=torch.float64, device=dev)
            for p_, q_ in ((u, u + s_step), (u + s_step, u)):
                dist.scatter_reduce_(0, torch.as_tensor(p_ * m + q_, device=dev), prior, reduce="amin")
        s_step *= 2
    dist = dist.view(m, m)
    dist.fill_diagonal_(0.0)
    for p in range(m):
        torch.minimum(dist, dist[:, p:p + 1] + dist[p:p + 1, :], out=dist)
    finite = torch.isfinite(dist)
    dist[~finite] = dist[finite].max() * 1.2                 # disconnected pieces: place far apart
    d2 = dist ** 2
    centred = d2 - d2.mean(0, keepdim=True) - d2.mean(1, keepdim=True) + d2.mean()
    evals, evecs = torch.linalg.eigh(-0.5 * centred)
    coarse = (evecs[:, -3:] * torch.sqrt(torch.clamp(evals[-3:], min=0.0))).cpu().numpy()
    if k == 1:
        return coarse
    centres = np.arange(m) * k + (k - 1) / 2
    beads = np.arange(n)
    return np.stack([np.interp(beads, centres, coarse[:, d]) for d in range(3)], axis=1)


def fit_structure(n: int, node_feat: np.ndarray, ci: np.ndarray, cj: np.ndarray, cm: np.ndarray,
                  cfg: FitConfig | None = None, b0: float = physics.B0_NM,
                  init_nm: np.ndarray | None = None,
                  progress: Callable[[str, int, int, dict[str, float]], None] | None = None) -> FitResult:
    """Minimise L_total = L_contact + lambda_smooth L_smooth + lambda_steric L_steric.

    The steric term uses a Verlet neighbour list (cutoff d_min + skin, rebuilt every
    `neighbor_every` epochs with the O(N) cell list), so memory never scales as N^2.
    `progress(stage, epoch, total_epochs, row)` is called once per epoch.
    """
    cfg = cfg or FitConfig()
    if n < 4:
        raise ValueError("A reconstruction needs at least 4 beads.")
    if len(ci) == 0:
        raise ValueError("This window has no contacts (e.g. an unassembled region); nothing to reconstruct.")
    dev = resolve_device(cfg.device)
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    t0 = time.time()

    m_ref = physics.reference_count(ci, cj, cm)
    target = physics.contact_target_distance(cm, m_ref, b0=1.0, alpha=cfg.alpha)
    ci_t = torch.as_tensor(ci, dtype=torch.long, device=dev)
    cj_t = torch.as_tensor(cj, dtype=torch.long, device=dev)
    tgt_t = torch.as_tensor(target, dtype=torch.float32, device=dev)
    if init_nm is not None:
        x0 = init_nm / b0
    elif cfg.init == "mds":
        x0 = shortest_path_mds(n, ci, cj, target, max_nodes=3000 if dev.type == "cuda" else 1000, device=dev)
    else:
        x0 = random_walk(n, rng)
    p = nn.Parameter(torch.as_tensor(x0, dtype=torch.float32, device=dev))

    history: dict[str, list[float]] = {k: [] for k in
                                       ("epoch", "stage", "contact", "smooth", "steric", "total", "grad_norm")}
    empty = torch.empty(0, dtype=torch.long, device=dev)

    def run_stage(name: str, epochs: int, forward: Callable[[], torch.Tensor], groups: list[dict],
                  steric_from: int) -> None:
        if epochs <= 0:
            return
        opt = torch.optim.AdamW(groups)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=0.0)
        params = [q for g in groups for q in g["params"]]
        pi = pj = empty
        offset = len(history["epoch"])
        for ep in range(epochs):
            steric_on = ep >= steric_from
            if steric_on and (ep - steric_from) % cfg.neighbor_every == 0:
                with torch.no_grad():
                    cur = forward().detach().cpu().numpy().astype(np.float64)
                i_np, j_np, _ = physics.neighbor_pairs(cur, cfg.d_min + cfg.skin, min_sep=2)
                pi, pj = torch.as_tensor(i_np, device=dev), torch.as_tensor(j_np, device=dev)
            opt.zero_grad(set_to_none=True)
            x = forward()
            lc = contact_loss(x, ci_t, cj_t, tgt_t)
            ls = smooth_loss(x)
            lst = steric_loss(x, pi, pj, cfg.d_min) if steric_on else x.sum() * 0.0
            total = lc + cfg.lambda_smooth * ls + cfg.lambda_steric * lst
            total.backward()
            gn = torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
            opt.step()
            sched.step()
            row = {"epoch": offset + ep + 1, "stage": name, "contact": lc.item(), "smooth": ls.item(),
                   "steric": lst.item(), "total": total.item(), "grad_norm": float(gn)}
            if not np.isfinite(row["total"]):
                raise FloatingPointError(f"Non-finite loss in {name} at epoch {ep + 1}: {row}")
            for k, v in row.items():
                history[k].append(v)
            if progress:
                progress(name, ep + 1, epochs, row)

    # Stage 1: contact embedding on free coordinates (phantom chain, then excluded volume)
    run_stage("embed", cfg.prefit_epochs, lambda: p,
              [{"params": [p], "lr": cfg.lr_prefit, "weight_decay": 0.0}],
              steric_from=int(cfg.phantom_fraction * cfg.prefit_epochs))

    # Stage 2: EGNN refinement conditioned on GC / H3K27ac node features
    n_params = 0
    forward_final: Callable[[], torch.Tensor] = lambda: p
    if cfg.refine_epochs > 0 and not cfg.refine_with_egnn:
        run_stage("refine", cfg.refine_epochs, forward_final,
                  [{"params": [p], "lr": cfg.lr_refine, "weight_decay": 0.0}], steric_from=0)
    elif cfg.refine_epochs > 0:
        graph = build_message_graph(n, ci, cj, cm, cfg.k_top).to_device(dev)
        feats = torch.as_tensor(node_feat, dtype=torch.float32, device=dev)
        model = ChromatinEGNN(node_dim=feats.shape[1], hidden=cfg.hidden, n_layers=cfg.n_layers).to(dev)
        n_params = sum(t.numel() for t in model.parameters())
        forward_final = lambda: model(p, feats, graph)[0]
        run_stage("refine", cfg.refine_epochs, forward_final,
                  [{"params": [p], "lr": cfg.lr_refine, "weight_decay": 0.0},
                   {"params": list(model.parameters()), "lr": cfg.lr_model, "weight_decay": cfg.weight_decay}],
                  steric_from=0)

    with torch.no_grad():
        coords = forward_final().detach().cpu().numpy().astype(np.float64) * b0
    return FitResult(coords, history, n_params, time.time() - t0,
                     asdict(cfg) | {"b0_nm": b0, "m_ref": m_ref, "device_used": str(dev)})


# ----------------------------------------------------------------------------------------
# Verification
# ----------------------------------------------------------------------------------------
def random_orthogonal(rng: np.random.Generator, proper: bool) -> np.ndarray:
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q = q @ np.diag(np.sign(np.diag(r)))
    if (np.linalg.det(q) > 0) != proper:
        q[:, 0] *= -1
    return q


def equivariance_check(n: int = 300, seed: int = 0) -> dict[str, float]:
    """Numerically test f(Qx + t) = Q f(x) + t and h(Qx + t) = h(x) in float64.

    Weights are re-drawn with a large scale so every MLP produces non-trivial output and the
    test cannot pass vacuously (a freshly initialised phi_x is ~0 by design).
    """
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    model = ChromatinEGNN(node_dim=3, hidden=16, n_layers=3, max_step=1.0).double()
    with torch.no_grad():
        for prm in model.parameters():
            prm.normal_(0.0, 0.3)
    x = torch.as_tensor(random_walk(n, rng) * 1.5, dtype=torch.float64)
    feats = torch.as_tensor(rng.normal(size=(n, 3)), dtype=torch.float64)
    ii = rng.integers(0, n, 4 * n)
    jj = rng.integers(0, n, 4 * n)
    keep = ii < jj
    g = build_message_graph(n, ii[keep], jj[keep], rng.integers(1, 50, keep.sum()).astype(float), 8,
                            dtype=torch.float64)
    out: dict[str, float] = {}
    with torch.no_grad():
        y, h = model(x, feats, g)
        for name, proper in (("rotation", True), ("reflection", False)):
            q = torch.as_tensor(random_orthogonal(rng, proper), dtype=torch.float64)
            t = torch.as_tensor(rng.normal(size=3) * 10, dtype=torch.float64)
            y_t, h_t = model(x @ q.T + t, feats, g)
            out[f"{name}_coord_err"] = float((y_t - (y @ q.T + t)).abs().max())
            out[f"{name}_feat_err"] = float((h_t - h).abs().max())
        out["displacement_scale"] = float((y - x).norm(dim=-1).mean())
    return out
