"""
Population (ensemble) reconstruction: a maximum-entropy Gaussian polymer ensemble, sampled with
exact Langevin (Ornstein-Uhlenbeck) dynamics. PyTorch; runs on the GPU when one is available.

Why a population. Hi-C / Micro-C and imaging contact frequencies average over thousands of cells,
and every cell folds differently. A single 3D structure cannot reproduce population statistics. On
held-out chromatin-tracing data (validation/, Bintu et al. 2018), even the best single structure
built from the *true* median distances recovers only ~55-65 % of the reproducible structure.

Model (lengths in units of the contact radius r_c)
1. Contact frequency -> pair spread. If the vector between beads i and j is Gaussian with
   per-axis variance s_ij, then f_ij = P(|r_ij| < r_c) = Maxwell_CDF(r_c / sqrt(s_ij)). Inverting
   gives s_ij. Rare contacts recover the familiar d ~ f^(-1/3) law, and frequent ones saturate
   correctly.
2. Maximum-entropy ensemble. The least-biased distribution of chains with given pair second
   moments is a Gaussian: p(X) ~ exp(-sum_{i<j} k_ij |x_i - x_j|^2 / 2), a network of springs k_ij
   (attractive or repulsive) acting on every pair (Shi & Thirumalai: HIPPS, PRX 2019; DIMES,
   Nat Commun 2023). It is parameterised by its covariance Sigma = A A^T (always valid). A is fitted
   so the model's pair variances match s_ij in log space, each pair weighted by its binomial
   reliability N f / (1 - f). Noisy, geometrically inconsistent pairs are thereby reconciled with
   well-measured ones.
3. Outputs are exact statistics of that ensemble:
   - median distance 1.5382 sqrt(s_ij) r_c (compared with microscopy);
   - contact probability (compared with the input, as the contact-map fit).
4. Trajectories. The spring network's energy defines overdamped Langevin dynamics,
   dx = -P x dt + sqrt(2) dW, with P = Sigma^+ (the learned couplings). The process is linear, so
   it is propagated exactly mode by mode. K replicas (default 100) start from equilibrium
   draws, so every frame is a genuine sample of the ensemble: a 4D movie of thermal fluctuations.

Limitations.
- No excluded volume and no confinement (Gaussian chains), so single sampled structures can overlap.
- It predicts population statistics, not any one cell's fold.
- Real pair distances are not exactly Gaussian (e.g. a loop formed in only some cells). Distance
  *rankings* stay accurate in that case, but absolute nm values for such pairs are less so.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field
from typing import Callable

import numpy as np
import torch

MAXWELL_MEDIAN = 1.5381722544550522          # median of |N(0, I_3)| in units of sigma
_X_GRID = np.geomspace(1e-3, 30.0, 20_000)    # r_c / sigma
_P_GRID = np.array([math.erf(x / math.sqrt(2)) - math.sqrt(2 / math.pi) * x * math.exp(-x * x / 2)
                    for x in _X_GRID])        # Maxwell CDF: P(|r| < r_c)


def contact_probability_from_sigma(sigma: np.ndarray, r_c: float = 1.0) -> np.ndarray:
    """P(|r| < r_c) for a Gaussian pair vector with per-axis standard deviation sigma."""
    x = r_c / np.maximum(np.asarray(sigma, dtype=np.float64), 1e-12)
    return np.interp(x, _X_GRID, _P_GRID)


def sigma_from_contact_probability(p: np.ndarray, r_c: float = 1.0) -> np.ndarray:
    """Inverse of contact_probability_from_sigma (p clipped to the tabulated range)."""
    p = np.clip(np.asarray(p, dtype=np.float64), _P_GRID[0], _P_GRID[-1])
    return r_c / np.interp(p, _P_GRID, _X_GRID)


def gaussian_median_distance(p: np.ndarray, r_c: float = 1.0) -> np.ndarray:
    """Median pair distance implied by a contact probability p = P(d < r_c), Gaussian pair vector.
    Small p recovers d ~ p^(-1/3); p = 1/2 gives exactly r_c."""
    return MAXWELL_MEDIAN * sigma_from_contact_probability(p, r_c)


@dataclass
class EnsembleConfig:
    iterations: int = 1500            # Adam steps fitting the covariance (tuned on practice data only)
    learning_rate: float = 0.01
    weighting: str = "binomial"       # "binomial": N f / (1 - f) per pair; "uniform"
    replicas: int = 100               # Langevin trajectories
    frames: int = 50                  # stored frames per trajectory
    frame_interval: float = 0.05      # in units of the slowest mode's relaxation time
    seed: int = 0
    device: str = "auto"


@dataclass
class EnsembleResult:
    median_distance_nm: np.ndarray    # (N, N) exact ensemble median distance
    contact_probability: np.ndarray   # (N, N) exact ensemble P(d < r_c)
    covariance_nm2: np.ndarray        # (N, N) per-axis covariance of bead positions (centred)
    couplings: np.ndarray             # (N, N) spring constants k_ij = -P_ij (1/nm^2); > 0 attractive
    trajectories_nm: np.ndarray       # (frames, replicas, N, 3) exact Langevin samples
    sampled_median_distance_nm: np.ndarray   # median over all stored frames (finite-sample check)
    representative_nm: np.ndarray     # (N, 3) sampled structure closest to the median map
    contact_fit: float                # Spearman(ensemble contact probability, input frequency)
    spread_cv: float                  # mean over pairs of std/mean distance across samples
    history: dict[str, list[float]]
    seconds: float
    config: dict = field(default_factory=dict)

    @property
    def frames_nm(self) -> np.ndarray:
        """Last frame of every trajectory: a population of `replicas` structures."""
        return self.trajectories_nm[-1]


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def _pair_variance(cov: np.ndarray) -> np.ndarray:
    d = np.diag(cov)
    return np.clip(d[:, None] + d[None] - 2 * cov, 0.0, None)


def fit_ensemble(freq: np.ndarray, n_observed: np.ndarray | float | None = None, r_c_nm: float = 150.0,
                 cfg: EnsembleConfig | None = None,
                 progress: Callable[[int, int, dict[str, float]], None] | None = None) -> EnsembleResult:
    """Fit a maximum-entropy population of chains to contact frequencies.

    freq        N x N contact frequencies in [0, 1] (fraction of cells / reads with d < r_c); NaN
                marks an unobserved pair, which is then set by the rest of the ensemble.
    n_observed  cells or reads behind each frequency (scalar or N x N). It bounds how close to 0 or
                1 a frequency is trusted (half a count) and sets the reliability weights.
    r_c_nm      the contact radius the frequencies refer to (150 nm for Bintu et al. tracing).
    """
    cfg = cfg or EnsembleConfig()
    t0 = time.time()
    f_in = np.asarray(freq, dtype=np.float64)
    n = len(f_in)
    if f_in.ndim != 2 or f_in.shape != (n, n) or n < 4:
        raise ValueError("freq must be a square matrix with at least 4 beads.")
    both = np.stack([f_in, f_in.T])                                   # symmetrise, ignoring NaN
    k_obs = np.isfinite(both).sum(0)
    f_in = np.where(k_obs > 0, np.nansum(both, 0) / np.maximum(k_obs, 1), np.nan)
    iu = np.triu_indices(n, 1)
    observed = np.isfinite(f_in[iu])
    if observed.sum() < n - 1:
        raise ValueError("Too few observed pairs to define an ensemble.")
    n_obs = np.broadcast_to(np.asarray(1000.0 if n_observed is None else n_observed, dtype=np.float64), (n, n))
    n_obs = np.maximum(n_obs[iu], 1.0)
    f = np.clip(np.where(observed, f_in[iu], 0.5), 0.5 / n_obs, 1 - 0.5 / n_obs)

    # 1. target per-axis pair variances (units r_c^2); unobserved pairs get a neutral placeholder
    s_target = sigma_from_contact_probability(f) ** 2
    # 2. starting covariance: projection of the target onto valid (PSD) covariances
    S = np.zeros((n, n))
    S[iu] = np.where(observed, s_target, np.nan)
    S = S + S.T
    if not observed.all():                    # fill gaps by genomic-separation medians for the start only
        sep = np.abs(np.subtract.outer(np.arange(n), np.arange(n)))
        for s in range(1, n):
            vals = np.diag(S, s)
            fill = np.nanmedian(vals) if np.isfinite(vals).any() else np.nanmedian(S[np.isfinite(S) & (sep > 0)])
            S[(sep == s) & ~np.isfinite(S)] = fill
    J = np.eye(n) - 1.0 / n
    w, v = np.linalg.eigh(-0.5 * J @ S @ J)
    a0 = v * np.sqrt(np.clip(w, 0.0, None))

    # 3. refine: weighted least squares on log pair variances
    dev = _device(cfg.device)
    torch.manual_seed(cfg.seed)
    wt = f * n_obs / (1 - f) if cfg.weighting == "binomial" else np.ones_like(f)
    wt = np.where(observed, wt, 0.0)
    wt = wt / wt[observed].mean()
    A = torch.tensor(a0, dtype=torch.float64, device=dev, requires_grad=True)
    ii = torch.as_tensor(iu[0], device=dev)
    jj = torch.as_tensor(iu[1], device=dev)
    tgt = torch.as_tensor(np.log(s_target), device=dev)
    wt_t = torch.as_tensor(wt, device=dev)
    opt = torch.optim.Adam([A], lr=cfg.learning_rate)
    history: dict[str, list[float]] = {"iteration": [], "loss": []}
    best, best_a = float("inf"), A.detach().clone()
    for it in range(cfg.iterations + 1):                                # last pass only evaluates
        opt.zero_grad(set_to_none=True)
        gram = A @ A.T                                                  # pair variance from the Gram matrix:
        g = torch.diagonal(gram)                                        # s_ij = G_ii + G_jj - 2 G_ij
        s = g[ii] + g[jj] - 2.0 * gram[ii, jj]
        loss = (wt_t * (torch.log(s + 1e-12) - tgt) ** 2).mean()
        val = loss.item()
        if not np.isfinite(val):
            raise FloatingPointError(f"Ensemble fit diverged at iteration {it}.")
        if val < best:                                                  # Adam jitters near the optimum:
            best, best_a = val, A.detach().clone()                      # keep the best covariance seen
        if it % 10 == 0 or it == cfg.iterations:
            history["iteration"].append(it)
            history["loss"].append(val)
            if progress:
                progress(it, cfg.iterations, {"loss": val})
        if it == cfg.iterations:
            break
        loss.backward()
        opt.step()
    history["best_loss"] = [best]
    t_fit = time.time() - t0

    # 4. exact ensemble statistics
    a = best_a.cpu().numpy()
    cov = J @ (a @ a.T) @ J                                              # centred, r_c^2 units
    s_model = _pair_variance(cov)
    sigma = np.sqrt(s_model)
    median = MAXWELL_MEDIAN * sigma * r_c_nm
    p_model = contact_probability_from_sigma(sigma)
    np.fill_diagonal(median, 0.0)
    np.fill_diagonal(p_model, 1.0)
    contact_fit = _spearman(p_model[iu][observed], f_in[iu][observed])

    # 5. exact Langevin (Ornstein-Uhlenbeck) trajectories, mode by mode
    t1 = time.time()
    lam, modes = np.linalg.eigh(cov)
    keep = lam > lam.max() * 1e-10                                      # drop the translation mode
    lam, modes = lam[keep], modes[:, keep]                              # per-axis variance of each mode
    rng = np.random.default_rng(cfg.seed)
    dt = cfg.frame_interval * lam.max()                                 # slowest relaxation time = lam.max()
    decay = np.exp(-dt / lam)                                           # precision of mode = 1 / lam
    y = rng.normal(size=(cfg.replicas, keep.sum(), 3)) * np.sqrt(lam)[None, :, None]   # equilibrium start
    frames = []
    for _ in range(cfg.frames):
        y = y * decay[None, :, None] + rng.normal(size=y.shape) * np.sqrt(lam * (1 - decay ** 2))[None, :, None]
        frames.append(np.einsum("nm,kmc->knc", modes, y))
    traj = np.stack(frames) * r_c_nm                                     # (frames, replicas, N, 3)

    def pair_d(x: np.ndarray) -> np.ndarray:                            # (k, N, 3) -> (k, pairs), float32
        return np.stack([np.linalg.norm(c[iu[0]] - c[iu[1]], axis=-1) for c in x]).astype(np.float32)

    budget = max(cfg.replicas, int(5e7 // max(len(iu[0]), 1)))        # ~200 MB of float32 at most
    step = max(1, int(np.ceil(cfg.frames * cfg.replicas / budget)))
    d = pair_d(traj.reshape(-1, n, 3)[::step])
    sampled = np.zeros((n, n))
    sampled[iu] = np.median(d, axis=0)
    sampled = sampled + sampled.T
    spread = float(np.mean(d.std(0) / np.maximum(d.mean(0), 1e-12)))
    last = pair_d(traj[-1]).astype(np.float64)
    rep = int(np.argmin(np.abs(np.log(np.maximum(last, 1e-9)) - np.log(np.maximum(median[iu], 1e-9))).mean(1)))

    prec = (modes / lam) @ modes.T                                      # pseudo-inverse of cov, 1/r_c^2
    couplings = -prec / r_c_nm ** 2
    np.fill_diagonal(couplings, 0.0)
    return EnsembleResult(median, p_model, cov * r_c_nm ** 2, couplings, traj, sampled, traj[-1][rep],
                          contact_fit, spread, history, time.time() - t0,
                          asdict(cfg) | {"r_c_nm": r_c_nm, "device_used": str(dev), "fit_seconds": t_fit,
                                         "sampling_seconds": time.time() - t1})


def counts_to_probability(counts: np.ndarray, p_adjacent: float = 0.5) -> np.ndarray:
    """Hi-C / Micro-C counts -> contact probabilities, for data that are not already frequencies.

    Sequencing gives relative counts only, so one number must be assumed: the contact probability
    of adjacent beads (`p_adjacent`). Counts are scaled so the median adjacent count maps to it,
    then capped below 1. The resulting absolute scale (nm) is an assumption, not a measurement;
    imaging-derived frequencies need no such step.
    """
    m = np.asarray(counts, dtype=np.float64)
    adj = np.diag(m, 1)
    adj = adj[np.isfinite(adj) & (adj > 0)]
    if adj.size == 0:
        raise ValueError("No adjacent-bead counts to anchor the probability scale.")
    return np.clip(m * (p_adjacent / float(np.median(adj))), 0.0, 0.999)


def fit_from_counts(ci: np.ndarray, cj: np.ndarray, cm: np.ndarray, n: int, valid: np.ndarray | None = None,
                    b0_nm: float = 50.0, p_adjacent: float = 0.5, cfg: EnsembleConfig | None = None,
                    progress: Callable[[int, int, dict[str, float]], None] | None = None) -> EnsembleResult:
    """Population model from a sequencing contact list (Hi-C / Micro-C counts), e.g. an app window.

    Sequencing counts are relative, so two assumptions are made explicit:
    - Adjacent beads touch with probability `p_adjacent`, which sets the probability scale via
      counts_to_probability. The effective number of cells is then median adjacent count / p_adjacent,
      so a pixel with zero reads means "rarer than one in N_eff", not "never".
    - The contact radius is chosen so adjacent beads sit b0 apart (median), which is the same length
      anchor as the v3.2 single-structure pipeline.
    Pairs touching an unassembled bin (valid = False) are treated as unobserved.
    """
    ci = np.asarray(ci, dtype=np.int64)
    cj = np.asarray(cj, dtype=np.int64)
    counts = np.zeros((n, n))
    np.add.at(counts, (ci, cj), np.asarray(cm, dtype=np.float64))
    counts = counts + counts.T
    np.fill_diagonal(counts, 0.0)
    adj = np.diag(counts, 1)
    adj = adj[adj > 0]
    if adj.size < 3:
        raise ValueError("Too few adjacent-bead contacts to set the probability scale for a population model.")
    p = counts_to_probability(counts, p_adjacent)
    n_eff = float(np.median(adj)) / p_adjacent
    if valid is not None:
        bad = ~np.asarray(valid, bool)
        p[bad, :] = np.nan
        p[:, bad] = np.nan
    r_c = b0_nm / float(gaussian_median_distance(p_adjacent, 1.0))
    res = fit_ensemble(p, max(n_eff, 1.0), r_c_nm=r_c, cfg=cfg, progress=progress)
    res.config.update({"input": "sequencing counts", "p_adjacent_assumed": p_adjacent, "n_effective": n_eff,
                       "anchor_b0_nm": b0_nm})
    return res
