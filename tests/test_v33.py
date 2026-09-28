"""Regression tests for v3.3: population (ensemble) reconstruction, maximum-entropy Gaussian ensemble
with exact Langevin sampling (chronocell.ensemble)."""

from __future__ import annotations

import numpy as np
import pytest

from chronocell import ensemble as E

FAST = E.EnsembleConfig(iterations=600, replicas=40, frames=20, device="cpu")


def _population(n: int = 14, bond: float = 0.6, cells: int = 6000, loop: tuple[int, int] | None = None,
                looped_fraction: float = 0.6, seed: int = 0) -> np.ndarray:
    """Planted population of Gaussian chains (units of r_c); optionally a fraction of cells has a
    closed loop between two beads (Brownian bridge, a physically consistent loop). Returns pair
    distances (cells, n, n)."""
    rng = np.random.default_rng(seed)
    steps = rng.normal(size=(cells, n - 1, 3)) * bond / np.sqrt(3)
    if loop:
        a, b = loop
        closed = rng.random(cells) < looped_fraction
        seg = steps[closed, a:b]
        end = rng.normal(size=(closed.sum(), 1, 3)) * 0.1          # loop anchors ~0.1 r_c apart
        steps[closed, a:b] = seg - (seg.sum(1, keepdims=True) - end) / (b - a)
    x = np.concatenate([np.zeros((cells, 1, 3)), np.cumsum(steps, axis=1)], axis=1)
    return np.linalg.norm(x[:, :, None] - x[:, None], axis=-1)


def test_contact_probability_and_distance_inversions_are_consistent():
    # P(d < r_c) = 1/2 means the median distance is exactly r_c
    assert E.gaussian_median_distance(0.5, 150.0) == pytest.approx(150.0, rel=1e-3)
    p = np.geomspace(1e-3, 0.9, 50)
    d = E.gaussian_median_distance(p)
    assert np.all(np.diff(d) < 0)                                  # more contact -> closer
    slope = np.polyfit(np.log(p[:10]), np.log(d[:10]), 1)[0]
    assert slope == pytest.approx(-1 / 3, abs=0.02)                # rare contacts: d ~ p^(-1/3)
    sig = np.geomspace(0.2, 5, 30)
    assert np.allclose(E.sigma_from_contact_probability(E.contact_probability_from_sigma(sig)), sig, rtol=1e-3)


def test_ensemble_recovers_a_planted_ideal_chain():
    d = _population()
    freq, truth = (d < 1.0).mean(0), np.median(d, axis=0) * 150.0
    res = E.fit_ensemble(freq, 6000, r_c_nm=150.0, cfg=FAST)
    iu = np.triu_indices(len(freq), 1)
    assert res.contact_fit > 0.95
    assert np.corrcoef(np.log(res.median_distance_nm[iu]), np.log(truth[iu]))[0, 1] > 0.98
    assert np.median(res.median_distance_nm[iu] / truth[iu]) == pytest.approx(1.0, abs=0.1)   # nm scale from r_c
    assert res.history["best_loss"][0] <= res.history["loss"][0] and res.history["best_loss"][0] < 1e-3
    assert res.trajectories_nm.shape == (20, 40, 14, 3) and res.frames_nm.shape == (40, 14, 3)
    assert res.representative_nm.shape == (14, 3)


def test_langevin_trajectories_sample_the_fitted_ensemble():
    d = _population(seed=3)
    res = E.fit_ensemble((d < 1.0).mean(0), 6000, cfg=E.EnsembleConfig(iterations=600, replicas=100, frames=50))
    iu = np.triu_indices(14, 1)
    ratio = res.sampled_median_distance_nm[iu] / res.median_distance_nm[iu]
    assert np.median(ratio) == pytest.approx(1.0, abs=0.05)         # finite sample -> exact statistics
    assert np.corrcoef(res.sampled_median_distance_nm[iu], res.median_distance_nm[iu])[0, 1] > 0.98
    # Maxwell distribution: std/mean of |r| is a constant ~0.42 for any Gaussian pair vector
    assert res.spread_cv == pytest.approx(0.42, abs=0.05)


def test_ensemble_captures_a_loop_present_in_part_of_the_population():
    d = _population(loop=(3, 11), seed=1)
    freq, truth = (d < 1.0).mean(0), np.median(d, axis=0)
    res = E.fit_ensemble(freq, 6000, cfg=FAST)
    assert abs(res.contact_probability[3, 11] - freq[3, 11]) < 0.1
    assert res.median_distance_nm[3, 11] < res.median_distance_nm[3, 10]      # loop anchors closer than inside
    assert truth[3, 11] < truth[3, 10]
    assert np.allclose(res.couplings, res.couplings.T) and np.isfinite(res.couplings).all()
    iu = np.triu_indices(len(freq), 1)
    # a partly-looped population is not Gaussian: ranks are right, nm values of loop pairs less so
    assert E._spearman(res.median_distance_nm[iu], truth[iu]) > 0.95


def test_unobserved_pairs_are_filled_by_the_ensemble():
    d = _population(seed=2)
    freq = (d < 1.0).mean(0)
    freq[0, 13] = freq[13, 0] = np.nan
    res = E.fit_ensemble(freq, 6000, cfg=FAST)
    assert np.isfinite(res.median_distance_nm).all()
    assert res.median_distance_nm[0, 13] > res.median_distance_nm[0, 7]      # still the farthest pair


def test_ensemble_input_validation():
    with pytest.raises(ValueError):
        E.fit_ensemble(np.ones((3, 3)))
    with pytest.raises(ValueError):
        E.fit_ensemble(np.ones((6, 5)))


def test_counts_to_probability_anchors_adjacent_pairs():
    n = 10
    s = np.abs(np.subtract.outer(np.arange(n), np.arange(n))).astype(float)
    counts = 400.0 / np.maximum(s, 1) ** 1.1
    p = E.counts_to_probability(counts, p_adjacent=0.5)
    assert np.median(np.diag(p, 1)) == pytest.approx(0.5)
    assert p.max() < 1.0 and p.min() >= 0.0


# ---------------------------------------------------------------- ICE balancing (chronocell.normalize)
def _biased_map(n: int = 200, seed: int = 0, drop_bin: int | None = None):
    """Circulant (equal-visibility) true map times planted per-bin biases, Poisson-sampled."""
    from chronocell import normalize as N  # noqa: F401
    rng = np.random.default_rng(seed)
    i, j = np.triu_indices(n, 1)
    sep = np.minimum(j - i, n - (j - i))
    true = 100.0 / sep ** 1.1
    bias = np.exp(rng.normal(0.0, 0.4, n))
    obs = rng.poisson(bias[i] * bias[j] * true * 50).astype(float) / 50
    if drop_bin is not None:                                   # an (almost) empty bin
        obs[(i == drop_bin) | (j == drop_bin)] = 0.0
        obs[(i == drop_bin) & (j == drop_bin + 1)] = 1.0
    return i, j, obs, true, bias


def test_ice_recovers_planted_biases_and_equalises_rows():
    from chronocell import normalize as N
    i, j, obs, true, bias = _biased_map()
    res = N.ice_balance(i, j, obs, 200, min_nnz=5)
    assert res.converged and res.row_sum_cv < 0.01
    assert np.corrcoef(np.log(res.bias), np.log(bias))[0, 1] > 0.99
    ok = obs > 0
    assert np.corrcoef(np.log(res.values[ok]), np.log(true[ok]))[0, 1] > \
        np.corrcoef(np.log(obs[ok]), np.log(true[ok]))[0, 1] + 0.05          # balancing removes the bias


def test_ice_masks_empty_bins_instead_of_inflating_them():
    from chronocell import normalize as N
    i, j, obs, *_ = _biased_map(drop_bin=50)
    res = N.ice_balance(i, j, obs, 200, min_nnz=5)
    assert res.masked[50] and np.isnan(res.bias[50]) and res.converged
    ci, cj, cm, notes = N.balanced_contacts(i, j, obs, 200, min_nnz=5)
    assert not np.any((ci == 50) | (cj == 50)) and np.all(cm > 0) and "converged" in notes[0]
    assert np.median(cm) == pytest.approx(np.median(obs[(obs > 0) & (i != 50) & (j != 50)]), rel=1e-6)
    with pytest.raises(ValueError):
        N.ice_balance([0], [1], [-1.0], 5)


# ---------------------------------------------------------------- bending stiffness / nuclear confinement
def test_bend_and_confinement_losses_match_their_numpy_twins():
    import torch
    from chronocell import egnn, physics
    x = np.cumsum(np.random.default_rng(0).normal(size=(60, 3)), axis=0) * 50.0
    xt = torch.tensor(x / 50.0)
    assert egnn.bend_loss(xt, 0.2).item() == pytest.approx(physics.loss_bend(x, 0.2), rel=1e-6)
    assert egnn.confinement_loss(xt, 3.0).item() == pytest.approx(physics.loss_confinement(x, 150.0, 50.0), rel=1e-6)
    straight = np.stack([np.arange(10.0), np.zeros(10), np.zeros(10)], axis=1)
    assert physics.loss_bend(straight, 1.0) == pytest.approx(0.0)
    assert physics.loss_confinement(straight, 100.0) == 0.0


def test_confinement_term_keeps_the_fold_inside_the_nucleus():
    from chronocell import egnn, physics
    rng = np.random.default_rng(1)
    truth = np.cumsum(rng.normal(size=(80, 3)), axis=0) * 50.0
    ci, cj, d = physics.neighbor_pairs(truth, 120.0, min_sep=1)
    cm = np.maximum(1.0, 50.0 * (50.0 / np.maximum(d, 1.0)) ** 3)
    feats = egnn.node_features(np.full(80, 0.42), np.zeros(80), np.ones(80, bool))
    base = egnn.FitConfig(prefit_epochs=200, refine_epochs=0, device="cpu")
    free = egnn.fit_structure(80, feats, ci, cj, cm, base, b0=50.0)
    r_free = np.linalg.norm(free.coords_nm - free.coords_nm.mean(0), axis=1).max()
    cfg = egnn.FitConfig(prefit_epochs=200, refine_epochs=0, device="cpu", lambda_confine=50.0,
                         confine_radius_nm=0.5 * r_free)
    held = egnn.fit_structure(80, feats, ci, cj, cm, cfg, b0=50.0)
    r_held = np.linalg.norm(held.coords_nm - held.coords_nm.mean(0), axis=1).max()
    assert r_held < 0.75 * r_free
    assert free.history["confine"][-1] == 0.0 and held.history["confine"][-1] >= 0.0
