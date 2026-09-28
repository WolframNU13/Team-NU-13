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
