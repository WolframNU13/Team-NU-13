"""EGNN equivariance, gradient stability and reconstruction accuracy."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from chronocell import egnn, physics, synthetic  # noqa: E402


def test_e3_equivariance_rotation_and_reflection():
    err = egnn.equivariance_check(n=200, seed=1)
    assert err["displacement_scale"] > 1e-2                  # the layers really move the beads
    for k in ("rotation_coord_err", "reflection_coord_err", "rotation_feat_err", "reflection_feat_err"):
        assert err[k] < 1e-10, (k, err[k])


def test_gradients_finite_for_coincident_beads():
    n = 50
    rng = np.random.default_rng(0)
    x = torch.tensor(egnn.random_walk(n, rng), dtype=torch.float32)
    x[10] = x[30]                                             # exact overlap: ||x_i - x_j|| = 0
    x.requires_grad_(True)
    g = egnn.build_message_graph(n, np.array([10, 3]), np.array([30, 40]), np.array([5.0, 2.0]))
    model = egnn.ChromatinEGNN()
    y, _ = model(x, torch.zeros(n, 3), g)
    pi, pj = torch.tensor([10]), torch.tensor([30])
    loss = (egnn.contact_loss(y, torch.tensor([10]), torch.tensor([30]), torch.tensor([1.0]))
            + egnn.smooth_loss(y) + egnn.steric_loss(y, pi, pj, 0.8))
    loss.backward()
    assert torch.isfinite(x.grad).all()


def test_reconstructs_planted_structure():
    syn = synthetic.build(seed=7)
    lo, hi = 2600, 3000
    m = (syn.ci >= lo) & (syn.ci < hi) & (syn.cj >= lo) & (syn.cj < hi)
    feats = egnn.node_features(syn.gc[lo:hi], syn.epi[lo:hi], syn.valid[lo:hi])
    cfg = egnn.FitConfig(prefit_epochs=800, refine_epochs=20)
    res = egnn.fit_structure(hi - lo, feats, syn.ci[m] - lo, syn.cj[m] - lo, syn.cm[m], cfg)
    truth = syn.coords[lo:hi]
    rmsd, _, _ = physics.kabsch_rmsd(truth, res.coords_nm)
    # Benchmark (AUDIT.md §5), five windows of 400-1,600 beads: RMSD/Rg 0.28-0.32, distance r 0.96-0.97.
    assert rmsd / physics.radius_of_gyration(truth) < 0.4
    assert physics.distance_correlation(truth, res.coords_nm) > 0.95
    assert physics.loss_steric(res.coords_nm).overlaps <= 5


def test_mds_initialisation_recovers_global_fold():
    syn = synthetic.build(seed=7)
    lo, hi = 3000, 3400
    m = (syn.ci >= lo) & (syn.ci < hi) & (syn.cj >= lo) & (syn.cj < hi)
    target = physics.contact_target_distance(syn.cm[m], physics.reference_count(syn.ci[m], syn.cj[m], syn.cm[m]), b0=1.0)
    x0 = egnn.shortest_path_mds(hi - lo, syn.ci[m] - lo, syn.cj[m] - lo, target)
    assert physics.distance_correlation(syn.coords[lo:hi], x0) > 0.85


def test_mds_keeps_contact_free_stretches_compact():
    """Assembly gaps (no contacts) must not be laid out as extended rods by the initialisation."""
    syn = synthetic.build(seed=7)
    target = physics.contact_target_distance(syn.cm, physics.reference_count(syn.ci, syn.cj, syn.cm), b0=1.0)
    x0 = egnn.shortest_path_mds(len(syn.coords), syn.ci, syn.cj, target) * physics.B0_NM
    assert physics.radius_of_gyration(x0) < 2.5 * physics.radius_of_gyration(syn.coords)
