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


# ---------------------------------------------------------------- app: population model, scores, probe, slicing
def test_app_population_model_two_scores_probe_and_slicing(tmp_path, monkeypatch):
    from pathlib import Path

    from streamlit.testing.v1 import AppTest
    import ui.common as C
    import ui.states_panel as SP
    monkeypatch.setattr(C, "SLOT_ROOT", tmp_path / "empty")
    monkeypatch.setattr(SP, "SLOT_ROOT", tmp_path / "empty")
    monkeypatch.setattr(SP, "DEMO_ROOT", tmp_path / "demo")
    at = AppTest.from_file(str(Path(__file__).resolve().parent.parent / "app.py"), default_timeout=900)
    at.run()
    ok = lambda: not at.exception and "cc-card-warn" not in "".join(m.value for m in at.markdown)  # noqa: E731
    assert ok(), [e.value for e in at.exception]
    at.session_state["custom_window"] = (2000, 2150)
    at.segmented_control(key="region_choice").set_value("custom").run()
    assert ok(), [e.value for e in at.exception]
    text = lambda: "".join(m.value for m in at.markdown)  # noqa: E731
    assert "Accuracy · two separate scores" in text()
    build = next(b for b in at.button if (b.label or "").startswith("Build population model"))
    build.click().run()
    assert ok(), [e.value for e in at.exception]
    assert len(at.session_state["ensembles"]) == 1
    assert any(r["Model"] == "v3.3 population" for r in at.session_state["telemetry"])
    assert "Population model" in text() and "Microscopy accuracy" in text()
    from chronocell import accuracy as ACC
    bench = ACC.load_benchmark()
    if bench:                                                   # the benchmark number shown is the stored one
        assert f"{bench['models']['ensemble_v3_3']['overall_percent_of_ceiling']:.1f} %" in text()
    labels = [d.label for d in at.get("download_button")]
    assert "Population (PDB, 100 models)" in labels
    at.toggle(key="probe_on").set_value(True).run()
    assert ok() and "Population median (all trajectories)" in text()
    at.toggle(key="clip_on").set_value(True).run()
    assert ok(), [e.value for e in at.exception]
    next(b for b in at.button if b.key == "agent_build_3d").click().run()
    assert ok() and any((d.label or "").startswith("Research dossier") for d in at.get("download_button"))


def test_pdf_dossier_prints_both_scores_separately():
    from chronocell import accuracy as ACC, agent as A, genome, pdf_report, physics, synthetic
    ch = genome.chrom("chr22")
    b0 = physics.bond_length_for(ch.resolution)
    ref = synthetic.build(ch, seed=7, b0=b0)
    m = A.compute_metrics(ref.coords[2000:2150], ref.epi[2000:2150], ref.valid[2000:2150], b0, ch, 2000)
    ctx = A.AgentContext(A.HEALTHY, True, "chr22", ch.resolution, "Custom window", "chr22:20,000,001-21,500,000",
                         "reference", "h3k27ac", True, True, b0, m, (), (), {})
    scores = ACC.two_scores("ensemble_v3_3", 0.9)
    pdf = pdf_report.build(ctx, "ok", "offline", accuracy=scores)
    assert pdf[:4] == b"%PDF" and len(pdf) > len(pdf_report.build(ctx, "ok", "offline"))   # the section adds content
    assert scores["contact_map_fit"]["value"] == 0.9
    assert "not measured on this" in (scores["microscopy_accuracy"] or {"scope": "not measured on this"})["scope"]


# ---------------------------------------------------------------- REST API handlers and run log (chronocell.api)
def _planted_counts(n: int = 40, seed: int = 0):
    """Counts from a planted population (Gaussian chains): P(d < r_c) scaled to reads."""
    d = _population(n=n, cells=4000, seed=seed)
    f = (d < 1.0).mean(0)
    i, j = np.triu_indices(n, 1)
    cnt = np.random.default_rng(seed).poisson(f[i, j] * 400).astype(float)
    keep = cnt > 0
    return {"i": i[keep].tolist(), "j": j[keep].tolist(), "count": cnt[keep].tolist()}, n


def test_api_reconstruct_population_reports_two_scores_and_logs_without_raw_data(tmp_path):
    from chronocell import api
    contacts, n = _planted_counts()
    log = api.AuditLog(tmp_path / "runs.jsonl")
    out = api.reconstruct({"contacts": contacts, "n_beads": n, "model": "population", "b0_nm": 50.0,
                           "include": ["median_distance"]}, log=log)
    assert out["model"] == "population_v3_3" and np.asarray(out["coords_nm"]).shape == (n, 3)
    assert np.asarray(out["median_distance_nm"]).shape == (n, n)
    assert out["accuracy"]["contact_map_fit"]["value"] > 0.8
    assert "not evidence of accuracy" in out["accuracy"]["contact_map_fit"]["definition"]
    mic = out["accuracy"]["microscopy_accuracy"]
    assert mic is None or "not measured on this structure" in mic["scope"]          # benchmark never passed off as local
    assert out["metrics"]["n_beads"] == n and len(out["input_sha256"]) == 64
    rec = log.read()
    assert len(rec) == 1 and rec[0]["status"] == "ok" and rec[0]["input_sha256"] == out["input_sha256"]
    assert rec[0]["parameters"]["contacts"]["i"].startswith("<array")                 # sizes only, no raw data
    assert str(contacts["count"][:3])[1:-1] not in (tmp_path / "runs.jsonl").read_text()


def test_api_single_structure_metrics_and_benchmark(tmp_path):
    from chronocell import api
    contacts, n = _planted_counts(n=30, seed=4)
    log = api.AuditLog(tmp_path / "runs.jsonl")
    out = api.reconstruct({"contacts": contacts, "n_beads": n, "model": "single", "b0_nm": 50.0}, log=log)
    assert out["model"] == "single_structure_v3_2" and "final_loss" in out["telemetry"]
    m = api.metrics({"coords_nm": out["coords_nm"], "b0_nm": 50.0, "contacts": contacts}, log=log)
    assert m["metrics"]["rg_nm"] > 0 and m["accuracy"]["microscopy_accuracy"] is None      # input: no benchmark claim
    from chronocell import accuracy as ACC
    if ACC.load_benchmark():
        b = api.benchmark(log=log)
        assert b["summary"]["ensemble_v3_3"] == ACC.load_benchmark()["models"]["ensemble_v3_3"]["overall_percent_of_ceiling"]
    assert [r["endpoint"] for r in log.read()][:2] == ["/api/v1/reconstruct", "/api/v1/metrics"]


def test_api_rejects_bad_requests_and_logs_them(tmp_path):
    from chronocell import api
    log = api.AuditLog(tmp_path / "runs.jsonl")
    bad = [{"contacts": {"i": [0], "j": [1]}},                                             # missing counts
           {"contacts": {"i": [0, 1], "j": [5, 2], "count": [1, -2]}, "n_beads": 6},       # negative count
           {"contacts": {"i": [0], "j": [9], "count": [1]}, "n_beads": 5},                 # index out of range
           {"contacts": {"i": [0], "j": [1], "count": [1]}, "n_beads": 500, "model": "population"},   # too large
           {"contacts": {"i": [0], "j": [1], "count": [1]}, "n_beads": 10, "model": "magic"}]
    for payload in bad:
        with pytest.raises(api.RequestError):
            api.reconstruct(payload, log=log)
    with pytest.raises(api.RequestError):
        api.metrics({"coords_nm": [[0, 0, 0]]}, log=log)
    with pytest.raises(api.RequestError):                                                  # model-level ValueError -> 400
        api.reconstruct({"contacts": {"i": [0, 5], "j": [9, 12], "count": [1, 1]}, "n_beads": 20}, log=log)
    assert [r["status"] for r in log.read()] == ["rejected"] * 7
    with pytest.raises(api.RequestError):                                                  # not a JSON object
        api.reconstruct(["not", "an", "object"], log=log)


def test_api_fastapi_wrapper_when_installed(tmp_path):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    from chronocell import api
    client = TestClient(api.create_app(api.AuditLog(tmp_path / "runs.jsonl")))
    contacts, n = _planted_counts(n=24, seed=5)
    r = client.post("/api/v1/reconstruct", json={"contacts": contacts, "n_beads": n})
    assert r.status_code == 200 and r.json()["model"] == "population_v3_3"
    assert client.post("/api/v1/reconstruct", json={"contacts": {}}).status_code == 400
