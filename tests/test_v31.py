"""Regression tests for v3.1: format-based state engine, ChronoAgent, shape metrics, loader
extensions, and the whole app under AppTest (states, colour modes, agent, 4D, stale modules)."""

from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import pytest

from chronocell import agent as A, demo_states, formats, genome, physics, states as S, synthetic

CH = genome.chrom("chr22")
B0 = physics.bond_length_for(CH.resolution)


def _npy(arr) -> bytes:
    buf = io.BytesIO()
    np.save(buf, arr)
    return buf.getvalue()


@pytest.fixture(scope="module")
def ref():
    return synthetic.build(CH, seed=7, b0=B0)


# ---------------------------------------------------------------- format sniffing
@pytest.mark.parametrize("shape,kind,n", [
    ((500, 3), S.COORDS, 500), ((3, 500), S.COORDS, 500), ((4, 500, 3), S.FRAMES, 500),
    ((500,), S.TRACK, 500), ((500, 1), S.TRACK, 500), ((500, 4), S.UNSUPPORTED, 0), ((2, 3), S.UNSUPPORTED, 0),
])
def test_npy_classified_by_shape(shape, kind, n):
    k, got_n, got_shape, _ = S.sniff("x.npy", data=_npy(np.zeros(shape, np.float32)))
    assert (k, got_n, got_shape) == (kind, n, shape)


def test_sniff_never_raises_on_bad_content():
    assert S.sniff("x.npy", data=b"\x93NUMPY garbage")[0] == S.UNSUPPORTED
    assert S.sniff("x.npy", data=_npy(np.array(["a", "b", "c", "d"])))[0] == S.UNSUPPORTED     # non-numeric
    assert S.sniff("x.pdb", data=b"HEADER nothing here\nEND\n")[0] == S.UNSUPPORTED
    assert S.sniff("x.npz", data=b"PK not a zip")[0] == S.UNSUPPORTED
    assert S.sniff("x.bed", data=b"chr22\t1\t2\n")[0] == S.UNSUPPORTED


def test_pdb_and_bundle_sniffed(ref):
    pdb, _ = formats.write_pdb(ref.coords[:300], 0, ref.gc, ref.epi, 10.0, source="t", method="t", chrom=CH)
    assert S.sniff("s.pdb", data=pdb.encode())[:2] == (S.COORDS, 300)
    buf = io.BytesIO()
    formats.write_bundle(buf, formats.StructureBundle("chr22", 10_000, np.stack([ref.coords[:300]] * 3), np.arange(3),
                                                      ["a", "b", "c"]))
    assert S.sniff("b.npz", data=buf.getvalue())[:2] == (S.FRAMES, 300)


# ---------------------------------------------------------------- state / chromosome inference
@pytest.mark.parametrize("path,state", [
    ("chr22/healthy/coords.npy", S.HEALTHY), ("HealthyControl_H3K27ac.npy", S.HEALTHY), ("ctrl/rep1.pdb", S.HEALTHY),
    ("tumour_K562.npy", S.DISEASE), ("abnormal_cells.npy", S.DISEASE), ("healthy/tumour_sample.npy", S.DISEASE),
    ("Senescent/IMR90.pdb", S.SENESCENT), ("OIS_day7.npy", S.SENESCENT),
    ("normalized_counts.npy", None), ("run_01.npy", None), ("graph_chr22.npz", None),
])
def test_state_from_path_words(path, state):
    assert S.infer_state(path) == state


def test_chromosome_from_path():
    assert S.infer_chrom("chr2/chr22_ctrl.npy") == "chr22"
    assert S.infer_chrom("x/chrX_sen.npy") == "chrX"
    assert S.infer_chrom("achr3.npy") is None and S.infer_chrom("healthy/a.npy") is None


def test_plan_pairs_tracks_by_length_then_name():
    mk = lambda name, kind, n: S.StateFile(f"folder:{name}", name, "folder", kind, n, (n,), S.DISEASE, None, "")
    coords = mk("cancer/tumour.npy", S.COORDS, 5082)
    tracks = (mk("cancer/other.npy", S.TRACK, 800), mk("cancer/tumour_h3k27ac.npy", S.TRACK, 5082),
              mk("x/signal.npy", S.TRACK, 5082))
    assert S.best_track(coords, tracks).name == "cancer/tumour_h3k27ac.npy"
    assert S.best_track(coords, tracks[:1]) is None                     # length never matches
    plan = S.plan([coords, *tracks], S.DISEASE)
    assert plan.structures == (coords,) and len(plan.tracks) == 3 and S.plan([coords], S.HEALTHY).empty


def test_load_track_accepts_column_vector():
    assert S.load_track(_npy(np.ones((7, 1)))).shape == (7,)
    with pytest.raises(ValueError):
        S.load_track(_npy(np.ones((7, 3))))


# ---------------------------------------------------------------- shape metrics
def test_max_span_is_exact(ref):
    rng = np.random.default_rng(0)
    for x in (rng.normal(size=(700, 3)) * 50, ref.coords[:900], np.c_[np.arange(50.0), np.zeros(50), np.zeros(50)]):
        brute = np.sqrt(((x[:, None] - x[None]) ** 2).sum(-1)).max()
        assert physics.max_span(x) == pytest.approx(brute, rel=1e-12)


def test_gyration_shape_limits():
    rod = physics.gyration_shape(np.c_[np.arange(100.0), np.zeros(100), np.zeros(100)])
    ball = physics.gyration_shape(np.random.default_rng(1).normal(size=(20_000, 3)))
    assert rod.asphericity == pytest.approx(1.0) and ball.asphericity < 0.02
    assert sum(rod.eigenvalues) == pytest.approx(physics.radius_of_gyration(np.c_[np.arange(100.0), np.zeros(100),
                                                                                    np.zeros(100)]) ** 2)


def test_local_density_counts_non_bonded_neighbours():
    x = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [0.5, 0.2, 0]], float)
    d = physics.local_density(x, r=0.9, min_sep=2)
    assert d.tolist() == [1.0, 1.0, 0.0, 2.0]          # bead 3 is close to 0 and 1, both >= 2 apart in sequence


# ---------------------------------------------------------------- ChronoAgent
def test_spearman_handles_ties_and_monotone_maps():
    a = np.arange(40.0)
    assert A.spearman(a, a ** 3) == pytest.approx(1.0) and A.spearman(a, -a) == pytest.approx(-1.0)
    assert np.isnan(A.spearman(np.ones(40), a)) and np.isnan(A.spearman(a[:5], a[:5]))
    assert A._rank(np.array([3.0, 1.0, 3.0, 2.0])).tolist() == [2.5, 0.0, 2.5, 1.0]


def _ctx(ref, state=A.DISEASE, has_data=True, placeholder=False, comparisons=True):
    lo, hi = 1800, 2400
    m = A.compute_metrics(ref.coords[lo:hi], ref.epi[lo:hi], ref.valid[lo:hi], B0, CH, lo, "H3K27ac", placeholder)
    x = ref.coords[lo:hi]
    swollen = synthetic.relax(x.mean(0) + (x - x.mean(0)) * 1.3, B0, 0.8 * B0, iters=20)
    md = A.compute_metrics(swollen, ref.epi[lo:hi], ref.valid[lo:hi], B0, CH, lo, "H3K27ac", placeholder)
    comps = (A.Comparison(A.HEALTHY, m, True, "ctrl.npy"),) if comparisons else ()
    return A.AgentContext(state, has_data, "chr22", 10_000, "Custom window", "chr22:18,000,001-24,000,000",
                          "tumour.npy", "h3k27ac.npy", False, False, B0, md,
                          tuple(g.name for g in genome.genes_in("chr22", lo * 10_000, hi * 10_000)), comps)


def test_metrics_are_finite_and_consistent(ref):
    ctx = _ctx(ref)
    m = ctx.metrics
    assert m.n_beads == 600 and m.rg_nm > 0 and m.span_nm > 2 * m.rg_nm and 0 < m.packing < 1
    assert 0 <= m.dense_fraction <= 1 and np.isfinite(m.mean_signal) and len(m.accessible) == 3
    assert m.rg_nm > ctx.comparisons[0].metrics.rg_nm                   # swollen copy is larger
    with pytest.raises(ValueError):
        A.compute_metrics(ref.coords[:10], ref.epi[:9], ref.valid[:10], B0, CH)


def test_heuristic_sections_and_decompaction_call(ref):
    text = A.heuristic_analysis(_ctx(ref), "which drugs, vs healthy?")
    for head in ("### Biophysical diagnosis", "### Therapeutic strategy (research hypotheses)",
                 "### Expression & accessibility insights", "### Answer to your question"):
        assert head in text
    assert "decompaction" in text and "BET" in text and "BCR" in text and A.DISCLAIMER in text


def test_heuristic_withholds_without_data_or_real_signal(ref):
    no_data = A.heuristic_analysis(_ctx(ref, has_data=False, comparisons=False))
    assert "Withheld" in no_data and "BET" not in no_data
    placeholder = A.heuristic_analysis(_ctx(ref, placeholder=True))
    assert "signal-based insights are withheld" in placeholder and "Spearman" not in placeholder


def test_llm_fallback_chain_and_key_hygiene(ref):
    ctx, key, seen = _ctx(ref), "AIzaSECRET123", []

    def transport(url, headers, payload, timeout):
        seen.append(url)
        assert key not in url and headers["x-goog-api-key"] == key
        if "flash-latest" in url:
            return 404, {"error": {"message": "model not found"}}
        return 200, {"candidates": [{"content": {"parts": [{"text": "### Biophysical diagnosis\nok"}]}}]}

    res = A.ask_llm(ctx, "q", "h", A.GEMINI, key, transport=transport)
    assert res.model == A.DEFAULT_MODELS[A.GEMINI][1] and len(seen) == 2

    def rejected(url, headers, payload, timeout):
        seen.append(url)
        return 401, {"error": {"message": f"bad key {key}"}}

    seen.clear()
    with pytest.raises(A.AgentError) as e:
        A.ask_llm(ctx, "", "h", A.OPENROUTER, key, transport=rejected)
    assert key not in str(e.value) and len(seen) == 1                    # a rejected key is not retried
    with pytest.raises(A.AgentError):
        A.ask_llm(ctx, "", "h", A.GEMINI, key, model="../../x?y=1", transport=transport)
    assert A.detect_provider("AIza...") == A.GEMINI and A.detect_provider("sk-or-v1-x") == A.OPENROUTER
    assert A.detect_provider("hello") is None


def test_report_markdown(ref):
    ctx = _ctx(ref)
    md = A.report_markdown(ctx, A.heuristic_analysis(ctx), "offline heuristic engine", "why?")
    assert md.startswith("# ChronoCell-5D Analysis Report")
    for part in ("## Structural metrics", "## Other biological states", "## ChronoAgent analysis", "Maximum 3D span"):
        assert part in md


# ---------------------------------------------------------------- loader: state track + condition
def test_load_dataset_applies_state_track(ref):
    from ui.common import load_dataset
    coords = ref.coords[:5082].astype(np.float32)
    track = np.linspace(0, 1, 5082)
    ds = load_dataset("chr22", 7, None, ("Healthy · c.npy", "c.npy", _npy(coords)), "nm", None, False,
                      ("Healthy · t.npy", "t.npy", _npy(track)), "Healthy Control")
    ok = ds.valid
    assert ds.condition == "Healthy Control" and not ds.signal_is_placeholder and "t.npy" in ds.signal_label
    assert np.allclose(ds.epi[ok], track[ok])
    bad = load_dataset("chr22", 7, None, ("x", "c.npy", _npy(coords)), "nm", None, False, ("y", "t.npy", _npy(track[:99])))
    assert bad.signal_is_placeholder and any("signal track ignored" in n for n in bad.notes)


def test_slot_ignores_tracks(tmp_path, monkeypatch):
    import ui.common as C
    (tmp_path / "chr22").mkdir()
    np.save(tmp_path / "chr22" / "coords.npy", np.zeros((50, 3)))
    np.save(tmp_path / "chr22" / "signal.npy", np.zeros(50))
    monkeypatch.setattr(C, "SLOT_ROOT", tmp_path)
    assert [p.name for p in C.slot_files("chr22")] == ["coords.npy"]


def test_demo_states_are_discovered(tmp_path):
    demo_states.write_demo(tmp_path, relax_iters=5)
    files = S.for_chromosome(S.scan_folder(tmp_path), "chr22")
    plans = {s: S.plan(files, s) for s in S.STATES}
    assert [len(plans[s].structures) for s in S.STATES] == [1, 1, 1]
    assert [len(plans[s].tracks) for s in S.STATES] == [1, 1, 1]
    sen = plans[S.SENESCENT]
    assert sen.structures[0].name.endswith(".pdb") and S.best_track(sen.structures[0], sen.tracks) is not None


# ---------------------------------------------------------------- the whole app (AppTest)
def test_app_end_to_end_with_states(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    import ui.common as C
    import ui.states_panel as SP
    demo_states.write_demo(tmp_path, relax_iters=5)
    monkeypatch.setattr(C, "SLOT_ROOT", tmp_path)
    monkeypatch.setattr(SP, "SLOT_ROOT", tmp_path)
    app = str(Path(__file__).resolve().parent.parent / "app.py")
    at = AppTest.from_file(app, default_timeout=300)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    page = lambda: "\n".join(m.value for m in at.markdown)
    assert "cc-cards" in page() and any("ChronoAgent" in e.label for e in at.expander)
    assert [t.label for t in at.sidebar.text_input][0] == "AI API Key"
    for state in (S.DISEASE, S.SENESCENT):
        at.sidebar.selectbox(key="bio_state").set_value(state).run()
        assert not at.exception, [e.value for e in at.exception]
        assert f'<span class="v s">{state}</span>' in page()
        assert "vs Healthy" in page()                                   # metric deltas against the control
    for mode in ("Residue Index Spectrum", "Epigenomic Signal Heatmap"):
        at.selectbox(key="disp_colour").set_value(mode).run()
        assert not at.exception
    at.segmented_control(key="workspace").set_value("4D dynamics").run()
    mode = next(s for s in at.segmented_control if (s.key or "").startswith("fourd_mode"))
    assert "Across conditions" in mode.options
    mode.set_value("Across conditions").run()
    assert not at.exception, [e.value for e in at.exception]
    # Last, because it re-imports every project module: what Streamlit's watcher does when a file is
    # saved (the cause of "module 'chronocell.genome' has no attribute 'MAIN_CHROMOSOMES'").
    del sys.modules["chronocell.genome"]
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert sys.modules["chronocell.genome"].MAIN_CHROMOSOMES[0] == "chr1"
