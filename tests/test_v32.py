"""Regression tests for v3.2: genes, neighbourhoods, drug lab, patient-data ingestion, snapshots, PDF,
secrets-based API key, linked viewports, and every page of the app under AppTest."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from chronocell import (agent as A, demo_states, domains as D, formats, genes as G, genome, ingest as I, pdf_report,
                        physics, snapshot as SN, synthetic, theme as T, therapy as TH)

CH = genome.chrom("chr22")
B0 = physics.bond_length_for(CH.resolution)


@pytest.fixture(scope="module")
def ref():
    return synthetic.build(CH, seed=7, b0=B0)


@pytest.fixture(scope="module")
def demo(tmp_path_factory):
    root = tmp_path_factory.mktemp("demo")
    demo_states.write_demo(root, relax_iters=10)
    load = lambda rel: np.load(root / "chr22" / rel).astype(float)  # noqa: E731
    sen, _ = formats.read_pdb((root / "chr22/senescent/synthetic_demo_senescent.pdb").read_text())
    return {"root": root, "healthy": load("healthy/synthetic_demo_coords.npy"),
            "tumour": load("cancer/synthetic_demo_tumour_coords.npy"),
            "sig_h": load("healthy/synthetic_demo_h3k27ac.npy"), "sig_t": load("cancer/synthetic_demo_tumour_h3k27ac.npy"),
            "senescent": sen, "valid": CH.assembled_mask()}


# ---------------------------------------------------------------- genes
def test_gene_annotation_is_complete_and_searchable():
    tab = G.table()
    assert len(tab) > 19_000 and set(tab["chrom"]) == set(genome.MAIN_CHROMOSOMES)
    bcr = G.find("bcr")
    assert bcr["chrom"] == "chr22" and bcr["start"] == 23_180_508 and G.find("TP53")["chrom"] == "chr17"
    assert G.find("") is None and G.category("BCR") == "cancer" and G.category("SNCA") == "neuro"


def test_accessibility_table_statuses(ref):
    tab = G.accessibility_table(ref.coords, ref.epi, ref.valid, B0, CH, 0)
    assert len(tab) == len(G.on_chromosome("chr22")[lambda d: d["tss"] < CH.size])
    assert set(tab["status"]) <= set(G.STATUS_ORDER)
    assert (tab.loc[tab["score"] >= 0.5, "status"] == G.OPEN).all()
    assert (tab.loc[tab["score"] <= -0.5, "status"] == G.BURIED).all()
    s = G.summary(tab)
    assert s["open"] + s["intermediate"] + s["buried"] == s["genes_in_view"]
    assert any(g["gene"] == "BCR" for g in s["flagged_disease_genes"])


def test_expression_parsing_and_agreement(ref):
    tab = G.accessibility_table(ref.coords, ref.epi, ref.valid, B0, CH, 0)
    # expression that follows the score perfectly must give rho ~ 1
    expr = pd.Series(np.exp(tab["score"].fillna(0).to_numpy() + 3), index=tab["gene"])
    text = "gene\ttpm\n" + "\n".join(f"{g}\t{v:.4f}" for g, v in expr.items())
    parsed = G.parse_expression(text.encode(), "rna.tsv")
    tab2 = G.accessibility_table(ref.coords, ref.epi, ref.valid, B0, CH, 0, expression=parsed)
    assert G.expression_agreement(tab2)["rho"] > 0.95
    with pytest.raises(ValueError):
        G.parse_expression(b"just one column\n1\n2\n")


# ---------------------------------------------------------------- neighbourhoods
def test_insulation_matches_brute_force():
    rng = np.random.default_rng(0)
    n, w = 120, 10
    ci = rng.integers(0, n - 1, 2000)
    cj = np.minimum(ci + rng.integers(1, 40, 2000), n - 1)
    keep = ci < cj
    ci, cj, cm = ci[keep], cj[keep], rng.random(keep.sum()) + 0.5
    fast = D.insulation(ci, cj, cm, n, w)
    for i in range(w, n - w):
        brute = cm[(ci >= i - w) & (ci < i) & (cj > i) & (cj <= i + w)].sum()
        assert np.isclose(2 ** fast[i] * np.mean([cm[(ci >= k - w) & (ci < k) & (cj > k) & (cj <= k + w)].sum()
                                                    for k in range(w, n - w)]), brute)


def test_boundaries_found_between_planted_domains():
    # three dense blocks -> boundaries near 40 and 80
    n = 120
    i, j = np.triu_indices(n, 1)
    block = (i // 40) == (j // 40)
    cm = np.where(block, 10.0, 0.2) / np.maximum(j - i, 1) ** 0.5
    rep = D.analyse(n, 10_000, B0, ci=i, cj=j, cm=cm)
    assert any(abs(b - 40) <= 2 for b in rep.boundaries) and any(abs(b - 80) <= 2 for b in rep.boundaries)
    assert len(rep.tads) >= 3


def test_compartment_sign_follows_orientation_track():
    n = 200
    i, j = np.triu_indices(n, 1)
    kind = (np.arange(n) // 25) % 2                      # alternating A/B blocks of 25 beads
    same = kind[i] == kind[j]
    cm = np.where(same, 2.0, 0.5) / np.maximum(j - i, 1) ** 0.8
    orient = np.where(kind == 1, 1.0, 0.0)               # "GC-rich" = kind 1
    ev, _ = D.compartments(i, j, cm, n, orient)
    assert np.mean(ev[kind == 1] > 0) > 0.9 and np.mean(ev[kind == 0] < 0) > 0.9


def test_domains_from_structure_agree_with_contacts(ref):
    a = D.analyse(CH.n_bins, CH.resolution, B0, ref.coords, ref.ci, ref.cj, ref.cm, orient=ref.gc)
    b = D.analyse(CH.n_bins, CH.resolution, B0, ref.coords, orient=ref.gc)
    assert a.source == "measured contacts" and b.source == "3D proximity" and b.loops == []
    nearest = [np.min(np.abs(np.array(b.boundaries) - x)) for x in a.boundaries]
    assert np.median(nearest) <= 3 and 0.9 < a.gamma < 1.8


# ---------------------------------------------------------------- drug lab
def test_drug_selectivity_matches_the_defect(demo):
    lo, hi = TH.suggest_window(demo["tumour"], demo["healthy"], demo["sig_t"], 800)
    assert 1600 <= lo <= 2000                                                   # the planted swollen region
    tumour = TH.compare_drugs(demo["tumour"][lo:hi], demo["sig_t"][lo:hi], demo["valid"][lo:hi], B0,
                              demo["healthy"][lo:hi], signal_ref=demo["sig_h"]).set_index("key")
    assert tumour.loc["bet", "restoration_pct"] > 10                            # over-open, hyper-acetylated -> BET
    assert tumour.loc["hdac", "restoration_pct"] < 2 and tumour.loc["ezh2", "restoration_pct"] < 2
    lo2, hi2 = 3000, 3800
    sen = TH.compare_drugs(demo["senescent"][lo2:hi2], demo["sig_h"][lo2:hi2] * 0.7, demo["valid"][lo2:hi2], B0,
                           demo["healthy"][lo2:hi2], signal_ref=demo["sig_h"]).set_index("key")
    assert sen.loc["hdac", "restoration_pct"] > 5 and sen.loc["bet", "restoration_pct"] < 2   # compacted -> opening drugs


def test_treatment_is_monotone_and_physical(demo):
    x, h = demo["tumour"][1800:2400], demo["healthy"][1800:2400]
    r = TH.simulate_treatment(x, demo["sig_t"][1800:2400], demo["valid"][1800:2400], B0, "bet", h,
                              signal_ref=demo["sig_h"])
    rest = r.metrics["restoration_pct"].to_numpy()
    assert rest[0] == 0 and np.all(np.diff(rest) > -1.0) and rest[-1] > rest[5] > 0
    assert r.frames.shape == (11, 600, 3)
    bonds = physics.bond_lengths(r.frames[-1]) / B0
    assert 0.9 < np.median(bonds) < 1.1
    mech = TH.simulate_treatment(x, demo["sig_t"][1800:2400], demo["valid"][1800:2400], B0, "bet", None,
                                 signal_ref=demo["sig_h"])
    assert mech.mode == "mechanism only" and mech.metrics["rg_nm"].iloc[-1] < mech.metrics["rg_nm"].iloc[0]
    with pytest.raises(ValueError):
        TH.simulate_treatment(x[:10], np.ones(10), np.ones(10, bool), B0, "hdac")


# ---------------------------------------------------------------- patient-data ingestion
def _cool_bytes(ref, res=5000):
    import h5py
    sizes = {"chr21": 46_709_983, "chr22": 50_818_468}
    names = list(sizes)
    nb = {c: -(-sizes[c] // res) for c in names}
    offs = np.array([0, nb["chr21"], nb["chr21"] + nb["chr22"]])
    starts = np.concatenate([np.arange(nb[c]) * res for c in names])
    b1 = np.r_[5, offs[1] + ref.ci * 2]
    b2 = np.r_[offs[1] + 7, offs[1] + ref.cj * 2]
    cnt = np.r_[9, ref.cm].astype(np.int32)
    order = np.lexsort((b2, b1))
    b1, b2, cnt = b1[order], b2[order], cnt[order]
    buf = io.BytesIO()
    with h5py.File(buf, "w") as f:
        f.attrs["bin-size"] = res
        g = f.create_group("chroms"); g["name"] = np.array(names, dtype="S"); g["length"] = list(sizes.values())
        g = f.create_group("bins"); g["chrom"] = np.repeat([0, 1], [nb["chr21"], nb["chr22"]]); g["start"] = starts
        g["end"] = starts + res
        g = f.create_group("pixels"); g["bin1_id"] = b1; g["bin2_id"] = b2; g["count"] = cnt
        g = f.create_group("indexes"); g["chrom_offset"] = offs
        g["bin1_offset"] = np.searchsorted(b1, np.arange(offs[-1] + 1))
    return buf.getvalue()


def test_cool_is_read_and_aggregated(ref):
    pytest.importorskip("h5py")
    ci, cj, cm, notes = I.read_contacts(_cool_bytes(ref), "patient.cool", CH)
    assert len(ci) == len(ref.ci) and cm.sum() == pytest.approx(ref.cm.sum()) and "aggregated" in notes[0]


def test_contact_tables_and_tracks(ref):
    three = "\n".join(f"{a}\t{b}\t{c:g}" for a, b, c in zip(ref.ci[:200], ref.cj[:200], ref.cm[:200])).encode()
    bedpe = "\n".join(f"chr22\t{a * 10000}\t{a * 10000 + 10000}\tchr22\t{b * 10000}\t{b * 10000 + 10000}\t{c:g}"
                      for a, b, c in zip(ref.ci[:100], ref.cj[:100], ref.cm[:100])).encode()
    assert I.classify_text(three) == "contacts" and len(I.read_contacts(three, "c.txt", CH)[0]) == 200
    assert I.classify_text(bedpe) == "contacts" and len(I.read_contacts(bedpe, "c.tsv", CH)[0]) == 100
    assert I.classify_text(b"0.1,0.2,0.3\n1.5,2.5,3.5\n") == "unknown"            # float triples are coordinates
    bg = "\n".join(f"chr22\t{s}\t{s + 5000}\t{2.0}" for s in range(0, 100_000, 5000)).encode()
    arr, _ = I.read_track(bg, "t.bedGraph", CH)
    assert len(arr) == CH.n_bins and np.allclose(arr[:10], 2.0) and np.isnan(arr[20])
    bed, _ = I.read_track(b"chr22\t0\t15000\tp\t4\n", "p.bed", CH)
    assert bed[0] == pytest.approx(4.0) and bed[1] == pytest.approx(2.0)
    with pytest.raises(ValueError):
        I.read_track(b"chr1\t0\t100\t1\n", "t.bedGraph", CH)
    try:
        import hicstraw  # noqa: F401
    except ImportError:
        with pytest.raises(ValueError, match="hic"):
            I.read_contacts(b"HIC", "x.hic", CH)


def test_loader_accepts_contact_map_as_graph(ref, tmp_path):
    from ui.common import load_dataset
    buf = io.BytesIO()
    np.save(buf, ref.coords.astype(np.float32))
    table = "\n".join(f"{a}\t{b}\t{c:g}" for a, b, c in zip(ref.ci, ref.cj, ref.cm)).encode()
    ds = load_dataset("chr22", 7, None, ("c", "c.npy", buf.getvalue()), "nm", ("patient contacts", "hic.tsv", table), False)
    assert ds.ci.size == len(ref.ci) and "contacts from patient contacts" in ds.tracks_label


# ---------------------------------------------------------------- snapshots, PDF, agent extras
def test_snapshot_png_and_gif(ref):
    img = SN.render(ref.coords[:500], SN.normalise(np.arange(500.0)), T.SCALES["Residue Index Spectrum"],
                    ref.valid[:500], size=(320, 240), title="t", scale_nm=SN.nice_scale(ref.coords[:500]))
    png = SN.png(img)
    assert png[:8] == b"\x89PNG\r\n\x1a\n" and img.size == (320, 240)
    assert SN.nice_scale(np.array([[0, 0, 0], [1900.0, 0, 0]])) == 200
    gif = SN.gif(np.stack([ref.coords[:200]] * 3), SN.normalise(np.arange(200.0)), T.SCALES["Genomic position"],
                 ["a", "b", "c"], size=(160, 120))
    assert gif[:6] in (b"GIF89a", b"GIF87a")


def _ctx(ref, extras=None):
    m = A.compute_metrics(ref.coords[1800:2400], ref.epi[1800:2400], ref.valid[1800:2400], B0, CH, 1800)
    return A.AgentContext(A.DISEASE, True, "chr22", 10_000, "Custom window", "chr22:18,000,001-24,000,000",
                          "tumour.npy", "h3k27ac.npy", False, False, B0, m, ("BCR",),
                          (A.Comparison(A.HEALTHY, m, True, "ctrl"),), extras or {})


def test_agent_uses_genes_domains_and_drug_lab(ref):
    extras = {"neighbourhoods": {"source": "3D proximity", "tads": 12, "median_tad_mb": 0.4,
                                 "a_compartment_fraction": 0.4, "loops": 0, "contact_decay_gamma": 1.2},
              "genes_on_fold": {"genes_in_view": 50, "open": 10, "intermediate": 30, "buried": 10, "top_open": ["X1"],
                                "top_buried": ["Y1"], "flagged_disease_genes": [{"gene": "BCR", "category": "cancer",
                                                                                 "status": G.BURIED}]},
              "drug_lab": {"drug": "BET bromodomain inhibitor", "mode": "toward healthy baseline", "region": "chr22:18-26 Mb",
                           "restoration_pct": 26.0, "best_drug": "BET bromodomain inhibitor"}}
    text = A.heuristic_analysis(_ctx(ref, extras))
    assert "12 TAD-like" in text and "BCR (buried)" in text and "restored 26%" in text
    assert "additional_analyses" in _ctx(ref, extras).payload()


def test_pdf_dossier(ref):
    ctx = _ctx(ref)
    tab = G.accessibility_table(ref.coords[1800:2400], ref.epi[1800:2400], ref.valid[1800:2400], B0, CH, 1800)
    png = SN.png(SN.render(ref.coords[1800:2400], SN.normalise(np.arange(600.0)), T.SCALES["Genomic position"],
                           size=(400, 300)))
    therapy = {"drug": "BET", "mode": "mechanism only", "efficacy": 0.8, "plain": "p",
               "table": pd.DataFrame({"dose_pct": [0, 50, 100], "rg_nm": [400, 390, 380], "nu": [0.4] * 3,
                                      "gamma": [1.3] * 3, "packing": [0.1] * 3})}
    pdf = pdf_report.build(ctx, A.heuristic_analysis(ctx, "why?") + "\n_italic_ **bold** α≈β → γ ≥ 1 🤖",
                           "offline", "why?", [("view", png)], physics.local_density(ref.coords[1800:2400], 1.5 * B0),
                           tab, G.summary(tab), {"tads": 3, "source": "3D proximity"}, therapy)
    assert pdf[:5] == b"%PDF-" and len(pdf) > 20_000


# ---------------------------------------------------------------- API key from secrets / environment
def test_stored_key_from_environment(monkeypatch):
    from ui import agent_panel
    monkeypatch.setattr(agent_panel, "_has_secrets_file", lambda: False)
    for k in agent_panel.KEY_NAMES:
        monkeypatch.delenv(k, raising=False)
    assert agent_panel.stored_key() == ("", None, "")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    key, provider, name = agent_panel.stored_key()
    assert key == "sk-or-test" and provider == A.OPENROUTER and name == "OPENROUTER_API_KEY"


def test_plotly_js_served_statically(tmp_path, monkeypatch):
    from ui import sync_view
    monkeypatch.setattr(sync_view, "STATIC_DIR", tmp_path)
    url = sync_view.ensure_plotly_js()
    assert url == f"app/static/{sync_view.JS_NAME}" and (tmp_path / sync_view.JS_NAME).stat().st_size > 1_000_000


# ---------------------------------------------------------------- every page, end to end
def test_every_page_with_demo_patients(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    import ui.common as C
    import ui.states_panel as SP
    monkeypatch.setattr(C, "SLOT_ROOT", tmp_path / "empty")
    monkeypatch.setattr(SP, "SLOT_ROOT", tmp_path / "empty")
    monkeypatch.setattr(SP, "DEMO_ROOT", tmp_path / "demo")
    app = str(Path(__file__).resolve().parent.parent / "app.py")
    at = AppTest.from_file(app, default_timeout=600)
    at.run()
    ok = lambda: not at.exception and "cc-card-warn" not in "".join(m.value for m in at.markdown)  # noqa: E731
    assert ok(), [e.value for e in at.exception]
    at.sidebar.toggle(key="demo_patients").set_value(True).run()
    at.sidebar.selectbox(key="bio_state").set_value(A.DISEASE).run()
    assert ok()
    for mode in ("A/B compartment", "TAD domains"):
        at.selectbox(key="disp_colour").set_value(mode).run()
        assert ok(), mode
    for page in ("4D dynamics", "Compare", "Drug lab", "Genes", "Guide", "3D structure"):
        at.segmented_control(key="workspace").set_value(page).run()
        assert ok(), (page, [e.value for e in at.exception])
        if page == "Drug lab":
            assert at.radio(key="lab_drug").value == "bet"                     # best match pre-selected
        if page == "Genes":
            next(s for s in at.selectbox if (s.key or "").startswith("genes_pick")).set_value("BCR").run()
            assert ok()
    build = next(b for b in at.button if b.key == "agent_build_3d")
    build.click().run()
    assert ok() and any((d.label or "").startswith("Research dossier") for d in at.get("download_button"))
