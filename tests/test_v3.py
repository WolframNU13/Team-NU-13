"""Regression tests for the v3 additions: window clamping, genome model, bundles, 4D scenarios."""

from __future__ import annotations

import io

import numpy as np
import pytest

from chronocell import formats, genome, physics, scenarios, synthetic
from ui.common import clamp_window


# ---------------------------------------------------------------- the Custom-window crash
@pytest.mark.parametrize("lo,hi,expected", [
    (2600, 3400, (2600, 3400)), (0, 10, (0, 10)), (5075, 5082, (5072, 5082)), (5081, 5082, (5072, 5082)),
    (3000, 2000, (2000, 3000)), (-50, 99_999, (0, 5082)), (None, None, (0, 5082)), (float("nan"), 7, (0, 10)),
    (4000, 4003, (4000, 4010)), (5082, 9000, (5072, 5082)),
])
def test_clamp_window_always_valid(lo, hi, expected):
    a, b = clamp_window(lo, hi, 5082)
    assert (a, b) == expected
    assert 0 <= a < b <= 5082 and b - a >= 10
    np.arange(5082)[a:b][0]                                   # slicing never produces an empty window


def test_clamp_window_tiny_dataset():
    assert clamp_window(3, 1, 5) == (0, 5)


# ---------------------------------------------------------------- genome model
def test_all_chromosomes_load_with_bounded_bead_counts():
    for name in genome.MAIN_CHROMOSOMES:
        ch = genome.chrom(name)
        assert ch.n_bins <= genome.MAX_DEFAULT_BEADS and ch.bands and ch.size == genome.chromosome_size(name)
    assert genome.chrom("chr22").resolution == 10_000 and genome.chrom("chr1").resolution == 50_000


def test_resolution_inferred_from_bead_count():
    assert genome.resolution_for_beads("chr4", genome.chrom("chr4", 25_000).n_bins) == 25_000


def test_gc_fraction_zero_length_bins_are_invalid():
    from chronocell import features
    f, valid = features.gc_fraction(b"GCGC" * 5, n_bins=3, resolution=20)
    assert valid.tolist() == [True, False, False] and np.isnan(f[1:]).all()


# ---------------------------------------------------------------- bundles and slot I/O
def test_bundle_roundtrip_multiframe_window():
    x = np.random.default_rng(0).normal(size=(3, 50, 3)) * 100
    b = formats.StructureBundle("chr22", 10_000, x, np.array([0, 6, 12.0]), ["0 h", "6 h", "12 h"],
                                condition="tumour", start_bin=2600, time_unit="hours")
    buf = io.BytesIO()
    formats.write_bundle(buf, b)
    back, notes = formats.read_bundle(buf.getvalue(), "t.npz")
    assert back.start_bin == 2600 and back.labels == b.labels and back.condition == "tumour"
    assert np.allclose(back.frames, x, atol=1e-3) and any("Window of 50 beads" in n for n in notes)


def test_bundle_rejects_window_past_chromosome_end():
    b = formats.StructureBundle("chr22", 10_000, np.zeros((1, 50, 3)) + np.arange(50)[None, :, None],
                                np.zeros(1), ["t0"], start_bin=5070)
    with pytest.raises(ValueError):
        b.validate()


def test_windowed_pdb_export_is_placed_back_on_its_loci():
    syn = synthetic.build(seed=7)
    pdb, _ = formats.write_pdb(syn.coords[2600:3000], 2600, syn.gc, syn.epi, 10.0, "t", "t")
    b, notes = formats.read_bundle(pdb.encode(), "w.pdb")
    assert (b.chrom, b.start_bin, b.resolution, b.n_beads) == ("chr22", 2600, 10_000, 400)


def test_pdb_other_chromosome_segment_and_resolution():
    ch = genome.chrom("chr4")
    x = synthetic.fractal_globule(300, b0=physics.bond_length_for(ch.resolution), seed=1)
    gc = np.full(ch.n_bins, 0.4)
    pdb, _ = formats.write_pdb(x, 1000, gc, gc, 1.0, "t", "t", chrom=ch)
    assert formats.validate_pdb(pdb).ok and "CH4 " in pdb and "ONE CA PER 40 KB BIN" in pdb


# ---------------------------------------------------------------- 4D scenarios
@pytest.fixture(scope="module")
def ref22():
    ch = genome.chrom("chr22")
    return ch, synthetic.fractal_globule(ch.n_bins, b0=50.0, seed=7)


@pytest.mark.parametrize("key", ["22q11del", "ph", "ewing"])
def test_disease_presets_relax_to_physical_chains(ref22, key):
    ch, x = ref22
    p = scenarios.PRESETS[key]
    tr = scenarios.simulate(x, ch, p.operation, scenarios.preset_region(p, ch), 50.0, p.title, p.summary, n_frames=8)
    assert tr.simulated and tr.n_frames == 8
    assert physics.bond_lengths(tr.frames[-1]).max() < 1.25 * 50.0          # junction bonds closed
    assert physics.bond_lengths(tr.frames[0]).max() > 2 * 50.0              # ...and they started open


def test_deletion_removes_exactly_the_22q11_span(ref22):
    ch, x = ref22
    reg = scenarios.preset_region(scenarios.PRESETS["22q11del"], ch)
    tr = scenarios.simulate(x, ch, "deletion", reg, 50.0, "d", "d", n_frames=4)
    assert tr.frames.shape[1] == ch.n_bins - (reg["b"] - reg["a"])
    assert not np.any((tr.bead_bin >= reg["a"]) & (tr.bead_bin < reg["b"]))


def test_translocation_partner_maps_to_partner_qter(ref22):
    ch, x = ref22
    reg = scenarios.preset_region(scenarios.PRESETS["ph"], ch)
    tr = scenarios.simulate(x, ch, "translocation", reg, 50.0, "t", "t", n_frames=4)
    partner = tr.bead_chrom == "chr9"
    assert partner.any() and tr.bead_bin[partner].max() == genome.chrom("chr9", 10_000).n_bins - 1
    assert genome.gene("ABL1").start // 10_000 == tr.bead_bin[partner].min()


def test_duplication_inserts_copies_and_trajectory_pdb_is_valid():
    ch = genome.chrom("chr4")
    x = synthetic.fractal_globule(ch.n_bins, b0=physics.bond_length_for(ch.resolution), seed=7)
    p = scenarios.PRESETS["snca3"]
    reg = scenarios.preset_region(p, ch)
    tr = scenarios.simulate(x, ch, p.operation, reg, physics.bond_length_for(ch.resolution), p.title, "", n_frames=4)
    assert tr.frames.shape[1] == ch.n_bins + 2 * (reg["b"] - reg["a"])
    segs = np.array([formats.segment_id(c) for c in tr.bead_chrom])
    chk = formats.validate_pdb(formats.write_pdb_trajectory(tr.frames, tr.bead_bin + 1, segs, ch, "t", "m"))
    assert chk.ok and chk.n_models == 4


def test_region_clamp_for_custom_scenarios():
    assert scenarios.clamp_region(90, 10, 100, 2) == (10, 90)
    assert scenarios.clamp_region(99, 500, 100, 2) == (98, 100)
