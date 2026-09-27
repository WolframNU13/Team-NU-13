"""Verification suite for the audited mathematics, physics and file formats."""

from __future__ import annotations

import numpy as np
import pytest

from chronocell import features, formats, genome, physics, synthetic


@pytest.fixture(scope="module")
def syn():
    return synthetic.build(seed=7)


# ---------------------------------------------------------------- genome constants
def test_bin_count_and_last_partial_bin():
    assert genome.N_BINS == 5082
    lengths = genome.bin_lengths()
    assert lengths[:-1].min() == lengths[:-1].max() == 10_000
    assert lengths[-1] == 8_468 and lengths.sum() == genome.CHROM_SIZE


def test_gap_mask_matches_ucsc_short_arm():
    mask = genome.assembled_mask()
    assert not mask[:1051].any()            # 0 - 10.51 Mb is telomere + short_arm N
    assert mask[1051] and not mask[-1]      # first assembled bin; final bin is telomere N


# ---------------------------------------------------------------- polymer metrics
def test_radius_of_gyration_equals_pairwise_form():
    x = np.random.default_rng(1).normal(size=(400, 3)) * 7
    d2 = ((x[:, None] - x[None]) ** 2).sum(-1)
    rg_pair = np.sqrt(d2.sum() / (2 * len(x) ** 2))
    assert physics.radius_of_gyration(x) == pytest.approx(rg_pair, rel=1e-12)


@pytest.mark.parametrize("kind,expected,tol", [("rod", 1.0, 0.01), ("ideal", 0.5, 0.06)])
def test_scaling_exponent_on_reference_chains(kind, expected, tol):
    rng = np.random.default_rng(3)
    n = 20_000
    if kind == "rod":
        x = np.arange(n)[:, None] * np.array([[1.0, 0, 0]])
    else:
        steps = rng.normal(size=(n, 3))
        x = np.cumsum(steps / np.linalg.norm(steps, axis=1, keepdims=True), axis=0)
    fit = physics.distance_scaling(x)
    assert fit.nu == pytest.approx(expected, abs=tol)


def test_synthetic_globule_is_physical(syn):
    x = syn.coords
    bonds = physics.bond_lengths(x)
    assert np.median(bonds) == pytest.approx(physics.B0_NM, rel=1e-6)
    fit = physics.distance_scaling(x)
    assert fit.regime == "fractal globule" and abs(fit.nu - 1 / 3) < 0.05
    assert physics.loss_steric(x).overlaps == 0


def test_cell_list_matches_brute_force():
    x = np.random.default_rng(5).normal(size=(900, 3)) * 5
    i, j, _ = physics.neighbor_pairs(x, 1.1, min_sep=2)
    d = np.linalg.norm(x[:, None] - x[None], axis=-1)
    bi, bj = np.nonzero(np.triu(d < 1.1, k=2))
    assert set(zip(i.tolist(), j.tolist())) == set(zip(bi.tolist(), bj.tolist()))


# ---------------------------------------------------------------- losses
def test_contact_targets_are_finite_monotone_and_anchored():
    cm = np.array([1.0, 5.0, 60.0, 500.0])
    d = physics.contact_target_distance(cm, m_ref=60.0)
    assert np.all(np.isfinite(d)) and np.all(np.diff(d) <= 0)
    assert d[2] == pytest.approx(physics.B0_NM)                     # M = M_ref  ->  d* = b0
    assert d.min() >= physics.D_MIN_FACTOR * physics.B0_NM          # never below excluded volume
    with pytest.raises(ValueError):
        physics.contact_target_distance(np.array([0.0, 3.0]), 60.0)


def test_smooth_loss_has_rest_length():
    x = np.arange(10)[:, None] * np.array([[physics.B0_NM, 0, 0]])
    assert physics.loss_smooth(x) == pytest.approx(0.0)
    assert physics.loss_smooth(x * 0.5) > 0                          # compression is penalised


def test_kabsch_recovers_mirror_image():
    x = np.random.default_rng(2).normal(size=(200, 3))
    mirrored = x * np.array([1, 1, -1]) + 3.0
    rmsd, _, reflected = physics.kabsch_rmsd(x, mirrored, allow_reflection=True)
    assert rmsd < 1e-9 and reflected
    assert physics.kabsch_rmsd(x, mirrored, allow_reflection=False)[0] > 0.1


# ---------------------------------------------------------------- formats
def _pdb(syn, lo=2000, hi=2300):
    ref = float(np.nanpercentile(syn.epi, 99.5))
    return formats.write_pdb(syn.coords[lo:hi], lo, syn.gc, syn.epi, ref, "test", "test")


def test_pdb_columns_and_conect(syn):
    text, frame = _pdb(syn)
    chk = formats.validate_pdb(text)
    assert chk.ok, chk.issues
    assert chk.n_atoms == 300 and chk.n_conect == 300
    atom = next(line for line in text.splitlines() if line.startswith("ATOM"))
    assert atom[0:6] == "ATOM  " and atom[12:16] == " CA " and atom[17:20] == "GNN" and atom[21] == "A"
    assert int(atom[22:26]) == 2001 and atom[76:78] == " C" and len(atom) == 80


def test_pdb_roundtrip_restores_nanometres(syn):
    text, _ = _pdb(syn)
    back, unit = formats.read_pdb(text)
    assert unit == "nm"
    assert np.abs(back - syn.coords[2000:2300]).max() < 1e-3 + 1e-9


def test_pdb_full_chromosome_fits_fixed_columns(syn):
    ref = float(np.nanpercentile(syn.epi, 99.5))
    text, frame = formats.write_pdb(syn.coords, 0, syn.gc, syn.epi, ref, "test", "test")
    assert formats.validate_pdb(text).ok


def test_validator_catches_broken_columns(syn):
    text, _ = _pdb(syn)
    lines = text.splitlines()
    k = next(i for i, line in enumerate(lines) if line.startswith("ATOM"))
    lines[k] = lines[k][:30] + " " + lines[k][30:79]                  # shift coordinates by one column
    assert not formats.validate_pdb("\n".join(lines)).ok


def test_canonical_contacts_drop_diagonal_and_mirror():
    i = np.array([0, 1, 2, 2, 3])
    j = np.array([0, 2, 1, 3, 2])
    m = np.array([9.0, 4.0, 4.0, 2.0, 2.0])
    ci, cj, cm, notes = formats.canonical_contacts(i, j, m, 5)
    assert list(zip(ci, cj, cm)) == [(1, 2, 4.0), (2, 3, 2.0)]
    assert any("diagonal" in s for s in notes)


# ---------------------------------------------------------------- features
def test_gc_fraction_ignores_n_and_partial_bins():
    seq = b"GGCC" * 5 + b"N" * 20 + b"atat" * 5            # bins of 20 bp: 100 % GC, all-N, 0 % GC
    f, valid = features.gc_fraction(seq, n_bins=3, resolution=20)
    assert f[0] == pytest.approx(1.0) and np.isnan(f[1]) and f[2] == pytest.approx(0.0)
    assert valid.tolist() == [True, False, True]


def test_binned_mean_is_nan_aware():
    v = np.array([1.0, 3.0, np.nan, np.nan, np.nan, 2.0])
    out = features.binned_mean(v, n_bins=3, resolution=2)
    assert out[0] == 2.0 and np.isnan(out[1]) and out[2] == 2.0
