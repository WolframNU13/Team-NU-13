# ChronoCell-5D

3D chromatin workstation for human chr22 (GRCh38) at 10 kb: contact-embedding reconstruction refined by an E(3)-equivariant graph network, with polymer-physics analytics and wwPDB-conformant export.

```bash
pip install -r requirements.txt
streamlit run app.py
```

With no uploads, the app opens on a planted synthetic chromosome (a Hilbert-curve fractal globule with band-informed tracks and Poisson Micro-C contacts), so reconstructions can be scored against a known answer.

## Pipeline

```bash
# 1. graph from raw inputs (needs pyfaidx, pyBigWig, cooler) — or --synthetic
python -m chronocell.build_graph --fasta chr22.fa --bigwig H3K27ac_signal.bigWig --mcool human_microc.mcool --out graph_chr22.npz

# 2. reconstruct (whole chromosome or --start/--end window); writes .npz, _history.csv, .pdb
python -m chronocell.train --graph graph_chr22.npz --out predicted_coords.npz

# 3. open the results in the app: Data → Coordinates (predicted_coords.npz) and Graph (graph_chr22.npz)
```

The legacy `graph_data.pt` and `predicted_coords.pt` files are still accepted. Loading PyG objects requires ticking the "Allow unpickling" box.

Upload whole-chromosome structures (5,082 beads) to the app. Windowed CLI runs (`--start/--end`) are for offline analysis: their PDB and CSV exports carry the correct loci, but the app would map a windowed array onto bins starting at 0.

## Verification

```bash
python -m pytest                    # 21 tests: equivariance, physics identities, PDB columns, features
python -m chronocell.benchmark      # reconstruction accuracy vs planted structures
```

See [AUDIT.md](AUDIT.md) for the flaw-by-flaw audit of v1, the corrected equations, and the benchmark.
