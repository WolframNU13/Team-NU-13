# ChronoCell-5D

3D / 4D chromatin workstation for human chromosomes (GRCh38). It reconstructs folds with contact embedding refined by an E(3)-equivariant graph network, and adds:
- polymer-physics analytics;
- playback of time courses and disease states;
- simulated disease rearrangements;
- biological-state comparison (healthy, disease/cancer, senescent);
- the ChronoAgent interpreter;
- wwPDB-conformant export.

```bash
pip install -r requirements.txt
streamlit run app.py
```

* **Coordinates:** drop files into `coordinates/<chrom>/` or upload them under *Data*. Until then a clearly labelled reference model is shown. See `coordinates/README.md`.
* **Biological states:** in the sidebar, pick *Healthy Control*, *Disease State / Cancer* or *Senescent State*. Files are recognised by content: `.npy` (N, 3) coordinates, `.npy` (N,) signal tracks, `.pdb` structures. They are assigned to a state by the words in their path (e.g. `coordinates/chr22/healthy/…`), or by uploading them into a state. Try it with `python -m chronocell.demo_states demo_states`, then run with `CHRONOCELL_COORDINATES=demo_states`.
* **🤖 ChronoAgent:** a structural-genomics interpreter below the viewport.
  - Paste a free Google AI Studio (Gemini) or OpenRouter key into the sidebar's *AI API Key* for LLM answers.
  - Without a key, an offline heuristic engine answers instantly from the measured metrics.
  - It exports `ChronoCell_Analysis_Report.md` and the current state's PDB.
* **GPU reconstruction:** `colab/ChronoCell5D_Colab.ipynb` on a Colab T4. Its output unzips straight into `coordinates/`.
* **Full guide** (every screen, graph, file and function): [APP_GUIDE.md](APP_GUIDE.md). **Audit of v1 and the benchmark:** [AUDIT.md](AUDIT.md).

## Command line

```bash
python -m chronocell.build_graph --fasta chr22.fa --bigwig H3K27ac.bigWig --mcool sample.mcool --out graph.npz   # or --synthetic
python -m chronocell.train --graph graph.npz --out predicted_coords.npz
python -m chronocell.benchmark
python -m chronocell.demo_states demo_states   # synthetic state files (opt-in)
python -m pytest                    # 83 tests
```
