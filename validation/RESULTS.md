# ChronoCell-5D: accuracy against real microscopy

**Real data.** The ground truth is Bintu et al., *Science* 2018 (chromatin tracing). It gives the
measured 3D position, in nanometres, of every 30 kb piece of DNA, in thousands of individual human
cells. Public data: github.com/BogdanBintu/ChromatinImaging.

**Rerun:**
- `python validation/validate_tracing.py` runs the test datasets (about 2 minutes).
- `python validation/validate_tracing.py --practice` runs the tuning datasets.

On the first run the data downloads into `validation/data/`, which is not committed. Raw numbers are
in `validation/results.json` and `validation/results_practice.json`.

## How it was measured

1. **Split** the cells of each experiment at random into half A and half B.
2. **Input:** from half A, keep only how often each pair of DNA pieces touches (closer than 150 nm).
   This is the same kind of information Hi-C gives. Every distance in half A is thrown away.
3. **Build:** ChronoCell-5D builds its 3D model from those touch frequencies alone.
4. **Answer key:** half B's measured median distance between every pair of pieces. The model never
   saw half B.
5. **Score (microscopy accuracy):** rank agreement with the answer key (Spearman ρ) after removing
   the obvious "further along the DNA = further apart" trend. It is expressed as a fraction of the
   **ceiling**, which is how well the two halves of the real experiment agree with each other.
   - Three datasets × three random splits.
   - **Overall = Σ model / Σ ceiling.** This rule was fixed before any tuning, so the noisy
     low-structure dataset cannot dominate.

## Two scores, never mixed

| Score | What it compares | What it proves |
|---|---|---|
| **Contact-map fit** | the model's contacts vs the **input** (half A) | the fit converged. It is *not* evidence of accuracy: any good optimiser scores high here. |
| **Microscopy accuracy** | the model's distances vs **unseen** measurements (half B) | whether the 3D model is right |

## Held-out protocol (why the number is credible)

- **Tuning used only practice datasets:** K562 28–30 Mb, HCT116 28–30 Mb (untreated and 6 h auxin),
  and HCT116 34–37 Mb.
- **The three test datasets below played no part in any design choice.** They were run once, after
  the settings were frozen (see `UPDATES.md`, 28 September 2026).

## Results on the held-out test datasets (mean over 3 splits)

| Dataset (cells) | v3.2 single structure | **v3.3 population model** | 100 sampled trajectories | No 3D (direct inversion) | Ceiling ρ |
|---|---|---|---|---|---|
| IMR90 chr21:28–30 Mb (4,832) | 39 % | **88 %** | 88 % | 81 % | 0.98 |
| A549 chr21:28–30 Mb (3,941) | 54 % | **91 %** | 91 % | 88 % | 0.95 |
| IMR90 chr21:18–20 Mb (1,277) | 35 % | **54 %** (±11 points across splits) | 53 % | 42 % | 0.25 |
| **Overall (Σ model / Σ ceiling)** | **45 %** | **85.6 %** | **85.6 %** | 79.7 % | |

The unweighted mean of the three percentages is 78 %.

| Other measures, v3.3 | IMR90 28–30 | A549 | IMR90 18–20 |
|---|---|---|---|
| Contact-map fit (ρ vs input) | 0.99 | 0.99 | 0.97 |
| Raw ρ, model (trend kept) | **0.976** | **0.952** | 0.868 |
| Raw ρ, genomic-distance baseline | 0.930 | 0.915 | **0.962** |
| Lin's CCC in nm (1 = exact sizes) | 0.97 | 0.93 | 0.46 |
| Model / real size | 1.00 | 0.92 | 0.82 |
| Cell-to-cell spread (CV), model vs measured | 0.42 vs 0.50 | 0.42 vs 0.55 | 0.42 vs 0.58 |

Practice datasets, for comparison (Σ/Σ over 4 datasets × 3 splits): v3.2 48 %, v3.3 92 %.

## What the v3.3 model is

- A **population of structures**, not one fold. Every cell folds differently, and Hi-C and imaging
  average over thousands of cells.
- It is a maximum-entropy Gaussian polymer ensemble. The method follows HIPPS/DIMES (Shi &
  Thirumalai, PRX 2019; Nat Commun 2023) and is not a new method of ours.
- Each contact frequency is converted to a pair spread. A valid 3D ensemble is then fitted to all
  pairs at once, with each pair weighted by how reliably it was measured.
- 100 Langevin trajectories are drawn exactly from that ensemble.
- The prediction is the ensemble's median distance. It is computed exactly, and checked against the
  100 sampled trajectories, which give the same score.
- Code: `chronocell/ensemble.py`; tests: `tests/test_v33.py`.

## Honest reading

- **On regions with real structure the target is met.** The model recovers 88–91 % of the structure
  the experiment itself can reproduce, up from 39–54 % in v3.2.
  - Absolute sizes are now right: CCC 0.93–0.97, where v3.2 was about 3× too small.
  - Overall rank agreement now beats the genomic-distance baseline.
- **The 3D population adds real information.** It beats inverting each contact frequency on its own
  (no 3D) on every dataset, because fitting all pairs jointly corrects noisy ones.
- **Where the data are weak, the model is weak.** On IMR90 18–20 Mb, even the experiment's two
  halves barely agree once the trend is removed (ceiling 0.25). There the model reaches 54 %, and:
  - it swings ±11 points between splits;
  - its overall rank agreement (0.87) is **below** the genomic-distance baseline (0.96).
- **Cells vary more than the model.** Real cell-to-cell spread (CV 0.50–0.58) is larger than the
  Gaussian model's fixed 0.42. The model gets population medians right, not the full shape of each
  pair's distance distribution.
- **Limits:**
  - The input is imaging-derived contact frequencies at a known 150 nm radius, not sequencing Hi-C.
    For Hi-C the probability scale must be assumed (`ensemble.counts_to_probability`), and a direct
    Hi-C → imaging test is still to do.
  - The population model is intended for windows of up to a few hundred beads.
  - Sampled single structures have no excluded volume.
