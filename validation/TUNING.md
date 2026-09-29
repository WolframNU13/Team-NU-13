# Tuning log: practice datasets only

Every design choice for v3.3 was made on **practice datasets** that are not part of the reported
validation. The test datasets (IMR90 chr21:28–30 Mb, IMR90 chr21:18–20 Mb, A549 chr21:28–30 Mb) were
run once, after the settings below were frozen. The time-stamped record is in `UPDATES.md`
(28 September 2026).

- **Practice datasets:** Bintu et al. 2018:
  - K562 chr21:28–30 Mb;
  - HCT116 chr21:28–30 Mb, untreated and 6 h auxin (cohesin depleted);
  - HCT116 chr21:34–37 Mb, untreated.
- **Excluded:** the IMR90 cell-cycle set, because it shares cell line and region with a test set.
- **Score:** trend-removed Spearman ρ against half B as a % of the ceiling. Overall = Σ model / Σ
  ceiling.

## 1. Where the v3.2 pipeline loses accuracy (split 0)

| | K562 | HCT116 28–30 |
|---|---|---|
| Contact frequencies alone (f^-1/3, no 3D) | 88 % | 82 % |
| Best single 3D structure from half A's *true* medians | 63 % | 56 % |
| v3.2: shortest-path MDS start only | 74 % | 75 % |
| v3.2: after gradient fit | 37 % | 53 % |
| v3.2: full, with EGNN | 37 % | 55 % |

**Conclusion.** The input carries more than 80 % of the reproducible structure. One 3D structure
cannot hold it (cap about 60–75 %), and v3.2's gradient stage loses accuracy relative to its own MDS
start. This motivates a population model.

## 2. Langevin contact-potential ensemble (abandoned)

- **Setup:** 100 replica chains, overdamped Langevin, pair potentials −ε·φ(d) learned so the
  ensemble's contact frequencies match the input.
- **Result:** overall 63.2 % (K562 81 %, HCT116 75 %, HCT116 + auxin **19 %**, HCT116 34–37 Mb 78 %).
- **Cause:** about 60 time units were simulated against a chain relaxation time of about 140, so the
  ensemble never equilibrated. The model's long-range contact frequencies ran about 50 % above the
  input, and the contact-map fit was only 0.77.

## 3. Maximum-entropy Gaussian ensemble (adopted: `chronocell/ensemble.py`)

This is the HIPPS/DIMES approach (Shi & Thirumalai, PRX 2019; Nat Commun 2023).

| Variant (split 0) | Overall |
|---|---|
| Direct Maxwell inversion (no 3D) | 86.7 % |
| Gaussian ensemble, PSD projection only | 73.1 % |
| Gaussian ensemble, refined, binomial weights | **91.3 %** |
| Refined, 100 sampled structures (independent draws) | 78.6 % |

| Setting (splits 0 / 1) | Overall |
|---|---|
| Binomial weights, 1,500 iterations | 91.3 % / 92.2 % |
| Binomial weights, 4,000 iterations | 91.3 % / 92.1 % |
| Uniform weights, 1,500 iterations | 91.1 % / 91.9 % |

**Frozen:** binomial weights, 1,500 iterations, learning rate 0.01, 100 trajectories × 50 frames.

The final practice run (`python validation/validate_tracing.py --practice`, 3 splits) gave:

| Model | Overall |
|---|---|
| v3.2 | 48.3 % |
| v3.3 ensemble | **92.0 %** |
| v3.3, 100 trajectories | 91.9 % |
| No 3D | 87.5 % |

## 4. Phase 2 physics terms on the v3.2 single structure (split 0)

| `egnn.FitConfig` | Overall | K562 / HCT116 / auxin / 34–37 |
|---|---|---|
| Default | 47.7 % | 37 / 55 / 47 / 52 |
| Bending stiffness λ = 1, cos θ₀ = 0 | 46.4 % | 37 / 53 / 47 / 49 |
| Bending stiffness λ = 1, cos θ₀ = 0.3 | 48.2 % | 44 / 52 / 48 / 48 |
| Nuclear confinement λ = 10, R = 300 nm | 47.7 % | unchanged: every bead already inside R |
| Nuclear confinement λ = 10, R = 500 nm | 47.7 % | unchanged |

**Conclusion.** Neither term changes accuracy beyond noise (±1–2 points), so both stay **off by
default**. They remain available for simulation and what-if use (`lambda_bend`, `lambda_confine`).

They are not added to the population model. Any quadratic prior on a Gaussian ensemble is absorbed by
the fitted couplings, and a hard wall would break the exact Gaussian statistics.
