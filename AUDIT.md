# ChronoCell-5D — audit and rewrite (v1 → v2)

**v3 note.** Errors fixed in v3 (custom-window crash, empty-window fit, etc.) are listed in APP_GUIDE.md §1.

**v3.2 note.** New pages (Compare, Drug lab, Genes, Guide), neighbourhoods, patient-data formats, PDF dossier and the fixes found while building them are in APP_GUIDE.md §15–19. docs/OVERVIEW.md explains the project in plain language.

**v3.1 note.** APP_GUIDE.md §1 lists the v3.1 fixes: the stale-module `MAIN_CHROMOSOMES` crash, the unreachable sidebar, silently dropped files, graph failures masking structures, and tracks mistaken for structures. §12–14 cover the biological-state engine, ChronoAgent and the dashboard.

**Scope.** The team specification's code blocks (Teammate 1 graph assembly, Teammate 2 EGNN training, Teammate 3 export) from *AI-Powered Codon Optimization for Vaccines.pdf* (both versions), and the v1 Streamlit app (kept for reference at `legacy/app_v1.py`).

**Method.** Every finding below was reproduced or measured in this repository. The fixes are covered by 21 tests (`python -m pytest`, all passing) and the benchmark in `python -m chronocell.benchmark`. Where a number is quoted, it was measured.

---

## 1. Critical flaws

Severity: **S1** = wrong results or crash · **S2** = misleading or unstable · **S3** = hygiene.

### [Math/Physics Error]

| # | Sev | Where | Flaw | Consequence | Fix |
|---|---|---|---|---|---|
| P1 | S1 | T2 loss | Contact target `d* = 1/(M+ε)` is an inverse-*linear* map on raw read counts. It has no power law and no length scale. | Targets depend on sequencing depth: at 1,000 reads a pair is "0.001 units" apart. They are not physical distances. | `physics.contact_target_distance`: `d* = b₀(M/M_ref)^(−1/α)`, α = 3 (capture volume; PASTIS default). `M_ref` is the median nearest-neighbour count, so M = M_ref ⇒ d* = b₀. Targets are clipped to [d_min, 8 b₀]. |
| P2 | S1 | T1 edges | `np.where(matrix > 0)` on the full symmetric matrix keeps the diagonal and both triangles. | Self-contacts ask for d = 0 (see D4), and every contact is double-weighted. | `formats.canonical_contacts`: i < j, no diagonal, mirrored pixels merged (tested). |
| P3 | S1 | T2 loss | `L_smooth = mean‖Δx‖²` has no rest length, and the code's mean over coordinate components is 1/3 of the documented formula. | Its minimum is a collapsed chain (all bonds 0), which fights the steric term by construction. | Harmonic backbone `mean(((‖Δx‖ − b₀)/b₀)²)` (`physics.loss_smooth`, `egnn.smooth_loss`). |
| P4 | S1 | whole pipeline | Coordinates are labelled Å. The v1 synthetic model used a 3.8 Å bond (the protein Cα–Cα distance). | A 10 kb bead is ≈ 50 nm = 500 Å, so v1 was about 130× too small. Every R_g and distance read-out carried a wrong unit. | Nanometres throughout, b₀ = 50 nm (30 nm-fibre coarse-graining at 3 kb per 30 nm, scaled to 10 kb; exposed as a parameter). Arbitrary model units are calibrated to b₀, never relabelled. |
| P5 | S1 | T2 | `d_min = 0.5` is unrelated to the target scale (P1 targets reach 10⁻³). | Contact and steric terms are mutually unsatisfiable. | d_min = 0.8 b₀, and targets are clipped to ≥ d_min so the terms cannot contradict each other. |
| P6 | S2 | T2 / spec | Three different steric objectives: a raw sum (summary), `1/|Ω|` (formula v2, which dilutes by ~N²/2), and a mean over 1,000 random pairs with \|i−j\| ∈ [3,10) (code). | The trained term never sees long-range overlaps; it is a biased, high-variance estimator of the wrong quantity. | Exact soft-core barrier over all \|i−j\| > 1, summed and divided by N, evaluated on an O(N) cell-list / Verlet neighbour list (`physics.neighbor_pairs`, verified against brute force). |
| P7 | S2 | T1 f_GC | Divides by B = 10,000 even for the 8,468 bp final bin and for N runs. | Assembly gaps read as 0 % GC (AT-rich), not as missing. **1,164 of 5,082 bins** are > 50 % N in hg38. | `features.gc_fraction`: GC/(A+C+G+T), NaN below 50 % called bases; mask from the UCSC `gap` track. |
| P8 | S2 | T1 f_epi | `nan_to_num` on bigWig values. | "No coverage" becomes "no signal"; the raw unbounded ChIP scale is fed to an MLP. | NaN-aware binned mean (`features.binned_mean`); log1p + z-score in `egnn.node_features`. |
| P9 | S2 | T1 fallback | Synthetic contacts are uniform random pairs. | No distance decay, so the model learns noise. | `synthetic.simulate_contacts`: Poisson counts with mean ∝ d^−α from a planted fractal globule, so reconstructions can be *scored*. |
| P10 | S2 | v1 app | Centromere preset at 12.2–17.4 Mb. | These are hg19 values. hg38 `acen` is **13.7–17.4 Mb** (UCSC cytoBand). | `genome.py` is built from UCSC hg38 cytoBand / gap / centromeres API records. |
| P11 | S2 | v1 app | ν fitted on mean ⟨R⟩ rather than RMS √⟨R²⟩, with no SE, R² or regime; "R_e²/R_g² ideal = 6" shown as a generic target; "inferred contact 1/d − ε" heat map. | Over-confident or non-physical read-outs. | `physics.distance_scaling`: RMS, log-spaced s, fit window [4, N/10], OLS SE and R², regime classification (1/3, 1/2, 0.588). |
| P12 | S1 | optimisation | Random initial coordinates. | Misfolded minima: RMSD/R_g 0.53–0.65 (table §5). | Shortest-path MDS initialisation (ShRec3D-style), then phantom-chain → excluded-volume schedule: RMSD/R_g 0.28–0.32. |

### [DL/Equivariance Bug]

| # | Sev | Where | Flaw | Consequence | Fix |
|---|---|---|---|---|---|
| D0 | **S1** | T2 loop | `idx_b = idx_a + randint(3, 10)` with `idx_a < N − 3` indexes past the last bead. | **Reproduced: `IndexError: index 5082 is out of bounds` at epoch 4. 45 % of epochs sample an out-of-range bead, so the 150/200-epoch run cannot complete.** | Deterministic Verlet neighbour list; no random index arithmetic. |
| D1 | S1 | EGNNLayer | `x_i += Σ_j (x_i − x_j)·φ_x(m)` uses the raw difference and a plain sum. | The step scales with distance × degree, and Micro-C hubs have hundreds of neighbours, so coordinates explode. | Unit direction (x_i − x_j)/(d_ij + ε), mean aggregation 1/\|N(i)\| (as C = 1/(M−1) in Satorras et al.), step bounded by s·tanh(φ_x). |
| D2 | S2 | EGNNLayer | Raw d² is an MLP input. | d² spans 10⁰–10⁶, which saturates SiLU and destabilises gradients. | Gaussian RBF expansion of d/b₀ (bounded, smooth, invariant). |
| D3 | S2 | EGNNLayer | Raw read counts are used as edge attributes. | The input scale varies by 10³–10⁴. | [log1p(M)/log1p(M_max), log1p\|i−j\|/log1p N, backbone flag]. |
| D4 | S1 | losses | `torch.norm` has no defined gradient at 0; the diagonal self-edges (P2) make d ≡ 0. | NaN gradients. | d = √(‖Δ‖² + ε²) everywhere (test: coincident beads give finite gradients). |
| D5 | S2 | naming / eval | Distance-only inputs make the network **E(3)**-equivariant (reflections included), not SE(3). Contact data cannot fix chirality. | "SE(3)" overstates the symmetry, and RMSD without reflection is biased. | Measured in float64: rotation 7e−15 and reflection 1.4e−14 coordinate error, with invariant features to 1e−15 (`egnn.equivariance_check`, in-app button). RMSD uses O(3) Kabsch. |
| D6 | S2 | T2 model | A single layer, and `h_out` is discarded. | GC / H3K27ac have almost no path to the coordinates. | 3 layers with residual, LayerNorm-ed invariant features. |
| D7 | S2 | design | Free coordinates `pos` are optimised jointly with an EGNN on one graph. | **The network is a reparameterisation of P, not a predictor.** Ablation (§5): EGNN refinement equals coordinate refinement to ≤ 0.001 RMSD/R_g at 2.5–3× the cost. It adds value only when trained across many structures (amortised inference). | EGNN kept as an optional stage 2 and reported honestly; `refine_with_egnn=False` gives the control arm. |
| D8 | S3 | T2 optimiser | AdamW's default weight decay (0.01) also applies to `pos`. | It biases the structure towards the origin (a ~1 % shrink over 200 epochs). | Separate parameter groups; no decay on coordinates. |
| D9 | S2 | T2 init | `torch.randn` initial positions. | A unit-variance cloud: no chain, wrong scale. | MDS fold (default) or a unit-bond random walk. |
| D10 | S2 | T1/T2 memory | A dense 5,082² float64 matrix (207 MB), and message passing over every pixel. | O(E·hidden) activations per layer. | Sparse cooler pixel table; message graph = backbone (\|i−j\| ≤ 2) + top-8 contacts per bead; all contacts still supervise the loss. |
| D11 | S2 | v1 app | Automatic `torch.load(weights_only=False)` fallback on uploads. | Arbitrary code execution from a crafted .pt file. | Opt-in "trust" switch; weights_only first. |
| D12 | S3 | T2 | No gradient clipping and no NaN guard. | Silent divergence. | Clip-norm 5; training raises on any non-finite loss. |

### [PDB Spec Violation]

| # | Sev | Where | Flaw | Fix |
|---|---|---|---|---|
| F1 | S1 | T3 / v1 | Real-scale nm coordinates centred on the origin reach about −1,000. `%8.3f` holds only −999.999…9999.999, so the columns silently shift (T3 has no check). | Translate into the positive octant (restoring the full 0–9999.999 range), scale by 10ᵏ only if still too large, record unit and offset in REMARK 250. `formats.read_pdb` inverts it exactly (round-trip test, < 1e−3 nm). |
| F2 | S2 | T3 | Model-unit values written into Å columns. | Unit and scale are explicit in REMARK 250 ("NM PER FILE UNIT (NOT A)"). |
| F3 | S2 | v1 | Free text in REMARK 1 (reserved for references), REMARK 2 (controlled `RESOLUTION.` syntax) and REMARK 3 (refinement). | `REMARK   2 RESOLUTION. NOT APPLICABLE.` plus REMARK 250 key/value experimental details. |
| F4 | S2 | T3 / v1 | CONECT is written one-way (i → i+1). | Every bead lists both sequence neighbours; the validator checks each bond is sequential and reciprocated. |
| F5 | S2 | v1 | Serial and resSeq wrapped modulo 10⁵ / 10⁴, silently corrupting IDs. | Hard error if a field overflows. |
| F6 | S3 | v1 | B-factor normalised by the export window's maximum, so it can't be compared across exports. | Fixed chromosome-wide map: 99.99·ln(1+f_epi)/ln(1+P99.5). |
| F7 | S3 | T3 | HEADER without date/idCode; no TER, CRYST1 or segID; ragged lines. | All records at exact columns, padded to 80; `formats.validate_pdb` checks record columns, decimal-point positions, separators, element justification, serial sequence and CONECT integrity. |

### [UI/UX Anti-Pattern]

| # | v1 | v2 |
|---|---|---|
| U1 | Neon cyan glows, pulsing dot, gradient text, emoji branding | Paper / ink / single cobalt accent from the references; hairlines instead of glow |
| U2 | Static "GPU: Active" badge (a claim, not a measurement) | Measured facts only: bead, assembled and contact counts; torch availability under About |
| U3 | Monospace on labels and cards | Inter Tight for UI; IBM Plex Mono only for loci and numeric values |
| U4 | KPI glow-card grid; analytics hidden in four tabs away from the structure | Dominant viewport + scrollable inspector (Polymer physics, Genomic features, Model & convergence, Export) |
| U5 | Dual-axis GC vs H3K27ac (unrelated units coupled by an arbitrary scale) | Aligned small multiples |
| U6 | Spectral/Turbo rainbows on near-black; gaps coloured as data | Calibrated scales starting clear of the paper colour; a distinct ghost state for unassembled bins and a context state for non-highlighted beads |
| U7 | Any display change re-ran the whole app, physics included | `st.fragment` stage: display changes re-render only the figure; physics cached by coordinates |
| U8 | `#5d6b82` on `#1f293d` = **2.70:1** (fails AA) | Measured: ink 15.0:1, ink-2 8.0:1, muted 5.3:1, accent 6.7:1 |
| U9 | Tooltip in Å; no cytoband; gap bins shown as data | Locus tooltip: exact bp (float64 — float32 rounds above 16.7 Mb), bin, band, GC, H3K27ac, or "unassembled" |
| U10 | Simulated loss curve shown next to real metrics | Only real fits are plotted |
| U11 | Fixed-pixel line "tube" | Lit triangle-mesh tube, rotation-minimising frames, adaptive LOD (≤ 70k vertices for the whole chromosome) |
| U12 | CSS keyed to internal test IDs | Styling scoped to `st.container(key=…)` hooks and theme config |

---

## 2. Corrected formulation (as implemented)

```
Lengths in nm; losses dimensionless.  b0 = 50 nm, alpha = 3, d_min = 0.8 b0.

Contact target      d*_ij = clip( b0 (M_ij / M_ref)^(-1/alpha), d_min, 8 b0 ),   M_ref = median M_{i,i+1}
L_contact           = (1/|C|) sum_C ((d_ij - d*_ij) / d*_ij)^2                 relative stress
L_smooth            = (1/(N-1)) sum_i ((||x_{i+1} - x_i|| - b0) / b0)^2         harmonic backbone
L_steric            = (1/N) sum_{|i-j|>1} ([d_min - d_ij]_+ / b0)^2              soft-core excluded volume
L_total             = L_contact + 1.0 L_smooth + 20 L_steric
d_ij                = sqrt(||x_i - x_j||^2 + eps^2)                               smooth at 0

EGNN layer          m_ij = phi_e(h_i, h_j, RBF(d_ij / b0), a_ij)
                    x_i' = x_i + (1/|N(i)|) sum_j (x_i - x_j)/(d_ij + eps) * s tanh(phi_x(m_ij))
                    h_i' = LayerNorm(h_i + phi_h(h_i, (1/|N(i)|) sum_j m_ij))

R_g^2               = (1/N) sum ||x_i - x_cm||^2  ( = (1/2N^2) sum_ij ||x_i - x_j||^2, tested )
R(s)                = sqrt(<||x_{i+s} - x_i||^2>),  nu = OLS slope of log R vs log s, s in [4, N/10]
```

Code: `chronocell/physics.py` (NumPy, the reference implementation), `chronocell/egnn.py` (differentiable twins, model and training), `chronocell/formats.py`, `chronocell/features.py`.

## 3. Architecture

```
chronocell/genome.py      hg38 chr22 constants from UCSC (cytoBand, gap, centromeres)
chronocell/physics.py     metrics, scaling fit, cell-list neighbours, contact mapping, losses, Kabsch
chronocell/egnn.py        E(3)-EGNN, message graph, MDS init, two-stage fit, equivariance check
chronocell/synthetic.py   Hilbert fractal globule, band-informed tracks, Poisson contacts
chronocell/features.py    vectorised f_GC / f_epi binning, contact extraction, H3K27ac hubs
chronocell/formats.py     wwPDB writer + validator + reader, XYZ, graph/structure loaders
chronocell/viz.py         Plotly builders (tube mesh, charts) — no physics
chronocell/theme.py       design tokens + stylesheet
chronocell/build_graph.py CLI: FASTA + bigWig + mcool -> graph .npz   (replaces Teammate 1)
chronocell/train.py       CLI: graph -> coordinates, history, PDB     (replaces Teammate 2/3)
app.py                    Streamlit shell: state, layout, caching
```

Numerics never import Streamlit. The app caches the dataset (`cache_resource`), per-window physics (`cache_data`, keyed on the coordinates), and hover labels. The 3D stage is an `st.fragment`.

## 4. Verification (`python -m pytest`: 21 passed)

- Genome: 5,082 bins with an 8,468 bp final bin; the gap mask matches UCSC (bins 0–1,050 and 5,081 unassembled).
- R_g centre-of-mass form equals the pairwise form (rel. 1e−12). ν = 1.00 on a rod and 0.489 on a 20,000-step ideal random walk (theory 0.5).
- The synthetic globule is in the fractal-globule regime (ν = 0.367, R² 0.989) with 0 overlaps.
- The cell list matches brute force exactly.
- Contact targets are finite, monotone, anchored (M_ref ↦ b₀), ≥ d_min, and zero counts are rejected.
- The harmonic backbone has its minimum at b₀.
- Kabsch recovers a mirror image only when reflection is allowed.
- PDB: exact columns, validator passes on window and whole-chromosome exports, catches a one-column shift, and round-trips to nm.
- Canonical contacts drop the diagonal and merge mirrored pixels. f_GC ignores N and partial bins; f_epi is NaN-aware.
- EGNN is E(3)-equivariant (< 1e−10), gradients stay finite for coincident beads, the MDS init recovers the global fold (r > 0.85), and the fit recovers a planted structure.

## 5. Reconstruction benchmark (`python -m chronocell.benchmark`, CPU)

Planted fractal globule (seed 7, 70,466 simulated contacts). The "truth" column is L_contact evaluated at the true structure: the Poisson noise floor.

| window (bins) | beads | arm | RMSD/R_g | distance r | overlaps | L_contact (truth) | time (s) |
|---|---|---|---|---|---|---|---|
| 1400–2199 | 800 | random walk, no refine | 0.645 | 0.726 | 0 | 0.0270 (0.0142) | 5.7 |
| 1400–2199 | 800 | MDS, no refine | 0.297 | 0.961 | 0 | 0.0145 (0.0142) | 4.9 |
| 1400–2199 | 800 | MDS + coordinate refine | 0.291 | 0.963 | 0 | 0.0144 (0.0142) | 5.5 |
| 1400–2199 | 800 | MDS + EGNN refine | 0.292 | 0.962 | 0 | 0.0144 (0.0142) | 13.7 |
| 2600–3399 | 800 | random walk, no refine | 0.568 | 0.917 | 0 | 0.0226 (0.0141) | 3.5 |
| 2600–3399 | 800 | MDS + coordinate refine | 0.311 | 0.968 | 2 | 0.0198 (0.0141) | 6.0 |
| 2600–3399 | 800 | MDS + EGNN refine | 0.311 | 0.968 | 2 | 0.0198 (0.0141) | 14.5 |
| 3800–4599 | 800 | random walk, no refine | 0.619 | 0.617 | 25 | 0.0487 (0.0140) | 3.8 |
| 3800–4599 | 800 | MDS + coordinate refine | 0.275 | 0.963 | 0 | 0.0166 (0.0140) | 6.3 |
| 3800–4599 | 800 | MDS + EGNN refine | 0.276 | 0.962 | 0 | 0.0166 (0.0140) | 14.6 |
| 2000–3599 | 1,600 | random walk, no refine | 0.571 | 0.900 | 22 | 0.0585 (0.0144) | 6.8 |
| 2000–3599 | 1,600 | MDS + coordinate refine | 0.285 | 0.958 | 12 | 0.0354 (0.0144) | 9.1 |
| 2000–3599 | 1,600 | MDS + EGNN refine | 0.285 | 0.958 | 10 | 0.0355 (0.0144) | 27.1 |
| 3000–3399 | 400 | random walk, no refine | 0.533 | 0.899 | 0 | 0.0250 (0.0133) | 2.7 |
| 3000–3399 | 400 | MDS + coordinate refine | 0.319 | 0.973 | 0 | 0.0151 (0.0133) | 3.3 |
| 3000–3399 | 400 | MDS + EGNN refine | 0.319 | 0.973 | 0 | 0.0151 (0.0133) | 8.5 |

**Reading.**
1. The initialisation is the decisive fix: RMSD/R_g falls from 0.53–0.65 to 0.28–0.32 in every window.
2. Where L_contact reaches the noise floor (window 1400), the residual ~0.3 R_g is data-limited: the fit explains the contacts as well as the truth does.
3. EGNN refinement is indistinguishable from coordinate refinement at 2.5–3× the cost (D7).
4. The 1,600-bead window stops above the floor (0.035 vs 0.014), so larger windows need more epochs than the in-app default.

## 6. Limitations (stated, not hidden)

- b₀ = 50 nm and α = 3 are literature anchors, not fitted values. With real data, check α against the measured P(s) exponent (γ ≈ α·ν).
- All tracks and contacts in the default view are synthetic and labelled as such. `build_graph.py` wires pyfaidx/pyBigWig/cooler; those packages and real chr22 files were not available here, so that glue is untested, while the per-bin functions it calls are unit-tested.
- In-app reconstruction is capped at 2,000 beads (~20 s per 800 beads on this CPU). Use `python -m chronocell.train` for the whole chromosome.
- A structure is determined only up to reflection (D5). Assembly gaps (1,164 bins) carry no data; their positions come from chain continuity alone and are drawn as ghosts.
