# ChronoCell-5D — development log

Project: ChronoCell-5D, a contact-guided 3D/4D chromatin reconstruction and interpretation
workstation for human chromosomes (GRCh38).

Repository: `github.com/WolframNU13/Team-NU-13`
Context: HackBlitz 2.0, ALLEN Global / Global City International School, Bengaluru — team HBZ-13
("Team NU — Null Pointers"), 27 September 2026.

---

## About this document

**This log was reconstructed on 28 September 2026 from git history and filesystem timestamps. It
was not written contemporaneously.** That matters if it is ever relied on as a record, so the
evidence behind every entry is stated below rather than presented as memory.

**Evidence classes used:**

| Class | Strength | What it proves |
|---|---|---|
| Git commit timestamps (3 commits) | Strong | The listed files existed in that state at that moment |
| File modification times (mtime) | Weaker | When a file was **last written** — not when it was created |

**Known limits of the timestamps — read before citing any of them:**

- **mtime is last-write, not creation.** A file created at 10:30 and edited at 14:10 shows only
  14:10. Several files below were certainly started earlier than the time shown.
- **This repository lives in a OneDrive folder.** Cloud sync can rewrite mtimes. Git commit times
  are the only timestamps here not subject to that.
- **Commit times are when work was committed,** which trails when it was written.
- **Times are local (IST, UTC+05:30).**

**Attribution is missing and must be added by hand.** All three commits are authored
`Shivoham Pandey`, because one machine did the committing. The git record therefore does **not**
reflect who contributed which idea or which module. Every `Contributors:` line below is left blank
deliberately — fill them in together, from memory, while it is still fresh. Do not guess, and do not
let one person fill in the other's entries.

**Contributors to be named:**

- Shivoham Pandey — <pandey.sharma@gmail.com>
- _(co-originator of the concept — name, email, and contribution to be filled in)_
- Dhruv Kumar, Kaanishka S, Harismitha Srinath — HBZ-13 teammates; individual contributions to be
  recorded below where applicable

---

## 26 September 2026 — before the event

Work exists on this date, the evening before the hackathon build window. It is recorded here because
it is what the evidence shows.

**20:17 — 20:18 · Reference material collected**
`AI-Powered Codon Optimization for Vaccines.pdf`, and a second copy.
The project's starting brief was codon optimization, not chromatin. The pivot is visible in the
commit history below.
Contributors: _______________

**20:40 · First prototype**
`legacy/app_v1.py` — the version later retired into `legacy/`.
Contributors: _______________

**21:32 · `chronocell` package created**
`chronocell/__init__.py`.
Contributors: _______________

**21:48 · First tests**
`tests/test_core.py`.
Contributors: _______________

**22:27 · Benchmark harness**
`chronocell/benchmark.py`.
Contributors: _______________

**23:00 · Architecture written**
`ARCHITECTURE.md`. The system design predates the build window.
Contributors: _______________

---

## 27 September 2026 — event day

Event schedule for reference: report 08:15 · build 10:00–13:00 and 14:00–16:30 · judging 17:30–19:00.

### 09:39:21 — Commit `326c3d6` "Initial commit: ChronoCell - AI-Powered Codon Optimization platform"

25 files. **The commit message names codon optimization, but the contents are already the chromatin
system** — `egnn.py`, `genome.py`, `physics.py`, `build_graph.py`, `features.py`, `formats.py`,
`synthetic.py`, `train.py`, `viz.py`, `theme.py`, `benchmark.py`, plus `ARCHITECTURE.md`,
`AUDIT.md`, `README.md`, `app.py`, `pytest.ini`, and the two reference PDFs.

This is the clearest single piece of evidence for when the pivot from codon optimization to chromatin
folding happened: **before this commit**, i.e. on or before the evening of 26 September.

Contributors: _______________
Conception note — whose idea was the pivot, and when was it discussed: _______________

### Morning build block, 10:00–13:00

**10:37 · Reference genome data + synthetic generator** — `chronocell/data/hg38.json`, `synthetic.py`
**10:38 · Feature extraction** — `features.py`
**10:44 · Disease scenarios** — `scenarios.py`
**10:49 · File formats (PDB/wwPDB export path)** — `formats.py`
**10:52 · UI split begins** — `ui/__init__.py`; the monolithic `app.py` starts becoming a package
**10:58 · Graph construction + Colab bridge** — `build_graph.py`, `colab.py`
**10:59 · Colab code packer** — `colab/pack_code.py`
**11:02 · E(3)-equivariant graph network** — `egnn.py`
**11:06 · GPU notebook + v3 tests** — `colab/ChronoCell5D_Colab.ipynb`, `tests/test_v3.py`
**11:08 · EGNN tests** — `tests/test_egnn.py`
**11:17 · Training loop** — `train.py`
**12:26 · Polymer physics analytics** — `physics.py` (R_g, ν, contact-decay exponent γ, packing)
**12:29 · Genome coordinate handling** — `genome.py`
**12:46 · Demo state generator** — `demo_states.py` (synthetic healthy / cancer / senescent)

Contributors, morning block: _______________

### 13:31:02 — Commit `24830e4` "Update codebase with new features and tests"

31 files. Adds the ChronoAgent interpreter (`agent.py`), the biological-state system (`states.py`),
disease scenarios, the Colab GPU pipeline, and the first real UI package — `ui/common.py`,
`ui/four_d.py`, `ui/agent_panel.py`, `ui/states_panel.py` — plus `tests/test_v3.py` and
`tests/test_v31.py` and `APP_GUIDE.md`.

Contributors: _______________

### Afternoon build block, 14:00–16:30

**13:39 · Gene annotation dataset** — `chronocell/data/genes_hg38.json.gz`
**13:43 · TAD / domain detection** — `domains.py`

**13:45 · `chronocell/therapy.py` — the drug-lab model** ⭐
The virtual treatment model: each drug class reduced to *where* it acts (target weights per bead from
the signal track and 3D crowding) and *which way* it pushes (+1 open / −1 compact / 0 re-loop); dose
scales displacement; direction-gated movement toward a healthy baseline; every treated conformation
relaxed back to valid polymer geometry (bond lengths → b₀, excluded volume); restoration scored as
`100 × (1 − RMSD(treated, healthy) / RMSD(untreated, healthy))`.
**This module is the strongest candidate for novel technical subject matter in the project.** Record
its conception carefully.
Contributors: _______________
Conception note — who proposed the two-axis drug reduction, the direction gating, and the restoration
metric, and when: _______________

**13:47 · Session snapshot/restore** — `snapshot.py`
**13:49 · PDF dossier generator** — `pdf_report.py`
**13:51 · Biological states reworked** — `states.py`
**13:52 · Shared UI/dataset assembly** — `ui/common.py`
**13:54 · Streamlit config** — `.streamlit/config.toml`
**13:55 · Theme + visualisation** — `theme.py`, `viz.py`
**13:56 · Compare workspace** — `ui/compare.py` (side-by-side states, linked rotation)
**13:58 · Genes workspace** — `ui/genes_view.py`
**13:59 · ChronoAgent + plain-language guide** — `agent.py`, `ui/guide.py`
**14:10 · Drug lab UI, agent panel, 4D workspace** — `ui/drug_lab.py`, `ui/agent_panel.py`,
`ui/four_d.py`. The drug lab pre-computes all 11 doses so the dose slider animates in-browser.
**14:11 · States panel** — `ui/states_panel.py`
**14:12 · Data ingest** — `ingest.py`
**14:14 — 14:16 · Test suites** — `tests/test_v31.py`, `tests/test_v32.py`
**14:17 · Gene lookup** — `genes.py`
**14:22 · Dependencies + sync view** — `requirements.txt`, `ui/sync_view.py`
**14:24 — 14:25 · Documentation** — `JUDGES_GUIDE.md`, `APP_GUIDE.md`, `AUDIT.md`,
`coordinates/README.md`
**14:30 · Application shell finalised** — `app.py`

Contributors, afternoon block: _______________

### 15:00 — 15:02 · Validation against real microscopy ⭐

**15:00** `validation/validate_tracing.py` · **15:01** `validation/results.json` ·
**15:02** `validation/RESULTS.md`

Validated against Bintu et al., *Science* 2018 chromatin tracing data (public:
`github.com/BogdanBintu/ChromatinImaging`) on three datasets × three random splits. Method: split
cells into halves A and B; feed only half A's contact frequencies (<150 nm) to the pipeline; score
against half B's measured median pairwise distances, which the model never saw.

Headline numbers, recorded as measured — including the unfavourable ones:

| Dataset | Model ρ | Baseline ρ | Model ρ, trend removed | Baseline, trend removed | Model / ceiling |
|---|---|---|---|---|---|
| IMR90 chr21:28–30 Mb (4,832 cells) | 0.86 ± 0.01 | 0.93 | **0.38 ± 0.02** | 0.01 | 39 % |
| A549 chr21:28–30 Mb (3,941 cells) | 0.83 ± 0.01 | 0.91 | **0.51 ± 0.03** | 0.00 | 54 % |
| IMR90 chr21:18–20 Mb (1,277 cells) | 0.51 ± 0.03 | 0.96 | **0.09 ± 0.02** | 0.00 | 36 % |

Stated limitations, from `validation/RESULTS.md`: on raw rank agreement the genomic-distance baseline
scores higher than the model; absolute distances are off by roughly 3× (the 72 nm bond length assumed
for 30 kb beads is too small for this data); and the input was imaging-derived contact frequencies
rather than sequencing Hi-C, so a direct Hi-C → imaging test remains to be done.

Contributors: _______________

**15:14 · Master README** — `README.md`

### 15:19:14 — Commit `af4d159` "v3.2: Master README, new modules (domains, genes, therapy, drug lab, PDF reports), validation suite, judges guide"

39 files — the state presented to the judges.

Contributors: _______________

---

## State at end of day

**Modules:** 20 in `chronocell/`, 10 in `ui/`.
**Tests:** 73 `def test_` functions across 5 files — `test_core.py` (16), `test_egnn.py` (5),
`test_v3.py` (14), `test_v31.py` (20), `test_v32.py` (18). The README quotes 83; the difference is
most likely parametrised cases, and the exact figure should be confirmed with
`python -m pytest --collect-only -q` before it is quoted anywhere.

**Six workspaces:** 3D structure · 4D dynamics · Compare · Drug lab · Genes · Guide.

**Public disclosure:** demonstrated to the HackBlitz judging panel on 27 September 2026, and pushed
to `github.com/WolframNU13/Team-NU-13`. If that repository is public, 27 September 2026 is the
worldwide publication date for everything in commit `af4d159`.

---

## Open items

- [ ] **Fill in every `Contributors:` line above, jointly.** This is the only part of the record that
      cannot be reconstructed from evidence later, and it is the part that matters most.
- [ ] **Add the co-originator as a git author.** Have them make commits under their own name and
      email from here on, so the record shows joint work going forward.
- [ ] Add `LICENSE`, `AUTHORS`, `CITATION.cff` — none currently exist, so the code is
      "all rights reserved" by default and authorship is recorded nowhere in the repo.
- [ ] Confirm whether `github.com/WolframNU13/Team-NU-13` is public or private.
- [ ] Check the HackBlitz registration terms for any IP assignment or licence clause.
- [ ] Confirm the real test count before quoting 83 anywhere.
- [ ] Direct Hi-C → imaging validation on IMR90 (named as the next step in `validation/RESULTS.md`).

---

## Maintaining this log

From here on, write entries **as work happens**, dated, with the contributor named. A
contemporaneous log is worth far more than a reconstructed one. Append to the bottom, never rewrite
history above, and commit each entry so its date is independently witnessed by git.

---

## 28 September 2026 — post-event cleanup (live log)

**Entries from here on are written as the work happens**, unlike the reconstructed sections above.
Times are the machine clock (IST, UTC+05:30); git commit times witness them independently.

Made by, for every entry in this section unless stated otherwise: **Claude (AI coding assistant,
Claude Code, model Claude Opus 5.5)**. Requested and approved by: **Shivoham Pandey**.

**15:02:34 · Commit `52c9174` on `main`: this log added to git.** It had existed only as an untracked
file. Committed at the requester's instruction before any cleanup began. Not pushed.

**15:02:35 · Branch `chore/hackathon-cleanup` created from `52c9174`.** All cleanup below happens on
this branch; `main` is untouched.

**15:05 · Repository location recorded.** The requester reports that the GitHub repository was
transferred and renamed from `github.com/WolframNU13/Team-NU-13` to
`github.com/Sh1voham/ChronoCell-5D`. It could not be independently verified from this machine: the
GitHub login here (WolframNU13) no longer has access, and the new repository is not public. The local
`origin` remote still points at the old name and was not changed.

**15:07 · Baseline before any change: 101 tests passed.**

**15:38:04 · Removed `legacy/app_v1.py`.** The original prototype app, imported by nothing. It
remains in git history: `git show 52c9174:legacy/app_v1.py`.

**15:38:04 · Stopped tracking `colab/chronocell_code.zip`.** A generated file (made by
`python colab/pack_code.py`) that `.gitignore` already listed. The local copy is kept.

**15:38:04 · `JUDGES_GUIDE.md` moved to `docs/OVERVIEW.md`,** then edited at **15:38:49**:
- the judge-specific wording and the 3-minute demo script were removed;
- "pitch" and "questions judges ask" were retitled "In 30 seconds" and "Frequently asked questions";
- the outdated accuracy paragraph ("imaging data weren't available") was replaced with the real
  microscopy validation results, including the unfavourable ones.

**15:38:05 — 15:38:17 · Stopped tracking `validation/data/*.csv` (16 MB) and added `validation/data/`
to `.gitignore`.** This is third-party data (Bintu et al., *Science* 2018) whose licence is
unconfirmed. `validation/validate_tracing.py` downloads it on first run. The local copies are kept.

**15:39:31 · Documentation and code references updated:**
- `APP_GUIDE.md` and `AUDIT.md`: references to `JUDGES_GUIDE.md` now point to `docs/OVERVIEW.md`,
  and the `legacy/app_v1.py` row is removed.
- `ARCHITECTURE.md`: folder name `Hack-a-thon/` changed to `ChronoCell-5D/`, and the legacy entries
  now point to git history.
- `chronocell/features.py` docstring: "Teammate 1 loop" changed to "original per-bin loop".
- `validation/validate_tracing.py` docstring corrected: it writes `results.json` only.
- `validation/RESULTS.md`: now notes the on-demand data download.
- `colab/ChronoCell5D_Colab.ipynb`: example Google Drive path changed to `MyDrive/ChronoCell-5D`.

**15:40 — 15:41 · Added `docs/images/fold.png` and `docs/images/compare.png`.** Rendered with
`chronocell.snapshot` from the synthetic reference model and the synthetic demo patients, and
captioned as synthetic in the README.

**15:42:20 · `README.md` rewritten.** The old file was wrapped in `<![CDATA[ ... ]]>`, which made
GitHub show it as one unformatted paragraph. Changes:
- new structure, with the two images and a pipeline diagram;
- a validation summary with its limits;
- test count corrected from 83 to 101;
- two claims the app does not support were removed: an "immunoglobulin cluster" preset and
  "persistence length";
- clone URL changed to `Sh1voham/ChronoCell-5D`;
- the "Team NU-13" section was removed at the requester's instruction.

**Licence: deliberately not added.** The old README claimed "MIT", but no `LICENSE` file exists, and
the open items above say to check the HackBlitz registration terms for an IP-assignment or licence
clause first. The claim was replaced with "no licence has been chosen yet". Choosing one remains an
open decision for the authors.

**Kept, by decision:**
- the two reference PDFs (at the requester's instruction);
- `UPDATES.md`;
- `.claude/launch.json` (local app-preview settings used by the assistant);
- the historical "Teammate" references in `AUDIT.md`, which describe the original specification.

**15:43 — 15:47 · Verification after the changes.** Every Python file compiles; the Colab notebook
and `validation/results.json` parse; **101 tests passed**, the same as the baseline. This includes
end-to-end runs through all six app pages.

**15:48 · Committed on `chore/hackathon-cleanup`** (the commit that adds this entry). Not pushed:
this machine's GitHub login has no access to the renamed repository.

Contributors to the ideas behind these changes: _______________ (the requester directed the cleanup;
fill in if others were involved).

## 28 September 2026 — reconstruction accuracy work (live log)

Made by Claude (Claude Code, Claude Opus 5.5); requested by Shivoham Pandey. Branch
`feat/microscopy-accuracy-v3.3`, created from `chore/hackathon-cleanup` at `124ed41` (which contains
`main` plus the cleanup commit).

**Goal set by the requester:** raise the accuracy measured against real microscopy (currently 36–54 %
of the experiment's own reproducibility, about 43 % on average) towards 80–90 %.

**Rules fixed before any tuning, so the final number is credible:**
- **Practice data (approved by the requester):** tuning uses only Bintu et al. datasets that were never part of the reported
  validation: K562 chr21:28–30 Mb, HCT116 chr21:28–30 Mb (untreated and 6 h auxin), and
  HCT116 chr21:34–37 Mb (untreated). The IMR90 cell-cycle set is excluded because it shares cell
  line and region with a test set.
- **Test data:** the three reported datasets (IMR90 chr21:28–30 Mb, IMR90 chr21:18–20 Mb,
  A549 chr21:28–30 Mb) stay untouched until the final run.
- **Headline number:** Σ model / Σ ceiling of the trend-removed Spearman ρ over the three test
  datasets, averaged over 3 random splits. This is the ceiling-weighted average of the per-dataset
  ratios, so the noisy 18–20 Mb set (ceiling 0.25) cannot dominate it. Per-dataset ratios, raw ρ,
  the genomic-distance baseline and Lin's CCC are reported alongside, favourable or not.
- **What the model sees:** only half A's contact frequencies (< 150 nm), as before.

**18:29 · Branch created.**

**18:35 · Branch renamed** from `feat/reconstruction-accuracy` to `feat/microscopy-accuracy-v3.3`, at the
requester's suggestion. It stays based on the cleanup commit so the restructured README is kept.

**Direction from the requester (18:35):**
- extend v3.2; keep all 101 tests passing;
- report two clearly separated scores:
  - *Contact map fit*: agreement with the input contact data;
  - *Independent microscopy accuracy*: against unseen imaging data;
- Phase 1: a microscopy validation pipeline and an ensemble Langevin population solver;
- Phase 2: physics fixes (ICE, positive fitted exponent, bending stiffness, nuclear confinement);
- Phase 3: UI additions (distance probe, slicing plane, timing table, loss charts);
- Phase 4: REST API and a JSON log without compliance claims.

**18:29 — 18:33 · Step 1 diagnostics, practice data only** (split 0; trend-removed Spearman ρ as a
% of each dataset's ceiling):

| | K562 28–30 Mb | HCT116 28–30 Mb |
|---|---|---|
| Ceiling (half A vs half B) | 0.983 | 0.930 |
| Contact frequencies alone, no 3D (f^-1/3) | 88 % | 82 % |
| Best single 3D structure, from half A's *true* medians | 63 % | 56 % |
| Current pipeline: shortest-path MDS start only | 74 % | 75 % |
| Current pipeline: after gradient fit | 37 % | 53 % |
| Current pipeline: full (with EGNN) | 37 % | 55 % |

Reading:
- the input carries enough information for more than 80 %;
- a single 3D structure is capped at roughly 60–75 %;
- the gradient stage of the current pipeline loses accuracy relative to its own starting point.

This motivates a population (ensemble) model.

**18:36 — 18:48 · First maximum-entropy Langevin ensemble (scratch prototype, practice data, split 0).**

How it works:
- 100 replica chains run overdamped Langevin dynamics as Gaussian chains, with soft excluded volume.
- Each pair of beads has a potential −ε_ij·φ(d), with φ a smooth step at the 150 nm contact radius.
- ε_ij is updated until the ensemble's contact frequency matches the input.
- The prediction is the ensemble's median distance.

This is a maximum-entropy inversion (the idea behind Zhang & Wolynes 2015), with lengths in units of
the contact radius. Results, untuned:

| | HCT116 28–30 Mb | K562 28–30 Mb |
|---|---|---|
| Trend-removed ρ vs half B (% of ceiling) | 0.694 (74.7 %) | 0.797 (81.1 %) |
| Raw ρ | 0.916 | 0.963 |
| Lin's CCC (nm) | 0.81 | 0.77 |
| Model / real scale | 1.01 | 1.16 |
| Contact-map fit ρ (ensemble vs input frequencies) | 0.794 | 0.754 |

Speed: about 5 ms per Langevin step on this CPU (float32, 100 replicas × 65 beads), about 35 s per
dataset.

**18:53 — 18:57 · Langevin prototype dropped.**
- On HCT116 + auxin it reached only 19 % of the ceiling, although contact frequencies alone give 82 %.
- Cause: the simulated time (about 60 units) is shorter than the chain's relaxation time (about 140),
  so long-range contacts never equilibrated. The model's contact frequencies ran about 50 % above the
  input.
- Overall on practice data: 63 %.

**18:57 — 19:10 · New population model: `chronocell/ensemble.py`.**
- **Model:** a maximum-entropy Gaussian polymer ensemble. This approach is prior art, not a novel
  method: HIPPS/DIMES, Shi & Thirumalai, PRX 2019 and Nat Commun 2023.
  - Each contact frequency is inverted to a pair spread through the Maxwell distribution.
  - A valid covariance is fitted by weighted least squares, weighting each pair by its binomial
    reliability.
  - Statistics are exact.
- **Trajectories:** 100 Langevin trajectories are sampled exactly (Ornstein–Uhlenbeck, mode by mode)
  from the fitted spring network.
- **Settings:** frozen after tuning on practice data only. Tested: 1,500 vs 4,000 iterations,
  binomial vs uniform weights, splits 0 and 1. The spread across settings was under 1 point.
- **Tests:** `tests/test_v33.py`, 7 tests on planted populations: ideal chain, partial loop, missing
  pairs, exact Langevin sampling, input checks. All pass.
- **`validation/validate_tracing.py`:**
  - now scores v3.2, the v3.3 ensemble (exact and 100 trajectories) and a no-3D reference;
  - reports contact-map fit and microscopy accuracy separately;
  - adds a `--practice` option.

**Practice-set results** (`validation/results_practice.json`; 4 datasets × 3 splits; trend-removed
Spearman ρ as a % of the ceiling, Σ model / Σ ceiling):

| | K562 | HCT116 | HCT116 + auxin | HCT116 34–37 Mb | **Overall** |
|---|---|---|---|---|---|
| v3.2 single structure | 31–39 % | 52–58 % | 47–50 % | 51–58 % | **48.3 %** |
| v3.3 ensemble (exact) | 92–93 % | 91–92 % | 87–90 % | 95–96 % | **92.0 %** |
| v3.3, 100 trajectories | 91–92 % | 92 % | 88–89 % | 95 % | **91.9 %** |
| No 3D (direct inversion) | | | | | **87.5 %** |

- Contact-map fit of the ensemble: 0.97–0.99.
- Raw ρ of the ensemble (0.92–0.98) now beats the genomic-distance baseline (0.77–0.93) on every
  practice set. v3.2 did not.

**Status:** the three held-out TEST datasets have NOT been run yet. The 80–90 % question is not
answered until they are. Next: run `python validation/validate_tracing.py` once, record the result
here, and update `validation/RESULTS.md` whatever it shows. The full 101-test suite was not rerun for
this commit; only new files and the validation script changed.

**19:53 — 19:55 · Held-out TEST run (once, settings frozen beforehand).** `python validation/validate_tracing.py`,
3 datasets × 3 splits → `validation/results.json`. Trend-removed Spearman ρ as a % of the ceiling:

| Test dataset | v3.2 single structure | **v3.3 ensemble** | 100 trajectories | No 3D (direct inversion) |
|---|---|---|---|---|
| IMR90 chr21:28–30 Mb | 39.1 % | **88.2 %** | 88.4 % | 81.4 % |
| A549 chr21:28–30 Mb | 54.2 % | **91.2 %** | 91.5 % | 87.9 % |
| IMR90 chr21:18–20 Mb (ceiling 0.25) | 35.2 % | **54.4 %** (±11 points across splits) | 53.2 % | 42.3 % |
| **Overall, Σ model / Σ ceiling (pre-registered)** | **45.2 %** | **85.6 %** | **85.7 %** | 79.7 % |

**Also recorded, favourable or not:**
- The unweighted mean of the per-dataset ratios is 77.9 %.
- Raw ρ of the v3.3 ensemble is 0.976 / 0.952 / 0.868. The genomic-distance baseline scores
  0.930 / 0.915 / **0.962**, so the model loses on raw ρ on the weak 18–20 Mb region.
- Lin's CCC (nm) is 0.97 / 0.93 / 0.46.
- Contact-map fit is 0.99 / 0.99 / 0.97.
- The ensemble's cell-to-cell spread (CV 0.42, fixed by the Gaussian model) is below the measured
  0.50–0.58.

**Conclusion:** the requester's target (overall 80–90 %) is met on held-out data under the rule fixed
before tuning: **85.6 %**. The weak-structure region remains far below it.

**19:58 · Documentation updated to match:**
- `validation/RESULTS.md` rewritten: protocol, the two scores, full tables and an honest reading.
- The accuracy sections of `README.md` and `docs/OVERVIEW.md` updated. Both say the population model
  runs on windows and is **not yet built into the app's pages**.
- Test count updated from 101 to 108.

**Full test suite: 108 passed** (the 101 existing tests plus 7 new).

**Found in passing, not a fix:** the requested "positive, data-fitted γ" is already how the code works
(`physics.contact_decay` / `domains.decay_exponent` fit P(s) ~ s^-γ from data, with γ > 0). The
negative-γ error was only in the pasted spec.

### Phase 2 — physics refinements (20:07)

Made by Claude (Claude Code, Claude Opus 5.5); requested by Shivoham Pandey.

- **ICE balancing, new `chronocell/normalize.py`.** ICE = Iterative Correction and Eigenvector
  decomposition (Imakaev et al., Nat Methods 2012).
  - Works on sparse contact lists, with cooler-style defaults: ignore 2 diagonals, MAD filter, and
    `min_nnz` re-filtering until stable.
  - The first version did not converge on sparse maps: row-sum CV was still 5 % after 500
    iterations. Two causes:
    - kept bins whose partners had been masked;
    - a stopping rule stricter than cooler's.
  - After fixing both, the synthetic chr22 map converges in 1,470 iterations (about 1 s). On a
    planted-bias test the bias is recovered with r = 0.999.
  - `balanced_contacts()` returns balanced (ci, cj, cm). `python -m chronocell.build_graph --balance`
    uses it.
- **Bending stiffness and nuclear-envelope confinement** added to `egnn.FitConfig` (`lambda_bend`,
  `bend_cos0`, `lambda_confine`, `confine_radius_nm`), with NumPy twins `physics.loss_bend` and
  `physics.loss_confinement`.
  - Measured on the practice sets: no accuracy change beyond ±1–2 points. Confinement is inactive at
    realistic radii.
  - **Both are off by default.** v3.2 behaviour is unchanged.
  - They are not added to the Gaussian population model, because the fitted couplings absorb any
    quadratic prior.
- **Positive, data-fitted γ:** already the case (see the Phase 1 note). No change needed.
- **Wording:** ChronoAgent's "Biophysical diagnosis" heading renamed to "Biophysical assessment"
  (agent, PDF output, test, APP_GUIDE, OVERVIEW), to avoid a medical-diagnosis reading. The exports
  already say "Research use only — not a clinical diagnostic" and contain no compliance claims.
- **`validation/TUNING.md` (new):** every practice-set experiment with its numbers, including the
  abandoned ones.
- **Tests: 112 passing** (4 new: ICE ×2, bend/confinement ×2). README count updated.

### Phase 3 — user interface (20:39)

Made by Claude (Claude Code, Claude Opus 5.5); requested by Shivoham Pandey.

- **Population model in the app** (3D structure → 03 Model & convergence → *Build population model*):
  - windows up to 400 beads; built from the window's sequencing counts (`ensemble.fit_from_counts`);
  - the adjacent-bead contact probability is an explicit user-visible assumption, and lengths are
    anchored to b₀;
  - *Population model* joins the structure switch; the view shows the representative member;
  - a note explains that members have no excluded volume;
  - multi-model PDB export of the 100 structures.
- **Speed-up:** the ensemble loss now uses the Gram matrix (A Aᵀ) instead of a (pairs × N) difference
  tensor. 300 beads went from more than 10 min to 22 s.
  - The held-out validation was re-run: the exact ensemble is unchanged (85.6 %).
  - The 100-trajectory sample moved from 85.7 % to 85.6 % (A549 from 91.5 % to 91.3 %) through
    sampling round-off. `validation/results.json` and `RESULTS.md` updated; the earlier numbers are
    left as logged above.
- **Two separate accuracy scores**, new `chronocell/accuracy.py`:
  - Contact-map fit, live on the window.
  - Microscopy accuracy, the method benchmark read from `validation/results.json` and labelled "not
    measured on this window".
  - Shown in 03, the PDF dossier (new section) and the JSON report (`accuracy` block).
- **Distance probe** (*Measure*): two beads by number. It shows the distance in the displayed
  structure, the separation along the DNA and the loci. With the population model it also gives the
  population median, the middle 50 % of cells and the contact probability. The pair is marked in 3D.
  - **Limitation, stated honestly:** beads are picked by number, not by clicking, because the app's
    3D chart widget does not return click events.
- **Slicing plane** (*Display*): x, y or z normal at a % of the fold's extent. Everything beyond it
  is hidden (tube faces, beads, line, context), and a translucent sheet marks the plane.
- **Execution telemetry table:** every reconstruction this session, per stage, with wall-clock
  time, ms per bead, device, final loss and contact-map fit. It uses real measurements.
- **Loss-convergence charts:** the existing v3.2 loss chart, plus a new chart for the population
  fit.
- **Docs:** APP_GUIDE §10 (real-microscopy accuracy) and new §20 (how to use v3.3); a Guide-page
  section "How accurate is it? Two scores, never mixed"; README and OVERVIEW updated. Version
  string changed to 3.3.
- **Verified in the running app** (in-app browser, chr22 30.5–32.0 Mb, 150 beads):
  - the population model builds;
  - contact-map fit 0.867; microscopy benchmark 85.6 % with its scope note;
  - the probe gives 196 nm in the displayed structure and a population median of 327 nm (IQR
    235–428 nm);
  - the slicing plane cuts the tube;
  - telemetry rows appear.
- **Tests: 114 passing** (2 new: end-to-end app test of the population model, scores, probe,
  slicing and dossier; and the PDF two-score section).

### Phase 4 — REST API and run log (21:41)

Made by Claude (Claude Code, Claude Opus 5.5); requested by Shivoham Pandey.

- **New `chronocell/api.py`:** handlers for `/api/v1/reconstruct` (population ≤ 400 beads or
  single ≤ 2,000), `/api/v1/metrics` and `/api/v1/benchmark`.
  - They are plain dict-in / dict-out functions; `create_app()` wraps them in FastAPI, and
    `python -m chronocell.api` serves them.
  - Responses keep the two accuracy scores separate.
  - Bad input, including model-level errors and non-object bodies, is rejected as a request error
    (HTTP 400), not a server error.
- **FastAPI is not installed and was NOT installed** (an install needs the requester's go-ahead).
  The wrapper's test is skipped until `pip install fastapi uvicorn`; the handlers are fully tested.
- **Run log:** JSON lines in `.chronocell_cache/api_run_log.jsonl` (git-ignored). Each line holds:
  - run id, UTC time, endpoint and software version;
  - parameters, as sizes only, never raw data;
  - the input's SHA-256;
  - run time, status and a result summary.
  It is documented as a reproducibility record, with no compliance claim.
- **Docs:** README (REST API section), APP_GUIDE, and an optional line in `requirements.txt`.
- **Tests: 117 passing, 1 skipped** (the FastAPI wrapper).

**21:41 · Pushed** branch `feat/microscopy-accuracy-v3.3` to github.com/Sh1voham/ChronoCell-5D
(private) at the requester's instruction. `origin` was updated from the old WolframNU13/Team-NU-13 URL.
No licence was added; `main` was not pushed.
