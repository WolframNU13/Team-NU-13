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
