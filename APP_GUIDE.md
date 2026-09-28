# ChronoCell-5D v3.2 — complete application guide

ChronoCell-5D is a workstation for the 3D and 4D structure of human chromosomes (GRCh38). It reconstructs spatial coordinates from Micro-C contacts, H3K27ac and DNA sequence, analyses them with polymer physics, plays them over time or disease state, and exports standard files.

It runs locally with Streamlit. Heavy reconstruction runs on a Google Colab T4 GPU, whose outputs drop into a folder the app watches.

```bash
pip install -r requirements.txt
streamlit run app.py              # opens http://localhost:8501
python -m pytest                  # 101 tests
python -m chronocell.demo_states demo_states   # optional: synthetic Healthy / Disease / Senescent files to try the states
```

**New in v3.2** (added without removing anything; details in §15–19):
- **Six pages:** 01 3D structure · 02 4D dynamics · **03 Compare** (two states, linked cameras) · **04 Drug lab** (virtual epigenetic drugs, real-time dose slider) · **05 Genes** (19,386 genes, predicted active/silenced, RNA-seq check) · **06 Guide** (plain language).
- **Neighbourhoods:** TADs, A/B compartments, loops and contact decay, as a 05 inspector panel and two new colour modes.
- **Patient data:** `.bed`, `.bedGraph`, `.bigWig` tracks; `.cool`, `.mcool`, `.hic` and text contact maps; RNA-seq tables.
- **Exports:** PDF research dossier (with 3D snapshots), animated GIF for 4D, PNG camera button, side-by-side image.
- **Demo patients** switch, **API key from `.streamlit/secrets.toml`**, and a reference model cached to disk (fast start).
- A plain-language line under every page title. A plain-language overview of the whole project: **docs/OVERVIEW.md**.

**New in v3.1** (added without removing anything):
- A **Biological state** selector (Healthy Control, Disease State / Cancer, Senescent State) backed by a format-based data engine: files are recognised by content, not by fixed names (§12).
- **🤖 ChronoAgent**, a structural-genomics interpreter. It uses a free LLM (Gemini or OpenRouter) when you enter a key, and an instant offline heuristic engine otherwise (§13).
- A **metric dashboard** above the viewport (R_g, max 3D span, mean signal, state, deltas vs healthy).
- Two new colour modes: **Residue Index Spectrum** and **Epigenomic Signal Heatmap**.
- One-click export of the current state's **PDB** and a Markdown **ChronoCell_Analysis_Report.md** (§14).
- Fixes for every error listed in §1, including the `MAIN_CHROMOSOMES` crash.

Contents:
1. [Errors found and fixed in this version](#1-errors-found-and-fixed-in-this-version)
2. [Files and folders](#2-files-and-folders)
3. [How data flows](#3-how-data-flows)
4. [The spatial-coordinate slot (and the reference model)](#4-the-spatial-coordinate-slot-and-the-reference-model)
5. [Screen-by-screen UI guide](#5-screen-by-screen-ui-guide)
6. [Every graph, and how to read it](#6-every-graph-and-how-to-read-it)
7. [4D: time courses, conditions and disease scenarios](#7-4d-time-courses-conditions-and-disease-scenarios)
8. [Google Colab (T4) workflow](#8-google-colab-t4-workflow)
9. [Function reference (every module)](#9-function-reference-every-module)
10. [Accuracy, limits and honest caveats](#10-accuracy-limits-and-honest-caveats)
11. [Troubleshooting](#11-troubleshooting)
12. [Biological states: the format-based data engine](#12-biological-states-the-format-based-data-engine)
13. [ChronoAgent: structural genomics interpreter](#13-chronoagent-structural-genomics-interpreter)
14. [Metric dashboard, colour modes and exports](#14-metric-dashboard-colour-modes-and-exports)
15. [v3.2: Compare](#15-v32-compare)
16. [v3.2: Drug lab](#16-v32-drug-lab)
17. [v3.2: Genes and multi-omics](#17-v32-genes-and-multi-omics)
18. [v3.2: Neighbourhoods, patient data, exports, API key](#18-v32-neighbourhoods-patient-data-exports-api-key)
19. [v3.2: errors found and fixed](#19-v32-errors-found-and-fixed)

---

## 1. Errors found and fixed in this version

### v3.1

| # | Symptom | Root cause | Fix |
|---|---|---|---|
| 10 | **`AttributeError: module 'chronocell.genome' has no attribute 'MAIN_CHROMOSOMES'`** (app.py line 200) on a server started before the upgrade | Streamlit's file watcher deletes an edited module from `sys.modules`, but the package keeps the old module object as an attribute, so `from chronocell import genome` returned the **pre-upgrade** module next to the new `app.py`. Reproduced exactly in a test. | A guard at the top of `app.py` (`_refresh_project_modules`) detects stale project modules (dropped by the watcher, or file newer than the import) and re-imports all of them together, clearing caches built from old classes. The check, purge and imports run under one process-wide lock (several browser sessions can't race). If a file is saved while a run is importing, the import is retried once. |
| 11 | Sidebar could never be reopened once collapsed | The header was hidden with `display:none`, and Streamlit puts the "expand sidebar" button inside the header. | The header is now zero-height and transparent. Only its toolbar is hidden, and the expand button is pinned top-left. |
| 12 | Other coordinate files that failed to load were **silently dropped** (4D "Across conditions" just lost them) | `except Exception: pass`. | Each failure is listed in a warning card ("N provided file(s) could not be loaded and were skipped"). |
| 13 | **A corrupted graph file made every structure fail**, so the app fell back to the reference model | The graph was read inside the same `try` as the coordinates. | The structure is retried without the graph, and a "Graph ignored" card explains why. |
| 14 | A 1-D signal `.npy` in `coordinates/<chrom>/` was offered as a *structure source* and crashed ("Expected an (N, 3) coordinate array") | Slot files were accepted by extension only. | `.npy` headers are sniffed, and 1-D arrays are treated as tracks (§12). |
| 15 | File-load errors appeared as a raw red `st.error` | — | Friendly warning cards that say what is shown instead (all file-derived text is HTML-escaped). |
| 16 | "GC and H3K27ac are placeholders" was shown even when a real signal track was loaded | The banner didn't know about per-state tracks. | The banner reports exactly which parts are placeholders. |
| 17 | Rapid 4D control changes re-ran the whole page | Only the 3D stage was a fragment. | The whole 4D workspace and the ChronoAgent panel are `st.fragment`s too. |

### v3

The whole app was scanned in two ways: a headless sweep (every region × colour × rendering style × export scope, both workspaces, all disease presets, custom structural variants, five chromosomes, the coordinate slot, a reconstruction), and code review. Every item below was reproduced first, then fixed, then covered by a test.

| # | Symptom | Root cause | Fix |
|---|---|---|---|
| 1 | **"List index out of range" after 05 Custom** | The *Strongest hubs* table (Genomic features) kept its selected row under a fixed widget key. After *Open this hub as a window*, or any window change, the old row number was used to index the new window's shorter hub list: `in_hubs[rows[0]]`. | The table key now includes the window, so a selection never outlives the window it came from, and the row index is bounds-checked before use. |
| 2 | Custom ranges could be reversed, negative, past the end, one bin wide, or NaN | The custom range was sliced without validation. | **`clamp_window(lo, hi, n)`** (ui/common.py) is the single gate for every user range: it swaps reversed bounds, clips to `[0, n)`, enforces ≥ 10 beads (shifted back inside the data at the end), and treats None/NaN as "whole". The slider and the new **Start/End (Mb)** inputs have hard `min_value`/`max_value` limits and write through `clamp_window` in callbacks. 12 parametrised tests cover it. |
| 3 | Custom range forgotten after visiting another region | The slider's session key was deleted by Streamlit whenever the slider wasn't drawn. | The range is stored in `ss.custom_window` (not a widget key); widgets are re-synced from it on each run. |
| 4 | **Crash when reconstructing a window with no data** (`ValueError: Contact list is empty`), e.g. a custom window in the unassembled chr22 p-arm | No guard. | The UI explains why the window can't be fitted (fewer than 20 contacts or assembled bins); `fit_structure` raises a clear error, which the UI shows instead of crashing. |
| 5 | `StreamlitDuplicateElementKey` | Three styling containers shared `key="region"`. | Unique `seg_*` keys, styled with an attribute selector. |
| 6 | Uploaded windowed exports were mapped to bins 0…N | Readers ignored the window. | Bundles carry `start_bin`; ChronoCell PDB/XYZ headers are parsed back (`region_hint`), so files land on their true loci. |
| 7 | Whole-chromosome reconstructions stretched assembly gaps into ~50 µm rods (R_g 12.9 µm, ν 0.93) | Shortest paths grow linearly along stretches without contacts. | Polymer prior edges only where data are missing: whole chr22 R_g 520 nm (true 581), ν 0.339. A regression test prevents a relapse. |
| 8 | GC fraction reported 0 % for bins past the sequence end | Zero-length bins passed the "called bases" test. | They are marked invalid (NaN). |
| 9 | Tooltips rounded loci above 16.7 Mb (from v2) | float32 custom data. | float64. |

Earlier audit findings (v1 → v2: physics, EGNN, PDB) are in `AUDIT.md`.

---

## 2. Files and folders

```
app.py                         Streamlit shell + 3D workspace (run this); stale-module guard
ui/common.py                   dataset assembly, coordinate slot, clamp_window, HTML helpers, warning cards
ui/four_d.py                   4D workspace (one st.fragment)
ui/states_panel.py             sidebar: biological state, file discovery, per-state uploads      (v3.1)
ui/agent_panel.py              sidebar AI key; metric dashboard; ChronoAgent panel (st.fragment) (v3.1)
chronocell/                    numerics — never imports Streamlit
  data/hg38.json               UCSC hg38: sizes, cytobands, gaps, centromeres of chr1–22, X, Y; gene anchors
  genome.py                    Chrom model (bins, loci, gaps, bands, resolution)
  physics.py                   polymer metrics, neighbour search, contact<->distance mapping, losses, Kabsch
  synthetic.py                 reference model: Hilbert fractal globule, tracks, simulated Micro-C
  features.py                  GC / H3K27ac binning, contact extraction, enhancer hubs
  formats.py                   PDB (single + multi-model), XYZ, bundles, graph and structure readers
  egnn.py                      E(3)-equivariant GNN, MDS initialisation, GPU-capable fitting
  scenarios.py                 4D: structural-variant operators, disease presets, trajectories
  states.py                    format sniffing, state/chromosome inference, structure-track pairing (v3.1)
  agent.py                     ChronoAgent: metrics, heuristic engine, Gemini/OpenRouter client, report (v3.1)
  demo_states.py               writes a synthetic demo set of state files (opt-in)                 (v3.1)
  colab.py                     GPU pipeline used by the notebook
  viz.py                       all Plotly figures
  theme.py                     design tokens and stylesheet
  build_graph.py, train.py     command-line pipeline; benchmark.py = accuracy benchmark
colab/ChronoCell5D_Colab.ipynb Colab notebook (T4 GPU)
colab/pack_code.py             zips chronocell/ for upload -> colab/chronocell_code.zip
coordinates/                   THE SLOT: put coordinates here (see §4)
tests/                         101 tests (pytest); test_v31.py / test_v32.py include end-to-end AppTests of every page
chronocell/genes.py            gene annotation (RefSeq Select), 3D accessibility, RNA-seq agreement      (v3.2)
chronocell/domains.py          TADs (insulation), A/B compartments, loops, contact decay                 (v3.2)
chronocell/therapy.py          drug lab: drug classes, targeting, dose simulation, ranking                (v3.2)
chronocell/ingest.py           BED / bedGraph / bigWig tracks; cool / mcool / hic / text contact maps      (v3.2)
chronocell/snapshot.py         Pillow 3D snapshots (PNG) and animations (GIF)                             (v3.2)
chronocell/pdf_report.py       PDF research dossier (fpdf2)                                              (v3.2)
chronocell/data/genes_hg38.json.gz  19,386 genes, UCSC hg38 ncbiRefSeqSelect                              (v3.2)
ui/compare.py, ui/sync_view.py Compare page; linked-camera dual viewport                                 (v3.2)
ui/drug_lab.py, ui/genes_view.py, ui/guide.py   Drug lab, Genes and Guide pages                          (v3.2)
static/                        plotly.js served locally for the linked viewports (auto-created)          (v3.2)
.chronocell_cache/             reference-model and demo-patient cache (auto-created, git-ignored)        (v3.2)
docs/OVERVIEW.md               the whole project in plain language with analogies                        (v3.2)
AUDIT.md                       v1 audit and benchmark; APP_GUIDE.md = this file
.streamlit/config.toml         theme (light paper / ink / cobalt), upload limit 512 MB
```

---

## 3. How data flows

```
             raw data (per condition, per time point)
   chrN.fa ──┐  H3K27ac.bigWig ──┐  Micro-C.mcool ──┐
             ▼                    ▼                  ▼
        build_graph.from_files  ->  graph: f_GC, f_epi, assembled mask, contacts (i<j, M>0)
                                          │
                     Colab T4 GPU:  egnn.fit_structure  (MDS init -> contact embedding -> EGNN refine)
                                          │  warm start frame t from frame t-1, align frames
                                          ▼
                   bundle  coordinates/<chrom>/<condition>.npz   (T frames × N beads × 3, nm)
                                          │
        ┌─────────────────────────────────┴──────────────────────────────┐
        ▼                                                                ▼
  3D workspace: one frame, region, physics, tracks,          4D workspace: play frames / conditions,
  maps, in-app reconstruction, PDB/XYZ/bundle export         or simulate a disease rearrangement
```

With no files anywhere, `synthetic.build` supplies a **reference model** for the chosen chromosome, and every screen works on it.

---

## 4. The spatial-coordinate slot (and the reference model)

**Priority**, per chromosome:

1. Files uploaded under **Data → Coordinates** (several allowed; each is a condition).
2. Files in **`coordinates/<chrom>/`** (e.g. `coordinates/chr22/healthy.npz`). They are read on every run, so new Colab output appears after a page refresh.
3. The **reference model**: a planted synthetic chromosome with an orange banner, *"Reference model — awaiting spatial coordinates"*. It stays until 1 or 2 provide coordinates. It then remains selectable under **Structure source** for comparison.

**Tracks and contacts** come from, in order: an uploaded graph → `coordinates/<chrom>/graph*.npz` → tracks embedded in the coordinate bundle → reference tracks. For real coordinates without tracks, the placeholder tracks are flagged and reconstruction is disabled.

**Accepted coordinate files**

| Format | Chromosome / window / resolution taken from | Units |
|---|---|---|
| Bundle `.npz` (Colab, app export) | the file (`chrom`, `resolution`, `start_bin`) | nm (declared) |
| ChronoCell `.pdb` / `.xyz` export | REMARK 250 / comment line | nm (declared) |
| `predicted_coords.pt`, `.npy`, `.csv`, plain `.npz` (N×3) | chromosome selector; resolution inferred from bead count | calibrated (below) |

**Units.** *Data → Coordinate unit*: *Auto* keeps declared nanometres and rescales anything else so the median bond equals b₀ (the note says by how much). You can also force nm, Å or µm. Model units are never relabelled as Å.

**Bundle specification** (`formats.StructureBundle`, format tag `chronocell-bundle-1`)

| key | shape / type | meaning |
|---|---|---|
| `frames` | (T, N, 3) float | coordinates in nm; T ≥ 1 is the 4th dimension |
| `times`, `labels`, `time_unit` | (T,), T strings, str | e.g. `[0, 24]`, `["0 h", "24 h"]`, `"hours"` |
| `chrom`, `resolution`, `start_bin` | str, int, int | chromosome, bin size (bp), first bin of the window |
| `condition`, `source` | str | e.g. `"tumour"`, `"Colab Micro-C"` |
| `gc`, `epi`, `valid` (optional) | (n_bins,) or (N,) | tracks |
| `ci`, `cj`, `cm` (optional) | (E,) | contacts, i < j, counts > 0 |

The file is validated on load. The chromosome must be chr1–22/X/Y, the window must fit inside the chromosome, and values must be finite.

---

## 5. Screen-by-screen UI guide

### Sidebar (v3.1)
- **Biological state**: Healthy Control / Disease State / Cancer / Senescent State.
  - **Structure file** and **Epigenomic track**: the state's files, auto-paired (a track with one value per bead is preferred).
  - **Show this state in the workstation** (on by default). When on, the state's structure and track drive every view, and the Structure source picker follows it (greyed out). When off, you pick the source manually; states remain available for comparison.
  - The detected files, each tagged coords / frames / track / graph / skipped, with the reason it was recognised or skipped.
  - **Add files to <state>**: upload .npy / .pdb (also .npz / .xyz / .csv) directly into the selected state. Uploads persist per state for the session; *Remove* clears them.
  - A count of structures/tracks per state, and of sub-folder files that name no state.
- **ChronoAgent**: **AI API Key** (password field), **Provider** (auto-detect from the key / Google Gemini / OpenRouter), **Model** (blank = free default models), and a status line.

### Top bar (both workspaces)
- **ChronoCell-5D** brand mark.
- **01 3D structure | 02 4D dynamics**: switches workspace. Data, chromosome and parameters are shared.
- **Chromosome selector** (chr1…chrY). The resolution is chosen automatically: the finest standard size with ≤ 6,000 beads (chr22 10 kb, chr4 40 kb, chr1 50 kb). A loaded file overrides it and the app says so.
- **Data**: coordinate uploads (multiple), folder-slot status, coordinate unit, graph upload, "allow unpickling" (needed for PyG `.pt` files; only for files you produced), reference seed.
- **Method**:
  - b₀ from resolution (50 nm × (res/10 kb)^(1/3)) or a manual value.
  - Contact exponent α (default 3).
  - Excluded-volume diameter d_min / b₀ (default 0.8).
- **About**: scope, citations, versions.

### Meta row and banners
- **Left:** a **Structure source** picker (when coordinates are provided; greyed out while a biological state drives the view), plus a one-line description.
- **Right:** assembly · chromosome · resolution, then bead, assembled-bin, contact and frame counts.
- **Warning cards** (v3.1): a file couldn't be loaded (and what is shown instead), a graph was ignored, some condition files were skipped.
- **Banners:**
  - orange = reference model or placeholder tracks;
  - blue = information;
  - notes = unit calibration, window placement, graph mismatch.
- **Frame** slider (3D workspace, multi-frame files): picks which frame is analysed.

### 3D workspace
- **Title block:** "Fig. 1 — Reconstructed fold · chrN", the region name, the exact locus, bin range and length. After a reconstruction, an **Input structure | EGNN reconstruction** toggle appears.
- **Region control** (numbered like the reference design):
  - `01 Whole`.
  - `02 Centromere`: the cytogenetic *acen* bands ± half their width, highlighted.
  - `03 Telomeres`: 1 Mb at each end, highlighted.
  - `04 Enhancer hubs`: runs of top-3 % H3K27ac, highlighted.
  - `05 Custom`: slider in bins plus **Start (Mb)** / **End (Mb)** boxes, all clamped (§1).
- **Ideogram:** cytobands coloured by Giemsa stain, the blue bracket = current window, blue ticks = highlighted regions, centromere position.
- **Metric dashboard** (v3.1, directly above the viewport): R_g (with ν and regime), max 3D span, mean signal intensity (and which track), and the biological state with its source file. When the healthy control is loaded on the same beads, each number shows its change vs Healthy (terracotta = up, cobalt = down).
- **Viewport** (Fig. 1):
  - The chromatin fibre is drawn as a lit tube.
  - Controls: drag to rotate, scroll to zoom, right-drag to pan, double-click to reset. On-canvas buttons: **Turntable / Pause** and camera presets **Iso · Front · Top · Side**.
  - A scale bar gives real nanometres, and the end beads are labelled with bin numbers.
  - Hovering shows the locus, bin, cytoband, GC and H3K27ac, or "unassembled".
  - **Display** popover:
    - rendering Tube / Beads / Line;
    - colour by Genomic position / GC content / H3K27ac / Monochrome (Monochrome + highlight = cobalt highlight) / **Residue Index Spectrum** / **Epigenomic Signal Heatmap** (v3.1, §14);
    - tube radius, bead size, "show rest of chromosome" (grey context line), viewport height.
  - Display changes redraw only the viewport; physics isn't recomputed.
- **Legend line** (above) and **spec line** (below): the active colour scale; the structure name, R_g, ν and regime, and overlap count.
- **Inspector** (scrolls independently):
  - **01 Polymer physics**: R_g, R_e, contour length, mean bond / b₀, scaling exponent ν ± SE with fit window and R², excluded-volume overlaps (deeper than 2 % b₀) with maximum penetration, R_e²/R_g², regime tag, then Graphs A and B.
  - **02 Genomic features**: assembled bins, median GC, median H3K27ac, contacts in window, contact-decay γ, hub count, then Graphs C–E and the **Strongest hubs** table (select a row → *Open this hub as a window*).
  - **03 Model & convergence**:
    - The equations in use, and the parameters b₀, α, M_ref, target clip, λ₁/λ₂ and compute device.
    - **Reconstruct this window** (embedding and EGNN epochs; limit 2,000 beads in-app). For larger windows, a button jumps to the most contact-dense 800 beads. Windows without data are explained, not attempted.
    - After a fit: runtime and device, parameter count, final L_contact, and, on the reference model, RMSD / R_g and distance correlation against the planted truth. Then Graph F.
    - **Verify E(3) equivariance**: rotation and reflection errors (≈ 1e−14).
  - **04 Export**: scope (window or whole structure), wwPDB column check, downloads (**PDB**, **XYZ**, **Bundle (.npz)** for re-import, **Report (JSON)** with provenance, parameters and metrics, **Bins (CSV)**), and a PDB preview.
- **🤖 ChronoAgent panel** (full width, below the inspector): see §13.
- **Status bar:** structure source, tracks source, chromosome · resolution · b₀ · α · d_min.

### 4D workspace
Described in §7. With several biological states loaded, **Across conditions** plays Healthy Control → Disease State / Cancer → Senescent State (when they cover the same beads). The ChronoAgent panel appears below it, analysing the whole loaded structure.

---

## 6. Every graph, and how to read it

| ID | Where | Shows | Read it as |
|---|---|---|---|
| Fig. 1 | 3D viewport | Lit tube along a Catmull-Rom spline through the beads, coloured by the chosen track. Grey = unassembled bins; pale = outside the highlight. | The fold itself. Compact interpenetrating domains are typical of a crumpled (fractal) globule. |
| A | Polymer physics | log–log RMS distance √⟨R²(s)⟩ against genomic separation s (kb). The blue segment is the fit window; dashed guides show ν = 1/3, 1/2, 0.59. | Slope = ν. About 1/3: compact globule; 1/2: ideal chain; 0.59: swollen coil; below 0.28 at large s: confinement plateau. |
| B | Polymer physics | Histogram of bond length / b₀; dashed line at 1. | A narrow peak at 1 means a physical chain; a tail means stretched bonds (breaks, gaps, unconverged fits). |
| C | Genomic features | Two aligned panels: GC fraction and H3K27ac along the window (Mb). Gaps are unassembled bins; blue bands are the highlight. | Separate axes because the units are unrelated (a dual-axis chart would imply a false scale link). |
| D | Genomic features | Contact map (log₁₀(1+M), Hi-C convention) or distance map (nm) of the window. Large windows are block-averaged, and the pixel size is stated. | Squares on the diagonal are domains; a checkerboard is compartments. The distance map is the 3D structure's own "contact map". |
| E | Genomic features | Contact decay P(s) (mean reads per pair against separation, log–log), with γ in the readout. | For M ∝ d^−3, γ ≈ 3ν: a consistency check between contacts and structure. |
| F | Model & convergence | Loss against epoch (log): L_contact, λ₁·L_smooth, λ₂·L_steric, L_total; the dotted line marks the start of EGNN refinement. | Flat L_contact near its noise floor means converged. The steric term appears when the phantom phase ends. |
| G | 4D stage | Animated trajectory (Play / Pause / frame slider, client-side), coloured by position, displacement from frame 0, or segment (native, duplicated, partner, inverted). | Watch where the fold changes. Axes are fixed across frames, so motion is real, not a camera jump. |
| H | 4D Dynamics | Small multiples against time: R_g, RMSD to frame 0, median bond. | Compaction or opening over time; convergence of simulated relaxation. |
| I | 4D Dynamics | Per-bead displacement (first → last frame) along the rearranged chain, coloured by segment. | Peaks locate where the structure changed most (e.g. deletion junction, docking partner segment). |

---

## 7. 4D: time courses, conditions and disease scenarios

The 4th dimension has three sources, chosen at the top right of the 4D workspace:

1. **Provided frames**: a multi-frame bundle (time course, cell-cycle phases, dose series), played in its own time units.
2. **Across conditions**: the first frame of each provided condition of the same chromosome and window (e.g. `healthy.npz` → `tumour.npz`), superimposed.
3. **Simulated scenario**: a structural rearrangement applied to the current structure, followed by relaxation of the polymer (bond lengths + excluded volume). The frames are relaxation sweeps, not biological time. **This is a hypothesis generator, not measured disease structure**, and it is labelled so on screen. It is the default until time-resolved or disease coordinates are provided.

**Disease presets** (gene anchors from UCSC hg38 ncbiRefSeqSelect)

| Preset | Chromosome | Operation | Region used | Disease relevance | Reference |
|---|---|---|---|---|---|
| 22q11.2 deletion (LCR22A–D) | chr22 | deletion | DGCR6 start → LZTR1 end, 18.91–21.00 Mb (approximate span) | 22q11.2 deletion syndrome; a risk factor for early-onset **Parkinson's disease** and schizophrenia | Butcher et al., JAMA Neurol 2013 |
| Philadelphia der(22) t(9;22) | chr22 + chr9 | translocation | chr22 pter → mid-BCR (≈23.25 Mb) fused to chr9 from ABL1 (130.84 Mb) → qter | **CML**, Ph+ ALL (BCR–ABL1) | Nowell & Hungerford 1960; Rowley 1973 |
| Ewing sarcoma der(22) t(11;22) | chr22 + chr11 | translocation | chr22 pter → mid-EWSR1 (≈29.28 Mb) fused to chr11 from mid-FLI1 (≈128.75 Mb) → qter | **Ewing sarcoma** (EWSR1–FLI1) | Delattre et al., Nature 1992 |
| SNCA triplication (PARK4) | chr4 | duplication × 2 | SNCA ± 0.5 Mb (illustrative; reported events are 0.4–4 Mb) | Familial **Parkinson's disease** | Singleton et al., Science 2003 |

**Custom structural variant** (any chromosome): deletion, duplication (1–4 extra copies), inversion or translocation to any partner chromosome from a chosen position to its qter. All positions are Mb inputs with hard min/max limits and are clamped to the loaded data. Translocation partner segments use a synthetic fractal-globule conformation, because the partner chromosome's own coordinates aren't part of this chromosome's data; hovering shows their true partner loci.

**4D inspector**
- **01 Scenario & frames**: source controls, preset description and citation, frame count, sweeps per frame, and the colour mode.
- **02 Dynamics**:
  - Readout: frames · beads, R_g first → last, RMSD, largest displacement with its locus, median bond.
  - Event log, then Graphs H and I.
- **03 Export trajectory**:
  - **Multi-model PDB**: one MODEL per frame; PyMOL (`load x.pdb` then use the movie controls) and ChimeraX play it as a movie. Partner beads carry their own segID, e.g. `CH9`.
  - **Trajectory .npz**: frames plus the bead → bin/chromosome/kind maps.
  - **Metrics CSV**.

---

## 8. Google Colab (T4) workflow

1. `python colab/pack_code.py` creates `colab/chronocell_code.zip`.
2. Open `colab/ChronoCell5D_Colab.ipynb` in Colab and choose *Runtime → Change runtime type → T4 GPU*.
3. **Cell 1** checks the GPU. **Cell 2** uploads the zip and installs pyBigWig, cooler and pyfaidx.
4. **Cell 3 (configure):**
   - `DEMO = True` runs everything on planted data (no downloads).
   - For real data, set `DEMO = False`, pick `CHROM`, and list `CONDITIONS`. Each condition has one `Frame(label, time, mcool=…, bigwig=…)` per time point; paths can be local or https.
   - Sources: 4DN Data Portal (Micro-C/Hi-C `.mcool`) and ENCODE (H3K27ac fold-change `.bigWig`). For cancer, K562 (a CML line carrying BCR–ABL1) has public Hi-C. For Parkinson's, use patient-derived or isogenic iPSC dopaminergic-neuron data where available.
   - The `.mcool` must contain the chosen resolution.
5. **Cell 4** downloads the UCSC hg38 FASTA for the chromosome and builds one graph per frame (GC, H3K27ac, contacts; chr-prefix mismatches are handled).
6. **Cell 5** reconstructs on the GPU:
   - Frame 1: shortest-path MDS (3,000-node Floyd–Warshall on the GPU), then 2,500 embedding epochs and 100 EGNN epochs.
   - Later frames are warm-started from the previous frame (300 epochs), then all frames are superimposed so the time axis is continuous.
7. **Cell 6 (QC):** per condition, R_g over frames, RMSD against frame 1, ν and regime, overlaps, final losses, an E(3) equivariance check on the GPU, and a 3D preview.
8. **Cell 7** writes `coordinates/<chrom>/<condition>.npz` plus `graph.npz` and downloads a zip.
9. Unzip it at the project root and refresh the app. The reference model is replaced, the conditions appear under *Structure source*, and the 4D workspace offers *Provided frames* and *Across conditions*.

The notebook was executed end to end in demo mode (CPU, reduced epochs) during development. It has not been run on a real T4 here, so the GPU timings above are estimates.

---

## 9. Function reference (every module)

### `chronocell/genome.py`: reference genome
- `Band`, `Gene`: cytoband and gene records.
- `gene(name)`: hg38 anchor (DGCR6, TBX1, COMT, LZTR1, BCR, EWSR1, SNCA, ABL1, FLI1).
- `chromosome_size(name)`, `default_resolution(name)`, `resolution_for_beads(name, n)`.
- `Chrom`: one chromosome at one resolution.
  - `n_bins`, `short`, `bin_start/bin_end(i)`, `bin_lengths()`, `interval_to_bins(start, end)` (clipped), `locus(i)`, `mb(i)`.
  - `gap_fraction()` (exact N overlap), `assembled_mask()`, `band_for_bins()`, `acen`, `with_resolution(r)`.
- `chrom(name, resolution=None)`, `chrom_for_beads(name, n)`.
- chr22 compatibility names: `CHROM`, `N_BINS`, `RESOLUTION`, `CYTOBANDS`, `GAPS`, `ACEN`, `bin_start()`, `assembled_mask()`, …

### `chronocell/physics.py`: polymer physics (NumPy, nm)
- `bond_length_for(res)`: b₀ scaling with resolution.
- `radius_of_gyration`, `end_to_end`, `bond_lengths`, `calibrate_to_bond_length`.
- `distance_scaling(x)` → `ScalingFit` (s, RMS R, local slope, ν, SE, R², window, regime); `classify_regime(ν)`.
- `neighbor_pairs(x, r, min_sep)`: O(N) cell-list pair search.
- `reference_count`, `contact_target_distance`: d* = b₀(M/M_ref)^(−1/α), clipped to [d_min, 8 b₀].
- `loss_contact`, `loss_smooth`, `loss_steric` (→ `StericReport`: loss, overlaps, max penetration, min distance).
- `kabsch_rmsd(ref, mobile, allow_reflection)`, `distance_correlation`.
- `contact_decay` (P(s), γ), `coarse_distance_map`, `coarse_contact_map`.

### `chronocell/synthetic.py`: reference model
- `hilbert_curve_3d`: vectorised Skilling algorithm.
- `relax`: position-based bond and excluded-volume relaxation with a Verlet list.
- `fractal_globule(n, b0, seed)`: the planted conformation.
- `synthetic_tracks(chrom)`: band-informed GC and H3K27ac; NaN in gaps.
- `simulate_contacts`: Poisson reads with mean ∝ d^−α; none in gaps.
- `build(chrom, seed, b0)` → `SyntheticChromosome`.

### `chronocell/features.py`: tracks from raw data
- `gc_fraction(seq, n_bins, res)`: GC/(A+C+G+T), NaN for N-rich or empty bins.
- `binned_mean`: NaN-aware bigWig binning.
- `signal_hubs(epi, valid)`: H3K27ac hubs.
- `contacts_from_pixels`: cooler pixel table → canonical contacts.

### `chronocell/formats.py`: files
- `segment_id`, `atom_record` (exact wwPDB columns), `pdb_frame`, `epi_to_bfactor`.
- `write_pdb(coords, start_bin, gc, epi, epi_ref, source, method, chrom, bead_bins, segments)`.
- `write_pdb_trajectory(frames, res_seq, segments, chrom, title, method)`: MODEL/ENDMDL.
- `validate_pdb(text)` → `PdbCheck` (atoms, CONECT, models, issues); `read_pdb`, `write_xyz`, `read_xyz`.
- `as_coords`, `read_structure(bytes, name)`.
- `GraphData`, `canonical_contacts`, `read_graph`, `write_graph_npz`.
- `StructureBundle` (`validate`, `n_frames`, `n_beads`), `write_bundle`, `read_bundle` (any file → bundle + notes), `region_hint`.
- `report_json`.

### `chronocell/egnn.py`: model and fitting (PyTorch; CPU or CUDA)
- `RadialBasis`, `EGNNLayer`, `ChromatinEGNN`: stabilised E(3)-equivariant layers.
- `MessageGraph` (`to`, `to_device`), `build_message_graph`: backbone + top-k contacts.
- `node_features(gc, epi, valid)`: standardised [GC, log H3K27ac, assembled].
- `contact_loss`, `smooth_loss`, `steric_loss`: differentiable twins of the physics losses.
- `FitConfig`: epochs, learning rates, λ, α, d_min, device ("auto" = CUDA when present), …
- `FitResult`, `resolve_device`, `random_walk`, `shortest_path_mds` (with the gap-only polymer prior).
- `fit_structure(n, feats, ci, cj, cm, cfg, b0, init_nm, progress)`.
- `random_orthogonal`, `equivariance_check`.

### `chronocell/scenarios.py`: 4D
- `Trajectory`: frames, times, labels, and bead → bin / chromosome / kind maps.
- `clamp_region`, `relax_frames`, `align_frames`.
- Structural-variant operators: `deletion`, `duplication`, `inversion`, `translocation`.
- `Preset`, `PRESETS`, `preset_region`.
- `simulate(x, chrom, op, params, b0, …)`, `from_bundle(...)`, `frame_metrics(traj)` (R_g, R_e, RMSD, displacement, median bond per frame).

### `chronocell/colab.py`: GPU pipeline
- `environment`, `download_fasta` (UCSC hg38).
- `Frame`, `Condition`, `GraphArrays`, `build_graph`.
- `reconstruct_condition` (warm-started frames, aligned) → `ConditionResult`.
- `write_outputs` (slot layout + zip), `demo_graphs` (planted healthy and rearranged time courses).

### `chronocell/viz.py`: figures
- Geometry and colour: `catmull_rom`, `tube_mesh` (rotation-minimising frames), `level_of_detail`, `state_colorscale`, `encode`.
- 3D: `viewport` (Fig. 1).
- 2D: `scaling_chart` (A), `bond_histogram` (B), `tracks_chart` (C), `matrix_chart` (D), `decay_chart` (E), `loss_chart` (F).
- 4D: `trajectory_figure` (G), `timeseries_chart` (H), `displacement_profile` (I).

### `chronocell/theme.py`
Design tokens (measured WCAG contrast), colour scales, `inject()` stylesheet, `plot_layout()`.

### Command line
- `python -m chronocell.build_graph`: FASTA + bigWig + mcool → graph `.npz` (`--synthetic` for a dry run).
- `python -m chronocell.train`: graph → coordinates, history CSV, PDB.
- `python -m chronocell.benchmark`: accuracy table (AUDIT.md §5).

### `chronocell/states.py` (v3.1)
- Constants `STATES`, `HEALTHY`, `DISEASE`, `SENESCENT`, kinds `COORDS`, `FRAMES`, `TRACK`, `GRAPH`, `UNSUPPORTED`.
- `sniff(name, data=None, path=None)` → (kind, n, shape, detail); reads only the .npy header, never unpickles, never raises.
- `load_track(bytes)`; `infer_state(path)`, `infer_chrom(path)`.
- `folder_signature(root)`, `scan_folder(root)`, `sniff_upload(state, name, data)`, `for_chromosome(files, chrom)`.
- `StateFile`, `StatePlan`, `plan(files, state)`, `best_track(structure, tracks, n_full)`.

### `chronocell/agent.py` (v3.1)
- `compute_metrics(coords, signal, valid, b0, chrom, first_bin, label, placeholder)` → `Metrics`: R_g, span, R_e, ν, packing, asphericity, crowding, dense fraction, signal statistics, Spearman ρ (signal vs crowding / radial position), hubs, most accessible and most compact loci.
- `AgentContext` (`payload()`, `fingerprint()`), `Comparison`.
- `heuristic_analysis(ctx, query)`: the offline engine.
- `ask_llm(ctx, query, heuristics, provider, api_key, model, transport)`, `detect_provider(key)`, `build_prompt`, `SYSTEM_PROMPT`, `DEFAULT_MODELS`, `AgentError`.
- `report_markdown(ctx, analysis, engine, query)`, `metric_rows`, `spearman`.

### `chronocell/physics.py` additions (v3.1)
- `max_span(x)`: exact maximum pairwise distance (pruned by the triangle inequality; chunked).
- `gyration_shape(x)`: gyration-tensor eigenvalues, asphericity, κ².
- `local_density(x, r)`: non-bonded neighbours within r (cell list).

### `chronocell/genome.py` addition (v3.1)
- `genes_in(chrom, start, end)`: anchor genes overlapping a locus.

### `ui/common.py`
- `clamp_window` (§1).
- `Dataset` (`n`, `n_frames`, `has_contacts`, `epi_ref`, `gbin`; v3.1 adds `signal_label`, `signal_is_placeholder`).
- `reference_chromosome`, `slot_files` (skips 1-D tracks), `slot_graph`.
- `load_dataset(…, track=None, condition=None)`: v3.1 adds an optional per-bead signal track and a condition label.
- `html`, `readout`, `fmt`, `banner`, `esc`, `warning_card`.
- `SLOT_ROOT` honours the environment variable `CHRONOCELL_COORDINATES`.

### `ui/states_panel.py` (v3.1)
- `sidebar(chrom, n_full)` → `BioSelection` (state, drive, plans, structure, track; `track_for(file)`).
- `as_source(file)`, `file_bytes(file)`, `_ingest` (upload callback).

### `ui/agent_panel.py` (v3.1)
- `sidebar_settings()` → `AgentSettings`.
- `metrics_for` (cached), `dataset_metrics`, `comparisons_for`, `build_context`.
- `metric_cards(ctx)`: the dashboard.
- `render(ctx, settings, pdb, scope)`: the panel (`st.fragment`).

### `ui/four_d.py`
- `render(ds, conditions, b0, frame)`: the 4D workspace.
- `_scenario_controls`: presets and custom structural variants, clamped.
- `_simulate` (cached), `_hover`.

### `app.py`
- Cached helpers: `read_slot_file`, `hover_labels`, `hubs_for`, `analyse` (all window physics), `window_contacts`, `equivariance_report`.
- `best_fit_window`, `full_track`.
- Region logic: `resolve_region`, `open_window` (callback), `_custom_from_slider`, `_custom_from_inputs`.
- `ideogram`, `stage` (fragment).
- v3.1:
  - `_refresh_project_modules`, `_purge_project_modules`, `_stamp_project_modules` (the stale-module guard).
  - `load_source(i)`: a graph failure degrades to "graph ignored".
  - `pdb_for`, `chrono_agent` (dashboard + panel).

---

## 10. Accuracy, limits and honest caveats

- **Reconstruction accuracy.** Measured against planted fractal globules, which are the only ground truth available here:
  - 400–1,600-bead windows: RMSD 0.28–0.32 R_g, distance correlation 0.96–0.97.
  - Whole chr22 with the GPU-sized initialisation: 0.32 R_g on assembled beads, r = 0.93.
  - Accuracy on real chromatin can't be stated without imaging ground truth (e.g. chromatin tracing).
- **Mirror images.** Contact data determine a structure only up to reflection.
- **EGNN.** For a single-structure fit, EGNN refinement is equivalent to plain coordinate refinement (AUDIT.md §5). Its value needs training across many structures.
- **Disease scenarios.** They are geometric consequences of karyotypes under a polymer model, not measured disease conformations. The "time" axis is relaxation sweeps. Parkinson's presets (22q11.2 deletion, SNCA triplication) model the structural lesion only; they say nothing about neuronal chromatin state.
- **Parameters.** b₀ and α are literature anchors. Check α against your data's P(s) exponent.
- **Reference model.** The default view is synthetic until you provide coordinates, and it says so.
- **ChronoAgent.**
  - Its statements are interpretations of the model's geometry, and the report says so.
  - Rule thresholds come from polymer theory (ν) and the reference globule (packing ≈ 0.19). They are not fitted to disease data.
  - The "therapeutic strategy" section is a list of research hypotheses (drug classes and mechanisms, each with its evidence level), never treatment advice.
  - LLM answers can be wrong: the prompt forces them to quote the metrics, and the offline analysis stays one click away.
  - The Gemini/OpenRouter HTTP path is tested against stubbed responses; no real API call was made while building this.
  - Free-tier model names change. If the defaults are retired, type a current model name in *Model*.
- **Biological-state assignment** is by words in the path. Check the sidebar list: it shows the state each file was assigned. Uploading into a state assigns it explicitly.
- **Raw-file ingestion.** `build_graph.from_files` needs pyfaidx, pyBigWig and cooler. Those libraries and real files weren't available where this was built, so the per-bin functions it calls are unit-tested but the glue has only been exercised through the demo path.

## 11. Troubleshooting

| Problem | Fix |
|---|---|
| Changes to the code don't appear | v3.1 re-imports changed modules automatically on the next rerun. If a very old server still misbehaves, stop and restart `streamlit run app.py`. |
| `module 'chronocell.genome' has no attribute …` | A pre-3.1 server mixing old and new modules. Restart it once; from v3.1 the guard prevents it. |
| A state shows "No files" | File names/folders must contain a state word (healthy, control, ctrl, wt · cancer, tumour, tumor, disease, patient, mutant · senescent, senescence, OIS, aged), or upload into the state from the sidebar. |
| A track is "ignored" | Its length must equal the structure's bead count, or the whole chromosome's bin count at the structure's resolution. |
| ChronoAgent says the key was rejected | Check the key and provider. Gemini keys start with `AIza`; OpenRouter keys start with `sk-or-`. |
| ChronoAgent: "no model answered" | The free models are busy or retired. Retry, or enter a current model name. The offline analysis is shown meanwhile. |
| Sidebar hidden | Click the small button at the top-left corner. |
| "graph ignored: bins do not match" | The graph's resolution differs from the structure's. Rebuild the graph at the same resolution. |
| Reconstruct button missing | The window has > 2,000 beads (use the suggested window or Colab), no contacts (unassembled region) or no graph. |
| PDB export unavailable | More than 9,999 beads with bins > 9,999: export a window or use a coarser resolution. |
| `.pt` file refused | Tick *Allow unpickling* (only for files you produced) or export to `.npz`. |
| Colab: `has N bp bins; requested M` | Pick a `RESOLUTION` present in the `.mcool`. |

---

## 12. Biological states: the format-based data engine

**What it does.** You choose a state in the sidebar: *Healthy Control*, *Disease State / Cancer* or *Senescent State*. The engine finds that state's files and loads its structure and signal track into every view in real time: 3D model, metric cards, colour modes, genomic-feature charts, hubs, 4D "Across conditions", and ChronoAgent.

**Recognised by format (content), not by file name** (`chronocell/states.py: sniff`):

| File | Content | Recognised as |
|---|---|---|
| `.npy` | shape `(N, 3)` (or `(3, N)`) | coordinates: one conformation |
| `.npy` | shape `(T, N, 3)` | frames: a time course or several conformations |
| `.npy` | shape `(N,)` or `(N, 1)` | 1-D epigenomic track: one value per bead (H3K27ac or any signal) |
| `.pdb` | ATOM/HETATM records | coordinates (first MODEL); several MODELs → frames |
| `.npz` | `frames` or a coordinate key | ChronoCell bundle / coordinate archive (a gc/epi npz = graph) |
| `.xyz`, `.csv` | 3 numeric columns | coordinates |

- Only the `.npy` header is read to classify arrays. Pickled/object arrays are never loaded.
- Corrupt files are reported as *skipped* with the reason. They never crash the app.

**Which state a file belongs to.** The engine reads the words in its path, deepest first (the file name, then its folders):

| State | Words (any case; long words may be part of a longer name) |
|---|---|
| Healthy Control | healthy, control, normal, wildtype, baseline, untreated, unaffected; tokens ctrl, wt, hc, mock |
| Disease State / Cancer | disease, cancer, tumour/tumor, malignant, carcinoma, leukaemia/leukemia, lymphoma, sarcoma, myeloma, glioma, melanoma, metasta…, patient, parkinson, alzheimer, mutant, affected, abnormal, deletion, translocation; tokens cml, aml, pd, mut, tum, ph |
| Senescent State | senescent, senescence, sasp, ageing/aging, replicative, presenescent; tokens sen, ois, ris, aged |

- The longest matching word wins, so `abnormal` beats `normal`. "normalized" and "uncontrolled" are ignored.
- A path component naming a chromosome (`chr9`, `chrX`) restricts the file to that chromosome.
- Files uploaded in the sidebar go to the selected state explicitly.

**Where files are found:** everywhere under `coordinates/` (up to 4 folders deep; folders starting with `.` or `_` are skipped). Set the environment variable `CHRONOCELL_COORDINATES` to scan another folder. Example layout:

```
coordinates/
  chr22/
    healthy/   coords.npy (5082×3)   h3k27ac.npy (5082)
    cancer/    tumour_K562.npy       tumour_K562_h3k27ac.npy
    senescent/ IMR90.pdb             signal.npy
```

**Pairing a structure with its track** (`best_track`):
1. The track must have one value per bead, or cover the whole chromosome at that resolution. A whole-chromosome track is windowed automatically.
2. Among matching tracks, the one in the same folder wins, then the one sharing the most name tokens. You can override the choice in the sidebar.
3. A track replaces H3K27ac as the signal. GC, the assembled mask and contacts still come from a graph or the reference.

**How a state drives the app** (with *Show this state in the workstation* on):
- The state's structure becomes the Structure source, loaded through the same pipeline as uploads (units, windows, validation).
- Its condition label is the state name.
- A state with a track but no structure colours the current structure with that track, if the lengths match.
- A state with no files leaves the current source in place, and the sidebar explains how to add files.

**Try it:** `python -m chronocell.demo_states demo_states`, then run with `CHRONOCELL_COORDINATES=demo_states`. All demo files are named `synthetic_demo_*`: healthy = reference-like globule; "disease" = 18–26 Mb swollen with a 2.2× signal gain; "senescent" = globally compacted, as a PDB. They are never written into `coordinates/` by default, so the reference model stays until you provide data.

---

## 13. ChronoAgent: structural genomics interpreter

The panel **🤖 ChronoAgent: Structural Genomics Interpreter** sits below the 3D workspace and below the 4D workspace. It is an `st.fragment`, so asking questions never re-runs the page.

**Inputs:**
1. The selected biological state, and whether the view actually comes from that state's files.
2. Metrics computed at runtime on the structure in view (the window, or the EGNN reconstruction), by `agent.compute_metrics`:
   - R_g, maximum 3D span, R_e, scaling exponent ν and regime;
   - packing fraction = N b₀³ / (8 R³) with R = √(5/3) R_g, the sphere of equal R_g;
   - asphericity;
   - crowding (non-bonded beads within 1.5 b₀) and the fraction of beads in dense neighbourhoods (≥ 12 neighbours ≈ local packing 0.44);
   - mean/median/SD signal;
   - Spearman ρ of signal vs crowding and vs radial position;
   - signal hubs;
   - the three most accessible loci (high signal, low crowding) and the three most compact, low-signal loci.
3. The same metrics for every other state with data: on the same beads when the bead sets match, otherwise for the whole structure (then R_g is not compared).
4. Anchor genes in view, provenance flags (reference model? placeholder tracks? reconstruction?) and your optional question.

**Engines:**
- **Offline heuristic engine** (no key; instant; deterministic). Rules:
  - Fold class from ν (< 0.28 over-compact, ≈ 1/3 globule, 0.45–0.56 relaxed, > 0.56 swollen) and packing vs the reference globule.
  - Net shift vs Healthy: R_g ±5 %, ν ±0.03 and dense-fraction ±5 points each count as one unit.
  - Therapeutic hypotheses by state and direction:
    - disease + open → BET / CDK7 / p300-CBP probes;
    - disease + compact → HDAC / DNMT / EZH2 probes;
    - senescent + crowded foci → SAHF, senolytics and senomorphics;
    - senescent + open → SADS / lamin B1 loss.
  - Gene notes for 22q11.2, BCR/ABL1, EWSR1/FLI1 and SNCA, each with its evidence level.
  - Expression and accessibility from the correlations and the ranked loci.
  - A question is routed by keywords to the relevant sections.
- **LLM engine** (key entered). Google Gemini (`generativelanguage.googleapis.com`, free AI Studio keys) or OpenRouter (free `:free` models).
  - The model receives the numeric context as JSON plus the heuristic findings. The system prompt requires it to quote the numbers, label hypotheses, frame therapeutics as research with evidence levels, never give dosing or patient advice, and use the same headings.
  - Default models are tried in turn if one is missing or rate-limited: Gemini `gemini-flash-latest` → `gemini-2.5-flash` → `gemini-2.0-flash`; OpenRouter `meta-llama/llama-3.3-70b-instruct:free` → `deepseek/deepseek-chat-v3-0324:free` → `google/gemma-3-27b-it:free`.
  - A rejected key stops immediately. Any failure shows a warning card and falls back to the offline analysis.
  - One request per **Analyse** press; answers are cached for the same view and question.

**Key handling:**
- The key is a password field kept only in this browser session's widget state.
- It is sent only to the chosen provider, in a request header (`x-goog-api-key` / `Authorization`), never in a URL.
- It is redacted from error messages and never written to the report.
- LLM output is rendered as Markdown only (no HTML), with remote images stripped.

**Output sections:** Biophysical assessment · Therapeutic strategy (research hypotheses) · Expression & accessibility insights · Answer to your question.

---

## 14. Metric dashboard, colour modes and exports

- **Dashboard** (above the 3D viewport): R_g · max 3D span · mean signal intensity · biological state, each with its change vs Healthy Control when comparable. The numbers come from the same cached `compute_metrics` call the agent uses, so the cards and the report always agree.
- **Colour modes** (Display → Colour by):
  - **Residue Index Spectrum**: a classic blue → teal → green → ochre → orange → red spectrum from bead 1 to bead N of the view (PyMOL-style), for following the chain.
  - **Epigenomic Signal Heatmap**: low signal blue → cyan → ochre → red → **magenta** for the highest signal (1st–99th percentile of the view). It shows whichever track is active: H3K27ac or the state's own track.
  - The original modes (Genomic position, GC content, H3K27ac, Monochrome) are unchanged.
- **Exports in the ChronoAgent panel:**
  - **Structure (.pdb) · current state**: the structure in view, wwPDB, with the same REMARK 250 locus records as 04 Export, so it re-imports onto the right bins.
  - **Report (ChronoCell_Analysis_Report.md)**: sample table (state, locus, provenance), structural metrics, the other states, your question, the full analysis (LLM or offline, labelled), method notes and the disclaimer.
- The 04 Export expander (PDB/XYZ/bundle/JSON/CSV) is unchanged.

---

## 15. v3.2: Compare

**Purpose:** two structures side by side, with linked cameras, and what changed between them.

- **Left / Right:** any loaded structure (the reference model, every uploaded, slot or state file). The defaults are Healthy Control on the left and the current state on the right.
- **Superposition:** the right structure is Kabsch-aligned onto the left one over their shared bins. Reflection is allowed, because contact data fix a fold only up to its mirror image; the page says when the mirror was used. The same camera then shows the same region of both.
- **Linked cameras** (`ui/sync_view.py`):
  - Both figures are drawn by plotly.js in one `st.iframe`. On every `plotly_relayouting` / `plotly_relayout` event, `scene.camera` is copied to the other view, throttled to animation frames.
  - plotly.js is served from `static/` (static serving is enabled in `.streamlit/config.toml`), so it works offline. It falls back to the public CDN if static serving is off (e.g. a server started before the setting existed).
  - This was verified in a browser: rotating one view sets the other's camera exactly.
- **Colour:**
  - *Difference between the two*: per-bead distance after superposition, pale → dark red.
  - Genomic position.
  - Each side's own signal on a shared scale.
- **Region:** whole shared region, the most different 800 beads, or a custom Mb range.
- **Numbers:** R_g, span, ν, packing, crowded beads, mean signal, contact decay γ and TAD count, each with the change and a plain meaning.
- **Where they differ most:** a smoothed difference profile, the top 5 loci, and the genes at each.
- **Download:** a side-by-side PNG.

## 16. v3.2: Drug lab

**Model** (`chronocell/therapy.py`): a mechanism sandbox, not pharmacology. Each drug class = *where it acts* + *which way it pushes*.

| Drug class | Target (per-bead weight) | Direction |
|---|---|---|
| EZH2 / EED inhibitor | crowded **and** low-signal beads (Polycomb-like) | open (+1) |
| HDAC inhibitor | low-signal beads (least acetylated) | open (+1) |
| BET bromodomain inhibitor | top-quartile signal hubs, broadened (super-enhancer-like) | compact (−1) |
| CTCF / cohesin loop stabiliser | loop anchors ± 2 beads (measured loops, else the longest-range 3D contacts) | pull anchors together (0) |

- Signal percentiles are ranked against the **whole chromosome** (the healthy track when available), so a region-wide gain is visible.
- **With a healthy baseline** (same beads): targeted beads move `dose × max-effect × weight` of the way to their healthy positions. This happens *only where that agrees with the drug's direction* (crowding gate): an opening drug never compacts.
- **Restoration** = `100 × (1 − RMSD(treated, healthy) / RMSD(untreated, healthy))`.
- **Without a baseline:** fold-scale moves about ±30-bead centres, in 4 rounds with relaxation in between.
- Every conformation is relaxed (bond lengths → b₀, excluded volume) so it stays a valid polymer.
- Doses 0, 10, …, 100 % are pre-computed, so the **dose slider under the 3D view** animates instantly in the browser.
- **Outputs:** cards for restoration at full dose, R_g untreated → treated vs healthy, P(s) slope (−γ) vs healthy, and the share of beads reached. Also dose–response curves, a ranking of all drug classes ("which mechanism fits this fold"), and a CSV.
- The best-matching drug is pre-selected and marked ★. A drug pushing the wrong way explains why it does nothing.
- **Validated behaviour** (tests):
  - Demo tumour (over-open, hyper-acetylated): BET restores > 10 % while HDAC and EZH2 restore < 2 %.
  - Demo senescent (over-compact): HDAC > 5 %, BET < 2 %.
  - Restoration rises with dose; bonds stay ≈ b₀.
- **Limits:** there are no pharmacokinetics, no cell-type specificity and no measured drug data. The page and dossier say so.

## 17. v3.2: Genes and multi-omics

**Data:** 19,386 genes, one RefSeq Select / MANE transcript per gene (UCSC REST API, hg38 `ncbiRefSeqSelect`), stored in `chronocell/data/genes_hg38.json.gz`.

**Accessibility** (`chronocell/genes.py`), at each gene's promoter bead (TSS), relative to the region shown:
- openness = −z(crowding), where crowding = non-bonded beads within 1.5 b₀, smoothed ± 2 beads;
- activity = z(log(1 + signal)), skipped when the signal is a placeholder;
- score = the mean of the two.
- Status: ≥ +0.5 **hyper-accessible (predicted active)**; ≤ −0.5 **buried (predicted silenced)**; otherwise intermediate.
- Curated flags: cancer genes (~100 well-established drivers and suppressors) and neuro-disease genes (Parkinson's, Alzheimer's, ALS, Huntington's, 22q11.2).

**The page:**
- **Gene search** covering every gene on the loaded structure.
- A **3D view** coloured buried (blue) → open (terracotta), with labels for disease genes, the most open and most buried genes, and the chosen gene (◆).
- **Touches in 3D:** genes whose promoters lie within 2 b₀ of the chosen one, with how far apart they are along the DNA.
- A filterable **table** with CSV download.
- **RNA-seq** (upload, or a state's expression file): measured values next to predictions, a scatter plot, and the Spearman ρ between accessibility and log expression, the honest test of the prediction.

## 18. v3.2: Neighbourhoods, patient data, exports, API key

**Neighbourhoods** (`chronocell/domains.py`):
- They work from measured contacts, or from **3D proximity** (beads closer than 1.5 b₀) when a structure has no contacts.
- **Insulation score:** a difference array in O(C + N), window ≈ 500 kb.
- **Boundaries:** minima deeper than 0.15 log2.
- **Compartments:** the first eigenvector of the O/E Pearson map on ≤ 500 coarse bins, oriented by GC / signal.
- **Candidate loops:** O/E ≥ 3, count ≥ max(p90, 5), 5 beads to 2 Mb, with non-maximum suppression; measured contacts only.
- **Contact decay γ.**
- **Where it appears:** the 3D inspector (05 Neighbourhoods), the colour modes *A/B compartment* and *TAD domains*, ChronoAgent and the dossier.
- On the reference model, boundaries from contacts and from 3D proximity agree within about 1 bead (median).

**Patient data** (`chronocell/ingest.py`, recognised by content):

| Kind | Formats |
|---|---|
| Tracks | `.npy`; `.bedGraph` / `.bdg` (length-weighted mean per bin); `.bed` (score-weighted coverage; BED3 = coverage); `.bigWig` (optional pyBigWig) |
| Contact maps | `.cool` / `.mcool` read with h5py (finer resolutions aggregated); `.hic` (optional hic-straw, else hic2cool); text tables with 3 columns (bin bin count), 5 columns (chrom pos chrom pos count) or 6–7 columns (BEDPE) |
| Expression | two columns: gene, value |

- A state's own contact map becomes that state's graph. The Data menu's "Graph or contact map" accepts contact maps too.

**Exports:**
- **PDF research dossier** (ChronoAgent panel → *Build PDF dossier*). It contains a 3D snapshot (healthy vs current side by side when available), metrics, a crowding histogram, other states, a gene table, neighbourhoods, the drug-lab result, the full analysis and method notes. "Research use only" appears on page 1 and every footer. It uses a Unicode font where available, with a Latin-1 fallback.
- **Animated GIF** (4D → 03 Export).
- A **PNG camera button** on the 3D viewport.
- A **side-by-side PNG** on Compare.
- A **Markdown report** and **PDB**, as before.

**API key:**
- Paste it into the sidebar, **or** copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and set `GEMINI_API_KEY` or `OPENROUTER_API_KEY` (optionally `CHRONOAGENT_MODEL`). Environment variables with the same names also work.
- The sidebar field always overrides the stored key.
- `secrets.toml` is in `.gitignore`.

**Demo patients:**
- The sidebar switch writes synthetic Healthy / Disease / Senescent chr22 files into `.chronocell_cache/demo_states/`, once. Your `coordinates/` folder is never touched.
- The files are labelled `demo/…` everywhere.

**Fast start:** the reference model is cached in `.chronocell_cache/`. The cache is keyed by the generator's code, so a code change rebuilds it.

## 19. v3.2: errors found and fixed

| # | Symptom | Cause | Fix |
|---|---|---|---|
| 18 | Deprecation warning on every Compare render ("replace `st.components.v1.html` with `st.iframe`", removal date passed) | Old component API | `st.iframe` (with a fallback for older Streamlit) |
| 19 | "Cannot serialize the return value (TreatmentResult)" right after a code reload | `st.cache_data` pickles results; an object from a reloaded module no longer matched its class | Caches that return project objects (treatment, metrics, domains, window physics, 4D simulation, folder scan) use `st.cache_resource` (no pickling) |
| 20 | RNA-seq tables rejected ("no gene-name column") | pandas 3 gives text columns a string dtype, not `object` | Gene column = first non-numeric column |
| 21 | A 3-column BED was read as an expression table | Classifier required ≥ 4 columns for tracks | BED3+ recognised as a track |
| 22 | PDF / GIF buttons failed outside a fragment rerun | `st.rerun(scope="fragment")` called during a full run | Build-then-download in the same run |
| 23 | A header-less 3-column float CSV could be mistaken for contacts | Ambiguous column structure | Contacts need non-negative integer bins; float triples stay coordinates |
| 24 | Drug lab opened on a drug with no effect on the demo tumour (0 %) | First drug in the list by default | The best-matching mechanism is pre-selected and marked ★; a wrong-direction drug explains itself |
