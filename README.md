<![CDATA[# 🧬 ChronoCell-5D

> **AI-powered 3D / 4D chromatin structure workstation for human chromosomes**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.50%2B-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![Tests](https://img.shields.io/badge/Tests-83%20passing-brightgreen)]()
[![License](https://img.shields.io/badge/License-MIT-blue)]()

---

## 📖 Overview

**ChronoCell-5D** reconstructs and visualises the three-dimensional spatial fold of human chromosomes from experimental genomics data. It combines **contact-frequency embedding** with an **E(3)-equivariant graph neural network (EGNN)** to produce physically realistic 3D structures, then layers polymer-physics analytics, biological-state comparison, and an AI-powered interpreter on top.

### The "5D"

| Dimension | What it captures |
|-----------|-----------------|
| **X, Y, Z** | Spatial 3D coordinates of each genomic locus |
| **4th D** | Time-course playback & condition transitions |
| **5th D** | Biological state (Healthy → Disease/Cancer → Senescent) |

---

## ✨ Key Features

### 🔬 Two-Stage AI Reconstruction Pipeline
- **Contact embedding** — converts Micro-C pairwise contact frequencies into target 3D distances via the power-law relationship *M ∝ d⁻ᵅ*, then optimises free coordinates with a multi-term loss (distance, smoothness, clash, steric)
- **E(3)-Equivariant GNN** — refines coordinates conditioned on genomic features (GC content, H3K27ac signal), producing structures invariant to rotation, reflection, and translation

### 🧪 Polymer-Physics Analytics
- Radius of gyration (R_g), contact probability scaling P(s), Flory exponent (ν)
- Neighbour search, Kabsch RMSD alignment, persistence length
- Metric dashboard with live deltas across biological states

### 🧬 Biological State Comparison
- Compare **Healthy Control** vs **Disease State / Cancer** vs **Senescent State**
- Format-based data engine — files are recognised by content, not names
- Simulated disease rearrangements (deletions, inversions, translocations)

### 🤖 ChronoAgent — Structural Genomics Interpreter
- Natural-language queries about chromatin structure ("Which TADs contain the most active enhancers?")
- Online mode with **Gemini** or **OpenRouter** LLM (free API key)
- Offline heuristic engine for instant answers without a key
- Exports analysis reports (Markdown + PDB)

### 📊 Interactive 3D Viewport
- Lit triangle-mesh tube rendering with multiple colour modes
- Contact maps, distance matrices, and genomic-feature tracks
- Residue Index Spectrum and Epigenomic Signal Heatmap colouring
- Region presets (full chromosome, centromere, telomeres, immunoglobulin cluster)

### 📤 Standards-Compliant Export
- **wwPDB-conformant PDB** files (loadable in PyMOL, Chimera, VMD)
- XYZ, CSV, NPZ coordinate formats
- PDF research dossiers and Markdown analysis reports

---

## 🏗️ Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│                       USER INTERFACE (app.py)                         │
│   Data I/O  ·  3D Viewport (Plotly)  ·  Inspector Panels  ·  Export  │
│   States Panel  ·  ChronoAgent  ·  4D Workspace  ·  Drug Lab        │
├────────────────────────────────────────────────────────────────────────┤
│                     UI COMPONENTS (ui/)                               │
│   common · four_d · states_panel · agent_panel · compare             │
│   drug_lab · genes_view · guide · sync_view                          │
├────────────────────────────────────────────────────────────────────────┤
│                    CORE LIBRARY (chronocell/)                         │
│                                                                       │
│   genome.py      GRCh38 constants, cytobands, centromere, gaps       │
│   features.py    Per-bin GC fraction, H3K27ac binning, enhancer hubs │
│   physics.py     Polymer physics engine (losses, R(s), Rg, RMSD)    │
│   egnn.py        E(3)-equivariant GNN model + two-stage training     │
│   synthetic.py   Hilbert-curve fractal globule + planted contacts    │
│   formats.py     PDB/XYZ/NPZ/PT/CSV I/O, graph loaders             │
│   viz.py         Plotly 3D viewport, 2D analytics charts             │
│   theme.py       CSS design tokens, colour scales                    │
│   states.py      Biological state data engine                        │
│   agent.py       ChronoAgent LLM + heuristic interpreter            │
│   scenarios.py   Disease rearrangement simulations                   │
│   domains.py     TAD / compartment domain detection                  │
│   genes.py       Gene annotation and overlay                         │
│   ingest.py      Multi-format file ingestion                         │
│   therapy.py     Therapeutic target analysis                         │
│   pdf_report.py  PDF research dossier generation                     │
│   colab.py       Colab notebook builder utilities                    │
│                                                                       │
│   CLI tools:                                                          │
│   build_graph.py   FASTA + bigWig + mcool → graph .npz              │
│   train.py         graph → reconstructed 3D coordinates              │
│   benchmark.py     Accuracy evaluation on planted structures         │
├────────────────────────────────────────────────────────────────────────┤
│                       TEST SUITE (tests/)                             │
│   test_core · test_egnn · test_v3 · test_v31 · test_v32  (83 tests) │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.10+**
- **pip** (or conda)
- **PyTorch 2.2+** (for reconstruction; the viewer runs without it)

### Installation

```bash
# Clone the repository
git clone https://github.com/WolframNU13/Team-NU-13.git
cd Team-NU-13

# Install dependencies
pip install -r requirements.txt
```

### Run the App

```bash
streamlit run app.py
# Opens at http://localhost:8501
```

### Run Tests

```bash
python -m pytest          # 83 tests
```

---

## 💻 Usage

### Web App (Streamlit)

1. Launch with `streamlit run app.py`
2. The app starts with a reference model of human chromosome 22 (GRCh38, 10 kb resolution)
3. Upload your own coordinates, or run a GPU reconstruction via Colab

**Coordinates:** Drop files into `coordinates/<chrom>/` or upload via the *Data* panel. Until then, a clearly labelled reference model is shown.

**Biological States:** In the sidebar, pick *Healthy Control*, *Disease State / Cancer*, or *Senescent State*. Files are recognised by content (`.npy` coordinates, `.npy` signal tracks, `.pdb` structures).

**ChronoAgent 🤖:** A structural-genomics interpreter below the viewport.
- Paste a free Google AI Studio (Gemini) or OpenRouter key for LLM answers
- Without a key, an offline heuristic engine answers instantly

### Command-Line Pipeline

```bash
# Stage 1: Build the graph from raw genomics data
python -m chronocell.build_graph \
  --fasta chr22.fa \
  --bigwig H3K27ac.bigWig \
  --mcool sample.mcool \
  --out graph.npz

# Or use synthetic data for testing
python -m chronocell.build_graph --synthetic --out graph.npz

# Stage 2: Reconstruct 3D structure
python -m chronocell.train --graph graph.npz --out predicted_coords.npz

# Stage 3: Benchmark accuracy
python -m chronocell.benchmark

# Generate synthetic biological state files (optional)
python -m chronocell.demo_states demo_states
```

### GPU Reconstruction (Google Colab)

For heavy reconstruction jobs, use the provided Colab notebook:

1. Open `colab/ChronoCell5D_Colab.ipynb` on a Colab T4 runtime
2. Run all cells — outputs unzip straight into `coordinates/`
3. Restart the Streamlit app to pick up the new structures

---

## 📁 Project Structure

```
Team-NU-13/
├── app.py                    # Streamlit entry point
├── requirements.txt          # Python dependencies
├── pytest.ini                # Test configuration
├── README.md                 # This file
├── ARCHITECTURE.md           # Detailed architecture documentation
├── APP_GUIDE.md              # Complete application guide (every screen & function)
├── AUDIT.md                  # Audit of v1 → v3.2 with corrected equations
│
├── chronocell/               # Core Python package (23 modules)
│   ├── genome.py             # GRCh38 chromosome constants
│   ├── features.py           # Genomic feature extraction
│   ├── physics.py            # Polymer physics engine
│   ├── egnn.py               # E(3)-equivariant GNN
│   ├── synthetic.py          # Synthetic ground truth generator
│   ├── formats.py            # File I/O and PDB export
│   ├── viz.py                # Plotly visualisation builders
│   ├── theme.py              # Design tokens and CSS
│   ├── states.py             # Biological state engine
│   ├── agent.py              # ChronoAgent interpreter
│   ├── scenarios.py          # Disease rearrangements
│   ├── domains.py            # TAD / compartment detection
│   ├── genes.py              # Gene annotation
│   ├── ingest.py             # Multi-format ingestion
│   ├── therapy.py            # Therapeutic analysis
│   ├── pdf_report.py         # PDF report generation
│   ├── colab.py              # Colab utilities
│   ├── snapshot.py           # State snapshot management
│   ├── demo_states.py        # Synthetic state file generator
│   ├── build_graph.py        # CLI: raw files → graph
│   ├── train.py              # CLI: graph → 3D coordinates
│   ├── benchmark.py          # CLI: accuracy benchmark
│   └── data/                 # Reference data (hg38.json)
│
├── ui/                       # Streamlit UI components (10 modules)
│   ├── common.py             # Dataset assembly, coordinate slot, helpers
│   ├── four_d.py             # 4D workspace (time-course playback)
│   ├── states_panel.py       # Biological state selector
│   ├── agent_panel.py        # ChronoAgent chat interface
│   ├── compare.py            # State comparison views
│   ├── drug_lab.py           # Therapeutic target explorer
│   ├── genes_view.py         # Gene annotation overlay
│   ├── guide.py              # In-app usage guide
│   └── sync_view.py          # Synchronised multi-view
│
├── tests/                    # Test suite (83 tests)
│   ├── test_core.py          # Physics, genome, formats, features
│   ├── test_egnn.py          # EGNN equivariance & gradients
│   ├── test_v3.py            # v3 regression tests
│   ├── test_v31.py           # v3.1 feature tests
│   └── test_v32.py           # v3.2 feature tests
│
├── colab/                    # Google Colab notebook + utilities
│   ├── ChronoCell5D_Colab.ipynb
│   ├── chronocell_code.zip
│   └── pack_code.py
│
├── coordinates/              # Reconstructed coordinate files
├── legacy/                   # Pre-audit reference code (v1)
└── .streamlit/               # Streamlit configuration
```

---

## 🧮 How It Works

### 1. Graph Assembly
Raw genomics data (FASTA sequence, H3K27ac ChIP-seq, Micro-C contacts) is binned at 10 kb resolution into a graph where nodes carry genomic features and edges carry contact frequencies.

### 2. Contact Embedding
Pairwise contact counts are converted to target 3D distances via the power-law *M ∝ d⁻ᵅ*. Free 3D coordinates are optimised with a composite loss:
- **Distance loss** — fit the target distances
- **Smoothness** — penalise sharp bends in the polymer backbone
- **Clash penalty** — prevent steric overlaps
- **Polymer prior** — fill gaps where contacts are missing

### 3. EGNN Refinement
An E(3)-equivariant graph neural network refines the embedded coordinates conditioned on per-node features (GC content, H3K27ac signal). The model is provably equivariant to rotations, reflections, and translations — verified by automated tests.

### 4. Physics Analysis
The reconstructed structure is analysed with polymer-physics metrics: radius of gyration, contact probability decay P(s) ∝ s⁻¹, Flory exponent, persistence length, and TAD/compartment detection.

---

## 📚 Documentation

| Document | Description |
|----------|-------------|
| [ARCHITECTURE.md](ARCHITECTURE.md) | Detailed architecture, module reference, equations |
| [APP_GUIDE.md](APP_GUIDE.md) | Complete application guide — every screen, graph, and function |
| [AUDIT.md](AUDIT.md) | Audit trail from v1 → v3.2 with corrected equations |
| [coordinates/README.md](coordinates/README.md) | Coordinate file format specification |

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| **Web Framework** | Streamlit 1.50+ |
| **3D Visualisation** | Plotly 6.0+ (WebGL mesh3d) |
| **ML Framework** | PyTorch 2.2+ |
| **Scientific Computing** | NumPy, Pandas, SciPy |
| **Contact Maps** | h5py (HDF5 / .mcool) |
| **PDF Export** | fpdf2 |
| **AI Interpreter** | Gemini API / OpenRouter (optional) |
| **GPU Training** | Google Colab (T4) |
| **Testing** | pytest (83 tests) |

---

## 🧪 Testing

The test suite covers physics identities, EGNN equivariance, format round-trips, UI regressions, and feature correctness:

```bash
python -m pytest                    # Run all 83 tests
python -m pytest tests/test_core.py # Core physics & formats
python -m pytest tests/test_egnn.py # EGNN model tests
python -m pytest -v                 # Verbose output
```

---

## 👥 Team NU-13

Built for the hackathon by **Team NU-13** ([@WolframNU13](https://github.com/WolframNU13)).

---

## 📄 License

This project is licensed under the MIT License.
]]>
