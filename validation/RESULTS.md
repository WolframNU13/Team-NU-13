# ChronoCell-5D: accuracy against real microscopy

**Real data.** The ground truth is Bintu et al., *Science* 2018 (chromatin tracing). It gives the measured 3D position, in nanometres, of every 30 kb piece of DNA, in thousands of individual human cells. Public data: github.com/BogdanBintu/ChromatinImaging.

**Rerun:** `python validation/validate_tracing.py` (about 1 minute). On the first run it downloads the three tracing datasets (about 16 MB) into `validation/data/`, which is not committed. Raw numbers are in `validation/results.json`.

## How it was measured

1. **Split** the cells of each experiment at random into half A and half B.
2. **Input:** from half A, keep only how often each pair of DNA pieces touches (closer than 150 nm). This is the same kind of information Hi-C gives. Every distance in half A is thrown away.
3. **Build:** ChronoCell-5D builds the 3D fold from those touch frequencies, using the same pipeline as the app.
4. **Answer key:** half B's measured median distances between every pair of pieces. The model never saw half B.
5. **Score:** how well the model's distances rank-agree with the measured ones (Spearman ρ).
   - Run on three datasets × three random splits.
   - Compared with a lazy baseline that only knows "further along the DNA = further apart".
   - Compared with a ceiling: how well the two halves of the real experiment agree with each other.

## Results (mean ± SD over 3 splits)

| Dataset (cells) | Model ρ | Baseline ρ | Ceiling ρ | **Model ρ, trend removed** | Baseline, trend removed | Ceiling, trend removed | Model / ceiling |
|---|---|---|---|---|---|---|---|
| IMR90 chr21:28–30 Mb (4,832) | 0.86 ± 0.01 | 0.93 | 1.00 | **0.38 ± 0.02** | 0.01 | 0.98 | 39 % |
| A549 chr21:28–30 Mb (3,941) | 0.83 ± 0.01 | 0.91 | 0.99 | **0.51 ± 0.03** | 0.00 | 0.95 | 54 % |
| IMR90 chr21:18–20 Mb (1,277) | 0.51 ± 0.03 | 0.96 | 0.94 | **0.09 ± 0.02** | 0.00 | 0.25 | 36 % |

- Absolute size: model distances are about **0.3× the measured nanometres** (Lin's CCC 0.03–0.13). The bond length assumed for 30 kb beads (72 nm) is about 3× too small for this microscopy data.

## Honest reading

- **Real structure is captured.** After removing the obvious distance trend, the model reproduces **36–54 % of the structure the experiment itself can reproduce**. The baseline gets 0 %. This is the part that holds the biology: the TAD-like domains.
- **Overall ranking is below the baseline.** On overall rank agreement (ρ 0.51–0.86), the lazy baseline scores higher (0.91–0.96). A single rebuilt fold does not yet beat "distance along the DNA" on that measure.
- **Absolute nanometres are off by about 3×.** The model's length scale must be calibrated before quoting sizes in nm.
- **Where the data is weak, the model is weak.** On the region with little structure and few cells (18–20 Mb), even the experiment's two halves barely agree once the trend is removed (0.25), and the model follows (0.09).
- **Limit:** the input was imaging-derived contact frequencies, not sequencing Hi-C. A direct Hi-C → imaging test on IMR90 is the next step; the public Hi-C file could not be reached in the time available.
