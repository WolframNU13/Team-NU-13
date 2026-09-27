# Coordinate slot

Put spatial coordinates here and ChronoCell-5D loads them automatically. Until something is here (or uploaded under **Data → Coordinates** or the sidebar's **Biological state**), the app shows the clearly labelled **reference model** for the selected chromosome.

```
coordinates/
  chr22/
    healthy.npz          # one condition (a Colab bundle; may hold many frames = the 4th dimension)
    tumour.npz           # another condition -> "Across conditions" in the 4D workspace
    graph.npz            # optional: GC, H3K27ac, Micro-C contacts for this chromosome (graph*.npz)
    healthy/             # biological-state folders (any depth; the words in the path pick the state)
      coords.npy         #   (N, 3) coordinates
      h3k27ac.npy        #   (N,)  signal track, paired with the structure automatically
    cancer/
      tumour_K562.pdb    #   wwPDB structure
    senescent/
      IMR90.npy
  chr4/
    ...
```

Accepted coordinate files:
- ChronoCell bundles (`.npz` with `frames`).
- ChronoCell PDB/XYZ exports: their headers carry chromosome, window and resolution.
- `predicted_coords.pt`.
- Plain `(N, 3)` arrays (`.npy`, `.csv`, `.npz`) and `(T, N, 3)` frame stacks.
- One-dimensional `.npy` arrays are **signal tracks**, never structures.

Also recognised by content (v3.2):
- **Tracks:** `.bedGraph`, `.bed`, `.bigWig`.
- **Contact maps:** `.cool`, `.mcool`, `.hic`, and bin/position text tables. A state's own contact map is used for that state.
- **RNA-seq tables** (gene, value), used by the Genes page.

Units: bundles declare nanometres. Anything else is calibrated so the median bond equals b₀ (see Data → Coordinate unit).

**Biological states.** A file belongs to *Healthy Control*, *Disease State / Cancer* or *Senescent State* when its name or a folder above it contains a state word:
- healthy, control, ctrl, wt, normal…
- cancer, tumour, tumor, disease, patient, mutant…
- senescent, senescence, OIS, aged…

Folders starting with `.` or `_` are ignored. The full rules are in APP_GUIDE.md §12.

The Colab notebook (`colab/ChronoCell5D_Colab.ipynb`) writes files in exactly this layout; unzip its output into this folder. To scan a different folder, set the environment variable `CHRONOCELL_COORDINATES`.
