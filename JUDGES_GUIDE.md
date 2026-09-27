# ChronoCell-5D, explained simply

*A guide for judges: what the website does, why it matters, and the ideas behind it, in plain words and everyday analogies. No biology or physics background needed.*

---

## The 30-second pitch

Every cell in your body carries about **two metres of DNA**, squeezed into a nucleus about 100 times thinner than a hair. The DNA isn't stuffed in randomly. **How it is folded decides which genes can be read.** When the folding goes wrong, cells can turn cancerous, age badly, or contribute to diseases such as Parkinson's.

**ChronoCell-5D is a workstation for that folding:**

| Dimension | What it means |
|---|---|
| 3D | The shape of a chromosome |
| 4D | How the shape changes over time or with disease |
| 5th dimension | An AI assistant, **ChronoAgent**, that explains what the changes mean |

You can:
- compare healthy and diseased cells side by side;
- see which genes end up switched on or buried;
- test **virtual drugs** that try to fold the DNA back into a healthy shape.

> **The one analogy to remember:** a chromosome is a very long **necklace stuffed into a small jewellery box.**
> - Beads near the lid are easy to reach: those genes are **on**.
> - Beads crushed at the bottom are hard to reach: those genes are **off**.
>
> Disease repacks the box. ChronoCell-5D shows you the box and what changed, and lets you try to repack it.

---

## 1. The problem, in one picture

- **DNA is an instruction book** with about 20,000 recipes (genes).
- **A cell never reads the whole book.** It keeps some pages open on the desk and locks others in a drawer. The folding decides which is which.
- **Folding is organised in layers, like a city:**

| Layer | Analogy |
|---|---|
| **Compartments A and B** | The busy city centre, where genes are active, and the quiet suburbs, where they're mostly off. |
| **TADs** (topologically associating domains) | The rooms of a house: DNA inside a room touches itself far more than it touches the next room. |
| **Loops** | Two distant points of the necklace held together by a clip, bringing a switch (enhancer) next to the gene it controls. |
| **Chemical marks** such as H3K27ac | Sticky notes that say "read me". Where there are many, genes tend to be active. |

In cancer, rooms get merged or split, clips move, and sticky notes pile up in the wrong places. In ageing (senescent) cells, parts of the DNA collapse into tight balls. **Seeing the shape is the first step to understanding and fixing it.**

---

## 2. How we get a 3D shape at all

Nobody can photograph a whole chromosome's fold directly. Instead scientists use **Micro-C / Hi-C** experiments, which produce a giant list of which pieces of DNA were touching which.

> **Analogy: the party seating chart.** You weren't at the party, but you have a list of who was seen talking to whom, and how often. People who talk often were probably sitting close together. From thousands of such clues you can rebuild the seating plan.

ChronoCell-5D does exactly that, in three steps:
1. **Convert "how often they touched" into "how far apart they were"**: often = close.
2. **Lay the chain out in 3D** so those distances are respected. This is like pinning a necklace so each pair of beads sits at the right distance.
3. **Polish it with an E(3)-equivariant graph neural network (EGNN).** This is a kind of AI that respects a basic fact of physics: *turning or mirroring an object doesn't change its shape*.
   - Analogy: a puzzle solver that gets the same answer however you hold the puzzle.
   - It also enforces physical rules: beads are chained at a fixed spacing and can't pass through each other.

The heavy version of this runs on a free **Google Colab GPU**, and the results drop straight into the website.

**Until real data are provided, the site shows a clearly labelled *reference model*.** Think of it as a mannequin in a shop window until the real person walks in.

---

## 3. A tour of the website

The website has **six pages** (top of the screen) and a **sidebar** (left). Under every data page sits the **ChronoAgent** panel.

### The sidebar: choosing whose DNA you're looking at

- **Load demo patients (synthetic):** one switch gives you a healthy, a "cancer" and a "senescent" chromosome 22. Every page then works without any data.
  - They are clearly labelled *demo* and are **not real patients**. They are for exploring the software.
- **Biological state:** Healthy Control, Disease State / Cancer, or Senescent State. Pick one and the whole site switches to that sample.
- **Bring your own files:** drop in 3D coordinates, activity tracks, contact maps or RNA-seq tables.
  - The site works out what each file is **from its contents, not its name**, like a postal sorter reading the parcel rather than trusting the label.
  - It files each one under the right state using words in its name (healthy, tumour, senescent, and so on).
- **AI API key:** optional. Paste your own free Google Gemini or OpenRouter key, or store it once in a settings file. Without a key, the offline expert still answers instantly.

### Page 01: 3D structure (the map)

**What you see:** the chromosome as a coloured tube you can spin, zoom and hover over. Each bead is a short stretch of DNA; on chr22 it's 10,000 letters.

**The four cards above the view:**

| Card | Plain meaning | Analogy |
|---|---|---|
| **Radius of gyration (R_g)** | Overall size of the fold | Measuring a ball of wool |
| **Max 3D span** | Widest distance across it | The ball's longest diameter |
| **Mean signal intensity** | Average "read me" marks | How many sticky notes there are |
| **Biological state** | Whose sample this is | The label on the sample tube |

When a healthy sample is loaded, each card also shows **"+12% vs Healthy"**, so a change jumps out immediately.

**Colour it different ways** (Display button):
- **Residue Index Spectrum:** a rainbow from one end of the DNA to the other, for following the chain.
- **Epigenomic Signal Heatmap:** blue where it's quiet, **red to magenta where it's busy**.
- **A/B compartment:** city centre (orange) vs suburbs (blue).
- **TAD domains:** alternating colours for each "room".
- The original modes are still there: position, GC content, activity mark, monochrome.

**The panels on the right** measure the fold like an engineer measures a spring:

| Panel | What it shows |
|---|---|
| **01 Polymer physics** | Size, length, and whether the fold is a tight crumpled ball or a loose coil |
| **02 Genomic features** | Sequence make-up, activity tracks, contact maps and the strongest activity hubs |
| **03 Model** | The equations, plus a button that rebuilds the 3D fold from contacts in front of you |
| **04 Export** | Standard files that other molecular viewers (PyMOL, ChimeraX) can open |
| **05 Neighbourhoods** | The rooms (TADs) and the city-centre/suburb split (compartments) |

The camera icon on the view saves a high-resolution picture.

### Page 02: 4D dynamics (the time-lapse)

**Analogy:** a time-lapse video of a room being rearranged.

- **Provided frames:** plays a real time course, such as a cell going through its cycle.
- **Across conditions:** morphs Healthy → Cancer → Senescent.
- **Simulated scenario:** stages a "DNA accident" and lets the fold settle. Examples:

| Scenario | The accident | Linked to |
|---|---|---|
| **22q11.2 deletion** | A missing piece of chromosome 22 | Much higher risk of early Parkinson's disease |
| **Philadelphia chromosome** | Chromosomes 9 and 22 swap ends, fusing two genes (BCR and ABL1) | Chronic myeloid leukaemia |
| **Ewing sarcoma** | Chromosomes 11 and 22 swap ends (EWSR1–FLI1) | A childhood bone cancer |
| **SNCA triplication** | Three copies of the Parkinson's gene SNCA | Inherited Parkinson's |

Press **Play** and watch the DNA move. You can export a movie file (PyMOL/ChimeraX) or an **animated GIF** for slides.

### Page 03: Compare (spot the difference)

**Analogy:** two photos of the same room, before and after, with changes highlighted.

- Healthy on the left, disease on the right.
- The two structures are **laid on top of each other** as closely as possible, so the same camera shows the same place.
- **The cameras are linked:** rotate or zoom one and the other follows.
- Colour **"Difference"** paints each piece of DNA by how far it moved. Pale means unchanged; dark red means moved a lot.
- A table compares size, compaction, crowding and neighbourhoods. A chart and list show **where** they differ most and **which genes live there**.
- A side-by-side picture can be downloaded.

### Page 04: Drug lab (a flight simulator for drugs)

**Analogy:** pilots train in a flight simulator before flying a real plane. The drug lab lets you "fly" a drug mechanism on a fold before any experiment.

Many modern cancer drugs are **epigenetic**: they don't attack the DNA sequence, they change how tightly it is packed. The lab turns each drug class into two simple facts:
1. **Where it acts**, for example on tightly packed, silent regions, or on over-active hubs.
2. **Which way it pushes**: opening the packing or tightening it.

| Drug class | Plain mechanism | Status in medicine |
|---|---|---|
| **EZH2 / EED inhibitor** | Stops the "keep this closed" mark, so tight regions loosen | Approved (EZH2) / trials (EED) |
| **HDAC inhibitor** | Stops cells erasing "open" marks, so DNA loosens broadly | Approved for some blood cancers |
| **BET inhibitor** | Pulls a reader protein off over-active hubs, which then settle down | Clinical trials |
| **CTCF / cohesin loop stabiliser** | Tightens the clips that fence DNA into rooms | Hypothetical / research |

**How to use it:**
1. Pick a region; by default it's the one where the fold is most abnormal.
2. Pick a drug. The lab marks the **"★ best match for this fold"**.
3. **Drag the dose slider** under the 3D view from 0% to 100%. The fold changes in real time.

**What you get:**
- **"Fold restored":** how much closer to the healthy shape the fold gets.
- **Dose–response curves:** size and "contact decay" against dose, with the healthy value as a dashed line.
- **A ranking** of which mechanism fits this particular fold best.

**It teaches the right lesson.** In the demo cancer, the damage is over-opened, over-active chromatin, so the BET inhibitor helps and the "opening" drugs do nothing. In the demo senescent cell the DNA is over-compacted, and the opening drugs help instead. **The right drug depends on the shape of the defect.**

> Honest label, shown on the page: this is a **mechanism simulator**, not a prediction of how well a drug works in patients.

### Page 05: Genes (who is awake, who is asleep)

**Analogy:** a hotel at night. Rooms facing the busy street, lights on and doors open, are awake. Rooms deep inside behind locked doors are asleep.

- All **19,386 human genes** (from the official RefSeq catalogue) are placed on the fold at their start position.
- For each gene the site measures how crowded its start is in 3D, and how many "read me" marks it carries. It then labels it:
  - **Hyper-accessible (predicted active)**: open and marked.
  - **Buried (predicted silenced)**: crowded and unmarked.
  - **Intermediate**: in between.
- Known **cancer genes** (BCR, NF2, CHEK2, TP53 and others) and **neuro-disease genes** (SNCA, COMT, LRRK2 and others) are flagged.
- **Pick a gene** (type "BCR") to see its status and **which other genes it touches in 3D**. These can sit millions of letters away along the DNA yet be neighbours in space, a classic hidden partnership.
- **Add RNA-seq** (measured gene activity) to check the prediction. The site reports how well "open in 3D" agrees with "actually active".
  - This is the honest test of the idea that shape controls activity.

### Page 06: Guide

A built-in plain-language manual, the 2-minute version of this document. Its buttons jump straight to each page.

### 🤖 ChronoAgent (the expert on call)

**Analogy:** a senior scientist who reads every gauge on your dashboard and writes you a short report.

- It uses the exact numbers on screen: size, compaction, crowding, activity, genes, neighbourhoods and your latest drug-lab result.
- It writes four sections:
  1. **Biophysical diagnosis**: what the shape means (for example, "decompacted, open domains").
  2. **Therapeutic strategy (research hypotheses)**: which drug mechanisms would be worth testing, and why.
  3. **Expression & accessibility insights**: which genes are likely switched on or off.
  4. **Answer to your question**: anything you type.
- **Two engines:**
  - **Offline rules** answer instantly, with no internet and no key.
  - **Your own AI key** (Google Gemini or OpenRouter) adds free-form answers. The AI is told to quote the numbers and to label every claim that goes beyond them as a hypothesis.
- **Three downloads:**
  - a Markdown report (`ChronoCell_Analysis_Report.md`);
  - the structure file (PDB);
  - a **PDF research dossier** with 3D pictures, all the metrics, a crowding chart, genes, neighbourhoods, the drug-lab result and the analysis.
- **Always labelled:** research use only, not medical advice.

---

## 4. The numbers, explained with everyday objects

| Number | Everyday meaning |
|---|---|
| **R_g (radius of gyration)** | How big a ball of wool is. Bigger = looser. |
| **ν (compaction exponent)** | How the size grows with length. **0.33** = paper crumpled tightly in your fist, which is normal for chromatin. **0.5** = a loose bowl of spaghetti. Higher = stretched out. |
| **Packing fraction** | How full the box is. |
| **Crowding** | How many other beads touch a bead, like people around you on a busy train. |
| **Contact decay (γ) / P(s) slope** | How fast the chance of two DNA pieces touching drops as they get further apart along the chain. **About −1** for normal crumpled chromatin; steeper means looser. |
| **Restoration %** | In the drug lab, how much of the way back to the healthy shape the treatment gets. |
| **Accessibility score** | Above +0.5: open, likely active. Below −0.5: buried, likely silenced. |

---

## 5. What is real, and what is simulated

We are strict about this, and the website labels it everywhere.

**Real:**
- The human genome map: chromosome sizes, cytogenetic bands, unassembled gaps and 19,386 genes, from public reference databases (UCSC / RefSeq).
- The physics formulas and the reconstruction method.
- Anything computed from files **you** provide.

**Synthetic or simulated, and labelled as such:**
- The reference model and the demo patients. They exist so the tool can be explored without data.
- The 4D disease scenarios. They show the geometric consequence of a DNA rearrangement, not a measured patient.
- The drug lab: a mechanism simulator, not an efficacy prediction.
- Gene "active / silenced" labels: predictions from shape and marks, which can be checked against RNA-seq.
- ChronoAgent's therapy section: research ideas, never treatment advice.

We measured the reconstruction's accuracy against structures whose true shape we knew (synthetic ones). The rebuilt shapes matched with a distance correlation of 0.93–0.97. Accuracy on real chromatin needs imaging data (chromatin tracing) that weren't available here.

---

## 6. Under the hood (for technical judges)

| Part | What it is |
|---|---|
| Interface | Streamlit (Python) with Plotly 3D. The heavy views are isolated so rotating or re-colouring doesn't reload the page. |
| Reconstruction | Contact-to-distance mapping (α = 3), shortest-path MDS initialisation, gradient refinement and an **E(3)-equivariant GNN** (Satorras et al., 2021), in PyTorch. Runs on CPU or a Colab T4 GPU. |
| Physics | Radius of gyration, R(s) scaling exponent, excluded volume, contact decay, gyration-tensor shape, crowding (cell-list neighbour search). |
| Neighbourhoods | TAD boundaries from the insulation score (Crane et al., 2015); A/B compartments from the eigenvector of the observed/expected correlation map (Lieberman-Aiden et al., 2009); candidate loops from enrichment. These work from measured contacts **or** from the 3D structure itself. |
| Drug lab | Mechanism-targeted, direction-gated moves toward the healthy fold (or local swelling/condensing without one), then polymer relaxation so the result stays physical. Doses 0–100% are pre-computed, so the slider is instant. |
| Genes | RefSeq Select / MANE annotation (UCSC REST API); promoter crowding and signal z-scores; Spearman agreement with RNA-seq. |
| Data formats | 3D: `.npy`, `.pdb`, `.npz`, `.xyz`, `.csv`. Tracks: `.npy`, `.bedGraph`, `.bed`, `.bigWig`. Contacts: `.cool`, `.mcool`, `.hic`, text tables. Expression: `.csv` / `.tsv`. Recognised by content. |
| AI | Google Gemini / OpenRouter over REST. The key travels only in a request header, is redacted from errors and never written to reports. Output is rendered as Markdown only. An offline fallback is always available. |
| Exports | wwPDB (single and multi-model), XYZ, bundle, JSON, CSV, PNG, GIF, Markdown report, PDF dossier. |
| Quality | **101 automated tests**, including an end-to-end run through every page, plus a real-browser check (for example, rotating one Compare view really rotates the other). |

---

## 7. A 3-minute demo script

1. **(20 s)** Open the site. It shows the reference chromosome 22, labelled as synthetic. In the sidebar, switch on **Load demo patients**.
2. **(30 s)** Page **01**: spin the chromosome. Point at the four cards. Switch the sidebar to **Disease State / Cancer** and read "+x% vs Healthy". Colour by **Epigenomic Signal Heatmap**: the magenta hotspots are over-active regions.
3. **(40 s)** Page **03 Compare**: healthy vs cancer. Rotate one view and the other follows. The dark-red region is what the disease changed; the list names the genes that live there.
4. **(40 s)** Page **04 Drug lab**: the BET inhibitor is marked ★ best match. **Drag the dose slider** and watch the fold contract toward healthy. Show the ranking, then switch to an HDAC inhibitor: "wrong direction for this defect, so no effect". Then say the key line: *"the right drug depends on the shape of the defect."*
5. **(30 s)** Page **05 Genes**: type **BCR** (the leukaemia gene). Is it open or buried, and which genes does it touch in 3D?
6. **(20 s)** Scroll to **ChronoAgent**, press **Analyse**, and download the **PDF dossier**.
7. **(Optional, 20 s)** Page **02 4D**: play the **Philadelphia chromosome** scenario.

---

## 8. Questions judges often ask

**Is this a diagnostic tool?**
No. It is a research workstation. Every simulated or predicted element is labelled, and the PDF says "research use only" on every page.

**Where would real data come from?**
Public projects such as 4DN and ENCODE publish Micro-C/Hi-C and ChIP-seq data for many cell types, including cancer lines. The Colab notebook turns those files into 3D structures that drop straight into the site.

**Why "5D"?**
3D space + time (4D) + interpretation, the step from "here is a shape" to "here is what it means and what you could test".

**What's new compared with existing genome browsers?**
Most tools show contact maps as flat 2D heatmaps. ChronoCell-5D combines, in one place:
- the 3D fold;
- its change over time and between states;
- genes and their predicted activity on that fold;
- a drug-mechanism sandbox;
- an explainer that turns numbers into a readable report.

**What would you do next?**
- Run the pipeline on public patient-derived Micro-C data.
- Validate against imaging (chromatin tracing).
- Train the equivariant network across many cells so it learns general folding rules.
- Calibrate the drug lab with before-and-after treatment data.

---

*ChronoCell-5D is for research and education, not medical advice. Full technical manual: `APP_GUIDE.md`.*
