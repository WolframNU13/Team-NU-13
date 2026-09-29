"""
Guide workspace: the whole app in plain language, with analogies, a guided tour, a glossary,
data formats, how to use your own AI key, and what is real versus simulated.
"""

from __future__ import annotations

import streamlit as st

from ui.common import html

ss = st.session_state


def _go(page: str, **extra) -> None:
    ss.workspace = page
    for k, v in extra.items():
        ss[k] = v


def render(version: str) -> None:
    html('<p class="cc-eyebrow" style="margin-top:14px">Guide · ChronoCell-5D ' + version + '</p>'
         '<h1 class="cc-title">The genome, folded — in plain words</h1>'
         '<p class="cc-lead">Every human cell holds about two metres of DNA, packed into a nucleus a hundred times '
         'thinner than a hair. <b>How</b> it is folded decides which genes can be read. In cancer, ageing and some brain '
         'diseases the folding goes wrong. ChronoCell-5D rebuilds that fold in 3D, watches it change over time (the 4th '
         'dimension), and explains what the changes mean (the 5th: interpretation).</p>')
    html('<div class="cc-analogy"><b>The big analogy.</b> Think of a chromosome as a very long necklace stuffed into a '
         'small box. Beads that sit near the lid are easy to reach (genes that are <b>switched on</b>); beads crushed at '
         'the bottom are hard to reach (genes that are <b>silenced</b>). Disease can repack the box.</div>')

    html('<h2 class="cc-h2">A 2-minute tour</h2>')
    html('<ol class="cc-steps">'
         '<li><b>Turn on demo patients</b> in the sidebar (Biological state → <i>Load demo patients</i>). You get a '
         'synthetic healthy, cancer and senescent chr22, clearly labelled as demo data.</li>'
         '<li><b>01 3D structure</b>: rotate the chromosome, read its size and signal in the cards above it.</li>'
         '<li>Pick <b>Disease State / Cancer</b> in the sidebar and watch the numbers change against Healthy.</li>'
         '<li><b>03 Compare</b>: healthy on the left, cancer on the right. Rotate one; the other follows. Dark red = '
         'what moved.</li>'
         '<li><b>04 Drug lab</b>: pick a drug class and drag the dose slider under the fold. See how far it moves the '
         'fold back toward healthy.</li>'
         '<li><b>05 Genes</b>: type a gene (BCR, NF2, CHEK2). Is it open (likely active) or buried (likely silenced)? '
         'Which genes does it touch in 3D?</li>'
         '<li>Scroll to <b>🤖 ChronoAgent</b> under any page, press <i>Analyse</i>, and download the Markdown report or '
         'the PDF dossier.</li></ol>')
    cols = st.columns(5)
    for col, (label, page) in zip(cols, [("Open 3D structure", "3D structure"), ("Open Compare", "Compare"),
                                         ("Open Drug lab", "Drug lab"), ("Open Genes", "Genes"),
                                         ("Open 4D dynamics", "4D dynamics")]):
        col.button(label, key=f"guide_go_{page}", on_click=_go, args=(page,), width="stretch")

    html('<h2 class="cc-h2">What each page does</h2>')
    left, right = st.columns(2, gap="large")
    with left:
        html('<div class="cc-callout"><h4>01 · 3D structure — the map</h4>The chromosome as a tube you can rotate. '
             'Each bead is a short stretch of DNA (10,000 letters on chr22). Colour it by position, activity marks '
             '(<i>Epigenomic Signal Heatmap</i>: blue quiet, magenta busy), neighbourhoods (TADs) or the classic '
             'rainbow. Panels on the right measure it like an engineer measures a coil.</div>')
        html('<div class="cc-callout"><h4>02 · 4D dynamics — the time-lapse</h4>Plays a time course, morphs one '
             'condition into another, or simulates a DNA accident (a deletion, or the swap that makes the '
             '"Philadelphia chromosome" in leukaemia) and lets the fold settle. Like a time-lapse of a room being '
             're-arranged.</div>')
        html('<div class="cc-callout"><h4>03 · Compare — spot the difference</h4>Two states side by side with '
             'linked cameras, the structures overlaid as well as possible, and every bead coloured by how far it '
             'moved. The table says what changed and the list says which genes sit where it changed most.</div>')
    with right:
        html('<div class="cc-callout"><h4>04 · Drug lab — a flight simulator for drugs</h4>Epigenetic drugs work by '
             'loosening or tightening DNA packing in specific places. Pick one; the lab applies <i>where</i> it acts and '
             '<i>which way</i> it pushes, over a dose range, and measures how far the fold returns to healthy. It is a '
             'simulator for ideas, not a prediction for patients.</div>')
        html('<div class="cc-callout"><h4>05 · Genes — who is awake, who is asleep</h4>All 19,386 human genes are '
             'placed on the fold. A gene whose start sits in open, well-marked chromatin is flagged <i>predicted '
             'active</i>; one buried in a crowded clump <i>predicted silenced</i>. Add RNA-seq to check the '
             'prediction against measured activity.</div>')
        html('<div class="cc-callout"><h4>🤖 ChronoAgent — the expert on call</h4>Reads every number on screen and '
             'writes a short report: what the shape means, which drug mechanisms could be tested, which genes are '
             'likely affected. Offline rules work instantly; with your own AI key it answers free-form questions. '
             'Exports a Markdown report and a PDF dossier.</div>')

    html('<h2 class="cc-h2">Reading the numbers</h2>')
    html('<dl class="cc-gloss">'
         '<dt>R<sub>g</sub> (radius of gyration)</dt><dd>How big the ball of DNA is: the average distance of every bead '
         'from the centre. Bigger = more open. <i>Like measuring a ball of wool.</i></dd>'
         '<dt>Max 3D span</dt><dd>The widest distance across the fold, bead to bead.</dd>'
         '<dt>ν (compaction exponent)</dt><dd>How the size grows with length. ≈ 0.33: tightly crumpled like paper '
         'squeezed in a fist (normal for chromatin). ≈ 0.5: a loose coil like cooked spaghetti. Higher: stretched out.</dd>'
         '<dt>Packing fraction</dt><dd>How much of its own space the DNA fills (0.19 for our reference fold).</dd>'
         '<dt>Crowding</dt><dd>How many other beads touch a bead. Crowded spots behave like silenced '
         'heterochromatin.</dd>'
         '<dt>Signal (H3K27ac)</dt><dd>A chemical tag that marks active switches (enhancers) — like sticky notes saying '
         '"read me". Your own track can replace it.</dd>'
         '<dt>TAD</dt><dd>A self-contained neighbourhood of DNA that mostly touches itself: rooms in a house.</dd>'
         '<dt>A / B compartment</dt><dd>The busy city centre (A, active) versus the quiet suburbs (B, inactive).</dd>'
         '<dt>Contact decay γ, P(s) slope</dt><dd>How fast the chance that two pieces of DNA touch drops as they get '
         'further apart along the chain. Slope ≈ −1 for normal crumpled chromatin; steeper = looser.</dd>'
         '<dt>Restoration (%)</dt><dd>In the drug lab: how much closer to the healthy shape the treated fold is.</dd>'
         '<dt>Reference model</dt><dd>A synthetic stand-in shown until real data arrive — a mannequin in the shop '
         'window until the real person walks in.</dd></dl>')

    html('<h2 class="cc-h2">How accurate is it? Two scores, never mixed</h2>')
    html('<div class="cc-analogy"><b>The exam analogy.</b> <i>Contact-map fit</i> is like checking a student against '
         'the homework they copied from: a high mark only shows they copied carefully. <i>Microscopy accuracy</i> is the '
         'real exam: questions they never saw. We report both, side by side, and never add them together.</div>')
    html('<dl class="cc-gloss">'
         '<dt>Contact-map fit</dt><dd>How well the 3D model reproduces the contact data it was built from (Spearman '
         '\u03c1, measured live on your window). It shows the fit converged, not that the shape is right.</dd>'
         '<dt>Microscopy accuracy</dt><dd>How well the <b>method</b> predicts distances measured under a microscope in '
         'cells it never saw (chromatin tracing, Bintu et al. 2018). The <b>population model</b> recovers 85.6 % of '
         'the folding pattern the experiment can reproduce; the older single-structure model recovers 45 %. It is a '
         'property of the method, not a measurement on your data. Full details are in <code>validation/RESULTS.md</code>.</dd>'
         '<dt>Population model</dt><dd>One chromosome folds differently in every cell, like a crowd of people each '
         'standing a little differently. The population model builds the whole crowd (100 trajectories), not one '
         'average person. Find it under 01 3D structure \u2192 03 Model &amp; convergence, on windows of up to 400 '
         'beads.</dd>'
         '<dt>Measure and Slicing plane</dt><dd><i>Measure</i> (above the view) gives the distance between any two '
         'beads, and for the population model the typical range across cells. <i>Display \u2192 Slicing plane</i> cuts '
         'the fold open like slicing a cake, to see inside.</dd></dl>')

    html('<h2 class="cc-h2">Bring your own data</h2>')
    st.dataframe([
        {"You have": "3D coordinates", "Formats": ".npy (N×3), .pdb, .npz bundle, .xyz, .csv",
         "Where": "sidebar → Add files to a state, Data → Coordinates, or coordinates/<chrom>/"},
        {"You have": "An activity / ChIP / ATAC track", "Formats": ".npy (one value per bead), .bedGraph, .bed, .bigWig*",
         "Where": "sidebar → Add files to a state"},
        {"You have": "Micro-C / Hi-C contacts", "Formats": ".cool, .mcool, .hic*, text table (bin bin count)",
         "Where": "sidebar (per state) or Data → Graph or contact map"},
        {"You have": "RNA-seq expression", "Formats": ".csv / .tsv (gene, value)", "Where": "05 Genes → Add measured expression"},
        {"You have": "Raw sequencing (FASTA + bigWig + mcool)", "Formats": "Colab notebook on a T4 GPU",
         "Where": "colab/ChronoCell5D_Colab.ipynb → unzip output into coordinates/"},
    ], hide_index=True, width="stretch")
    html('<p class="cc-note">* bigWig needs the optional package pyBigWig; .hic needs hic-straw (or convert with hic2cool). '
         'Files are matched to Healthy / Disease / Senescent by words in their name or folder (healthy, control, tumour, '
         'cancer, senescent…), or by uploading them into a state.</p>')

    html('<h2 class="cc-h2">Using your own AI key</h2>')
    html('<ol class="cc-steps">'
         '<li>Get a free key: <b>Google AI Studio</b> (aistudio.google.com → Get API key; starts with <code>AIza</code>) '
         'or <b>OpenRouter</b> (openrouter.ai → Keys; starts with <code>sk-or-</code>).</li>'
         '<li><b>Quick way:</b> paste it into the sidebar field <i>AI API Key</i>. It stays in this browser tab only.</li>'
         '<li><b>Permanent way:</b> copy <code>.streamlit/secrets.toml.example</code> to <code>.streamlit/secrets.toml</code> '
         'and put your key there (<code>GEMINI_API_KEY = "…"</code>). The app picks it up automatically; the file is '
         'git-ignored.</li>'
         '<li>Press <b>Analyse</b> in the ChronoAgent panel. One press = one request. Without a key, the offline '
         'engine answers instantly.</li></ol>')
    html('<div class="cc-callout"><h4>Keep the key private</h4>The key is sent only to the provider you choose, inside a '
         'request header, and never appears in reports or on screen. If you deploy the app publicly with a key in '
         'secrets, every visitor spends <i>your</i> quota: keep public deployments keyless or protected.</div>')

    html('<h2 class="cc-h2">What is real, and what is simulated</h2>')
    html('<div class="cc-callout"><b>Real:</b> the human genome map (chromosome sizes, bands, gaps, 19,386 genes from '
         'RefSeq), the physics formulas, the reconstruction algorithm, and anything computed from files you provide.'
         '<br><b>Synthetic / simulated:</b> the reference model and the demo patients (labelled everywhere), the 4D '
         'disease scenarios, and the drug lab (a mechanism simulator). Gene "active / silenced" labels are predictions '
         'from structure. ChronoAgent\'s therapy section lists research ideas, never medical advice.</div>')
