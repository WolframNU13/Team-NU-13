"""
ChronoAgent research dossier (PDF, fpdf2).

Contents: sample and provenance, 3D snapshot(s), structural metrics, crowding distribution, the
other biological states, genes and their predicted accessibility, TADs / compartments, the drug-lab
result (if one was run), the full ChronoAgent analysis and method notes.

It is explicitly a research document: the first page and every page footer say it is not a
clinical diagnostic. Greek letters and symbols are printed with a Unicode system font (Arial /
DejaVu Sans); where none exists the text is transliterated to Latin-1 (nu, gamma, ~, >=).
"""

from __future__ import annotations

import datetime as dt
import io
import os
import re

import numpy as np

from . import agent as A, genome

# (regular, bold, italic, bold-italic); missing italic faces fall back to the upright ones.
FONT_CANDIDATES = [
    ("C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/arialbd.ttf", "C:/Windows/Fonts/ariali.ttf", "C:/Windows/Fonts/arialbi.ttf"),
    ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
     "/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-BoldOblique.ttf"),
    ("/System/Library/Fonts/Supplemental/Arial.ttf", "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
     "/System/Library/Fonts/Supplemental/Arial Italic.ttf", "/System/Library/Fonts/Supplemental/Arial Bold Italic.ttf"),
    ("/Library/Fonts/Arial.ttf", "/Library/Fonts/Arial Bold.ttf", "/Library/Fonts/Arial Italic.ttf",
     "/Library/Fonts/Arial Bold Italic.ttf"),
]
# Symbols not every font has (subscripts, arrows, math, emoji) -> safe equivalents, always applied.
_ALWAYS = {"₀": "0", "₁": "1", "₂": "2", "₃": "3", "→": "->", "←": "<-", "↔": "<->", "≥": ">=", "≤": "<=",
           "≈": "~", "✓": "v", "🤖": "", "–": "-", "—": "-", "…": "...", "−": "-", "√": "sqrt", "‖": "|", "∝": "~",
           "Σ": "sum ", "⟨": "<", "⟩": ">", "\u00a0": " ", "\u2009": " ", "\u202f": " "}
# Only when no Unicode font is available.
_LATIN = {"ν": "nu", "γ": "gamma", "α": "alpha", "ρ": "rho", "κ": "kappa", "λ": "lambda", "Δ": "delta ", "π": "pi",
          "“": '"', "”": '"', "‘": "'", "’": "'", "•": "-"}
INK, MUTED, ACCENT, RULE, WARN_BG, WARN_INK = (28, 30, 27), (98, 100, 95), (51, 64, 209), (216, 216, 211), (247, 237, 228), (122, 62, 18)
DISCLAIMER_SHORT = "Research use only - computational model, not a clinical diagnostic."


def _font_paths():
    for regular, bold, italic, bold_italic in FONT_CANDIDATES:
        if os.path.exists(regular) and os.path.exists(bold):
            it = italic if os.path.exists(italic) else regular
            bi = bold_italic if os.path.exists(bold_italic) else bold
            return regular, bold, it, bi
    return None


class _Doc:
    def __init__(self):
        from fpdf import FPDF

        class PDF(FPDF):
            def footer(self_inner):
                self_inner.set_y(-12)
                self_inner.set_font(self_inner.body_font, "", 7.5)
                self_inner.set_text_color(*MUTED)
                self_inner.cell(0, 5, self_inner.clean(f"ChronoCell-5D research dossier · {DISCLAIMER_SHORT} · page {self_inner.page_no()}"),
                                align="C")

        self.pdf = PDF(format="A4", unit="mm")
        self.pdf.set_auto_page_break(True, margin=16)
        self.pdf.set_margins(16, 16, 16)
        fonts = _font_paths()
        if fonts:
            for style, path in zip(("", "B", "I", "BI"), fonts):
                self.pdf.add_font("Body", style, path)
            self.unicode = True
            self.pdf.body_font = "Body"
        else:
            self.unicode = False
            self.pdf.body_font = "helvetica"
        self.pdf.clean = self.clean

    def clean(self, text) -> str:
        t = str(text)
        for a, b in _ALWAYS.items():
            t = t.replace(a, b)
        if not self.unicode:
            for a, b in _LATIN.items():
                t = t.replace(a, b)
            t = t.encode("latin-1", "replace").decode("latin-1")
        return t

    # ---- primitives ------------------------------------------------------------------
    def font(self, size: float, bold: bool = False, color=INK):
        self.pdf.set_font(self.pdf.body_font, "B" if bold else "", size)
        self.pdf.set_text_color(*color)

    def para(self, text: str, size: float = 9.5, bold: bool = False, color=INK, h: float = 4.8, markdown: bool = False,
             indent: float = 0.0):
        self.font(size, bold, color)
        self.pdf.set_x(self.pdf.l_margin + indent)
        self.pdf.multi_cell(self.pdf.epw - indent, h, self.clean(text), markdown=markdown, new_x="LMARGIN", new_y="NEXT")

    def heading(self, text: str, size: float = 13):
        self.pdf.ln(3)
        self.font(size, True)
        self.pdf.multi_cell(self.pdf.epw, 7, self.clean(text), new_x="LMARGIN", new_y="NEXT")
        y = self.pdf.get_y()
        self.pdf.set_draw_color(*RULE)
        self.pdf.line(self.pdf.l_margin, y, self.pdf.l_margin + self.pdf.epw, y)
        self.pdf.ln(2)

    def banner(self, text: str):
        self.pdf.set_fill_color(*WARN_BG)
        self.font(8.8, True, WARN_INK)
        self.pdf.multi_cell(self.pdf.epw, 5, self.clean(text), fill=True, new_x="LMARGIN", new_y="NEXT", padding=2)
        self.pdf.ln(2)

    def table(self, rows: list[list], widths: tuple, header: bool = True, size: float = 8.3):
        from fpdf.fonts import FontFace
        self.font(size)
        with self.pdf.table(col_widths=widths, line_height=4.6, text_align="LEFT", first_row_as_headings=header,
                            headings_style=FontFace(emphasis="BOLD", color=INK, fill_color=(236, 236, 231)),
                            borders_layout="HORIZONTAL_LINES") as t:
            for r in rows:
                row = t.row()
                for c in r:
                    row.cell(self.clean("" if c is None else c))
        self.pdf.ln(2)

    def image(self, png: bytes, width_frac: float = 1.0):
        w = self.pdf.epw * width_frac
        self.pdf.image(io.BytesIO(png), x=self.pdf.l_margin + (self.pdf.epw - w) / 2, w=w)
        self.pdf.ln(2)

    def histogram(self, values: np.ndarray, label: str, bins: int = 24):
        v = np.asarray(values, float)
        v = v[np.isfinite(v)]
        if v.size < 5:
            return
        counts, edges = np.histogram(v, bins=bins)
        x0, y0, w, h = self.pdf.l_margin, self.pdf.get_y() + 2, self.pdf.epw * 0.6, 28
        if y0 + h + 12 > self.pdf.h - self.pdf.b_margin:
            self.pdf.add_page()
            y0 = self.pdf.get_y() + 2
        bw = w / bins
        self.pdf.set_fill_color(*ACCENT)
        for k, c in enumerate(counts):
            bh = h * c / max(counts.max(), 1)
            self.pdf.rect(x0 + k * bw + 0.3, y0 + h - bh, bw - 0.6, bh, style="F")
        self.pdf.set_draw_color(*RULE)
        self.pdf.line(x0, y0 + h, x0 + w, y0 + h)
        self.font(7.5, color=MUTED)
        self.pdf.set_xy(x0, y0 + h + 1)
        self.pdf.cell(w, 4, self.clean(f"{label}: {edges[0]:.0f} ... {edges[-1]:.0f} (n = {v.size:,}; median {np.median(v):.1f})"))
        self.pdf.set_y(y0 + h + 7)

    def markdown(self, md: str):
        """The subset of Markdown ChronoAgent (and LLMs) produce: ### headings, bullets, **bold**."""
        for raw in md.splitlines():
            line = raw.rstrip()
            if not line.strip():
                self.pdf.ln(1.5)
                continue
            line = line.replace("`", "").replace("--", "-")
            line = re.sub(r"(?<![\w*])_(?!_)(.+?)(?<!_)_(?![\w*])", r"__\1__", line)     # _italic_ -> fpdf2 italic
            if re.match(r"^\s*#{1,6}\s", line):
                self.pdf.ln(1.5)
                self.para(re.sub(r"^\s*#{1,6}\s*", "", line).replace("**", ""), size=10.5, bold=True, h=5.6)
            elif re.match(r"^\s*([-*]|\d+\.)\s+", line):
                depth = (len(line) - len(line.lstrip())) // 2
                body = re.sub(r"^\s*([-*]|\d+\.)\s+", "", line)
                self.para("•  " + body if self.unicode else "-  " + body, size=9.2, markdown=True, indent=3 + 4 * depth)
            elif line.lstrip().startswith(">"):
                self.para(line.lstrip("> "), size=9.2, color=MUTED, markdown=True, indent=3)
            else:
                self.para(line, size=9.2, markdown=True)

    def output(self) -> bytes:
        return bytes(self.pdf.output())


def build(ctx: A.AgentContext, analysis_md: str, engine: str, query: str = "", images: list[tuple[str, bytes]] = (),
          crowding: np.ndarray | None = None, genes_table=None, genes_summary: dict | None = None,
          domains_summary: dict | None = None, therapy: dict | None = None, software: str = "ChronoCell-5D") -> bytes:
    d = _Doc()
    pdf = d.pdf
    pdf.add_page()
    when = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    d.font(20, True)
    pdf.multi_cell(pdf.epw, 9, d.clean("ChronoCell-5D · Research dossier"), new_x="LMARGIN", new_y="NEXT")
    d.para(f"{ctx.state} · {ctx.chrom} · {ctx.locus} · generated {when} · engine: {engine}", size=9, color=MUTED)
    pdf.ln(2)
    d.banner("FOR RESEARCH USE ONLY. This dossier describes a computational model of chromatin structure. It is not a "
             "clinical diagnostic, and the therapeutic section lists research hypotheses, not treatment recommendations.")

    d.heading("1. Sample and provenance")
    d.table([["Field", "Value"],
             ["Biological state", ctx.state + ("" if ctx.state_has_data else " (no files for this state; another source shown)")],
             ["Chromosome · assembly", f"{ctx.chrom} · {genome.ASSEMBLY} · {ctx.resolution / 1000:g} kb beads"],
             ["Region", f"{ctx.region} · {ctx.locus}"],
             ["Structure", ctx.structure + (" (EGNN reconstruction)" if ctx.reconstruction else "")],
             ["Signal / tracks", ctx.tracks],
             ["Synthetic reference model", "yes: illustrates the method, not biology" if ctx.is_reference else "no"],
             ["Bond length b0", f"{ctx.b0:.0f} nm"]], (45, 135))

    for title, png in images:
        d.heading(f"3D snapshot · {title}", size=11)
        d.image(png)

    d.heading("2. Structural metrics")
    d.table([["Metric", "Value"]] + [list(r) for r in A.metric_rows(ctx.metrics)], (110, 70))
    if crowding is not None:
        d.para("Distribution of local crowding (non-bonded beads within 1.5 b0 of each bead):", size=8.5, color=MUTED)
        d.histogram(crowding, "neighbours per bead")

    if ctx.comparisons:
        d.heading("3. Other biological states")
        rows = [["State", "Same beads", "Rg (nm)", "Span (nm)", "nu", "Packing", "Dense %", "Mean signal"]]
        for c in ctx.comparisons:
            o = c.metrics
            f = lambda v, nd=0: "n/a" if v is None or not np.isfinite(v) else f"{v:,.{nd}f}"  # noqa: E731
            rows.append([c.state, "yes" if c.same_window else "no", f(o.rg_nm), f(o.span_nm), f(o.nu, 3), f(o.packing, 3),
                         f(100 * o.dense_fraction, 1), f(o.mean_signal, 3)])
        d.table(rows, (40, 18, 20, 22, 16, 20, 18, 26))

    if genes_summary:
        d.heading("4. Genes on the fold (predicted accessibility)")
        d.para(f"{genes_summary.get('genes_in_view', 0)} genes with their promoter in view: "
               f"{genes_summary.get('open', 0)} hyper-accessible (predicted active), "
               f"{genes_summary.get('intermediate', 0)} intermediate, {genes_summary.get('buried', 0)} buried "
               "(predicted silenced). Predictions come from 3D crowding and signal at each promoter, relative to this region.",
               size=9)
        if genes_table is not None and len(genes_table):
            flagged = genes_table[genes_table["category"] != ""]
            show = flagged if len(flagged) else genes_table
            rows = [["Gene", "Locus (TSS)", "Flag", "Score", "Status", "Expression"]]
            for _, r in show.head(25).iterrows():
                ex = "" if not np.isfinite(r.get("expression", np.nan)) else f"{r['expression']:.2f}"
                rows.append([r["gene"], r["locus"], r["category"], f"{r['score']:.2f}", r["status"], ex])
            d.table(rows, (22, 36, 16, 14, 64, 26), size=7.8)

    if domains_summary:
        d.heading("5. Neighbourhoods: TADs and compartments")
        d.table([["Measure", "Value"]] + [[k.replace("_", " "), "n/a" if v is None else str(v)] for k, v in domains_summary.items()],
                (80, 100))

    if therapy:
        d.heading("6. Virtual drug lab (simulation)")
        d.para(f"{therapy['drug']} · mode: {therapy['mode']} · maximum effect {therapy['efficacy']:.0%}. "
               f"{therapy.get('plain', '')}", size=9)
        tab = therapy["table"]
        cols = [c for c in ("dose_pct", "rg_nm", "nu", "gamma", "packing", "restoration_pct") if c in tab.columns]
        rows = [[c.replace("_", " ") for c in cols]]
        for _, r in tab.iloc[::2].iterrows():
            rows.append([f"{r[c]:.0f}" if c in ("dose_pct", "rg_nm") else f"{r[c]:.3f}" if c != "restoration_pct"
                         else f"{r[c]:.1f}%" for c in cols])
        d.table(rows, tuple([180 / len(cols)] * len(cols)))
        d.para("A what-if simulation of the drug's mechanism on this fold (where it acts, which way it pushes). "
               "It is not a prediction of clinical efficacy.", size=8.3, color=MUTED)

    d.heading("7. ChronoAgent analysis")
    if query.strip():
        d.para(f"Question: {query.strip()}", size=9, color=MUTED)
    d.markdown(analysis_md)

    d.heading("8. Method notes")
    for line in ("Rg = sqrt(mean ||x_i - x_cm||^2); span = maximum pairwise bead distance; nu from RMS R(s) ~ s^nu.",
                 "Packing = N b0^3 / (8 R^3) with R = sqrt(5/3) Rg; crowding = non-bonded beads within 1.5 b0.",
                 "Gene accessibility: promoter crowding and signal, z-scored within the region shown.",
                 "TADs: insulation-score minima; compartments: first eigenvector of the O/E correlation map.",
                 A.DISCLAIMER):
        d.para("- " + line, size=8.5, color=MUTED)
    d.para(f"Generated by {software}.", size=8, color=MUTED)
    return d.output()
