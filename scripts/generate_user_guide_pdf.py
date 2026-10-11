#!/usr/bin/env python3
"""
Executive PDF User Manual Generator
====================================
Generates a publication-grade, beautifully styled user manual:
'HELP_how_to_use_app.pdf'

Design Specifications:
- Built with ReportLab Platypus and NumberedCanvas (running header, footer, dynamic page counts).
- Executive clinical theme: Deep Navy (#1E3A8A), Slate (#334155), Emerald (#059669), Soft Grey fills.
- Distinct 4-page structured layout with zero orphaned elements or split tables:
    Page 1: Title Header, Meta Overview, Section 1: Executive Overview & Quickstart
    Page 2: Section 2: Comprehensive Walkthrough (Part I: Primary Causal Discovery & In Silico Perturbation)
    Page 3: Section 2: Comprehensive Walkthrough (Part II: Cellular Hierarchy & Translational Pharmacology)
    Page 4: Section 3: Data Export & Integration, Section 4: Troubleshooting & FAQ, Citations
"""

import os
import sys
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.pdfgen import canvas

sys.stdout.reconfigure(line_buffering=True)

# -----------------------------------------------------------------------------
# Color Palette
# -----------------------------------------------------------------------------
COLOR_PRIMARY = colors.HexColor("#1E3A8A")     # Deep Navy
COLOR_SECONDARY = colors.HexColor("#0D5C75")   # Deep Teal / Slate Blue
COLOR_BODY = colors.HexColor("#334155")        # Slate Body Text
COLOR_MUTED = colors.HexColor("#64748B")       # Muted Slate
COLOR_ACCENT = colors.HexColor("#059669")      # Emerald Accent
COLOR_ACCENT_BG = colors.HexColor("#ECFDF5")   # Light Emerald
COLOR_WARN = colors.HexColor("#D97706")        # Amber Accent
COLOR_WARN_BG = colors.HexColor("#FFFBEB")     # Light Amber
COLOR_BG_CARD = colors.HexColor("#F8FAFC")     # Soft Grey Background
COLOR_BG_CODE = colors.HexColor("#0F172A")     # Dark Slate / Terminal
COLOR_BORDER = colors.HexColor("#CBD5E1")      # Border Line
COLOR_WHITE = colors.HexColor("#FFFFFF")


# -----------------------------------------------------------------------------
# Numbered Canvas for Running Headers, Footers, and Dynamic Page Counts
# -----------------------------------------------------------------------------
class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_decorations(self, page_count):
        self.saveState()
        page_w = 612.0  # Letter width (8.5 in * 72)
        page_h = 792.0  # Letter height (11.0 in * 72)
        margin = 54.0   # 0.75 in * 72

        # Running Header (Pages 2+)
        if self._pageNumber > 1:
            self.setFont("Helvetica", 8)
            self.setFillColor(COLOR_MUTED)
            self.drawString(margin, page_h - 38, "AML Single-Cell Causal Discovery Dashboard | User Manual")
            self.setFont("Helvetica-Bold", 8)
            self.setFillColor(COLOR_PRIMARY)
            self.drawRightString(page_w - margin, page_h - 38, "GSE116256 Clinical Guide")
            
            # Header rule
            self.setStrokeColor(COLOR_BORDER)
            self.setLineWidth(0.6)
            self.line(margin, page_h - 43, page_w - margin, page_h - 43)

        # Running Footer (All Pages)
        self.setStrokeColor(COLOR_BORDER)
        self.setLineWidth(0.6)
        self.line(margin, 46, page_w - margin, 46)

        self.setFont("Helvetica", 8)
        self.setFillColor(COLOR_MUTED)
        self.drawString(margin, 34, "Antigravity Bioinformatics & Translational Oncology Platform")
        
        self.setFont("Helvetica-Bold", 8)
        self.setFillColor(COLOR_BODY)
        self.drawRightString(page_w - margin, 34, f"Page {self._pageNumber} of {page_count}")

        self.restoreState()


# -----------------------------------------------------------------------------
# Document Builder
# -----------------------------------------------------------------------------
def build_pdf(filename: Path):
    print(f"[*] Initializing PDF builder for: {filename}...", flush=True)
    doc = SimpleDocTemplate(
        str(filename),
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54,
    )
    printable_w = 504.0  # 612 - 108

    # Style Registry
    styles = getSampleStyleSheet()

    style_title = ParagraphStyle(
        "DocTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=COLOR_PRIMARY,
        spaceAfter=4,
    )
    style_subtitle = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=13.5,
        textColor=COLOR_MUTED,
        spaceAfter=10,
    )
    style_meta = ParagraphStyle(
        "DocMeta",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8.5,
        leading=12,
        textColor=COLOR_SECONDARY,
    )
    style_h1 = ParagraphStyle(
        "Heading1_Custom",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=12.5,
        leading=16,
        textColor=COLOR_PRIMARY,
        spaceBefore=8,
        spaceAfter=4,
        keepWithNext=True,
    )
    style_h2 = ParagraphStyle(
        "Heading2_Custom",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=10,
        leading=13.5,
        textColor=COLOR_SECONDARY,
        spaceBefore=6,
        spaceAfter=3,
        keepWithNext=True,
    )
    style_body = ParagraphStyle(
        "Body_Custom",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.4,
        leading=11.8,
        textColor=COLOR_BODY,
        spaceAfter=4,
    )
    style_body_bold = ParagraphStyle(
        "BodyBold_Custom",
        parent=style_body,
        fontName="Helvetica-Bold",
    )
    style_bullet = ParagraphStyle(
        "Bullet_Custom",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.2,
        leading=11.4,
        textColor=COLOR_BODY,
        leftIndent=12,
        firstLineIndent=-8,
        spaceAfter=2.5,
    )
    style_cli = ParagraphStyle(
        "CLI_Custom",
        parent=styles["Normal"],
        fontName="Courier-Bold",
        fontSize=8.0,
        leading=10.5,
        textColor=colors.HexColor("#38BDF8"),  # Bright Cyan
    )
    style_cli_comment = ParagraphStyle(
        "CLIComment_Custom",
        parent=styles["Normal"],
        fontName="Courier-Oblique",
        fontSize=7.5,
        leading=9.5,
        textColor=colors.HexColor("#94A3B8"),  # Slate Grey
    )
    style_callout_title = ParagraphStyle(
        "CalloutTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8.5,
        leading=11,
        textColor=COLOR_ACCENT,
    )
    style_callout_text = ParagraphStyle(
        "CalloutText",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.0,
        leading=11.0,
        textColor=colors.HexColor("#065F46"),
    )
    style_table_cell = ParagraphStyle(
        "TableCell",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=7.6,
        leading=10.0,
        textColor=COLOR_BODY,
    )
    style_table_cell_bold = ParagraphStyle(
        "TableCellBold",
        parent=style_table_cell,
        fontName="Helvetica-Bold",
        textColor=COLOR_PRIMARY,
    )
    style_table_hdr = ParagraphStyle(
        "TableHdr",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=7.8,
        leading=10.0,
        textColor=COLOR_WHITE,
    )

    def callout_box(title: str, text: str, kind: str = "tip"):
        bg_col = COLOR_ACCENT_BG if kind == "tip" else COLOR_WARN_BG
        brd_col = COLOR_ACCENT if kind == "tip" else COLOR_WARN
        t_style = style_callout_title if kind == "tip" else ParagraphStyle("WarnT", parent=style_callout_title, textColor=COLOR_WARN)
        b_style = style_callout_text if kind == "tip" else ParagraphStyle("WarnB", parent=style_callout_text, textColor=colors.HexColor("#92400E"))

        content = [
            Paragraph(f"<b>{title}</b>", t_style),
            Spacer(1, 2),
            Paragraph(text, b_style)
        ]
        t = Table([[content]], colWidths=[printable_w])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), bg_col),
            ('BOX', (0,0), (-1,-1), 0.8, brd_col),
            ('TOPPADDING', (0,0), (-1,-1), 5),
            ('BOTTOMPADDING', (0,0), (-1,-1), 5),
            ('LEFTPADDING', (0,0), (-1,-1), 8),
            ('RIGHTPADDING', (0,0), (-1,-1), 8),
        ]))
        return t

    def cli_box(lines: list[tuple[str, str]]):
        flowables = []
        for cmd, comment in lines:
            flowables.append(Paragraph(f"$ {cmd}", style_cli))
            if comment:
                flowables.append(Paragraph(f"# {comment}", style_cli_comment))
            flowables.append(Spacer(1, 2))
        
        t = Table([[flowables]], colWidths=[printable_w])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), COLOR_BG_CODE),
            ('BOX', (0,0), (-1,-1), 0.7, colors.HexColor("#334155")),
            ('TOPPADDING', (0,0), (-1,-1), 5),
            ('BOTTOMPADDING', (0,0), (-1,-1), 5),
            ('LEFTPADDING', (0,0), (-1,-1), 8),
            ('RIGHTPADDING', (0,0), (-1,-1), 8),
        ]))
        return t

    story = []

    # =========================================================================
    # PAGE 1: TITLE, EXECUTIVE OVERVIEW & RAPID DEPLOYMENT
    # =========================================================================
    header_table_data = [
        [
            Paragraph("<b>ANTIGRAVITY TRANSLATIONAL ONCOLOGY SUITE</b>", style_meta),
            Paragraph("<b>VERSION 3.0 | CLINICAL RELEASE</b>", ParagraphStyle("RMeta", parent=style_meta, alignment=2))
        ]
    ]
    t_hdr = Table(header_table_data, colWidths=[printable_w * 0.6, printable_w * 0.4])
    t_hdr.setStyle(TableStyle([
        ('BOTTOMPADDING', (0,0), (-1,-1), 2),
        ('TOPPADDING', (0,0), (-1,-1), 0),
        ('LEFTPADDING', (0,0), (-1,-1), 0),
        ('RIGHTPADDING', (0,0), (-1,-1), 0),
    ]))
    story.append(t_hdr)
    story.append(Spacer(1, 4))
    story.append(HRFlowable(width="100%", thickness=1.5, color=COLOR_PRIMARY, spaceBefore=0, spaceAfter=8))

    story.append(Paragraph("AML Single-Cell Causal Discovery & Drug Repurposing Dashboard", style_title))
    story.append(Paragraph("User & Exploration Guide: Interactive Graphical Modeling, In Silico Perturbations, Leukemic Hierarchy Rewiring & Clinical Pharmacology", style_subtitle))

    meta_box_data = [
        [
            Paragraph("<b>Target Audience:</b> Computational Biologists, Hematologist-Oncologists, Pharmacologists", style_table_cell),
            Paragraph("<b>Clinical Dataset:</b> van Galen et al. (GSE116256 / AML1012-D0)", style_table_cell),
        ],
        [
            Paragraph("<b>Inference Engine:</b> Depth-Bounded PC Algorithm (depth=2, Fisher-Z)", style_table_cell),
            Paragraph("<b>Web Application:</b> Streamlit v1.65+ on http://localhost:8501", style_table_cell),
        ]
    ]
    t_meta = Table(meta_box_data, colWidths=[printable_w * 0.55, printable_w * 0.45])
    t_meta.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), COLOR_BG_CARD),
        ('BOX', (0,0), (-1,-1), 0.7, COLOR_BORDER),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('RIGHTPADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(t_meta)
    story.append(Spacer(1, 8))

    story.append(Paragraph("1. Executive Overview & Rapid Deployment", style_h1))
    story.append(HRFlowable(width="100%", thickness=0.8, color=COLOR_SECONDARY, spaceBefore=0, spaceAfter=6))
    
    story.append(Paragraph(
        "Acute Myeloid Leukemia (AML) is characterized by blockades in hematopoietic differentiation and the clonal proliferation of immature leukemic blasts. "
        "Standard transcriptomic co-expression analyses (e.g., WGCNA) identify thousands of correlated gene pairs but fundamentally suffer from confounding: "
        "they cannot distinguish causative oncogenic drivers from passenger correlations, nor can they determine edge directionality ($X \\to Y$ vs. $Y \\to X$). "
        "This dashboard provides an autonomous, clinical-grade interface to explore a <b>rigorously constrained causal Directed Acyclic Graph (DAG)</b> "
        "derived from van Galen single-cell bone marrow aspirates.",
        style_body
    ))
    story.append(Spacer(1, 4))

    story.append(Paragraph("System Prerequisites & Local Execution:", style_h2))
    story.append(Paragraph(
        "The application is engineered on top of Python 3.10+ and Streamlit. All heavy computational graph discovery steps "
        "have been pre-computed and stored within the local <code>./results</code> directory. The dashboard serves real-time interactive physics networks, "
        "dynamic in silico knockout math, and subpopulation comparative queries without latency.",
        style_body
    ))

    cli_cmds = [
        ("cd \"D:\\new project on 10102026\"", "Navigate to the project root workspace"),
        ("streamlit run dashboard/app.py", "Launch dashboard locally (opens default browser at http://localhost:8501)"),
        ("streamlit run dashboard/app.py --server.port 8502", "Optional: specify alternate port if 8501 is occupied"),
        ("streamlit run dashboard/app.py --server.headless true", "Optional: execute in headless mode for remote cloud server hosting"),
    ]
    story.append(cli_box(cli_cmds))
    story.append(Spacer(1, 6))

    story.append(callout_box(
        "PRO-TIP: VERIFYING ARTIFACT AVAILABILITY & DIRECTORY PATH",
        "Upon launch, the sidebar checks the <code>./results</code> folder for key assets: <code>top_master_regulators.csv</code>, "
        "<code>stable_causal_edges.csv</code>, <code>in_silico_knockout_effects.csv</code>, <code>subpopulation_causal_diff.csv</code>, "
        "and <code>drug_target_interactions.csv</code>. If you store outputs in an alternate location, update the 'Artifacts Directory' path directly in the sidebar.",
        "tip"
    ))

    # Clean Break to Page 2
    story.append(PageBreak())

    # =========================================================================
    # PAGE 2: MODULE WALKTHROUGH (PART I: CAUSAL DISCOVERY & IN SILICO KO)
    # =========================================================================
    story.append(Paragraph("2. Module Walkthrough: Primary Causal Network & In Silico Perturbations", style_h1))
    story.append(HRFlowable(width="100%", thickness=0.8, color=COLOR_SECONDARY, spaceBefore=0, spaceAfter=6))

    story.append(Paragraph("Tab 1: Causal Network Explorer", style_h2))
    story.append(Paragraph(
        "The Causal Network Explorer visualizes the entire 55-node, 69-edge directed causal network using interactive PyVis physics simulation. "
        "Unlike undirected correlation networks, directed arrows indicate the inferred direction of regulatory transmission.",
        style_body
    ))
    story.append(Paragraph("&bull; <b>Causal Out-Degree Filter:</b> Sliders allow you to isolate hub master regulators controlling 3, 4, or 5+ downstream targets.", style_bullet))
    story.append(Paragraph("&bull; <b>Stability Frequency Cutoff:</b> Constrain the visible graph to edges meeting stringent bootstrap retention (default $\\ge 60\\%$, up to $100\\%$).", style_bullet))
    story.append(Paragraph("&bull; <b>Gene Search & Highlighting:</b> Typing a gene symbol (e.g., <code>NPM1</code> or <code>S100A9</code>) highlights its direct causal parents (upstream controllers in gold) and direct children (downstream targets in cyan).", style_bullet))
    story.append(Paragraph("&bull; <b>Physics Interaction:</b> Nodes can be clicked and dragged to untangle dense clusters; click 'Freeze Physics' in PyVis to fix positions for export.", style_bullet))
    story.append(Spacer(1, 4))

    story.append(Paragraph("Tab 2: Master Regulator Inspector", style_h2))
    story.append(Paragraph(
        "This module delivers deep-dive analytical profiling for individual driver genes. Selecting a candidate gene from the dropdown provides:",
        style_body
    ))
    story.append(Paragraph("&bull; <b>Topology Card:</b> Summarizes Master Regulator Score (MRS), global rank, out-degree, in-degree, and betweenness centrality.", style_bullet))
    story.append(Paragraph("&bull; <b>Upstream vs. Downstream Projections:</b> Displays all immediate upstream regulators and all downstream reachable cascade targets.", style_bullet))
    story.append(Paragraph("&bull; <b>Curated Oncogenic Biology Card:</b> Displays peer-reviewed functional mechanisms for established AML regulators (e.g., *NPM1* nucleolar stress, *FLT3* tyrosine kinase signaling, *BCL2* apoptotic threshold, *RUNX1* myeloid differentiation blockade).", style_bullet))
    story.append(Paragraph("&bull; <b>Actionable Therapeutics Card:</b> Directly lists approved and investigational compounds mapped to the regulator.", style_bullet))
    story.append(Spacer(1, 4))

    story.append(Paragraph("Tab 3: In Silico Knockout Simulator (Pearl's $do$-Operator)", style_h2))
    story.append(Paragraph(
        "A marquee innovation of this platform is the implementation of Judea Pearl's structural causal intervention calculus ($do(X = 0)$). "
        "Rather than computing passive conditional correlations, the simulator perturbs the upstream node to zero and propagates the causal shock "
        "downstream through linear Structural Causal Model (SCM) path coefficients ($\\beta_{u \\to v}$) following forward topological sort order.",
        style_body
    ))
    story.append(Paragraph("&bull; <b>Knockout Selector:</b> Interactively simulate complete ablations of top master regulators: <code>NPM1</code>, <code>NAMPT</code>, <code>S100A9</code>, <code>RNASE2</code>, and <code>FCN1</code>.", style_bullet))
    story.append(Paragraph("&bull; <b>Knockout Efficiency Slider (10% to 100%):</b> Model dose-dependent pharmacologic inhibition (e.g., 50% target knockdown vs. 100% CRISPR knockout).", style_bullet))
    story.append(Paragraph("&bull; <b>Cascade Hierarchy Filter:</b> Inspect direct children (causal distance = 1) versus deep indirect cascade targets (distance $\\ge 2$).", style_bullet))
    story.append(Paragraph("&bull; <b>Interactive Waterfall Bar Chart:</b> Displays percentage expression shifts ($\\Delta \\%$) across downstream genes, identifying catastrophic downstream collapses.", style_bullet))
    story.append(Spacer(1, 4))

    story.append(callout_box(
        "LANDMARK IN SILICO DISCOVERIES: TARGETED CASCADE COLLAPSES",
        "1. <b>NPM1 -> MRPL42 (-62.5%):</b> Ablating NPM1 starves mitochondrial ribosomal protein translation, explaining why NPM1 disruption selectively impairs oxidative phosphorylation in leukemic stem cells.<br/>"
        "2. <b>S100A9 -> THBS1 (-87.9%) & VCAN (-52.0%):</b> Silencing S100A9 shuts down downstream extracellular matrix protection and cell adhesion.<br/>"
        "3. <b>RNASE2 -> S100A8/A9 (-32.1% / -25.4%):</b> RNASE2 acts as an apex trigger destabilizing the calprotectin alarmin complex.",
        "tip"
    ))

    # Clean Break to Page 3
    story.append(PageBreak())

    # =========================================================================
    # PAGE 3: MODULE WALKTHROUGH (PART II: HIERARCHY REWIRING & PHARMACOLOGY)
    # =========================================================================
    story.append(Paragraph("2. Module Walkthrough: Differentiation Hierarchy & Translational Pharmacology", style_h1))
    story.append(HRFlowable(width="100%", thickness=0.8, color=COLOR_SECONDARY, spaceBefore=0, spaceAfter=6))

    story.append(Paragraph("Tab 4: Leukemic Hierarchy & Subpopulation Rewiring", style_h2))
    story.append(Paragraph(
        "Therapeutic failure in AML often stems from Leukemic Stem Cells (LSCs) that harbor distinct transcriptional wiring compared to mature blasts. "
        "This module stratifies single cells into <b>Immature / Stem-like Blasts</b> (LSC-like / HSC-like / Prog-like, $n=535$) and "
        "<b>Differentiated Blasts</b> (Mono-like / Gran-like / cDC-like, $n=535$) and evaluates causal edge rewiring via 100x bootstrap stability.",
        style_body
    ))
    story.append(Paragraph("&bull; <b>The Four Functional Rewiring Categories:</b>", style_body_bold))
    story.append(Paragraph("  1. <i>Stemness-Maintained ($n=16$):</i> Active selectively in LSC-like blasts (e.g., <code>HDC -> MS4A2</code> $\\Delta \\beta = +1.07$, <code>PRG2 -> CLC</code> $\\Delta \\beta = +0.85$, <code>RNASE2 -> SRGN</code> $\\Delta \\beta = +0.42$). These represent selective anti-leukemic stem cell vulnerabilities.", style_bullet))
    story.append(Paragraph("  2. <i>Maturation-Induced ($n=26$):</i> Regulatory cascades emerging exclusively upon differentiation (e.g., <code>HDC -> CXCL8</code> $\\Delta \\beta = -1.08$, <code>TPSB2 -> MS4A2</code> $\\Delta \\beta = -0.81$, <code>MTMR7 -> NAMPT</code> $\\Delta \\beta = -0.48$).", style_bullet))
    story.append(Paragraph("  3. <i>Conserved Core ($n=11$):</i> Baseline transcriptional machinery operating invariantly across all maturation states (e.g., <code>SRGN -> ELANE</code>, <code>NAMPT -> RGS2</code>).", style_bullet))
    story.append(Paragraph("  4. <i>Context-Dependent ($n=16$):</i> Transitional or low-magnitude interactions.", style_bullet))
    story.append(Spacer(1, 4))

    story.append(Paragraph("Tab 5: Centrality & Target Ranking", style_h2))
    story.append(Paragraph(
        "Ranks all 55 network genes by graph-theoretical centrality metrics. Genes with high <b>Out-Degree</b> govern broad downstream transcriptional programs, "
        "while high <b>Betweenness Centrality</b> indicates vital network information bottlenecks. "
        "The composite Master Regulator Score (MRS) combines Out-Degree (50%), Betweenness Centrality (30%), and Downstream Reachability (20%).",
        style_body
    ))
    story.append(Spacer(1, 4))

    story.append(Paragraph("Tab 6: Biological Pathway Enrichment", style_h2))
    story.append(Paragraph(
        "Displays overrepresentation analysis across Gene Ontology (GO Biological Process 2023) and KEGG Pathways. "
        "Interactive sliders allow filtering by adjusted $p$-value ($-\\log_{10} p_{adj}$). Top enriched biological processes include "
        "<b>Innate Immune Defense</b> ($p = 3.34 \\times 10^{-4}$), <b>Positive Regulation of Apoptotic Signaling</b> ($p = 1.98 \\times 10^{-3}$), "
        "and <b>NF-$\\kappa$B Transcription Factor Activation</b> ($p = 9.13 \\times 10^{-3}$).",
        style_body
    ))
    story.append(Spacer(1, 4))

    story.append(Paragraph("Tab 7: Targeted Therapeutics & Drug Repurposing", style_h2))
    story.append(Paragraph(
        "Bridges causal discovery with clinical pharmacology. Cross-references the top 10 master regulators against ChEMBL and DGIdb, "
        "curating <b>31 actionable therapeutic candidates</b> (12 FDA-Approved, 19 Investigational).",
        style_body
    ))
    story.append(Paragraph("&bull; <b>Lead Therapeutic Candidates in AML:</b>", style_body_bold))
    story.append(Paragraph("  - <b>NAMPT:</b> <code>Daporinad (FK866)</code> & <code>Teglarinad (KPT-9274)</code> &mdash; Triggers acute NAD+ exhaustion, starving metabolically addicted leukemic blasts.", style_bullet))
    story.append(Paragraph("  - <b>NPM1:</b> <code>Actinomycin D</code> & <code>Selinexor (KPT-330)</code> &mdash; Restores p53 tumor suppression via selective cytoplasmic NPM1 degradation and XPO1 inhibition.", style_bullet))
    story.append(Paragraph("  - <b>S100A9/A8:</b> <code>Tasquinimod</code> & <code>Auranofin</code> &mdash; Blocks TLR4/RAGE alarmin signaling, clearing bone marrow niche chemoresistance.", style_bullet))
    story.append(Paragraph("  - <b>TPSAB1:</b> <code>Midostaurin</code> (FDA approved) &mdash; Multi-kinase inhibitor targeting blast degranulation and survival.", style_bullet))
    story.append(Spacer(1, 4))

    story.append(Paragraph("Tab 8: Preprocessing & QC Audit", style_h2))
    story.append(Paragraph(
        "Provides complete transparency into Scanpy single-cell quality control metrics: cell filtering ($>50$ genes), gene filtering ($>5$ cells), "
        "mitochondrial exclusion (<25% MT reads), library normalization ($10^4$ target), and Seurat highly variable gene dispersion selection (top 80 HVGs).",
        style_body
    ))

    # Clean Break to Page 4
    story.append(PageBreak())

    # =========================================================================
    # PAGE 4: DATA EXPORT, TROUBLESHOOTING & FAQ
    # =========================================================================
    story.append(Paragraph("3. Data Export & Downstream Bioinformatics Integration", style_h1))
    story.append(HRFlowable(width="100%", thickness=0.8, color=COLOR_SECONDARY, spaceBefore=0, spaceAfter=6))
    
    story.append(Paragraph(
        "All analytical outputs generated by the platform are exportable in standardized bioinformatics formats "
        "for immediate downstream analysis in R (Seurat, Bioconductor) or Python (Scanpy, NetworkX):",
        style_body
    ))

    export_table_data = [
        [
            Paragraph("<b>Artifact File</b>", style_table_hdr),
            Paragraph("<b>File Format</b>", style_table_hdr),
            Paragraph("<b>Primary Application & Downstream Utility</b>", style_table_hdr),
        ],
        [
            Paragraph("<code>top_master_regulators.csv</code>", style_table_cell_bold),
            Paragraph("CSV Table", style_table_cell),
            Paragraph("Ranked driver leaderboard; input for CRISPR/shRNA screens and survival analysis.", style_table_cell),
        ],
        [
            Paragraph("<code>stable_causal_edges.csv</code>", style_table_cell_bold),
            Paragraph("CSV Table", style_table_cell),
            Paragraph("Stability-selected causal links with bootstrap frequencies ($\\ge 60\\%$).", style_table_cell),
        ],
        [
            Paragraph("<code>causal_graph.graphml</code>", style_table_cell_bold),
            Paragraph("GraphML (XML)", style_table_cell),
            Paragraph("Cytoscape & Gephi compatible DAG file with node and edge attributes.", style_table_cell),
        ],
        [
            Paragraph("<code>in_silico_knockout_effects.csv</code>", style_table_cell_bold),
            Paragraph("CSV Table", style_table_cell),
            Paragraph("Pearl $do$-operator simulated expression shifts ($\\Delta \\%$, $z$-score, causal distance).", style_table_cell),
        ],
        [
            Paragraph("<code>subpopulation_causal_diff.csv</code>", style_table_cell_bold),
            Paragraph("CSV Table", style_table_cell),
            Paragraph("Hierarchy rewiring metrics ($\\Delta \\beta = \\beta_{imm} - \\beta_{diff}$) for LSC targeting.", style_table_cell),
        ],
        [
            Paragraph("<code>drug_target_interactions.csv</code>", style_table_cell_bold),
            Paragraph("CSV Table", style_table_cell),
            Paragraph("Actionable compounds curated from ChEMBL/DGIdb with clinical trial status.", style_table_cell),
        ],
        [
            Paragraph("<code>results/*.png</code>", style_table_cell_bold),
            Paragraph("300 DPI PNG", style_table_cell),
            Paragraph("Publication-grade vector-quality figures formatted for manuscripts and grants.", style_table_cell),
        ],
    ]
    t_exp = Table(export_table_data, colWidths=[printable_w * 0.35, printable_w * 0.18, printable_w * 0.47])
    t_exp.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), COLOR_PRIMARY),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ('LEFTPADDING', (0,0), (-1,-1), 5),
        ('RIGHTPADDING', (0,0), (-1,-1), 5),
        ('GRID', (0,0), (-1,-1), 0.5, COLOR_BORDER),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [COLOR_WHITE, COLOR_BG_CARD]),
    ]))
    story.append(t_exp)
    story.append(Spacer(1, 8))

    story.append(Paragraph("4. Troubleshooting & Frequently Asked Questions (FAQ)", style_h1))
    story.append(HRFlowable(width="100%", thickness=0.8, color=COLOR_SECONDARY, spaceBefore=0, spaceAfter=6))

    faq_items = [
        ("Q1: Port 8501 is already in use by another service.",
         "Streamlit defaults to port 8501. Launch with <code>streamlit run dashboard/app.py --server.port 8502</code> or terminate existing background processes using PowerShell: <code>Get-Process -Name streamlit | Stop-Process</code>."),
        ("Q2: Dashboard displays 'Awaiting Artifacts' or warnings in sidebar.",
         "Ensure your working directory is the repository root (<code>D:\\new project on 10102026</code>). Verify that <code>./results</code> contains the exported CSV files. You can enter an absolute path directly in the sidebar's 'Artifacts Directory' text input."),
        ("Q3: Force-directed network graph renders slowly or lags during dragging.",
         "Ensure hardware acceleration is enabled in your web browser settings (Chrome/Edge: 'Settings > System > Use graphics acceleration when available'). Additionally, increase the 'Minimum Causal Out-Degree' slider to filter out peripheral nodes."),
        ("Q4: Can I run this dashboard on a remote cloud server or Docker container?",
         "Yes. Launch with <code>streamlit run dashboard/app.py --server.headless true --server.address 0.0.0.0 --server.port 8501</code> to allow external reverse proxy or browser connections."),
    ]

    for q, a in faq_items:
        story.append(Paragraph(f"<b>{q}</b>", style_h2))
        story.append(Paragraph(a, style_body))
        story.append(Spacer(1, 2))

    story.append(Spacer(1, 6))
    story.append(callout_box(
        "TECHNICAL SUPPORT & CITATIONS",
        "If you encounter pipeline issues or require custom algorithmic extensions, consult the repository README at "
        "<code>README.md</code>. Please cite van Galen et al. (Cell 2019) and the Py-Why causal-learn consortium.",
        "tip"
    ))

    print(f"[*] Building document with {len(story)} flowables...", flush=True)
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"[+] Document successfully compiled to: {filename}", flush=True)


if __name__ == "__main__":
    out_pdf = Path("HELP_how_to_use_app.pdf")
    build_pdf(out_pdf)
