#!/usr/bin/env python3
"""
Generate High-Resolution Drug Repurposing Summary Plot
======================================================
Produces a 2-panel, publication-grade figure summarizing targeted therapeutics
and clinical translation status for the top 10 AML master regulators:
- Panel A (Left): Horizontal stacked bar chart of Approved vs. Investigational compounds per target.
- Panel B (Right): Donut chart illustrating the global clinical development phase breakdown.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def generate_drug_plot(
    csv_path: Path = Path("results/drug_target_interactions.csv"),
    output_path: Path = Path("results/drug_repurposing_summary.png"),
    dpi: int = 300
):
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing input CSV at: {csv_path}")

    df = pd.read_csv(csv_path)
    print(f"[*] Loaded {len(df)} drug interactions from {csv_path}")

    # Ensure consistent classification
    df["Category"] = df["Clinical_Phase"].apply(lambda p: "FDA-Approved" if p == "Approved" else "Investigational")

    # Order genes by Rank ascending (so Rank 1 is top of chart when plotted inverted)
    rank_order = df[["Gene", "Rank"]].drop_duplicates().sort_values("Rank")
    ordered_genes = rank_order["Gene"].tolist()

    # Aggregate counts per gene and category
    pivot_df = pd.crosstab(df["Gene"], df["Category"]).reindex(ordered_genes).fillna(0)
    for col in ["FDA-Approved", "Investigational"]:
        if col not in pivot_df.columns:
            pivot_df[col] = 0
    pivot_df = pivot_df[["FDA-Approved", "Investigational"]]
    pivot_df["Total"] = pivot_df["FDA-Approved"] + pivot_df["Investigational"]

    # Invert so Rank 1 appears at the very top of the horizontal bar chart
    plot_genes = ordered_genes[::-1]
    plot_df = pivot_df.loc[plot_genes]

    # Global Style Configuration
    sns.set_theme(style="whitegrid", font="sans-serif")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 11,
        "figure.titlesize": 15,
        "figure.autolayout": False
    })

    fig, axes = plt.subplots(1, 2, figsize=(16, 6.8), gridspec_kw={"width_ratios": [1.4, 1.0]}, dpi=dpi)

    # -------------------------------------------------------------------------
    # Panel A: Stacked Horizontal Bar Chart
    # -------------------------------------------------------------------------
    ax_bar = axes[0]
    y_pos = np.arange(len(plot_genes))
    bar_height = 0.62

    c_app = "#10b981"   # Emerald Green
    c_inv = "#2563eb"   # Royal Blue

    app_vals = plot_df["FDA-Approved"].values
    inv_vals = plot_df["Investigational"].values
    total_vals = plot_df["Total"].values

    # Plot bars
    bars_app = ax_bar.barh(y_pos, app_vals, height=bar_height, label="FDA-Approved Repurposing", color=c_app, edgecolor="#064e3b", linewidth=1.1, alpha=0.92)
    bars_inv = ax_bar.barh(y_pos, inv_vals, left=app_vals, height=bar_height, label="Investigational / Clinical Trial", color=c_inv, edgecolor="#1e3a8a", linewidth=1.1, alpha=0.92)

    # Bar labels
    gene_labels_with_rank = []
    for g in plot_genes:
        r = rank_order[rank_order["Gene"] == g]["Rank"].values[0]
        gene_labels_with_rank.append(f"#{r} {g}")

    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(gene_labels_with_rank, fontweight="bold", color="#1e293b")
    ax_bar.set_xlabel("Number of Mapped Therapeutic Candidates", fontweight="bold", labelpad=10)
    ax_bar.set_title("A. Therapeutic Probes per AML Master Regulator", fontweight="bold", pad=14, loc="left", color="#0f172a")

    # Add numeric count annotations at bar ends
    for i, (tot, app, inv) in enumerate(zip(total_vals, app_vals, inv_vals)):
        label_text = f"{int(tot)}"
        ax_bar.text(tot + 0.12, i, label_text, va="center", ha="left", fontsize=10, fontweight="bold", color="#334155")
        # Internal labels if wide enough
        if app >= 1:
            ax_bar.text(app / 2, i, f"{int(app)}", va="center", ha="center", fontsize=9.5, fontweight="bold", color="#ffffff")
        if inv >= 1:
            ax_bar.text(app + (inv / 2), i, f"{int(inv)}", va="center", ha="center", fontsize=9.5, fontweight="bold", color="#ffffff")

    ax_bar.set_xlim(0, max(total_vals) + 1.2)
    ax_bar.grid(axis="x", linestyle="--", alpha=0.55)
    ax_bar.grid(axis="y", visible=False)
    ax_bar.legend(loc="lower right", frameon=True, framealpha=0.95, edgecolor="#cbd5e1", fontsize=10)

    # -------------------------------------------------------------------------
    # Panel B: Donut Chart (Clinical Phase Breakdown)
    # -------------------------------------------------------------------------
    ax_pie = axes[1]
    phase_order = ["Approved", "Phase 3", "Phase 2", "Phase 1", "Preclinical"]
    phase_counts = df["Clinical_Phase"].value_counts().reindex(phase_order).dropna()
    phase_counts = phase_counts[phase_counts > 0]

    color_palette = {
        "Approved": "#10b981",    # Emerald
        "Phase 3": "#f59e0b",     # Amber
        "Phase 2": "#3b82f6",     # Blue
        "Phase 1": "#8b5cf6",     # Purple
        "Preclinical": "#94a3b8"  # Slate
    }
    pie_colors = [color_palette[p] for p in phase_counts.index]
    pie_labels = [f"{p}\n({int(cnt)})" for p, cnt in zip(phase_counts.index, phase_counts.values)]

    wedges, texts, autotexts = ax_pie.pie(
        phase_counts.values,
        labels=pie_labels,
        autopct="%1.1f%%",
        pctdistance=0.74,
        startangle=140,
        colors=pie_colors,
        wedgeprops=dict(width=0.46, edgecolor="white", linewidth=2.0),
        textprops=dict(fontsize=10, fontweight="medium", color="#1e293b")
    )

    for autotext in autotexts:
        autotext.set_fontsize(9.5)
        autotext.set_fontweight("bold")
        autotext.set_color("#ffffff")

    # Center label inside donut
    total_compounds = len(df)
    ax_pie.text(
        0, 0,
        f"{total_compounds}\nCompounds\n(10 Targets)",
        ha="center", va="center",
        fontsize=12, fontweight="bold", color="#0f172a",
        linespacing=1.2
    )

    ax_pie.set_title("B. Global Clinical Development Phase Breakdown", fontweight="bold", pad=14, loc="left", color="#0f172a")

    # Overall figure title & layout
    fig.suptitle(
        "AML Master Regulator Drug Repurposing & Translational Probe Landscape (GSE116256)",
        fontsize=15, fontweight="heavy", y=0.98, color="#0f172a"
    )

    plt.tight_layout(rect=[0, 0.03, 1, 0.94])

    # Save to primary results directory and subfolder
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")

    mirror_path = output_path.parent / "figures" / output_path.name
    mirror_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(mirror_path, dpi=dpi, bbox_inches="tight")

    plt.close()
    print(f"[+] Successfully exported high-resolution figure to: {output_path}")
    print(f"[+] Mirrored copy saved to: {mirror_path}")

if __name__ == "__main__":
    generate_drug_plot()
