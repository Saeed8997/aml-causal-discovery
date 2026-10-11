#!/usr/bin/env python3
"""
Subpopulation Causal Rewiring & Leukemic Hierarchy Analysis
===========================================================
Investigates phenotypic state-specific causal rewiring across AML blast hierarchy:
1. Ingests van Galen GSE116256 single-cell expression and cell annotations.
2. Stratifies cells into two primary phenotypic compartments:
   - Immature / Stem-like AML Blasts (LSC-like / HSC-like / Progenitor-like / GMP-like)
   - Differentiated AML Blasts (Mono-like / Gran-like / cDC-like / Differentiated)
3. Evaluates learned causal DAG edges across both subpopulations via OLS path modeling
   and bootstrap stability selection (100 resamplings).
4. Computes edge retention rates, path coefficients (beta weights), and state-specific
   differential delta: Delta_beta = beta_immature - beta_differentiated.
5. Classifies edges into:
   - Stemness-Maintained Axes (therapeutic vulnerabilities in leukemic stem cells)
   - Maturation-Induced Axes (differentiation-driven transcriptional programs)
   - Conserved Core Regulatory Circuits
6. Exports:
   - 'results/subpopulation_causal_diff.csv'
   - 'results/subpopulation_causal_wiring.png' (300 DPI publication figure)
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import networkx as nx
import statsmodels.api as sm
from sklearn.utils import resample
import matplotlib.pyplot as plt
import seaborn as sns

sys.stdout.reconfigure(line_buffering=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Single-Cell Subpopulation Causal Rewiring Analysis in AML."
    )
    parser.add_argument(
        "--dem-path",
        type=Path,
        default=Path("results/GSM3587923_AML1012-D0.dem.txt.gz"),
        help="Path to digital expression matrix (.dem.txt.gz)",
    )
    parser.add_argument(
        "--anno-path",
        type=Path,
        default=Path("results/GSM3587924_AML1012-D0.anno.txt.gz"),
        help="Path to cell annotations (.anno.txt.gz)",
    )
    parser.add_argument(
        "--edges-path",
        type=Path,
        default=Path("results/stable_causal_edges.csv"),
        help="Path to stability-selected causal edges CSV",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("results/subpopulation_causal_diff.csv"),
        help="Output path for differential causal CSV",
    )
    parser.add_argument(
        "--out-png",
        type=Path,
        default=Path("results/subpopulation_causal_wiring.png"),
        help="Output path for 300 DPI summary figure",
    )
    parser.add_argument(
        "--n-bootstraps",
        type=int,
        default=100,
        help="Number of bootstrap resamplings per subpopulation (default: 100)",
    )
    return parser.parse_args()


def load_and_preprocess_data(dem_path: Path, anno_path: Path):
    """Loads expression matrix and cell annotations, performing library normalization."""
    if not dem_path.exists():
        raise FileNotFoundError(f"Expression matrix not found at: {dem_path}")
    if not anno_path.exists():
        raise FileNotFoundError(f"Annotation file not found at: {anno_path}")

    print(f"[*] Loading cell annotations from: {anno_path}...", flush=True)
    df_anno = pd.read_csv(anno_path, sep="\t", compression="gzip")
    print(f"[+] Loaded annotations for {len(df_anno)} single cells.", flush=True)

    print(f"[*] Loading raw digital expression matrix from: {dem_path}...", flush=True)
    df_dem = pd.read_csv(dem_path, sep="\t", compression="gzip", index_col=0)
    
    # Rows = cells, Columns = genes
    expr = df_dem.T
    print(f"[+] Raw expression matrix: {expr.shape[0]} cells x {expr.shape[1]} genes.", flush=True)

    # Library size normalization (10,000 counts) and log1p transform
    print("[*] Performing library size normalization (1e4 target) and log1p transform...", flush=True)
    library_size = expr.sum(axis=1)
    expr_norm = expr.div(library_size, axis=0) * 1e4
    expr_log = np.log1p(expr_norm)

    return expr_log, df_anno


def stratify_cell_subpopulations(df_anno: pd.DataFrame):
    """
    Stratifies cells into Immature / Stem-like blasts vs. Differentiated blasts
    based on van Galen GSE116256 cell type definitions.
    """
    print("\n[*] Stratifying cells into phenotypic hierarchy compartments...", flush=True)
    
    # Immature / Stem-like: HSC-like, Prog-like, GMP-like, HSC, Prog, GMP
    immature_pattern = "HSC|Prog|GMP"
    immature_mask = df_anno["CellType"].str.contains(immature_pattern, case=False, na=False)
    immature_barcodes = df_anno.loc[immature_mask, "Cell"].tolist()

    # Differentiated: Mono-like, ProMono-like, cDC-like, Mono, ProMono, cDC
    diff_pattern = "Mono|cDC|Gran"
    diff_mask = df_anno["CellType"].str.contains(diff_pattern, case=False, na=False)
    diff_barcodes = df_anno.loc[diff_mask, "Cell"].tolist()

    print(f"[+] Compartment 1 (Immature / Stem-like Blasts): {len(immature_barcodes)} cells", flush=True)
    print(f"    - Subtypes: {dict(df_anno.loc[immature_mask, 'CellType'].value_counts())}", flush=True)
    print(f"[+] Compartment 2 (Differentiated Blasts): {len(diff_barcodes)} cells", flush=True)
    print(f"    - Subtypes: {dict(df_anno.loc[diff_mask, 'CellType'].value_counts())}", flush=True)

    return immature_barcodes, diff_barcodes


def compute_differential_causal_wiring(
    expr_log: pd.DataFrame,
    edges_df: pd.DataFrame,
    immature_barcodes: list,
    diff_barcodes: list,
    n_bootstraps: int = 100,
):
    """
    Computes subpopulation-specific path coefficients (betas) and bootstrap stability
    for all stability-selected causal edges.
    """
    print(f"\n[*] Evaluating causal edges across subpopulations (Bootstraps={n_bootstraps})...", flush=True)
    
    imm_idx = expr_log.index.intersection(immature_barcodes)
    diff_idx = expr_log.index.intersection(diff_barcodes)

    # Build DiGraph from edges
    G = nx.DiGraph()
    edge_order = []
    for _, row in edges_df.iterrows():
        src = str(row["Source"])
        tgt = str(row["Target"])
        G.add_edge(src, tgt)
        edge_order.append((src, tgt))

    # 1. Full-sample OLS regressions
    full_sample_metrics = {}
    for target in G.nodes():
        parents = [p for p in G.predecessors(target) if p in expr_log.columns]
        if not parents or target not in expr_log.columns:
            continue
        
        # Fit Immature
        y_imm = expr_log.loc[imm_idx, target]
        X_imm = expr_log.loc[imm_idx, parents]
        X_imm_c = sm.add_constant(X_imm, has_constant="add")
        model_imm = sm.OLS(y_imm, X_imm_c).fit()

        # Fit Differentiated
        y_diff = expr_log.loc[diff_idx, target]
        X_diff = expr_log.loc[diff_idx, parents]
        X_diff_c = sm.add_constant(X_diff, has_constant="add")
        model_diff = sm.OLS(y_diff, X_diff_c).fit()

        for p in parents:
            b_imm = float(model_imm.params.get(p, 0.0))
            se_imm = float(model_imm.bse.get(p, 0.0))
            pval_imm = float(model_imm.pvalues.get(p, 1.0))

            b_diff = float(model_diff.params.get(p, 0.0))
            se_diff = float(model_diff.bse.get(p, 0.0))
            pval_diff = float(model_diff.pvalues.get(p, 1.0))

            full_sample_metrics[(p, target)] = {
                "beta_immature": b_imm,
                "se_immature": se_imm,
                "pval_immature": pval_imm,
                "beta_diff": b_diff,
                "se_diff": se_diff,
                "pval_diff": pval_diff,
                "r2_immature": float(model_imm.rsquared),
                "r2_diff": float(model_diff.rsquared),
            }

    # 2. Multi-Core / Fast Bootstrap Stability Resampling
    print(f"[*] Running {n_bootstraps} bootstrap iterations to determine edge retention rates...", flush=True)
    boot_counts_imm = {e: 0 for e in edge_order}
    boot_counts_diff = {e: 0 for e in edge_order}
    boot_betas_imm = {e: [] for e in edge_order}
    boot_betas_diff = {e: [] for e in edge_order}

    np.random.seed(42)
    sample_size_imm = int(0.85 * len(imm_idx))
    sample_size_diff = int(0.85 * len(diff_idx))

    for b in range(n_bootstraps):
        if (b + 1) % 25 == 0 or b == 0:
            print(f"  -> Bootstrap progress: {b + 1}/{n_bootstraps} ({((b+1)/n_bootstraps)*100:.0f}%)", flush=True)
            
        b_imm_idx = np.random.choice(imm_idx, size=sample_size_imm, replace=True)
        b_diff_idx = np.random.choice(diff_idx, size=sample_size_diff, replace=True)

        for target in G.nodes():
            parents = [p for p in G.predecessors(target) if p in expr_log.columns]
            if not parents or target not in expr_log.columns:
                continue

            # Subsample Immature
            sub_y_imm = expr_log.loc[b_imm_idx, target]
            sub_X_imm = expr_log.loc[b_imm_idx, parents]
            m_imm = sm.OLS(sub_y_imm, sm.add_constant(sub_X_imm, has_constant="add")).fit()

            # Subsample Differentiated
            sub_y_diff = expr_log.loc[b_diff_idx, target]
            sub_X_diff = expr_log.loc[b_diff_idx, parents]
            m_diff = sm.OLS(sub_y_diff, sm.add_constant(sub_X_diff, has_constant="add")).fit()

            for p in parents:
                edge = (p, target)
                if edge not in boot_counts_imm:
                    continue

                p_val_i = m_imm.pvalues.get(p, 1.0)
                param_i = m_imm.params.get(p, 0.0)
                boot_betas_imm[edge].append(param_i)
                if p_val_i < 0.05 and abs(param_i) > 0.05:
                    boot_counts_imm[edge] += 1

                p_val_d = m_diff.pvalues.get(p, 1.0)
                param_d = m_diff.params.get(p, 0.0)
                boot_betas_diff[edge].append(param_d)
                if p_val_d < 0.05 and abs(param_d) > 0.05:
                    boot_counts_diff[edge] += 1

    # Consolidate results table
    rows = []
    for src, tgt in edge_order:
        edge = (src, tgt)
        if edge not in full_sample_metrics:
            continue

        fs = full_sample_metrics[edge]
        ret_imm = boot_counts_imm[edge] / n_bootstraps
        ret_diff = boot_counts_diff[edge] / n_bootstraps
        delta_ret = ret_imm - ret_diff
        delta_b = fs["beta_immature"] - fs["beta_diff"]

        # Classification logic:
        # Stemness-Maintained: Predominant in immature (Delta_b >= 0.15 or (Delta_b > 0 and delta_ret >= 0.20 and beta_imm > 0.10))
        # Maturation-Induced: Predominant in differentiated (Delta_b <= -0.15 or (Delta_b < 0 and delta_ret <= -0.20 and beta_diff > 0.10))
        # Conserved Core: Active in both compartments (both retentions >= 0.30 or consistent positive beta)
        # Context-Dependent: Weak or transitional
        if delta_b >= 0.15 or (delta_b > 0 and delta_ret >= 0.20 and fs["beta_immature"] > 0.10):
            category = "Stemness-Maintained"
        elif delta_b <= -0.15 or (delta_b < 0 and delta_ret <= -0.20 and fs["beta_diff"] > 0.10):
            category = "Maturation-Induced"
        elif (ret_imm >= 0.30 and ret_diff >= 0.30) or (abs(delta_b) < 0.15 and fs["beta_immature"] > 0.05 and fs["beta_diff"] > 0.05):
            category = "Conserved Core"
        else:
            category = "Context-Dependent"

        rows.append({
            "Source": src,
            "Target": tgt,
            "Edge": f"{src} -> {tgt}",
            "Beta_Immature": round(fs["beta_immature"], 4),
            "Beta_Differentiated": round(fs["beta_diff"], 4),
            "Delta_Beta": round(delta_b, 4),
            "SE_Immature": round(fs["se_immature"], 4),
            "SE_Differentiated": round(fs["se_diff"], 4),
            "Pval_Immature": float(f"{fs['pval_immature']:.4e}"),
            "Pval_Differentiated": float(f"{fs['pval_diff']:.4e}"),
            "Retention_Immature": round(ret_imm, 3),
            "Retention_Differentiated": round(ret_diff, 3),
            "Delta_Retention": round(delta_ret, 3),
            "Wiring_Category": category,
        })

    diff_df = pd.DataFrame(rows)
    return diff_df


def generate_publication_figure(diff_df: pd.DataFrame, out_png: Path):
    """
    Generates a 300 DPI publication figure with 2 panels:
    - Panel A: Comparative edge strength scatter/bubble plot (Immature vs. Differentiated).
    - Panel B: Top differential regulatory axes horizontal bar plot.
    """
    print(f"\n[*] Rendering publication figure to: {out_png}...", flush=True)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    # Style configuration
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
    plt.rcParams["axes.edgecolor"] = "#2C3E50"
    plt.rcParams["axes.linewidth"] = 0.9

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8), dpi=300)

    category_palette = {
        "Stemness-Maintained": "#D9381E",  # Warm Crimson / Coral
        "Maturation-Induced": "#0D5C75",  # Deep Cyan / Teal
        "Conserved Core": "#2E7D32",      # Forest Green
        "Context-Dependent": "#8E8E93",   # Muted Slate Gray
    }

    # -------------------------------------------------------------
    # Panel A: Comparative Edge Strength Scatter Plot
    # -------------------------------------------------------------
    for cat, color in category_palette.items():
        subset = diff_df[diff_df["Wiring_Category"] == cat]
        sizes = np.clip(np.abs(subset["Delta_Beta"]) * 140 + 40, 30, 220)
        ax1.scatter(
            subset["Beta_Immature"],
            subset["Beta_Differentiated"],
            c=color,
            s=sizes,
            alpha=0.82,
            edgecolors="black",
            linewidths=0.6,
            label=f"{cat} (n={len(subset)})",
            zorder=4,
        )

    # Reference diagonal y = x
    min_val = min(diff_df["Beta_Immature"].min(), diff_df["Beta_Differentiated"].min()) - 0.15
    max_val = max(diff_df["Beta_Immature"].max(), diff_df["Beta_Differentiated"].max()) + 0.15
    ax1.plot([min_val, max_val], [min_val, max_val], "--", color="#7F8C8D", linewidth=1.2, zorder=2, label="Conserved Equivalence Line (y = x)")
    ax1.axhline(0, color="#BDC3C7", linestyle=":", linewidth=0.9, zorder=1)
    ax1.axvline(0, color="#BDC3C7", linestyle=":", linewidth=0.9, zorder=1)

    # Annotate top extreme edges
    top_stemness = diff_df.nlargest(6, "Delta_Beta")
    top_maturation = diff_df.nsmallest(6, "Delta_Beta")
    annot_candidates = pd.concat([top_stemness, top_maturation]).drop_duplicates(subset=["Edge"])

    for _, row in annot_candidates.iterrows():
        ax1.annotate(
            row["Edge"],
            xy=(row["Beta_Immature"], row["Beta_Differentiated"]),
            xytext=(6, 5),
            textcoords="offset points",
            fontsize=8.5,
            fontweight="bold",
            color="#1A252C",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#BDC3C7", alpha=0.85, lw=0.5),
            zorder=6,
        )

    ax1.set_title(
        "A: Differential Causal Edge Path Strength\n(Immature LSC-like vs. Differentiated Blasts)",
        fontsize=13,
        fontweight="bold",
        pad=12,
        color="#1A252C",
    )
    ax1.set_xlabel("Immature / Stem-like Blast Path Coefficient (Beta)", fontsize=11, fontweight="bold", labelpad=8)
    ax1.set_ylabel("Differentiated Blast Path Coefficient (Beta)", fontsize=11, fontweight="bold", labelpad=8)
    ax1.set_xlim(min_val, max_val)
    ax1.set_ylim(min_val, max_val)
    ax1.grid(True, linestyle="--", alpha=0.35, zorder=0)
    ax1.legend(loc="lower right", frameon=True, facecolor="#F8F9FA", edgecolor="#BDC3C7", fontsize=9.5)

    # -------------------------------------------------------------
    # Panel B: Top Differential Regulatory Axes Bar Plot
    # -------------------------------------------------------------
    # Select top 8 stemness-maintained (Delta_Beta > 0) and top 8 maturation-induced (Delta_Beta < 0)
    top_stem = diff_df.nlargest(8, "Delta_Beta").sort_values("Delta_Beta", ascending=True)
    top_diff = diff_df.nsmallest(8, "Delta_Beta").sort_values("Delta_Beta", ascending=True)
    plot_df = pd.concat([top_diff, top_stem])

    bar_colors = [
        category_palette["Stemness-Maintained"] if val > 0 else category_palette["Maturation-Induced"]
        for val in plot_df["Delta_Beta"]
    ]

    y_pos = np.arange(len(plot_df))
    bars = ax2.barh(y_pos, plot_df["Delta_Beta"], color=bar_colors, edgecolor="black", linewidth=0.7, height=0.68, zorder=3)
    ax2.axvline(0, color="#2C3E50", linewidth=1.1, zorder=4)

    # Value labels
    for bar, val in zip(bars, plot_df["Delta_Beta"]):
        offset = 0.03 if val >= 0 else -0.03
        ha = "left" if val >= 0 else "right"
        ax2.text(
            val + offset,
            bar.get_y() + bar.get_height() / 2,
            f"{val:+.2f}",
            va="center",
            ha=ha,
            fontsize=8.5,
            fontweight="bold",
            color="#1A252C",
        )

    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(plot_df["Edge"], fontsize=9.5, fontweight="bold")
    ax2.set_xlabel("Differential Path Strength Delta (Beta_Immature - Beta_Diff)", fontsize=11, fontweight="bold", labelpad=8)
    ax2.set_title(
        "B: Top Phenotypic Rewiring Regulatory Axes\n(Stemness Vulnerabilities vs. Maturation Programs)",
        fontsize=13,
        fontweight="bold",
        pad=12,
        color="#1A252C",
    )
    ax2.grid(True, axis="x", linestyle="--", alpha=0.35, zorder=0)

    # Custom annotations on panel B
    ax2.text(
        0.05,
        0.94,
        "Red: Stemness-Maintained (LSC Vulnerability)",
        transform=ax2.transAxes,
        fontsize=9,
        fontweight="bold",
        color="#D9381E",
        bbox=dict(boxstyle="square,pad=0.25", fc="#FBEAE7", ec="#D9381E", lw=0.8),
    )
    ax2.text(
        0.05,
        0.86,
        "Blue: Maturation-Induced Program",
        transform=ax2.transAxes,
        fontsize=9,
        fontweight="bold",
        color="#0D5C75",
        bbox=dict(boxstyle="square,pad=0.25", fc="#E6F2F5", ec="#0D5C75", lw=0.8),
    )

    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()

    # Mirror figure into results/figures/
    mirror_png = Path("results/figures") / out_png.name
    mirror_png.parent.mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copyfile(out_png, mirror_png)
    print(f"[+] Publication figure saved to {out_png} and mirrored to {mirror_png}", flush=True)


def main():
    args = parse_args()

    print("====================================================================", flush=True)
    print("STEP 4: SUBPOPULATION CAUSAL REWIRING & LEUKEMIC HIERARCHY ANALYSIS", flush=True)
    print("====================================================================", flush=True)

    # 1. Load Data
    expr_log, df_anno = load_and_preprocess_data(args.dem_path, args.anno_path)

    # 2. Stratify Cells
    immature_barcodes, diff_barcodes = stratify_cell_subpopulations(df_anno)

    # 3. Load Causal Edges
    if not args.edges_path.exists():
        raise FileNotFoundError(f"Stable causal edges file not found at: {args.edges_path}")
    edges_df = pd.read_csv(args.edges_path)
    print(f"[+] Loaded {len(edges_df)} stability-selected causal edges from: {args.edges_path}", flush=True)

    # 4. Compute Differential Rewiring
    diff_df = compute_differential_causal_wiring(
        expr_log=expr_log,
        edges_df=edges_df,
        immature_barcodes=immature_barcodes,
        diff_barcodes=diff_barcodes,
        n_bootstraps=args.n_bootstraps,
    )

    # 5. Save Table
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    diff_df.to_csv(args.out_csv, index=False)
    print(f"[+] Differential metrics exported to: {args.out_csv} ({len(diff_df)} edges)", flush=True)

    # Mirror table to results/tables/
    mirror_csv = Path("results/tables") / args.out_csv.name
    mirror_csv.parent.mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copyfile(args.out_csv, mirror_csv)
    print(f"[+] Mirrored table copy saved to: {mirror_csv}", flush=True)

    # 6. Generate Publication Figure
    generate_publication_figure(diff_df, args.out_png)

    # 7. Summary Leaderboards
    print("\n-------------------------------------------------------------", flush=True)
    print("LEADERBOARD: TOP STEMNESS-MAINTAINED REGULATORY AXES (LSC-SPECIFIC)", flush=True)
    print("-------------------------------------------------------------", flush=True)
    top_stem = diff_df.sort_values(by="Delta_Beta", ascending=False).head(8)
    for idx, (_, r) in enumerate(top_stem.iterrows(), 1):
        print(
            f"  {idx}. {r['Edge']:<18} | Beta_Imm: {r['Beta_Immature']:+0.3f} | Beta_Diff: {r['Beta_Differentiated']:+0.3f} | Delta_Beta: {r['Delta_Beta']:+0.3f} | Ret_Imm: {r['Retention_Immature']:.1%}",
            flush=True,
        )

    print("\n-------------------------------------------------------------", flush=True)
    print("LEADERBOARD: TOP MATURATION-INDUCED REGULATORY AXES", flush=True)
    print("-------------------------------------------------------------", flush=True)
    top_mat = diff_df.sort_values(by="Delta_Beta", ascending=True).head(8)
    for idx, (_, r) in enumerate(top_mat.iterrows(), 1):
        print(
            f"  {idx}. {r['Edge']:<18} | Beta_Imm: {r['Beta_Immature']:+0.3f} | Beta_Diff: {r['Beta_Differentiated']:+0.3f} | Delta_Beta: {r['Delta_Beta']:+0.3f} | Ret_Diff: {r['Retention_Differentiated']:.1%}",
            flush=True,
        )

    print("\n====================================================================", flush=True)
    print("[SUCCESS] Step 4 Phase 1 Complete!", flush=True)
    print("====================================================================", flush=True)


if __name__ == "__main__":
    main()
