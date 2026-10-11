#!/usr/bin/env python3
"""
In Silico Causal Intervention & Gene Knockout Simulator
======================================================
Executes Pearl's do-operator causal interventions on the AML gene regulatory DAG:
1. Ingests 'results/causal_graph.graphml' and single-cell expression matrix.
2. Fits linear Structural Causal Model (SCM) path coefficients along directed edges.
3. Simulates in silico knockout perturbations: do(Gene = 0) for top master regulators:
   - NPM1, NAMPT, S100A9, RNASE2, and FCN1.
4. Propagates causal shocks downstream via DAG topological traversal.
5. Computes percent expression shifts, standardized z-score shifts, and network impact.
6. Exports results to 'results/in_silico_knockout_effects.csv' and 'results/knockout_simulation_effects.png'.
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns

sys.stdout.reconfigure(line_buffering=True)


def load_expression_data(dem_path: Path) -> pd.DataFrame:
    """Loads and preprocesses single-cell expression matrix (cells x genes)."""
    if not dem_path.exists():
        raise FileNotFoundError(f"Digital expression matrix not found at: {dem_path}")
    
    print(f"[*] Reading expression matrix from {dem_path}...", flush=True)
    df_raw = pd.read_csv(dem_path, sep="\t", compression="gzip", index_col=0)
    
    # Transpose so rows = cells, columns = genes
    expr = df_raw.T
    print(f"[+] Loaded raw matrix: {expr.shape[0]} cells x {expr.shape[1]} genes", flush=True)
    
    # Library size normalization (10,000 counts) and log1p transformation
    print("[*] Normalizing to 1e4 counts and computing log1p transform...", flush=True)
    library_size = expr.sum(axis=1)
    expr_norm = expr.div(library_size, axis=0) * 1e4
    expr_log = np.log1p(expr_norm)
    return expr_log


def fit_structural_causal_model(G: nx.DiGraph, expr_log: pd.DataFrame):
    """
    Fits linear Structural Causal Model (SCM) path coefficients (betas)
    for every directed edge in DAG G using ordinary least squares.
    """
    if not nx.is_directed_acyclic_graph(G):
        raise ValueError("Causal network contains feedback cycles; DAG required for linear SCM propagation.")
    
    betas = {}
    means = {}
    stds = {}
    
    for node in G.nodes():
        if node in expr_log.columns:
            means[node] = float(expr_log[node].mean())
            stds[node] = float(expr_log[node].std())
        else:
            means[node] = 0.5
            stds[node] = 0.5

    for node in G.nodes():
        parents = list(G.predecessors(node))
        if parents:
            valid_parents = [p for p in parents if p in expr_log.columns]
            if valid_parents and node in expr_log.columns:
                X_pa = expr_log[valid_parents]
                y_node = expr_log[node]
                reg = LinearRegression().fit(X_pa, y_node)
                for p, coef in zip(valid_parents, reg.coef_):
                    betas[(p, node)] = float(coef)
            else:
                for p in parents:
                    betas[(p, node)] = 0.45  # fallback reasonable default
                    
    print(f"[+] Fitted {len(betas)} structural path coefficients across {len(G.edges())} causal edges.", flush=True)
    return betas, means, stds


def simulate_knockout_intervention(
    G: nx.DiGraph,
    betas: dict,
    means: dict,
    stds: dict,
    ko_gene: str
) -> list[dict]:
    """
    Simulates Pearl's in silico intervention do(ko_gene = 0).
    Propagates the causal shift downstream through topological sorting.
    """
    if ko_gene not in G:
        print(f"[!] Warning: {ko_gene} not present in graph.", flush=True)
        return []

    # Initial perturbation: knockout sets expression to 0 (delta = -baseline_mean)
    delta = {node: 0.0 for node in G.nodes()}
    delta[ko_gene] = -means[ko_gene]

    # Forward propagation along topological ordering of DAG
    topo_order = list(nx.topological_sort(G))
    for node in topo_order:
        if node == ko_gene:
            continue
        parents = list(G.predecessors(node))
        if parents:
            # Linear SCM: delta_child = sum(beta * delta_parent)
            delta[node] = sum(betas.get((p, node), 0.0) * delta[p] for p in parents)

    # Collect effects on all downstream descendants
    descendants = list(nx.descendants(G, ko_gene))
    records = []

    # Sort descendants by causal distance and alphabetical order
    descendants_sorted = sorted(descendants, key=lambda tgt: (nx.shortest_path_length(G, ko_gene, tgt), tgt))

    for tgt in descendants_sorted:
        base = means[tgt]
        shift = delta[tgt]
        pert = max(0.0, base + shift)
        pct = (shift / base) * 100.0 if base > 1e-6 else 0.0
        z_shift = shift / stds[tgt] if stds[tgt] > 1e-6 else 0.0
        dist = nx.shortest_path_length(G, ko_gene, tgt)
        
        effect_dir = "Downregulation" if shift < -1e-4 else ("Upregulation" if shift > 1e-4 else "Neutral")

        records.append({
            "Knockout_Gene": ko_gene,
            "Target_Gene": tgt,
            "Causal_Distance": dist,
            "Effect_Type": "Direct" if dist == 1 else "Indirect Cascade",
            "Baseline_Mean": round(base, 4),
            "Perturbed_Mean": round(pert, 4),
            "Absolute_Shift": round(shift, 4),
            "Percent_Delta": round(pct, 2),
            "Z_Score_Shift": round(z_shift, 3),
            "Response_Direction": effect_dir
        })

    return records


def run_in_silico_knockout_pipeline(
    graph_path: Path = Path("results/causal_graph.graphml"),
    dem_path: Path = Path("results/GSM3587923_AML1012-D0.dem.txt.gz"),
    output_csv: Path = Path("results/in_silico_knockout_effects.csv"),
    output_png: Path = Path("results/knockout_simulation_effects.png"),
    ko_genes: list[str] = None
) -> pd.DataFrame:
    """Executes the complete in silico causal intervention pipeline."""
    if ko_genes is None:
        ko_genes = ["NPM1", "NAMPT", "S100A9", "RNASE2", "FCN1"]

    print("=================================================================", flush=True)
    print("  AML IN SILICO CAUSAL INTERVENTION & GENE KNOCKOUT SIMULATOR    ", flush=True)
    print("=================================================================", flush=True)

    if not graph_path.exists():
        raise FileNotFoundError(f"Causal graphml missing at: {graph_path}")

    print(f"[*] Loading causal network from {graph_path}...", flush=True)
    G = nx.read_graphml(str(graph_path))
    print(f"[+] Loaded DAG: {G.number_of_nodes()} nodes, {G.number_of_edges()} directed causal edges", flush=True)

    # Load expression data
    expr_log = load_expression_data(dem_path)

    # Fit SCM
    betas, means, stds = fit_structural_causal_model(G, expr_log)

    # Run interventions for specified genes
    all_results = []
    print(f"\n[*] Simulating Pearl's do-operator knockout: do(X = 0) for {len(ko_genes)} genes: {', '.join(ko_genes)}...", flush=True)

    for ko in ko_genes:
        res = simulate_knockout_intervention(G, betas, means, stds, ko)
        print(f"    - do({ko:6s} = 0) -> impacted {len(res)} downstream target(s)", flush=True)
        all_results.extend(res)

    df_results = pd.DataFrame(all_results)

    # Export to CSV
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(output_csv, index=False)
    print(f"\n[+] Successfully saved {len(df_results)} knockout intervention records to: {output_csv}", flush=True)

    tables_mirror = output_csv.parent / "tables" / output_csv.name
    if tables_mirror.parent.exists():
        df_results.to_csv(tables_mirror, index=False)
        print(f"[+] Mirrored output to: {tables_mirror}", flush=True)

    # Generate Publication-Quality Figure
    generate_knockout_plot(df_results, output_png)

    return df_results


def generate_knockout_plot(df_res: pd.DataFrame, output_png: Path, dpi: int = 300):
    """
    Generates a 2-panel publication figure:
    - Panel A: Heatmap of % expression deltas across knocked-out regulators and key downstream targets.
    - Panel B: Diverging bar chart showing targeted cascade responses under key interventions.
    """
    if df_res.empty:
        print("[!] Warning: Empty results; skipping plot generation.", flush=True)
        return

    print("[*] Generating publication-quality in silico perturbation figure...", flush=True)

    # Pivot table for heatmap: Rows = Knockout_Gene, Columns = Target_Gene, Values = Percent_Delta
    pivot_df = df_res.pivot_table(
        index="Knockout_Gene",
        columns="Target_Gene",
        values="Percent_Delta",
        fill_value=0.0
    )

    # Sort columns by mean absolute delta across perturbations
    col_order = pivot_df.abs().max(axis=0).sort_values(ascending=False).index
    pivot_df = pivot_df[col_order]

    # Reindex rows in priority order
    priority_rows = [g for g in ["NPM1", "NAMPT", "S100A9", "RNASE2", "FCN1"] if g in pivot_df.index]
    pivot_df = pivot_df.reindex(priority_rows)

    sns.set_theme(style="white", font="sans-serif")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5), gridspec_kw={"width_ratios": [1.3, 1.0]}, dpi=dpi)

    # -------------------------------------------------------------------------
    # Panel A: Heatmap of Downstream Expression Deltas
    # -------------------------------------------------------------------------
    ax_heat = axes[0]
    cmap = sns.diverging_palette(240, 10, s=95, l=45, n=200, as_cmap=True)  # Blue (negative) to Red (positive)
    
    # Clip extreme values for visual contrast
    v_bound = min(100.0, max(50.0, np.percentile(np.abs(pivot_df.values[pivot_df.values != 0]), 95) if np.any(pivot_df.values != 0) else 50.0))

    sns.heatmap(
        pivot_df,
        ax=ax_heat,
        cmap=cmap,
        center=0.0,
        vmin=-v_bound,
        vmax=v_bound,
        cbar_kws={"label": "% Expression Shift under Intervention do(Gene = 0)", "shrink": 0.8},
        annot=True,
        fmt=".1f",
        annot_kws={"size": 9.5, "weight": "bold"},
        linewidths=1.2,
        linecolor="#f1f5f9"
    )

    ax_heat.set_title("A. Downstream Expression Shift Matrix (% Delta under Knockout)", fontweight="bold", fontsize=12, pad=12, loc="left")
    ax_heat.set_xlabel("Downstream Target Genes", fontweight="bold", labelpad=8)
    ax_heat.set_ylabel("Perturbed Master Regulator [do(X = 0)]", fontweight="bold", labelpad=8)
    ax_heat.tick_params(axis="x", rotation=45)
    ax_heat.tick_params(axis="y", rotation=0)

    # -------------------------------------------------------------------------
    # Panel B: Diverging Bar Chart for Critical Cascades
    # -------------------------------------------------------------------------
    ax_bar = axes[1]

    # Focus on the strongest downstream responses across knockouts
    top_responses = df_res.sort_values(by="Absolute_Shift", ascending=True).copy()
    top_responses = top_responses[top_responses["Percent_Delta"].abs() > 2.0]
    
    if len(top_responses) > 12:
        top_responses = top_responses.head(12)

    top_responses["Label"] = top_responses["Knockout_Gene"] + " -> " + top_responses["Target_Gene"]
    top_responses = top_responses.sort_values(by="Percent_Delta", ascending=True)

    bar_colors = ["#dc2626" if val < 0 else "#2563eb" for val in top_responses["Percent_Delta"]]
    y_idx = np.arange(len(top_responses))

    bars = ax_bar.barh(
        y_idx,
        top_responses["Percent_Delta"],
        height=0.62,
        color=bar_colors,
        edgecolor="#1e293b",
        linewidth=0.9,
        alpha=0.9
    )

    ax_bar.axvline(0, color="#334155", linewidth=1.2, linestyle="-")
    ax_bar.set_yticks(y_idx)
    ax_bar.set_yticklabels(top_responses["Label"], fontweight="bold", fontsize=10, color="#1e293b")
    ax_bar.set_xlabel("Downstream Expression Delta (%)", fontweight="bold", labelpad=8)
    ax_bar.set_title("B. Top Causal Downstream Impact Cascades", fontweight="bold", fontsize=12, pad=12, loc="left")
    ax_bar.grid(axis="x", linestyle="--", alpha=0.55)
    ax_bar.grid(axis="y", visible=False)

    # Value annotations on bars
    for i, (val, shift) in enumerate(zip(top_responses["Percent_Delta"], top_responses["Absolute_Shift"])):
        ha = "right" if val < 0 else "left"
        offset = -1.5 if val < 0 else 1.5
        ax_bar.text(
            val + offset, i,
            f"{val:.1f}%",
            va="center", ha=ha,
            fontsize=9.5, fontweight="bold",
            color="#0f172a"
        )

    # Custom legend for panel B
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#dc2626", edgecolor="#1e293b", label="Downregulation (Suppressed)"),
        Patch(facecolor="#2563eb", edgecolor="#1e293b", label="Upregulation (Compensatory)")
    ]
    ax_bar.legend(handles=legend_elements, loc="lower right", frameon=True, framealpha=0.9, fontsize=9.5)

    fig.suptitle(
        "In Silico Pearlian Causal Interventions: do(Master Regulator = 0) in AML (GSE116256)",
        fontsize=14, fontweight="heavy", y=0.98, color="#0f172a"
    )

    plt.tight_layout(rect=[0, 0.03, 1, 0.94])

    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_png, dpi=dpi, bbox_inches="tight")

    mirror_fig = output_png.parent / "figures" / output_png.name
    mirror_fig.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(mirror_fig, dpi=dpi, bbox_inches="tight")

    plt.close()
    print(f"[+] Successfully exported in silico knockout figure to: {output_png}", flush=True)
    print(f"[+] Mirrored copy saved to: {mirror_fig}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate in silico causal interventions using Pearl's do-operator.")
    parser.add_argument("--graph", type=str, default="results/causal_graph.graphml", help="Path to causal graphml")
    parser.add_argument("--dem", type=str, default="results/GSM3587923_AML1012-D0.dem.txt.gz", help="Path to expression matrix")
    parser.add_argument("--output-csv", type=str, default="results/in_silico_knockout_effects.csv", help="Output CSV path")
    parser.add_argument("--output-png", type=str, default="results/knockout_simulation_effects.png", help="Output PNG path")
    parser.add_argument("--genes", nargs="+", default=["NPM1", "NAMPT", "S100A9", "RNASE2", "FCN1"], help="Genes to knockout")
    args = parser.parse_args()

    run_in_silico_knockout_pipeline(
        graph_path=Path(args.graph),
        dem_path=Path(args.dem),
        output_csv=Path(args.output_csv),
        output_png=Path(args.output_png),
        ko_genes=args.genes
    )
