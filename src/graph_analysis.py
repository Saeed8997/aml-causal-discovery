"""
Network graph construction, topological centrality computation, and master regulator ranking.
"""

from pathlib import Path
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt

def analyze_causal_network(
    stable_df: pd.DataFrame,
    all_genes: list[str],
    tab_dir: Path = None,
    net_dir: Path = None,
    fig_dir: Path = None
) -> tuple[nx.DiGraph, pd.DataFrame]:
    """
    Constructs networkx DiGraph, computes Out/In degree, Betweenness Centrality,
    PageRank, and identifies Master Regulators.
    """
    G = nx.DiGraph()

    active_nodes = set(stable_df["Source"]).union(set(stable_df["Target"]))
    for node in active_nodes:
        G.add_node(node)

    for _, row in stable_df.iterrows():
        G.add_edge(row["Source"], row["Target"], weight=float(row["Stability_Frequency"]))

    out_degree = dict(G.out_degree())
    in_degree = dict(G.in_degree())
    betweenness = nx.betweenness_centrality(G, weight="weight")
    closeness = nx.closeness_centrality(G)
    try:
        pagerank = nx.pagerank(G, weight="weight")
    except Exception:
        pagerank = {n: 1.0 / len(G) for n in G.nodes()}

    reachability = {}
    downstream_targets = {}
    for node in G.nodes():
        descendants = nx.descendants(G, node)
        reachability[node] = len(descendants)
        downstream_targets[node] = ", ".join(sorted(list(descendants))) if descendants else "None"

    max_out = max(out_degree.values()) if out_degree.values() and max(out_degree.values()) > 0 else 1
    max_bet = max(betweenness.values()) if betweenness.values() and max(betweenness.values()) > 0 else 1
    max_reach = max(reachability.values()) if reachability.values() and max(reachability.values()) > 0 else 1

    records = []
    for node in G.nodes():
        norm_out = out_degree[node] / max_out
        norm_bet = betweenness[node] / max_bet
        norm_reach = reachability[node] / max_reach
        norm_in = in_degree[node] / (max(in_degree.values()) if max(in_degree.values()) > 0 else 1)
        mrs = (0.40 * norm_out) + (0.35 * norm_bet) + (0.25 * norm_reach) - (0.10 * norm_in)

        records.append({
            "Gene": node,
            "Master_Regulator_Score": round(mrs, 4),
            "Out_Degree": out_degree[node],
            "In_Degree": in_degree[node],
            "Betweenness_Centrality": round(betweenness[node], 5),
            "Closeness_Centrality": round(closeness[node], 5),
            "PageRank": round(pagerank[node], 5),
            "Downstream_Targets_Count": reachability[node],
            "Downstream_Targets_List": downstream_targets[node]
        })

    regulators_df = pd.DataFrame(records)
    regulators_df = regulators_df.sort_values(by=["Master_Regulator_Score", "Out_Degree"], ascending=[False, False]).reset_index(drop=True)
    regulators_df["Rank"] = range(1, len(regulators_df) + 1)

    if tab_dir:
        tab_dir.mkdir(parents=True, exist_ok=True)
        regulators_df.to_csv(tab_dir / "top_master_regulators.csv", index=False)

    if net_dir:
        net_dir.mkdir(parents=True, exist_ok=True)
        nx.write_graphml(G, str(net_dir / "causal_graph.graphml"))

    if fig_dir:
        fig_dir.mkdir(parents=True, exist_ok=True)
        # Summary plots
        sub_df = regulators_df.head(12).sort_values(by="Master_Regulator_Score", ascending=True)
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        axes[0].barh(sub_df["Gene"], sub_df["Out_Degree"], color="#e74c3c", alpha=0.85, edgecolor="black")
        axes[0].set_title("Causal Out-Degree", fontsize=12, fontweight='bold')
        axes[0].set_xlabel("Targets")

        axes[1].barh(sub_df["Gene"], sub_df["Betweenness_Centrality"], color="#3498db", alpha=0.85, edgecolor="black")
        axes[1].set_title("Betweenness Centrality", fontsize=12, fontweight='bold')
        axes[1].set_xlabel("Centrality")
        plt.tight_layout()
        plt.savefig(fig_dir / "master_regulators_centrality.png", dpi=200)
        plt.close()

    return G, regulators_df
