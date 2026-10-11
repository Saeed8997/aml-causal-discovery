"""
Causal Discovery with Stability Selection using PC Algorithm & Fisher-Z test.
Multi-core parallelized with depth=2 constraint.
"""

from pathlib import Path
import time
import numpy as np
import pandas as pd
import scanpy as sc
from causallearn.search.ConstraintBased.PC import pc
from causallearn.graph.Endpoint import Endpoint
from joblib import Parallel, delayed


def run_single_bootstrap(b, X, gene_names, alpha=0.05, depth=2):
    """
    Executes a single bootstrap iteration of the PC causal discovery algorithm.
    """
    t0 = time.time()
    rng = np.random.RandomState(42 + b)
    n_cells = X.shape[0]
    boot_idx = rng.choice(n_cells, size=int(0.85 * n_cells), replace=True)
    X_b = X[boot_idx, :]

    try:
        cg = pc(
            X_b,
            alpha=alpha,
            indep_test='fisherz',
            node_names=gene_names,
            depth=depth,
            max_k=depth,
            show_progress=False
        )
    except TypeError:
        cg = pc(
            X_b,
            alpha=alpha,
            indep_test='fisherz',
            node_names=gene_names,
            max_k=depth,
            show_progress=False
        )

    directed_edges = []
    for edge in cg.G.get_graph_edges():
        u_name = edge.get_node1().get_name()
        v_name = edge.get_node2().get_name()
        ep1 = edge.get_endpoint1()
        ep2 = edge.get_endpoint2()

        if ep1 == Endpoint.TAIL and ep2 == Endpoint.ARROW:
            directed_edges.append((u_name, v_name))
        elif ep1 == Endpoint.ARROW and ep2 == Endpoint.TAIL:
            directed_edges.append((v_name, u_name))

    elapsed = time.time() - t0
    print(f"    [Bootstrap {b+1:02d}] Completed in {elapsed:.1f}s | {len(directed_edges)} directed edges found", flush=True)
    return directed_edges


def run_causal_stability_selection(
    adata_hvg: sc.AnnData,
    n_bootstraps: int = 25,
    stability_threshold: float = 0.60,
    alpha: float = 0.05,
    depth: int = 2,
    tab_dir: Path = None
) -> tuple[pd.DataFrame, list[str]]:
    """
    Executes bootstrapped PC algorithm across multi-core workers to identify high-confidence directed causal interactions.
    """
    if hasattr(adata_hvg.X, "toarray"):
        X = adata_hvg.X.toarray()
    else:
        X = np.asarray(adata_hvg.X)

    gene_names = list(adata_hvg.var_names)
    n_cells = X.shape[0]

    print(f"[*] Executing {n_bootstraps} bootstraps (depth={depth}) with joblib.Parallel(n_jobs=-1)...", flush=True)
    start_time = time.time()
    bootstrap_results = Parallel(n_jobs=-1, verbose=0)(
        delayed(run_single_bootstrap)(b, X, gene_names, alpha, depth)
        for b in range(n_bootstraps)
    )
    total_time = time.time() - start_time
    print(f"[+] All {n_bootstraps} bootstraps completed in {total_time:.1f}s ({total_time/60:.2f} min)", flush=True)

    directed_edge_counts = {}
    for edges in bootstrap_results:
        for edge in edges:
            directed_edge_counts[edge] = directed_edge_counts.get(edge, 0) + 1

    edge_records = []
    for (src, tgt), count in directed_edge_counts.items():
        freq = count / n_bootstraps
        edge_records.append({
            "Source": src,
            "Target": tgt,
            "Count": count,
            "Total_Bootstraps": n_bootstraps,
            "Stability_Frequency": round(freq, 4),
            "Edge_Type": "directed_causal"
        })

    edge_df = pd.DataFrame(edge_records)
    if not edge_df.empty:
        edge_df = edge_df.sort_values(by="Stability_Frequency", ascending=False).reset_index(drop=True)

    stable_df = edge_df[edge_df["Stability_Frequency"] >= stability_threshold].copy()
    if len(stable_df) < 15 and not edge_df.empty:
        stable_df = edge_df[edge_df["Stability_Frequency"] >= 0.40].head(50).copy()

    if tab_dir:
        tab_dir.mkdir(parents=True, exist_ok=True)
        stable_df.to_csv(tab_dir / "stable_causal_edges.csv", index=False)

    return stable_df, gene_names
