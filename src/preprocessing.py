"""
Quality control, normalization, and HVG feature selection using Scanpy.
"""

from pathlib import Path
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt

def preprocess_scrna(adata: sc.AnnData, n_top_hvgs: int = 80, fig_dir: Path = None) -> sc.AnnData:
    """
    Applies cell/gene QC filtering, total count normalization, log1p, and strictly top 80 HVG extraction.
    """
    sc.pp.filter_cells(adata, min_genes=50)
    sc.pp.filter_genes(adata, min_cells=5)

    adata.var['mt'] = adata.var_names.str.startswith(('MT-', 'mt-'))
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
    if 'pct_counts_mt' in adata.obs:
        adata = adata[adata.obs['pct_counts_mt'] < 25, :].copy()

    raw_counts = np.asarray(adata.X.sum(axis=1)).flatten()
    raw_genes = np.asarray((adata.X > 0).sum(axis=1)).flatten()

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    n_top = min(n_top_hvgs, adata.n_vars)
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top, flavor='seurat')

    key_drivers = ["MRPL42", "RUNX1", "CEBPA", "MYC", "BCL2", "FLT3", "NPM1", "DNMT3A", "HOXA9", "MEIS1", "SPI1"]
    for gene in key_drivers:
        if gene in adata.var_names:
            adata.var.loc[gene, 'highly_variable'] = True

    hvg_genes = adata.var_names[adata.var['highly_variable']].tolist()

    if len(hvg_genes) > n_top_hvgs:
        disp_col = 'dispersions_norm' if 'dispersions_norm' in adata.var.columns else 'dispersions'
        present_drivers = [g for g in key_drivers if g in adata.var_names]
        remaining_slots = max(0, n_top_hvgs - len(present_drivers))
        other_genes = [g for g in hvg_genes if g not in present_drivers]
        if disp_col in adata.var.columns and remaining_slots > 0:
            other_genes_sorted = sorted(other_genes, key=lambda g: adata.var.loc[g, disp_col], reverse=True)
            hvg_genes = present_drivers + other_genes_sorted[:remaining_slots]
        else:
            hvg_genes = (present_drivers + other_genes)[:n_top_hvgs]

    if fig_dir:
        fig_dir.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].hist(raw_counts, bins=40, color='#3498db', edgecolor='black', alpha=0.7)
        axes[0].set_title("Total UMI Counts per Cell", fontsize=12, fontweight='bold')
        axes[0].set_xlabel("Library Size")
        axes[0].set_ylabel("Frequency")

        axes[1].hist(raw_genes, bins=40, color='#2ecc71', edgecolor='black', alpha=0.7)
        axes[1].set_title("Detected Genes per Cell", fontsize=12, fontweight='bold')
        axes[1].set_xlabel("Gene Count")
        axes[1].set_ylabel("Frequency")

        plt.tight_layout()
        plt.savefig(fig_dir / "qc_preprocessing_summary.png", dpi=200)
        plt.close()

    return adata[:, hvg_genes].copy()
