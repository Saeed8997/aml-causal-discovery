import os
import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import streamlit.components.v1 as components
import matplotlib.pyplot as plt
import seaborn as sns
from pyvis.network import Network

# Page configuration
st.set_page_config(
    page_title="AML Causal Target Discovery | Single-Cell Pipeline",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1e3a8a;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 1.1rem;
        color: #475569;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .badge-hub {
        background-color: #ef4444;
        color: white;
        padding: 0.2rem 0.6rem;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: bold;
    }
    .badge-target {
        background-color: #3b82f6;
        color: white;
        padding: 0.2rem 0.6rem;
        border-radius: 12px;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Data Loading & Artifact Resolution Helper
# -----------------------------------------------------------------------------
def locate_artifact(filename: str, search_dirs: list[Path]) -> Path | None:
    """Locate artifact file across potential paths."""
    for base in search_dirs:
        candidate = base / filename
        if candidate.exists():
            return candidate
        # Look in subdirectories
        for sub in ["tables", "figures", "networks"]:
            cand_sub = base / sub / filename
            if cand_sub.exists():
                return cand_sub
    return None

@st.cache_data
def load_pipeline_artifacts(results_path_str: str):
    """Loads all CSV tables and GraphML files from the results folder."""
    base_path = Path(results_path_str)
    search_dirs = [
        base_path,
        Path("./results"),
        Path("."),
        Path("../results"),
        Path("./tables"),
        Path("./networks")
    ]

    artifacts = {
        "regulators": None,
        "edges": None,
        "enrichment": None,
        "graph": None,
        "graphml_path": None,
        "drugs": None,
        "knockout": None,
        "subpopulation": None
    }

    # 1. Master Regulators
    reg_file = locate_artifact("top_master_regulators.csv", search_dirs)
    if reg_file:
        artifacts["regulators"] = pd.read_csv(reg_file)

    # 2. Stable Causal Edges
    edge_file = locate_artifact("stable_causal_edges.csv", search_dirs)
    if edge_file:
        artifacts["edges"] = pd.read_csv(edge_file)

    # 3. Enrichment Results
    enr_file = locate_artifact("enrichment_results.csv", search_dirs)
    if enr_file:
        artifacts["enrichment"] = pd.read_csv(enr_file)

    # 4. Network Graph
    graph_file = locate_artifact("causal_graph.graphml", search_dirs)
    if graph_file:
        try:
            artifacts["graph"] = nx.read_graphml(str(graph_file))
            artifacts["graphml_path"] = graph_file
        except Exception:
            pass

    # If graphml not found but edges exist, construct DiGraph from edges
    if artifacts["graph"] is None and artifacts["edges"] is not None:
        G = nx.DiGraph()
        for _, r in artifacts["edges"].iterrows():
            G.add_edge(r["Source"], r["Target"], weight=float(r.get("Stability_Frequency", 1.0)))
        artifacts["graph"] = G

    # 5. Targeted Therapeutics & Drug Interactions
    drug_file = locate_artifact("drug_target_interactions.csv", search_dirs)
    if drug_file:
        artifacts["drugs"] = pd.read_csv(drug_file)

    # 6. In Silico Knockout Perturbations
    ko_file = locate_artifact("in_silico_knockout_effects.csv", search_dirs)
    if ko_file:
        artifacts["knockout"] = pd.read_csv(ko_file)

    # 7. Subpopulation Causal Rewiring
    subpop_file = locate_artifact("subpopulation_causal_diff.csv", search_dirs)
    if subpop_file:
        artifacts["subpopulation"] = pd.read_csv(subpop_file)

    return artifacts

# -----------------------------------------------------------------------------
# Sidebar Configuration
# -----------------------------------------------------------------------------
st.sidebar.image("https://img.icons8.com/color/96/dna-helix.png", width=64)
st.sidebar.title("AML Causal Discovery")
st.sidebar.caption("Single-Cell Causal Inference & Target Identification")

results_input = st.sidebar.text_input("Artifacts Directory", value="./results")
artifacts = load_pipeline_artifacts(results_input)

st.sidebar.markdown("---")
st.sidebar.subheader("🔬 Pipeline Parameters")
st.sidebar.info(r"""
- **Dataset:** GSE116256 (van Galen 2019)
- **Cell Cohort:** AML Blast & Progenitors
- **Feature Selection:** Top 80 Highly Variable Genes
- **Algorithm:** Parallel PC Algorithm (Fisher-Z test, depth=2)
- **Significance ($\alpha$):** 0.05
- **Stability Bootstraps:** 25 Resamplings (Multi-Core)
- **Edge Cutoff:** $\ge 60\%$ Directed Consistency
""")

# Check artifacts availability
missing = []
for k in ["regulators", "edges", "enrichment"]:
    if artifacts[k] is None:
        missing.append(k)

if missing:
    st.warning(f"⚠️ Notice: Artifacts `{', '.join(missing)}` not yet detected in `{results_input}`. Run or sync the Kaggle kernel output to view full results.")

# -----------------------------------------------------------------------------
# Header
# -----------------------------------------------------------------------------
st.markdown('<div class="main-title">🧬 Acute Myeloid Leukemia (AML) Causal Discovery</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Interactive Single-Cell Causal Regulatory Network & Master Regulator Drug Discovery Dashboard</div>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Tabs
# -----------------------------------------------------------------------------
tab_net, tab_target, tab_ko, tab_subpop, tab_rank, tab_enr, tab_drugs, tab_qc = st.tabs([
    "🌐 Causal Network Explorer",
    "🎯 Master Regulator Inspector",
    "⚡ In Silico Knockout Simulator",
    "🔀 Leukemic Hierarchy & Subpopulation Rewiring",
    "📊 Centrality & Target Ranking",
    "🧪 Biological Pathway Enrichment",
    "💊 Targeted Therapeutics & Drug Repurposing",
    "🔬 Preprocessing & QC Audit"
])

# =============================================================================
# TAB 1: CAUSAL NETWORK EXPLORER
# =============================================================================
with tab_net:
    st.subheader("Interactive Directed Causal Regulatory Network")
    st.markdown("""
    This network represents **statistically stable directed causal relationships** ($X \to Y$) identified across bootstrap resamplings.
    Arrows denote causal direction where perturbations in upstream regulators propagate downstream.
    """)

    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([2, 2, 2])
    with col_ctrl1:
        min_out = st.slider("Minimum Causal Out-Degree", min_value=0, max_value=15, value=0, step=1)
    with col_ctrl2:
        min_freq = st.slider("Minimum Stability Frequency", min_value=0.40, max_value=1.0, value=0.60, step=0.05)
    with col_ctrl3:
        search_gene = st.text_input("Highlight Gene of Interest", value="").strip().upper()

    if artifacts["edges"] is not None and not artifacts["edges"].empty:
        df_edges = artifacts["edges"].copy()
        df_edges = df_edges[df_edges["Stability_Frequency"] >= min_freq]

        # Build subnetwork
        G_sub = nx.DiGraph()
        for _, r in df_edges.iterrows():
            G_sub.add_edge(r["Source"], r["Target"], weight=float(r["Stability_Frequency"]))

        # Filter by out-degree
        if min_out > 0:
            nodes_keep = [n for n in G_sub.nodes() if G_sub.out_degree(n) >= min_out or G_sub.in_degree(n) > 0]
            G_sub = G_sub.subgraph(nodes_keep).copy()

        # Identify top master hubs
        top_hubs = set()
        if artifacts["regulators"] is not None and not artifacts["regulators"].empty:
            top_hubs = set(artifacts["regulators"].head(10)["Gene"])

        # PyVis Graph construction
        net = Network(height="650px", width="100%", bgcolor="#ffffff", font_color="#1e293b", directed=True)
        net.barnes_hut(gravity=-3000, central_gravity=0.3, spring_length=120, spring_strength=0.05, damping=0.95)

        for node in G_sub.nodes():
            out_d = G_sub.out_degree(node)
            in_d = G_sub.in_degree(node)

            # Styling
            is_hub = node in top_hubs
            is_searched = (search_gene and node == search_gene)
            is_target_of_search = False
            if search_gene and search_gene in G_sub and node in G_sub.successors(search_gene):
                is_target_of_search = True

            if is_searched:
                color = "#f59e0b"  # Gold
                size = 36
                border_width = 4
            elif is_target_of_search:
                color = "#10b981"  # Emerald green downstream
                size = 28
                border_width = 3
            elif is_hub:
                color = "#ef4444"  # Red Master Regulator
                size = 30
                border_width = 2
            elif out_d > 0:
                color = "#3b82f6"  # Blue Intermediate
                size = 20
                border_width = 1
            else:
                color = "#94a3b8"  # Slate Gray Target
                size = 14
                border_width = 1

            title_text = f"<b>{node}</b><br>Out-Degree: {out_d}<br>In-Degree: {in_d}"
            net.add_node(
                node,
                label=node,
                color=color,
                size=size,
                title=title_text,
                borderWidth=border_width,
                font={"size": 14, "face": "arial", "bold": is_hub or is_searched}
            )

        for u, v, data in G_sub.edges(data=True):
            freq = data.get("weight", 0.6)
            edge_color = "#dc2626" if u in top_hubs else "#64748b"
            net.add_edge(
                u,
                v,
                value=float(freq) * 3,
                title=f"Causal Influence: {u} &rarr; {v}<br>Stability: {freq*100:.1f}%",
                color=edge_color,
                arrows="to"
            )

        # Render HTML
        html_content = net.generate_html()
        components.html(html_content, height=680, scrolling=False)

        st.caption("🔴 Red: Top Master Regulators | 🔵 Blue: Intermediate Regulators | ⚪ Gray: Downstream Targets | 🟡 Gold: Queried Gene | 🟢 Green: Direct Targets of Query")
    else:
        st.info("Awaiting network artifacts. Ensure `stable_causal_edges.csv` is generated.")

# =============================================================================
# TAB 2: MASTER REGULATOR INSPECTOR
# =============================================================================
with tab_target:
    st.subheader("🎯 Master Regulator & Causal Target Profiler")
    st.markdown("""
    Detailed inspection of top-ranked master regulators, their downstream causal influence cascades,
    and therapeutic relevance in Acute Myeloid Leukemia.
    """)

    if artifacts["regulators"] is not None and not artifacts["regulators"].empty:
        reg_df = artifacts["regulators"]
        genes_list = reg_df["Gene"].tolist()

        selected_gene = st.selectbox("Select Master Regulator to Inspect:", genes_list, index=0)
        gene_data = reg_df[reg_df["Gene"] == selected_gene].iloc[0]

        # Metric cards
        m1, m2, m3, m4, m5 = st.columns(5)
        with m1:
            st.metric("Global Rank", f"#{int(gene_data.get('Rank', 1))}")
        with m2:
            st.metric("Master Score", f"{gene_data.get('Master_Regulator_Score', 0.0):.3f}")
        with m3:
            st.metric("Causal Out-Degree", f"{int(gene_data.get('Out_Degree', 0))}")
        with m4:
            st.metric("In-Degree", f"{int(gene_data.get('In_Degree', 0))}")
        with m5:
            st.metric("Downstream Reach", f"{int(gene_data.get('Downstream_Targets_Count', 0))}")

        st.markdown("---")
        col_down1, col_down2 = st.columns([1, 1])

        with col_down1:
            st.markdown(f"#### 🌐 Downstream Causal Targets of `{selected_gene}`")
            raw_targets = str(gene_data.get("Downstream_Targets_List", "None"))
            if raw_targets and raw_targets != "None":
                target_genes = [t.strip() for t in raw_targets.split(",")]
                st.write(f"Direct and indirect downstream targets ({len(target_genes)} genes):")
                
                # Show as tags/badges
                badges = " ".join([f'<span class="badge-target">{tg}</span>' for tg in target_genes])
                st.markdown(badges, unsafe_allow_html=True)

                # Filter edges originating from this gene
                if artifacts["edges"] is not None:
                    out_edges = artifacts["edges"][artifacts["edges"]["Source"] == selected_gene]
                    if not out_edges.empty:
                        st.markdown("##### Direct Causal Projections:")
                        st.dataframe(
                            out_edges[["Source", "Target", "Stability_Frequency"]].rename(
                                columns={"Stability_Frequency": "Bootstrap Stability"}
                            ),
                            use_container_width=True
                        )
            else:
                st.info(f"`{selected_gene}` does not exhibit downstream reachable targets at the current cutoff.")

        with col_down2:
            st.markdown(f"#### 💊 Therapeutic & Biological Rationale (`{selected_gene}`)")
            
            # Annotated biology database for known AML drivers
            aml_knowledge = {
                "MRPL42": {
                    "role": "Mitochondrial Ribosomal Protein L42 (Mitochondrial Translation)",
                    "mechanism": "Leukemia stem cells (LSCs) are uniquely dependent on mitochondrial oxidative phosphorylation and mitochondrial protein synthesis for survival.",
                    "drug_status": "Vulnerable to mitochondrial translation inhibitors (e.g., Tigecycline, Doxycycline, Bedaquiline).",
                    "evidence": "van Galen et al., Cell 2019; Skrtić et al., Cancer Cell 2011"
                },
                "RUNX1": {
                    "role": "Core-Binding Factor Subunit Alpha-2 (Master Hematopoietic TF)",
                    "mechanism": "Essential regulator of myeloid differentiation. Somatic mutations or transcriptional rewiring cause differentiation arrest.",
                    "drug_status": "CBFbeta/RUNX1 allosteric inhibitors (AI-10-47, Ro5-3335).",
                    "evidence": "Groschel et al., Nat Genet 2014"
                },
                "CEBPA": {
                    "role": "CCAAT/Enhancer-Binding Protein Alpha",
                    "mechanism": "Key transcription factor for granulocytic lineage commitment; loss of function promotes blast accumulation.",
                    "drug_status": "Epigenetic reactivation via hypomethylating agents (Azacitidine, Decitabine).",
                    "evidence": "Pabst et al., Nat Genet 2001"
                },
                "MYC": {
                    "role": "Proto-oncogene c-Myc Transcription Factor",
                    "mechanism": "Master driver of cell cycle progression, ribosome biogenesis, and metabolic reprogramming in AML blasts.",
                    "drug_status": "BET bromodomain inhibitors (JQ1, OTX015, INCB054329) and CDK9 inhibitors.",
                    "evidence": "Delmore et al., Cell 2011"
                },
                "BCL2": {
                    "role": "B-cell Lymphoma 2 (Anti-Apoptotic Regulator)",
                    "mechanism": "Prevents mitochondrial outer membrane permeabilization, sequestering pro-apoptotic BH3-only proteins in LSCs.",
                    "drug_status": "Venetoclax (FDA approved for AML in combination with Azacitidine/LDAC).",
                    "evidence": "DiNardo et al., NEJM 2020; Pollyea et al., Nat Med 2018"
                },
                "FLT3": {
                    "role": "FMS-like Tyrosine Kinase 3",
                    "mechanism": "Constitutively active receptor tyrosine kinase driving cell survival and proliferation via STAT5/ERK/AKT.",
                    "drug_status": "Gilteritinib, Midostaurin, Quizartinib (FDA approved for FLT3-mutated AML).",
                    "evidence": "Perl et al., NEJM 2019"
                }
            }

            info = aml_knowledge.get(selected_gene, {
                "role": "Predicted Causal Upstream Regulator",
                "mechanism": "High causal out-degree and betweenness indicates a key transcriptional/signaling nexus in AML blasts.",
                "drug_status": "Novel candidate for functional genomic screening (CRISPR/shRNA) and small molecule targeting.",
                "evidence": "AML Causal Discovery Pipeline"
            })

            st.info(f"**Biological Function:** {info['role']}")
            st.write(f"**Oncogenic Mechanism:** {info['mechanism']}")
            st.success(f"**Targeting Strategy:** {info['drug_status']}")
            st.caption(f"**Reference:** {info['evidence']}")

            if artifacts.get("drugs") is not None and not artifacts["drugs"].empty:
                gene_drugs = artifacts["drugs"][artifacts["drugs"]["Gene"] == selected_gene]
                if not gene_drugs.empty:
                    st.markdown("##### 💊 Mapped Therapeutics & Repurposing Candidates:")
                    for _, d_row in gene_drugs.iterrows():
                        icon = "🟢" if d_row["Clinical_Phase"] == "Approved" else "🔵"
                        st.markdown(f"- **{icon} {d_row['Drug_Name']}** ({d_row['Approval_Status']}) — *{d_row['Interaction_Type']}*: {d_row['Mechanism_Of_Action']}")
    else:
        st.info("Awaiting master regulators data. Ensure `top_master_regulators.csv` exists.")

# =============================================================================
# TAB 3: IN SILICO CAUSAL PERTURBATION & KNOCKOUT SIMULATOR
# =============================================================================
with tab_ko:
    st.subheader("⚡ In Silico Causal Intervention & Gene Knockout Simulator")
    st.markdown(r"""
    Simulating **Pearl's $do$-operator** on the AML gene regulatory DAG:
    Modeling the systemic effect of target ablation $do(\text{Gene} = 0)$ and propagating the downstream
    causal shock through structural equation path coefficients ($\beta_{u \to v}$) in topological order.
    """)

    if artifacts.get("knockout") is not None and not artifacts["knockout"].empty:
        df_ko = artifacts["knockout"].copy()

        # Summary KPIs
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.metric("Master Regulators Tested", df_ko["Knockout_Gene"].nunique())
        with k2:
            st.metric("Downstream Cascades Analyzed", len(df_ko))
        with k3:
            max_down = df_ko["Percent_Delta"].min()
            st.metric("Max Downregulation", f"{max_down:.1f}%", help="Strongest observed downstream expression suppression")
        with k4:
            direct_cnt = len(df_ko[df_ko["Effect_Type"] == "Direct"])
            st.metric("Direct Causal Projections", direct_cnt, help="Immediate downstream children in DAG")

        st.markdown("---")
        st.subheader("🎯 Interactive Knockout Simulation")

        c_ko1, c_ko2, c_ko3 = st.columns([2, 2, 2])
        with c_ko1:
            ko_options = sorted(list(df_ko["Knockout_Gene"].unique()))
            selected_ko = st.selectbox("Select Master Regulator to Knock Out:", ko_options, index=0)
        with c_ko2:
            ko_eff = st.slider("Knockout Efficiency (Inhibition Level %):", min_value=10, max_value=100, value=100, step=10)
        with c_ko3:
            filter_eff_type = st.selectbox("Effect Hierarchy:", ["All Targets (Direct + Cascade)", "Direct Targets Only (Distance = 1)", "Indirect Cascades Only (Distance ≥ 2)"])

        # Filter dataset for selected gene
        gene_ko_df = df_ko[df_ko["Knockout_Gene"] == selected_ko].copy()

        # Scale by efficiency factor
        eff_factor = ko_eff / 100.0
        gene_ko_df["Simulated_Percent_Delta"] = gene_ko_df["Percent_Delta"] * eff_factor
        gene_ko_df["Simulated_Perturbed_Mean"] = np.clip(gene_ko_df["Baseline_Mean"] + (gene_ko_df["Absolute_Shift"] * eff_factor), 0.0, None)
        gene_ko_df["Simulated_Z_Score"] = gene_ko_df["Z_Score_Shift"] * eff_factor

        if filter_eff_type == "Direct Targets Only (Distance = 1)":
            gene_ko_df = gene_ko_df[gene_ko_df["Causal_Distance"] == 1]
        elif filter_eff_type == "Indirect Cascades Only (Distance ≥ 2)":
            gene_ko_df = gene_ko_df[gene_ko_df["Causal_Distance"] >= 2]

        st.markdown(f"#### 🌊 Downstream Response Cascade for $do(\\text{{{selected_ko}}} = 0)$ at {ko_eff}% Efficiency")

        if not gene_ko_df.empty:
            fig_bar, ax_bar = plt.subplots(figsize=(10, max(3.5, len(gene_ko_df) * 0.45)))
            
            plot_df = gene_ko_df.sort_values(by="Simulated_Percent_Delta", ascending=True)
            bar_colors = ["#dc2626" if v < 0 else "#2563eb" for v in plot_df["Simulated_Percent_Delta"]]

            y_pos = np.arange(len(plot_df))
            ax_bar.barh(y_pos, plot_df["Simulated_Percent_Delta"], color=bar_colors, edgecolor="#1e293b", linewidth=0.8, alpha=0.88)
            ax_bar.axvline(0, color="#334155", linestyle="-", linewidth=1.1)

            labels = [f"{r['Target_Gene']} (d={r['Causal_Distance']})" for _, r in plot_df.iterrows()]
            ax_bar.set_yticks(y_pos)
            ax_bar.set_yticklabels(labels, fontweight="bold", fontsize=10)
            ax_bar.set_xlabel("Expression Shift (% Delta relative to Baseline Mean)", fontweight="bold")
            ax_bar.set_title(f"Target Expression Response following in silico {selected_ko} Knockout", fontweight="bold", fontsize=12)
            ax_bar.grid(axis="x", linestyle="--", alpha=0.55)
            ax_bar.grid(axis="y", visible=False)

            for i, val in enumerate(plot_df["Simulated_Percent_Delta"]):
                ha = "right" if val < 0 else "left"
                offset = -1.2 if val < 0 else 1.2
                ax_bar.text(val + offset, i, f"{val:.1f}%", va="center", ha=ha, fontsize=9.5, fontweight="bold")

            plt.tight_layout()
            st.pyplot(fig_bar)
            plt.close()

            # Detailed response table
            st.markdown("##### 📋 Perturbation Response Metrics")
            show_cols = [
                "Target_Gene", "Causal_Distance", "Effect_Type",
                "Baseline_Mean", "Simulated_Perturbed_Mean",
                "Simulated_Percent_Delta", "Simulated_Z_Score", "Response_Direction"
            ]
            rename_dict = {
                "Target_Gene": "Target Gene",
                "Causal_Distance": "Path Distance",
                "Effect_Type": "Hierarchy",
                "Baseline_Mean": "Baseline Mean",
                "Simulated_Perturbed_Mean": "Perturbed Mean",
                "Simulated_Percent_Delta": "Delta (%)",
                "Simulated_Z_Score": "Z-Shift",
                "Response_Direction": "Direction"
            }
            st.dataframe(gene_ko_df[show_cols].rename(columns=rename_dict), use_container_width=True)

            csv_ko = gene_ko_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label=f"📥 Download {selected_ko} Knockout Table (CSV)",
                data=csv_ko,
                file_name=f"knockout_{selected_ko}_effects.csv",
                mime="text/csv"
            )

        else:
            st.info(f"No targets match the filter criteria for {selected_ko}.")

        st.markdown("---")
        st.subheader("🔬 Global Causal Shock Matrix & Publication Summary")

        search_dirs = [Path(results_input), Path("./results"), Path(".")]
        ko_static_img = locate_artifact("knockout_simulation_effects.png", search_dirs)
        if ko_static_img and ko_static_img.exists():
            st.image(str(ko_static_img), caption="Global In Silico Knockout Effect Heatmap & Top Cascades (300 DPI)", use_container_width=True)
        else:
            st.info("Run `scripts/in_silico_knockout.py` to generate the global perturbation figure.")

        st.markdown(r"""
        #### 💡 Mechanistic Insights from In Silico Knockout
        1. **`NPM1 ➔ MRPL42` Collapse (-62.5%):** Silencing *NPM1* triggers a profound downregulation of *MRPL42* (mitochondrial ribosomal protein L42). In AML, leukemic stem cells rely heavily on oxidative phosphorylation and mitochondrial translation; this demonstrates how targeted nucleolar disruption starves blast bioenergetics.
        2. **`S100A9 ➔ THBS1 / VCAN` Suppression (-87.9% / -52.0%):** Ablating *S100A9* causes immediate shutoff of downstream extracellular matrix regulators (*THBS1* and *VCAN* / versican), disrupting the protective inflammatory bone marrow microenvironment.
        3. **`RNASE2 ➔ S100A8/S100A9 / SRGN` Cascade (-32.1% / -25.4% / -17.4%):** As the top-ranked causal driver, *RNASE2* knockout propagates across 14 downstream genes, attenuating the alarmin network and elastase activation.
        """)

    else:
        st.info("Awaiting in silico knockout artifacts. Run `scripts/in_silico_knockout.py` to generate perturbation data.")

# =============================================================================
# TAB 4: LEUKEMIC HIERARCHY & SUBPOPULATION REWIRING
# =============================================================================
with tab_subpop:
    st.subheader("🔀 Leukemic Hierarchy & Subpopulation Causal Rewiring")
    st.markdown(r"""
    Investigating how the AML causal regulatory network rewires across the leukemic differentiation hierarchy:
    comparing **Immature / Stem-like AML Blasts** (LSC-like / HSC-like / Progenitor-like / GMP-like) against
    **Differentiated AML Blasts** (Mono-like / Gran-like / cDC-like / Differentiated) from van Galen et al. (GSE116256).
    """)

    if artifacts.get("subpopulation") is not None and not artifacts["subpopulation"].empty:
        df_sub = artifacts["subpopulation"].copy()

        # Summary KPIs
        sk1, sk2, sk3, sk4, sk5 = st.columns(5)
        with sk1:
            st.metric("Total Causal Edges", len(df_sub))
        with sk2:
            stem_cnt = len(df_sub[df_sub["Wiring_Category"] == "Stemness-Maintained"])
            st.metric("Stemness Axes", stem_cnt, help="Causal links active preferentially in primitive LSC-like blasts (vulnerabilities)")
        with sk3:
            mat_cnt = len(df_sub[df_sub["Wiring_Category"] == "Maturation-Induced"])
            st.metric("Maturation Axes", mat_cnt, help="Causal links emerging upon monocytic differentiation")
        with sk4:
            core_cnt = len(df_sub[df_sub["Wiring_Category"] == "Conserved Core"])
            st.metric("Conserved Circuits", core_cnt, help="Causal links operating robustly across all maturation stages")
        with sk5:
            max_delta = df_sub["Delta_Beta"].max()
            st.metric("Max Stemness Δβ", f"{max_delta:+.2f}", help="Strongest stemness-specific path coefficient difference")

        st.markdown("---")
        st.subheader("🔍 Interactive Rewiring Explorer & Comparative Filters")

        sc1, sc2, sc3 = st.columns([2, 2, 2])
        with sc1:
            category_options = ["All Categories"] + sorted(df_sub["Wiring_Category"].unique().tolist())
            selected_cat = st.selectbox("Wiring Category Filter", category_options)
        with sc2:
            min_delta_b = st.slider("Minimum Absolute Delta (|Δβ|)", min_value=0.0, max_value=1.2, value=0.10, step=0.05)
        with sc3:
            gene_search = st.text_input("Filter by Gene (Source or Target)", value="").strip().upper()

        # Apply filtering
        filt_sub = df_sub.copy()
        if selected_cat != "All Categories":
            filt_sub = filt_sub[filt_sub["Wiring_Category"] == selected_cat]
        filt_sub = filt_sub[filt_sub["Delta_Beta"].abs() >= min_delta_b]
        if gene_search:
            filt_sub = filt_sub[(filt_sub["Source"] == gene_search) | (filt_sub["Target"] == gene_search)]

        st.caption(f"Showing **{len(filt_sub)}** causal regulatory edges matching filter criteria.")

        # Interactive Visuals
        vcol1, vcol2 = st.columns([1, 1])

        with vcol1:
            st.markdown("##### 📈 Comparative Edge Strength Scatter (Beta_Imm vs. Beta_Diff)")
            fig_sc, ax_sc = plt.subplots(figsize=(7, 6), dpi=150)
            cat_palette = {
                "Stemness-Maintained": "#D9381E",
                "Maturation-Induced": "#0D5C75",
                "Conserved Core": "#2E7D32",
                "Context-Dependent": "#8E8E93"
            }
            for cat, color in cat_palette.items():
                cat_data = filt_sub[filt_sub["Wiring_Category"] == cat]
                if not cat_data.empty:
                    ax_sc.scatter(
                        cat_data["Beta_Immature"],
                        cat_data["Beta_Differentiated"],
                        c=color,
                        label=cat,
                        s=65,
                        alpha=0.85,
                        edgecolors="black",
                        linewidth=0.5
                    )
            # Diagonal
            b_min = min(df_sub["Beta_Immature"].min(), df_sub["Beta_Differentiated"].min()) - 0.1
            b_max = max(df_sub["Beta_Immature"].max(), df_sub["Beta_Differentiated"].max()) + 0.1
            ax_sc.plot([b_min, b_max], [b_min, b_max], "--", color="#7F8C8D", linewidth=1.0, label="Conserved (y=x)")
            ax_sc.axhline(0, color="#BDC3C7", linestyle=":", linewidth=0.8)
            ax_sc.axvline(0, color="#BDC3C7", linestyle=":", linewidth=0.8)
            ax_sc.set_xlabel("Immature / LSC-like Beta", fontsize=9, fontweight="bold")
            ax_sc.set_ylabel("Differentiated Beta", fontsize=9, fontweight="bold")
            ax_sc.set_xlim(b_min, b_max)
            ax_sc.set_ylim(b_min, b_max)
            ax_sc.legend(loc="lower right", fontsize=8, frameon=True)
            ax_sc.grid(True, linestyle="--", alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig_sc)
            plt.close()

        with vcol2:
            st.markdown("##### 📊 Top Differential Regulatory Axes (|Δβ|)")
            if not filt_sub.empty:
                top_edges = filt_sub.sort_values(by="Delta_Beta", ascending=False).head(12)
                fig_bar, ax_bar = plt.subplots(figsize=(7, 6), dpi=150)
                colors = [cat_palette.get(cat, "#8E8E93") for cat in top_edges["Wiring_Category"]]
                y_pos = np.arange(len(top_edges))
                ax_bar.barh(y_pos, top_edges["Delta_Beta"], color=colors, edgecolor="black", linewidth=0.6, height=0.65)
                ax_bar.axvline(0, color="#2C3E50", linewidth=0.9)
                ax_bar.set_yticks(y_pos)
                ax_bar.set_yticklabels(top_edges["Edge"], fontsize=8.5, fontweight="bold")
                ax_bar.set_xlabel("Delta Beta (Immature - Differentiated)", fontsize=9, fontweight="bold")
                ax_bar.invert_yaxis()
                ax_bar.grid(True, axis="x", linestyle="--", alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig_bar)
                plt.close()
            else:
                st.info("No edges found matching criteria.")

        # Leaderboard Table
        st.markdown("##### 📋 Subpopulation Causal Differential Metrics Table")
        disp_cols = [
            "Edge", "Wiring_Category", "Beta_Immature", "Beta_Differentiated",
            "Delta_Beta", "Retention_Immature", "Retention_Differentiated",
            "Pval_Immature", "Pval_Differentiated"
        ]
        rename_map = {
            "Edge": "Causal Edge",
            "Wiring_Category": "Wiring Category",
            "Beta_Immature": "Immature β",
            "Beta_Differentiated": "Diff β",
            "Delta_Beta": "Δβ (Imm - Diff)",
            "Retention_Immature": "Imm Retention",
            "Retention_Differentiated": "Diff Retention",
            "Pval_Immature": "P-val (Imm)",
            "Pval_Differentiated": "P-val (Diff)"
        }
        st.dataframe(filt_sub[disp_cols].rename(columns=rename_map), use_container_width=True)

        csv_sub = filt_sub.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Subpopulation Rewiring Table (CSV)",
            data=csv_sub,
            file_name="subpopulation_causal_diff.csv",
            mime="text/csv"
        )

        st.markdown("---")
        st.subheader("🔬 Publication Figure & Biological Insights")
        search_dirs = [Path(results_input), Path("./results"), Path(".")]
        sub_fig = locate_artifact("subpopulation_causal_wiring.png", search_dirs)
        if sub_fig and sub_fig.exists():
            st.image(str(sub_fig), caption="Subpopulation Causal Rewiring in AML (300 DPI Publication Figure)", use_container_width=True)

        st.markdown(r"""
        #### 💡 Key Biological Findings across the AML Differentiation Hierarchy:
        1. **Stemness-Specific Vulnerabilities (`HDC ➔ MS4A2`, `PRG2 ➔ CLC`, `RNASE2 ➔ SRGN`, `HBB ➔ DNAJC14`):**
           - These regulatory connections are active selectively in primitive **LSC-like and Progenitor-like** compartments ($\Delta \beta > 0.40$), representing prime targets to eliminate the leukemic reservoir without compromising mature myeloid defense.
        2. **Maturation-Driven Cascades (`HDC ➔ CXCL8`, `TPSB2 ➔ MS4A2 / HPGD`, `PPP1R3D ➔ ELANE`, `MTMR7 ➔ NAMPT`):**
           - As leukemic blasts progress towards monocytic differentiation, the transcriptional wiring pivots: *HDC* shifts from MS4A2 activation toward driving inflammatory interleukin-8 (*CXCL8*) chemokine expression ($\Delta \beta = -1.08$), and *MTMR7* activates *NAMPT* NAD+ salvage synthesis ($\Delta \beta = -0.48$).
        3. **Conserved Core Circuits (`SRGN ➔ ELANE`, `NAMPT ➔ RGS2`, `HIST1H1B ➔ CENPF`):**
           - Persist across all differentiation stages with high bootstrap retention ($\ge 75\%$), representing baseline leukemic survival machinery.
        """)

    else:
        st.info("Awaiting subpopulation rewiring analysis. Run `scripts/subpopulation_causal_wiring.py` to generate `subpopulation_causal_diff.csv`.")

# =============================================================================
# TAB 5: CENTRALITY & TARGET RANKING
# =============================================================================
with tab_rank:
    st.subheader("📊 Master Regulator Centrality & Ranking Leaderboard")
    st.markdown("""
    Topological ranking of candidate genes based on causal influence.
    Genes with high **Out-Degree** and **Betweenness Centrality** control regulatory cascades across leukemic cell states.
    """)

    if artifacts["regulators"] is not None and not artifacts["regulators"].empty:
        reg_df = artifacts["regulators"].copy()

        c1, c2 = st.columns([3, 1])
        with c1:
            search_str = st.text_input("Filter by Gene Symbol:", "")
        with c2:
            min_score = st.number_input("Min Master Score:", value=0.0, step=0.1)

        if search_str:
            reg_df = reg_df[reg_df["Gene"].str.contains(search_str, case=False, na=False)]
        if min_score > 0:
            reg_df = reg_df[reg_df["Master_Regulator_Score"] >= min_score]

        display_cols = [
            "Rank", "Gene", "Master_Regulator_Score",
            "Out_Degree", "In_Degree", "Betweenness_Centrality",
            "Downstream_Targets_Count"
        ]
        available_cols = [c for c in display_cols if c in reg_df.columns]
        
        st.dataframe(
            reg_df[available_cols].style.background_gradient(
                subset=["Master_Regulator_Score", "Out_Degree", "Betweenness_Centrality"],
                cmap="YlOrRd"
            ),
            use_container_width=True,
            height=400
        )

        # Download button
        csv_data = reg_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Master Regulators Table (CSV)",
            data=csv_data,
            file_name="aml_master_regulators.csv",
            mime="text/csv"
        )

        st.markdown("---")
        st.subheader("Topological Centrality Distributions")
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].scatter(reg_df["Out_Degree"], reg_df["Betweenness_Centrality"], color="#e74c3c", alpha=0.8, edgecolor="black")
        axes[0].set_title("Out-Degree vs. Betweenness Centrality", fontweight='bold')
        axes[0].set_xlabel("Causal Out-Degree")
        axes[0].set_ylabel("Betweenness Centrality")

        for _, r in reg_df.head(6).iterrows():
            axes[0].annotate(r["Gene"], (r["Out_Degree"] + 0.1, r["Betweenness_Centrality"] + 0.005), fontsize=9, fontweight='bold')

        axes[1].hist(reg_df["Master_Regulator_Score"], bins=15, color="#3498db", edgecolor="black", alpha=0.75)
        axes[1].set_title("Master Regulator Score Distribution", fontweight='bold')
        axes[1].set_xlabel("Score")
        axes[1].set_ylabel("Gene Frequency")

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    else:
        st.info("Awaiting centrality table.")

# =============================================================================
# TAB 4: BIOLOGICAL PATHWAY ENRICHMENT
# =============================================================================
with tab_enr:
    st.subheader("🧪 Functional Biological Validation & Pathway Overrepresentation")
    st.markdown("""
    Overrepresentation analysis of the top causal subnetwork targets across **Gene Ontology (Biological Process)**
    and **KEGG Human Pathways** using `gseapy`.
    """)

    if artifacts["enrichment"] is not None and not artifacts["enrichment"].empty:
        enr_df = artifacts["enrichment"].copy()

        # Metrics
        e1, e2, e3 = st.columns(3)
        with e1:
            st.metric("Total Pathways Tested", len(enr_df))
        with e2:
            sig_cnt = len(enr_df[enr_df["Adjusted_P_value"] < 0.05]) if "Adjusted_P_value" in enr_df.columns else 0
            st.metric("Significant Pathways (FDR < 0.05)", sig_cnt)
        with e3:
            min_p = enr_df["Adjusted_P_value"].min() if "Adjusted_P_value" in enr_df.columns else 1.0
            st.metric("Lowest Adjusted P-Value", f"{min_p:.2e}")

        st.markdown("---")
        st.subheader("Top Enriched Pathways Bar Plot")
        
        top_terms_n = st.slider("Number of Terms to Display", min_value=5, max_value=20, value=10)
        sub_enr = enr_df.head(top_terms_n).copy()

        if "Adjusted_P_value" in sub_enr.columns:
            sub_enr["neg_log10_p"] = -np.log10(np.clip(sub_enr["Adjusted_P_value"].astype(float), 1e-15, 1.0))
        else:
            sub_enr["neg_log10_p"] = 2.0

        sub_enr = sub_enr.sort_values(by="neg_log10_p", ascending=True)

        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.barh(sub_enr["Term"], sub_enr["neg_log10_p"], color="#8b5cf6", edgecolor="black", alpha=0.85)
        ax.axvline(-np.log10(0.05), color="#ef4444", linestyle="--", label="FDR = 0.05 cutoff")
        ax.set_xlabel("-log10(Adjusted P-Value)", fontsize=11, fontweight='bold')
        ax.set_title(f"Top {top_terms_n} Enriched Pathways in Causal Subnetwork", fontsize=13, fontweight='bold')
        ax.legend(loc="lower right")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.markdown("---")
        st.subheader("Enrichment Results Table")
        st.dataframe(enr_df, use_container_width=True)

        csv_enr = enr_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Enrichment Table (CSV)",
            data=csv_enr,
            file_name="aml_enrichment_results.csv",
            mime="text/csv"
        )
    else:
        st.info("Awaiting enrichment results.")

# =============================================================================
# TAB 5: TARGETED THERAPEUTICS & DRUG REPURPOSING
# =============================================================================
with tab_drugs:
    st.subheader("💊 Targeted Therapeutics & Drug Repurposing Explorer")
    st.markdown("""
    Precision oncology translation: Mapping upstream **AML Master Regulators** to **FDA-approved drugs**
    and **investigational antineoplastic agents** using the ChEMBL API, DGIdb, and curated hematology pharmacology.
    """)

    if artifacts.get("drugs") is not None and not artifacts["drugs"].empty:
        df_drugs = artifacts["drugs"].copy()

        # Top summary KPIs
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        with kpi1:
            st.metric("Total Drug Interactions", len(df_drugs))
        with kpi2:
            app_cnt = len(df_drugs[df_drugs["Clinical_Phase"] == "Approved"])
            st.metric("FDA-Approved Compounds", app_cnt, help="Approved drugs eligible for immediate clinical repurposing")
        with kpi3:
            inv_cnt = len(df_drugs[df_drugs["Clinical_Phase"] != "Approved"])
            st.metric("Investigational Agents", inv_cnt, help="Compounds in Phase 1-3 clinical trials or preclinical evaluation")
        with kpi4:
            genes_cov = df_drugs["Gene"].nunique()
            st.metric("Master Regulators Targeted", f"{genes_cov} / 10", help="Top master regulator targets with actionable chemical probes")

        st.markdown("---")
        st.subheader("🔍 Interactive Drug Search & Filtering")

        f_col1, f_col2, f_col3, f_col4 = st.columns([2, 2, 2, 2])

        with f_col1:
            approval_filter = st.selectbox(
                "FDA Clinical Status:",
                ["All Compounds", "FDA Approved Only", "Investigational Only (Phase I-III / Preclinical)"]
            )

        with f_col2:
            all_genes = ["All Genes"] + sorted(list(df_drugs["Gene"].unique()))
            selected_gene_filter = st.selectbox("Target Master Regulator:", all_genes)

        with f_col3:
            all_types = ["All Types"] + sorted(list(df_drugs["Interaction_Type"].dropna().unique()))
            selected_type_filter = st.selectbox("Interaction Mechanism:", all_types)

        with f_col4:
            keyword_search = st.text_input("Search Drug or Mechanism:", "").strip().lower()

        # Apply Filters
        filtered_drugs = df_drugs.copy()

        if approval_filter == "FDA Approved Only":
            filtered_drugs = filtered_drugs[filtered_drugs["Clinical_Phase"] == "Approved"]
        elif approval_filter == "Investigational Only (Phase I-III / Preclinical)":
            filtered_drugs = filtered_drugs[filtered_drugs["Clinical_Phase"] != "Approved"]

        if selected_gene_filter != "All Genes":
            filtered_drugs = filtered_drugs[filtered_drugs["Gene"] == selected_gene_filter]

        if selected_type_filter != "All Types":
            filtered_drugs = filtered_drugs[filtered_drugs["Interaction_Type"] == selected_type_filter]

        if keyword_search:
            filtered_drugs = filtered_drugs[
                filtered_drugs["Drug_Name"].str.lower().str.contains(keyword_search, na=False) |
                filtered_drugs["Mechanism_Of_Action"].str.lower().str.contains(keyword_search, na=False) |
                filtered_drugs["AML_Relevance"].str.lower().str.contains(keyword_search, na=False) |
                filtered_drugs["Target_Class"].str.lower().str.contains(keyword_search, na=False)
            ]

        st.caption(f"Showing **{len(filtered_drugs)}** matching therapeutic candidate(s) of {len(df_drugs)} total.")

        # Interactive Table Display
        display_cols = [
            "Rank", "Gene", "Drug_Name", "Approval_Status", "Clinical_Phase",
            "Interaction_Type", "Mechanism_Of_Action", "Target_Class", "AML_Relevance", "Data_Source"
        ]
        avail_cols = [c for c in display_cols if c in filtered_drugs.columns]

        st.dataframe(
            filtered_drugs[avail_cols],
            use_container_width=True,
            height=380
        )

        # Download button
        csv_drugs_data = filtered_drugs.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Drug Repurposing Table (CSV)",
            data=csv_drugs_data,
            file_name="aml_drug_repurposing_candidates.csv",
            mime="text/csv"
        )

        st.markdown("---")
        st.subheader("📊 Drug Repurposing Landscape by Target & Clinical Phase")

        c_v1, c_v2 = st.columns([1, 1])

        with c_v1:
            gene_phase_counts = pd.crosstab(df_drugs["Gene"], df_drugs["Clinical_Phase"].apply(lambda p: "Approved" if p == "Approved" else "Investigational"))
            fig_dp, ax_dp = plt.subplots(figsize=(8, 4.5))
            gene_phase_counts.plot(kind="bar", stacked=True, ax=ax_dp, color=["#10b981", "#3b82f6"], edgecolor="black", alpha=0.85)
            ax_dp.set_title("Drug Candidates per Master Regulator", fontweight="bold", fontsize=11)
            ax_dp.set_xlabel("Target Gene", fontweight="bold")
            ax_dp.set_ylabel("Number of Compounds", fontweight="bold")
            ax_dp.legend(title="Status")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            st.pyplot(fig_dp)
            plt.close()

        with c_v2:
            phase_counts = df_drugs["Clinical_Phase"].value_counts()
            fig_ph, ax_ph = plt.subplots(figsize=(8, 4.5))
            colors = ["#10b981", "#3b82f6", "#f59e0b", "#ec4899", "#8b5cf6"]
            ax_ph.pie(phase_counts, labels=phase_counts.index, autopct="%1.1f%%", startangle=140, colors=colors[:len(phase_counts)], explode=[0.05]*len(phase_counts))
            ax_ph.set_title("Clinical Development Phase Breakdown", fontweight="bold", fontsize=11)
            plt.tight_layout()
            st.pyplot(fig_ph)
            plt.close()

        st.markdown("---")
        st.subheader("💡 High-Priority Therapeutic Opportunities for AML")

        sp1, sp2 = st.columns(2)
        with sp1:
            st.markdown("""
            #### 🧬 Axis 1: NAMPT Metabolic Exhaustion
            - **Lead Candidates:** `Daporinad (FK866)` (Phase 2), `Teglarinad (KPT-9274)` (Phase 1/2)
            - **Causal Context:** `NAMPT` is a top causal driver projecting to `MRPL42` and `NPM1`.
            - **Clinical Rationale:** Leukemic blast cells exhibit strict dependency on the NAD+ salvage pathway. Catalytic NAMPT inhibition induces acute intracellular NAD+ depletion and mitochondrial bioenergetic collapse, showing strong synergy with venetoclax.
            """)
            st.markdown("""
            #### 🛡️ Axis 2: S100A8/S100A9 Alarmin Niche Blockade
            - **Lead Candidates:** `Tasquinimod` (Phase 2/3 in MDS/AML), `Paquinimod`, `Auranofin` (FDA Approved)
            - **Causal Context:** `S100A9` and `S100A8` form an inflammatory cascade controlling `VCAN` and `RGS2`.
            - **Clinical Rationale:** S100A8/A9 heterodimers activate TLR4 and RAGE to sustain leukemic stem cell survival and chemoresistance. Allosteric small-molecule inhibition clears inflammatory marrow protection.
            """)

        with sp2:
            st.markdown("""
            #### 🔬 Axis 3: NPM1 Oncogenic Restoration & Nucleolar Stress
            - **Lead Candidates:** `Actinomycin D` (FDA Approved), `Selinexor` (FDA Approved), `Pevonedistat` (Phase 3)
            - **Causal Context:** `NPM1` is a core bottleneck regulating mitochondrial translation (`MRPL42`).
            - **Clinical Rationale:** Low-dose Actinomycin D selectively triggers nucleolar disruption and proteasomal degradation of mutant cytoplasmic NPM1, restoring wild-type p53 tumor suppression in unfit AML patients.
            """)
            st.markdown("""
            #### ⚙️ Axis 4: Protease & Extracellular Remodeling
            - **Lead Candidates:** `Midostaurin` (FDA Approved), `Doxycycline` (FDA Approved), `Nafamostat`
            - **Causal Context:** `TPSAB1` (tryptase) and `VCAN` (versican) promote blast tissue invasion and stemness.
            - **Clinical Rationale:** Multi-kinase FLT3/KIT inhibition blocks mast/blast degranulation, while matrix metalloproteinase inhibition blocks versican proteolytic activation.
            """)

    else:
        st.info("Awaiting drug interaction mapping. Run `scripts/drug_target_mapping.py` to generate `drug_target_interactions.csv`.")

# =============================================================================
# TAB 6: PREPROCESSING & QC AUDIT
# =============================================================================
with tab_qc:
    st.subheader("🔬 Quality Control & Preprocessing Audit")
    st.markdown("""
    Transparent summary of Scanpy single-cell preprocessing, cell filtering thresholds,
    and library size normalization performed prior to causal discovery.
    """)

    # Look for QC image
    search_dirs = [Path(results_input), Path("./results"), Path(".")]
    qc_fig = locate_artifact("qc_preprocessing_summary.png", search_dirs)
    net_fig = locate_artifact("causal_network_summary.png", search_dirs)
    mr_fig = locate_artifact("master_regulators_centrality.png", search_dirs)
    drug_fig = locate_artifact("drug_repurposing_summary.png", search_dirs)

    col_q1, col_q2 = st.columns(2)
    with col_q1:
        if qc_fig and qc_fig.exists():
            st.image(str(qc_fig), caption="Scanpy Preprocessing QC (Cell Library Size & Detected Gene Distributions)", use_container_width=True)
        else:
            st.info("QC figure will display once generated.")

    with col_q2:
        if mr_fig and mr_fig.exists():
            st.image(str(mr_fig), caption="Top Master Regulators Centrality Comparison", use_container_width=True)
        else:
            st.info("Centrality figure will display once generated.")

    if net_fig and net_fig.exists():
        st.markdown("---")
        st.image(str(net_fig), caption="Global Causal Network Summary (Static High-Res Rendering)", use_container_width=True)

    if drug_fig and drug_fig.exists():
        st.markdown("---")
        st.image(str(drug_fig), caption="Targeted Therapeutics & Drug Repurposing Summary (300 DPI)", use_container_width=True)

    st.markdown("---")
    st.markdown(r"""
    #### ⚙️ Computational Pipeline Protocol
    1. **Data Ingestion:** Single-cell RNA sequencing profiles of bone marrow aspirates from AML patients (van Galen et al., Cell 2019).
    2. **Scanpy QC:** Removal of empty droplets (<50 genes) and low-abundance genes (<5 cells). Mitochondrial filtering (<25% MT counts).
    3. **Normalization:** Count depth normalization to $10^4$ transcripts per cell followed by log-transformation $\ln(1 + x)$.
    4. **Feature Selection:** Seurat highly variable gene dispersion ranking selecting top 80 key drivers and transcriptional regulators.
    5. **Causal Discovery:** Parallel PC Algorithm using Fisher's Z conditional independence tests at $\alpha = 0.05$ with search depth bounded at 2.
    6. **Stability Selection:** 25 bootstrap iterations across multi-core workers with 85% subsampling. Only directed edges with $\ge 60\%$ stability retained.
    7. **Master Regulator Identification:** Composite scoring prioritizing high out-degree, betweenness centrality, and downstream cascade reachability.
    """)
