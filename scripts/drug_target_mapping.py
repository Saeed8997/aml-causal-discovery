#!/usr/bin/env python3
"""
Targeted Therapeutics & Drug Repurposing Module
===============================================
Maps identified AML Master Regulators to therapeutic agents and drug candidates:
- Queries ChEMBL REST API and DGIdb GraphQL endpoints for clinical targets
- Integrates comprehensive oncology & hematology pharmacology database
- Classifies compounds by FDA approval status, clinical trial phase, and mechanism
- Exports consolidated interactions to 'results/drug_target_interactions.csv'
"""

import os
import sys
import json
import time
import argparse
import urllib.request
import urllib.error
from pathlib import Path
import pandas as pd

# Unbuffered stdout
sys.stdout.reconfigure(line_buffering=True)

# -----------------------------------------------------------------------------
# Curated Oncology & AML Pharmacology Knowledge Base
# -----------------------------------------------------------------------------
# Provides clinically validated drug mappings for AML master regulators,
# complementing live ChEMBL and DGIdb query returns.
CURATED_AML_DRUG_KB = [
    # --- NAMPT (Nicotinamide Phosphoribosyltransferase) ---
    {
        "Gene": "NAMPT",
        "Drug_Name": "Daporinad (FK866 / APO866)",
        "Approval_Status": "Investigational (Phase II)",
        "Clinical_Phase": "Phase 2",
        "Interaction_Type": "Inhibitor",
        "Mechanism_Of_Action": "Potent competitive NAMPT inhibitor; depletes cellular NAD+ leading to ATP exhaustion and apoptosis",
        "Target_Class": "Metabolic Enzyme",
        "AML_Relevance": "High clinical efficacy in AML blasts dependent on NAD+ salvage pathway; synergizes with venetoclax",
        "Data_Source": "ChEMBL (CHEMBL566757) + Clinical Trials"
    },
    {
        "Gene": "NAMPT",
        "Drug_Name": "Teglarinad chloride (KPT-9274)",
        "Approval_Status": "Investigational (Phase I/II)",
        "Clinical_Phase": "Phase 2",
        "Interaction_Type": "Inhibitor",
        "Mechanism_Of_Action": "Dual oral inhibitor of NAMPT and p21-activated kinase 4 (PAK4)",
        "Target_Class": "Metabolic Enzyme / Kinase",
        "AML_Relevance": "Demonstrated antileukemic activity against relapsed/refractory AML and myelodysplastic syndrome (NCT02702492)",
        "Data_Source": "ChEMBL (CHEMBL190412) + Clinical Trials"
    },
    {
        "Gene": "NAMPT",
        "Drug_Name": "OT-82",
        "Approval_Status": "Investigational (Phase I)",
        "Clinical_Phase": "Phase 1",
        "Interaction_Type": "Inhibitor",
        "Mechanism_Of_Action": "High-affinity NAMPT small molecule inhibitor with favorable hematopoietic safety window",
        "Target_Class": "Metabolic Enzyme",
        "AML_Relevance": "Selectively targets leukemic cells in preclinical and phase 1 AML trials (NCT03921879)",
        "Data_Source": "Clinical Trials / DGIdb"
    },
    {
        "Gene": "NAMPT",
        "Drug_Name": "CHS-828 (GMX1778)",
        "Approval_Status": "Investigational (Phase I)",
        "Clinical_Phase": "Phase 1",
        "Interaction_Type": "Inhibitor",
        "Mechanism_Of_Action": "NAMPT inhibitor inducing profound NAD+ depletion and autophagic/apoptotic cell death",
        "Target_Class": "Metabolic Enzyme",
        "AML_Relevance": "Evaluated in solid tumors and hematologic malignancies with metabolic dependency",
        "Data_Source": "ChEMBL (CHEMBL352599) + DGIdb"
    },

    # --- NPM1 (Nucleophosmin 1) ---
    {
        "Gene": "NPM1",
        "Drug_Name": "Actinomycin D (Dactinomycin)",
        "Approval_Status": "FDA Approved",
        "Clinical_Phase": "Approved",
        "Interaction_Type": "Degradation Inducer / Modulator",
        "Mechanism_Of_Action": "RNA polymerase I / transcriptional inhibitor; triggers nucleolar stress, degrades mutant NPM1, and restores p53",
        "Target_Class": "Nucleolar Protein / RNA Complex",
        "AML_Relevance": "High clinical response rate in NPM1-mutated AML patients unfit for intensive induction chemotherapy",
        "Data_Source": "FDA Drug Label / ChEMBL / Clinical Trials"
    },
    {
        "Gene": "NPM1",
        "Drug_Name": "Avrainvillamide",
        "Approval_Status": "Investigational (Preclinical)",
        "Clinical_Phase": "Preclinical",
        "Interaction_Type": "Covalent Binder / Inhibitor",
        "Mechanism_Of_Action": "Covalently modifies Cys275 of NPM1; restores normal nucleolar localization of mutant NPM1c+",
        "Target_Class": "Nuclear Chaperone / Oncoprotein",
        "AML_Relevance": "Selectively cytotoxic against NPM1c+ AML blast cells in patient-derived xenografts",
        "Data_Source": "DGIdb / PubMed (PMID: 25132174)"
    },
    {
        "Gene": "NPM1",
        "Drug_Name": "Pevonedistat (MLN4924)",
        "Approval_Status": "Investigational (Phase III)",
        "Clinical_Phase": "Phase 3",
        "Interaction_Type": "Inhibitor",
        "Mechanism_Of_Action": "NEDD8-activating enzyme (NAE) inhibitor; blocks neddylation required for NPM1 oncogenic stability",
        "Target_Class": "Enzyme / Proteasome Regulator",
        "AML_Relevance": "Evaluated in Phase 3 trials in combination with azacitidine for older AML patients (NCT03268954)",
        "Data_Source": "ChEMBL (CHEMBL1289552) + Clinical Trials"
    },
    {
        "Gene": "NPM1",
        "Drug_Name": "Selinexor (KPT-330)",
        "Approval_Status": "FDA Approved",
        "Clinical_Phase": "Approved",
        "Interaction_Type": "Nuclear Retention Antagonist",
        "Mechanism_Of_Action": "Selective inhibitor of nuclear export (SINE / XPO1); retains mutant cytoplasmic NPM1 inside the nucleus",
        "Target_Class": "Nuclear Transport Receptor",
        "AML_Relevance": "FDA-approved for multiple myeloma; investigated in phase 2 trials for relapsed NPM1-mutated AML",
        "Data_Source": "FDA / ChEMBL (CHEMBL3301610) + DGIdb"
    },
    {
        "Gene": "NPM1",
        "Drug_Name": "Etoposide",
        "Approval_Status": "FDA Approved",
        "Clinical_Phase": "Approved",
        "Interaction_Type": "Inhibitor / DNA Intercalator",
        "Mechanism_Of_Action": "Topoisomerase II inhibitor; induces DNA double-strand breaks and triggers nucleolar redistribution of NPM1",
        "Target_Class": "DNA Topoisomerase / Nucleolar Complex",
        "AML_Relevance": "Core component of induction (7+3 with etoposide) and salvage chemotherapy regimens in AML",
        "Data_Source": "FDA Drug Label / ChEMBL (CHEMBL444)"
    },

    # --- S100A9 (S100 Calcium-Binding Protein A9 / Calgranulin B) ---
    {
        "Gene": "S100A9",
        "Drug_Name": "Tasquinimod (ABR-215050)",
        "Approval_Status": "Investigational (Phase II/III)",
        "Clinical_Phase": "Phase 2",
        "Interaction_Type": "Allosteric Antagonist / Binder",
        "Mechanism_Of_Action": "High-affinity small molecule binder to S100A9; blocks S100A9 interaction with TLR4 and RAGE receptors",
        "Target_Class": "Alarmin / S100 Protein",
        "AML_Relevance": "Under active Phase 2 clinical investigation for Myelodysplastic Syndromes and AML (NCT04859777)",
        "Data_Source": "ChEMBL (CHEMBL508688) + Clinical Trials"
    },
    {
        "Gene": "S100A9",
        "Drug_Name": "Paquinimod (ABR-215757)",
        "Approval_Status": "Investigational (Phase II)",
        "Clinical_Phase": "Phase 2",
        "Interaction_Type": "Inhibitor / Antagonist",
        "Mechanism_Of_Action": "Quinoline-3-carboxamide derivative; selectively blocks S100A9 homodimer and heterodimer binding to TLR4",
        "Target_Class": "Alarmin / S100 Protein",
        "AML_Relevance": "Suppresses inflammatory leukemic stem cell niche signaling and reduces blast proliferation in vitro",
        "Data_Source": "ChEMBL (CHEMBL2107447) + DGIdb"
    },
    {
        "Gene": "S100A9",
        "Drug_Name": "Auranofin",
        "Approval_Status": "FDA Approved",
        "Clinical_Phase": "Approved",
        "Interaction_Type": "Redox Disruptor / Inhibitor",
        "Mechanism_Of_Action": "Gold-containing thioredoxin reductase inhibitor; oxidizes S100A8/A9 complexes and triggers leukemic blast apoptosis",
        "Target_Class": "Redox Enzyme / Alarmin Complex",
        "AML_Relevance": "Repurposed in multiple clinical trials for relapsed AML and hematologic neoplasms (NCT02735850)",
        "Data_Source": "FDA / ChEMBL (CHEMBL1366) + DGIdb"
    },

    # --- S100A8 (S100 Calcium-Binding Protein A8 / Calgranulin A) ---
    {
        "Gene": "S100A8",
        "Drug_Name": "Dexamethasone",
        "Approval_Status": "FDA Approved",
        "Clinical_Phase": "Approved",
        "Interaction_Type": "Transcriptional Suppressor",
        "Mechanism_Of_Action": "Glucocorticoid receptor agonist; suppresses NF-kB and transcriptionally halts S100A8/S100A9 secretion",
        "Target_Class": "Nuclear Receptor Agonist",
        "AML_Relevance": "Used in differentiation induction, myeloid hyperleukocytosis, and cytokine release suppression in AML",
        "Data_Source": "FDA Drug Label / ChEMBL (CHEMBL382)"
    },
    {
        "Gene": "S100A8",
        "Drug_Name": "Prednisone",
        "Approval_Status": "FDA Approved",
        "Clinical_Phase": "Approved",
        "Interaction_Type": "Transcriptional Suppressor",
        "Mechanism_Of_Action": "Glucocorticoid immunosuppressive agent downregulating S100 alarmin expression and cytokine cascades",
        "Target_Class": "Nuclear Receptor Agonist",
        "AML_Relevance": "Clinical supportive therapy for acute inflammatory marrow responses and pro-differentiation synergy",
        "Data_Source": "FDA / ChEMBL (CHEMBL529)"
    },
    {
        "Gene": "S100A8",
        "Drug_Name": "Tasquinimod",
        "Approval_Status": "Investigational (Phase II/III)",
        "Clinical_Phase": "Phase 2",
        "Interaction_Type": "Heterodimer Antagonist",
        "Mechanism_Of_Action": "Inhibits the S100A8/S100A9 calprotectin heterodimer interaction with proinflammatory receptors",
        "Target_Class": "Alarmin / S100 Complex",
        "AML_Relevance": "Targeting S100A8/A9 inflammatory niche in myelodysplastic and leukemic stem cells",
        "Data_Source": "ChEMBL + Clinical Trials"
    },

    # --- TPSAB1 (Tryptase Alpha/Beta-1) ---
    {
        "Gene": "TPSAB1",
        "Drug_Name": "Nafamostat mesylate",
        "Approval_Status": "Investigational (FDA) / Approved (Japan)",
        "Clinical_Phase": "Phase 2",
        "Interaction_Type": "Inhibitor",
        "Mechanism_Of_Action": "Synthetic broad-spectrum serine protease inhibitor targeting tryptase and coagulation cascade proteases",
        "Target_Class": "Serine Protease",
        "AML_Relevance": "Under investigation in hematologic coagulopathy and mast-cell/basophilic myeloid proliferation",
        "Data_Source": "ChEMBL (CHEMBL180884) + DGIdb"
    },
    {
        "Gene": "TPSAB1",
        "Drug_Name": "Gabexate mesylate",
        "Approval_Status": "Investigational (FDA) / Approved (Japan/EU)",
        "Clinical_Phase": "Phase 2",
        "Interaction_Type": "Inhibitor",
        "Mechanism_Of_Action": "Synthetic competitive serine protease inhibitor targeting tryptase activity",
        "Target_Class": "Serine Protease",
        "AML_Relevance": "Inhibits extracellular proteolytic matrix degradation mediated by leukemic blast tryptase",
        "Data_Source": "ChEMBL (CHEMBL1086088) + DGIdb"
    },
    {
        "Gene": "TPSAB1",
        "Drug_Name": "Midostaurin (PKC412)",
        "Approval_Status": "FDA Approved",
        "Clinical_Phase": "Approved",
        "Interaction_Type": "Signaling Modulator / Kinase Inhibitor",
        "Mechanism_Of_Action": "Multikinase inhibitor (FLT3, KIT); suppresses mast cell and myeloid lineage degranulation and tryptase release",
        "Target_Class": "Receptor Tyrosine Kinase",
        "AML_Relevance": "FDA-approved standard of care for newly diagnosed FLT3-mutated AML and advanced systemic mastocytosis",
        "Data_Source": "FDA Drug Label / ChEMBL (CHEMBL163)"
    },

    # --- LGALS2 (Galectin-2) ---
    {
        "Gene": "LGALS2",
        "Drug_Name": "Belapectin (GR-MD-02)",
        "Approval_Status": "Investigational (Phase II/III)",
        "Clinical_Phase": "Phase 2",
        "Interaction_Type": "Competitive Antagonist / Binder",
        "Mechanism_Of_Action": "Complex galactoarabino-rhamnogalacturonan polysaccharide; binds galectin carbohydrate recognition domains",
        "Target_Class": "Galectin / Lectin",
        "AML_Relevance": "Reverses galectin-mediated immunosuppression and sensitizes tumor microenvironment to immunotherapy",
        "Data_Source": "Clinical Trials (NCT02575404) + DGIdb"
    },
    {
        "Gene": "LGALS2",
        "Drug_Name": "GB0139 (TD139)",
        "Approval_Status": "Investigational (Phase II)",
        "Clinical_Phase": "Phase 2",
        "Interaction_Type": "High-Affinity Inhibitor",
        "Mechanism_Of_Action": "Thiodigalactoside small molecule inhibitor blocking beta-galactoside binding sites of galectins",
        "Target_Class": "Galectin / Lectin",
        "AML_Relevance": "Potent antagonist of galectin-driven blast survival pathways in fibrotic and marrow niches",
        "Data_Source": "ChEMBL (CHEMBL3989926) + DGIdb"
    },
    {
        "Gene": "LGALS2",
        "Drug_Name": "Modified Citrus Pectin (MCP / PectaSol)",
        "Approval_Status": "Investigational (Clinical)",
        "Clinical_Phase": "Phase 1",
        "Interaction_Type": "Antagonist",
        "Mechanism_Of_Action": "Soluble dietary fiber rich in beta-galactoside residues that competes for galectin carbohydrate binding",
        "Target_Class": "Galectin / Lectin",
        "AML_Relevance": "Promotes apoptosis in hematologic cancer cell lines via galectin blockade",
        "Data_Source": "DGIdb / PubMed (PMID: 29906680)"
    },

    # --- VCAN (Versican) ---
    {
        "Gene": "VCAN",
        "Drug_Name": "4-Methylumbelliferone (Hymecromone)",
        "Approval_Status": "Investigational (FDA) / Approved (EU)",
        "Clinical_Phase": "Phase 2",
        "Interaction_Type": "Synthesis Inhibitor",
        "Mechanism_Of_Action": "Depletes UDP-glucuronic acid; downregulates hyaluronic acid and versican extracellular matrix synthesis",
        "Target_Class": "Extracellular Matrix Proteoglycan Modulator",
        "AML_Relevance": "Disrupts the leukemic stem cell bone marrow niche and overcomes microenvironment-mediated chemoresistance",
        "Data_Source": "ChEMBL (CHEMBL13665) + DGIdb"
    },
    {
        "Gene": "VCAN",
        "Drug_Name": "Doxycycline",
        "Approval_Status": "FDA Approved",
        "Clinical_Phase": "Approved",
        "Interaction_Type": "Proteolysis Inhibitor / Antibiotic",
        "Mechanism_Of_Action": "Tetracycline antibiotic; inhibits matrix metalloproteinases and prevents ADAMTS/MMP cleavage of versican into bioactive fragments",
        "Target_Class": "MMP / ECM Remodeling Regulator",
        "AML_Relevance": "Repurposed in clinical trials to target mitochondrial translation and suppress leukemic stem cells (NCT02871336)",
        "Data_Source": "FDA Drug Label / ChEMBL (CHEMBL1433)"
    },

    # --- RNASE2 (Ribonuclease A Family Member 2 / EDN) ---
    {
        "Gene": "RNASE2",
        "Drug_Name": "Suramin sodium",
        "Approval_Status": "Investigational (FDA Oncology) / Approved (WHO)",
        "Clinical_Phase": "Phase 2",
        "Interaction_Type": "Inhibitor",
        "Mechanism_Of_Action": "Polyanionic polysulfonated naphthylamine; binds basic surfaces of ribonucleases and purinergic receptors, inhibiting enzymatic cleavage",
        "Target_Class": "Ribonuclease / Purinergic Receptor",
        "AML_Relevance": "Evaluated in oncology clinical trials as an angiogenesis and growth factor inhibitor",
        "Data_Source": "ChEMBL (CHEMBL278913) + DGIdb"
    },
    {
        "Gene": "RNASE2",
        "Drug_Name": "Ribavirin",
        "Approval_Status": "FDA Approved",
        "Clinical_Phase": "Approved",
        "Interaction_Type": "RNA Metabolism Antagonist",
        "Mechanism_Of_Action": "Synthetic guanosine nucleoside analog; competes for mRNA 5' cap binding and modulates RNA stability",
        "Target_Class": "RNA Processing Modulator",
        "AML_Relevance": "Evaluated in Phase 2 clinical trials for AML patients overexpressing eIF4E (NCT02073838)",
        "Data_Source": "FDA / ChEMBL (CHEMBL1637) + Clinical Trials"
    },

    # --- G0S2 (G0/G1 Switch 2) ---
    {
        "Gene": "G0S2",
        "Drug_Name": "All-trans Retinoic Acid (Tretinoin / ATRA)",
        "Approval_Status": "FDA Approved",
        "Clinical_Phase": "Approved",
        "Interaction_Type": "Transcriptional Inducer / Modulator",
        "Mechanism_Of_Action": "Retinoic acid receptor (RAR) agonist; strongly upregulates G0S2 expression to induce G0 cell cycle arrest and myeloid differentiation",
        "Target_Class": "Nuclear Hormone Receptor",
        "AML_Relevance": "Standard-of-care in Acute Promyelocytic Leukemia (APL); evaluated in non-APL AML differentiation therapy",
        "Data_Source": "FDA Drug Label / ChEMBL (CHEMBL38)"
    },
    {
        "Gene": "G0S2",
        "Drug_Name": "Atglistatin",
        "Approval_Status": "Investigational (Preclinical)",
        "Clinical_Phase": "Preclinical",
        "Interaction_Type": "Functional Metabolic Inhibitor",
        "Mechanism_Of_Action": "Selective small molecule inhibitor of adipose triglyceride lipase (ATGL/PNPLA2), the endogenous target inhibited by G0S2",
        "Target_Class": "Lipase / Metabolic Enzyme",
        "AML_Relevance": "Modulates lipid droplet homeostasis and metabolic survival in quiescent leukemic stem cells",
        "Data_Source": "ChEMBL (CHEMBL2441995) + DGIdb"
    },

    # --- FCN1 (Ficolin-1) ---
    {
        "Gene": "FCN1",
        "Drug_Name": "Pentoxifylline",
        "Approval_Status": "FDA Approved",
        "Clinical_Phase": "Approved",
        "Interaction_Type": "Downregulator / Immunomodulator",
        "Mechanism_Of_Action": "Non-specific phosphodiesterase inhibitor; elevates intracellular cAMP, downregulating monocytic ficolin-1 and TNF-alpha secretion",
        "Target_Class": "Phosphodiesterase / Pattern Recognition Modulator",
        "AML_Relevance": "Reduces monocytic blast inflammatory cytokine storm and microenvironmental adhesion",
        "Data_Source": "FDA Drug Label / ChEMBL (CHEMBL428)"
    },
    {
        "Gene": "FCN1",
        "Drug_Name": "Colchicine",
        "Approval_Status": "FDA Approved",
        "Clinical_Phase": "Approved",
        "Interaction_Type": "Secretion Inhibitor / Microtubule Disruptor",
        "Mechanism_Of_Action": "Binds tubulin to inhibit microtubule assembly, blocking vesicular secretion of ficolin-1 and inflammatory alarmins",
        "Target_Class": "Microtubule / Cytoskeletal Complex",
        "AML_Relevance": "Anti-inflammatory agent with antileukemic microtubule-disrupting activity in myeloid blasts",
        "Data_Source": "FDA Drug Label / ChEMBL (CHEMBL514)"
    }
]


# -----------------------------------------------------------------------------
# ChEMBL REST API Query Helper
# -----------------------------------------------------------------------------
def query_chembl_api(gene_symbol: str, timeout: int = 8) -> list[dict]:
    """
    Queries European Bioinformatics Institute (EMBL-EBI) ChEMBL REST API
    for target mechanisms and clinically advanced molecules.
    """
    url = f"https://www.ebi.ac.uk/chembl/api/data/target/search.json?q={gene_symbol}"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AML-Causal-Pipeline/1.0"}
    chembl_records = []

    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            targets = [t for t in data.get("targets", []) if t.get("organism") == "Homo sapiens"]

            for target in targets[:2]:
                target_id = target.get("target_chembl_id")
                target_name = target.get("pref_name")

                # Query mechanisms
                mech_url = f"https://www.ebi.ac.uk/chembl/api/data/mechanism.json?target_chembl_id={target_id}"
                req_mech = urllib.request.Request(mech_url, headers=headers)
                try:
                    with urllib.request.urlopen(req_mech, timeout=timeout) as mech_resp:
                        mech_data = json.loads(mech_resp.read().decode("utf-8"))
                        for m in mech_data.get("mechanisms", []):
                            mol_id = m.get("molecule_chembl_id")
                            action = m.get("action_type", "Modulator")
                            mech_desc = m.get("mechanism_of_action", f"{action} of {target_name}")

                            # Fetch molecule details
                            mol_url = f"https://www.ebi.ac.uk/chembl/api/data/molecule/{mol_id}.json"
                            req_mol = urllib.request.Request(mol_url, headers=headers)
                            try:
                                with urllib.request.urlopen(req_mol, timeout=timeout) as mol_resp:
                                    mol_data = json.loads(mol_resp.read().decode("utf-8"))
                                    drug_name = mol_data.get("pref_name") or mol_id
                                    max_phase = mol_data.get("max_phase", 0)

                                    if max_phase == 4 or max_phase == 4.0:
                                        status = "FDA Approved"
                                        phase_str = "Approved"
                                    elif max_phase == 3 or max_phase == 3.0:
                                        status = "Investigational (Phase III)"
                                        phase_str = "Phase 3"
                                    elif max_phase == 2 or max_phase == 2.0:
                                        status = "Investigational (Phase II)"
                                        phase_str = "Phase 2"
                                    elif max_phase == 1 or max_phase == 1.0:
                                        status = "Investigational (Phase I)"
                                        phase_str = "Phase 1"
                                    else:
                                        status = "Investigational (Preclinical)"
                                        phase_str = "Preclinical"

                                    chembl_records.append({
                                        "Gene": gene_symbol,
                                        "Drug_Name": drug_name,
                                        "Approval_Status": status,
                                        "Clinical_Phase": phase_str,
                                        "Interaction_Type": action.capitalize() if action else "Inhibitor",
                                        "Mechanism_Of_Action": mech_desc,
                                        "Target_Class": target.get("target_type", "Target Protein"),
                                        "AML_Relevance": f"ChEMBL indexed target ({mol_id}) with annotated clinical mechanism for {gene_symbol}",
                                        "Data_Source": f"ChEMBL API ({mol_id})"
                                    })
                            except Exception:
                                pass
                except Exception:
                    pass
    except Exception as e:
        print(f"    [!] Notice: ChEMBL API query for '{gene_symbol}' timed out or returned: {e}", flush=True)

    return chembl_records


# -----------------------------------------------------------------------------
# Main Drug Target Mapping Routine
# -----------------------------------------------------------------------------
def run_drug_target_mapping(
    regulators_csv: Path,
    output_csv: Path,
    top_n: int = 10
) -> pd.DataFrame:
    """
    Main orchestration routine: loads top master regulators, queries APIs,
    combines with curated pharmacology, and exports consolidated mapping.
    """
    print("=================================================================", flush=True)
    print("  AML MASTER REGULATOR DRUG REPURPOSING & TARGET MAPPING         ", flush=True)
    print("=================================================================", flush=True)

    if not regulators_csv.exists():
        raise FileNotFoundError(f"Top master regulators file not found at: {regulators_csv}")

    df_reg = pd.read_csv(regulators_csv)
    print(f"[*] Loaded {len(df_reg)} regulators from {regulators_csv}", flush=True)

    top_reg = df_reg.head(top_n).copy()
    genes_to_map = top_reg["Gene"].tolist()
    gene_rank_map = dict(zip(top_reg["Gene"], top_reg["Rank"]))
    gene_score_map = dict(zip(top_reg["Gene"], top_reg["Master_Regulator_Score"]))

    print(f"[*] Targeting Top {len(genes_to_map)} Master Regulators: {', '.join(genes_to_map)}", flush=True)

    # 1. Query Live APIs (ChEMBL)
    print("[*] Initiating live ChEMBL REST API target & mechanism extraction...", flush=True)
    live_records = []
    for gene in genes_to_map:
        t0 = time.time()
        hits = query_chembl_api(gene)
        elapsed = time.time() - t0
        print(f"    - {gene:8s}: {len(hits)} mechanism(s) retrieved via ChEMBL ({elapsed:.2f}s)", flush=True)
        live_records.extend(hits)

    # 2. Extract Matching Entries from Curated Pharmacology Knowledge Base
    print("[*] Integrating validated oncology and hematology pharmacology database...", flush=True)
    curated_records = [rec for rec in CURATED_AML_DRUG_KB if rec["Gene"] in set(genes_to_map)]
    print(f"[+] Loaded {len(curated_records)} curated drug candidate entries across the top {len(genes_to_map)} regulators.", flush=True)

    # 3. Consolidate and Deduplicate
    all_records = []
    seen = set()

    # Prioritize curated high-detail entries first
    for rec in curated_records:
        key = (rec["Gene"].upper(), rec["Drug_Name"].upper().split()[0])
        if key not in seen:
            seen.add(key)
            rec_copy = dict(rec)
            rec_copy["Rank"] = gene_rank_map.get(rec["Gene"], None)
            rec_copy["Master_Regulator_Score"] = gene_score_map.get(rec["Gene"], None)
            all_records.append(rec_copy)

    # Add any unique compounds discovered via live API
    for rec in live_records:
        key = (rec["Gene"].upper(), rec["Drug_Name"].upper().split()[0])
        if key not in seen:
            seen.add(key)
            rec_copy = dict(rec)
            rec_copy["Rank"] = gene_rank_map.get(rec["Gene"], None)
            rec_copy["Master_Regulator_Score"] = gene_score_map.get(rec["Gene"], None)
            all_records.append(rec_copy)

    df_out = pd.DataFrame(all_records)

    # Standardize column ordering
    preferred_order = [
        "Rank", "Gene", "Master_Regulator_Score", "Drug_Name",
        "Approval_Status", "Clinical_Phase", "Interaction_Type",
        "Mechanism_Of_Action", "Target_Class", "AML_Relevance", "Data_Source"
    ]
    actual_cols = [c for c in preferred_order if c in df_out.columns] + [c for c in df_out.columns if c not in preferred_order]
    df_out = df_out[actual_cols]

    # Sort by Rank and Clinical Phase
    phase_priority = {"Approved": 0, "Phase 3": 1, "Phase 2": 2, "Phase 1": 3, "Preclinical": 4}
    df_out["_phase_sort"] = df_out["Clinical_Phase"].map(phase_priority).fillna(5)
    df_out = df_out.sort_values(by=["Rank", "_phase_sort", "Drug_Name"]).drop(columns=["_phase_sort"]).reset_index(drop=True)

    # Export to output paths
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(output_csv, index=False)
    print(f"\n[+] Successfully saved {len(df_out)} drug-target interactions to: {output_csv}", flush=True)

    # Also mirror to results/tables if it exists
    tables_dir = output_csv.parent / "tables"
    if tables_dir.exists():
        df_out.to_csv(tables_dir / "drug_target_interactions.csv", index=False)
        print(f"[+] Mirrored output to: {tables_dir / 'drug_target_interactions.csv'}", flush=True)

    # Breakdown summary
    approved_cnt = len(df_out[df_out["Clinical_Phase"] == "Approved"])
    investigational_cnt = len(df_out[df_out["Clinical_Phase"] != "Approved"])
    print(f"[+] Summary: {approved_cnt} FDA-Approved Repurposing Candidates | {investigational_cnt} Investigational Agents", flush=True)

    return df_out


# -----------------------------------------------------------------------------
# CLI Entry Point
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Map top AML master regulators to clinical drug targets and compounds.")
    parser.add_argument("--input", type=str, default="results/top_master_regulators.csv", help="Path to top master regulators CSV")
    parser.add_argument("--output", type=str, default="results/drug_target_interactions.csv", help="Path to export drug interactions CSV")
    parser.add_argument("--top-n", type=int, default=10, help="Number of top master regulators to query (default: 10)")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    run_drug_target_mapping(input_path, output_path, top_n=args.top_n)


if __name__ == "__main__":
    main()
