# Where each manuscript table and figure comes from

| Manuscript | Content | Script | Output file (results/) |
|---|---|---|---|
| Table 1 | Sample characteristics | 00_prepare.py | T2_covariate_descriptives.csv |
| Table 2 | Item distributions | 00_prepare.py | T1_item_descriptives.csv |
| Table 3 | LCA fit indices, K = 1–7 (incl. BLRT) | 02_mixture.py | T4_mixture_fit_indices.csv |
| Table 4 | Class-specific item means (SD) | 02_mixture.py | S8_lca_K5_class_item_means.csv |
| Table 5 | Centrality with bootstrap CIs | 01_network.py | T6_centrality.csv |
| Table 6 | BCH relative risk ratios (survey-weighted) | 03_regression.py | T12_bch_K5.csv |
| Figure 1 | Analytic workflow | 09_workflow_figure.py | F1_workflow.png |
| Figure 2 | Five latent classes | 05_final_figures.py | F4_final_classes_K5.png |
| Figure 3 | Symptom network | 01_network.py | F2_network.png |
| Figure 4 | Centrality indices | 01_network.py | F3_centrality.png |
| Table S1 | Participant flow | 00_prepare.py | S1_sample_flow.csv |
| Table S2 | Original listwise sample vs excluded women | 00_prepare.py | S2_original_listwise_vs_dropped.csv |
| Table S3 | Reliability | 00_prepare.py | T3_reliability.csv |
| Table S4 | Codebook | 06_codebook.py | S11_codebook.csv |
| Table S5 | Network stability and sensitivity | 01_network.py | T7_cs_coefficients.csv, T8_network_sensitivity.csv |
| Table S6 | Gaussian LPA sensitivity analysis | 02_mixture.py | T4_mixture_fit_indices.csv (LPA rows) |
| Table S7 | Level vs shape | 02_mixture.py | T9_level_vs_shape.csv |
| Table S8 | Bivariate residuals | 07_local_independence.py | T15_local_independence_summary.csv, S13_bivariate_residuals.csv |
| Table S9 | Bootstrap CIs of class sizes and means | 10_lca_robustness.py | S15_lca_uncertainty_K5.csv |
| Table S10 | Survey-weighted LCA | 10_lca_robustness.py | S14_weighted_lca_K5.csv |
| Table S11 | Classification-error matrix | 03_regression.py | S12_classification_error_K5.csv |
| Table S12 | GVIF | 03_regression.py | T11_gvif_K5.csv |
| Tables S13–S17 | Modal-assignment models (unadjusted, adjusted, urban, rural, decision-making) | 03_regression.py | T10_mnlogit_K5.csv |
| Table S18 | BCH for the four-class solution | 03_regression.py | T12_bch_K4.csv |
| Figures S1–S3 | Case-drop stability, edge CIs, strength differences | 01_network.py | S1_case_drop_stability.png, S2_edge_bootstrap.png, S3_strength_difference.png |
| Figure S4 | Information criteria | 02_mixture.py | S4_information_criteria.png |
| Figure S5 | Four-class solution | 05_final_figures.py | F4_final_classes_K4.png |
