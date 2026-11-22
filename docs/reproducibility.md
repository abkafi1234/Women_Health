# Reproducibility

- **Environment:** Python 3.12.15; package versions pinned in `requirements.txt`. BLAS threads fixed to 1, all random procedures seeded (network bootstraps, case-dropping, LCA random starts, BLRT, BCH and LCA bootstraps). The number of parallel jobs does not affect results.
- **Runtime:** about 2 hours on 16–20 CPU cores (network bootstraps ≈ 20 min, LCA + BLRT ≈ 55 min, BCH bootstraps ≈ 25 min, LCA robustness ≈ 13 min).
- **Verification:** a clean re-run of the full pipeline from an empty results folder reproduced **43 of 44 result tables identically** (maximum absolute difference 0) relative to an earlier independent run. The only non-identical table came from a reduced smoke-test run. Compare two runs with `python analysis/13_compare_runs.py OLD_RESULTS NEW_RESULTS`.
- **Validation of custom code:**
  - the survey-weighted multinomial logit matches `statsmodels` (coefficients to 1e-5, standard errors to 1e-7, including cluster-robust SEs);
  - the bivariate-normal CDF used for polychoric correlations matches SciPy to machine precision;
  - the BCH three-step estimator recovers known coefficients in simulation, where StepMix 3.0's built-in BCH did not and also ignores sampling weights.
