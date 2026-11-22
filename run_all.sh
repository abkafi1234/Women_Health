#!/usr/bin/env bash
# Reproduce every result, table and figure of the revised manuscript from the BDHS 2022 IR file.
#
#   1. Place BDIR81FL.DTA (DHS Program, registration required) in data/ (see data/README.md).
#   2. python3 -m venv .venv && .venv/bin/pip install -r requirements.txt   (from the repository root)
#   3. PYTHON=.venv/bin/python N_JOBS=16 ./run_all.sh
#
# Runtime on 16-20 cores: about 1.5-2 hours (network bootstraps, BLRT and BCH bootstraps dominate).
# All random procedures use fixed seeds, so results are identical across runs with the pinned versions.
# Outputs: results/ (override with SMTS_OUT); LaTeX tables: results/latex (SMTS_MANUSCRIPT).
set -euo pipefail
cd "$(dirname "$0")/analysis"
PY=${PYTHON:-python3}
case "$PY" in /*) ;; */*) PY="$(cd .. && pwd)/$PY" ;; esac   # relative path -> absolute
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1   # deterministic, no BLAS oversubscription
export N_JOBS=${N_JOBS:-16}
OUTDIR=${SMTS_OUT:-../results}
mkdir -p "$OUTDIR"
log() { echo "[$(date '+%F %T')] $*" | tee -a "$OUTDIR/run_all.log"; }
run() {  # run "<ENV=VAL ...>" script.py logname
  log "START $2 $1"
  env $1 "$PY" "$2" > "$OUTDIR/log_$3.txt" 2>&1 || { log "FAILED $2 (see $OUTDIR/log_$3.txt)"; exit 1; }
  log "DONE  $2"
}
log "Python: $("$PY" --version 2>&1)"
run "X=0"                          00_prepare.py              00_prepare
run "N_BOOT=1000"                  01_network.py              01_network
run "N_INIT=20 N_BLRT=100"         02_mixture.py              02_mixture
for k in 5 4; do
  run "K=$k"                       05_final_figures.py        05_final_figures_K$k
  run "K=$k N_BOOT_BCH=200"        03_regression.py           03_regression_K$k
done
run "X=0"                          04_reproduce_original.py   04_reproduce_original
run "X=0"                          06_codebook.py             06_codebook
run "X=0"                          07_local_independence.py   07_local_independence
run "K=5 N_BOOT_LCA=200"           10_lca_robustness.py       10_lca_robustness
run "X=0"                          08_latex_tables.py         08_latex_tables
run "X=0"                          09_workflow_figure.py      09_workflow_figure
log "ALL DONE"
