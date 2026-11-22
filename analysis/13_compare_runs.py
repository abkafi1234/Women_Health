"""Compare two output folders (default: ../output_run1 vs ../output) table by table.

For every CSV present in both, reports the maximum absolute difference over all numeric cells and whether
non-numeric cells are identical. Used to document that a clean re-run reproduces the earlier results.
Usage: python 13_compare_runs.py [OLD_DIR] [NEW_DIR]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from common import REV

old = Path(sys.argv[1]) if len(sys.argv) > 1 else REV / "results_previous"
new = Path(sys.argv[2]) if len(sys.argv) > 2 else REV / "results"
rows = []
for f in sorted((new / "tables").glob("*.csv")):
    g = old / "tables" / f.name
    if not g.exists():
        rows.append((f.name, "new in this run", np.nan, ""))
        continue
    a, b = pd.read_csv(g), pd.read_csv(f)
    if a.shape != b.shape:
        rows.append((f.name, f"shape {a.shape} vs {b.shape}", np.nan, ""))
        continue
    num_a, num_b = a.select_dtypes("number"), b.select_dtypes("number")
    same_cols = list(num_a.columns) == list(num_b.columns)
    diff = float(np.nanmax(np.abs(num_a.to_numpy(float) - num_b.to_numpy(float)))) if same_cols and num_a.size else 0.0
    txt_a = a.select_dtypes(exclude="number").fillna("").astype(str)
    txt_b = b.select_dtypes(exclude="number").fillna("").astype(str)
    txt_same = txt_a.shape == txt_b.shape and (txt_a.to_numpy() == txt_b.to_numpy()).all()
    rows.append((f.name, "identical" if diff == 0 and txt_same and same_cols else "differs", diff,
                 "" if txt_same else "text cells differ" if same_cols else "columns differ"))
for f in sorted((old / "tables").glob("*.csv")):
    if not (new / "tables" / f.name).exists():
        rows.append((f.name, "missing in new run", np.nan, ""))
res = pd.DataFrame(rows, columns=["table", "status", "max_abs_numeric_diff", "note"])
pd.set_option("display.width", 200)
print(res.to_string(index=False))
res.to_csv(new / "reproducibility_comparison.csv", index=False)
