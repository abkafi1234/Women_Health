"""Local-independence diagnostics for the selected LCA: bivariate residuals (BVR) per item pair.

BVR = Pearson X^2 of the observed vs model-implied 4x4 cross-table divided by its df (9), as in
Latent GOLD. Values well above ~3.84 indicate residual association not explained by the classes.
With N ~ 20,000 even trivial misfit inflates X^2, so we also report a size-free effect measure
(Cramer's-V-type: sqrt(X^2 / (N * 3))).
"""
import os
import pickle

import numpy as np
import pandas as pd

from common import ITEMS, OUT, SHORT, save_table

df = pd.read_pickle(OUT / "analytic.pkl")
X = df[ITEMS].to_numpy(int)
n = len(X)
rows = []
for K in [4, 5]:
    with open(OUT / "models" / "lca_models.pkl", "rb") as f:
        m, perm = pickle.load(f)[K]
    pis = m.get_parameters()["measurement"]["pis"].reshape(K, len(ITEMS), 4)
    w = m.get_parameters()["weights"]
    for i in range(16):
        for j in range(i + 1, 16):
            obs = np.zeros((4, 4))
            np.add.at(obs, (X[:, i], X[:, j]), 1)
            exp = n * np.einsum("k,ka,kb->ab", w, pis[:, i], pis[:, j])
            x2 = np.sum((obs - exp) ** 2 / np.clip(exp, 1e-9, None))
            rows.append({"K": K, "item_i": SHORT[ITEMS[i]], "item_j": SHORT[ITEMS[j]], "X2": x2,
                         "BVR": x2 / 9, "effect_V": np.sqrt(x2 / (n * 3))})
bvr = pd.DataFrame(rows)
save_table(bvr, "S13_bivariate_residuals")
summ = bvr.groupby("K").agg(pairs=("BVR", "size"), median_BVR=("BVR", "median"), max_BVR=("BVR", "max"),
                            n_BVR_gt_3_84=("BVR", lambda s: int((s > 3.84).sum())),
                            median_V=("effect_V", "median"), max_V=("effect_V", "max"))
save_table(summ.reset_index(), "T15_local_independence_summary")
print(summ.round(3).to_string())
print(bvr[bvr.K == 5].sort_values("BVR", ascending=False).head(8).round(3).to_string(index=False))
