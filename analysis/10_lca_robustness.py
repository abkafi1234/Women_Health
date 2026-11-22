"""Robustness of the selected latent class solution (K = 5 by default).

(a) Survey-weighted LCA: the measurement model re-estimated with the BDHS sampling weights, compared with the
    unweighted solution (class sizes, item means, agreement of modal assignments).
(b) Uncertainty of class sizes and class-specific item means: stratified PSU bootstrap (resampling PSUs with
    replacement within strata), re-estimating the LCA in every replicate; classes are matched to the main
    solution on their full item profile (Hungarian assignment). Percentile 95% intervals.
"""
import os
import pickle

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score
from stepmix import StepMix

from common import ITEMS, ITEM_INFO, OUT, SHORT, save_table

K = int(os.environ.get("K", 5))
N_BOOT = int(os.environ.get("N_BOOT_LCA", 200))
N_JOBS = int(os.environ.get("N_JOBS", 16))

df = pd.read_pickle(OUT / "analytic.pkl")
X = df[ITEMS].astype(int).reset_index(drop=True)
w = (df["weight"] / df["weight"].mean()).to_numpy()
PSU = df["v021"].to_numpy()
STR = df["v023"].astype(str).to_numpy()


def item_means(m):
    pis = m.get_parameters()["measurement"]["pis"].reshape(K, len(ITEMS), 4)
    return (pis * np.arange(4)).sum(-1)


def fit(Xd, seed, n_init, tol, sw=None):
    return StepMix(n_components=K, measurement="categorical", n_init=n_init, max_iter=2000, abs_tol=tol,
                   random_state=seed, progress_bar=0, verbose=0).fit(Xd, sample_weight=sw)


def match(mu, ref):
    _, o = linear_sum_assignment(((ref[:, None, :] - mu[None, :, :]) ** 2).sum(-1))
    return o


with open(OUT / "models" / "lca_models.pkl", "rb") as f:
    m0, perm0 = pickle.load(f)[K]
MU0 = item_means(m0)[perm0]
post0 = m0.predict_proba(X)[:, perm0]
modal0 = post0.argmax(1)

# ---------------------------------------------------------------- (a) survey-weighted LCA
mw = fit(X, 11, n_init=20, tol=1e-8, sw=w)
ow = match(item_means(mw), MU0)
MUw = item_means(mw)[ow]
postw = mw.predict_proba(X)[:, ow]
comp = pd.DataFrame({
    "class": [f"C{c + 1}" for c in range(K)],
    "size_unweighted_model_%": m0.get_parameters()["weights"][perm0] * 100,
    "size_weighted_model_%": mw.get_parameters()["weights"][ow] * 100,
    "max_abs_diff_item_mean": np.abs(MUw - MU0).max(1),
})
ari = adjusted_rand_score(modal0, postw.argmax(1))
agree = (modal0 == postw.argmax(1)).mean()
save_table(comp, f"S14_weighted_lca_K{K}", floatfmt="%.3f")
with open(OUT / f"weighted_lca_note_K{K}.txt", "w") as fh:
    fh.write(f"Weighted vs unweighted LCA (K={K}): ARI = {ari:.3f}; identical modal class for {agree * 100:.1f}% "
             f"of women; largest absolute difference in any class item mean = {np.abs(MUw - MU0).max():.3f}\n")
print(open(OUT / f"weighted_lca_note_K{K}.txt").read())
print(comp.round(3).to_string(index=False))

# ---------------------------------------------------------------- (b) PSU bootstrap uncertainty
strata_psus = {h: np.unique(PSU[STR == h]) for h in np.unique(STR)}
rows_of_psu = pd.Series(np.arange(len(PSU))).groupby(PSU).apply(np.asarray).to_dict()


def boot(seed):
    rng = np.random.default_rng(seed)
    idx = np.concatenate([rows_of_psu[c] for ps in strata_psus.values() for c in rng.choice(ps, len(ps), replace=True)])
    Xb = X.iloc[idx].reset_index(drop=True)
    try:
        mb = fit(Xb, seed, n_init=3, tol=1e-6)
    except Exception:
        return None
    o = match(item_means(mb), MU0)
    pb = mb.predict_proba(Xb)[:, o]
    size_w = (pb * w[idx][:, None]).sum(0) / w[idx].sum()   # survey-weighted posterior class size
    return item_means(mb)[o], size_w


res = [r for r in Parallel(n_jobs=N_JOBS)(delayed(boot)(20_000 + b) for b in range(N_BOOT)) if r is not None]
MUb = np.array([r[0] for r in res])
SZb = np.array([r[1] for r in res])
size0 = (post0 * w[:, None]).sum(0) / w.sum()

rows = []
for c in range(K):
    rows.append({"row": "class size, % (weighted)", "class": f"C{c + 1}", "estimate": size0[c] * 100,
                 "ci_low": np.quantile(SZb[:, c], .025) * 100, "ci_high": np.quantile(SZb[:, c], .975) * 100})
for i, v in enumerate(ITEMS):
    for c in range(K):
        rows.append({"row": f"{SHORT[v]} {ITEM_INFO.label[i]}", "class": f"C{c + 1}", "estimate": MU0[c, i],
                     "ci_low": np.quantile(MUb[:, c, i], .025), "ci_high": np.quantile(MUb[:, c, i], .975)})
unc = pd.DataFrame(rows)
unc["n_boot_ok"] = len(res)
save_table(unc, f"S15_lca_uncertainty_K{K}", floatfmt="%.3f")
print(unc[unc.row.str.startswith("class size")].round(2).to_string(index=False))
print("max CI width of item means:", round(float((unc.ci_high - unc.ci_low)[unc.row != "class size, % (weighted)"].max()), 3))
