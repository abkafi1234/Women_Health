"""Reproduce the originally submitted results and quantify each error (for the response letter)."""
import numpy as np
import pandas as pd
from sklearn.covariance import GraphicalLasso
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from statsmodels.discrete.discrete_model import MNLogit

from common import ITEMS, LABELS, OUT, SHORT, save_table

df = pd.read_pickle(OUT / "analytic.pkl")
flag = pd.read_pickle(OUT / "original_sample_flag.pkl")["in_original_sample"]
d = df[flag]
X = d[ITEMS].to_numpy(float)
Z = StandardScaler().fit_transform(X)
lines = [f"Original listwise sample reproduced: n = {len(d)} (paper reports 10,374; 32 rows with "
         "'don't know'/'refused' were silently dropped before the LPA/network)"]

# original paper labels (PHQ first) vs. true BDHS labels
orig_labels = ['Anhedonia', 'Sadness', 'Sleep', 'Fatigue', 'Appetite', 'Guilt', 'Concentration', 'Motor',
               'Suicide', 'Nervousness', 'Control Worry', 'Excess Worry', 'Relaxation', 'Restlessness',
               'Irritability', 'Fear']
map_tab = pd.DataFrame({"var": ITEMS, "label_in_submission": orig_labels,
                        "true_item": [SHORT[c] for c in ITEMS], "true_label": [LABELS[c] for c in ITEMS]})
save_table(map_tab, "R1_label_mapping_error")

# original network: glasso alpha=0.01 on z-scores, 'strength' = row sums of |precision| (not partial corr.)
g = GraphicalLasso(alpha=0.01, mode="cd", tol=1e-4, max_iter=100).fit(Z)
K = g.precision_.copy()
np.fill_diagonal(K, 0)
raw = np.abs(K).sum(1)
dd = np.sqrt(np.diag(g.precision_))
P = -g.precision_ / np.outer(dd, dd)
np.fill_diagonal(P, 0)
pc = np.abs(P).sum(1)
net = pd.DataFrame({"var": ITEMS, "label_in_submission": orig_labels, "true_label": [LABELS[c] for c in ITEMS],
                    "submitted_'strength'(|precision| sums)": raw, "strength_from_partial_corr": pc}
                   ).sort_values("submitted_'strength'(|precision| sums)", ascending=False)
save_table(net, "R2_original_network_reproduction")
top = net.iloc[0]
lines.append(f"Original network top node: {top['var']} labelled '{top['label_in_submission']}' in the paper, "
             f"actually '{top['true_label']}'; value {top.iloc[3]:.4f} (paper: 1.958096) is a sum of precision-"
             f"matrix entries; strength from partial correlations = {top.iloc[4]:.3f}")
second = net.iloc[1]
lines.append(f"Second node: {second['var']} '{second['label_in_submission']}' -> actually '{second['true_label']}' "
             f"({second.iloc[3]:.4f}; paper 1.713212)")

# original LPA: GMM full covariance, z-scores, single start, seed 42
gm = GaussianMixture(n_components=3, covariance_type="full", random_state=42).fit(Z)
lab = gm.predict(Z)
sizes = np.bincount(lab) / len(lab) * 100
means = pd.DataFrame(X).groupby(lab).mean()
lines.append("Original 3-class GMM sizes by raw label 0/1/2: " + ", ".join(f"{s:.1f}%" for s in sizes))
lines.append("Mean item score per raw label: " + ", ".join(f"{i}: {m:.2f}" for i, m in means.mean(1).items()))
fits = []
for seed in range(10):
    l2 = GaussianMixture(n_components=3, covariance_type="full", random_state=seed).fit(Z)
    fits.append((seed, l2.score(Z) * len(Z), sorted(np.round(np.bincount(l2.predict(Z)) / len(Z) * 100, 1))))
lines.append("Same GMM, 10 seeds (LL, sorted class %): " + "; ".join(f"s{s}: {ll:.0f} {p}" for s, ll, p in fits))

# original MNLogit: Profile recoded via categories [1,0,2]; no intercept
dm = d.copy()
dm["Profile"] = pd.Categorical(lab).reorder_categories([1, 0, 2]).codes
# the original model used the within-residence wealth index (v190a)
w0 = dm["wealth_within_residence"] if "wealth_within_residence" in dm else dm["wealth"]
dummies = pd.DataFrame({
    "v190a_poorest": w0 == "poorest", "v190a_poorer": w0 == "poorer",
    "v190a_middle": w0 == "middle", "v190a_richer": w0 == "richer",
    "v106_no education": dm.education == "no education", "v106_primary": dm.education == "primary",
    "v106_secondary": dm.education == "secondary", "v102_rural": dm.residence == "rural"}).astype(float)
res_noint = MNLogit(dm["Profile"], dummies).fit(method="ncg", maxiter=200, disp=0)
res_int = MNLogit(dm["Profile"], dummies.assign(const=1.0)).fit(method="bfgs", maxiter=2000, disp=0)
comp = pd.DataFrame({"OR_no_intercept(submitted)": np.exp(res_noint.params.iloc[:, -1]),
                     "OR_with_intercept": np.exp(res_int.params.iloc[:-1, -1].reindex(dummies.columns)),
                     "p_with_intercept": res_int.pvalues.iloc[:-1, -1].reindex(dummies.columns)})
save_table(comp.reset_index().rename(columns={"index": "term"}), "R3_original_mnlogit_intercept")
lines.append("Outcome categories after reorder (code: raw label): 0:1, 1:0, 2:2 -> reference is raw label 1 "
             f"(size {sizes[1]:.1f}%), last contrast is raw label 2 (size {sizes[2]:.1f}%)")

with open(OUT / "reproduction_note.txt", "w") as f:
    f.write("\n".join(lines) + "\n")
print("\n".join(lines))
print(net.round(3).to_string(index=False))
print(comp.round(3).to_string())
