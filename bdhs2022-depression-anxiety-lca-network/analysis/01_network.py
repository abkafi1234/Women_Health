"""Psychometric network analysis: polychoric EBICglasso + bootnet-style accuracy/stability."""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from common import (FIG, ITEMS, ITEM_INFO, OUT, SHORT, bridge_strength, cor_matrix, ebic_glasso,
                    estimate_network, expected_influence, save_table, strength)

N_BOOT = int(os.environ.get("N_BOOT", 1000))
N_JOBS = int(os.environ.get("N_JOBS", 18))
SEED = 2026

df = pd.read_pickle(OUT / "analytic.pkl")
X = df[ITEMS].to_numpy(int)
n, p = X.shape
names = [SHORT[c] for c in ITEMS]
labels = [f"{SHORT[c]}\n{ITEM_INFO.set_index('var').loc[c, 'label']}" for c in ITEMS]
comm = ITEM_INFO["scale"].to_numpy()
iu = np.triu_indices(p, 1)


def centralities(W):
    return np.vstack([strength(W), expected_influence(W), bridge_strength(W, comm)])


# ------------------------------------------------------------------ main network
R = cor_matrix(X, "polychoric")
W, lam, path = ebic_glasso(R, n, gamma=0.5, return_path=True)
save_table(pd.DataFrame(R, index=names, columns=names), "S3_polychoric_correlations", index=True)
save_table(pd.DataFrame(W, index=names, columns=names), "S4_network_edge_weights", index=True)
path.to_csv(OUT / "tables" / "S5_ebic_lambda_path.csv", index=False)
C0 = centralities(W)
n_edges = int((np.abs(W[iu]) > 0).sum())
print(f"EBICglasso: lambda={lam:.4f}, {n_edges}/{len(iu[0])} non-zero edges, "
      f"{(W[iu] < 0).sum()} negative")


# ------------------------------------------------------------------ nonparametric bootstrap
def boot_np(seed):
    rng = np.random.default_rng(seed)
    Wb = estimate_network(X[rng.integers(0, n, n)])
    return Wb[iu], centralities(Wb)


res = Parallel(n_jobs=N_JOBS)(delayed(boot_np)(SEED + b) for b in range(N_BOOT))
E_b = np.array([r[0] for r in res])               # (B, edges)
C_b = np.array([r[1] for r in res])               # (B, 3, p)

edges = pd.DataFrame({
    "node_i": [names[i] for i in iu[0]], "node_j": [names[j] for j in iu[1]],
    "label_i": [ITEM_INFO.label[i] for i in iu[0]], "label_j": [ITEM_INFO.label[j] for j in iu[1]],
    "weight": W[iu], "boot_mean": E_b.mean(0),
    "ci_low": np.quantile(E_b, .025, 0), "ci_high": np.quantile(E_b, .975, 0),
    "prop_nonzero": (np.abs(E_b) > 0).mean(0),
}).sort_values("weight", ascending=False)
save_table(edges, "T5_edges_bootstrap", floatfmt="%.3f")

cent_names = ["strength", "expected_influence", "bridge_strength"]
cent = pd.DataFrame({"item": names, "label": ITEM_INFO.label, "scale": comm})
for k, nm in enumerate(cent_names):
    cent[nm] = C0[k]
    cent[f"{nm}_ci_low"] = np.quantile(C_b[:, k], .025, 0)
    cent[f"{nm}_ci_high"] = np.quantile(C_b[:, k], .975, 0)
    cent[f"{nm}_z"] = (C0[k] - C0[k].mean()) / C0[k].std(ddof=1)
cent = cent.sort_values("strength", ascending=False)
save_table(cent, "T6_centrality", floatfmt="%.3f")

# difference tests (bootnet "differenceTest"): CI of bootstrapped difference excludes 0
def diff_matrix(k):
    D = np.zeros((p, p), bool)
    for i in range(p):
        for j in range(p):
            if i != j:
                d = C_b[:, k, i] - C_b[:, k, j]
                lo, hi = np.quantile(d, [.025, .975])
                D[i, j] = (lo > 0) or (hi < 0)
    return D


Dstr = diff_matrix(0)
order = np.argsort(-C0[0])
save_table(pd.DataFrame(Dstr[np.ix_(order, order)], index=np.array(names)[order],
                        columns=np.array(names)[order]).astype(int),
           "S6_strength_difference_tests", index=True)
top = order[0]
print("Top strength:", names[top], ITEM_INFO.label[top], f"{C0[0, top]:.3f};",
      "significantly stronger than", int(Dstr[top].sum()), "of", p - 1, "nodes")

# ------------------------------------------------------------------ case-dropping bootstrap
drop_props = np.round(np.arange(0.05, 0.80, 0.05), 2)
per = max(N_BOOT // len(drop_props), 50)


def boot_cd(seed, prop):
    rng = np.random.default_rng(seed)
    m = int(round(n * (1 - prop)))
    Wb = estimate_network(X[rng.choice(n, m, replace=False)])
    Cb = centralities(Wb)
    cors = [np.corrcoef(C0[k], Cb[k])[0, 1] for k in range(3)]
    cors.append(np.corrcoef(W[iu], Wb[iu])[0, 1])
    return prop, *cors


jobs = [(SEED + 10_000 + i, pr) for i, pr in enumerate(np.repeat(drop_props, per))]
cd = pd.DataFrame(Parallel(n_jobs=N_JOBS)(delayed(boot_cd)(s, pr) for s, pr in jobs),
                  columns=["drop", *cent_names, "edge"])
cd.to_csv(OUT / "tables" / "S7_case_drop_raw.csv", index=False)


def cs_coef(col, thr=0.7, cert=0.95):
    ok = cd.groupby("drop")[col].apply(lambda s: (s >= thr).mean() >= cert)
    ok = ok[ok]
    return float(ok.index.max()) if len(ok) else 0.0


cs = pd.DataFrame({"measure": cent_names + ["edge"],
                   "CS_coefficient(r>=0.7, 95%)": [cs_coef(c) for c in cent_names + ["edge"]]})
save_table(cs, "T7_cs_coefficients")
print(cs.to_string(index=False))

# ------------------------------------------------------------------ sensitivity analyses
sens = []
for label, method, g in [("Spearman, gamma=0.5", "spearman", 0.5), ("Pearson, gamma=0.5", "pearson", 0.5),
                         ("Polychoric, gamma=0.25", "polychoric", 0.25),
                         ("Polychoric, gamma=0", "polychoric", 0.0)]:
    Ws = ebic_glasso(cor_matrix(X, method), n, gamma=g)
    Cs = centralities(Ws)
    sens.append({"model": label, "n_edges": int((np.abs(Ws[iu]) > 0).sum()),
                 "r_edges": np.corrcoef(W[iu], Ws[iu])[0, 1],
                 "r_strength": np.corrcoef(C0[0], Cs[0])[0, 1],
                 "top3_strength": ", ".join(ITEM_INFO.label[np.argsort(-Cs[0])[:3]])})
sens.insert(0, {"model": "Main: polychoric, gamma=0.5", "n_edges": n_edges, "r_edges": 1.0,
                "r_strength": 1.0, "top3_strength": ", ".join(ITEM_INFO.label[order[:3]])})
save_table(pd.DataFrame(sens), "T8_network_sensitivity")
print(pd.DataFrame(sens).round(3).to_string(index=False))

# ------------------------------------------------------------------ figures
col = {"GAD": "#E8A33D", "PHQ": "#4C8BC9"}
G = nx.Graph()
for i in range(p):
    G.add_node(i)
for i, j in zip(*iu):
    if W[i, j] != 0:
        G.add_edge(i, j, weight=W[i, j])
pos = nx.spring_layout(G, weight="weight", seed=7, k=0.9, iterations=500)
fig, ax = plt.subplots(figsize=(9, 8))
wmax = np.abs(W).max()
for i, j, dct in G.edges(data=True):
    w = dct["weight"]
    ax.plot([pos[i][0], pos[j][0]], [pos[i][1], pos[j][1]], color="#2B6CB0" if w > 0 else "#C53030",
            lw=0.3 + 7 * abs(w) / wmax, alpha=0.25 + 0.75 * abs(w) / wmax, zorder=1,
            ls="-" if w > 0 else "--")
for i in range(p):
    ax.scatter(*pos[i], s=1500, color=col[comm[i]], edgecolor="black", zorder=2)
    ax.text(*pos[i], names[i].replace("-", "\n"), ha="center", va="center", fontsize=8,
            fontweight="bold", zorder=3)
from matplotlib.lines import Line2D
leg = [Line2D([], [], marker="o", ls="", ms=12, mfc=col[k], mec="black", label=f"{k} item") for k in col]
leg += [Line2D([], [], color="#2B6CB0", lw=3, label="positive partial correlation"),
        Line2D([], [], color="#C53030", lw=3, ls="--", label="negative partial correlation")]
ax.legend(handles=leg, loc="lower left", fontsize=8, frameon=False)
ax.text(1.0, 0.0, "\n".join(f"{SHORT[c]}: {ITEM_INFO.set_index('var').loc[c, 'label']}" for c in ITEMS),
        transform=ax.transAxes, fontsize=7, va="bottom", ha="left")
ax.set_axis_off()
ax.set_title(f"Regularised partial-correlation network (polychoric EBICglasso, N = {n:,})")
fig.savefig(FIG / "F2_network.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# centrality plot with bootstrap CIs (z not used: raw values with CIs)
fig, axes = plt.subplots(1, 3, figsize=(13, 6), sharey=True)
o = order[::-1]
for k, (ax, nm) in enumerate(zip(axes, ["Strength", "Expected influence", "Bridge strength"])):
    lo = np.quantile(C_b[:, k], .025, 0)
    hi = np.quantile(C_b[:, k], .975, 0)
    ax.errorbar(C0[k][o], range(p), xerr=[C0[k][o] - lo[o], hi[o] - C0[k][o]], fmt="o",
                color="black", ecolor="grey", capsize=2)
    ax.set_title(nm)
    ax.grid(axis="x", ls=":")
axes[0].set_yticks(range(p))
axes[0].set_yticklabels([f"{names[i]} {ITEM_INFO.label[i]}" for i in o], fontsize=8)
fig.suptitle("Node centrality with 95% nonparametric bootstrap intervals")
fig.tight_layout()
fig.savefig(FIG / "F3_centrality.png", dpi=300)
plt.close(fig)

# case-drop stability
fig, ax = plt.subplots(figsize=(7, 4.5))
for c, nm in zip(cent_names + ["edge"], ["Strength", "Expected influence", "Bridge strength", "Edge weights"]):
    g = cd.groupby("drop")[c]
    ax.plot(g.mean().index, g.mean().values, marker="o", label=nm)
    ax.fill_between(g.mean().index, g.quantile(.025), g.quantile(.975), alpha=0.15)
ax.axhline(0.7, color="red", ls="--", lw=1)
ax.set_xlabel("Proportion of cases dropped")
ax.set_ylabel("Correlation with full-sample estimate")
ax.set_title("Case-dropping bootstrap stability")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIG / "S1_case_drop_stability.png", dpi=300)
plt.close(fig)

# edge-weight bootstrap CIs
fig, ax = plt.subplots(figsize=(7, 14))
e = edges.sort_values("weight")
ax.hlines(range(len(e)), e.ci_low, e.ci_high, color="grey")
ax.plot(e.weight, range(len(e)), "o", ms=3, color="black")
ax.set_yticks(range(len(e)))
ax.set_yticklabels(e.node_i + " - " + e.node_j, fontsize=5)
ax.axvline(0, color="red", lw=0.8)
ax.set_xlabel("Edge weight (partial correlation) with 95% bootstrap CI")
fig.tight_layout()
fig.savefig(FIG / "S2_edge_bootstrap.png", dpi=300)
plt.close(fig)

# strength difference test plot
fig, ax = plt.subplots(figsize=(7, 6))
M = Dstr[np.ix_(order, order)].astype(float)
np.fill_diagonal(M, np.nan)
ax.imshow(M, cmap="Greys", vmin=0, vmax=1.5)
ax.set_xticks(range(p))
ax.set_yticks(range(p))
ax.set_xticklabels(np.array(names)[order], rotation=90, fontsize=7)
ax.set_yticklabels([f"{names[i]} ({C0[0, i]:.2f})" for i in order], fontsize=7)
ax.set_title("Strength difference tests (dark = 95% bootstrap CI of difference excludes 0)", fontsize=9)
fig.tight_layout()
fig.savefig(FIG / "S3_strength_difference.png", dpi=300)
plt.close(fig)
print(cent[["item", "label", "strength", "strength_ci_low", "strength_ci_high", "bridge_strength"]]
      .round(3).to_string(index=False))
