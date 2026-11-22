"""Mixture models for the 16 symptom items.

Primary: polytomous latent class analysis (items as 4-category indicators; StepMix 'categorical').
Sensitivity: Gaussian latent profile analysis (diagonal covariance, class-varying variances) on
raw 0-3 scores. Without a variance floor the Gaussian model is degenerate on these data (a class
of all-zero responders has ~0 variance, so the likelihood is unbounded); we therefore use a floor
(reg_covar) and report this explicitly.
"""
import os
import pickle

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture
from stepmix import StepMix

from common import FIG, GAD, ITEMS, ITEM_INFO, OUT, PHQ, SHORT, save_table

N_JOBS = int(os.environ.get("N_JOBS", 16))
N_INIT = int(os.environ.get("N_INIT", 20))
N_BLRT = int(os.environ.get("N_BLRT", 100))
KMAX = 7
REG_COVAR = float(os.environ.get("REG_COVAR", 0.01))

df = pd.read_pickle(OUT / "analytic.pkl")
Xdf = df[ITEMS].astype(int)
X = Xdf.to_numpy()
n = len(X)
MODELS = OUT / "models"
MODELS.mkdir(exist_ok=True)


# ------------------------------------------------------------------ helpers
def fit_lca(k, seed, data=X, n_init=1, tol=1e-8):
    m = StepMix(n_components=k, measurement="categorical", n_init=n_init, max_iter=2000,
                abs_tol=tol, random_state=seed, progress_bar=0, verbose=0)
    m.fit(data)
    return m.score(data) * len(data), m


def best_lca(k):
    fits = Parallel(n_jobs=N_JOBS)(delayed(fit_lca)(k, 1000 * k + s) for s in range(N_INIT))
    lls = np.array([f[0] for f in fits])
    best = fits[int(lls.argmax())][1]
    # proportion of starts replicating the best log-likelihood (local-maxima check)
    rep = float(np.mean(lls.max() - lls < 0.01))
    return best, lls.max(), rep


def fit_lpa(k, seed=0, data=X):
    g = GaussianMixture(n_components=k, covariance_type="diag", reg_covar=REG_COVAR,
                        n_init=N_INIT, max_iter=2000, tol=1e-6, random_state=seed)
    g.fit(data.astype(float))
    return g


def indices(ll, npar, post):
    k = post.shape[1]
    modal = post.argmax(1)
    ent = 1.0 if k == 1 else 1 + np.sum(post * np.log(post + 1e-300)) / (n * np.log(k))
    sizes = np.bincount(modal, minlength=k) / n
    avepp = [post[modal == c, c].mean() if (modal == c).any() else np.nan for c in range(k)]
    return {"LL": ll, "npar": npar, "AIC": -2 * ll + 2 * npar, "BIC": -2 * ll + npar * np.log(n),
            "SABIC": -2 * ll + npar * np.log((n + 2) / 24), "CAIC": -2 * ll + npar * (np.log(n) + 1),
            "entropy": ent, "smallest_class_%": sizes.min() * 100, "min_AvePP": np.nanmin(avepp),
            "AvePP": "; ".join(f"{a:.2f}" for a in avepp)}


def order_by_severity(means):
    """Return permutation ordering classes by mean item score (ascending)."""
    return np.argsort(means.mean(axis=1))


def lca_item_stats(m):
    """Class-specific expected item scores and SDs from response probabilities."""
    pis = m.get_parameters()["measurement"]["pis"]  # (k, items*4) ordered item-major
    k = pis.shape[0]
    P = pis.reshape(k, len(ITEMS), 4)
    cats = np.arange(4)
    mean = (P * cats).sum(-1)
    sd = np.sqrt((P * cats ** 2).sum(-1) - mean ** 2)
    # sanity check of the (item, category) layout: posterior-weighted data means must agree
    post = m.predict_proba(Xdf)
    emp = (post.T @ X) / post.sum(0)[:, None]
    assert np.allclose(emp, mean, atol=0.02), "unexpected StepMix parameter layout"
    return mean, sd, P


# ------------------------------------------------------------------ LCA K = 1..7
lca_rows, lca_models = [], {}
# cache of fitted models, keyed by the number of random starts and a hash of the data (avoids stale reuse)
import hashlib
_hash = hashlib.sha1(X.tobytes()).hexdigest()[:12]
cache = MODELS / f"lca_fits_cache_ninit{N_INIT}_{_hash}.pkl"
cached = pickle.load(open(cache, "rb")) if cache.exists() else {}
for k in range(1, KMAX + 1):
    if k not in cached:
        cached[k] = best_lca(k)
        pickle.dump(cached, open(cache, "wb"))
    m, ll, rep = cached[k]
    post = m.predict_proba(Xdf)
    mean, _, _ = lca_item_stats(m)
    perm = order_by_severity(mean)
    lca_models[k] = (m, perm)
    r = {"model": "LCA", "K": k, **indices(ll, m.n_parameters, post[:, perm]),
         "best_LL_replicated_%": rep * 100}
    lca_rows.append(r)
    print(f"LCA K={k}: LL={ll:.1f} BIC={r['BIC']:.1f} entropy={r['entropy']:.3f} "
          f"min class={r['smallest_class_%']:.1f}%", flush=True)
with open(MODELS / "lca_models.pkl", "wb") as f:
    pickle.dump(lca_models, f)


# ------------------------------------------------------------------ LPA (Gaussian) K = 1..7
lpa_rows, lpa_models = [], {}
for k in range(1, KMAX + 1):
    g = fit_lpa(k)
    perm = order_by_severity(g.means_)
    post = g.predict_proba(X.astype(float))[:, perm]
    ll = g.score(X.astype(float)) * n
    npar = g._n_parameters()
    lpa_models[k] = (g, perm)
    lpa_rows.append({"model": f"LPA (diag, reg_covar={REG_COVAR})", "K": k, **indices(ll, npar, post)})
    print(f"LPA K={k}: BIC={lpa_rows[-1]['BIC']:.1f} entropy={lpa_rows[-1]['entropy']:.3f}", flush=True)
with open(MODELS / "lpa_models.pkl", "wb") as f:
    pickle.dump(lpa_models, f)

# demonstrate degeneracy without a floor (reviewer question on GMM for ordinal items)
g_raw = GaussianMixture(n_components=4, covariance_type="diag", reg_covar=1e-6, n_init=5,
                        random_state=0).fit(X.astype(float))
degenerate_note = (f"Gaussian LPA K=4 with sklearn default reg_covar=1e-6: LL={g_raw.score(X) * n:.1f}, "
                   f"min class variance={g_raw.covariances_.min():.2e} (unbounded likelihood)")
print(degenerate_note)

fit = pd.DataFrame(lca_rows + lpa_rows)
save_table(fit, "T4_mixture_fit_indices", floatfmt="%.3f")
with open(OUT / "mixture_note.txt", "w") as f:
    f.write(degenerate_note + "\n")

# ------------------------------------------------------------------ class descriptions for candidate K
summ = []
for k in range(2, KMAX + 1):
    m, perm = lca_models[k]
    mean, sd, P = lca_item_stats(m)
    mean, sd, P = mean[perm], sd[perm], P[perm]
    post = m.predict_proba(Xdf)[:, perm]
    modal = post.argmax(1)
    w = np.bincount(modal, minlength=k) / n
    pd.DataFrame(post, index=df.index, columns=[f"C{c + 1}" for c in range(k)]).to_pickle(
        MODELS / f"lca_post_K{k}.pkl")

    tab = pd.DataFrame({"item": [SHORT[c] for c in ITEMS], "label": ITEM_INFO.label})
    for c in range(k):
        tab[f"C{c + 1} mean"] = mean[c]
        tab[f"C{c + 1} SD"] = sd[c]
        tab[f"C{c + 1} P(>=2)"] = P[c, :, 2:].sum(-1)
    tab.loc[len(tab)] = ["", "class size % (modal)", *sum([[w[c] * 100, np.nan, np.nan] for c in range(k)], [])]
    save_table(tab, f"S8_lca_K{k}_class_item_means", floatfmt="%.3f")

    # level vs shape: share of between-class variance in item means captured by one dimension
    M = mean - (w[:, None] * mean).sum(0)
    Mw = np.sqrt(w)[:, None] * M
    sv = np.linalg.svd(Mw, compute_uv=False) ** 2
    first = sv[0] / sv.sum()
    # crossings: pairs of adjacent classes whose item-mean ordering reverses on >=1 item
    cross = [int(np.sum(mean[c + 1] < mean[c])) for c in range(k - 1)]
    gad_phq = [mean[c, :7].mean() / max(mean[c, 7:].mean(), 1e-9) for c in range(k)]
    summ.append({"K": k, "share_between_class_variance_1st_dim_%": first * 100,
                 "items_where_next_class_lower": "; ".join(map(str, cross)),
                 "GAD/PHQ mean ratio by class": "; ".join(f"{x:.2f}" for x in gad_phq),
                 "class sizes %": "; ".join(f"{x * 100:.1f}" for x in w)})
save_table(pd.DataFrame(summ), "T9_level_vs_shape", floatfmt="%.2f")
print(pd.DataFrame(summ).to_string(index=False))

# agreement LCA vs LPA and vs the original 3-class GMM labelling is done in 04_reproduce_original.py
agree = []
for k in range(2, KMAX + 1):
    a = lca_models[k][0].predict(Xdf)
    g, _ = lpa_models[k]
    b = g.predict(X.astype(float))
    agree.append({"K": k, "ARI_LCA_vs_LPA": adjusted_rand_score(a, b)})
save_table(pd.DataFrame(agree), "S9_lca_lpa_agreement")

# ------------------------------------------------------------------ figures
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
for ax, mdl in zip(axes, ["LCA", f"LPA (diag, reg_covar={REG_COVAR})"]):
    f_ = fit[fit.model == mdl]
    for c in ["AIC", "BIC", "SABIC"]:
        ax.plot(f_.K, f_[c], marker="o", label=c)
    ax.set_title(mdl)
    ax.set_xlabel("Number of classes")
    ax.legend()
    ax.grid(ls=":")
fig.tight_layout()
fig.savefig(FIG / "S4_information_criteria.png", dpi=300)
plt.close(fig)

for k in range(2, 6):
    m, perm = lca_models[k]
    mean, sd, _ = lca_item_stats(m)
    mean = mean[perm]
    w = np.bincount(m.predict_proba(Xdf)[:, perm].argmax(1), minlength=k) / n
    fig, ax = plt.subplots(figsize=(11, 4.8))
    for c in range(k):
        ax.plot(range(16), mean[c], marker="o", lw=2, label=f"Class {c + 1} ({w[c] * 100:.1f}%)")
    ax.axvline(6.5, color="grey", ls="--", lw=1)
    ax.text(3, ax.get_ylim()[1] * 0.97, "GAD-7", ha="center", va="top")
    ax.text(11.5, ax.get_ylim()[1] * 0.97, "PHQ-9", ha="center", va="top")
    ax.set_xticks(range(16))
    ax.set_xticklabels([f"{SHORT[c]}\n{ITEM_INFO.set_index('var').loc[c, 'label']}" for c in ITEMS],
                       rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("Expected item score (0 = never ... 3 = always)")
    ax.set_title(f"Latent class item profiles, K = {k} (N = {n:,})")
    ax.legend(fontsize=8)
    ax.grid(ls=":")
    fig.tight_layout()
    fig.savefig(FIG / f"F4_lca_profiles_K{k}.png", dpi=300)
    plt.close(fig)
print(fit[["model", "K", "LL", "npar", "BIC", "SABIC", "entropy", "smallest_class_%", "min_AvePP"]
          + [c for c in ["BLRT_p"] if c in fit]].round(3).to_string(index=False))

# ------------------------------------------------------------------ BLRT (parametric bootstrap, K vs K-1)
def blrt_rep(k, seed):
    m0, _ = lca_models[k - 1]
    Xs = m0.sample(n)[0]  # returns (X, Y, labels)
    Xs = pd.DataFrame(np.asarray(Xs).astype(int)[:, :len(ITEMS)], columns=ITEMS)
    ll0 = max(fit_lca(k - 1, seed + s, Xs, tol=1e-6)[0] for s in range(3))
    ll1 = max(fit_lca(k, seed + 100 + s, Xs, tol=1e-6)[0] for s in range(3))
    return 2 * (ll1 - ll0)


lr_obs = {k: 2 * (lca_rows[k - 1]["LL"] - lca_rows[k - 2]["LL"]) for k in range(2, KMAX + 1)}
if N_BLRT > 0:
    for k in range(2, KMAX + 1):
        null = np.array(Parallel(n_jobs=N_JOBS)(delayed(blrt_rep)(k, 50_000 + 1000 * k + b * 7)
                                                for b in range(N_BLRT)))
        p = (1 + np.sum(null >= lr_obs[k])) / (N_BLRT + 1)
        lca_rows[k - 1].update({"BLRT_LR": lr_obs[k], "BLRT_p": p, "BLRT_null_max": null.max()})
        print(f"BLRT {k - 1} vs {k}: LR={lr_obs[k]:.1f}, null max={null.max():.1f}, p={p:.4f}", flush=True)

    fit = pd.DataFrame(lca_rows + lpa_rows)
    save_table(fit, "T4_mixture_fit_indices", floatfmt="%.3f")
    print(fit[["model", "K", "BIC", "BLRT_LR", "BLRT_null_max", "BLRT_p"]].round(3).to_string(index=False))
