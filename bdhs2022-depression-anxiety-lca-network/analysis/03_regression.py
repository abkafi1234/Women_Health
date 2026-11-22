"""Predictors of latent-class membership.

* Multinomial logit on modal class (with intercept; the original model had none), reporting
  relative-risk ratios (RRR) and 95% CIs. Survey-weighted with cluster-robust (Taylor) SEs
  when v005/v021/v023 are available; otherwise unweighted with model-based SEs.
* Bias-adjusted three-step estimation (BCH; StepMix) as the primary classification-error-
  corrected estimate, with bootstrap SEs.
* Collinearity diagnostics (GVIF), unadjusted vs adjusted models, and residence-stratified models.
"""
import os

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import optimize, stats
from scipy.optimize import linear_sum_assignment
from stepmix import StepMix

from common import ITEMS, OUT, save_table

K = int(os.environ.get("K", 5))
N_BOOT = int(os.environ.get("N_BOOT_BCH", 200))
N_JOBS = int(os.environ.get("N_JOBS", 16))

df = pd.read_pickle(OUT / "analytic.pkl")
post = pd.read_pickle(OUT / "models" / f"lca_post_K{K}.pkl")
df["cls"] = post.to_numpy().argmax(1)
WEIGHTED = "weight" in df.columns and "v021" in df.columns
_w = df["weight"] if WEIGHTED else pd.Series(1.0, index=df.index)
prev = pd.DataFrame({"class": [f"C{c + 1}" for c in range(K)],
                     "n_modal": np.bincount(df["cls"], minlength=K),
                     "%_modal_unweighted": np.bincount(df["cls"], minlength=K) / len(df) * 100,
                     "%_modal_weighted": [(_w[df["cls"] == c]).sum() / _w.sum() * 100 for c in range(K)],
                     "%_posterior_weighted": (post.to_numpy() * _w.to_numpy()[:, None]).sum(0) / _w.sum() * 100})
save_table(prev, f"T13_class_prevalence_K{K}", floatfmt="%.2f")
print(prev.round(2).to_string(index=False))

SES = ["education", "wealth", "residence"]
# 'no current partner' is the same 997 women in husband_edu, decisions and marital status, so only
# husband_edu (with a 'no current partner' level) enters the main model; decision-making is examined
# among currently married women (M4).
FULL = SES + ["age_group", "parity", "division", "religion", "working", "husband_edu", "hh_head_female",
              "hh_size", "internet_12m", "owns_mobile", "wifebeating_justified"]
NICE = {"education": "Education", "wealth": "Wealth quintile (national)", "age_group": "Age group",
        "parity": "Children ever born", "marital": "Marital status", "residence": "Residence",
        "division": "Division", "religion": "Religion", "working": "Currently working",
        "husband_edu": "Husband's education", "hh_head_female": "Female household head",
        "hh_size": "Household size (per member)", "children_u5": "Children under 5 (per child)",
        "decisions": "Household decisions participated in", "internet_12m": "Internet use (12 mo)",
        "owns_mobile": "Owns a mobile phone", "wifebeating_justified": "Justifies wife-beating (any)"}


def design(d, covs):
    parts, names, groups = [np.ones((len(d), 1))], ["Intercept"], [None]
    for v in covs:
        if str(d[v].dtype) == "category":
            levels = list(d[v].cat.categories)
            for lev in levels[1:]:
                parts.append((d[v] == lev).to_numpy(float)[:, None])
                names.append(f"{v}={lev}")
                groups.append(v)
        else:
            parts.append(d[[v]].to_numpy(float))
            names.append(v)
            groups.append(v)
    return np.hstack(parts), names, groups


def mlogit(y, Xd, w=None, cluster=None, strata=None, robust_inverse=False):
    """Weighted multinomial logit; reference = class 0. Returns (B, V) with B shape (p, J-1)."""
    n, p = Xd.shape
    J = int(y.max()) + 1
    w = np.ones(n) if w is None else w
    Y = np.eye(J)[y]

    def nll(b):
        B = np.column_stack([np.zeros(p), b.reshape(p, J - 1)])
        eta = Xd @ B
        eta -= eta.max(1, keepdims=True)
        lp = eta - np.log(np.exp(eta).sum(1, keepdims=True))
        P = np.exp(lp)
        g = -(Xd.T @ (w[:, None] * (Y - P)))[:, 1:]
        return -np.sum(w * (Y * lp).sum(1)), g.ravel()

    res = optimize.minimize(nll, np.zeros(p * (J - 1)), jac=True, method="L-BFGS-B",
                            options={"maxiter": 5000, "gtol": 1e-8})
    b = res.x
    B = np.column_stack([np.zeros(p), b.reshape(p, J - 1)])
    eta = Xd @ B
    P = np.exp(eta - eta.max(1, keepdims=True))
    P /= P.sum(1, keepdims=True)
    Pm = P[:, 1:]
    # Hessian of the negative log-likelihood
    H = np.zeros((p * (J - 1), p * (J - 1)))
    for a in range(J - 1):
        for c in range(J - 1):
            wac = w * (Pm[:, a] * ((a == c) - Pm[:, c]))
            H[a::J - 1, c::J - 1] = (Xd * wac[:, None]).T @ Xd
    if robust_inverse:  # point estimates only (BCH weights can be negative)
        return b.reshape(p, J - 1), None, res
    Hinv = np.linalg.inv(H)
    if cluster is None:
        V = Hinv
    else:  # Taylor-linearised, clusters within strata (as in Stata svy)
        S = (w[:, None, None] * Xd[:, :, None] * (Y - P)[:, None, 1:]).reshape(n, -1)
        meat = np.zeros_like(H)
        st = np.zeros(n) if strata is None else strata
        for h in np.unique(st):
            idx = st == h
            sc = pd.DataFrame(S[idx]).groupby(cluster[idx]).sum().to_numpy()
            nh = len(sc)
            if nh > 1:
                sc = sc - sc.mean(0)
                meat += nh / (nh - 1) * sc.T @ sc
        V = Hinv @ meat @ Hinv
    return b.reshape(p, J - 1), V, res


def tidy(B, V, names, label):
    p, J1 = B.shape
    se = np.sqrt(np.diag(V)).reshape(p, J1)
    rows = []
    for j in range(J1):
        for i, nm in enumerate(names):
            if nm == "Intercept":
                continue
            z = B[i, j] / se[i, j]
            rows.append({"model": label, "contrast": f"C{j + 2} vs C1", "term": nm,
                         "RRR": np.exp(B[i, j]), "ci_low": np.exp(B[i, j] - 1.96 * se[i, j]),
                         "ci_high": np.exp(B[i, j] + 1.96 * se[i, j]), "p": 2 * stats.norm.sf(abs(z))})
    return rows


def fit_model(d, covs, label):
    d = d.dropna(subset=covs)
    Xd, names, _ = design(d, covs)
    kw = {}
    if WEIGHTED:
        kw = dict(w=d["weight"].to_numpy(), cluster=d["v021"].to_numpy(), strata=d["v023"].to_numpy())
    B, V, _ = mlogit(d["cls"].to_numpy(), Xd, **kw)
    out = tidy(B, V, names, label)
    for r in out:
        r["n"] = len(d)
    return out


def gvif(d, covs):
    d = d.dropna(subset=covs)
    Xd, names, groups = design(d, covs)
    Xd = Xd[:, 1:]
    groups = np.array(groups[1:])
    R = np.corrcoef(Xd, rowvar=False)
    detR = np.linalg.det(R)
    rows = []
    for v in covs:
        a = groups == v
        g = np.linalg.det(R[np.ix_(a, a)]) * np.linalg.det(R[np.ix_(~a, ~a)]) / detR
        rows.append({"variable": NICE[v], "df": int(a.sum()), "GVIF": g,
                     "GVIF^(1/(2df))": g ** (1 / (2 * a.sum()))})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ models
rows = []
for v in FULL:
    rows += fit_model(df, [v], "M0 unadjusted")
rows += fit_model(df, SES, "M1 education + wealth + residence")
rows += fit_model(df, FULL, "M2 fully adjusted")
for res_level in ["urban", "rural"]:
    rows += fit_model(df[df.residence == res_level], ["education", "wealth"] + [c for c in FULL if c not in SES],
                      f"M3 stratified: {res_level}")
married = df[df.husband_edu != "no current partner"].copy()
married["husband_edu"] = married["husband_edu"].cat.remove_categories(["no current partner"])
married["decisions"] = married["decisions"].cat.remove_categories(["no current partner"])
rows += fit_model(married, FULL + ["decisions"], "M4 currently married + decision-making")
rows += fit_model(df.assign(wealth=df["wealth_within_residence"]), FULL,
                  "M5 sensitivity: wealth quintile within urban/rural (v190a)")
reg = pd.DataFrame(rows)
reg["design"] = "survey-weighted, cluster-robust" if WEIGHTED else "unweighted, model-based SE"
save_table(reg, f"T10_mnlogit_K{K}", floatfmt="%.4f")

save_table(pd.concat([gvif(df, SES).assign(model="M1"), gvif(df, FULL).assign(model="M2")]),
           f"T11_gvif_K{K}", floatfmt="%.3f")

# ------------------------------------------------------------------ bias-adjusted 3-step (BCH)
# Implemented directly, because StepMix's three-step BCH ignores sample_weight (verified).
# Step 1: unweighted polytomous LCA (as in 02_mixture.py). Step 2: modal assignment.
# Step 3: weighted multinomial logit on the K-fold expanded data with BCH weights
#   w_i * H[s_i, t], H = D^-1, D[t, s] = P(modal = s | true class t) estimated with survey weights
#   (Bolck, Croon & Hagenaars, 2004; Vermunt, 2010). SEs/CIs: stratified PSU bootstrap that
#   refits the measurement model in every replicate.
bch_covs = SES + ["age_group", "parity", "division", "religion", "working", "husband_edu", "internet_12m",
                  "owns_mobile"]
dd = df.dropna(subset=bch_covs)
Zd, znames, _ = design(dd, bch_covs)
Xi = dd[ITEMS].astype(int).reset_index(drop=True)
W_bch = (dd["weight"] / dd["weight"].mean()).to_numpy() if WEIGHTED else np.ones(len(dd))
PSU = dd["v021"].to_numpy() if WEIGHTED else np.arange(len(dd))
STR = dd["v023"].astype(str).to_numpy() if WEIGHTED else np.zeros(len(dd))


def fit_lca(X, seed, n_init, tol):
    return StepMix(n_components=K, measurement="categorical", n_init=n_init, max_iter=2000, abs_tol=tol,
                   random_state=seed, progress_bar=0, verbose=0).fit(X)


def item_means(m):
    pis = m.get_parameters()["measurement"]["pis"].reshape(K, len(ITEMS), 4)
    return (pis * np.arange(4)).sum(-1)


def bch_fit(post, Zd_, w):
    modal = post.argmax(1)
    D = np.array([[np.sum(w * post[:, t] * (modal == s)) for s in range(K)] for t in range(K)])
    D /= D.sum(1, keepdims=True)
    H = np.linalg.inv(D)
    n = len(post)
    y_long = np.repeat(np.arange(K)[None, :], n, 0).ravel()
    X_long = np.repeat(Zd_, K, axis=0)
    w_long = (w[:, None] * H[modal, :]).ravel()
    B, _, res = mlogit(y_long, X_long, w=w_long, robust_inverse=True)
    return B[1:], D  # drop intercept row -> (p, K-1)


def classes_ordered(m, X, ref=None):
    mu = item_means(m)
    if ref is None:
        o = np.argsort(mu.mean(1))
    else:
        _, o = linear_sum_assignment(((ref[:, None, :] - mu[None, :, :]) ** 2).sum(-1))
    return m.predict_proba(X)[:, o], mu[o]


m_main = fit_lca(Xi, 7, n_init=10, tol=1e-8)
post0, MU0 = classes_ordered(m_main, Xi)
B0, D0 = bch_fit(post0, Zd, W_bch)
save_table(pd.DataFrame(D0, index=[f"true C{t + 1}" for t in range(K)],
                        columns=[f"assigned C{s + 1}" for s in range(K)]), f"S12_classification_error_K{K}",
           index=True)
print("BCH main fit done; diagonal of D:", np.round(np.diag(D0), 3), flush=True)

strata_psus = {h: np.unique(PSU[STR == h]) for h in np.unique(STR)}
rows_of_psu = pd.Series(np.arange(len(PSU))).groupby(PSU).apply(np.asarray).to_dict()


def boot_bch(seed):
    """Stratified cluster bootstrap: resample PSUs with replacement within each stratum."""
    rng = np.random.default_rng(seed)
    idx = np.concatenate([rows_of_psu[c] for h, ps in strata_psus.items()
                          for c in rng.choice(ps, len(ps), replace=True)])
    try:
        Xb = Xi.iloc[idx].reset_index(drop=True)
        mb = fit_lca(Xb, seed, n_init=3, tol=1e-6)
        pb, _ = classes_ordered(mb, Xb, ref=MU0)
        return bch_fit(pb, Zd[idx], W_bch[idx])[0]
    except Exception:
        return None


Bb = [b for b in Parallel(n_jobs=N_JOBS)(delayed(boot_bch)(9000 + s) for s in range(N_BOOT)) if b is not None]
Bb = np.array(Bb)
brows = []
for j in range(K - 1):
    for i, nm in enumerate(znames[1:]):
        b = Bb[:, i, j]
        # percentile CI and percentile-based two-sided p, so that CI and p always agree
        p_boot = min(1.0, 2 * min((b <= 0).mean(), (b >= 0).mean()) + 1 / (len(b) + 1))
        brows.append({"model": "BCH three-step (survey-weighted; stratified PSU bootstrap)",
                      "contrast": f"C{j + 2} vs C1", "term": nm, "RRR": np.exp(B0[i, j]),
                      "ci_low": np.exp(np.quantile(b, .025)), "ci_high": np.exp(np.quantile(b, .975)),
                      "p": p_boot, "boot_se_log": b.std(ddof=1), "boot_mean_RRR": np.exp(b.mean()),
                      "n": len(Xi), "n_boot_ok": len(Bb)})
save_table(pd.DataFrame(brows), f"T12_bch_K{K}", floatfmt="%.4f")


show = reg[reg.model.isin(["M1 education + wealth + residence", "M2 fully adjusted"])
           & reg.term.str.match("education|wealth|residence")]
print(show.round(3).to_string(index=False))
print(pd.DataFrame(brows).query("term.str.match('education|wealth|residence')", engine="python")
      .round(3).to_string(index=False))
print(gvif(df, FULL).round(2).to_string(index=False))
