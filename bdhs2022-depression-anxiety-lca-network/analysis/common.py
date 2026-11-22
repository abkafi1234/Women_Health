"""Shared helpers for the analyses.

Corrections relative to the originally submitted analysis:
  * Item labels follow the BDHS variable labels: mth1-mth7 = GAD-7, mth8-mth16 = PHQ-9
    (the original script labelled them the other way round).
  * The analytic sample is defined by complete PHQ-9/GAD-7 responses, not by listwise
    deletion over 90 variables (which dropped women without a smartphone, women who never
    moved, and women without a current husband - all structurally "not applicable").
  * Polychoric correlations + EBICglasso replace a fixed-alpha glasso on z-scores.
"""
from __future__ import annotations

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, special, stats
from sklearn.covariance import graphical_lasso

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parents[1]          # repository root
REV = ROOT
# Results and LaTeX tables can be redirected (e.g. for a clean reproducibility re-run)
OUT = Path(os.environ.get("SMTS_OUT", ROOT / "results"))
MANU = Path(os.environ.get("SMTS_MANUSCRIPT", ROOT / "results" / "latex"))
TAB = OUT / "tables"
FIG = OUT / "figures"
for d in (TAB, FIG):
    d.mkdir(parents=True, exist_ok=True)

# Optional: full BDHS 2022 individual recode (adds weights, clusters, strata, age, parity,
# marital status). If present it is used instead of the three position-aligned extracts.
FULL_IR_CANDIDATES = [ROOT / "data" / "BDIR81FL.DTA", ROOT / "data" / "BDIR81FL.dta"]
# The 119 variables of the three extracts used in the original submission (mental-health module,
# husband/work/empowerment, woman's characteristics). Listed explicitly so that the pipeline runs from the
# official DHS file alone; also used to replicate the original listwise-deletion sample.
EXTRACT_VARS = [
    "mth1", "mth2", "mth3", "mth4", "mth5", "mth6", "mth7", "mth8", "mth9", "mth10", "mth11", "mth12", "mth13",
    "mth14", "mth15", "mth16", "mth17", "mth19", "mth20", "mth21", "mth22", "mth23", "mth24",
    "v701", "v702", "v704", "v704a", "v705", "v714", "v714a", "v715", "v716", "v717", "v719", "v721", "v729",
    "v730", "v731", "v732", "v739", "v740", "v741", "v743a", "v743b", "v743c", "v743d", "v743e", "v743f",
    "v744a", "v744b", "v744c", "v744d", "v744e", "v745a", "v745b", "v745c", "v745d", "v746",
    "v101", "v102", "v103", "v104", "v105", "v106", "v107", "v113", "v115", "v116", "v119", "v120", "v121",
    "v122", "v123", "v124", "v125", "v127", "v128", "v129", "v130", "v131", "v133", "v134", "v135", "v136",
    "v137", "v138", "v139", "v140", "v141", "v149", "v150", "v151", "v152", "v153", "v155", "v156", "v157",
    "v158", "v159", "v160", "v161", "v166", "v167", "v168", "v169a", "v169b", "v169c", "v170", "v171a",
    "v171b", "v172", "v173", "v174m", "v174y", "v175", "v176", "v177", "v190a", "v191a"]
IR_EXTRA = ["caseid", "v001", "v002", "v003", "v005", "v021", "v022", "v023", "v012", "v013", "v201",
            "v501", "v502", "v024", "v025", "v190", "seligbm"]

# ---------------------------------------------------------------- items (BDHS labels)
ITEMS = [f"mth{i}" for i in range(1, 17)]
ITEM_INFO = pd.DataFrame(
    [
        ("mth1", "GAD-1", "GAD", "Nervous", "Feeling nervous, anxious or on edge"),
        ("mth2", "GAD-2", "GAD", "Uncontrolled worry", "Not able to stop or control worrying"),
        ("mth3", "GAD-3", "GAD", "Excessive worry", "Worrying too much about different things"),
        ("mth4", "GAD-4", "GAD", "Trouble relaxing", "Trouble relaxing"),
        ("mth5", "GAD-5", "GAD", "Restlessness", "Being so restless that it is hard to sit still"),
        ("mth6", "GAD-6", "GAD", "Irritability", "Becoming easily annoyed or irritable"),
        ("mth7", "GAD-7", "GAD", "Fear", "Feeling afraid as if something awful might happen"),
        ("mth8", "PHQ-1", "PHQ", "Anhedonia", "Little interest or pleasure in doing things"),
        ("mth9", "PHQ-2", "PHQ", "Sad mood", "Feeling down, depressed or hopeless"),
        ("mth10", "PHQ-3", "PHQ", "Sleep", "Trouble falling/staying asleep, or sleeping too much"),
        ("mth11", "PHQ-4", "PHQ", "Fatigue", "Feeling tired or having little energy"),
        ("mth12", "PHQ-5", "PHQ", "Appetite", "Poor appetite or overeating"),
        ("mth13", "PHQ-6", "PHQ", "Worthlessness", "Feeling bad about yourself / a failure"),
        ("mth14", "PHQ-7", "PHQ", "Concentration", "Trouble concentrating"),
        ("mth15", "PHQ-8", "PHQ", "Psychomotor", "Moving/speaking slowly or being fidgety"),
        ("mth16", "PHQ-9", "PHQ", "Suicidal ideation", "Thoughts of being better off dead / self-harm"),
    ],
    columns=["var", "item", "scale", "label", "wording"],
)
LABELS = dict(zip(ITEM_INFO["var"], ITEM_INFO["label"]))
SHORT = dict(zip(ITEM_INFO["var"], ITEM_INFO["item"]))
GAD = ITEMS[:7]
PHQ = ITEMS[7:]
SCALE_MAP = {"never": 0, "rarely": 1, "often": 2, "always": 3}  # BDHS 2022 value labels of mth1-mth16
RESPONSE_NOTE = "0 = never, 1 = rarely, 2 = often, 3 = always (BDHS response labels)"


# ---------------------------------------------------------------- data loading
def _read(path: Path) -> pd.DataFrame:
    return pd.read_stata(path, convert_categoricals=True)


def load_raw() -> tuple[pd.DataFrame, str]:
    """Return the merged raw data and a note describing the source."""
    for p in FULL_IR_CANDIDATES:
        if p.exists():
            # read only the variables used (the IR file has >5,000 variables)
            return (pd.read_stata(p, columns=list(dict.fromkeys(EXTRACT_VARS + IR_EXTRA)), convert_categoricals=True),
                    f"full BDHS 2022 IR file {p.name} (weights v005, PSU v021, strata v023 available)")
    raise FileNotFoundError(
        "BDHS 2022 individual recode not found. Download BDIR81FL.DTA from the DHS Program "
        "(https://dhsprogram.com, registration required) and place it in data/ (see data/README.md).")


def code_items(df: pd.DataFrame) -> pd.DataFrame:
    """Map item responses to 0-3; 'don't know'/'refused'/missing -> NaN."""
    out = pd.DataFrame(index=df.index)
    for c in ITEMS:
        s = df[c].astype("object").map(lambda v: SCALE_MAP.get(str(v).strip().lower(), np.nan)
                                       if not isinstance(v, (int, float, np.integer, np.floating))
                                       else (v if v in (0, 1, 2, 3) else np.nan))
        out[c] = pd.to_numeric(s, errors="coerce")
    return out


def has_design(df: pd.DataFrame) -> bool:
    return all(v in df.columns for v in ("v005", "v021", "v023"))


# ---------------------------------------------------------------- polychoric correlation
def bvn_cdf(h, k, r):
    """Vectorised standard bivariate normal CDF P(X<=h, Y<=k) via Owen's T function."""
    h = np.asarray(h, float)
    k = np.asarray(k, float)
    h, k = np.broadcast_arrays(h, k)
    h = np.where(np.abs(h) < 1e-10, 1e-10, h)
    k = np.where(np.abs(k) < 1e-10, 1e-10, k)
    s = np.sqrt(1.0 - r * r)
    ah = (k - r * h) / (h * s)
    ak = (h - r * k) / (k * s)
    beta = np.where(h * k > 0, 0.0, 0.5)
    val = 0.5 * (special.ndtr(h) + special.ndtr(k)) - special.owens_t(h, ah) - special.owens_t(k, ak) - beta
    return np.clip(val, 0.0, 1.0)


def _thresholds(x: np.ndarray, ncat: int) -> np.ndarray:
    counts = np.bincount(x, minlength=ncat).astype(float)
    cum = np.cumsum(counts)[:-1] / counts.sum()
    cum = np.clip(cum, 1e-6, 1 - 1e-6)
    t = stats.norm.ppf(cum)
    return np.concatenate([[-8.0], t, [8.0]])


def polychoric_pair(x: np.ndarray, y: np.ndarray, ncat: int = 4) -> float:
    """Two-step ML polychoric correlation for two integer-coded ordinal variables."""
    tx, ty = _thresholds(x, ncat), _thresholds(y, ncat)
    tab = np.zeros((ncat, ncat))
    np.add.at(tab, (x, y), 1)
    H, K = np.meshgrid(tx, ty, indexing="ij")

    def nll(r):
        F = bvn_cdf(H, K, r)
        P = F[1:, 1:] - F[:-1, 1:] - F[1:, :-1] + F[:-1, :-1]
        return -np.sum(tab * np.log(np.clip(P, 1e-300, None)))

    res = optimize.minimize_scalar(nll, bounds=(-0.995, 0.995), method="bounded",
                                   options={"xatol": 1e-6})
    return float(res.x)


def nearest_pd(R: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    w, V = np.linalg.eigh((R + R.T) / 2)
    if w.min() >= eps:
        return R
    w = np.clip(w, eps, None)
    R2 = V @ np.diag(w) @ V.T
    d = np.sqrt(np.diag(R2))
    return R2 / np.outer(d, d)


def polychoric_matrix(X: np.ndarray, ncat: int = 4) -> np.ndarray:
    X = np.asarray(X, int)
    p = X.shape[1]
    R = np.eye(p)
    for i in range(p):
        for j in range(i + 1, p):
            R[i, j] = R[j, i] = polychoric_pair(X[:, i], X[:, j], ncat)
    return nearest_pd(R)


def cor_matrix(X: np.ndarray, method: str = "polychoric") -> np.ndarray:
    if method == "polychoric":
        return polychoric_matrix(X)
    if method == "spearman":
        return stats.spearmanr(X).statistic
    if method == "pearson":
        return np.corrcoef(X, rowvar=False)
    raise ValueError(method)


# ---------------------------------------------------------------- EBICglasso (qgraph-style)
def ebic_glasso(S: np.ndarray, n: int, gamma: float = 0.5, nlambda: int = 100,
                lambda_min_ratio: float = 0.01, return_path: bool = False):
    """Replicates qgraph::EBICglasso defaults (gamma=0.5, 100 lambdas, min ratio 0.01,
    unpenalised diagonal, no refit/threshold). Returns the partial-correlation matrix."""
    p = S.shape[0]
    lam_max = np.max(np.abs(S - np.diag(np.diag(S))))
    lambdas = np.exp(np.linspace(np.log(lam_max * lambda_min_ratio), np.log(lam_max), nlambda))
    path = []
    best = None
    for lam in lambdas:
        try:
            # tol 1e-5 gives the same selected lambda and edges within 2e-4 of tol 1e-6, 6x faster
            _, K = graphical_lasso(S, alpha=lam, max_iter=200, tol=1e-5, enet_tol=1e-5, mode="cd")
        except FloatingPointError:
            continue
        ll = n / 2 * (np.linalg.slogdet(K)[1] - np.trace(S @ K))
        E = int(np.sum(np.abs(np.triu(K, 1)) > 1e-8))
        ebic = -2 * ll + E * np.log(n) + 4 * E * gamma * np.log(p)
        path.append((lam, ebic, E))
        if best is None or ebic < best[0]:
            best = (ebic, lam, K)
    _, lam, K = best
    d = np.sqrt(np.diag(K))
    W = -K / np.outer(d, d)
    np.fill_diagonal(W, 0.0)
    W[np.abs(W) < 1e-8] = 0.0
    if return_path:
        return W, lam, pd.DataFrame(path, columns=["lambda", "EBIC", "n_edges"])
    return W


def strength(W):
    return np.abs(W).sum(axis=0)


def expected_influence(W):
    return W.sum(axis=0)


def bridge_strength(W, communities):
    communities = np.asarray(communities)
    mask = communities[:, None] != communities[None, :]
    return np.abs(W * mask).sum(axis=0)


def estimate_network(X: np.ndarray, method: str = "polychoric", gamma: float = 0.5) -> np.ndarray:
    return ebic_glasso(cor_matrix(X, method), X.shape[0], gamma=gamma)


# ---------------------------------------------------------------- misc
def cronbach_alpha(X: np.ndarray) -> float:
    X = np.asarray(X, float)
    k = X.shape[1]
    return k / (k - 1) * (1 - X.var(axis=0, ddof=1).sum() / X.sum(axis=1).var(ddof=1))


def ordinal_alpha(R: np.ndarray) -> float:
    k = R.shape[0]
    rbar = (R.sum() - k) / (k * (k - 1))
    return k * rbar / (1 + (k - 1) * rbar)


def omega_total(R: np.ndarray) -> float:
    """McDonald's omega (total) from a one-factor solution on R (principal axis, iterated)."""
    k = R.shape[0]
    h2 = 1 - 1 / np.diag(np.linalg.inv(R))
    for _ in range(200):
        Rr = R.copy()
        np.fill_diagonal(Rr, h2)
        w, V = np.linalg.eigh(Rr)
        lam = V[:, -1] * np.sqrt(max(w[-1], 1e-12))
        new = np.clip(lam ** 2, 1e-4, 0.999)
        if np.max(np.abs(new - h2)) < 1e-7:
            break
        h2 = new
    lam = np.abs(lam)
    return lam.sum() ** 2 / (lam.sum() ** 2 + (1 - lam ** 2).sum())


def save_table(df: pd.DataFrame, name: str, index: bool = False, floatfmt: str = "%.3f"):
    df.to_csv(TAB / f"{name}.csv", index=index, float_format=floatfmt)
    return df
