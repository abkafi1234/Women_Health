"""Build the analytic dataset, sample-flow table, codebook, item descriptives and reliability."""
import numpy as np
import pandas as pd
from scipy import stats

from common import (EXTRACT_VARS, GAD, ITEMS, ITEM_INFO, LABELS, OUT, PHQ, SHORT, TAB, code_items, cronbach_alpha,
                    has_design, load_raw, omega_total, ordinal_alpha, polychoric_matrix, save_table)

raw, source = load_raw()
items = code_items(raw)

# ------------------------------------------------------------------ sample flow
answered_any = raw[ITEMS].notna().any(axis=1)
complete = items.notna().all(axis=1)
flow = [
    ("BDHS 2022 individual records in extract", len(raw)),
    ("Not administered the mental-health module (short-questionnaire households; all 16 items missing)", int((~answered_any).sum())),
    ("Administered the mental-health module", int(answered_any.sum())),
    ("Excluded: 'don't know' / 'refused' / missing on >=1 of the 16 items",
     int((answered_any & ~complete).sum())),
    ("Analytic sample (complete PHQ-9 and GAD-7)", int(complete.sum())),
]
save_table(pd.DataFrame(flow, columns=["step", "n"]), "S1_sample_flow")

d = raw.loc[complete].copy()
X = items.loc[complete]

# ------------------------------------------------------------------ covariates
def cat(s, mapping=None, order=None):
    s = s.astype("object").astype(str).str.strip().str.lower()
    s = s.replace({"nan": np.nan, "none": np.nan})
    if mapping:
        s = s.map(lambda v: mapping.get(v, v) if isinstance(v, str) else v)
    return pd.Categorical(s, categories=order) if order else s


cov = pd.DataFrame(index=d.index)
cov["residence"] = cat(d["v102"], order=["urban", "rural"])
cov["division"] = cat(d["v101"], order=["dhaka", "barishal", "chattogram", "khulna", "mymensingh",
                                         "rajshahi", "rangpur", "sylhet"])
cov["education"] = cat(d["v106"], order=["no education", "primary", "secondary", "higher"])
wq = ["poorest", "poorer", "middle", "richer", "richest"]
# national (combined) wealth index when available; v190a is computed separately within urban and rural
cov["wealth"] = cat(d["v190"] if "v190" in d else d["v190a"], order=wq)
cov["wealth_within_residence"] = cat(d["v190a"], order=wq)
if "v012" in d:
    age = pd.to_numeric(d["v012"], errors="coerce")
    cov["age"] = age
    cov["age_group"] = pd.cut(age, [14, 24, 34, 49], labels=["15-24", "25-34", "35-49"])
if "v201" in d:
    par = pd.to_numeric(d["v201"], errors="coerce")
    cov["parity"] = pd.cut(par, [-1, 0, 2, 99], labels=["0", "1-2", "3+"])
if "v501" in d:
    cov["marital"] = pd.Categorical(np.where(cat(d["v501"]) == "married", "currently married",
                                             "formerly married"),
                                    categories=["currently married", "formerly married"])
cov["religion"] = pd.Categorical(np.where(cat(d["v130"]) == "islam", "muslim", "non-muslim"),
                                 categories=["muslim", "non-muslim"])
cov["working"] = cat(d["v714"], order=["no", "yes"])
hed = cat(d["v701"]).copy()
hed = pd.Series(hed, index=d.index).fillna("no current partner").replace({"don't know": np.nan})
cov["husband_edu"] = pd.Categorical(hed, categories=["no education", "primary", "secondary", "higher",
                                                     "no current partner"])
cov["hh_head_female"] = pd.Categorical(np.where(cat(d["v151"]) == "female", "yes", "no"),
                                       categories=["no", "yes"])
cov["owns_mobile"] = cat(d["v169a"], order=["no", "yes"])
net = cat(d["v171a"])
cov["internet_12m"] = pd.Categorical(np.where(net == "yes, last 12 months", "yes", "no"),
                                     categories=["no", "yes"])
cov["hh_size"] = pd.to_numeric(d["v136"], errors="coerce")
cov["children_u5"] = pd.to_numeric(d["v137"], errors="coerce")

# decision-making: number of 3 decisions (own health care, large purchases, family visits)
# taken alone or jointly; women without a current partner get their own category
dm_cols = ["v743a", "v743b", "v743d"]
part = d[dm_cols].apply(lambda s: cat(s).isin(["respondent alone", "respondent and husband/partner"]))
no_partner = d["v743a"].isna()
dm = part.sum(axis=1).astype(object)
dm[no_partner] = "no current partner"
dm = dm.map(lambda v: v if isinstance(v, str) else ("0" if v == 0 else ("1-2" if v < 3 else "3")))
cov["decisions"] = pd.Categorical(dm, categories=["0", "1-2", "3", "no current partner"])
# attitude: wife-beating justified for >=1 of 5 reasons ("don't know" treated as no)
wb = d[["v744a", "v744b", "v744c", "v744d", "v744e"]].apply(lambda s: cat(s) == "yes").any(axis=1)
cov["wifebeating_justified"] = pd.Categorical(np.where(wb, "yes", "no"), categories=["no", "yes"])

design_cols = [c for c in ["v001", "v002", "v003", "v005", "v021", "v022", "v023", "v012", "v013",
                           "v201", "v501", "v024", "v025", "v190"] if c in d.columns]
analytic = pd.concat([X.astype(int), cov, d[design_cols]], axis=1)
if "v005" in analytic:
    analytic["weight"] = analytic["v005"] / 1e6
analytic["GAD7"] = analytic[GAD].sum(axis=1)
analytic["PHQ9"] = analytic[PHQ].sum(axis=1)
analytic.to_pickle(OUT / "analytic.pkl")

# ------------------------------------------------------------------ the original sample, for comparison
raw_x = raw[[c for c in EXTRACT_VARS if c in raw.columns]]  # exactly the original 119 variables
miss = raw_x.isnull().mean() * 100
orig = raw_x.drop(columns=miss[miss >= 40].index).dropna()
orig_idx = orig.index.intersection(analytic.index)
analytic.assign(in_original_sample=analytic.index.isin(orig_idx)).loc[:, ["in_original_sample"]] \
    .to_pickle(OUT / "original_sample_flag.pkl")

comp = []
inorig = analytic.index.isin(orig_idx)
for v in ["residence", "education", "wealth", "owns_mobile", "husband_edu"]:
    for lev in analytic[v].cat.categories:
        a = (analytic.loc[inorig, v] == lev).mean() * 100
        b = (analytic.loc[~inorig, v] == lev).mean() * 100
        comp.append((v, lev, a, b))
comp = pd.DataFrame(comp, columns=["variable", "level", "% retained in original listwise sample",
                                   "% dropped by original listwise deletion"])
comp.loc[len(comp)] = ["PHQ-9 sum (mean)", "", analytic.loc[inorig, "PHQ9"].mean(),
                       analytic.loc[~inorig, "PHQ9"].mean()]
comp.loc[len(comp)] = ["GAD-7 sum (mean)", "", analytic.loc[inorig, "GAD7"].mean(),
                       analytic.loc[~inorig, "GAD7"].mean()]
comp.loc[len(comp)] = ["n", "", inorig.sum(), (~inorig).sum()]
save_table(comp, "S2_original_listwise_vs_dropped", floatfmt="%.1f")

# ------------------------------------------------------------------ item descriptives
rows = []
for _, r in ITEM_INFO.iterrows():
    x = analytic[r["var"]]
    p = x.value_counts(normalize=True).reindex([0, 1, 2, 3], fill_value=0) * 100
    rows.append([r["item"], r["label"], r["wording"], x.mean(), x.std(), stats.skew(x),
                 stats.kurtosis(x), *p.values])
desc = pd.DataFrame(rows, columns=["item", "label", "wording", "mean", "sd", "skew", "excess_kurtosis",
                                   "% 0 (never)", "% 1", "% 2", "% 3"])
save_table(desc, "T1_item_descriptives")

# ------------------------------------------------------------------ covariate descriptives
crows = []
wts = analytic["weight"] if "weight" in analytic else pd.Series(1.0, index=analytic.index)
for v in [c for c in cov.columns if str(cov[c].dtype) == "category"]:
    vc = analytic[v].value_counts(dropna=False, sort=False)
    for lev, n in vc.items():
        mask = analytic[v].isna() if pd.isna(lev) else analytic[v] == lev
        crows.append((v, "missing" if pd.isna(lev) else lev, n, n / len(analytic) * 100,
                      wts[mask].sum() / wts.sum() * 100))
for v in [c for c in ["age", "hh_size", "children_u5"] if c in analytic]:
    m = np.average(analytic[v], weights=wts)
    crows.append((v, "mean (SD)", f"{analytic[v].mean():.2f} ({analytic[v].std():.2f})", np.nan, m))
save_table(pd.DataFrame(crows, columns=["variable", "level", "n", "% unweighted", "% weighted (mean if numeric)"]),
           "T2_covariate_descriptives", floatfmt="%.1f")
if "weight" in analytic:
    wsum = pd.DataFrame({"item": [SHORT[c] for c in ITEMS],
                         "weighted_mean": [np.average(analytic[c], weights=wts) for c in ITEMS]})
    save_table(wsum, "S10_item_weighted_means")

# ------------------------------------------------------------------ reliability
rel = []
for name, cols in [("GAD-7", GAD), ("PHQ-9", PHQ), ("All 16 items", ITEMS)]:
    Xi = analytic[cols].to_numpy()
    R = polychoric_matrix(Xi)
    rel.append((name, len(cols), cronbach_alpha(Xi), ordinal_alpha(R), omega_total(R)))
save_table(pd.DataFrame(rel, columns=["scale", "k", "cronbach_alpha", "ordinal_alpha",
                                      "omega_total_polychoric"]), "T3_reliability")

with open(OUT / "data_note.txt", "w") as f:
    f.write(f"Source: {source}\nSurvey design variables present: {has_design(raw)}\n")
    f.write(f"Analytic N = {len(analytic)}; original listwise sample within it = {inorig.sum()}\n")
print(open(OUT / "data_note.txt").read())
print(pd.DataFrame(flow).to_string(index=False, header=False))
print(comp.to_string(index=False))
print(desc[["item", "label", "mean", "sd", "skew", "% 0 (never)", "% 3"]].round(2).to_string(index=False))
print(pd.read_csv(TAB / "T3_reliability.csv").round(3).to_string(index=False))
