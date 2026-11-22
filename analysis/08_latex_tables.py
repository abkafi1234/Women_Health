"""Generate LaTeX (booktabs) tables for the manuscript and supplement from output/tables/*.csv."""
import numpy as np
import pandas as pd

from common import MANU, OUT, TAB

OUTD = MANU / "tables"
OUTD.mkdir(parents=True, exist_ok=True)


def esc(s):
    return (str(s).replace("&", r"\&").replace("%", r"\%").replace("_", r"\_").replace("#", r"\#")
            .replace(">=", r"$\geq$").replace("<=", r"$\leq$"))


def f(x, d=2):
    return "" if pd.isna(x) else f"{x:,.{d}f}"


def write(name, body, caption, label, colspec, header, notes=None, size=r"\small", env="table"):
    lines = [rf"\begin{{{env}}}[htbp]", r"\centering", size, rf"\caption{{{caption}}}", rf"\label{{{label}}}",
             rf"\begin{{tabular}}{{{colspec}}}", r"\toprule", header + r" \\", r"\midrule"]
    rules = ("\\midrule", "\\toprule", "\\bottomrule", "\\addlinespace")
    lines += [r if r.strip().startswith(rules) else r + r" \\" for r in body]
    lines += [r"\bottomrule", r"\end{tabular}"]
    if notes:
        lines += [r"\par\smallskip", rf"\parbox{{\linewidth}}{{\footnotesize\textit{{Note.}} {notes}}}"]
    lines += [rf"\end{{{env}}}"]
    (OUTD / f"{name}.tex").write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------- Table 1: items
d = pd.read_csv(TAB / "T1_item_descriptives.csv")
body = [f"{r["item"]} & {esc(r.label)} & {f(r["mean"])} & {f(r.sd)} & {f(r["skew"])} & {f(r['% 0 (never)'], 1)} & "
        f"{f(r['% 2'] + r['% 3'], 1)}" for _, r in d.iterrows()]
write("tab1_items", body, "PHQ-9 and GAD-7 item distributions in the analytic sample ($N = 19{,}930$)", "tab:items",
      "llrrrrr", r"Item & Symptom & $M$ & $SD$ & Skew & \% 0 & \% $\geq$2",
      "Items refer to the past two weeks and are scored 0 (never), 1 (rarely), 2 (often) and 3 (always), "
      "following the BDHS response labels. \\% 0 = never; \\% $\\geq$2 = often or always. Unweighted. Full wording in Table~\\Sref{tab:S11}.")

# ---------------------------------------------------------------- Table 2: sample characteristics
c = pd.read_csv(TAB / "T2_covariate_descriptives.csv")
keep = {"age_group": "Age group (years)", "parity": "Children ever born", "marital": "Marital status",
        "education": "Education", "husband_edu": "Husband's education", "wealth": "Wealth quintile (national)",
        "residence": "Residence", "division": "Division", "religion": "Religion", "working": "Currently working",
        "internet_12m": "Internet use, past 12 months", "owns_mobile": "Owns a mobile phone"}
body = []
for v, lab in keep.items():
    s = c[c.variable == v]
    body.append(rf"\multicolumn{{3}}{{l}}{{\textit{{{esc(lab)}}}}}")
    for _, r in s.iterrows():
        body.append(rf"\quad {esc(r.level)} & {int(float(r.n)):,} & {f(r['% weighted (mean if numeric)'], 1)}")
age = c[c.variable == "age"].iloc[0]
body.append(rf"Age, years: weighted mean & & {f(age['% weighted (mean if numeric)'], 1)}")
write("tab2_sample", body, "Characteristics of the analytic sample ($N = 19{,}930$)", "tab:sample", "lrr",
      r"Characteristic & $n$ & Weighted \%", "Percentages use the BDHS women's sampling weight (v005). "
      "``No current partner'' denotes widowed, divorced or separated women.", size=r"\footnotesize")

# ---------------------------------------------------------------- Table 3: LCA fit
fit = pd.read_csv(TAB / "T4_mixture_fit_indices.csv")
lca = fit[fit.model == "LCA"]
body = []
for _, r in lca.iterrows():
    blrt = "" if pd.isna(r.get("BLRT_p")) else ("$<$.01" if r.BLRT_p < 0.01 else f"{r.BLRT_p:.2f}".lstrip("0"))
    body.append(f"{int(r.K)} & {f(r.LL, 1)} & {int(r.npar)} & {f(r.BIC, 1)} & {f(r.SABIC, 1)} & {f(r.CAIC, 1)} & "
                f"{f(r.entropy)} & {f(r['smallest_class_%'], 1)} & {f(r.min_AvePP)} & "
                f"{f(r['best_LL_replicated_%'], 0)} & {blrt}")
write("tab3_fit", body, "Fit and classification indices of polytomous latent class models with 1--7 classes",
      "tab:fit", "rrrrrrrrrrr",
      r"$K$ & LL & $p$ & BIC & SABIC & CAIC & Entropy & Min.\ \% & Min.\ AvePP & Rep.\ \% & BLRT $p$",
      "LL = log-likelihood (total, not per observation); $p$ = number of free parameters; SABIC = sample-size-"
      "adjusted BIC; Min.\\ \\% = smallest class (modal assignment); Min.\\ AvePP = lowest class-specific average "
      "posterior probability; Rep.\\ \\% = percentage of 20 random starts replicating the best LL; BLRT = bootstrap "
      "likelihood-ratio test of $K-1$ vs $K$ classes (100 replicates). Unweighted; $N = 19{,}930$. Gaussian latent "
      "profile models are reported in Table~\\Sref{tab:S5}.", size=r"\scriptsize")

# ---------------------------------------------------------------- Table 4: class item means (K=5)
t = pd.read_csv(TAB / "S8_lca_K5_class_item_means.csv")
prev = pd.read_csv(TAB / "T13_class_prevalence_K5.csv")
sizes = t.iloc[-1]
t = t.iloc[:-1]
body = [f"{r['item']} & {esc(r['label'])} & " + " & ".join(f"{f(r[f'C{c} mean'])} ({f(r[f'C{c} SD'])})"
                                                          for c in range(1, 6)) for _, r in t.iterrows()]
body.append(r"\midrule")
body.append("& Class size, \\% (weighted) & " + " & ".join(f(x, 1) for x in prev["%_posterior_weighted"]))
names = ["Minimal", "Mild,\\newline somatic-pred.", "Mild,\\newline anxiety-pred.", "Moderate,\\newline generalised",
         "High,\\newline generalised"]
write("tab4_classes", body,
      "Model-implied item means (SD) by latent class, five-class solution", "tab:classes",
      "ll" + "p{1.75cm}" * 5, "Item & Symptom & " + " & ".join(rf"C{i + 1}: {n}" for i, n in enumerate(names)),
      "Expected item scores (0 = never to 3 = always) and standard deviations implied by the class-specific response probabilities. "
      "Class sizes are posterior-probability weighted with survey weights. ", size=r"\scriptsize")

# ---------------------------------------------------------------- Table 5: centrality
ce = pd.read_csv(TAB / "T6_centrality.csv")
body = [f"{r['item']} & {esc(r['label'])} & {f(r.strength)} [{f(r.strength_ci_low)}, {f(r.strength_ci_high)}] & "
        f"{f(r.expected_influence)} [{f(r.expected_influence_ci_low)}, {f(r.expected_influence_ci_high)}] & "
        f"{f(r.bridge_strength)} [{f(r.bridge_strength_ci_low)}, {f(r.bridge_strength_ci_high)}]"
        for _, r in ce.iterrows()]
write("tab5_centrality", body, "Node centrality in the regularised partial-correlation network", "tab:centrality",
      "llccc", r"Item & Symptom & Strength [95\% CI] & Expected influence [95\% CI] & Bridge strength [95\% CI]",
      "Polychoric EBICglasso network ($\\gamma = 0.5$), $N = 19{,}930$. Intervals are 2.5th--97.5th percentiles of "
      "1,000 nonparametric bootstrap estimates. Bridge strength uses the a priori PHQ-9 and GAD-7 communities. "
      "Ordered by strength.", size=r"\scriptsize")

# ---------------------------------------------------------------- Table 6: correlates (BCH, K=5)
b = pd.read_csv(TAB / "T12_bch_K5.csv")
m = pd.read_csv(TAB / "T10_mnlogit_K5.csv")
termlab = {
    "age_group=25-34": ("Age (ref.\\ 15--24)", "25--34"), "age_group=35-49": ("", "35--49"),
    "parity=1-2": ("Children ever born (ref.\\ 0)", "1--2"), "parity=3+": ("", "3+"),
    "education=primary": ("Education (ref.\\ none)", "Primary"), "education=secondary": ("", "Secondary"),
    "education=higher": ("", "Higher"),
    "husband_edu=primary": ("Husband's education (ref.\\ none)", "Primary"),
    "husband_edu=secondary": ("", "Secondary"), "husband_edu=higher": ("", "Higher"),
    "husband_edu=no current partner": ("", "No current partner"),
    "wealth=poorer": ("Wealth (ref.\\ poorest)", "Poorer"), "wealth=middle": ("", "Middle"),
    "wealth=richer": ("", "Richer"), "wealth=richest": ("", "Richest"),
    "residence=rural": ("Residence (ref.\\ urban)", "Rural"),
    "religion=non-muslim": ("Religion (ref.\\ Muslim)", "Non-Muslim"),
    "working=yes": ("Currently working", "Yes"), "internet_12m=yes": ("Internet use, 12 mo", "Yes"),
    "owns_mobile=yes": ("Owns mobile phone", "Yes"),
}


def cell(df, term, contrast):
    r = df[(df.term == term) & (df.contrast == contrast)]
    if r.empty:
        return ""
    r = r.iloc[0]
    star = "$^{*}$" if (r.ci_low > 1 or r.ci_high < 1) else ""  # 95% CI excludes 1
    return f"{f(r.RRR)} [{f(r.ci_low)}, {f(r.ci_high)}]{star}"


body = []
for term, (grp, lev) in termlab.items():
    if grp:
        body.append(rf"\multicolumn{{5}}{{l}}{{\textit{{{grp}}}}}")
    body.append(rf"\quad {lev} & " + " & ".join(cell(b, term, f"C{k} vs C1") for k in range(2, 6)))
write("tab6_correlates", body,
      "Bias-adjusted (BCH) relative risk ratios for latent class membership (reference: C1, minimal symptoms)",
      "tab:correlates", "lcccc",
      r"Covariate & C2: Mild, somatic & C3: Mild, anxiety & C4: Moderate & C5: High",
      "RRR [95\\% CI] from a survey-weighted three-step model correcting for classification error "
      "\\citep{Vermunt2010,Bolck2004}; CIs are percentile intervals from 200 stratified PSU bootstrap replicates "
      "that re-estimate the latent class model. $^{*}$95\\% CI excludes 1. All covariates are entered "
      "simultaneously, together with division; $N = 19{,}897$. Modal-assignment survey-weighted "
      "estimates (Table~\\Sref{tab:S9b}) lead to the same conclusions.", size=r"\scriptsize")

# ---------------------------------------------------------------- Supplementary tables
sup = []
s1 = pd.read_csv(TAB / "S1_sample_flow.csv")
write("S1_flow", [f"{esc(r.step)} & {int(r.n):,}" for _, r in s1.iterrows()], "Participant flow", "tab:S1", "lr",
      r"Step & $n$", "Data: BDHS 2022 individual recode (BDIR81FL). The mental-health module was administered to "
      "ever-married women aged 15--49 in households selected for the full woman's questionnaire.")
s2 = pd.read_csv(TAB / "S2_original_listwise_vs_dropped.csv")
write("S2_listwise", [f"{esc(r.variable)} & {esc(r.level) if pd.notna(r.level) else ''} & "
                      f"{f(r.iloc[2], 1)} & {f(r.iloc[3], 1)}" for _, r in s2.iterrows()],
      "Composition of the original submission's listwise-deleted sample versus the women it excluded",
      "tab:S2", "llrr", r"Variable & Level & Retained (\%) & Excluded (\%)",
      "The original analysis deleted every case with a missing value on any of about 90 variables, including "
      "variables that are structurally inapplicable (smartphone type, previous residence, husband's characteristics).")
r3 = pd.read_csv(TAB / "T3_reliability.csv")
write("S3_reliability", [f"{esc(r.scale)} & {r.k} & {f(r.cronbach_alpha)} & {f(r.ordinal_alpha)} & "
                         f"{f(r.omega_total_polychoric)}" for _, r in r3.iterrows()],
      "Internal consistency in the analytic sample", "tab:S3", "lrrrr",
      r"Scale & $k$ & Cronbach's $\alpha$ & Ordinal $\alpha$ & $\omega_{t}$ (polychoric)")
cs = pd.read_csv(TAB / "T7_cs_coefficients.csv")
ns = pd.read_csv(TAB / "T8_network_sensitivity.csv")
write("S4_network_robustness",
      [f"\\multicolumn{{5}}{{l}}{{\\textit{{Correlation-stability coefficients}}}}"] +
      [f"\\quad {esc(r.measure)} & \\multicolumn{{4}}{{l}}{{{f(r.iloc[1])}}}" for _, r in cs.iterrows()] +
      [r"\midrule", r"Model & Edges & $r$ (edges) & $r$ (strength) & Three strongest nodes"] +
      [f"{esc(r.model)} & {r.n_edges} & {f(r.r_edges)} & {f(r.r_strength)} & {esc(r.top3_strength)}"
       for _, r in ns.iterrows()],
      "Network stability and sensitivity analyses", "tab:S4", "lrrrp{5cm}", r"\multicolumn{5}{l}{}",
      "CS-coefficient = largest proportion of cases that can be dropped while the correlation with the full-sample "
      "estimate remains $\\geq$ .70 in 95\\% of case-dropping bootstrap samples (maximum tested: .75). With "
      "$N = 19{,}930$, EBIC selected the smallest $\\lambda$ on the path for $\\gamma = 0$, .25 and .5.",
      size=r"\scriptsize")
lpa = fit[fit.model != "LCA"]
write("S5_lpa", [f"{int(r.K)} & {f(r.LL, 1)} & {int(r.npar)} & {f(r.BIC, 1)} & {f(r.SABIC, 1)} & {f(r.entropy)} & "
                 f"{f(r['smallest_class_%'], 1)} & {f(r.min_AvePP)}" for _, r in lpa.iterrows()],
      "Sensitivity analysis: Gaussian latent profile models (diagonal, class-varying variances)", "tab:S5",
      "rrrrrrrr", r"$K$ & LL & $p$ & BIC & SABIC & Entropy & Min.\ \% & Min.\ AvePP",
      "Fitted to raw 0--3 scores with a variance floor of 0.01. Without a floor the likelihood is unbounded "
      "(a class of all-zero responders has variance $\\to 0$), and with a floor BIC continues to decrease through "
      "$K = 7$, indicating misfit of the Gaussian model to these zero-inflated ordinal items. Agreement with the "
      "LCA classification: adjusted Rand index .27--.54.", size=r"\scriptsize")
lv = pd.read_csv(TAB / "T9_level_vs_shape.csv")
write("S6_level_shape", [f"{int(r.K)} & {f(r.iloc[1], 1)} & {esc(r.iloc[2])} & {esc(r.iloc[3])}"
                         for _, r in lv.iterrows()],
      "Quantitative (severity) versus qualitative (shape) differences between classes", "tab:S6", "rrll",
      r"$K$ & \% variance, 1st dim. & Items where next class is lower & GAD/PHQ mean ratio by class",
      "Second column: share of the size-weighted between-class variance in item means captured by its first "
      "singular dimension (100\\% = classes differ only in overall level). Third column: for each pair of adjacent "
      "classes (ordered by mean severity), the number of items on which the more severe class scores lower.",
      size=r"\scriptsize")
li = pd.read_csv(TAB / "T15_local_independence_summary.csv")
bv = pd.read_csv(TAB / "S13_bivariate_residuals.csv")
top = bv[bv.K == 5].sort_values("BVR", ascending=False).head(8)
write("S7_local_independence",
      [f"{int(r.K)} & {int(r.pairs)} & {f(r.median_BVR)} & {f(r.max_BVR)} & {int(r.n_BVR_gt_3_84)} & "
       f"{f(r.median_V, 3)} & {f(r.max_V, 3)}" for _, r in li.iterrows()] +
      [r"\midrule", r"\multicolumn{7}{l}{\textit{Largest residual associations, $K = 5$}}"] +
      [f"\\quad {r.item_i}--{r.item_j} & & & {f(r.BVR)} & & & {f(r.effect_V, 3)}" for _, r in top.iterrows()],
      "Local-independence diagnostics: bivariate residuals (BVR) of the latent class models", "tab:S7",
      "lrrrrrr", r"$K$ & Pairs & Median BVR & Max BVR & BVR $>$ 3.84 & Median $V$ & Max $V$",
      "BVR = Pearson $\\chi^2$ of the observed vs model-implied $4\\times4$ table divided by its df. $V$ = "
      "$\\sqrt{\\chi^2/(3N)}$, a sample-size-free effect size. The largest residual dependencies coincide with the "
      "strongest network edges.", size=r"\scriptsize")
cl = pd.read_csv(TAB / "S12_classification_error_K5.csv")
write("S12_classification", [f"{esc(r.iloc[0])} & " + " & ".join(f(x, 3) for x in r.iloc[1:]) for _, r in cl.iterrows()],
      "Classification-error matrix $D$ of the five-class solution (rows: true class; columns: modal assignment)",
      "tab:S12", "lrrrrr", "& " + " & ".join(f"Assigned C{i}" for i in range(1, 6)),
      "Estimated with survey weights; used for the BCH correction.", size=r"\footnotesize")
g = pd.read_csv(TAB / "T11_gvif_K5.csv")
g = g[g.model == "M2"]
write("S10_gvif", [f"{esc(r.variable)} & {int(r.df)} & {f(r.GVIF)} & {f(r.iloc[3])}" for _, r in g.iterrows()],
      "Collinearity diagnostics for the fully adjusted model", "tab:S10", "lrrr",
      r"Covariate & df & GVIF & GVIF$^{1/(2\,df)}$",
      "Generalised variance inflation factors \\citep{FoxMonette1992}. Values of GVIF$^{1/(2df)}$ below about 2.2 "
      "(GVIF $<$ 5 for df = 1) are conventionally taken to indicate no problematic collinearity.")


def modal_table(name, model, caption, label):
    s = m[m.model == model]
    body = []
    for term, (grp, lev) in termlab.items():
        if grp:
            body.append(rf"\multicolumn{{5}}{{l}}{{\textit{{{grp}}}}}")
        body.append(rf"\quad {lev} & " + " & ".join(cell(s, term, f"C{k} vs C1") for k in range(2, 6)))
    write(name, body, caption, label, "lcccc", r"Covariate & C2 vs C1 & C3 vs C1 & C4 vs C1 & C5 vs C1",
          f"{s.design.iloc[0]}; $N$ = {int(s.n.max()):,}. $^{{*}}$95\\% CI excludes 1.", size=r"\scriptsize")


modal_table("S9a_modal_unadjusted", "M0 unadjusted", "Unadjusted survey-weighted multinomial models "
            "(modal class assignment; each covariate entered separately)", "tab:S9a")
modal_table("S9b_modal_adjusted", "M2 fully adjusted", "Fully adjusted survey-weighted multinomial model "
            "(modal class assignment)", "tab:S9b")
modal_table("S9c_urban", "M3 stratified: urban", "Fully adjusted model, urban women only", "tab:S9c")
modal_table("S9d_rural", "M3 stratified: rural", "Fully adjusted model, rural women only", "tab:S9d")
termlab_dm = dict(termlab)
termlab_dm.pop("husband_edu=no current partner")
termlab_dm["decisions=1-2"] = ("Household decisions participated in (ref.\\ 0 of 3)", "1--2")
termlab_dm["decisions=3"] = ("", "All 3")
s = m[m.model == "M4 currently married + decision-making"]
body = []
for term, (grp, lev) in termlab_dm.items():
    if grp:
        body.append(rf"\multicolumn{{5}}{{l}}{{\textit{{{grp}}}}}")
    body.append(rf"\quad {lev} & " + " & ".join(cell(s, term, f"C{k} vs C1") for k in range(2, 6)))
write("S9e_married_decisions", body, "Fully adjusted model among currently married women, adding participation in "
      "household decision-making (modal class assignment)", "tab:S9e", "lcccc",
      r"Covariate & C2 vs C1 & C3 vs C1 & C4 vs C1 & C5 vs C1",
      f"{s.design.iloc[0]}; $N$ = {int(s.n.max()):,}. Decisions: own health care, large household purchases, "
      "visits to family or relatives, made alone or jointly with the husband. $^{*}$95\\% CI excludes 1.",
      size=r"\scriptsize")
b4 = pd.read_csv(TAB / "T12_bch_K4.csv")
body = []
for term, (grp, lev) in termlab.items():
    if grp:
        body.append(rf"\multicolumn{{4}}{{l}}{{\textit{{{grp}}}}}")
    body.append(rf"\quad {lev} & " + " & ".join(cell(b4, term, f"C{k} vs C1") for k in range(2, 5)))
write("S11_bch_k4", body, "Sensitivity analysis: BCH relative risk ratios for the four-class solution",
      "tab:S11b", "lccc", r"Covariate & Mild vs minimal & Moderate vs minimal & High vs minimal",
      "Same specification as Table 6. $^{*}$95\\% CI excludes 1.", size=r"\scriptsize")

# ---------------------------------------------------------------- LCA robustness (10_lca_robustness.py)
wp = TAB / "S14_weighted_lca_K5.csv"
if wp.exists():
    wl = pd.read_csv(wp)
    note = (OUT / "weighted_lca_note_K5.txt").read_text().strip()
    write("S13w_weighted_lca",
          [f"{r['class']} & {f(r.iloc[1], 1)} & {f(r.iloc[2], 1)} & {f(r.iloc[3], 3)}" for _, r in wl.iterrows()],
          "Sensitivity analysis: survey-weighted versus unweighted five-class latent class model", "tab:S13w", "lrrr",
          r"Class & Size, unweighted model (\%) & Size, weighted model (\%) & Max.\ $|\Delta|$ item mean",
          esc(note) + ". The weighted model re-estimates the measurement model with the BDHS sampling weights; "
          "classes are matched on their item profiles.", size=r"\footnotesize")
up = TAB / "S15_lca_uncertainty_K5.csv"
if up.exists():
    u = pd.read_csv(up)
    u["cell"] = [f"{f(r.estimate, 1 if r.row.startswith('class size') else 2)} [{f(r.ci_low, 1 if r.row.startswith('class size') else 2)}, "
                 f"{f(r.ci_high, 1 if r.row.startswith('class size') else 2)}]" for _, r in u.iterrows()]
    wide = u.pivot_table(index="row", columns="class", values="cell", aggfunc="first")
    order = ["class size, % (weighted)"] + [r for r in u.row.unique() if not r.startswith("class size")]
    wide = wide.loc[order]
    body = [f"{esc(i)} & " + " & ".join(wide.loc[i, f"C{c}"] for c in range(1, 6)) for i in wide.index]
    write("S15u_lca_uncertainty", body,
          "Uncertainty of class sizes and class-specific item means (five-class solution): estimate [95\\% CI]",
          "tab:S15u", "l" + "c" * 5, r"Item & C1 & C2 & C3 & C4 & C5",
          f"Percentile intervals from {int(u.n_boot_ok.iloc[0])} stratified PSU bootstrap replicates, each "
          "re-estimating the latent class model (3 random starts); classes matched to the main solution on their "
          "item profiles. Class sizes are survey-weighted posterior proportions.", size=r"\scriptsize", env="table*")
cb = pd.read_csv(TAB / "S11_codebook.csv")
write("S11_codebook", [f"{esc(r.variable)} & {esc(r['BDHS name'])} & {esc(r['description / wording'])} & "
                       f"{esc(r.coding)}" for _, r in cb.iterrows()],
      "Variable description (codebook)", "tab:S11", "p{2.6cm}p{1.6cm}p{5cm}p{5.2cm}",
      r"Variable & BDHS name & Description / wording & Coding", size=r"\scriptsize", env="table*")
print(sorted(p.name for p in OUTD.glob("*.tex")))

# ---------------------------------------------------------------- separate supplementary file
# One ordered list defines the supplement; it generates both the supplement content and the S-number map
# used by the main manuscript (\Sref{label}), so the two documents cannot disagree.
SUPP_TABLES = [("S1_flow", "tab:S1"), ("S2_listwise", "tab:S2"), ("S3_reliability", "tab:S3"),
               ("S11_codebook", "tab:S11"), ("S4_network_robustness", "tab:S4"), ("S5_lpa", "tab:S5"),
               ("S6_level_shape", "tab:S6"), ("S7_local_independence", "tab:S7"),
               ("S15u_lca_uncertainty", "tab:S15u"), ("S13w_weighted_lca", "tab:S13w"),
               ("S12_classification", "tab:S12"), ("S10_gvif", "tab:S10"), ("S9a_modal_unadjusted", "tab:S9a"),
               ("S9b_modal_adjusted", "tab:S9b"), ("S9c_urban", "tab:S9c"), ("S9d_rural", "tab:S9d"),
               ("S9e_married_decisions", "tab:S9e"), ("S11_bch_k4", "tab:S11b")]
SUPP_FIGS = [
    ("S1_case_drop_stability.png", "fig:S1", 0.75, "Case-dropping bootstrap stability of centrality indices and edge "
     "weights (mean and 95\\% interval of the correlation with the full-sample estimate; red line: .70)."),
    ("S2_edge_bootstrap.png", "fig:S2", 0.6, "Edge weights with 95\\% nonparametric bootstrap intervals (1,000 samples)."),
    ("S3_strength_difference.png", "fig:S3", 0.7, "Bootstrapped strength-difference tests. Dark cells: the 95\\% "
     "interval of the difference between two nodes excludes zero. Strength values in parentheses."),
    ("S4_information_criteria.png", "fig:S4", 1.0, "Information criteria for latent class (left) and Gaussian latent "
     "profile (right) models with 1--7 classes."),
    ("F4_final_classes_K4.png", "fig:S5", 1.0, "Sensitivity analysis: model-implied item means of the four-class solution."),
]
missing_t = [n for n, _ in SUPP_TABLES if not (OUTD / f"{n}.tex").exists()]
if missing_t:
    raise SystemExit(f"supplementary tables not generated: {missing_t}")
content = ["% Generated by analysis/08_latex_tables.py -- do not edit by hand.",
           "\\setcounter{table}{0}\\renewcommand{\\thetable}{S\\arabic{table}}",
           "\\setcounter{figure}{0}\\renewcommand{\\thefigure}{S\\arabic{figure}}"]
content += [f"\\input{{tables/{n}}}" for n, _ in SUPP_TABLES]
for fname, lab, wd, cap in SUPP_FIGS:
    content += ["\\begin{figure}[htbp]\\centering", f"\\includegraphics[width={wd}\\linewidth]{{{fname}}}",
                f"\\caption{{{cap}}}\\label{{{lab}}}", "\\end{figure}"]
(MANU / "supplementary_content.tex").write_text("\n".join(content) + "\n")
nums = ["% Generated by analysis/08_latex_tables.py: S-numbers of supplementary items, used via \\Sref{label}.",
        "\\providecommand{\\Sref}[1]{\\ifcsname supp@#1\\endcsname\\csname supp@#1\\endcsname\\else\\textbf{??}\\fi}"]
nums += [f"\\expandafter\\def\\csname supp@{lab}\\endcsname{{S{i}}}" for i, (_, lab) in enumerate(SUPP_TABLES, 1)]
nums += [f"\\expandafter\\def\\csname supp@{lab}\\endcsname{{S{i}}}" for i, (_, lab, _, _) in enumerate(SUPP_FIGS, 1)]
(OUTD / "supp_numbers.tex").write_text("\n".join(nums) + "\n")
smap = {lab: f"S{i}" for i, (_, lab) in enumerate(SUPP_TABLES, 1)}
smap.update({lab: f"S{i}" for i, (_, lab, _, _) in enumerate(SUPP_FIGS, 1)})
import json as _json
(OUTD / "supp_numbers.json").write_text(_json.dumps(smap, indent=1))
print("supplement:", len(SUPP_TABLES), "tables,", len(SUPP_FIGS), "figures")
