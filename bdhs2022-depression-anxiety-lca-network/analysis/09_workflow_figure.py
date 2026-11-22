"""Figure 1: analytic workflow (complementary person-centred and network analyses)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from common import FIG

fig, ax = plt.subplots(figsize=(10, 6.2))
ax.set_xlim(0, 10)
ax.set_ylim(0, 6.2)
ax.axis("off")


def box(x, y, w, h, title, body, fc):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.12", fc=fc, ec="#333333",
                                lw=1.1))
    ax.text(x + w / 2, y + h - 0.2, title, ha="center", va="top", fontsize=10, fontweight="bold")
    ax.text(x + w / 2, y + h - 0.55, body, ha="center", va="top", fontsize=8.2, linespacing=1.35)


def arrow(x1, y1, x2, y2, style="-|>", ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=14, lw=1.2, color="#333333",
                                 linestyle=ls))


box(2.6, 4.75, 4.8, 1.3, "Data: BDHS 2022 (individual recode)",
    "Ever-married women aged 15-49, full-questionnaire households\n"
    "PHQ-9 + GAD-7 items (0-3), complete responses: N = 19,930\n"
    "Survey weights, 674 PSUs, 16 strata; covariates", "#EDEDED")
box(0.2, 2.15, 4.3, 2.1, "A. Person-centred: latent classes",
    "Polytomous LCA, K = 1-7 (20 random starts)\n"
    "BIC, SABIC, CAIC, entropy, AvePP, BLRT,\nbest-LL replication; level vs shape\n"
    "Sensitivity: Gaussian LPA; K = 4\nLocal independence: bivariate residuals", "#DCEBF7")
box(5.5, 2.15, 4.3, 2.1, "B. Variable-centred: symptom network",
    "Polychoric GGM, EBICglasso (gamma = 0.5)\n"
    "Strength, expected influence, bridge strength\n"
    "1,000 bootstraps: edge/centrality CIs,\ndifference tests; case-drop CS-coefficients\n"
    "Sensitivity: Spearman, Pearson, gamma", "#FBE7CF")
box(0.2, 0.1, 4.3, 1.55, "C. Correlates of class membership",
    "Survey-weighted multinomial logit (modal class)\n"
    "BCH bias-adjusted three-step, PSU bootstrap\n"
    "GVIF; unadjusted / adjusted / stratified", "#DCEBF7")
box(5.5, 0.1, 4.3, 1.55, "Complementary interpretation",
    "Classes: between-person heterogeneity\n(level and shape of symptom profiles)\n"
    "Network: item-level conditional associations\n(incl. residual dependence within classes)", "#E8E8E8")
arrow(4.2, 4.75, 2.6, 4.27)
arrow(5.8, 4.75, 7.4, 4.27)
arrow(2.35, 2.15, 2.35, 1.68)
arrow(4.52, 3.2, 5.48, 3.2, style="<|-|>", ls="--")
ax.text(5.0, 3.33, "compared,\nnot integrated", ha="center", va="bottom", fontsize=7.5, style="italic")
arrow(7.65, 2.15, 7.65, 1.68)
arrow(4.52, 0.85, 5.48, 0.85)
fig.tight_layout()
fig.savefig(FIG / "F1_workflow.png", dpi=300)

print("Figure 1 written to", FIG / "F1_workflow.png")
