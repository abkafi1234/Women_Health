"""Publication figure for the selected latent class solution, with descriptive class names."""
import os
import pickle

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import FIG, ITEMS, ITEM_INFO, OUT, SHORT, save_table

K = int(os.environ.get("K", 5))
NAMES = {
    5: ["Minimal symptoms", "Mild, somatic-predominant", "Mild, anxiety-predominant",
        "Moderate, generalised", "High, generalised"],
    4: ["Minimal symptoms", "Mild symptoms", "Moderate symptoms", "High symptoms"],
}[K]
STYLE = ["#4D4D4D", "#2A9D8F", "#E9A23B", "#C2185B", "#5E35B1"][:K]

df = pd.read_pickle(OUT / "analytic.pkl")
tab = pd.read_csv(OUT / "tables" / f"S8_lca_K{K}_class_item_means.csv")
sizes = tab.iloc[-1, 2::3].astype(float).to_numpy()
tab = tab.iloc[:-1]
post = pd.read_pickle(OUT / "models" / f"lca_post_K{K}.pkl").to_numpy()
w = df["weight"].to_numpy() if "weight" in df else np.ones(len(df))
wsize = (post * w[:, None]).sum(0) / w.sum() * 100

fig, ax = plt.subplots(figsize=(11, 6.2))
for c in range(K):
    m = tab[f"C{c + 1} mean"].to_numpy()
    ax.plot(range(16), m, marker="o", lw=2, color=STYLE[c],
            label=f"{NAMES[c]} ({sizes[c]:.1f}%; weighted {wsize[c]:.1f}%)")
ax.axvline(6.5, color="grey", ls="--", lw=1)
ax.text(3, 2.1, "GAD-7", ha="center")
ax.text(11.5, 2.1, "PHQ-9", ha="center")
ax.set_ylim(-0.05, 2.25)
ax.set_xticks(range(16))
ax.set_xticklabels([f"{SHORT[c]}\n{ITEM_INFO.set_index('var').loc[c, 'label']}" for c in ITEMS],
                   rotation=60, ha="right", fontsize=8)
ax.set_ylabel("Expected item score\n(0 = never ... 3 = always)")
ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.36), ncol=3, frameon=False)
ax.grid(ls=":", alpha=0.6)
ax.set_title(f"Latent classes of depressive and anxiety symptoms (polytomous LCA, K = {K}, N = {len(df):,})")
fig.tight_layout()
fig.savefig(FIG / f"F4_final_classes_K{K}.png", dpi=300)
plt.close(fig)

# PHQ-9 / GAD-7 sum scores and conventional cut-offs by class (helps readers map classes onto screening use)
cls = post.argmax(1)
rows = []
for c in range(K):
    s = df[cls == c]
    rows.append({"class": f"C{c + 1}", "name": NAMES[c], "n": len(s),
                 "PHQ-9 mean (SD)": f"{s.PHQ9.mean():.1f} ({s.PHQ9.std():.1f})",
                 "GAD-7 mean (SD)": f"{s.GAD7.mean():.1f} ({s.GAD7.std():.1f})",
                 "% PHQ-9 >= 10": (s.PHQ9 >= 10).mean() * 100, "% GAD-7 >= 10": (s.GAD7 >= 10).mean() * 100})
save_table(pd.DataFrame(rows), f"T14_class_sumscores_K{K}", floatfmt="%.1f")
with open(OUT / f"class_names_K{K}.pkl", "wb") as f:
    pickle.dump(NAMES, f)
print(pd.DataFrame(rows).round(1).to_string(index=False))
