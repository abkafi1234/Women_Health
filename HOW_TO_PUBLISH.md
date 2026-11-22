# Publishing the repository

The folder `bdhs2022-depression-anxiety-lca-network/` is ready to push. It contains no microdata (checked), and `.gitignore` keeps the BDHS file out of git.

## 1. Push to GitHub
```bash
cd bdhs2022-depression-anxiety-lca-network
git init -b main
git add .
git commit -m "Analysis code and aggregate results for BDHS 2022 depression/anxiety LCA and network study"
# create an empty repository named bdhs2022-depression-anxiety-lca-network on github.com (no README), then:
git remote add origin https://github.com/<your-account>/bdhs2022-depression-anxiety-lca-network.git
git push -u origin main
```

## 2. Make it findable
- **Description (About box):** "Reproducible Python code: latent class and psychological network analysis of PHQ-9/GAD-7 depression and anxiety symptoms among women in Bangladesh (BDHS 2022 DHS)".
- **Topics:** `depression` `anxiety` `phq-9` `gad-7` `latent-class-analysis` `network-analysis` `psychometrics` `mental-health` `womens-health` `bangladesh` `demographic-and-health-surveys` `dhs` `survey-data` `python` `reproducible-research`
- After publication, add the article citation and DOI to `README.md` and `CITATION.cff`, and archive a release on **Zenodo** (GitHub → Zenodo integration). A citable DOI makes the code itself citable.

## 3. During double-anonymised peer review
Your GitHub account name and `CITATION.cff` reveal who you are. Do **not** put the GitHub link in the anonymised manuscript. Instead, create an anonymised mirror at <https://anonymous.4open.science> and list the four author names as terms to redact. Use that link in the manuscript (§2.3.4 and Data availability) until acceptance. The alternative is a view-only anonymised OSF link.

## 4. Check before going public
- In `CITATION.cff`, check that each author's given and family names are split the way you want, e.g. "Al Mamun" or "Al Kafi" as the family name.
- The licence is MIT ("Copyright (c) 2026 The authors"). Change it if your institution requires something else.
