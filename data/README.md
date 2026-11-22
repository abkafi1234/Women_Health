# Data

The analyses use the **2022 Bangladesh Demographic and Health Survey (BDHS 2022), individual recode (IR)**, Stata format: `BDIR81FL.DTA`.

The data are distributed by the DHS Program. The terms of use **do not allow redistribution**, so the file is not in this repository.

1. Register at <https://dhsprogram.com> and request access to the Bangladesh 2022 Standard DHS (state a short research purpose).
2. Download the Individual Recode in **Stata** format and unzip it.
3. Put `BDIR81FL.DTA` in this `data/` folder.

`.gitignore` blocks every data file here except this README from being committed.

Design variables used: women's sampling weight `v005` (divided by 10⁶), primary sampling unit `v021`, stratum `v023`. Symptom items: `mth1`–`mth7` = GAD-7 items 1–7; `mth8`–`mth16` = PHQ-9 items 1–9. Responses are labelled never / rarely / often / always (coded 0–3). The full codebook is in `results/tables/S11_codebook.csv`.
