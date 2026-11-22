"""Variable description table (codebook) for the supplement."""
import pandas as pd

from common import ITEM_INFO, save_table

rows = [(r.item, r["var"], r.wording, "Frequency in the past two weeks, recorded as 0 = never, 1 = rarely, 2 = often, 3 = always "
         "(BDHS response labels); 'don't know'/'refused' treated as missing (excluded)", "Symptom indicator")
        for _, r in ITEM_INFO.iterrows()]
rows += [
    ("Residence", "v102", "Type of place of residence", "urban (ref), rural", "Covariate"),
    ("Division", "v101", "Administrative division", "Dhaka (ref), Barishal, Chattogram, Khulna, Mymensingh, Rajshahi, "
     "Rangpur, Sylhet", "Covariate"),
    ("Education", "v106", "Highest educational level", "no education (ref), primary, secondary, higher", "Covariate"),
    ("Wealth quintile", "v190", "DHS wealth index (national)", "poorest (ref) ... richest", "Covariate"),
    ("Wealth quintile within residence", "v190a", "DHS wealth index computed separately for urban/rural",
     "poorest (ref) ... richest", "Sensitivity analysis"),
    ("Age group", "v012", "Respondent's current age", "15-24 (ref), 25-34, 35-49", "Covariate"),
    ("Children ever born", "v201", "Total children ever born", "0 (ref), 1-2, 3+", "Covariate"),
    ("Religion", "v130", "Religion", "Muslim (ref), non-Muslim (Hindu, Buddhist, Christian, other)", "Covariate"),
    ("Currently working", "v714", "Respondent currently working", "no (ref), yes", "Covariate"),
    ("Husband's education", "v701; v501", "Husband/partner's education level; current marital status",
     "no education (ref), primary, secondary, higher, no current partner (widowed/divorced/separated); "
     "'don't know' = missing", "Covariate"),
    ("Female household head", "v151", "Sex of household head", "no (ref), yes", "Covariate"),
    ("Household size", "v136", "Number of household members", "continuous", "Covariate"),
    ("Internet use (12 months)", "v171a", "Use of internet", "no (ref; never or before last 12 months), yes", "Covariate"),
    ("Owns a mobile phone", "v169a", "Owns a mobile telephone", "no (ref), yes", "Covariate"),
    ("Justifies wife-beating", "v744a-e", "Beating justified for at least one of five reasons",
     "no (ref; includes 'don't know'), yes", "Covariate"),
    ("Household decision-making", "v743a, v743b, v743d",
     "Number of three decisions (own health care, large purchases, visits to family) made alone or jointly",
     "0 (ref), 1-2, 3; currently married women only", "Covariate (M4)"),
    ("Survey weight", "v005", "Women's individual sample weight / 1,000,000", "", "Design"),
    ("Primary sampling unit", "v021", "PSU (cluster)", "674 PSUs", "Design"),
    ("Stratum", "v023", "Stratification used in sample design", "16 strata", "Design"),
]
save_table(pd.DataFrame(rows, columns=["variable", "BDHS name", "description / wording", "coding", "role"]),
           "S11_codebook")
print(len(rows), "rows written")
