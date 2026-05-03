import pandas as pd
import numpy as np
import re

# 1) Load your Compustat export
df = pd.read_csv("compustatNAraw.csv")

# 2) Parse date & tidy names
df["datadate"] = pd.to_datetime(df["datadate"], errors="coerce")
df["conm"] = df["conm"].astype(str).str.strip()

# 3) Keep only what we need
keep = ["gvkey","conm","tic","fic","sic","naics","gsubind","datadate","costat"]
df = df[keep].copy()

# 4) latest record per gvkey
df = df.sort_values(["gvkey","datadate"])
latest = df.groupby("gvkey", as_index=False).tail(1).reset_index(drop=True)

# 5) derive GICS hierarchy from 8-digit subindustry code
def derive_gics(sub):
    if pd.isna(sub):
        return pd.Series([np.nan,np.nan,np.nan])
    try:
        c = int(sub)
    except Exception:
        return pd.Series([np.nan,np.nan,np.nan])
    return pd.Series([c//10**6, c//10**4, c//10**2])  # sector, group, industry

latest[["gics_sector_code","gics_indgrp_code","gics_industry_code"]] = \
    latest["gsubind"].apply(derive_gics)

#sector names mapping
gics_sector_names = {
    10:"Energy",15:"Materials",20:"Industrials",25:"Consumer Discretionary",
    30:"Consumer Staples",35:"Health Care",40:"Financials",45:"Information Technology",
    50:"Communication Services",55:"Utilities",60:"Real Estate"
}
latest["gics_sector_name"] = latest["gics_sector_code"].map(gics_sector_names)

# industry group names mapping
gics_indgrp_names = {
    1010: "Energy Equipment & Services",
    1015: "Oil, Gas & Consumable Fuels",
    1510: "Materials",
    2010: "Capital Goods",
    2020: "Commercial & Professional Services",
    2030: "Transportation",
    2510: "Automobiles & Components",
    2520: "Consumer Durables & Apparel",
    2530: "Consumer Services",
    2550: "Consumer Discretionary Distribution & Retail",
    3010: "Food & Staples Retailing",
    3020: "Food, Beverage & Tobacco",
    3030: "Household & Personal Products",
    3510: "Health Care Equipment & Services",
    3520: "Pharmaceuticals, Biotechnology & Life Sciences",
    4010: "Banks",
    4020: "Diversified Financials",
    4030: "Insurance",
    4510: "Software & Services",
    4520: "Technology Hardware & Equipment",
    4530: "Semiconductors & Semiconductor Equipment",
    5010: "Telecommunication Services",
    5020: "Media & Entertainment",
    5510: "Utilities",
    6010: "Real Estate"
}
latest["gics_indgrp_name"] = latest["gics_indgrp_code"].map(gics_indgrp_names)

# 6) clean company names for fuzzy/joins
LEGAL_SUFFIXES = r'\b(incorporated|inc|corp|corporation|ltd|limited|plc|ag|sa|nv|oyj|ab|as|spa|s\.p\.a|co|company|holdings?|group|se|kk|gmbh|sarl|sas|bv|oy)\b'
def clean_name(s):
    s = str(s).lower()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(LEGAL_SUFFIXES, " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

latest["conm_clean"] = latest["conm"].apply(clean_name)

# 7) final reference table
ref = latest.rename(columns={
    "conm":"company_name","tic":"ticker","fic":"country_iso",
    "sic":"sic_code","naics":"naics_code","gsubind":"gics_subindustry_code"
})[[
    "gvkey","company_name","conm_clean","ticker","country_iso",
    "sic_code","naics_code","gics_subindustry_code",
    "gics_sector_code","gics_sector_name",
    "gics_indgrp_code","gics_indgrp_name",   # <-- new columns here
    "datadate","costat"
]]

ref.to_csv("compustat_ref_NA.csv", index=False)
print("Saved compustat_ref_NA.csv with", len(ref), "unique firms")
