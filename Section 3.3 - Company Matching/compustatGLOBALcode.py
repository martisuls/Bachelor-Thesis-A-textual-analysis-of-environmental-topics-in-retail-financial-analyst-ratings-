import pandas as pd
import numpy as np
import re

# 1) Load your Compustat Global export
df = pd.read_csv("compustatGLOBALraw.csv")

# 2) Parse date & tidy names
df["datadate"] = pd.to_datetime(df["datadate"], errors="coerce")
df["conm"] = df["conm"].astype(str).str.strip()

# 3) Keep only what we need (Global provides isin + exchange_code)
keep = ["gvkey","conm","isin","exchg","fic","sic","naics","gsubind","datadate","costat"]
df = df[keep].copy()

# 4) latest record per gvkey
df = df.sort_values(["gvkey","datadate"])
latest = df.groupby("gvkey", as_index=False).tail(1).reset_index(drop=True)

# 5) derive GICS hierarchy (sector, industry group, industry) from 8-digit subindustry
def derive_gics(sub):
    if pd.isna(sub):
        return pd.Series([np.nan, np.nan, np.nan])
    try:
        c = int(float(sub))  # handle decimals like 45201020.0
    except Exception:
        return pd.Series([np.nan, np.nan, np.nan])
    return pd.Series([c//10**6, c//10**4, c//10**2])  # sector(2d), group(4d), industry(6d)

latest[["gics_sector_code","gics_indgrp_code","gics_industry_code"]] = \
    latest["gsubind"].apply(derive_gics)

# sector names mapping
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

# 6) clean company names
LEGAL_SUFFIXES = r'\b(incorporated|inc|corp|corporation|ltd|limited|plc|ag|sa|nv|oyj|ab|as|spa|s\.p\.a|co|company|holdings?|group|se|kk|gmbh|sarl|sas|bv|oy)\b'
def clean_name(s):
    s = str(s).lower()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(LEGAL_SUFFIXES, " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

latest["conm_clean"] = latest["conm"].apply(clean_name)

# 7) final reference table with unified schema (ticker placeholder for Global)
ref = latest.rename(columns={
    "conm":"company_name",
    "isin":"isin",
    "exchg":"exchange_code",
    "fic":"country_iso",
    "sic":"sic_code",
    "naics":"naics_code",
    "gsubind":"gics_subindustry_code"
})
ref["ticker"] = np.nan  # Global doesn't provide ticker

# Optional: tidy numeric codes to Int64 so you don't see trailing ".0"
for col in ["exchange_code","sic_code","naics_code","gics_subindustry_code",
            "gics_sector_code","gics_indgrp_code","gics_industry_code"]:
    if col in ref.columns:
        ref[col] = pd.to_numeric(ref[col], errors="coerce").astype("Int64")

# Reorder columns to match NA + extra Global info + new indgrp columns
ref = ref[[
    "gvkey","company_name","conm_clean","ticker","isin","exchange_code","country_iso",
    "sic_code","naics_code","gics_subindustry_code",
    "gics_sector_code","gics_sector_name",
    "gics_indgrp_code","gics_indgrp_name",
    "datadate","costat"
]]

# 8) Save Global
ref.to_csv("compustat_ref_Global.csv", index=False)
print("Saved compustat_ref_Global.csv with", len(ref), "unique firms")

# 9) Merge with NA
na = pd.read_csv("compustat_ref_NA.csv")
gl = pd.read_csv("compustat_ref_Global.csv")
allref = pd.concat([na, gl], ignore_index=True)

# deduplication of GVKEY
allref = allref.sort_values(["gvkey","datadate"])
allref = allref.groupby("gvkey", as_index=False).tail(1)
assert allref["gvkey"].is_unique

allref.to_csv("compustat_ref_All.csv", index=False)
print("Saved compustat_ref_All.csv with", len(allref), "rows")
