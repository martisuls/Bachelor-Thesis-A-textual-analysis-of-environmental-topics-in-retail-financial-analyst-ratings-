import sys
from pathlib import Path

import pandas as pd

# ================= CONFIG =================
INPUT_FINAL_DATA = Path("final_data_all.csv")
INPUT_REFERENCE = Path("companies dictionary") / "merged_companies_prime_gics.csv"
OUTPUT_CSV = Path("final_data_all_gics.csv")
# ==========================================


def main() -> None:
    if not INPUT_FINAL_DATA.exists():
        sys.exit(f"ERROR: cannot find {INPUT_FINAL_DATA.resolve()}")
    print(f"Reading {INPUT_FINAL_DATA} ...")
    final_df = pd.read_csv(INPUT_FINAL_DATA)
    print(f"  {len(final_df):,} rows loaded")

    if not INPUT_REFERENCE.exists():
        sys.exit(f"ERROR: cannot find {INPUT_REFERENCE.resolve()}")
    print(f"Reading {INPUT_REFERENCE} ...")
    ref_df = pd.read_csv(INPUT_REFERENCE, usecols=["id", "gics_subindustry_code"])
    print(f"  {len(ref_df):,} rows loaded")

  
    ref_df = ref_df.drop_duplicates(subset=["id"], keep="first")

    drop_cols = [
        c for c in ("gics_sector_name", "gics_indgrp_name") if c in final_df.columns
    ]
    if drop_cols:
        print(f"Dropping columns: {drop_cols}")
        final_df = final_df.drop(columns=drop_cols)

  
    print("Joining on 'id' ...")
    merged = final_df.merge(ref_df, on="id", how="left")

   
    tail_order = ["date", "companyName", "gics_subindustry_code", "country_iso"]
    head_cols = [c for c in merged.columns if c not in tail_order]
    merged = merged[head_cols + [c for c in tail_order if c in merged.columns]]

    print(f"Writing {OUTPUT_CSV} ...")
    merged.to_csv(OUTPUT_CSV, index=False)

    n_total = len(merged)
    n_with_code = merged["gics_subindustry_code"].notna().sum()
    n_missing = n_total - n_with_code
    print(
        f"Done. {n_total:,} rows written "
        f"({n_with_code:,} with gics_subindustry_code, "
        f"{n_missing:,} missing).",
    )
    if n_missing:
        print(
            "  Reports missing a subindustry code usually correspond to the\n"
            "  8 companies patched manually by force_add_sector.py (e.g., Brookfield\n"
            "  Real Assets, China Agritech, XOMA, etc.). Run that patch AFTER\n"
            "  adding_gicsname.py to fill gics_sector_name / gics_indgrp_name\n"
            "  for those rows.",
        )


if __name__ == "__main__":
    main()
