#!/usr/bin/env python3
"""
reduce_companies_columns.py

Reduces the raw companies.csv export to the identifier columns used by the
company-matching pipeline (Section 3.3). Drops all other fields.

Input:  companies.csv          (raw Seeking Alpha company export)
Output: companies_new.csv      (six identifier columns)

Run from the project root that contains companies.csv:
    python reduce_companies_columns.py
"""

import os
import pandas as pd


INPUT_CSV = "companies.csv"
OUTPUT_CSV = "companies_new.csv"

KEEP_COLS = ["slug", "iexSlug", "name", "companyName", "tradingViewSlug", "id"]


def main() -> None:
    df = pd.read_csv(INPUT_CSV)

    # Keep only the identifier columns. Create empty placeholders for any
    # column that is missing from the input so the schema is stable.
    existing = [c for c in KEEP_COLS if c in df.columns]
    df_new = df[existing].copy()
    for col in KEEP_COLS:
        if col not in df_new.columns:
            df_new[col] = ""

    # Enforce final column order
    df_new = df_new[KEEP_COLS]

    df_new.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print(f"Reduced companies file written to: {os.path.abspath(OUTPUT_CSV)}")
    print(f"Rows: {len(df_new):,}")
    print(f"Columns kept: {', '.join(KEEP_COLS)}")


if __name__ == "__main__":
    main()
