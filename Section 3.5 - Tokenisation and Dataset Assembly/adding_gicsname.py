import pandas as pd
import numpy as np

# Manually define the 11 GICS Sectors
SECTORS = {
    10: "Energy",
    15: "Materials",
    20: "Industrials",
    25: "Consumer Discretionary",
    30: "Consumer Staples",
    35: "Health Care",
    40: "Financials",
    45: "Information Technology",
    50: "Communication Services",
    55: "Utilities",
    60: "Real Estate"
}

# Manually define the 25 GICS Industry Groups
INDUSTRY_GROUPS = {
    1010: "Energy",
    1510: "Materials",
    2010: "Capital Goods",
    2020: "Commercial & Professional Services",
    2030: "Transportation",
    2510: "Automobiles & Components",
    2520: "Consumer Durables & Apparel",
    2530: "Consumer Services",
    2550: "Consumer Discretionary Distribution & Retail",
    3010: "Consumer Staples Distribution & Retail",
    3020: "Food Beverage & Tobacco",
    3030: "Household & Personal Products",
    3510: "Health Care Equipment & Services",
    3520: "Pharmaceuticals Biotechnology & Life Sciences",
    4010: "Banks",
    4020: "Financial Services",
    4030: "Insurance",
    4510: "Software & Services",
    4520: "Technology Hardware & Equipment",
    4530: "Semiconductors & Semiconductor Equipment",
    5010: "Telecommunication Services",
    5020: "Media & Entertainment",
    5510: "Utilities",
    6010: "Equity Real Estate Investment Trusts (REITs)",
    6020: "Real Estate Management & Development"
}

# Manually define the 74 GICS Industries
INDUSTRIES = {
    101010: "Energy Equipment & Services",
    101020: "Oil Gas & Consumable Fuels",
    151010: "Chemicals",
    151020: "Construction Materials",
    151030: "Containers & Packaging",
    151040: "Metals & Mining",
    151050: "Paper & Forest Products",
    201010: "Aerospace & Defense",
    201020: "Building Products",
    201030: "Construction & Engineering",
    201040: "Electrical Equipment",
    201050: "Industrial Conglomerates",
    201060: "Machinery",
    201070: "Trading Companies & Distributors",
    202010: "Commercial Services & Supplies",
    202020: "Professional Services",
    203010: "Air Freight & Logistics",
    203020: "Passenger Airlines",
    203030: "Marine Transportation",
    203040: "Ground Transportation",
    203050: "Transportation Infrastructure",
    251010: "Automobile Components",
    251020: "Automobiles",
    252010: "Household Durables",
    252020: "Leisure Products",
    252030: "Textiles Apparel & Luxury Goods",
    253010: "Hotels Restaurants & Leisure",
    253020: "Diversified Consumer Services",
    255010: "Distributors",
    255030: "Broadline Retail",
    255040: "Specialty Retail",
    301010: "Consumer Staples Distribution & Retail",
    302010: "Beverages",
    302020: "Food Products",
    302030: "Tobacco",
    303010: "Household Products",
    303020: "Personal Care Products",
    351010: "Health Care Equipment & Supplies",
    351020: "Health Care Providers & Services",
    351030: "Health Care Technology",
    352010: "Biotechnology",
    352020: "Pharmaceuticals",
    352030: "Life Sciences Tools & Services",
    401010: "Banks",
    402010: "Financial Services",
    402020: "Consumer Finance",
    402030: "Capital Markets",
    402040: "Mortgage Real Estate Investment Trusts (REITs)",
    403010: "Insurance",
    451020: "IT Services",
    451030: "Software",
    452010: "Communications Equipment",
    452020: "Technology Hardware Storage & Peripherals",
    452030: "Electronic Equipment Instruments & Components",
    453010: "Semiconductors & Semiconductor Equipment",
    501010: "Diversified Telecommunication Services",
    501020: "Wireless Telecommunication Services",
    502010: "Media",
    502020: "Entertainment",
    502030: "Interactive Media & Services",
    551010: "Electric Utilities",
    551020: "Gas Utilities",
    551030: "Multi-Utilities",
    551040: "Water Utilities",
    551050: "Independent Power and Renewable Electricity Producers",
    601010: "Diversified REITs",
    601025: "Industrial REITs",
    601030: "Hotel & Resort REITs",
    601040: "Office REITs",
    601050: "Health Care REITs",
    601060: "Residential REITs",
    601070: "Retail REITs",
    601080: "Specialized REITs",
    602010: "Real Estate Management & Development"
}


def extract_gics_codes(subindustry_code):
    """
    Extract sector, industry group, and industry codes from subindustry code.
    Subindustry code format: 8 digits (SSIIGGNN)
    - First 2 digits: Sector
    - First 4 digits: Industry Group
    - First 6 digits: Industry
    """
    if pd.isna(subindustry_code):
        return None, None, None

    # Convert to string and remove decimal point if present
    code_str = str(int(subindustry_code))

    # Pad with zeros if necessary (should be 8 digits)
    code_str = code_str.zfill(8)

    sector_code = int(code_str[:2])
    indgrp_code = int(code_str[:4])
    industry_code = int(code_str[:6])

    return sector_code, indgrp_code, industry_code


def map_gics_names(df):
    """
    Add GICS sector, industry group, and industry names to the dataframe.
    """
    # Create new columns
    df['gics_sector_name'] = None
    df['gics_indgrp_name'] = None
    df['gics_industry_name'] = None

    # Iterate through each row and map the codes
    for idx, row in df.iterrows():
        subindustry_code = row['gics_subindustry_code']

        if pd.notna(subindustry_code):
            sector_code, indgrp_code, industry_code = extract_gics_codes(subindustry_code)

            # Map to names
            df.at[idx, 'gics_sector_name'] = SECTORS.get(sector_code, None)
            df.at[idx, 'gics_indgrp_name'] = INDUSTRY_GROUPS.get(indgrp_code, None)
            df.at[idx, 'gics_industry_name'] = INDUSTRIES.get(industry_code, None)

    return df


def main():
    # Read the input CSV
    print("Reading input CSV...")
    df = pd.read_csv('final_data_all_gics.csv')

    print(f"Total rows: {len(df)}")
    print(f"Rows with subindustry code: {df['gics_subindustry_code'].notna().sum()}")

    # Map GICS names
    print("\nMapping GICS names...")
    df_with_names = map_gics_names(df)

    # Reorder columns to put new columns after gics_subindustry_code
    cols = df.columns.tolist()
    subind_idx = cols.index('gics_subindustry_code')

    # Insert new columns after gics_subindustry_code
    new_cols = (cols[:subind_idx+1] +
                ['gics_sector_name', 'gics_indgrp_name', 'gics_industry_name'] +
                cols[subind_idx+1:])

    df_with_names = df_with_names[new_cols]

    # Save to new CSV
    output_file = 'final_data_all_gics_named.csv'
    print(f"\nSaving to {output_file}...")
    df_with_names.to_csv(output_file, index=False)

    print(f"Done! Output saved to {output_file}")

    # Print some statistics
    print("\nMapping statistics:")
    print(f"Rows with sector name: {df_with_names['gics_sector_name'].notna().sum()}")
    print(f"Rows with industry group name: {df_with_names['gics_indgrp_name'].notna().sum()}")
    print(f"Rows with industry name: {df_with_names['gics_industry_name'].notna().sum()}")

    # Show a sample
    print("\nSample of output (first 5 rows with GICS codes):")
    sample = df_with_names[df_with_names['gics_subindustry_code'].notna()].head()
    print(sample[['companyName', 'gics_subindustry_code', 'gics_sector_name',
                  'gics_indgrp_name', 'gics_industry_name']].to_string())


if __name__ == "__main__":
    main()
