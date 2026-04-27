import pandas as pd

# Read the CSV file
df = pd.read_csv('articles_dates.csv')

# Keep only id and publishOn columns
df_cleaned = df[['id', 'publishOn']].copy()

# Convert publishOn to datetime with UTC timezone handling
df_cleaned['publishOn'] = pd.to_datetime(df_cleaned['publishOn'], utc=True, errors='coerce')

# Format to show year-month-day (YYYY-MM-DD format)
df_cleaned['publishOn'] = df_cleaned['publishOn'].dt.strftime('%Y-%m-%d')

# Save to a new CSV file using the filename expected by Section 3.5
# (final_data_all.py reads articles_datesformat_filtered.csv).
df_cleaned.to_csv('articles_datesformat_filtered.csv', index=False)

print(f"Cleaning complete!")
print(f"Output saved to: articles_datesformat_filtered.csv")

# Display first few rows to verify
print("\nFirst 5 rows of cleaned data:")
print(df_cleaned.head())
