import csv

# Input file name
input_file = 'merged_companies_prime.csv'

# Output file names
no_match_file = 'no_match_companies.csv'
matched_file = 'matched_companies.csv'

# Read the input CSV and separate the data
with open(input_file, 'r', encoding='utf-8') as infile:
    reader = csv.DictReader(infile)

    # Get the fieldnames from the input file
    fieldnames = reader.fieldnames

    # Open both output files
    with open(no_match_file, 'w', newline='', encoding='utf-8') as no_match_out, \
         open(matched_file, 'w', newline='', encoding='utf-8') as matched_out:

        # Create writers for both output files
        no_match_writer = csv.DictWriter(no_match_out, fieldnames=fieldnames)
        matched_writer = csv.DictWriter(matched_out, fieldnames=fieldnames)

        # Write headers to both files
        no_match_writer.writeheader()
        matched_writer.writeheader()

        # Process each row
        no_match_count = 0
        matched_count = 0

        for row in reader:
            if row['match_type'] == 'no_match':
                no_match_writer.writerow(row)
                no_match_count += 1
            else:
                matched_writer.writerow(row)
                matched_count += 1

print(f"Processing complete!")
print(f"No match records: {no_match_count} -> saved to '{no_match_file}'")
print(f"Matched records: {matched_count} -> saved to '{matched_file}'")

# Input file name
input_file = 'matched_companies.csv'

# Output file name
output_file = 'matched_companies_filtered.csv'

# Columns to keep
columns_to_keep = [
    'id',
    'companyName',
    'gics_sector_name',
    'gics_indgrp_name',
    'country_iso'
]

# Read the input CSV and write filtered output
with open(input_file, 'r', encoding='utf-8') as infile:
    reader = csv.DictReader(infile)

    # Open output file
    with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=columns_to_keep)

        # Write header
        writer.writeheader()

        # Write filtered rows
        row_count = 0
        for row in reader:
            # Create a new row with only the columns we want
            filtered_row = {col: row[col] for col in columns_to_keep}
            writer.writerow(filtered_row)
            row_count += 1

print(f"Processing complete!")
print(f"Filtered {row_count} rows")
print(f"Output saved to '{output_file}' with columns: {', '.join(columns_to_keep)}")
