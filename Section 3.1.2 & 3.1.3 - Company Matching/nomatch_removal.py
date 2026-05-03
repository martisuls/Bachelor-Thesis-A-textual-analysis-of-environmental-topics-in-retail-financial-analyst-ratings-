import csv
import sys

# Increase the CSV field size limit to handle large content fields
csv.field_size_limit(sys.maxsize)

# Input files
no_match_file = 'no_match_companies.csv'
articles_file = 'articles_clean.csv'

# Output file
output_file = 'articles_clean_filtered.csv'

# Step 1: Read all IDs from no_match_companies.csv
print("Reading IDs from no_match_companies.csv...")
no_match_ids = set()

with open(no_match_file, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        no_match_ids.add(row['id'])

print(f"Found {len(no_match_ids)} unique IDs to remove")

# Step 2: Filter articles_clean.csv
print("Filtering articles_clean.csv...")

with open(articles_file, 'r', encoding='utf-8') as infile:
    reader = csv.DictReader(infile)
    fieldnames = reader.fieldnames

    with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()

        removed_count = 0
        kept_count = 0

        for row in reader:
            if row['id'] in no_match_ids:
                removed_count += 1
            else:
                writer.writerow(row)
                kept_count += 1

print(f"\nProcessing complete!")
print(f"Rows removed: {removed_count}")
print(f"Rows kept: {kept_count}")
print(f"Filtered data saved to '{output_file}'")
