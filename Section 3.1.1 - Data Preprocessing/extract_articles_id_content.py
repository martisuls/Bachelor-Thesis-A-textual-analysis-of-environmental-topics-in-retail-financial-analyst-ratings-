import csv
import os
import sys


INPUT_CSV = "articles_clean_filtered.csv"
OUTPUT_CSV = "articles_id_content.csv"

KEEP_COLS = ["id", "content"]


def main() -> None:
    # Allow large content cells (some HTML-cleaned reports are very long)
    csv.field_size_limit(sys.maxsize)

    if not os.path.exists(INPUT_CSV):
        sys.exit(
            f"ERROR: cannot find {INPUT_CSV} in {os.getcwd()}.\n"
            "Run Section 3.3 (nomatch_removal.py) first to produce it."
        )

    rows_written = 0
    with open(INPUT_CSV, "r", encoding="utf-8", newline="") as infile:
        reader = csv.DictReader(infile)

        missing = [c for c in KEEP_COLS if c not in (reader.fieldnames or [])]
        if missing:
            sys.exit(f"ERROR: input file is missing required columns: {missing}")

        with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as outfile:
            writer = csv.DictWriter(outfile, fieldnames=KEEP_COLS)
            writer.writeheader()
            for row in reader:
                writer.writerow({c: row[c] for c in KEEP_COLS})
                rows_written += 1

    print(f"Extracted (id, content) -> {os.path.abspath(OUTPUT_CSV)}")
    print(f"Rows written: {rows_written:,}")


if __name__ == "__main__":
    main()
