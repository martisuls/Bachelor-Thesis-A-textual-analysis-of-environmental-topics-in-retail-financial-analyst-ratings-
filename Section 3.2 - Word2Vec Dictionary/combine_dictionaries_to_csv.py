#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Combine all dictionary .txt files into a single CSV for review

Reads from: words/*.txt
Outputs: environmental_dictionary.csv
"""

import os
import pandas as pd
from glob import glob

# Configuration
WORDS_FOLDER = 'words'
OUTPUT_CSV = 'environmental_dictionary.csv'

def combine_dictionaries():
    """
    Read all .txt files from words/ folder and combine into CSV
    """

    print("=" * 60)
    print("COMBINING DICTIONARIES TO CSV")
    print("=" * 60)

    # Find all .txt files
    txt_files = glob(os.path.join(WORDS_FOLDER, '*.txt'))

    if not txt_files:
        print(f"ERROR: No .txt files found in {WORDS_FOLDER}/")
        return

    print(f"Found {len(txt_files)} dictionary files:")
    for f in txt_files:
        print(f"  - {os.path.basename(f)}")
    print()

    # Collect all words
    all_words = []

    for txt_file in txt_files:
        # Get category name from filename
        category = os.path.basename(txt_file).replace('.txt', '')

        # Read words from file
        with open(txt_file, 'r', encoding='utf-8') as f:
            words = [line.strip() for line in f if line.strip()]

        # Add to list with category
        for word in words:
            all_words.append({
                'word': word,
                'category': category
            })

        print(f"{category}: {len(words)} words")

    # Create DataFrame
    df = pd.DataFrame(all_words)

    # Save to CSV
    df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8')

    print()
    print("=" * 60)
    print("SUCCESS!")
    print(f"Total words: {len(df):,}")
    print(f"Categories: {df['category'].nunique()}")
    print(f"Output saved to: {OUTPUT_CSV}")
    print()
    print("Category breakdown:")
    print(df['category'].value_counts().to_string())
    print("=" * 60)


if __name__ == '__main__':
    combine_dictionaries()
