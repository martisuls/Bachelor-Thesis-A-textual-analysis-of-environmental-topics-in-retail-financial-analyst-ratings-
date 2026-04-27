#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
STEP 2: Dump all processed sentences to single text file

WHAT THIS DOES:
- Reads all .feather files from data/processed_df/
- Extracts processed sentences
- Writes to all_sentences/all.txt (one sentence per line)
- Creates input format for Word2Vec training

=== OPTIMIZATIONS ===
1. Buffered writing (faster I/O)
2. Progress tracking
3. Memory efficient (processes one file at a time)

STEP 2 (Dump Sentences) - Changes from Original:
CHANGED:

Added: Buffered writing (1MB buffer for faster I/O)
Added: Progress bar with tqdm
Added: Detailed statistics (doc count, sentence count, file size)
Added: Better logging messages

UNCHANGED:

Core logic: reads .feather files, extracts sentences, writes to all.txt
Output format: one sentence per line, space-separated words
File paths and structure

@author: Optimized from Yan LIN's code
"""

import pandas as pd
from glob import glob
import logging
import os
from tqdm import tqdm

# === CONFIGURATION ===
PROCESSED_DF_PATH = 'data/processed_df'
DUMP_PATH = 'all_sentences'
LOG_PATH = 'log'

os.makedirs(DUMP_PATH, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_PATH,'dumping.log')),
        logging.StreamHandler()
    ]
)


def run_dump_all_sentences():
    """
    OPTIMIZED: Faster writing with buffering and progress tracking
    """

    # Find all processed feather files
    process_files = sorted(glob(os.path.join(PROCESSED_DF_PATH,'*.feather')))
    out_file = os.path.join(DUMP_PATH,'all.txt')

    logging.info("=" * 60)
    logging.info("STEP 2: DUMPING SENTENCES")
    logging.info("=" * 60)
    logging.info(f"Input: {len(process_files)} .feather files")
    logging.info(f"Output: {out_file}")
    logging.info(f"Expected time: 5-10 minutes for 360K docs")
    logging.info("=" * 60)

    total_docs = 0
    total_sentences = 0

    # OPTIMIZATION: Buffered writing (faster I/O)
    with open(out_file, 'w', encoding='utf-8', buffering=1024*1024) as f:

        # Process each feather file with progress bar
        for f_name in tqdm(process_files, desc="Processing chunks"):

            # Read feather file
            df = pd.read_feather(f_name)

            # Extract and write sentences
            for doc in df.processed_docs.tolist():
                for sentence in doc:
                    # Skip empty sentences
                    if len(sentence) == 0:
                        continue

                    # Write sentence as space-separated words
                    f.write(' '.join(sentence) + '\n')
                    total_sentences += 1

            total_docs += df.shape[0]

    # Final statistics
    file_size_mb = os.path.getsize(out_file) / (1024 * 1024)

    logging.info("=" * 60)
    logging.info("SUCCESS: Dumping complete!")
    logging.info(f"Documents processed: {total_docs:,}")
    logging.info(f"Total sentences: {total_sentences:,}")
    logging.info(f"Avg sentences/doc: {total_sentences/total_docs:.1f}")
    logging.info(f"Output file size: {file_size_mb:.2f} MB")
    logging.info(f"Output file: {out_file}")
    logging.info("=" * 60)


if __name__ == '__main__':
    run_dump_all_sentences()
