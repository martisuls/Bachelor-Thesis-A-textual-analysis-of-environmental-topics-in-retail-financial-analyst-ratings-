#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STEP 5: Generate expanded ESG dictionary from seed words

WHAT THIS DOES:
- Loads your trained Word2Vec model
- Takes seed words from seedwords.py (e.g., "carbon_footprint")
- Finds top 500 most similar words for each category
- Removes duplicate words (assigns to most relevant category)
- Saves expanded dictionaries to words/ folder

EXAMPLE:
Input seed: "carbon_footprint"
Expands to: carbon_emission, greenhouse_gas, co2, emission_reduction, etc.

=== CHANGES FROM ORIGINAL ===
1. UNCHANGED: Core dictionary expansion logic (same algorithm)
2. ADDED: Progress tracking and statistics
3. ADDED: Sample word output for verification
4. ADDED: Category statistics and overlap reporting


@author: Optimized from Yan LIN's code
"""

import gensim
import os
import numpy as np
import logging
from datetime import datetime

# === CONFIGURATION ===
RESULT_PATH = 'word2vec_model'
OUTPUT_PATH = '.'
LOG_PATH = 'log'

TOP_N = 50

# Import seed words
from seedwords import SEED_WORD_DICT

os.makedirs(os.path.join(OUTPUT_PATH, 'words'), exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_PATH, 'gen_dictionary_'+ f"{datetime.today().strftime('%Y-%m-%d_%H-%M-%S')}" + '.log')),
        logging.StreamHandler()
    ]
)


def sort_word_list(keywords_list: dict) -> dict:
    """
    UNCHANGED: Original function to remove duplicate words
    Assigns each word to the category where it has highest similarity

    If "emission" appears in both Climate_Change and Pollution_Waste,
    it goes to whichever has higher similarity score
    """
    word_category_simi = {}  # word --> [(category, similarity), ...]

    # Collect all categories for each word
    for keyword, v in keywords_list.items():
        for w in v:
            if w[0] in word_category_simi:
                word_category_simi[w[0]].append((keyword, w[1]))
            else:
                word_category_simi[w[0]] = [(keyword, w[1])]

    # Sort by similarity (highest first)
    for k, v in word_category_simi.items():
        v.sort(reverse=True, key=lambda x: x[1])

    # Assign each word to its best category
    keywords_list = {}
    for word, category_simi in word_category_simi.items():
        category = category_simi[0][0]
        similarity = category_simi[0][1]
        if category in keywords_list:
            keywords_list[category].append((word, similarity))
        else:
            keywords_list[category] = [(word, similarity)]

    # Sort each category by similarity
    for k, v in keywords_list.items():
        v.sort(reverse=True, key=lambda x: x[1])

    # Convert to word-only lists
    results = {}
    for k, v in keywords_list.items():
        results[k] = [x[0] for x in v]

    return results


def get_word_list():
    """
    OPTIMIZED: Expand seed words with progress tracking
    Core expansion logic UNCHANGED
    """

    logging.info("=" * 60)
    logging.info("STEP 5: ESG DICTIONARY GENERATION")
    logging.info("=" * 60)
    logging.info(f"Input: Seed words from seedwords.py")
    logging.info(f"Model: {os.path.join(RESULT_PATH, 'all.word2vec')}")
    logging.info(f"Output: words/*.txt (one file per category)")
    logging.info(f"Expansion: Top {TOP_N} similar words per category")
    logging.info("=" * 60)

    # Show seed word categories
    logging.info("\nSeed word categories:")
    for category, words in SEED_WORD_DICT.items():
        logging.info(f"  {category}: {len(words)} seed words")
    logging.info("")

    # Preprocess seed words (lowercase, strip)
    seed_word_dict_new = {}
    for k, word_list in SEED_WORD_DICT.items():
        seed_word_dict_new[k] = [w.lower().strip() for w in word_list]
    seed_word_dict = seed_word_dict_new

    # Load Word2Vec model
    logging.info("Loading Word2Vec model...")
    model = gensim.models.Word2Vec.load(os.path.join(RESULT_PATH, 'all.word2vec'))
    logging.info(f"Model loaded: {len(model.wv):,} words in vocabulary")
    logging.info("")

    # Expand each category
    keywords_list = {}
    total_before_dedup = 0

    logging.info("Expanding seed words to find similar words...")
    logging.info("")

    for k, word_list in seed_word_dict.items():
        logging.info(f"Processing category: {k}")
        logging.info(f"  Seed words: {', '.join(word_list[:5])}{'...' if len(word_list) > 5 else ''}")

        keywords_list[k] = []
        l_simi = []  # Similarity scores for all words to each seed word

        found_seeds = 0
        missing_seeds = []

        # For each seed word, get similarities to all vocabulary words
        for w in word_list:
            if w in model.wv:
                # Get similarity of all words to this seed word
                l_simi.append(1 - model.wv.distances(w))
                found_seeds += 1
            else:
                missing_seeds.append(w)

        if missing_seeds:
            logging.info(f"  Warning: {len(missing_seeds)} seed words not in model: {', '.join(missing_seeds[:3])}{'...' if len(missing_seeds) > 3 else ''}")

        if len(l_simi) == 0:
            logging.info(f"  ERROR: No seed words found in model for {k}!")
            continue

        # Stack similarities and take maximum for each word
        # (if a word is similar to ANY seed word, use that similarity)
        l_simi = np.stack(l_simi, axis=1)
        l_simi = l_simi.max(axis=1)

        # Get top N most similar words
        indexes = np.argpartition(l_simi, -TOP_N)[-TOP_N:]
        keywords_list[k] = list(zip([model.wv.index_to_key[x] for x in indexes], l_simi[indexes]))

        total_before_dedup += len(keywords_list[k])

        logging.info(f"  Found {len(keywords_list[k])} similar words")
        logging.info(f"  Top 5 matches: {', '.join([w for w, s in sorted(keywords_list[k], key=lambda x: -x[1])[:5]])}")
        logging.info("")

    # Remove duplicates (assign each word to best category)
    logging.info("Removing duplicate words across categories...")
    logging.info(f"  Total words before deduplication: {total_before_dedup:,}")

    keywords_list = sort_word_list(keywords_list)

    total_after_dedup = sum(len(v) for v in keywords_list.values())
    duplicates_removed = total_before_dedup - total_after_dedup

    logging.info(f"  Total words after deduplication: {total_after_dedup:,}")
    logging.info(f"  Duplicates removed: {duplicates_removed:,}")
    logging.info("")

    # Save dictionaries
    logging.info("Saving dictionaries to files...")
    for k, word_list in keywords_list.items():
        output_file = os.path.join(OUTPUT_PATH, 'words', k + '.txt')
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(word_list))
        logging.info(f"  {k}: {len(word_list)} words -> {output_file}")

    logging.info("")
    logging.info("=" * 60)
    logging.info("SUCCESS: ESG Dictionary Generated!")
    logging.info("")
    logging.info("Final statistics:")
    for k, word_list in keywords_list.items():
        logging.info(f"  {k}: {len(word_list):,} words")
    logging.info(f"  TOTAL: {total_after_dedup:,} unique ESG keywords")
    logging.info("")
    logging.info("Dictionary files saved in: words/")
    logging.info("=" * 60)
    logging.info("READY FOR STEP 6: Analyze Documents")
    logging.info("=" * 60)


if __name__ == "__main__":
    get_word_list()
