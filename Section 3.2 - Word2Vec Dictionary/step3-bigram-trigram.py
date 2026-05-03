import logging
import os
from gensim.models import Phrases
from gensim.models.phrases import Phraser, ENGLISH_CONNECTOR_WORDS
from gensim.models.word2vec import LineSentence
import pickle

# === CONFIGURATION ===
LOG_PATH = 'log'
DUMP_PATH = 'all_sentences'
NGRAM_PATH = 'bigram_trigram_model'

MIN_COUNT = 5   # Phrase must appear at least 5 times
THRESHOLD = 1   # Lower = more phrases detected (1 = very permissive)

os.makedirs(NGRAM_PATH, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_PATH,'get_bigram_trigram.log')),
        logging.StreamHandler()
    ]
)


def run_bigram_trigram():
    """
    OPTIMIZED: Build bigram and trigram models with progress tracking
    """

    out_file = os.path.join(DUMP_PATH,'all.txt')
    ngram_file_path = os.path.join(NGRAM_PATH, f'all_bigram_trigram_{THRESHOLD}.pkl')

    logging.info("=" * 60)
    logging.info("STEP 3: BIGRAM/TRIGRAM DETECTION")
    logging.info("=" * 60)
    logging.info(f"Input: {out_file}")
    logging.info(f"Min count: {MIN_COUNT} (phrase must appear 5+ times)")
    logging.info(f"Threshold: {THRESHOLD} (lower = more phrases)")
    logging.info(f"Expected time: 15-30 minutes")
    logging.info("=" * 60)

    # Check if already exists
    if os.path.exists(ngram_file_path):
        logging.info("Bigram/trigram model already exists. Skipping...")

        # Show what's in it
        with open(ngram_file_path, 'rb') as f:
            bigram, trigram = pickle.load(f)
        logging.info(f"Loaded existing model:")
        logging.info(f"  Bigram vocab size: {len(bigram.vocab):,}")
        logging.info(f"  Trigram vocab size: {len(trigram.vocab):,}")
        return

    # Load sentences
    logging.info("Loading sentences (streaming mode)...")
    processed_sentences = LineSentence(out_file)

    # STEP 3A: Build bigram model
    logging.info("Building BIGRAM model (pass 1/2)...")
    logging.info("This scans all 20M sentences to find 2-word phrases...")

    bigram = Phrases(
        processed_sentences,
        min_count=MIN_COUNT,
        connector_words=ENGLISH_CONNECTOR_WORDS,
        delimiter="_",
        threshold=THRESHOLD
    )

    logging.info(f"Bigram model complete! Found {len(bigram.vocab):,} vocabulary items")

    # STEP 3B: Build trigram model on top of bigrams
    logging.info("Building TRIGRAM model (pass 2/2)...")
    logging.info("This scans sentences again to find 3-word phrases...")

    # Reload sentences (LineSentence is a generator, need fresh one)
    processed_sentences = LineSentence(out_file)

    trigram = Phrases(
        bigram[processed_sentences],  # Apply bigram first, then find trigrams
        min_count=MIN_COUNT,
        connector_words=ENGLISH_CONNECTOR_WORDS,
        delimiter="_",
        threshold=THRESHOLD
    )

    logging.info(f"Trigram model complete! Found {len(trigram.vocab):,} vocabulary items")

    # Save both models
    logging.info("Saving models...")
    with open(ngram_file_path,'wb') as f:
        pickle.dump((bigram, trigram), f)

    file_size_mb = os.path.getsize(ngram_file_path) / (1024 * 1024)

    logging.info("=" * 60)
    logging.info("SUCCESS: Bigram/Trigram models saved!")
    logging.info(f"Output file: {ngram_file_path}")
    logging.info(f"File size: {file_size_mb:.2f} MB")
    logging.info(f"Bigram vocab: {len(bigram.vocab):,} items")
    logging.info(f"Trigram vocab: {len(trigram.vocab):,} items")

    # Show sample phrases detected
    logging.info("=" * 60)
    logging.info("SAMPLE PHRASES DETECTED:")

    # Reload sentences for sampling
    processed_sentences = LineSentence(out_file)
    sample_bigrams = list(bigram.export_phrases(processed_sentences))[:20]

    logging.info("\nTop 20 bigrams:")
    for phrase, score in sample_bigrams:
        phrase_str = phrase[0].decode('utf-8') + '_' + phrase[1].decode('utf-8')
        logging.info(f"  {phrase_str} (score: {score:.2f})")

    logging.info("=" * 60)


if __name__ == '__main__':
    run_bigram_trigram()
