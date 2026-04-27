import logging
import gensim
import pickle
import os
from gensim.models.word2vec import LineSentence
from datetime import datetime

# === CONFIGURATION ===
LOG_PATH = 'log'
DUMP_PATH = 'all_sentences'
NGRAM_PATH = 'bigram_trigram_model'
RESULT_PATH = 'word2vec_model'

# Word2Vec hyperparameters (UNCHANGED from paper)
VECTOR_SIZE = 300    # Dimensionality of word vectors
WINDOW_SIZE = 5      # Context window (5 words before/after)
MIN_COUNT = 5        # Ignore words appearing < 5 times
EPOCHS = 20          # Training iterations
THRESHOLD = 1        # For bigram/trigram model

os.makedirs(RESULT_PATH, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_PATH,'run_word2vec.log')),
        logging.StreamHandler()
    ]
)


def trim_word(w, count, min_count):
    """
    UNCHANGED: Original word filtering function
    Remove single letter words and stop words
    """
    if (w in gensim.parsing.preprocessing.STOPWORDS) or len(w) < 2:
        return gensim.utils.RULE_DISCARD
    else:
        return gensim.utils.RULE_DEFAULT


def run_word2vec():
    """
    OPTIMIZED: Train Word2Vec with progress tracking
    Core training logic UNCHANGED from original
    """

    out_file = os.path.join(DUMP_PATH, 'all.txt')
    processed_sentences = LineSentence(out_file)
    ngram_file_path = os.path.join(NGRAM_PATH, f'all_bigram_trigram_{THRESHOLD}.pkl')
    model_path = os.path.join(RESULT_PATH, 'all.word2vec')

    logging.info("=" * 60)
    logging.info("STEP 4: WORD2VEC TRAINING")
    logging.info("=" * 60)
    logging.info(f"Input: {out_file}")
    logging.info(f"Bigram/trigram model: {ngram_file_path}")
    logging.info(f"Output: {model_path}")
    logging.info("")
    logging.info("Training parameters:")
    logging.info(f"  Vector size: {VECTOR_SIZE}")
    logging.info(f"  Window size: {WINDOW_SIZE}")
    logging.info(f"  Min count: {MIN_COUNT}")
    logging.info(f"  Epochs: {EPOCHS}")
    logging.info(f"  Workers: 20 (parallel training)")
    logging.info("")
    logging.info("Expected time: 30-60 minutes")
    logging.info("=" * 60)

    # Check if model already exists
    if os.path.exists(model_path):
        logging.info("Word2Vec model already exists!")
        model = gensim.models.Word2Vec.load(model_path)
        logging.info(f"Loaded existing model:")
        logging.info(f"  Vocabulary size: {len(model.wv):,} words")
        logging.info(f"  Vector dimensions: {model.wv.vector_size}")
        logging.info("Skipping training...")
        return

    # Load bigram/trigram models
    logging.info("Loading bigram/trigram models...")
    with open(ngram_file_path, 'rb') as f:
        bigram, trigram = pickle.load(f)

    # UNCHANGED: Freeze models for faster processing
    bigram = bigram.freeze()
    trigram = trigram.freeze()

    # UNCHANGED: Inject ESG-specific phrases
    # These ensure certain ESG terms are always treated as single units
    logging.info("Injecting ESG-specific phrases...")
    bigram.phrasegrams['scope_1'] = float('inf')
    bigram.phrasegrams['scope_2'] = float('inf')
    bigram.phrasegrams['scope_3'] = float('inf')
    bigram.phrasegrams['ecological_impact'] = float('inf')

    bigram.phrasegrams['employee_engagement'] = float('inf')
    bigram.phrasegrams['customer_welfare'] = float('inf')
    bigram.phrasegrams['product_safety'] = float('inf')
    bigram.phrasegrams['responsible_marketing'] = float('inf')
    bigram.phrasegrams['product_quality'] = float('inf')

    bigram.phrasegrams['community_development'] = float('inf')
    bigram.phrasegrams['community_relation'] = float('inf')
    bigram.phrasegrams['social_capital'] = float('inf')
    bigram.phrasegrams['social_impact'] = float('inf')

    trigram.phrasegrams['supply_chain_sustainability'] = float('inf')

    logging.info("ESG phrases injected successfully")

    # Train Word2Vec model
    logging.info("")
    logging.info("Starting Word2Vec training...")
    logging.info("This will take 30-60 minutes. Progress updates every few minutes.")
    logging.info("")

    start_time = datetime.now()

    # UNCHANGED: Core Word2Vec training (same parameters as paper)
    model = gensim.models.Word2Vec(
        sentences=trigram[bigram[processed_sentences]],
        vector_size=VECTOR_SIZE,
        window=WINDOW_SIZE,
        min_count=MIN_COUNT,
        workers=20,
        epochs=EPOCHS,
        trim_rule=trim_word
    )

    end_time = datetime.now()
    training_time = (end_time - start_time).total_seconds() / 60

    # Save model
    logging.info("")
    logging.info("Training complete! Saving model...")
    model.save(model_path)

    file_size_mb = os.path.getsize(model_path) / (1024 * 1024)

    # Final statistics
    logging.info("=" * 60)
    logging.info("SUCCESS: Word2Vec model trained and saved!")
    logging.info("")
    logging.info(f"Training time: {training_time:.1f} minutes")
    logging.info(f"Model saved to: {model_path}")
    logging.info(f"File size: {file_size_mb:.2f} MB")
    logging.info("")
    logging.info("Model statistics:")
    logging.info(f"  Vocabulary size: {len(model.wv):,} words")
    logging.info(f"  Vector dimensions: {model.wv.vector_size}")
    logging.info(f"  Total training epochs: {model.epochs}")
    logging.info("")

    # Show some example word similarities
    logging.info("Sample word similarities (testing model quality):")
    test_words = ['climate_change', 'carbon', 'employee', 'sustainability']
    for word in test_words:
        if word in model.wv:
            similar = model.wv.most_similar(word, topn=5)
            logging.info(f"\n  Most similar to '{word}':")
            for sim_word, score in similar:
                logging.info(f"    {sim_word}: {score:.3f}")

    logging.info("")
    logging.info("=" * 60)
    logging.info("READY FOR STEP 5: Dictionary Generation")
    logging.info("=" * 60)


if __name__ == '__main__':
    run_word2vec()
