import pandas as pd
import re
from collections import Counter
import time
import multiprocessing as mp

# =========================
# Globals for worker pool
# =========================
compiled_patterns = None   # list of tuples: (term_original, category, compiled_regex)
all_categories = None      # set of categories present in dictionary


def init_worker(patterns, categories):
    """Initializer for each worker process to share compiled patterns."""
    global compiled_patterns, all_categories
    compiled_patterns = patterns
    all_categories = categories


# -------------------------
# Pattern construction
# -------------------------
def _plural_last_token_regex(token: str) -> str:
    """
    Very small pluralization helper for the LAST token only:
      - energy -> (?:energy|energies)
      - class -> classes?  (allow optional 'es' or 's')
      - box   -> boxes?
      - match -> matches?
      - gas   -> gases?
      - default: s?
    """
    t = re.escape(token)
    # Common endings that take 'es'
    es_endings = ("s", "x", "z", "ch", "sh")
    if token.endswith("y") and len(token) > 1 and token[-2] not in "aeiou":
        # consonant + y -> y | ies
        return r"(?:%s|%s)" % (re.escape(token), re.escape(token[:-1] + "ies"))
    elif token.endswith(es_endings):
        return r"%s(?:es)?" % t
    else:
        return r"%s(?:s)?" % t


def build_compiled_patterns(term_to_category):
    """
    Build compiled regex patterns (longest-first).
    Dictionary terms:
      - underscores treated as spaces in the term
      - allow either spaces or hyphens between tokens in text: (?:\\s+|-)
      - pluralize ONLY the last token
      - match with word boundaries
    """
    patterns = []

    for term, category in term_to_category.items():
        term_norm = term.strip().lower().replace("_", " ")
        tokens = term_norm.split()
        if not tokens:
            continue

        # pluralize only the last token
        if len(tokens) == 1:
            parts = [_plural_last_token_regex(tokens[0])]
        else:
            parts = [re.escape(t) for t in tokens[:-1]]
            parts.append(_plural_last_token_regex(tokens[-1]))

        sep = r"(?:\s+|-)"  # allow spaces or hyphens between tokens
        pattern_str = r"\b" + sep.join(parts) + r"\b"
        pattern = re.compile(pattern_str, re.IGNORECASE)

        patterns.append((term, category, pattern, len(term)))

    # longest-first
    patterns.sort(key=lambda x: x[3], reverse=True)
    patterns = [(term, cat, pat) for term, cat, pat, _ in patterns]
    return patterns


# -------------------------
# Matching core (worker)
# -------------------------
def tokenize_batch(batch_data):
    """
    Process a batch of (id, content) with LONGEST-MATCH-FIRST and
    non-overlapping span selection.
    """
    global compiled_patterns, all_categories
    results = []

    # fixed output categories (keep your original columns)
    fixed_categories = [
        'Climate_Change',
        'Natural_Resources',
        'Environmental_Goals',
        'Pollution',
        'Sustainable_Reporting',
        'Ecosystem'
    ]

    for report_id, content in batch_data:
        if pd.isna(content):
            content = ""

        text = str(content).lower()
        total_word_count = len(text.split())

        # Initialize counts
        category_counts = {c: 0 for c in fixed_categories}
        all_terms = []

        # Track occupied character spans to prevent overlaps
        occupied = []  # list of (start, end) half-open

        def overlaps(span):
            s, e = span
            for os, oe in occupied:
                if not (e <= os or oe <= s):
                    return True
            return False

        # Greedy longest-first: earlier (longer) patterns fill space first
        for term, category, pattern in compiled_patterns:
            for m in pattern.finditer(text):
                span = m.span()
                if overlaps(span):
                    continue
                occupied.append(span)
                all_terms.append(term)

                # Only increment known/fixed columns; ignore unexpected cats
                if category in category_counts:
                    category_counts[category] += 1

        env_term_count = len(all_terms)
        unique_env_terms = len(set(all_terms))
        has_env_content = 1 if env_term_count > 0 else 0
        env_density_per_1000 = (env_term_count / total_word_count * 1000) if total_word_count > 0 else 0.0
        category_diversity = sum(1 for c in category_counts.values() if c > 0)
        dominant_category = max(category_counts.items(), key=lambda x: x[1])[0] if env_term_count > 0 else 'None'

        results.append({
            'id': report_id,
            'has_env_content': has_env_content,
            'env_term_count': env_term_count,
            'unique_env_terms': unique_env_terms,
            'env_density_per_1000': round(env_density_per_1000, 2),
            'climate_change_count': category_counts['Climate_Change'],
            'natural_resources_count': category_counts['Natural_Resources'],
            'environmental_goals_count': category_counts['Environmental_Goals'],
            'pollution_count': category_counts['Pollution'],
            'sustainable_reporting_count': category_counts['Sustainable_Reporting'],
            'ecosystem_count': category_counts['Ecosystem'],
            'total_word_count': total_word_count,
            'category_diversity': category_diversity,
            'dominant_category': dominant_category
        })

    return results


# -------------------------
# Main
# -------------------------
def main():
    print("=" * 70)
    print("ENVIRONMENTAL TOKENIZATION - LONGEST-MATCH-FIRST (underscore->space/hyphen)")
    print("=" * 70)

    # Detect CPU cores
    NUM_CORES = mp.cpu_count()
    print(f"Detected {NUM_CORES} CPU cores - will use all for parallel processing")

    # Load dictionary
    start_time = time.time()
    dictionary_df = pd.read_csv('environmental_dictionary_cleaned.csv')
    print(f"Loaded {len(dictionary_df)} environmental terms from dictionary")

    # Map term -> category
    term_to_category = dict(zip(dictionary_df['word'].astype(str).str.strip(),
                                dictionary_df['category'].astype(str).str.strip()))

    # Build patterns (longest-first)
    print("Precompiling regex patterns with underscore->space/hyphen and longest-match priority...")
    patterns = build_compiled_patterns(term_to_category)
    all_cats = set(dictionary_df['category'].astype(str).unique())
    print(f"Compiled and sorted {len(patterns)} patterns (longest first)")
    print(f"   Longest example: '{patterns[0][0]}' ({len(patterns[0][0])} chars)")

    # Load reports
    print(f"Loading 'articles_id_content.csv'...")
    reports_df = pd.read_csv('articles_id_content.csv', dtype={'id': str, 'content': str})
    total_reports = len(reports_df)
    print(f"Loaded {total_reports:,} reports")
    print(f"Initialization: {time.time() - start_time:.2f}s\n")

    # Prepare batches
    print("=" * 70)
    print("PROCESSING WITH MULTIPROCESSING")
    print("=" * 70)

    batch_size = max(1, total_reports // (NUM_CORES * 4))  # ~4 batches per core
    data_tuples = list(zip(reports_df['id'], reports_df['content']))

    batches = []
    for i in range(0, len(data_tuples), batch_size):
        batches.append(data_tuples[i:i + batch_size])

    print(f"Split into {len(batches)} batches (~{batch_size:,} reports/batch)")
    print(f"Starting parallel processing on {NUM_CORES} cores...\n")

    process_start = time.time()

    all_results = []
    with mp.Pool(processes=NUM_CORES, initializer=init_worker, initargs=(patterns, all_cats)) as pool:
        completed = 0
        for batch_results in pool.imap_unordered(tokenize_batch, batches):
            all_results.extend(batch_results)
            completed += len(batch_results)

            elapsed = time.time() - process_start
            rps = completed / elapsed if elapsed > 0 else 0
            remaining = total_reports - completed
            eta = remaining / rps if rps > 0 else 0
            pct = completed / total_reports * 100

            print(f"Progress: {completed:>7,}/{total_reports:,} ({pct:>5.1f}%) | "
                  f"{rps:>6.0f} reports/sec | ETA: {eta:>5.0f}s",
                  end='\r', flush=True)

    process_time = time.time() - process_start
    print(f"\n\nProcessing complete: {process_time:.2f}s ({total_reports / process_time:,.0f} reports/sec)")

    # Create output dataframe
    output_df = pd.DataFrame(all_results)

    # -------------------------
    # DIAGNOSTIC REPORT
    # -------------------------
    print("\n" + "=" * 70)
    print("DIAGNOSTIC REPORT")
    print("=" * 70)

    reports_with_env = output_df['has_env_content'].sum()
    reports_without_env = total_reports - reports_with_env

    print(f"\nENVIRONMENTAL CONTENT PREVALENCE:")
    print(f"  Reports WITH environmental terms:    {reports_with_env:>8,} ({reports_with_env/total_reports*100:>5.2f}%)")
    print(f"  Reports WITHOUT environmental terms: {reports_without_env:>8,} ({reports_without_env/total_reports*100:>5.2f}%)")

    env_reports = output_df[output_df['has_env_content'] == 1]

    if len(env_reports) > 0:
        print(f"\nENVIRONMENTAL TERM STATISTICS ({len(env_reports):,} reports with content):")
        print(f"  Total env terms found:       {env_reports['env_term_count'].sum():>10,}")
        print(f"  Mean terms per report:       {env_reports['env_term_count'].mean():>10.2f}")
        print(f"  Median terms per report:     {env_reports['env_term_count'].median():>10.2f}")
        print(f"  Max terms in a report:       {env_reports['env_term_count'].max():>10,}")

        print(f"\nUNIQUE TERM STATISTICS:")
        print(f"  Mean unique terms/report:    {env_reports['unique_env_terms'].mean():>10.2f}")
        print(f"  Median unique terms/report:  {env_reports['unique_env_terms'].median():>10.2f}")
        print(f"  Max unique terms:            {env_reports['unique_env_terms'].max():>10,}")

        print(f"\nENVIRONMENTAL DENSITY:")
        print(f"  Mean (terms/1000 words):     {env_reports['env_density_per_1000'].mean():>10.2f}")
        print(f"  Median density:              {env_reports['env_density_per_1000'].median():>10.2f}")
        print(f"  Max density:                 {env_reports['env_density_per_1000'].max():>10.2f}")

    print(f"\nCATEGORY DISTRIBUTION:")
    categories = [
        ('Climate Change', 'climate_change_count'),
        ('Natural Resources', 'natural_resources_count'),
        ('Environmental Goals', 'environmental_goals_count'),
        ('Pollution', 'pollution_count'),
        ('Sustainable Reporting', 'sustainable_reporting_count'),
        ('Ecosystem', 'ecosystem_count')
    ]
    for cat_name, cat_col in categories:
        count = (output_df[cat_col] > 0).sum()
        pct = count / total_reports * 100
        total_mentions = output_df[cat_col].sum()
        print(f"  {cat_name:<25} {count:>8,} reports ({pct:>5.2f}%) | {total_mentions:>10,} mentions")

    print(f"\nCATEGORY DIVERSITY:")
    diversity_dist = output_df['category_diversity'].value_counts().sort_index()
    for div, count in diversity_dist.items():
        print(f"  {div} categories: {count:>8,} reports ({count/total_reports*100:>5.2f}%)")

    print(f"\nDOMINANT CATEGORIES ({len(env_reports):,} reports with env content):")
    if len(env_reports) > 0:
        dominant_dist = env_reports['dominant_category'].value_counts()
        for cat, count in dominant_dist.items():
            print(f"  {cat:<30} {count:>8,} reports ({count/len(env_reports)*100:>5.2f}%)")

    print(f"\nTOP 10 REPORTS BY ENVIRONMENTAL CONTENT:")
    top_10 = output_df.nlargest(10, 'env_term_count')[
        ['id', 'env_term_count', 'unique_env_terms', 'env_density_per_1000', 'dominant_category']
    ]
    print(top_10.to_string(index=False))

    # Save output
    output_filename = 'environmental_metrics_outputGPT.csv'
    print(f"\n" + "=" * 70)
    print("SAVING RESULTS...")
    save_start = time.time()
    output_df.to_csv(output_filename, index=False)
    save_time = time.time() - save_start

    print(f"Saved to: {output_filename}")
    print(f"   Rows: {len(output_df):,} | Columns: {len(output_df.columns)}")
    print(f"   Save time: {save_time:.2f}s")

    total_time = time.time() - start_time
    print(f"\nTOTAL EXECUTION TIME: {total_time:.2f}s ({total_time/60:.2f} min)")
    print(f"THROUGHPUT: {total_reports/total_time:,.0f} reports/second")
    print("=" * 70)


if __name__ == '__main__':
    mp.freeze_support()  # for Windows safety
    main()
