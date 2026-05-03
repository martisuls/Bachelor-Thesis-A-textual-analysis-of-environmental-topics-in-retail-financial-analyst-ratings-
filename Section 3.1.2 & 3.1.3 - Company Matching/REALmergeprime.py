import pandas as pd
import numpy as np
from rapidfuzz import fuzz, process
import re
from tqdm import tqdm
import time
from collections import defaultdict
import multiprocessing as mp
from functools import partial
import warnings
warnings.filterwarnings('ignore')

# Common words that shouldn't be primary match targets
COMMON_CORPORATE_WORDS = {
    'corporation', 'corp', 'incorporated', 'inc', 'limited', 'ltd', 'company', 'co',
    'group', 'holdings', 'international', 'intl', 'partners', 'partnership',
    'aktiengesellschaft', 'ag', 'gmbh', 'brands', 'enterprises', 'industries',
    'systems', 'solutions', 'services', 'technologies', 'tech', 'global',
    'worldwide', 'usa', 'america', 'american', 'united', 'national', 'general',
    'first', 'new', 'capital', 'financial', 'investment', 'management',
    'sa', 'spa', 'srl', 'nv', 'bv', 'plc', 'llc', 'lp', 'llp', 'pvt', 'pte', 'pty',
    'bank', 'trust', 'fund', 'holding', 'subsidiary'
}

def get_distinctive_tokens(name):
    """
    Extract the distinctive (non-common) tokens from a company name
    """
    if not name:
        return []

    # Tokenize
    tokens = re.findall(r'\b[a-z]+\b', name.lower())

    # Filter out common corporate words and very short tokens
    distinctive = [t for t in tokens
                   if t not in COMMON_CORPORATE_WORDS and len(t) > 2]

    return distinctive

def smart_normalize(name, preserve_suffix=True):
    """
    Improved normalization that preserves distinctive company identifiers
    """
    if pd.isna(name) or not name:
        return '', []

    # Basic cleaning
    name = str(name).strip()
    clean_name = name.lower()

    # Remove extra spaces and standardize punctuation
    clean_name = re.sub(r'\s+', ' ', clean_name)
    clean_name = re.sub(r'[^\w\s]', ' ', clean_name)
    clean_name = re.sub(r'\s+', ' ', clean_name).strip()

    # Get distinctive tokens for matching
    distinctive = get_distinctive_tokens(clean_name)

    if not preserve_suffix:
        # Remove common suffixes
        for word in COMMON_CORPORATE_WORDS:
            clean_name = re.sub(r'\b' + word + r'\b', '', clean_name)
        clean_name = re.sub(r'\s+', ' ', clean_name).strip()

    return clean_name, distinctive

def extract_suffix(name):
    """
    Extract standardized suffix
    """
    if not name:
        return ''

    name_lower = str(name).lower()

    # Expanded suffix patterns
    suffix_patterns = {
        r'\b(inc\.?|incorporated)$': 'inc',
        r'\b(corp\.?|corporation)$': 'corp',
        r'\b(ltd\.?|limited)$': 'ltd',
        r'\b(aktiengesellschaft|ag)$': 'ag',
        r'\b(gmbh|gesellschaft mit beschrankter haftung)$': 'gmbh',
        r'\b(llc|l\.l\.c\.?)$': 'llc',
        r'\b(plc|p\.l\.c\.?)$': 'plc',
        r'\b(sa|s\.a\.?)$': 'sa',
        r'\b(spa|s\.p\.a\.?)$': 'spa',
    }

    for pattern, standard in suffix_patterns.items():
        if re.search(pattern, name_lower, re.IGNORECASE):
            return standard

    return ''

def token_similarity(tokens1, tokens2):
    """
    Calculate similarity based on distinctive tokens
    """
    if not tokens1 or not tokens2:
        return 0

    set1 = set(tokens1)
    set2 = set(tokens2)

    # Jaccard similarity with length consideration
    intersection = len(set1 & set2)
    union = len(set1 | set2)

    if union == 0:
        return 0

    jaccard = intersection / union

    # Bonus for matching important tokens (first tokens)
    first_match_bonus = 0
    if tokens1 and tokens2:
        if tokens1[0] == tokens2[0]:
            first_match_bonus = 0.3

    return min(jaccard + first_match_bonus, 1.0)

def prepare_compustat_data(compustat_df):
    """
    Prepare data with improved indexing
    """
    print("Preparing Compustat data with smart normalization...")

    # Sort by date
    compustat_df = compustat_df.sort_values('datadate', ascending=False)

    # Apply smart normalization
    print("  Normalizing company names...")
    norm_results = compustat_df['company_name'].apply(
        lambda x: smart_normalize(x, preserve_suffix=True)
    )
    compustat_df['normalized'] = norm_results.apply(lambda x: x[0])
    compustat_df['distinctive_tokens'] = norm_results.apply(lambda x: x[1])

    # Base name normalization
    base_results = compustat_df['company_name'].apply(
        lambda x: smart_normalize(x, preserve_suffix=False)
    )
    compustat_df['base_normalized'] = base_results.apply(lambda x: x[0])
    compustat_df['base_tokens'] = base_results.apply(lambda x: x[1])

    # Extract suffix
    compustat_df['suffix'] = compustat_df['company_name'].apply(extract_suffix)

    # Drop duplicates
    compustat_df = compustat_df.drop_duplicates(
        subset=['company_name', 'country_iso'],
        keep='first'
    ).reset_index(drop=True)

    # Create multiple lookup structures
    print("  Building lookup indices...")

    # 1. Exact lookup (normalized with suffix)
    exact_lookup = {}
    for idx, row in compustat_df.iterrows():
        if row['normalized']:
            exact_lookup[row['normalized']] = idx

    # 2. Token-based lookup (for smart fuzzy matching)
    token_lookup = defaultdict(list)
    for idx, row in compustat_df.iterrows():
        for token in row['distinctive_tokens']:
            if token:
                token_lookup[token].append(idx)

    # 3. Base name lookup
    base_lookup = defaultdict(list)
    for idx, row in compustat_df.iterrows():
        if row['base_normalized']:
            base_lookup[row['base_normalized']].append(idx)

    print(f"Indexed {len(exact_lookup):,} unique companies")
    print(f"Created {len(token_lookup):,} token indices")
    print(f"Found {len(base_lookup):,} unique base names")

    return compustat_df, exact_lookup, token_lookup, base_lookup

def find_best_match_improved(company_name, compustat_df, exact_lookup, token_lookup,
                             base_lookup, threshold=75):
    """
    Improved matching that avoids bad fuzzy matches
    """
    if not company_name:
        return None

    # Normalize input
    norm_full, input_tokens = smart_normalize(company_name, preserve_suffix=True)
    norm_base, base_tokens = smart_normalize(company_name, preserve_suffix=False)
    input_suffix = extract_suffix(company_name)

    # Stage 1: Exact match
    if norm_full in exact_lookup:
        idx = exact_lookup[norm_full]
        return {
            'idx': idx,
            'score': 100,
            'type': 'exact'
        }

    # Stage 2: Token-based matching (avoids bad fuzzy matches)
    if input_tokens:
        # Find candidates that share distinctive tokens
        candidate_indices = set()
        for token in input_tokens[:3]:  # Use first 3 distinctive tokens
            if token in token_lookup:
                candidate_indices.update(token_lookup[token])

        if candidate_indices:
            # Score each candidate
            best_idx = None
            best_score = 0
            best_type = ''

            for idx in candidate_indices:
                candidate = compustat_df.iloc[idx]

                # Calculate token similarity
                token_sim = token_similarity(
                    input_tokens,
                    candidate['distinctive_tokens']
                )

                # Calculate string similarity
                string_sim = fuzz.ratio(norm_full, candidate['normalized']) / 100

                # Combined score (weighted average)
                combined_score = (token_sim * 0.6 + string_sim * 0.4) * 100

                # Suffix bonus/penalty
                if input_suffix and candidate['suffix']:
                    if input_suffix == candidate['suffix']:
                        combined_score = min(combined_score * 1.1, 100)
                    else:
                        combined_score *= 0.9

                if combined_score > best_score and combined_score >= threshold:
                    best_idx = idx
                    best_score = combined_score
                    best_type = 'token_match'

            if best_idx is not None:
                return {
                    'idx': best_idx,
                    'score': best_score,
                    'type': best_type
                }

    # Stage 3: Base name exact match
    if norm_base in base_lookup:
        candidates = base_lookup[norm_base]

        if len(candidates) == 1:
            return {
                'idx': candidates[0],
                'score': 85,
                'type': 'base_single'
            }
        else:
            # Match by suffix
            for idx in candidates:
                if compustat_df.iloc[idx]['suffix'] == input_suffix:
                    return {
                        'idx': idx,
                        'score': 90,
                        'type': 'base_suffix_match'
                    }

            # No suffix match, return first
            return {
                'idx': candidates[0],
                'score': 75,
                'type': 'base_no_suffix'
            }

    # Stage 4: Conservative fuzzy matching (only on base name, high threshold)
    if base_tokens and len(base_tokens) >= 2:  # Need at least 2 distinctive tokens
        base_names_list = list(base_lookup.keys())
        if base_names_list:
            result = process.extractOne(
                norm_base,
                base_names_list,
                scorer=fuzz.ratio,
                score_cutoff=85  # Higher threshold for fuzzy
            )

            if result:
                matched_base, score, _ = result
                candidates = base_lookup[matched_base]

                # Verify token overlap to avoid bad matches
                candidate_tokens = compustat_df.iloc[candidates[0]]['base_tokens']
                token_overlap = token_similarity(base_tokens, candidate_tokens)

                if token_overlap >= 0.5:  # Need significant token overlap
                    idx = candidates[0]

                    # Check suffix if multiple candidates
                    if len(candidates) > 1 and input_suffix:
                        for cand_idx in candidates:
                            if compustat_df.iloc[cand_idx]['suffix'] == input_suffix:
                                idx = cand_idx
                                break

                    return {
                        'idx': idx,
                        'score': score * token_overlap,  # Adjust score by token overlap
                        'type': 'fuzzy_conservative'
                    }

    return None

def process_batch_improved(args):
    """
    Process a batch of companies (for multiprocessing)
    """
    batch_companies, compustat_df, exact_lookup, token_lookup, base_lookup, threshold = args
    results = []

    for company_name in batch_companies:
        if pd.isna(company_name) or not company_name:
            results.append({
                'matched_company': '',
                'match_score': 0,
                'match_type': 'no_input',
                'gics_sector_name': '',
                'gics_indgrp_name': '',
                'gics_subindustry_code': '',
                'country_iso': ''
            })
            continue

        match_result = find_best_match_improved(
            company_name, compustat_df, exact_lookup,
            token_lookup, base_lookup, threshold
        )

        if match_result:
            matched_row = compustat_df.iloc[match_result['idx']]
            results.append({
                'matched_company': matched_row.get('company_name', ''),
                'match_score': round(match_result['score'], 1),
                'match_type': match_result['type'],
                'gics_sector_name': matched_row.get('gics_sector_name', ''),
                'gics_indgrp_name': matched_row.get('gics_indgrp_name', ''),
                'gics_subindustry_code': matched_row.get('gics_subindustry_code', ''),
                'country_iso': matched_row.get('country_iso', '')
            })
        else:
            results.append({
                'matched_company': '',
                'match_score': 0,
                'match_type': 'no_match',
                'gics_sector_name': '',
                'gics_indgrp_name': '',
                'gics_subindustry_code': '',
                'country_iso': ''
            })

    return results

def merge_datasets_fast(compustat_file, id_company_file, output_file='merged_output.csv',
                        threshold=75, use_multiprocessing=True, n_cores=None):
    """
    Fast version with multiprocessing and improved matching
    """
    print("="*70)
    print("High-Performance Smart Company Matcher")
    print("   - Avoids bad fuzzy matches")
    print("   - Uses token-based matching")
    print("   - Parallel processing for speed")
    print("="*70)

    # Load data
    print("\nLoading datasets...")
    compustat_df = pd.read_csv(compustat_file, low_memory=False)
    id_company_df = pd.read_csv(id_company_file)

    print(f"Compustat: {len(compustat_df):,} records")
    print(f"ID Company: {len(id_company_df):,} records")

    # Prepare data
    compustat_df, exact_lookup, token_lookup, base_lookup = prepare_compustat_data(compustat_df)

    # Setup for processing
    print(f"\nStarting matching process...")
    print(f"   Threshold: {threshold}%")
    print(f"   Multiprocessing: {use_multiprocessing}")

    company_names = id_company_df['companyName'].tolist()
    total = len(company_names)

    if use_multiprocessing and total > 1000:
        # Determine number of cores
        if n_cores is None:
            n_cores = min(mp.cpu_count() - 1, 8)

        print(f"   Using {n_cores} CPU cores")

        # Split into chunks for multiprocessing
        chunk_size = max(100, total // (n_cores * 10))
        chunks = [company_names[i:i+chunk_size]
                 for i in range(0, total, chunk_size)]

        # Prepare arguments for each chunk
        process_args = [(chunk, compustat_df, exact_lookup, token_lookup,
                         base_lookup, threshold) for chunk in chunks]

        # Process in parallel with progress bar
        all_results = []

        with mp.Pool(n_cores) as pool:
            with tqdm(total=total, desc="Processing") as pbar:
                for chunk_results in pool.imap(process_batch_improved, process_args):
                    all_results.extend(chunk_results)
                    pbar.update(len(chunk_results))
    else:
        # Single-threaded processing
        print("   Using single-threaded processing")
        all_results = process_batch_improved(
            (company_names, compustat_df, exact_lookup, token_lookup,
             base_lookup, threshold)
        )

    # Add results to dataframe
    for col in ['matched_company', 'match_score', 'match_type',
                'gics_sector_name', 'gics_indgrp_name', 'gics_subindustry_code', 'country_iso']:
        id_company_df[col] = [r[col] for r in all_results]

    # Calculate statistics
    stats = id_company_df['match_type'].value_counts().to_dict()
    matched_count = len(id_company_df[id_company_df['match_score'] > 0])

    print(f"\nMatching complete!")
    print(f"   Matched: {matched_count:,} ({matched_count/total*100:.1f}%)")
    print(f"   Unmatched: {total - matched_count:,}")

    # Show match type breakdown
    print(f"\nMatch Type Breakdown:")
    for match_type, count in stats.items():
        if match_type != 'no_input':
            pct = count / total * 100
            confidence = "HIGH" if match_type in ['exact', 'token_match'] else "MEDIUM"
            print(f"   {match_type:<20} {count:>7,} ({pct:>5.1f}%) [{confidence}]")

    # Save results
    print(f"\nSaving to '{output_file}'...")
    id_company_df.to_csv(output_file, index=False)
    print(f"Saved successfully")

    # Show examples
    print_sample_matches(id_company_df)

    return id_company_df

def print_sample_matches(df):
    """
    Print sample matches for verification
    """
    print("\nSample Matches:")

    # Check for potential bad matches
    print("\nChecking for suspicious matches...")

    # Find matches where company names are very different
    suspicious = []
    for idx, row in df[df['match_score'] > 0].iterrows():
        if row['match_score'] < 80:
            orig_tokens = get_distinctive_tokens(row['companyName'])
            match_tokens = get_distinctive_tokens(row['matched_company'])

            if orig_tokens and match_tokens:
                overlap = len(set(orig_tokens) & set(match_tokens))
                if overlap == 0:
                    suspicious.append(row)

    if suspicious:
        print(f"Found {len(suspicious)} potentially incorrect matches:")
        for row in suspicious[:5]:
            print(f"  '{row['companyName']}' -> '{row['matched_company']}' ({row['match_score']:.1f}%)")
    else:
        print("  No obviously incorrect matches found")

    # Show good matches
    print("\nHigh confidence matches:")
    high = df[df['match_type'] == 'exact']
    for _, row in high.head(3).iterrows():
        print(f"  '{row['companyName'][:40]}' -> '{row['matched_company'][:40]}' [{row['country_iso']}]")

    print("\nToken-based matches:")
    token = df[df['match_type'] == 'token_match']
    for _, row in token.head(3).iterrows():
        print(f"  '{row['companyName'][:40]}' -> '{row['matched_company'][:40]}' ({row['match_score']:.1f}%) [{row['country_iso']}]")

if __name__ == "__main__":
    # Configuration
    COMPUSTAT_FILE = "compustat_ref_All.csv"
    ID_COMPANY_FILE = "id_company_output.csv"
    OUTPUT_FILE = "merged_companies_prime_gics.csv"
    MATCH_THRESHOLD = 75
    USE_MULTIPROCESSING = True
    N_CORES = None  # None = auto-detect

    try:
        start_time = time.time()

        # Run merge
        result_df = merge_datasets_fast(
            COMPUSTAT_FILE,
            ID_COMPANY_FILE,
            OUTPUT_FILE,
            MATCH_THRESHOLD,
            USE_MULTIPROCESSING,
            N_CORES
        )

        elapsed = time.time() - start_time
        print(f"\nTotal time: {elapsed:.1f} seconds")

        # Quick validation
        print("\nQuick Validation:")
        matched = result_df[result_df['match_score'] > 0]
        print(f"  Average match score: {matched['match_score'].mean():.1f}%")
        print(f"  Matches above 90%: {(matched['match_score'] >= 90).sum():,}")
        print(f"  Matches below 80%: {(matched['match_score'] < 80).sum():,}")

        print("\nComplete! Check merged_companies.csv")

    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()
