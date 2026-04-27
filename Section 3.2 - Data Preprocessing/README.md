# Section 3.2 — Data Preprocessing

This folder contains the preprocessing scripts that transform raw Seeking
Alpha exports into the cleaned files consumed by Section 3.3 (Company
Matching), Section 3.4 (Word2Vec Dictionary), and Section 3.5 (Tokenisation
and Dataset Assembly).

## Run order

1. **`cleandatacodespeed.py`** — strips HTML from the `content` column of
   the raw articles export, producing single-line plain text per report.
   Chunked and benchmarked; uses BeautifulSoup with an `lxml` parser when
   available and falls back to a regex tag-stripper otherwise.

   ```
   python cleandatacodespeed.py articles_raw.csv articles_clean.csv
   ```

2. **`format_article_dates.py`** — parses the `publishOn` field into ISO
   `YYYY-MM-DD` format and keeps only the `id` and `publishOn` columns.

   ```
   python format_article_dates.py
   ```

   Reads `articles_dates.csv` and writes `articles_datesformat_filtered.csv`.

3. **`reduce_companies_columns.py`** — reduces the raw `companies.csv` to
   the six identifier columns used downstream by the company-matching
   pipeline.

   ```
   python reduce_companies_columns.py
   ```

   Reads `companies.csv` and writes `companies_new.csv`.

4. **(Run after Section 3.3.)** **`extract_articles_id_content.py`** —
   extracts the `(id, content)` columns from the matched-and-filtered
   article file produced by `nomatch_removal.py` (Section 3.3). The output
   feeds Word2Vec preprocessing (Section 3.4) and environmental-term
   tokenisation (Section 3.5).

   ```
   python extract_articles_id_content.py
   ```

   Reads `articles_clean_filtered.csv` and writes `articles_id_content.csv`.

   This step enforces the methodology requirement that Word2Vec training
   and environmental-term tokenisation are performed only on reports
   linked to a Compustat-matched company (the 360'437-report analytical
   sample). Reports that failed company matching are removed earlier by
   `nomatch_removal.py` and therefore do not appear here.

## File-name handoffs

| Produced by                            | Output file                            | Consumed by                                                                  |
|----------------------------------------|----------------------------------------|------------------------------------------------------------------------------|
| `cleandatacodespeed.py`                | `articles_clean.csv`                   | Section 3.3 `nomatch_removal.py`                                             |
| `format_article_dates.py`              | `articles_datesformat_filtered.csv`    | Section 3.5 `final_data_all.py`                                              |
| `reduce_companies_columns.py`          | `companies_new.csv`                    | Section 3.3 (company-side matching input)                                    |
| Section 3.3 `nomatch_removal.py`       | `articles_clean_filtered.csv`          | This folder's `extract_articles_id_content.py`                               |
| `extract_articles_id_content.py`       | `articles_id_content.csv`              | Section 3.4 `step1-preprocessing.py`; Section 3.5 `tokenization_scriptGPT.py`|

## Inputs not produced in this folder

- `articles_raw.csv` — raw Seeking Alpha articles export (HTML content,
  publish dates, company associations).
- `articles_dates.csv` — Seeking Alpha articles dates export (the `id` and
  `publishOn` columns), used as input to `format_article_dates.py`.
- `companies.csv` — raw Seeking Alpha companies export.

These are obtained directly from the Seeking Alpha extraction described in
Section 3.1.1 of the thesis and are not redistributed here.

## Dependencies

- `pandas`
- `beautifulsoup4` and `lxml` (optional but recommended for
  `cleandatacodespeed.py`; falls back to regex-only HTML stripping if
  unavailable)
