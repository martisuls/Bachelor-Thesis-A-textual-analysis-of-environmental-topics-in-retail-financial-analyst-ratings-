import argparse
import os
import time
import re
import pandas as pd
from html import unescape
from pathlib import Path

# --- Optional dependency flags (keep same behavior as original) ---
try:
    from bs4 import BeautifulSoup
    _HAS_BS4 = True
except ImportError:
    _HAS_BS4 = False

# --- Precompiled regex (used for fallback and whitespace collapse) ---
_TAG_BLOCK_RE = re.compile(
    r"(?is)<(script|style|noscript|iframe|video|audio|canvas)[^>]*>.*?</\1>"
)
_TAG_INLINE_RE = re.compile(
    r"(?is)<(img|picture|figure|form|button|input|object|embed)[^>]*>"
)
_ALL_TAGS_RE = re.compile(r"(?is)<[^>]+>")
_WS_RE = re.compile(r"\s+")

# --- HTML cleaner (same utility & parser semantics as your code) ---
def clean_html(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = unescape(text)

    if _HAS_BS4:
        # Keep the same BeautifulSoup + "lxml" parser choice
        # (If lxml isn't installed in this Python env, bs4 will raise;
        #  catch & fall back gracefully to regex so job still runs.)
        try:
            soup = BeautifulSoup(text, "lxml")
            for tag in soup([
                "script","style","noscript","iframe","video","audio","canvas",
                "picture","figure","img","form","button","input","object","embed"
            ]):
                tag.decompose()
            text = soup.get_text(separator=" ", strip=True)
        except Exception:
            # Fallback behavior (same as original branch without bs4)
            text = _TAG_BLOCK_RE.sub("", text)
            text = _TAG_INLINE_RE.sub("", text)
            text = _ALL_TAGS_RE.sub(" ", text)
    else:
        text = _TAG_BLOCK_RE.sub("", text)
        text = _TAG_INLINE_RE.sub("", text)
        text = _ALL_TAGS_RE.sub(" ", text)

    # Collapse whitespace to one line
    return _WS_RE.sub(" ", text).strip()


def process_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    if "content" not in chunk.columns:
        raise KeyError("CSV has no 'content' column")
    out = chunk.copy()
    # apply() keeps semantics clear and is fine chunked
    out["content"] = out["content"].apply(clean_html)
    return out


def main():
    ap = argparse.ArgumentParser(description="Clean 'content' column to single-line plain text (chunked, benchmarked).")
    ap.add_argument("input_csv", help="Path to input CSV")
    ap.add_argument("output_csv", help="Path to output CSV")
    ap.add_argument("--chunksize", type=int, default=50_000,
                    help="Rows per chunk (default: 50,000). Lower if RAM is tight.")
    ap.add_argument("--engine", default=None, choices=[None, "c", "python", "pyarrow"],
                    help="pandas.read_csv engine. Default: pandas decides. For speed, set to 'pyarrow' if installed.")
    ap.add_argument("--encoding", default=None, help="CSV encoding (optional)")
    ap.add_argument("--no-header-reset", action="store_true",
                    help="Do not delete existing output file (append mode). Default behavior removes existing file.")
    args = ap.parse_args()

    inp = Path(args.input_csv)
    outp = Path(args.output_csv)

    if not inp.exists():
        raise FileNotFoundError(f"Input CSV not found: {inp}")

    # Safe default: remove old output unless user opts out
    if outp.exists() and not args.no_header_reset:
        outp.unlink()

    total_rows = 0
    chunks = 0
    t0 = time.time()
    header_written = outp.exists()

    # Construct read_csv kwargs mirroring your original
    read_kwargs = dict(
        dtype=str,
        keep_default_na=False
    )
    if args.engine:
        read_kwargs["engine"] = args.engine
    if args.encoding:
        read_kwargs["encoding"] = args.encoding

    # Begin chunked read
    print(f"Reading: {inp}")
    print(f"Writing: {outp}")
    print(f"Chunksize: {args.chunksize:,} | Engine: {read_kwargs.get('engine', 'auto')}")
    if _HAS_BS4:
        print("Parser: BeautifulSoup('lxml') when available; regex fallback on parser errors")
    else:
        print("Parser: regex fallback only (bs4 not installed)")

    try:
        reader = pd.read_csv(inp, chunksize=args.chunksize, **read_kwargs)
    except TypeError:
        # Older pandas may not like some args—retry minimally
        reader = pd.read_csv(inp, chunksize=args.chunksize, dtype=str, keep_default_na=False)

    for chunk_idx, chunk in enumerate(reader, start=1):
        t_chunk_start = time.time()
        n_rows = len(chunk)
        cleaned = process_chunk(chunk)
        mode = "a" if header_written else "w"
        cleaned.to_csv(outp, index=False, mode=mode, header=not header_written)
        header_written = True

        dt = time.time() - t_chunk_start
        total_rows += n_rows
        chunks += 1
        rps = n_rows / dt if dt > 0 else float("inf")
        elapsed = time.time() - t0
        print(f"[Chunk {chunk_idx}] rows={n_rows:,} | {dt:.2f}s | {rps:,.1f} rows/s | elapsed={elapsed:.1f}s | total={total_rows:,}")

    total_time = time.time() - t0
    overall_rps = total_rows / total_time if total_time > 0 else float("inf")
    print("-" * 72)
    print(f"DONE  | chunks={chunks} | rows={total_rows:,} | time={total_time:.2f}s | avg throughput={overall_rps:,.1f} rows/s")
    print(f"Output written to: {outp.resolve()}")


if __name__ == "__main__":
    main()
