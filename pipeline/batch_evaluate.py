#!/usr/bin/env python3
"""
pipeline/batch_evaluate.py
Author: Mourya Reddy Udumula
Role: ML Architecture & Adversarial Research

Batch URL evaluation tool for the SentinEL phishing detection system.

Supports:
  - Single URL evaluation
  - CSV batch evaluation with live progress and ASCII bar charts
  - Offline (default) and enriched (--mode enriched) feature modes
  - Optional homoglyph adversarial attack simulation (--adversarial)
  - Configurable worker threads for concurrent enriched evaluation
  - URL selection: random sample, first-N slice, 1-based range, or hand-picked row numbers

Usage
-----
    # Single URL (offline mode):
    python pipeline/batch_evaluate.py --url "http://example.com"

    # Single URL (enriched, live network):
    python pipeline/batch_evaluate.py --url "http://example.com" --mode enriched

    # CSV batch evaluation (all URLs):
    python pipeline/batch_evaluate.py --csv data/url_list.csv

    # Random sample of 500 URLs (reproducible, same seed = same 500 every time):
    python pipeline/batch_evaluate.py --csv data/url_list.csv --sample 500

    # Random sample with a custom seed (different seed = different URLs):
    python pipeline/batch_evaluate.py --csv data/url_list.csv --sample 500 --seed 7

    # First 200 URLs (deterministic, no randomness):
    python pipeline/batch_evaluate.py --csv data/url_list.csv --first 200

    # Rows 13 to 60 inclusive (1-based, i.e. URL #13 through URL #60):
    python pipeline/batch_evaluate.py --csv data/url_list.csv --range 13 60

    # Specific non-sequential URLs by their 1-based row numbers:
    python pipeline/batch_evaluate.py --csv data/url_list.csv --pick 13 93 44 210 7

    # CSV with adversarial attack simulation:
    python pipeline/batch_evaluate.py --csv data/url_list.csv --adversarial

    # CSV with enriched mode and 8 workers:
    python pipeline/batch_evaluate.py --csv data/url_list.csv --mode enriched --workers 8

CSV format (--csv):
    The CSV must have a column named 'url'. An optional 'label' column (0=legit, 1=phish)
    enables accuracy computation. Example:
        url,label
        http://google.com,0
        http://paypal-secure-login.xyz/update,1

Selection flags (mutually exclusive — only one may be used at a time):
    --sample N      Randomly sample N URLs. Use --seed to control reproducibility.
    --first N       Take the first N rows from the CSV. Deterministic.
    --range S E     Take rows S through E inclusive. 1-based (row 1 = first URL in file).
    --pick N [N ...]  Evaluate specific URLs by their 1-based row numbers.
                    Numbers can be in any order and may be non-sequential.
                    Example: --pick 13 93 44 7 prints them in the order given.

Note on --sample reproducibility:
    Default seed is 42. Running --sample 500 without --seed always gives the
    same 500 URLs, which is important for consistent benchmarking. Use --seed
    with a different value when you deliberately want a different random subset.

Note on row numbering (--range and --pick):
    Row numbers are 1-based to match how you naturally count URLs in a spreadsheet.
    Row 1 = the first data row (header is not counted). Row 11000 = the last URL.
"""

import sys
import os
import json
import pickle
import hashlib
import pathlib
import argparse
import time

import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from feature_extractor import FeatureExtractor
from adversarial.attack_generator import apply_homoglyphs_to_url

# ── Paths ──────────────────────────────────────────────────────────────────────
MODEL_PATH    = os.path.join(ROOT, 'data', 'sentinel_ultima.pkl')
CHECKSUM_PATH = os.path.join(ROOT, 'data', 'model_checksum.sha256')

# ── ANSI colour codes (degrade gracefully if terminal does not support) ────────
_RESET  = '\033[0m'
_RED    = '\033[91m'
_GREEN  = '\033[92m'
_YELLOW = '\033[93m'
_CYAN   = '\033[96m'
_BOLD   = '\033[1m'

def _c(text: str, code: str) -> str:
    """Apply ANSI colour if stdout is a TTY."""
    if sys.stdout.isatty():
        return code + text + _RESET
    return text

def _red(t):    return _c(t, _RED)
def _green(t):  return _c(t, _GREEN)
def _yellow(t): return _c(t, _YELLOW)
def _cyan(t):   return _c(t, _CYAN)
def _bold(t):   return _c(t, _BOLD)


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def _verify_integrity() -> None:
    """Verify SHA-256 checksum before loading pickle."""
    if not os.path.exists(CHECKSUM_PATH):
        raise FileNotFoundError(
            f"Checksum not found: {CHECKSUM_PATH}\n"
            "Run: python experiments/baseline_eval.py --retrain"
        )
    expected = pathlib.Path(CHECKSUM_PATH).read_text().strip()
    computed  = hashlib.sha256(pathlib.Path(MODEL_PATH).read_bytes()).hexdigest()
    if computed != expected:
        raise RuntimeError(
            "Model integrity check FAILED. "
            "File may be corrupted or tampered."
        )


def load_model():
    """Load and return the trained sklearn Pipeline."""
    _verify_integrity()
    with open(MODEL_PATH, 'rb') as f:
        return pickle.load(f)['model']


# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction helpers
# ─────────────────────────────────────────────────────────────────────────────

def _extract(url: str, mode: str) -> list:
    """Extract features; returns empty list on any error."""
    try:
        return FeatureExtractor.extract_features(url, mode=mode)
    except Exception:
        n = 17 if mode == 'offline' else 25
        return [0] * n


# ─────────────────────────────────────────────────────────────────────────────
# Single-URL evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_url(url: str, model, mode: str = 'offline',
                 adversarial: bool = False, threshold: float = 0.5) -> dict:
    """
    Evaluate a single URL and return a result dict.

    Returns:
        {
            url, mode, adversarial,
            features: [float, ...],
            prob_phishing: float,
            verdict: 'PHISHING' | 'SUSPICIOUS' | 'LEGITIMATE',
            elapsed_ms: float,
        }
    """
    t0 = time.perf_counter()
    eval_url = apply_homoglyphs_to_url(url) if adversarial else url
    feats = _extract(eval_url, mode)
    names = FeatureExtractor.get_feature_names(mode)
    # The trained model always uses the 17 offline features.
    # In enriched mode, the extra 8 network features are stored for inspection
    # but the model receives only the first 17 (offline portion).
    model_names = FeatureExtractor.get_feature_names('offline')
    X = pd.DataFrame([feats[:17]], columns=model_names)
    prob = float(model.predict_proba(X)[0][1])
    verdict = (
        'PHISHING'   if prob >= threshold else
        'SUSPICIOUS' if prob >= threshold - 0.2 else
        'LEGITIMATE'
    )
    elapsed = (time.perf_counter() - t0) * 1000
    return {
        'url':           url,
        'eval_url':      eval_url,
        'mode':          mode,
        'adversarial':   adversarial,
        'features':      feats,
        'prob_phishing': round(prob, 4),
        'verdict':       verdict,
        'elapsed_ms':    round(elapsed, 2),
    }


def print_single_result(r: dict, names: list) -> None:
    """Pretty-print a single-URL evaluation result."""
    verdict_str = r['verdict']
    if verdict_str == 'PHISHING':
        verdict_coloured = _red(_bold(verdict_str))
    elif verdict_str == 'SUSPICIOUS':
        verdict_coloured = _yellow(_bold(verdict_str))
    else:
        verdict_coloured = _green(_bold(verdict_str))

    print()
    print(_bold("=" * 65))
    print(_bold("  SentinEL — URL Evaluation"))
    print(_bold("=" * 65))
    print(f"  URL      : {r['url']}")
    if r['adversarial']:
        print(f"  Attacked : {r['eval_url']}")
    print(f"  Mode     : {r['mode']}")
    print(f"  Verdict  : {verdict_coloured}")
    print(f"  P(phish) : {r['prob_phishing']:.4f}  ({r['prob_phishing']*100:.1f}%)")
    print(f"  Time     : {r['elapsed_ms']:.1f} ms")
    print()

    # ASCII bar chart for top features
    feats = r['features']
    # Show top 8 features by name
    top_n = min(8, len(names))
    max_val = max(abs(v) for v in feats[:top_n]) or 1.0
    print("  Feature snapshot (first 8 features):")
    bar_width = 30
    for i in range(top_n):
        v = feats[i]
        bar_len = int(abs(v) / max_val * bar_width)
        bar = '#' * bar_len
        print(f"    {names[i]:<18}  {v:>8.3f}  |{bar:<{bar_width}}|")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# CSV selection helper
# ─────────────────────────────────────────────────────────────────────────────

def _select_rows(df: pd.DataFrame, sample: int, first: int,
                 row_range: tuple, pick: list, seed: int) -> tuple[pd.DataFrame, str]:
    """
    Apply URL selection to a DataFrame.

    All row numbers exposed to the user (--range, --pick) are 1-based so they
    match natural spreadsheet counting. Internally they are converted to
    zero-based DataFrame indices before slicing.

    Exactly one of (sample, first, row_range, pick) should be set.
    If all are None/empty, the full DataFrame is returned unchanged.

    Args:
        df:        Full DataFrame loaded from CSV.
        sample:    Randomly sample this many rows. Uses `seed` for reproducibility.
        first:     Take the first N rows (head).
        row_range: Tuple (start, end) — 1-based inclusive range.
                   E.g. (13, 60) selects rows 13 through 60 (the 13th–60th URLs).
        pick:      List of 1-based row numbers in any order.
                   E.g. [13, 93, 44, 7] evaluates exactly those 4 URLs in that order.
        seed:      Random seed used when sample is set.

    Returns:
        (selected_df, selection_description_string)
    """
    total = len(df)

    # ── --sample ───────────────────────────────────────────────────────────────
    if sample is not None:
        if sample >= total:
            print(
                f"  [WARNING] --sample {sample:,} >= dataset size ({total:,}). "
                f"Using all {total:,} rows."
            )
            return df.reset_index(drop=True), f"all {total:,} URLs (--sample exceeded dataset size)"
        selected = df.sample(n=sample, random_state=seed).reset_index(drop=True)
        desc = f"{sample:,} / {total:,} URLs  (random sample, seed={seed})"
        return selected, desc

    # ── --first ────────────────────────────────────────────────────────────────
    if first is not None:
        n = min(first, total)
        if first > total:
            print(
                f"  [WARNING] --first {first:,} > dataset size ({total:,}). "
                f"Using all {total:,} rows."
            )
        selected = df.head(n).reset_index(drop=True)
        desc = f"{n:,} / {total:,} URLs  (first {n:,} rows)"
        return selected, desc

    # ── --range (1-based, inclusive) ───────────────────────────────────────────
    if row_range is not None:
        start_1, end_1 = row_range
        # Validate and clamp to 1-based bounds
        if start_1 < 1:
            print(f"  [WARNING] --range start {start_1} < 1; clamped to 1.")
            start_1 = 1
        if end_1 > total:
            print(f"  [WARNING] --range end {end_1} > dataset size ({total:,}); clamped to {total:,}.")
            end_1 = total
        if start_1 > end_1:
            raise ValueError(f"--range start ({start_1}) must be <= end ({end_1}).")
        # Convert to zero-based iloc slice
        selected = df.iloc[start_1 - 1 : end_1].reset_index(drop=True)
        n = len(selected)
        desc = f"{n:,} / {total:,} URLs  (rows {start_1}–{end_1} inclusive, 1-based)"
        return selected, desc

    # ── --pick (1-based, arbitrary order) ─────────────────────────────────────
    if pick:
        invalid = [r for r in pick if r < 1 or r > total]
        if invalid:
            raise ValueError(
                f"--pick row numbers out of range (valid: 1–{total:,}): {invalid}"
            )
        # Convert to zero-based indices, preserving the order the user specified
        zero_based = [r - 1 for r in pick]
        selected = df.iloc[zero_based].reset_index(drop=True)
        n = len(selected)
        pick_display = ", ".join(str(r) for r in pick[:10])
        if len(pick) > 10:
            pick_display += f", ... (+{len(pick)-10} more)"
        desc = f"{n:,} / {total:,} URLs  (hand-picked rows: {pick_display})"
        return selected, desc

    # ── No selection — use everything ──────────────────────────────────────────
    return df, f"all {total:,} URLs"


# ─────────────────────────────────────────────────────────────────────────────
# CSV batch evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_csv(csv_path: str, model, mode: str = 'offline',
                 adversarial: bool = False, workers: int = 4,
                 threshold: float = 0.5,
                 sample: int = None, first: int = None,
                 row_range: tuple = None, pick: list = None,
                 seed: int = 42) -> dict:
    """
    Evaluate URLs in a CSV file with optional subsetting.

    Args:
        csv_path:   Path to CSV with at least a 'url' column.
                    Optional 'label' column enables accuracy reporting.
        model:      Trained sklearn Pipeline.
        mode:       'offline' or 'enriched'.
        adversarial: Apply homoglyph attack before evaluation.
        workers:    Thread-pool size for concurrent feature extraction.
        threshold:  Phishing probability threshold.
        sample:     If set, randomly sample this many rows.
        first:      If set, use only the first N rows.
        row_range:  If set, tuple (start, end) — 1-based inclusive range.
        pick:       If set, list of 1-based row numbers in any order.
        seed:       Random seed for reproducibility when using --sample.

    Returns:
        Summary dict with counts, accuracy (if labels present), and per-row results.
        The summary also includes 'selection' describing what subset was used.
    """
    df_full = pd.read_csv(csv_path)
    if 'url' not in df_full.columns:
        raise ValueError(f"CSV must contain a 'url' column. Found: {list(df_full.columns)}")

    # ── Apply selection ────────────────────────────────────────────────────────
    df, selection_desc = _select_rows(df_full, sample, first, row_range, pick, seed)

    has_labels = 'label' in df.columns
    urls   = df['url'].astype(str).tolist()
    labels = df['label'].tolist() if has_labels else [None] * len(urls)
    n      = len(urls)
    names  = FeatureExtractor.get_feature_names(mode)

    # ── Print header ───────────────────────────────────────────────────────────
    print()
    print(_bold("=" * 65))
    print(_bold("  SentinEL -- Batch Evaluation"))
    print(_bold("=" * 65))
    print(f"  CSV       : {csv_path}")
    print(f"  Selection : {selection_desc}")
    print(f"  Mode      : {mode}")
    print(f"  Attack    : {adversarial}")
    print(f"  Workers   : {workers}")
    print()

    results = [None] * n
    t_start = time.perf_counter()

    # ── Concurrent feature extraction + prediction ─────────────────────────────
    def _eval_one(idx_url):
        idx, url, label = idx_url
        r = evaluate_url(url, model, mode=mode,
                         adversarial=adversarial, threshold=threshold)
        r['label'] = label
        return idx, r

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_eval_one, (i, u, lb)): i
            for i, (u, lb) in enumerate(zip(urls, labels))
        }
        done = 0
        for fut in as_completed(futures):
            idx, r = fut.result()
            results[idx] = r
            done += 1
            if done % max(1, n // 20) == 0 or done == n:
                pct = done / n * 100
                bar_len = int(pct / 100 * 40)
                bar = '#' * bar_len + '-' * (40 - bar_len)
                elapsed = time.perf_counter() - t_start
                print(f"\r  [{bar}] {done:>{len(str(n))}}/{n}  {pct:5.1f}%  {elapsed:.1f}s",
                      end='', flush=True)

    print()  # newline after progress bar

    elapsed = time.perf_counter() - t_start

    # ── Aggregate ──────────────────────────────────────────────────────────────
    verdicts = [r['verdict'] for r in results]
    probs    = [r['prob_phishing'] for r in results]
    n_phish  = verdicts.count('PHISHING')
    n_susp   = verdicts.count('SUSPICIOUS')
    n_legit  = verdicts.count('LEGITIMATE')

    summary = {
        'n_total':      n,
        'selection':    selection_desc,
        'n_phishing':   n_phish,
        'n_suspicious': n_susp,
        'n_legitimate': n_legit,
        'mean_prob':    round(float(np.mean(probs)), 4),
        'mode':         mode,
        'adversarial':  adversarial,
        'elapsed_s':    round(elapsed, 2),
        'urls_per_sec': round(n / elapsed, 1),
    }

    if has_labels:
        preds   = [1 if r['verdict'] == 'PHISHING' else 0 for r in results]
        correct = sum(p == int(l) for p, l in zip(preds, labels) if l is not None)
        summary['accuracy'] = round(correct / n, 4)

    # ── Print results ──────────────────────────────────────────────────────────
    print()
    print(_bold("  Results"))
    print("  " + "-" * 45)
    print(f"  {'PHISHING':<20} : {_red(str(n_phish)):>20}  ({n_phish/n*100:.1f}%)")
    print(f"  {'SUSPICIOUS':<20} : {_yellow(str(n_susp)):>20}  ({n_susp/n*100:.1f}%)")
    print(f"  {'LEGITIMATE':<20} : {_green(str(n_legit)):>20}  ({n_legit/n*100:.1f}%)")
    print()

    if 'accuracy' in summary:
        print(f"  Accuracy (labels) : {summary['accuracy']*100:.2f}%")

    print(f"  Mean P(phish)     : {summary['mean_prob']:.4f}")
    print(f"  Throughput        : {summary['urls_per_sec']:.0f} URLs/s  ({elapsed:.1f}s total)")

    # Distribution bar chart
    print()
    print("  Verdict distribution:")
    bar_max = 40
    for label, count in [('PHISHING', n_phish), ('SUSPICIOUS', n_susp), ('LEGITIMATE', n_legit)]:
        bar_len = int(count / n * bar_max) if n else 0
        bar = '#' * bar_len
        print(f"    {label:<12}  |{bar:<{bar_max}}|  {count:>5}")

    summary['results'] = results
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

    parser = argparse.ArgumentParser(
        description='SentinEL batch URL evaluator.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # All 11,000 URLs (original behavior — unchanged)
  python pipeline/batch_evaluate.py --csv data/url_list.csv

  # Random 500 URLs (reproducible — same command = same 500 every time)
  python pipeline/batch_evaluate.py --csv data/url_list.csv --sample 500

  # Random 500 URLs with a different seed (different 500 from above)
  python pipeline/batch_evaluate.py --csv data/url_list.csv --sample 500 --seed 7

  # First 200 URLs (deterministic, no randomness)
  python pipeline/batch_evaluate.py --csv data/url_list.csv --first 200

  # URLs 13 through 60 inclusive (1-based: row 13 = 13th URL in file)
  python pipeline/batch_evaluate.py --csv data/url_list.csv --range 13 60

  # Hand-pick specific URLs by their 1-based row numbers (any order)
  python pipeline/batch_evaluate.py --csv data/url_list.csv --pick 13 93 44 210 7

  # Single URL
  python pipeline/batch_evaluate.py --url "http://github.com"

  # Adversarial attack on a hand-picked set
  python pipeline/batch_evaluate.py --csv data/url_list.csv --pick 13 93 44 --adversarial

Note: --sample, --first, --range, and --pick are mutually exclusive.
      They only apply when using --csv, not --url.
        """
    )

    # ── Input source (mutually exclusive) ─────────────────────────────────────
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--url', type=str,
        help='Single URL to evaluate.'
    )
    input_group.add_argument(
        '--csv', type=str,
        help='CSV file with a "url" column.'
    )

    # ── URL selection flags (mutually exclusive, CSV-only) ─────────────────────
    select_group = parser.add_mutually_exclusive_group()
    select_group.add_argument(
        '--sample', type=int, metavar='N',
        help=(
            'Randomly sample N URLs from the CSV. '
            'Use --seed to control reproducibility (default seed: 42). '
            'Same seed + same N always produces the same subset.'
        )
    )
    select_group.add_argument(
        '--first', type=int, metavar='N',
        help='Use only the first N rows from the CSV. Deterministic.'
    )
    select_group.add_argument(
        '--range', type=int, nargs=2, metavar=('START', 'END'),
        dest='row_range',
        help=(
            '1-based inclusive row range. Row 1 = first URL in the file. '
            'Example: --range 13 60 evaluates URLs 13 through 60.'
        )
    )
    select_group.add_argument(
        '--pick', type=int, nargs='+', metavar='N',
        help=(
            '1-based row numbers of specific URLs to evaluate, in any order. '
            'Example: --pick 13 93 44 210 7  evaluates exactly those 5 URLs '
            'in the order given.'
        )
    )

    # ── Seed for reproducibility ───────────────────────────────────────────────
    parser.add_argument(
        '--seed', type=int, default=42, metavar='N',
        help='Random seed for --sample (default: 42).'
    )

    # ── Existing arguments (unchanged) ────────────────────────────────────────
    parser.add_argument(
        '--mode', choices=['offline', 'enriched'], default='offline',
        help='Feature mode: offline (17 features, fast) or enriched (25 features, network).'
    )
    parser.add_argument(
        '--adversarial', action='store_true',
        help='Apply Cyrillic homoglyph attack before evaluation.'
    )
    parser.add_argument(
        '--workers', type=int, default=4,
        help='Thread-pool size for concurrent enriched evaluation (default: 4).'
    )
    parser.add_argument(
        '--threshold', type=float, default=0.5,
        help='Phishing probability threshold (default: 0.5).'
    )
    parser.add_argument(
        '--out', type=str, default=None,
        help='Save JSON results to this path.'
    )

    args = parser.parse_args()

    # ── Guard: selection flags are CSV-only ────────────────────────────────────
    if args.url and (args.sample or args.first or args.row_range or args.pick):
        parser.error("--sample, --first, --range, and --pick can only be used with --csv, not --url.")

    # ── Load model ─────────────────────────────────────────────────────────────
    try:
        model = load_model()
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    names = FeatureExtractor.get_feature_names(args.mode)

    # ── Single URL path (unchanged) ────────────────────────────────────────────
    if args.url:
        r = evaluate_url(
            args.url, model,
            mode=args.mode,
            adversarial=args.adversarial,
            threshold=args.threshold,
        )
        print_single_result(r, names)
        if args.out:
            out = {k: v for k, v in r.items() if k != 'features'}
            pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            pathlib.Path(args.out).write_text(json.dumps(out, indent=2))
            print(f"  Saved to {args.out}")

    # ── CSV batch path ─────────────────────────────────────────────────────────
    else:
        summary = evaluate_csv(
            args.csv, model,
            mode=args.mode,
            adversarial=args.adversarial,
            workers=args.workers,
            threshold=args.threshold,
            sample=args.sample,
            first=args.first,
            row_range=tuple(args.row_range) if args.row_range else None,
            pick=args.pick if args.pick else None,
            seed=args.seed,
        )
        if args.out:
            out = {k: v for k, v in summary.items() if k != 'results'}
            pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            pathlib.Path(args.out).write_text(json.dumps(out, indent=2))
            print(f"  Saved to {args.out}")


if __name__ == '__main__':
    main()