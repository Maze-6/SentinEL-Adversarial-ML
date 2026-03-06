"""
adversarial_eval.py
Author: Mourya Reddy Udumula
Role: ML Architecture & Adversarial Research
Evaluates classifier accuracy on adversarially manipulated URL dataset.
Attack: Cyrillic homoglyph substitution (see adversarial/attack_generator.py)
Result: 75.64% accuracy under attack — 3.91 percentage point degradation
        (clean baseline: 79.55%, test set: 2,200 URLs).
Attack rate: 99.3% (2,185/2,200 URLs successfully perturbed).
Features sensitive to attack: Entropy (+Shannon entropy from Cyrillic symbol
        expansion) and Risky_TLD (homoglyphs bypass TLD string matching).
Keywords neutralised: 67.6% (150 of 222 keyword matches eliminated).
Key finding: model learns superficial string patterns (keyword substring
        matching, TLD string comparison) that are bypassable via Unicode
        substitution. Network features (DNS, WHOIS, SSL) are attack-resistant
        because domain infrastructure is unchanged by visual homoglyph swaps.
"""

import sys
import os
import re
import math
import json
import pickle
import hashlib
import pathlib
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from feature_extractor import FeatureExtractor
from adversarial.attack_generator import generate_adversarial_urls, attack_summary
import config

_PRIMARY_DATASET  = os.path.join(ROOT, 'data', 'phishing_dataset.csv')
_FALLBACK_DATASET = os.path.join(ROOT, 'data', 'dataset.csv')
DATASET_PATH  = _PRIMARY_DATASET if os.path.exists(_PRIMARY_DATASET) else _FALLBACK_DATASET
URL_LIST_PATH = os.path.join(ROOT, 'data', 'url_list.csv')
MODEL_PATH    = os.path.join(ROOT, 'data', 'sentinel_ultima.pkl')
CHECKSUM_PATH = os.path.join(ROOT, 'data', 'model_checksum.sha256')
RESULTS_PATH  = os.path.join(ROOT, 'experiments', 'results_adversarial.json')

FEATURE_NAMES = FeatureExtractor.get_feature_names()
LEXICAL_INDICES = list(range(8))


def verify_model_integrity(pkl_path: str) -> None:
    """
    Security: Verify model file integrity before deserialisation.
    Pickle files can execute arbitrary code on load — we verify the
    SHA-256 checksum against a known-good value to prevent tampering.

    SECURITY NOTE: This checksum guards against accidental file corruption only.
    It does NOT protect against a malicious actor who has write access to this
    repository — an attacker with write access could replace both sentinel_ultima.pkl
    and model_checksum.sha256 simultaneously, bypassing this check.
    For production deployment, the checksum would be stored out-of-band:
    e.g., cryptographically signed and hosted separately from the model artefact,
    or verified against a hardware-backed root of trust.
    This is a research implementation — the control is documented accordingly.
    """
    if not os.path.exists(CHECKSUM_PATH):
        raise FileNotFoundError(
            f"Checksum file not found: {CHECKSUM_PATH}\n"
            "Run experiments/baseline_eval.py first to generate the checksum."
        )
    expected = pathlib.Path(CHECKSUM_PATH).read_text().strip()
    computed = hashlib.sha256(pathlib.Path(pkl_path).read_bytes()).hexdigest()
    if computed != expected:
        raise RuntimeError(
            f"Model integrity check FAILED for {pkl_path}\n"
            f"  Expected : {expected}\n"
            f"  Computed : {computed}\n"
            "The model file may have been tampered with or corrupted."
        )
    print("Model integrity verified [OK]")


def extract_lexical_only(url: str) -> list:
    """
    Recompute the 8 lexical features from *url* without any network calls.

    Feature order (matches FEATURE_NAMES[0:8]):
        Length, Digits, Entropy, Risky_TLD, IP_Usage,
        Subdomains, Hyphens, Keywords
    """
    from urllib.parse import urlparse
    parsed = urlparse(url)
    domain = parsed.netloc.split(':')[0]

    is_risky_tld = 1 if any(domain.endswith(t) for t in config.RISKY_TLDS) else 0
    has_ip       = 1 if re.search(r'\d{1,3}\.\d{1,3}', url) else 0
    has_keyword  = 1 if any(k in url for k in config.RISK_KEYWORDS) else 0

    return [
        len(url),
        sum(c.isdigit() for c in url),
        FeatureExtractor.shannon_entropy(url),
        is_risky_tld,
        has_ip,
        url.count('.') - 1,
        url.count('-'),
        has_keyword,
    ]


def build_attacked_features(df: pd.DataFrame, mode: str = 'offline'):
    """
    Attack URL strings with homoglyphs, recompute lexical features, keep
    network/HTTP features from the pre-computed CSV columns.

    Strategy:
      - Columns 0-7  (lexical)  -> recomputed from attacked URL string
      - Columns 8-16 (network)  -> kept from original CSV (network state unchanged)

    df must have a 'url' column plus the 17 FEATURE_NAMES columns.
    """
    urls = df['url'].tolist()
    attacked_urls = generate_adversarial_urls(urls)

    # Feature columns only
    X_original = df[FEATURE_NAMES].copy()
    X_attacked = X_original.copy()

    for row_idx, atk_url in enumerate(attacked_urls):
        new_lex = extract_lexical_only(atk_url)
        for col_idx, val in zip(LEXICAL_INDICES, new_lex):
            X_attacked.iloc[row_idx, col_idx] = val

    if mode == 'enriched':
        enriched_names = FeatureExtractor.get_feature_names('enriched')[17:]
        enriched_rows = []
        for atk_url in attacked_urls:
            feats = FeatureExtractor.extract_features(atk_url, mode='enriched')
            enriched_rows.append(feats[17:])
        enriched_df = pd.DataFrame(enriched_rows, columns=enriched_names,
                                   index=X_attacked.index)
        return pd.concat([X_attacked, enriched_df], axis=1), attacked_urls

    return X_attacked, attacked_urls


def print_feature_diagnostic(df_test: pd.DataFrame,
                             X_clean: pd.DataFrame,
                             X_attacked: pd.DataFrame,
                             attacked_urls: list) -> None:
    """Print side-by-side feature comparison for the first 3 test URLs."""
    n_samples = min(3, len(df_test))
    sample_urls = df_test['url'].tolist()[:n_samples]
    sample_atk  = attacked_urls[:n_samples]

    print()
    print("  --- Attack Diagnostic (first 3 URLs) ---")
    print(f"  {'Feature':<15}  {'Original':>10}  {'Attacked':>10}  {'Delta':>10}")
    print("  " + "-" * 52)

    total_changes = 0
    for i in range(n_samples):
        orig_row = X_clean.iloc[i]
        atk_row  = X_attacked.iloc[i]
        changed_features = []
        for feat in FEATURE_NAMES:
            delta = atk_row[feat] - orig_row[feat]
            if abs(delta) > 1e-9:
                changed_features.append((feat, orig_row[feat], atk_row[feat], delta))
                total_changes += 1

        print(f"\n  URL {i+1}: {sample_urls[i][:55]}")
        print(f"  Attacked: {sample_atk[i][:55]}")
        if changed_features:
            for feat, orig_v, atk_v, delta in changed_features:
                print(f"    {feat:<15}  {orig_v:>10.4f}  {atk_v:>10.4f}  {delta:>+10.4f}")
        else:
            print("    (no feature changes for this URL)")

    print()
    sensitive = set()
    for feat in FEATURE_NAMES:
        if any(abs(X_attacked.iloc[i][feat] - X_clean.iloc[i][feat]) > 1e-9
               for i in range(n_samples)):
            sensitive.add(feat)

    print(f"  URLs attacked          : {len(attacked_urls)}/{len(df_test)}")
    print(f"  Features sensitive to attack (of {len(FEATURE_NAMES)}) : "
          f"{len(sensitive)} ({', '.join(sorted(sensitive)) or 'none'})")

    print()
    print("  Note: len(url) counts characters not bytes, so homoglyph substitution")
    print("  does NOT change the Length feature. Entropy IS sensitive — Cyrillic")
    print("  characters expand the symbol alphabet, increasing Shannon entropy.")

    if total_changes == 0:
        print()
        print("  WARNING: Homoglyph attack produced no feature-level changes.")
        print("  The 15.8 pp degradation reported in the research was measured on")
        print("  11,000 URLs where keyword features were sensitive to substitution.")


def main():
    parser = argparse.ArgumentParser(
        description='SentinEL adversarial evaluation (homoglyph attack).'
    )
    parser.add_argument(
        '--mode', choices=['offline', 'enriched'], default='offline',
        help='offline (default) or enriched feature mode.'
    )
    args = parser.parse_args()
    mode = args.mode

    print("=" * 60)
    print("  SentinEL -- Adversarial Evaluation (Homoglyph Attack)")
    print("=" * 60)
    print(f"  Feature mode : {mode}")

    # 1. Load feature dataset (phishing_dataset.csv — 17 features + label)
    df_feats = pd.read_csv(DATASET_PATH)
    if len(df_feats) < 50:
        df_feats = pd.concat([df_feats] * 5, ignore_index=True)

    # 2. Attach 'url' column from url_list.csv (same row order, built in same pass)
    if os.path.exists(URL_LIST_PATH):
        df_urls = pd.read_csv(URL_LIST_PATH)
        df_feats['url'] = df_urls['url'].values
    else:
        print("  WARNING: url_list.csv not found — using placeholder URLs.")
        print("  Degradation figure will not reflect real homoglyph substitution.")
        df_feats['url'] = 'http://placeholder.example.com'

    y = df_feats['label']
    X_clean = df_feats[FEATURE_NAMES].copy()

    # 3. Train/test split — identical seed to baseline_eval.py
    _, X_test_clean, _, y_test = train_test_split(
        X_clean, y, test_size=0.2, random_state=42, stratify=y
    )
    df_reset = df_feats.reset_index(drop=True)
    _, df_test = train_test_split(
        df_reset, test_size=0.2, random_state=42,
        stratify=df_reset['label']
    )
    print(f"Test set : {len(df_test)} rows")

    # 4. Load model with integrity check
    verify_model_integrity(MODEL_PATH)
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)['model']

    # 5. Baseline accuracy on clean test set
    clean_acc = accuracy_score(y_test, model.predict(X_test_clean))

    # 6. Build attacked feature matrix
    X_test_attacked, attacked_urls = build_attacked_features(
        df_test.reset_index(drop=True), mode=mode
    )
    attack_info = attack_summary(df_test['url'].tolist(), attacked_urls)

    # 7. Evaluate on attacked test set
    X_for_model = X_test_attacked.iloc[:, :17] if mode == 'enriched' else X_test_attacked
    attacked_acc = accuracy_score(y_test, model.predict(X_for_model))
    degradation  = clean_acc - attacked_acc

    # 8. Keyword change breakdown
    orig_kw   = int(X_test_clean.iloc[:, 7].sum())
    attack_kw = int(X_for_model.iloc[:, 7].sum())
    kw_lost   = orig_kw - attack_kw

    # 9. Print results
    print()
    print(f"  Clean accuracy      : {clean_acc*100:.2f}%")
    print(f"  Attacked accuracy   : {attacked_acc*100:.2f}%")
    print(f"  Degradation         : {degradation*100:.2f} pp")
    print()
    print(f"  URLs successfully attacked : {attack_info['attacked_count']}/{attack_info['total']}")
    print(f"  Keyword matches (original) : {orig_kw}")
    print(f"  Keyword matches (attacked) : {attack_kw}")
    print(f"  Keywords neutralised       : {kw_lost}")

    # 10. Feature-level diagnostic
    print_feature_diagnostic(
        df_test.reset_index(drop=True),
        X_test_clean.reset_index(drop=True),
        X_test_attacked.reset_index(drop=True),
        attacked_urls
    )

    # 11. Save JSON
    results = {
        'clean_accuracy':       round(clean_acc, 4),
        'attacked_accuracy':    round(attacked_acc, 4),
        'degradation_pp':       round(degradation * 100, 2),
        'n_test':               len(y_test),
        'attack_rate':          attack_info['attack_rate'],
        'keywords_neutralised': kw_lost,
    }
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {RESULTS_PATH}")

    return results


if __name__ == '__main__':
    import sys as _sys
    if _sys.stdout.encoding and _sys.stdout.encoding.lower() not in ('utf-8', 'utf-16'):
        _sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    main()