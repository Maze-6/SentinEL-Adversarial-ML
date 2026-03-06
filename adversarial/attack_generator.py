"""
attack_generator.py
Author: Mourya Reddy Udumula
Role: ML Architecture & Adversarial Research
Applies Cyrillic homoglyph substitutions to URL datasets to generate
adversarial inputs. Used to evaluate robustness of phishing detection
classifiers under character-encoding attacks.
Results: 97.2% clean accuracy -> 81.4% under attack (15.8 pp degradation)
See: experiments/adversarial_eval.py
"""

import sys
import os
import pandas as pd
from urllib.parse import urlparse, urlunparse

# Allow imports from project root when run directly from adversarial/ subdir
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adversarial.homoglyph_map import apply_homoglyphs, contains_homoglyphs


def apply_homoglyphs_to_url(url: str) -> str:
    """
    Apply homoglyph substitution ONLY to the netloc and path of a URL.

    The scheme (http:// / https://) and query string are deliberately
    left as ASCII so the URL remains syntactically valid and the attack
    is confined to the domain and path components visible to the user.

    Args:
        url: A raw URL string.

    Returns:
        The URL with Cyrillic homoglyphs applied to netloc + path only.
        scheme and query remain untouched.

    Examples:
        >>> apply_homoglyphs_to_url('http://secure-paypal.com/verify')
        'http://sесurе-раурal.соm/vеrіfу'   # scheme stays ASCII
    """
    if not isinstance(url, str):
        raise TypeError(f"Expected str, got {type(url).__name__}")
    if not url:
        return url

    parsed = urlparse(url)
    attacked_netloc = apply_homoglyphs(parsed.netloc)
    attacked_path   = apply_homoglyphs(parsed.path)
    return urlunparse((
        parsed.scheme,    # untouched: http/https stays ASCII
        attacked_netloc,  # attacked: domain gets homoglyphs
        attacked_path,    # attacked: path gets homoglyphs
        parsed.params,    # untouched
        parsed.query,     # untouched
        parsed.fragment,  # untouched
    ))


def generate_adversarial_urls(url_list: list) -> list:
    """
    Apply Cyrillic homoglyph substitution to every URL in *url_list*.

    Only the netloc (domain) and path components are attacked.
    The URL scheme ('http://', 'https://') and query string are preserved
    as ASCII so URLs remain syntactically valid.

    Args:
        url_list: List of raw URL strings.

    Returns:
        New list of the same length with homoglyph-attacked URLs.

    Raises:
        TypeError: If url_list is None or not a list.
    """
    if url_list is None:
        raise TypeError("url_list cannot be None")
    if not isinstance(url_list, list):
        raise TypeError(
            f"url_list must be a list, got {type(url_list).__name__}"
        )
    if len(url_list) == 0:
        print("Warning: empty url_list provided")
        return []

    result = []
    for i, item in enumerate(url_list):
        if not isinstance(item, str):
            print(f"Warning: skipping non-string item at index {i}: "
                  f"{type(item).__name__}")
            continue
        result.append(apply_homoglyphs_to_url(item))
    return result


def generate_adversarial_dataset(df: pd.DataFrame, url_column: str) -> pd.DataFrame:
    """
    Return a copy of *df* with the URL column replaced by homoglyph-attacked URLs.

    All other columns (pre-computed features, labels) are preserved unchanged.
    Callers should recompute URL-derived lexical features after calling this function.

    Args:
        df:         DataFrame containing at least *url_column*.
        url_column: Name of the column holding raw URL strings.

    Returns:
        A new deep-copied DataFrame with *url_column* attacked.

    Raises:
        TypeError: If df is None.
        ValueError: If url_column is not a column in df.
    """
    if df is None:
        raise TypeError("DataFrame cannot be None")
    if url_column not in df.columns:
        raise ValueError(
            f"Column '{url_column}' not found. "
            f"Available: {list(df.columns)}"
        )
    if df.empty:
        print("Warning: empty DataFrame provided")
        return df.copy()

    adv_df = df.copy(deep=True)
    adv_df[url_column] = generate_adversarial_urls(adv_df[url_column].tolist())
    return adv_df


def attack_summary(original_urls: list, attacked_urls: list) -> dict:
    """
    Return a summary dict comparing original and attacked URL lists.

    Keys returned:
    - total:          total number of URLs
    - attacked_count: number of URLs containing at least one homoglyph
    - attack_rate:    fraction of URLs successfully attacked (0.0-1.0)
    - sample_pairs:   first 3 (original, attacked) pairs for visual inspection
    """
    attacked_count = sum(1 for u in attacked_urls if contains_homoglyphs(u))
    return {
        'total': len(original_urls),
        'attacked_count': attacked_count,
        'attack_rate': round(attacked_count / len(original_urls), 4) if original_urls else 0.0,
        'sample_pairs': list(zip(original_urls[:3], attacked_urls[:3])),
    }


def test_edge_cases() -> None:
    """Run edge-case validation for the attack functions."""
    print("\n=== Edge-Case Validation ===")

    # 1. Empty string URL
    result = apply_homoglyphs_to_url('')
    assert result == '', f"Empty string: expected '', got {result!r}"
    print("  [OK] apply_homoglyphs_to_url('') -> ''")

    # 2. None in list (should skip with warning)
    print("  [Testing None in list — expect warning below]")
    result_list = generate_adversarial_urls(['http://test.com', None, 'http://ok.com'])
    assert len(result_list) == 2, f"Expected 2 items (None skipped), got {len(result_list)}"
    print(f"  [OK] None in list skipped: {len(result_list)} URLs returned (None excluded)")

    # 3. Integer in list (should skip with warning)
    print("  [Testing integer in list — expect warning below]")
    result_list = generate_adversarial_urls([42, 'http://valid.com'])
    assert len(result_list) == 1, f"Expected 1 item (int skipped), got {len(result_list)}"
    print(f"  [OK] Integer in list skipped: {len(result_list)} URL returned")

    # 4. Empty list (should return [] with warning)
    print("  [Testing empty list — expect warning below]")
    result_list = generate_adversarial_urls([])
    assert result_list == [], f"Expected [], got {result_list}"
    print("  [OK] Empty list returned []")

    print("All edge cases passed.\n")


if __name__ == '__main__':
    # Ensure stdout can handle Cyrillic characters on Windows (CP1252 terminals)
    if sys.stdout.encoding and sys.stdout.encoding.lower() not in ('utf-8', 'utf-16'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')

    sample_urls = [
        'http://secure-verify-paypal.com.xyz',
        'http://apple-id-update.net',
        'http://192.168.1.1/login',
        'http://microsoft-team-invite.club',
        'http://verify-identity-irs.gov.bad.com',
        'http://netflix-account-hold.tk',
        'http://amazon-security-update.cn',
        'http://google.com',
        'http://facebook.com',
        'http://amazon.com',
    ]

    print("=== Homoglyph Attack Generator \u2014 Demo ===\n")
    attacked = generate_adversarial_urls(sample_urls)

    print(f"{'Original URL':<45}  {'Attacked URL':<50}  Changed?")
    print('-' * 105)
    for orig, atk in zip(sample_urls, attacked):
        changed = 'YES' if orig != atk else 'no'
        print(f"{orig:<45}  {atk:<50}  {changed}")

    summary = attack_summary(sample_urls, attacked)
    print(f"\nSummary: {summary['attacked_count']}/{summary['total']} URLs attacked "
          f"({summary['attack_rate'] * 100:.1f}%)")

    test_edge_cases()
