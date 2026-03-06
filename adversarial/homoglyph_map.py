"""
homoglyph_map.py
Author: Mourya Reddy Udumula
Role: ML Architecture & Adversarial Research
Cyrillic-to-Latin character substitution map used in homoglyph attack generation.
Maps visually identical Unicode characters (e.g., Cyrillic 'а' → Latin 'a') to
craft adversarial URLs that bypass string-matching phishing detectors.
Used by: adversarial/attack_generator.py
Research context: SentinEL adversarial robustness study
"""

# Latin → Cyrillic homoglyph substitution map.
# Keys are Latin characters; values are their visually identical Cyrillic counterparts.
# Sourced from Unicode confusables data — see SentinEL Technical Report Section 3.2.
HOMOGLYPH_MAP = {
    # Lowercase — highest-frequency substitutions targeting phishing keywords
    'a': '\u0430',  # Latin 'a' → Cyrillic 'а' (U+0430)
    'e': '\u0435',  # Latin 'e' → Cyrillic 'е' (U+0435)
    'o': '\u043e',  # Latin 'o' → Cyrillic 'о' (U+043E)
    'p': '\u0440',  # Latin 'p' → Cyrillic 'р' (U+0440)
    'c': '\u0441',  # Latin 'c' → Cyrillic 'с' (U+0441)
    'x': '\u0445',  # Latin 'x' → Cyrillic 'х' (U+0445)
    'y': '\u0443',  # Latin 'y' → Cyrillic 'у' (U+0443)
    'i': '\u0456',  # Latin 'i' → Cyrillic 'і' (U+0456, Ukrainian i)
    # Uppercase
    'B': '\u0412',  # Latin 'B' → Cyrillic 'В' (U+0412)
    'H': '\u041d',  # Latin 'H' → Cyrillic 'Н' (U+041D)
    'T': '\u0422',  # Latin 'T' → Cyrillic 'Т' (U+0422)
    'M': '\u041c',  # Latin 'M' → Cyrillic 'М' (U+041C)
    'A': '\u0410',  # Latin 'A' → Cyrillic 'А' (U+0410)
    'E': '\u0415',  # Latin 'E' → Cyrillic 'Е' (U+0415)
    'O': '\u041e',  # Latin 'O' → Cyrillic 'О' (U+041E)
    'P': '\u0420',  # Latin 'P' → Cyrillic 'Р' (U+0420)
    'C': '\u0421',  # Latin 'C' → Cyrillic 'С' (U+0421)
    'X': '\u0425',  # Latin 'X' → Cyrillic 'Х' (U+0425)
    'K': '\u041a',  # Latin 'K' → Cyrillic 'К' (U+041A)
}

# Reverse map: Cyrillic → Latin (for normalisation / detection)
REVERSE_MAP = {v: k for k, v in HOMOGLYPH_MAP.items()}


def apply_homoglyphs(text: str) -> str:
    """
    Replace Latin characters in *text* with their Cyrillic homoglyph equivalents.

    Digits, dots, slashes, colons, and hyphens are deliberately excluded from
    the map so URL structure remains syntactically valid.

    Args:
        text: Any string — typically a raw URL or URL component.

    Returns:
        A new string with applicable Latin characters replaced by Cyrillic ones.
        Returns '' if text is None or empty.

    Raises:
        TypeError: If text is not a str (and not None).
    """
    if text is None:
        return ""
    if not isinstance(text, str):
        raise TypeError(f"Expected str, got {type(text).__name__}")
    if text == "":
        return ""
    return ''.join(HOMOGLYPH_MAP.get(ch, ch) for ch in text)


def contains_homoglyphs(text: str) -> bool:
    """Return True if *text* contains any Cyrillic homoglyph character."""
    return any(ch in REVERSE_MAP for ch in text)


def normalise(text: str) -> str:
    """Replace Cyrillic homoglyphs in *text* back to their Latin equivalents."""
    return ''.join(REVERSE_MAP.get(ch, ch) for ch in text)


if __name__ == '__main__':
    samples = [
        'http://verify-identity-irs.gov.bad.com',
        'http://amazon-security-update.cn',
        'http://secure-verify-paypal.com.xyz',
        'http://192.168.1.1/login',
    ]
    print(f"{'Original':<45}  {'Attacked':<45}  Homoglyphs?")
    print('-' * 100)
    for url in samples:
        attacked = apply_homoglyphs(url)
        print(f"{url:<45}  {attacked:<45}  {contains_homoglyphs(attacked)}")
