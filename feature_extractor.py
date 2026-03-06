"""
feature_extractor.py
Author: Mourya Reddy Udumula
Role: ML Architecture & Adversarial Research

Two-mode URL feature extractor for the SentinEL phishing detection system.

Modes
-----
'offline'  (17 features) — Pure string/heuristic analysis. No network calls.
                           Safe for bulk dataset construction, adversarial
                           testing, and high-throughput scanning.
'enriched' (25 features) — 17 offline features + 8 concurrent live network
                           features (WHOIS, DNS, SSL, MX, ASN, registrar risk,
                           country risk). Use for single-URL production scanning.

Task 1 fix: Length feature uses len(url) — character count (Unicode code points)
            in Python 3 str, NOT byte count. Cyrillic homoglyphs are 1 char each
            (same as Latin originals), so character-count Length is invariant to
            homoglyph substitution. Byte-count (len(url.encode('utf-8'))) would
            be sensitive, which would be an encoding artefact, not a real signal.

Task 2 fix: Domain is IDNA/punycode-encoded before TLD and subdomain checks.
            This correctly normalises Cyrillic/Unicode netloc strings (e.g.
            'paypal.сom' with Cyrillic 'с' -> 'paypal.xn--om-7cd') so TLD
            membership tests work correctly on adversarially attacked URLs.

Task 5: Added mode parameter to extract_features() and get_feature_names().
        Enriched network features are computed concurrently via
        ThreadPoolExecutor and cached per-domain via functools.lru_cache.
"""

import re
import math
import ssl
import socket
import datetime
from urllib.parse import urlparse
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor

import config


class FeatureExtractor:
    """
    Extracts 17 (offline) or 25 (enriched) features from a URL.

    All public methods are static — no instance state is needed.
    """

    # ── Offline defaults for network features (indices 8-16) ──────────────────
    # Applied in 'offline' mode and as the fixed portion of 'enriched' features.
    # Mirror the failure-state values used by the original extractor so that
    # offline predictions are consistent with the model's training distribution.
    _NET_DEFAULTS = [
        1,    # DNS_Rec      — conservative: assume resolvable
        -1,   # Domain_Age   — -1 = unknown (same as WHOIS failure return)
        300,  # Expiry       — hardcoded constant (matches original extractor)
        0,    # Has_Form     — unknown
        0,    # Pass_Field   — unknown
        0,    # IFrame       — unknown
        0.1,  # Link_Ratio   — hardcoded constant (matches original extractor)
        0,    # HTTP_Code    — unknown
        0,    # SSL_Risk     — unknown; assume safe (conservative)
    ]

    # ── Enriched-mode helpers ──────────────────────────────────────────────────
    # High-risk country codes (ISO-3166-1 alpha-2, lowercase).
    _COUNTRY_RISK = {
        'cn': 0.85, 'ru': 0.80, 'ng': 0.75, 'ua': 0.65,
        'pk': 0.60, 'ro': 0.55, 'br': 0.50, 'tr': 0.50,
        'in': 0.45, 'id': 0.45, 'us': 0.15, 'gb': 0.10,
        'de': 0.10, 'fr': 0.12, 'jp': 0.10, 'au': 0.10,
        'ca': 0.12, 'nl': 0.12, 'se': 0.08, 'ch': 0.08,
    }
    # Registrar substrings associated with elevated abuse rates.
    _RISKY_REGISTRARS = {
        'namecheap', 'namesilo', 'pdr.ltd', 'onlinenic', 'eranet',
    }

    # ─────────────────────────────────────────────────────────────────────────
    # Core helpers
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def shannon_entropy(text: str) -> float:
        """Shannon entropy of *text* in bits-per-character."""
        if not text:
            return 0.0
        probs = [float(text.count(c)) / len(text) for c in dict.fromkeys(text)]
        return -sum(p * math.log(p) / math.log(2.0) for p in probs)

    @staticmethod
    def _normalise_domain(domain: str) -> str:
        """
        IDNA/punycode-encode *domain* for ASCII-safe TLD comparison.

        Task 2 fix: Without normalisation, Cyrillic domains such as
        'paypal.com' written with Cyrillic 'c' fail .com membership tests
        because the raw string does not end with the ASCII '.com' suffix.
        After IDNA encoding the domain converts to punycode, correctly
        avoiding false-negative risky-TLD hits on homoglyph URLs.
        """
        try:
            return domain.encode('idna').decode('ascii')
        except (UnicodeError, UnicodeDecodeError):
            return domain

    # ─────────────────────────────────────────────────────────────────────────
    # Lexical feature extraction (features 0-7, no network)
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _lexical(url: str) -> list:
        """
        Compute the 8 purely lexical/heuristic features.

        Task 1: Length is len(url) — character count (Unicode code points).
                Cyrillic homoglyph substitution replaces each Latin char with
                one Cyrillic char, so character-count Length is attack-invariant.
        Task 2: domain_ascii is used for all TLD comparisons.
        """
        parsed = urlparse(url)
        domain = parsed.netloc.split(':')[0]
        domain_ascii = FeatureExtractor._normalise_domain(domain)

        risky_tld = 1 if any(domain_ascii.endswith(t) for t in config.RISKY_TLDS) else 0
        ip_usage  = 1 if re.search(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', url) else 0
        keywords  = 1 if any(k in url for k in config.RISK_KEYWORDS) else 0

        return [
            len(url),                              # 0 Length (character count, not bytes)
            sum(c.isdigit() for c in url),         # 1 Digits
            FeatureExtractor.shannon_entropy(url), # 2 Entropy
            risky_tld,                             # 3 Risky_TLD
            ip_usage,                              # 4 IP_Usage
            url.count('.') - 1,                    # 5 Subdomains
            url.count('-'),                        # 6 Hyphens
            keywords,                              # 7 Keywords
        ]

    # ─────────────────────────────────────────────────────────────────────────
    # Legacy WHOIS helper (preserved for backward compat with classifier.py)
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def get_domain_age(domain: str) -> int:
        """Fetch domain age in days via WHOIS. Returns -1 on failure."""
        try:
            import whois
            w = whois.whois(domain)
            creation = w.creation_date
            if isinstance(creation, list):
                creation = creation[0]
            if isinstance(creation, (datetime.datetime, datetime.date)):
                if getattr(creation, 'tzinfo', None) is not None:
                    creation = creation.replace(tzinfo=None)
                return (datetime.datetime.now() - creation).days
        except Exception:
            return -1
        return -1

    # ─────────────────────────────────────────────────────────────────────────
    # Enriched network features (features 17-24)
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    @lru_cache(maxsize=256)
    def _cached_whois(domain: str):
        """Cached WHOIS lookup — result shared across enriched feature sub-tasks."""
        try:
            import whois
            return whois.whois(domain)
        except Exception:
            return None

    @staticmethod
    def _enriched_network(domain: str) -> list:
        """
        Compute 8 live network-based enriched features concurrently.

        Uses ThreadPoolExecutor for parallel I/O; domain-level lru_cache
        prevents redundant WHOIS calls within a single enriched extraction.

        Enriched feature indices (17-24):
            17 domain_age_days      WHOIS creation date age in days; -1=unknown
            18 dns_resolves         1 if DNS A record resolves, 0 otherwise
            19 ssl_valid            1 if SSL cert valid (>30 days remaining)
            20 ssl_cert_age_days    Age of SSL cert in days; -1=unknown
            21 mx_record_exists     1 if domain has MX records
            22 ip_in_high_risk_asn  0.0-1.0 heuristic ASN risk score
            23 registrar_risk_score 1.0 if registrar in high-abuse list, else 0.0
            24 whois_country_risk   0.0-1.0 registrant country risk score
        """
        def _dns_resolves():
            try:
                import dns.resolver
                dns.resolver.resolve(domain, 'A')
                return 1
            except Exception:
                return 0

        def _ssl_info():
            try:
                ctx = ssl.create_default_context()
                ctx.check_hostname = False
                with ctx.wrap_socket(socket.socket(), server_hostname=domain) as s:
                    s.settimeout(3.0)
                    s.connect((domain, 443))
                    cert = s.getpeercert()
                    not_after_str  = cert.get('notAfter', '')
                    not_before_str = cert.get('notBefore', '')
                    if not_after_str:
                        not_after = datetime.datetime.strptime(
                            not_after_str, '%b %d %H:%M:%S %Y %Z'
                        )
                        valid = 1 if (not_after - datetime.datetime.now()).days > 30 else 0
                        age_days = -1
                        if not_before_str:
                            not_before = datetime.datetime.strptime(
                                not_before_str, '%b %d %H:%M:%S %Y %Z'
                            )
                            age_days = (datetime.datetime.now() - not_before).days
                        return valid, age_days
                    return 0, -1
            except Exception:
                return 0, -1

        def _mx_exists():
            try:
                import dns.resolver
                dns.resolver.resolve(domain, 'MX')
                return 1
            except Exception:
                return 0

        def _asn_risk():
            """Simplified heuristic: flag IP ranges associated with high-abuse ASNs."""
            try:
                ip = socket.gethostbyname(domain)
                first_octet = int(ip.split('.')[0])
                high_risk_first_octets = {
                    1, 14, 27, 36, 42, 43, 49, 58, 59, 60, 61,
                    101, 106, 110, 111, 112, 113, 114, 115, 116,
                    117, 118, 119, 120, 121, 122, 123, 124, 125,
                    182, 183, 218, 219, 220, 221, 222, 223,
                }
                return 0.6 if first_octet in high_risk_first_octets else 0.0
            except Exception:
                return 0.0

        # Run I/O-bound lookups concurrently
        with ThreadPoolExecutor(max_workers=4) as pool:
            f_dns   = pool.submit(_dns_resolves)
            f_ssl   = pool.submit(_ssl_info)
            f_mx    = pool.submit(_mx_exists)
            f_asn   = pool.submit(_asn_risk)
            f_whois = pool.submit(FeatureExtractor._cached_whois, domain)

        dns_ok           = f_dns.result()
        ssl_valid, ssl_age = f_ssl.result()
        mx_ok            = f_mx.result()
        asn_risk         = f_asn.result()
        w                = f_whois.result()

        # WHOIS-derived features
        domain_age     = -1
        registrar_risk = 0.0
        country_risk   = 0.3   # default: slight uncertainty

        if w is not None:
            try:
                creation = w.creation_date
                if isinstance(creation, list):
                    creation = creation[0]
                if isinstance(creation, (datetime.datetime, datetime.date)):
                    if getattr(creation, 'tzinfo', None) is not None:
                        creation = creation.replace(tzinfo=None)
                    domain_age = (datetime.datetime.now() - creation).days
            except Exception:
                pass

            try:
                registrar = (getattr(w, 'registrar', '') or '').lower()
                registrar_risk = (
                    1.0 if any(r in registrar
                               for r in FeatureExtractor._RISKY_REGISTRARS)
                    else 0.0
                )
            except Exception:
                pass

            try:
                country = (getattr(w, 'country', '') or '').lower()[:2]
                country_risk = FeatureExtractor._COUNTRY_RISK.get(country, 0.3)
            except Exception:
                pass

        return [
            domain_age,     # 17 domain_age_days
            dns_ok,         # 18 dns_resolves
            ssl_valid,      # 19 ssl_valid
            ssl_age,        # 20 ssl_cert_age_days
            mx_ok,          # 21 mx_record_exists
            asn_risk,       # 22 ip_in_high_risk_asn
            registrar_risk, # 23 registrar_risk_score
            country_risk,   # 24 whois_country_risk
        ]

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def extract_features(url: str, mode: str = 'offline') -> list:
        """
        Extract URL features in the specified mode.

        Args:
            url:  URL string to analyse.
            mode: 'offline'  — 17 features, pure string analysis, no network.
                  'enriched' — 25 features: 17 offline + 8 live network.

        Returns:
            List of floats/ints of length 17 (offline) or 25 (enriched).

        Raises:
            ValueError: If mode is not 'offline' or 'enriched'.
            TypeError:  If url is not a str (len(None) raises TypeError).
        """
        lex = FeatureExtractor._lexical(url)            # 8 string features
        net = list(FeatureExtractor._NET_DEFAULTS)      # 9 network-defaults

        if mode == 'offline':
            return lex + net   # 17 total, no network calls

        if mode == 'enriched':
            parsed = urlparse(url)
            domain = parsed.netloc.split(':')[0]
            domain = FeatureExtractor._normalise_domain(domain)
            enriched = FeatureExtractor._enriched_network(domain)  # 8 live
            return lex + net + enriched   # 25 total

        raise ValueError(f"Unknown mode {mode!r}. Use 'offline' or 'enriched'.")

    @staticmethod
    def get_feature_names(mode: str = 'offline') -> list:
        """
        Return semantic feature name list for the given mode.

        Args:
            mode: 'offline' -> 17 names, 'enriched' -> 25 names.

        Returns:
            List of str feature names in extract_features() output order.
        """
        offline = [
            "Length", "Digits", "Entropy", "Risky_TLD", "IP_Usage",
            "Subdomains", "Hyphens", "Keywords",
            "DNS_Rec", "Domain_Age", "Expiry",
            "Has_Form", "Pass_Field", "IFrame", "Link_Ratio",
            "HTTP_Code", "SSL_Risk",
        ]
        enriched_extra = [
            "domain_age_days", "dns_resolves", "ssl_valid", "ssl_cert_age_days",
            "mx_record_exists", "ip_in_high_risk_asn",
            "registrar_risk_score", "whois_country_risk",
        ]
        return offline + enriched_extra if mode == 'enriched' else offline


if __name__ == '__main__':
    import sys
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

    test_urls = [
        "http://google.com",
        "http://paypal-secure-login.xyz/update",
        "http://192.168.1.1/admin",
        "http://verify-identity-irs.gov.bad.com",
    ]
    names = FeatureExtractor.get_feature_names()
    print(f"Offline features ({len(names)}):")
    print(f"  {'URL':<45}  {'Len':>4}  {'Entropy':>7}  {'RiskyTLD':>8}  {'KW':>3}")
    print("  " + "-" * 72)
    for url in test_urls:
        feats = FeatureExtractor.extract_features(url, mode='offline')
        print(f"  {url:<45}  {feats[0]:>4}  {feats[2]:>7.3f}  {feats[3]:>8}  {feats[7]:>3}")
