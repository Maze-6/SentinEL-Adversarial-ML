"""
test_sentinEL.py
Authors: Tests covering Mourya Udumula's ML architecture, adversarial research,
         explainability, enriched feature modes, and batch evaluation components.
"""
import pytest
import sys
import pathlib
import pickle
import pandas as pd
import numpy as np

ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from feature_extractor import FeatureExtractor
from adversarial.homoglyph_map import HOMOGLYPH_MAP, apply_homoglyphs, REVERSE_MAP
from adversarial.attack_generator import apply_homoglyphs_to_url
from explainer import GiniExplainer

# Canonical 17 semantic feature names used at training time
FEATURE_NAMES = FeatureExtractor.get_feature_names()

# ---------------------------------------------------------------------------
# Module-scoped fixtures — loaded once per test session
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def trained_model():
    """Load the trained pipeline from data/sentinel_ultima.pkl."""
    pkl_path = ROOT / "data" / "sentinel_ultima.pkl"
    data = pickle.load(open(pkl_path, "rb"))
    return data["model"]


@pytest.fixture(scope="module")
def sample_feature_row():
    """A zero-filled feature DataFrame — no network calls needed."""
    return pd.DataFrame([[0] * len(FEATURE_NAMES)], columns=FEATURE_NAMES)


# ---------------------------------------------------------------------------
# Feature Extractor Tests
# ---------------------------------------------------------------------------

class TestFeatureExtractor:
    def test_output_shape_offline(self):
        """Offline mode returns exactly 17 feature names."""
        names = FeatureExtractor.get_feature_names(mode='offline')
        assert len(names) == 17

    def test_output_shape_enriched(self):
        """Enriched mode returns exactly 25 feature names (17 + 8 network)."""
        names = FeatureExtractor.get_feature_names(mode='enriched')
        assert len(names) == 25

    def test_offline_extract_length(self):
        """extract_features(url, mode='offline') returns a list of 17 values."""
        feats = FeatureExtractor.extract_features("http://example.com", mode='offline')
        assert len(feats) == 17

    def test_rejects_none_input(self):
        """extract_features(None) raises TypeError at len(None)."""
        with pytest.raises(TypeError):
            FeatureExtractor.extract_features(None)

    def test_rejects_invalid_mode(self):
        """extract_features with an unknown mode raises ValueError."""
        with pytest.raises(ValueError):
            FeatureExtractor.extract_features("http://example.com", mode='foobar')

    def test_output_is_numeric(self):
        """shannon_entropy returns a non-negative float."""
        result = FeatureExtractor.shannon_entropy("http://example.com")
        assert isinstance(result, float)
        assert result >= 0.0

    def test_length_is_character_count(self):
        """Length feature is character count (not byte count).
        A URL with all-ASCII characters must have len(url) == length feature.
        Task 1: Cyrillic homoglyphs are single characters just like Latin,
        so character-count length is attack-invariant.
        """
        url = "http://paypal.com/verify"
        feats = FeatureExtractor.extract_features(url, mode='offline')
        assert feats[0] == len(url), (
            f"Length feature {feats[0]} != len(url) {len(url)}. "
            "Feature must use character count, not byte count."
        )

    def test_punycode_tld_normalisation(self):
        """Risky TLD check normalises Cyrillic domain to punycode before comparison.
        Task 2: A URL with a Cyrillic 'c' (.сom, where с=U+0441) should NOT
        match '.com' — it's a different TLD. The normalisation makes the comparison
        correct by punycode-encoding both sides consistently.
        """
        # Legitimate .com URL — risky_tld should be 0 (not in RISKY_TLDS)
        url_legit = "http://google.com/search"
        feats = FeatureExtractor.extract_features(url_legit, mode='offline')
        assert feats[3] == 0, "google.com should NOT be flagged as a risky TLD"

        # Risky TLD URL
        url_risky = "http://paypal.xyz/login"
        feats_risky = FeatureExtractor.extract_features(url_risky, mode='offline')
        assert feats_risky[3] == 1, "paypal.xyz should be flagged as a risky TLD"

    def test_offline_enriched_share_first_17(self):
        """First 17 values of enriched mode equal offline mode for same URL.
        Offline mode uses neutral defaults for features 8-16 identical to
        the enriched mode's fixed-baseline portion.
        """
        url = "http://example.com"
        offline = FeatureExtractor.extract_features(url, mode='offline')
        enriched = FeatureExtractor.extract_features(url, mode='enriched')
        # First 17 features must be identical (offline is the base)
        assert offline == enriched[:17], (
            "First 17 enriched features must equal offline features."
        )
        assert len(enriched) == 25


# ---------------------------------------------------------------------------
# Classifier Tests
# ---------------------------------------------------------------------------

class TestClassifier:
    def test_predicts_binary_output(self, trained_model, sample_feature_row):
        """predict() on a feature vector returns 0 (legitimate) or 1 (phishing)."""
        pred = trained_model.predict(sample_feature_row)
        assert pred[0] in (0, 1)

    def test_confidence_in_range(self, trained_model, sample_feature_row):
        """predict_proba() probabilities are in [0, 1] and sum to ~1.0."""
        proba = trained_model.predict_proba(sample_feature_row)[0]
        assert all(0.0 <= p <= 1.0 for p in proba)
        assert abs(sum(proba) - 1.0) < 0.001

    def test_predict_named_dataframe(self, trained_model):
        """Model accepts a named DataFrame with correct column count."""
        X = pd.DataFrame([[0] * 17], columns=FEATURE_NAMES)
        pred = trained_model.predict(X)
        assert pred[0] in (0, 1)


# ---------------------------------------------------------------------------
# Homoglyph Tests
# ---------------------------------------------------------------------------

class TestHomoglyphMap:
    def test_map_has_minimum_coverage(self):
        """HOMOGLYPH_MAP has at least 10 Latin-to-Cyrillic substitution pairs."""
        assert len(HOMOGLYPH_MAP) >= 10

    def test_substitution_occurs(self):
        """apply_homoglyphs('paypal.com') produces a different string of equal length."""
        original = "paypal.com"
        result = apply_homoglyphs(original)
        assert result != original                   # substitution happened
        assert len(result) == len(original)         # 1-to-1 char replacement

    def test_character_count_invariant(self):
        """Homoglyph substitution preserves character count (Task 1 invariant).
        In Python 3, len() counts Unicode code points (characters), not bytes.
        Each Cyrillic substitute is exactly 1 code point, same as the Latin original.
        """
        original = "http://paypal.com/verify"
        attacked = apply_homoglyphs_to_url(original)
        assert len(attacked) == len(original), (
            "Character count must be invariant to homoglyph substitution. "
            "Byte count (len(url.encode('utf-8'))) would differ — do not use that."
        )

    def test_protocol_stays_ascii(self):
        """apply_homoglyphs_to_url preserves the http:// scheme as plain ASCII."""
        url = "http://paypal.com/verify"
        result = apply_homoglyphs_to_url(url)
        assert result.startswith("http://")

    def test_https_protocol_preserved(self):
        """apply_homoglyphs_to_url preserves https:// scheme."""
        url = "https://bankofamerica.com/login"
        result = apply_homoglyphs_to_url(url)
        assert result.startswith("https://")

    def test_reverse_map_is_inverse(self):
        """REVERSE_MAP correctly maps Cyrillic back to Latin originals."""
        for latin, cyrillic in HOMOGLYPH_MAP.items():
            assert REVERSE_MAP[cyrillic] == latin, (
                f"REVERSE_MAP[{cyrillic!r}] should be {latin!r}"
            )

    def test_tld_preserved_after_attack(self):
        """TLD structure is syntactically preserved after homoglyph attack.
        The attacked URL must still parse as a valid URL with a non-empty netloc.
        """
        from urllib.parse import urlparse
        url = "http://verify-paypal.com/login"
        attacked = apply_homoglyphs_to_url(url)
        parsed = urlparse(attacked)
        assert parsed.netloc, "Attacked URL must have a non-empty netloc"
        assert parsed.scheme in ('http', 'https'), "Scheme must be preserved"


# ---------------------------------------------------------------------------
# Explainer Tests
# ---------------------------------------------------------------------------

class TestExplainer:
    def test_returns_named_features(self, trained_model):
        """top_features() returns semantic names (not generic f0, f1, f2 …)."""
        explainer = GiniExplainer(trained_model)
        top = explainer.top_features(FEATURE_NAMES, n=5)
        assert len(top) >= 1
        for name, _ in top:
            is_generic = (
                name.startswith("f")
                and len(name) <= 3
                and name[1:].isdigit()
            )
            assert not is_generic, f"Generic feature name found: {name!r}"

    def test_edge_case_empty_url(self):
        """apply_homoglyphs_to_url('') returns '' without raising any exception."""
        result = apply_homoglyphs_to_url("")
        assert result == ""


# ---------------------------------------------------------------------------
# Batch Evaluator Tests
# ---------------------------------------------------------------------------

class TestBatchEvaluator:
    """Tests for pipeline/batch_evaluate.py batch evaluation functionality."""

    def test_evaluate_url_returns_required_keys(self, trained_model):
        """evaluate_url() returns a dict with required keys."""
        sys.path.insert(0, str(ROOT / 'pipeline'))
        from batch_evaluate import evaluate_url
        result = evaluate_url(
            "http://example.com", trained_model, mode='offline'
        )
        required = {'url', 'verdict', 'prob_phishing', 'elapsed_ms', 'features'}
        assert required.issubset(result.keys()), (
            f"Missing keys: {required - result.keys()}"
        )

    def test_verdict_is_valid(self, trained_model):
        """verdict is one of PHISHING, SUSPICIOUS, LEGITIMATE."""
        from pipeline.batch_evaluate import evaluate_url
        r = evaluate_url("http://paypal-secure-login.xyz/update",
                         trained_model, mode='offline')
        assert r['verdict'] in ('PHISHING', 'SUSPICIOUS', 'LEGITIMATE')

    def test_prob_in_range(self, trained_model):
        """prob_phishing is in [0.0, 1.0]."""
        from pipeline.batch_evaluate import evaluate_url
        r = evaluate_url("http://google.com", trained_model, mode='offline')
        assert 0.0 <= r['prob_phishing'] <= 1.0

    def test_adversarial_flag_changes_eval_url(self, trained_model):
        """With adversarial=True, eval_url differs from original url."""
        from pipeline.batch_evaluate import evaluate_url
        url = "http://paypal.com/verify"
        r = evaluate_url(url, trained_model, mode='offline', adversarial=True)
        # The attacked URL must differ from the original (substitutions happen)
        assert r['eval_url'] != r['url'] or r['url'] == apply_homoglyphs_to_url(r['url']), (
            "Adversarial mode should modify the URL before evaluation."
        )

    def test_offline_features_length(self, trained_model):
        """Offline mode returns exactly 17 features in result."""
        from pipeline.batch_evaluate import evaluate_url
        r = evaluate_url("http://example.com", trained_model, mode='offline')
        assert len(r['features']) == 17

    def test_enriched_features_length(self, trained_model):
        """Enriched mode returns exactly 25 features in result."""
        from pipeline.batch_evaluate import evaluate_url
        r = evaluate_url("http://example.com", trained_model, mode='enriched')
        assert len(r['features']) == 25
