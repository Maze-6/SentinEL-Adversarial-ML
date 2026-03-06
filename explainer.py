"""
explainer.py
Author: Mourya Reddy Udumula
Role: ML Architecture & Adversarial Research
Native Gini-based feature attribution for the Random Forest phishing classifier.
Provides < 2ms explainability suitable for real-time SOC alert enrichment.
Compared against SHAP and LIME in experiments/explainability_bench.py.
"""

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier


class GiniExplainer:
    """
    Wraps a trained RandomForestClassifier (or a Pipeline whose 'clf' step is one)
    and exposes native Gini impurity-based feature importance.

    Gini importance is computed during training and accessed in O(1) — typically
    sub-millisecond — making it viable for real-time SOC alert enrichment where
    SHAP (50-80 ms) and LIME (50-80 ms) introduce unacceptable latency under EU
    AI Act transparency requirements.
    """

    def __init__(self, model):
        """
        Args:
            model: A fitted RandomForestClassifier *or* a fitted
                   sklearn.pipeline.Pipeline whose 'clf' step is a
                   RandomForestClassifier.

        Raises:
            TypeError: If no RandomForestClassifier can be extracted.
        """
        if isinstance(model, Pipeline):
            if 'clf' in model.named_steps:
                self._rf = model.named_steps['clf']
            else:
                self._rf = list(model.named_steps.values())[-1]
        elif isinstance(model, RandomForestClassifier):
            self._rf = model
        else:
            raise TypeError(
                f"Expected RandomForestClassifier or Pipeline, got {type(model).__name__}"
            )

        if not isinstance(self._rf, RandomForestClassifier):
            raise TypeError(
                f"Resolved model step is {type(self._rf).__name__}, "
                "expected RandomForestClassifier"
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def explain(self, feature_names: list) -> dict:
        """
        Return global Gini feature importances sorted in descending order.

        Args:
            feature_names: List of feature name strings matching the number of
                           features the model was trained on.

        Returns:
            {feature_name: importance_score} ordered most → least important.
            Scores sum to approximately 1.0.
        """
        importances = self._rf.feature_importances_
        paired = dict(zip(feature_names, importances.tolist()))
        return dict(sorted(paired.items(), key=lambda kv: kv[1], reverse=True))

    def explain_prediction(self, X_row, feature_names: list) -> dict:
        """
        Return per-tree mean feature importances weighted by prediction confidence.

        Because native Gini importances are global (model-level), this method
        returns the global importances weighted by each tree's prediction
        confidence for the given row — a lightweight proxy for instance-level
        explanation without SHAP overhead.

        Args:
            X_row:         Array-like of shape (1, n_features) or (n_features,).
            feature_names: Feature name strings.

        Returns:
            {feature_name: weighted_importance} sorted descending.
        """
        X_arr = np.array(X_row).reshape(1, -1)
        # Gather per-tree confidences for the positive (phishing) class
        tree_probs = np.array([
            tree.predict_proba(X_arr)[0][1]
            for tree in self._rf.estimators_
        ])
        weights = tree_probs / (tree_probs.sum() + 1e-12)

        # Weight each tree's feature importances by that tree's confidence
        weighted_importances = np.array([
            tree.feature_importances_ * w
            for tree, w in zip(self._rf.estimators_, weights)
        ]).sum(axis=0)

        paired = dict(zip(feature_names, weighted_importances.tolist()))
        return dict(sorted(paired.items(), key=lambda kv: kv[1], reverse=True))

    def top_features(self, feature_names: list, n: int = 5) -> list:
        """
        Return the top-n most important features as (name, importance) tuples.

        Note: feature_names should be the actual column names from dataset.csv,
        not generic placeholders. Use get_feature_names_from_dataset() to
        retrieve them, or FeatureExtractor.get_feature_names() for the
        canonical names.

        Args:
            feature_names: Feature name strings (e.g. from dataset.csv columns).
            n:             Number of top features to return (default 5).

        Returns:
            List of up to n (feature_name, importance_score) tuples,
            sorted most → least important.
        """
        ranked = self.explain(feature_names)
        return list(ranked.items())[:n]

    @staticmethod
    def get_feature_names_from_dataset(csv_path: str) -> list:
        """
        Load actual feature column names from dataset.csv.

        Returns all column names except 'url' and 'label', giving the
        real feature names to pass into explain(), top_features(), etc.
        Use this instead of passing generic f0/f1/f2 placeholders.

        Note: feature_names should be the actual column names from dataset.csv,
        not generic placeholders. Use this method to retrieve them.

        Args:
            csv_path: Path to the dataset CSV file.

        Returns:
            List of feature column name strings.
        """
        import pandas as pd
        df = pd.read_csv(csv_path, nrows=0)  # headers only, no data
        return [c for c in df.columns if c not in ('url', 'label')]

    def importance_table(self, feature_names: list) -> str:
        """Return a formatted ASCII table of all features and their importances."""
        ranked = self.explain(feature_names)
        lines = [f"{'Feature':<20}  {'Importance':>10}  Bar"]
        lines.append('-' * 55)
        for name, score in ranked.items():
            bar = '\u2588' * int(score * 40)
            lines.append(f"{name:<20}  {score:>10.4f}  {bar}")
        return '\n'.join(lines)


if __name__ == '__main__':
    import sys as _sys
    import pickle
    import os
    # Ensure stdout can handle Unicode bar/dash characters on Windows CP1252
    if _sys.stdout.encoding and _sys.stdout.encoding.lower() not in ('utf-8', 'utf-16'):
        _sys.stdout.reconfigure(encoding='utf-8', errors='replace')

    # Resolve paths relative to project root regardless of cwd
    _HERE = os.path.dirname(os.path.abspath(__file__))
    model_path   = os.path.join(_HERE, 'data', 'sentinel_ultima.pkl')
    dataset_path = os.path.join(_HERE, 'data', 'dataset.csv')

    try:
        data = pickle.load(open(model_path, 'rb'))
        pipeline = data['model']
    except FileNotFoundError:
        print(f"Model not found at {model_path}. Run experiments/baseline_eval.py first.")
        raise SystemExit(1)

    # Use descriptive feature names from FeatureExtractor (Length, Entropy, etc.)
    # The CSV stores generic f1/f2/... column headers, but FeatureExtractor.get_feature_names()
    # provides the semantic mapping used when the model was trained.
    from feature_extractor import FeatureExtractor
    feature_names = FeatureExtractor.get_feature_names()
    explainer = GiniExplainer(pipeline)

    print("=== GiniExplainer \u2014 Global Feature Importance ===\n")
    print(explainer.importance_table(feature_names))
    print(f"\nTop 5 features: {[(n, round(s, 4)) for n, s in explainer.top_features(feature_names)]}")
