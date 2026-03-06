"""
baseline_eval.py
Author: Mourya Reddy Udumula
Role: ML Architecture & Adversarial Research

Evaluates Random Forest classifier on the clean URL dataset.
Baseline result: 97.2% accuracy on held-out test set (80/20 split).
See classifier.py for model definition and training procedure.

Usage
-----
    # Evaluate pre-trained model:
    python experiments/baseline_eval.py

    # Retrain model on phishing_dataset.csv then evaluate:
    python experiments/baseline_eval.py --retrain
"""

import sys
import os
import json
import pickle
import hashlib
import pathlib
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_auc_score,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from feature_extractor import FeatureExtractor

# ── Paths ──────────────────────────────────────────────────────────────────────
DATASET_PATH   = os.path.join(ROOT, 'data', 'phishing_dataset.csv')
FALLBACK_PATH  = os.path.join(ROOT, 'data', 'dataset.csv')
MODEL_PATH     = os.path.join(ROOT, 'data', 'sentinel_ultima.pkl')
CHECKSUM_PATH  = os.path.join(ROOT, 'data', 'model_checksum.sha256')
RESULTS_PATH   = os.path.join(ROOT, 'experiments', 'results_baseline.json')

FEATURE_NAMES = FeatureExtractor.get_feature_names()   # 17 offline feature names


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
            "Run with --retrain to generate a new model and checksum."
        )
    expected = pathlib.Path(CHECKSUM_PATH).read_text().strip()
    computed  = hashlib.sha256(pathlib.Path(pkl_path).read_bytes()).hexdigest()
    if computed != expected:
        raise RuntimeError(
            f"Model integrity check FAILED for {pkl_path}\n"
            f"  Expected : {expected}\n"
            f"  Computed : {computed}\n"
            "The model file may have been tampered with or corrupted."
        )
    print("  Model integrity verified [OK]")


def load_data(dataset_path: str = None):
    """
    Load feature dataset. Prefers phishing_dataset.csv (~11k rows).
    Falls back to dataset.csv with 5x duplication if the full dataset
    is not yet built.

    Returns (X, y) DataFrames.
    """
    path = dataset_path or DATASET_PATH

    if not os.path.exists(path):
        print(f"  WARNING: {path} not found.")
        print(f"  Falling back to {FALLBACK_PATH} (demo dataset, ~70 rows).")
        print("  Run 'python data/build_dataset.py' to build the full 11k dataset.")
        path = FALLBACK_PATH

    df = pd.read_csv(path)

    if len(df) < 1_000:
        print(f"  WARNING: Dataset has only {len(df)} rows (expected >= 1,000).")
        print("  Results may not match the published research metrics.")
        if len(df) < 50:
            df = pd.concat([df] * 5, ignore_index=True)
            print(f"  Applied 5x duplication -> {len(df)} rows.")

    # Drop the label column to get features — CSV already has correct column names
    y = df['label']
    X = df.drop(columns=['label'])
    return X, y


def load_model():
    """Load the pre-trained Pipeline from data/sentinel_ultima.pkl."""
    verify_model_integrity(MODEL_PATH)
    with open(MODEL_PATH, 'rb') as f:
        payload = pickle.load(f)
    return payload['model']


def retrain_model(dataset_path: str = None):
    """
    Retrain the Random Forest Pipeline on phishing_dataset.csv and
    save to sentinel_ultima.pkl with a fresh SHA-256 checksum.

    Uses PhishModel.train_model() from classifier.py (single source of truth
    for training hyperparameters).
    """
    from classifier import PhishModel
    path = dataset_path or DATASET_PATH
    print(f"  Training on {path} ...")
    pm = PhishModel.__new__(PhishModel)
    pm.model_file = MODEL_PATH
    pm.feature_names = FEATURE_NAMES
    pm.f1 = pm.precision = pm.recall = 0.0
    pm.cm = None
    pm.roc_fpr = pm.roc_tpr = None
    pm.roc_auc = 0.0
    pm.loaded_data = None
    pm.model = None
    ok, msg = pm.train_model(path)
    if not ok:
        raise RuntimeError(f"Training failed: {msg}")
    print(f"  {msg}")
    # Write checksum
    digest = hashlib.sha256(pathlib.Path(MODEL_PATH).read_bytes()).hexdigest()
    pathlib.Path(CHECKSUM_PATH).write_text(digest)
    print(f"  Checksum written to {CHECKSUM_PATH}")
    return pm.model


def evaluate(model, X_test: pd.DataFrame, y_test: pd.Series,
             X_all: pd.DataFrame = None, y_all: pd.Series = None) -> dict:
    """
    Compute full metric suite:
      accuracy, precision, recall, F1 (macro + weighted),
      ROC-AUC, FNR, FPR, confusion matrix, 5-fold CV accuracy.
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, int(cm.sum()))

    acc       = accuracy_score(y_test, y_pred)
    prec      = precision_score(y_test, y_pred, zero_division=0)
    rec       = recall_score(y_test, y_pred, zero_division=0)
    f1        = f1_score(y_test, y_pred, zero_division=0)
    f1_macro  = f1_score(y_test, y_pred, average='macro',    zero_division=0)
    f1_weight = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    roc_auc   = roc_auc_score(y_test, y_prob) if len(set(y_test)) > 1 else float('nan')
    fnr       = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    fpr       = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    result = {
        'accuracy':        round(acc,       4),
        'precision':       round(prec,      4),
        'recall':          round(rec,       4),
        'f1':              round(f1,        4),
        'f1_macro':        round(f1_macro,  4),
        'f1_weighted':     round(f1_weight, 4),
        'roc_auc':         round(roc_auc,   4) if not np.isnan(roc_auc) else None,
        'fnr':             round(fnr,       4),
        'fpr':             round(fpr,       4),
        'confusion_matrix': cm.tolist(),
        'confusion_labels': ['TN', 'FP', 'FN', 'TP'],
        'n_test':          int(len(y_test)),
        'n_positives':     int(y_test.sum()),
        'n_negatives':     int((y_test == 0).sum()),
    }

    # 5-fold cross-validation on full dataset
    if X_all is not None and y_all is not None:
        cv_scores = cross_val_score(model, X_all, y_all, cv=5, scoring='accuracy')
        result['cv_5fold_mean'] = round(float(cv_scores.mean()), 4)
        result['cv_5fold_std']  = round(float(cv_scores.std()),  4)
        result['cv_5fold_all']  = [round(float(s), 4) for s in cv_scores]

    return result


def main():
    parser = argparse.ArgumentParser(
        description='SentinEL baseline evaluation with full metric suite.'
    )
    parser.add_argument(
        '--retrain', action='store_true',
        help='Retrain the model on phishing_dataset.csv before evaluating.'
    )
    parser.add_argument(
        '--dataset', type=str, default=None,
        help='Override dataset path (default: data/phishing_dataset.csv).'
    )
    args = parser.parse_args()

    print("=" * 65)
    print("  SentinEL -- Baseline Evaluation (Clean Dataset)")
    print("=" * 65)

    # 1. Load data
    X, y = load_data(args.dataset)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Dataset : {len(X):,} rows  ({X_train.shape[0]:,} train / {X_test.shape[0]:,} test)")

    # 2. Load or retrain model
    if args.retrain:
        print()
        print("  [--retrain] Training new model ...")
        model = retrain_model(args.dataset)
    else:
        print()
        model = load_model()
        print(f"  Model   : {type(model).__name__} loaded from {MODEL_PATH}")

    # 3. Evaluate
    print()
    print("  Computing metrics ...")
    metrics = evaluate(model, X_test, y_test, X_all=X, y_all=y)

    # 4. Print results
    cm = metrics['confusion_matrix']
    print()
    print(f"  {'Accuracy':<22} : {metrics['accuracy']*100:.2f}%")
    print(f"  {'Precision':<22} : {metrics['precision']*100:.2f}%")
    print(f"  {'Recall':<22} : {metrics['recall']*100:.2f}%")
    print(f"  {'F1 (binary)':<22} : {metrics['f1']*100:.2f}%")
    print(f"  {'F1 (macro)':<22} : {metrics['f1_macro']*100:.2f}%")
    print(f"  {'F1 (weighted)':<22} : {metrics['f1_weighted']*100:.2f}%")
    if metrics['roc_auc'] is not None:
        print(f"  {'ROC-AUC':<22} : {metrics['roc_auc']:.4f}")
    print(f"  {'FNR (miss rate)':<22} : {metrics['fnr']*100:.2f}%")
    print(f"  {'FPR (false alarm)':<22} : {metrics['fpr']*100:.2f}%")
    if 'cv_5fold_mean' in metrics:
        print(f"  {'5-fold CV (mean)':<22} : {metrics['cv_5fold_mean']*100:.2f}%"
              f"  (+/- {metrics['cv_5fold_std']*100:.2f}%)")
    print()
    print("  Confusion Matrix:")
    print(f"                Pred Legit  Pred Phish")
    if len(cm) == 2:
        print(f"  Actual Legit  {cm[0][0]:>10}  {cm[0][1]:>10}")
        print(f"  Actual Phish  {cm[1][0]:>10}  {cm[1][1]:>10}")

    # 5. Save JSON
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, 'w') as f:
        json.dump(metrics, f, indent=2)
    print()
    print(f"  Results saved to {RESULTS_PATH}")

    return metrics


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    main()