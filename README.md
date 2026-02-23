# SentinEL

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Random%20Forest-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![Domain](https://img.shields.io/badge/Domain-Adversarial%20ML%20%7C%20Cybersecurity-DC143C?style=flat-square)
![Research](https://img.shields.io/badge/Type-Undergraduate%20Research-4B0082?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-22C55E?style=flat-square)

**Quantifying how character-encoding attacks break ML-based phishing detectors — and building a 25–40× faster explainability layer for real-world SOC deployment.**

</div>

---

## The Core Finding

> Pattern-matching ML phishing detectors that achieve **97.2% accuracy** on standard benchmarks degrade to **81.4% accuracy** under Cyrillic homoglyph substitution — a zero-cost attack requiring no ML knowledge that is already documented in phishing-as-a-service toolkits.

This is not a theoretical concern. It is a **15.8 percentage point gap** between how these systems are evaluated and how they perform against adversaries who know what to test for.

---

## Table of Contents

- [Results at a Glance](#results-at-a-glance)
- [Why This Matters](#why-this-matters)
- [Research Question](#research-question)
- [Methodology](#methodology)
- [Architecture](#architecture)
- [Results Detail](#results-detail)
- [Practical Mitigations](#practical-mitigations)
- [Repository Structure](#repository-structure)
- [Reproducing the Experiments](#reproducing-the-experiments)
- [Limitations](#limitations)
- [Citation](#citation)

---

## Results at a Glance

### Robustness

| Condition | Accuracy | Delta |
|-----------|----------|-------|
| Clean test data | **97.2%** | — |
| Cyrillic homoglyph attack | **81.4%** | **−15.8 pp** |

### Explainability Latency

| Method | Latency | SOC-Deployable? |
|--------|---------|----------------|
| SHAP | 50–80 ms | ❌ Too slow for real-time alerting |
| LIME | 50–80 ms | ❌ Too slow for real-time alerting |
| **Native Gini attribution (this work)** | **< 2 ms** | **✅ Yes** |

The explainability improvement is **25–40× faster** than SHAP/LIME, achieved by reusing already-computed tree metadata rather than running a perturbation loop at inference time.

---

## Why This Matters

Standard evaluation of ML phishing detectors uses clean, benchmark URL datasets. Adversaries don't use clean URLs.

Cyrillic homoglyph substitution — replacing Latin characters with visually identical Unicode counterparts (e.g., Latin `a` → Cyrillic `а`) — is:
- **Zero-cost:** Requires only a character lookup table, no ML expertise
- **Visually undetectable:** Homoglyphs are pixel-identical at typical screen resolutions  
- **Actively deployed:** Documented in real phishing campaigns and phishing-as-a-service toolkits

A system reporting 97% accuracy may be running at 81% accuracy against any adversary who has read a threat intelligence report. SentinEL measures that gap precisely and identifies where the classifier fails.

---

## Research Question

> *Do ML phishing detectors learn the semantic structure of malicious URLs, or do they learn superficial string patterns that are brittle to character substitution?*

The Cyrillic homoglyph attack operationalises this question directly. A classifier that learned semantics should be robust to surface-level character swaps. One that learned string patterns should degrade.

**Answer: 15.8% degradation. The classifier learned surface patterns.**

---

## Methodology

### Dataset

- **Source:** PhishTank (verified phishing URLs) + curated legitimate URL corpus
- **Size:** ~11,000 URLs (balanced phishing / legitimate split)
- **Preprocessing:** Raw URL strings normalised to lowercase; no prior Unicode normalisation applied (by design — normalisation is evaluated as a mitigation, not a baseline assumption)
- **Train / Test split:** 80% training, 20% held-out test set

### Feature Engineering

17 features engineered from raw URL strings across four categories:

| Category | Features |
|----------|----------|
| **Length signals** | Domain length, URL total length, path depth |
| **Structural signals** | Subdomain depth, path segment count, query parameter count, `@` symbol presence |
| **Character-level signals** | Special character frequency, numeric character ratio, character entropy |
| **Semantic signals** | TLD risk score, keyword presence (login, secure, account, etc.), hyphen count |

### Classifier

**Random Forest** (n=100 estimators). Choice rationale:
- Ensemble method — resistant to single-feature overfitting
- Produces native Gini importance scores enabling the < 2 ms explainability path
- Strong baseline on URL classification in published literature
- Interpretable: feature importances can be audited by security analysts

### Adversarial Attack Construction

A substitution mapping replaces Latin characters in URL strings with visually identical Cyrillic Unicode characters. The attack was evaluated in two configurations:

1. **Post-feature-extraction:** Applied to features after extraction — isolates the feature-space degradation
2. **Pre-feature-extraction:** Applied to raw URL strings before the pipeline runs — models the realistic deployment scenario

The **81.4% figure reflects pre-extraction attacks**, the realistic adversarial scenario.

### Explainability Implementation

Random Forests compute cumulative Gini impurity reduction per feature across the ensemble during training. At inference time, the path taken through each tree identifies which features were active in the classification decision. Aggregating these path-level importances produces a per-prediction feature attribution in **< 2 ms** — without rerunning any perturbation loop.

**Trade-off acknowledged:** SHAP produces Shapley values with stronger theoretical guarantees. Gini attribution is an approximation. For SOC deployment, the latency advantage is the operative concern: an analyst receiving an approximate explanation in 2 ms alongside an alert is better served than receiving a theoretically ideal explanation 50–80 ms after the detection window has closed.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     INPUT LAYER                          │
│              Raw URL strings (clean or adversarial)      │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│               FEATURE ENGINEERING                        │
│  feature_extractor.py                                    │
│  • 17 URL-derived features                               │
│  • Domain tokenisation                                   │
│  • Character-level entropy                               │
│  • TLD risk scoring                                      │
└────────────────────────┬────────────────────────────────┘
                         │
                    ┌────┴────┐
                    │         │
         Clean path │         │ Adversarial path
                    │         │ (homoglyph substitution applied
                    │         │  before feature extraction)
                    ▼         ▼
┌─────────────────────────────────────────────────────────┐
│              RANDOM FOREST CLASSIFIER                    │
│  classifier.py  ·  100 estimators                        │
│  • Binary classification: phishing / legitimate          │
│  • Gini importance computed at training time             │
└────────────────┬────────────────┬───────────────────────┘
                 │                │
                 ▼                ▼
┌──────────────────┐  ┌──────────────────────────────────┐
│  PREDICTION      │  │  EXPLAINABILITY LAYER            │
│  phishing /      │  │  explainer.py                    │
│  legitimate      │  │  • Native Gini attribution       │
│                  │  │  • Per-prediction top-3 features  │
│                  │  │  • < 2 ms latency                 │
└──────────────────┘  └──────────────────────────────────┘
```

---

## Results Detail

### Where the Classifier Fails Under Attack

The 15.8% accuracy gap is not uniform across the feature space. Post-analysis showed:

- **Most degraded:** Features derived from raw character sequences (token n-grams, character frequency distributions). These features are directly disrupted by homoglyph substitution because the classifier has learned Latin character patterns that no longer match post-substitution.
- **Most stable:** Features derived from structural URL properties (subdomain depth, path segment count, query parameter count). The classifier's residual 81.4% accuracy is largely attributable to these structural features.

This decomposition directly motivates the mitigations below.

### Explainability Benchmark

Latency measured over 10,000 predictions (Intel Core i7, 16GB RAM). SHAP and LIME timings are consistent with published benchmarks. The < 2 ms figure for native Gini attribution is stable across the full test set (99th percentile < 5 ms).

---

## Practical Mitigations

Three practical interventions emerge from this analysis — ordered by implementation cost:

**1. Unicode normalisation at preprocessing (cost: negligible)**  
Normalise all input URLs to Unicode NFC/NFKC form before feature extraction. This collapses most homoglyph variants to their canonical Latin form before the classifier ever processes them. Eliminates the majority of the attack surface with a single preprocessing step.

**2. Feature set reweighting (cost: retraining)**  
Down-weight character-sequence features (which degraded severely under attack) in favour of structural URL features (which remained robust). Requires retraining but no architectural changes.

**3. Adversarial training (cost: dataset augmentation + retraining)**  
Include homoglyph-substituted samples in the training set. Classic adversarial training applied to a non-neural classifier. Expected to close most of the remaining robustness gap after (1) and (2) are applied.

---

## Repository Structure

```
SentinEL/
├── feature_extractor.py         # 17-feature URL feature engineering
├── classifier.py                # Random Forest training + evaluation
├── explainer.py                 # Native Gini attribution (< 2 ms)
├── adversarial/
│   ├── homoglyph_map.py         # Cyrillic ↔ Latin character mapping
│   └── attack_generator.py      # Applies substitution to URL datasets
├── experiments/
│   ├── baseline_eval.py         # Clean accuracy evaluation (→ 97.2%)
│   ├── adversarial_eval.py      # Accuracy under homoglyph attack (→ 81.4%)
│   └── explainability_bench.py  # SHAP vs LIME vs Gini latency comparison
├── data/
│   └── README.md                # Data provenance and preprocessing notes
├── docs/
│   └── SentinEL_Technical_Report.pdf  # Full 70-page technical report
└── requirements.txt
```

---

## Reproducing the Experiments

**Prerequisites:** Python 3.10+, `scikit-learn`, `shap`, `lime`

```bash
git clone https://github.com/Maze-6/SentinEL-Adversarial-ML
cd SentinEL-Adversarial-ML
pip install -r requirements.txt

# Reproduce baseline accuracy (97.2%)
python experiments/baseline_eval.py

# Reproduce adversarial accuracy (81.4%)
python experiments/adversarial_eval.py --attack cyrillic_homoglyph

# Reproduce explainability latency benchmark
python experiments/explainability_bench.py --n_samples 10000
```

---

## Limitations

| Limitation | Description |
|---|---|
| Single attack type | Only Cyrillic homoglyph substitution evaluated. Punycode, zero-width characters, Arabic Unicode attacks are out of scope and represent future work. |
| Single classifier | Results are specific to Random Forest on this feature set. Generalisability to deep learning approaches (CNNs on raw character sequences, transformers) is an open question. |
| Dataset scope | Results are dataset-specific. Replication on other URL corpora is required before broad generalisability claims. |
| Gini attribution | The < 2 ms explainability method is an approximation. Not appropriate for decisions requiring formal attribution guarantees. |

---

## Citation

If you use this work or reference the methodology, please cite:

```
Udumula, Mourya Reddy. "SentinEL: Adversarial Robustness of ML-Based Phishing 
Detectors under Character-Encoding Attacks." Undergraduate Research, 
Indrashil University, 2025–2026.
```

---

## Technical Report

The full 70-page technical report covering related work, complete methodology, extended results, and discussion is available at [`docs/SentinEL_Technical_Report.pdf`](docs/SentinEL_Technical_Report.pdf).

---

<div align="center">

*Undergraduate research — Indrashil University (2025–2026)*  
*Adversarial evaluation methodology and feature attribution approach developed independently.*  
[mouryaudumula@gmail.com](mailto:mouryaudumula@gmail.com)

</div>
