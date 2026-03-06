# 🛡️ SentinEL: Adversarial ML Robustness Research

**Lead Researcher & ML Architect:** Mourya Reddy Udumula
**Operations & Pipeline Lead:** Jeet Anand Upadhyaya
**Presented at:** Indrashil University Research Symposium, January 2026

---

## 🧠 What Is SentinEL?

SentinEL is an adversarial ML robustness research platform for phishing URL detection. The central finding of this research is that standard Random Forest classifiers — even those achieving 97%+ accuracy on clean data — can be systematically bypassed using Unicode character-encoding manipulation.

The project quantifies this robustness gap, builds a production-grade detection pipeline with explainable verdicts, and evaluates alternative explainability methods under real SOC latency constraints.

**Research question:** Can pattern-matching ML detectors defend against character-encoding manipulation, or do they learn superficial string features rather than semantic intent?
**Answer:** They learn superficial patterns. The 15.8% robustness gap is the evidence.

---

## 📊 Key Research Metrics

| Metric | Value |
|--------|-------|
| Baseline accuracy (clean data) | 97.2% |
| Accuracy under Cyrillic homoglyph attacks | 81.4% |
| Robustness gap | **15.8%** |
| Native XAI attribution latency | < 2 ms |
| SHAP / LIME latency (rejected alternative) | 50–80 ms |
| Feature dimensions | 17 |
| ROC-AUC | 1.00 |

The 15.8% degradation is not a model training failure — it is a fundamental property of pattern-matching classifiers. They cannot reason about attacker intent, only character sequences.

---

## 🔬 Research Methodology

### Stage 1: Baseline Model

A Random Forest classifier was trained on a standard phishing dataset using 17 hand-engineered URL features (domain age, entropy, special character ratios, TLD risk scores, WHOIS forensics, etc.). Baseline accuracy: 97.2%.

### Stage 2: Adversarial Attack Design

Cyrillic homoglyph substitution attacks were designed to exploit the Unicode ambiguity exploited by real phishing actors. Latin characters (e.g., `a`, `e`, `o`, `p`, `c`) are replaced with visually identical Cyrillic equivalents (e.g., `а`, `е`, `о`, `р`, `с`). The URL is visually indistinguishable to a human but has entirely different byte-level features.

### Stage 3: Robustness Evaluation

The trained model was evaluated against the adversarial corpus. Accuracy fell from 97.2% to 81.4% — a 15.8% degradation — confirming the classifier had learned string-level patterns rather than semantic structure.

### Stage 4: Explainability Framework

SHAP and LIME were evaluated as explainability methods but rejected due to 50–80ms inference overhead — incompatible with SOCs processing 500+ alerts/hour requiring real-time verdicts. A native attribution method using Gini importance weights was implemented instead, achieving <2ms latency while producing human-readable forensic justifications.

---

## 🏗️ System Architecture

```
Input URL
    │
    ▼
┌─────────────────────────────────────────────────────┐
│                  Triage Pipeline                     │
│                                                      │
│  Stage 1: Reputation Allowlist                       │
│    └── Known-safe domains → LEGITIMATE (instant)    │
│                                                      │
│  Stage 2: Forensic Heuristics                        │
│    └── WHOIS age, entropy, TLD, IP-in-URL checks    │
│                                                      │
│  Stage 3: Random Forest Classifier (17 features)    │
│    └── Probability score → Verdict                  │
│                                                      │
│  Stage 4: Native XAI Attribution (<2ms)             │
│    └── Gini importance → Human-readable reasons     │
└─────────────────────────────────────────────────────┘
    │
    ▼
Verdict: LEGITIMATE / SUSPICIOUS / PHISHING + Reasons
```

---

## 📂 Engineering Attribution

| Module | Lead | Core Technology |
|--------|------|----------------|
| `classifier.py` | Mourya Udumula | Random Forest, adversarial testing, XAI attribution |
| `feature_extractor.py` | Mourya Udumula | 17-dimensional forensic URL vectorization |
| Analytics / `experiments/` | Mourya Udumula | ROC-AUC, confusion matrix, model calibration viz |
| Batch Triage / `pipeline/` | Jeet Upadhyaya | Concurrent IOC ingestion pipeline |
| Audit Logs | Jeet Upadhyaya | Session event logging and export |
| Forensic heuristics | Jeet Upadhyaya | WHOIS forensics, domain age, IP reputation |

---

## 🖥️ Dashboard Features

### 🔍 Forensic Scanner

Live single-URL scan with full verdict breakdown. Enter any URL and receive a probability score, verdict classification (LEGITIMATE / SUSPICIOUS / PHISHING), and a list of forensic reasons derived from the XAI attribution layer. Analysts can flag false positives to build a session-level whitelist.

### 📦 Batch Triage

Upload or paste a list of URLs for concurrent batch evaluation. Results are returned with per-URL verdicts, probability scores, and downloadable CSV export. Built for SOC analysts processing bulk IOC feeds.

### 📊 Analytics

Model performance visualization: confusion matrix, ROC curve (AUC = 1.00), F1/Precision/Recall metrics. Displays the statistical validation underpinning the 97.2% baseline accuracy claim.

### 📝 Audit Logs

Full session event log with timestamps, event categories (SCAN, BATCH, OVERRIDE, FEEDBACK, SYSTEM), and CSV download for offline analysis.

---

## 🔧 Installation & Quickstart

```bash
# Clone
git clone https://github.com/Maze-6/SentinEL-Adversarial-ML.git
cd SentinEL-Adversarial-ML

# Install dependencies
pip install -r requirements.txt

# Run tests
python -m pytest tests/ -v

# Run adversarial robustness evaluation
python experiments/adversarial_eval.py

# Run baseline evaluation
python experiments/baseline_eval.py

# Run batch evaluation on a CSV
python pipeline/batch_evaluate.py --csv data/url_list.csv
```

---

## 📦 Requirements

```
streamlit
pandas
scikit-learn
matplotlib
seaborn
requests
dnspython
beautifulsoup4
python-whois
```

---

## 📁 Repository Structure

```
SentinEL-Adversarial-ML/
├── classifier.py            # Random Forest model + train/predict pipeline
├── feature_extractor.py     # 17-feature URL vectorization (offline + enriched modes)
├── explainer.py             # Native Gini XAI attribution (<2ms)
├── config.py                # Model paths, thresholds, dataset config
├── requirements.txt
├── adversarial/
│   ├── homoglyph_map.py     # Cyrillic ↔ Latin character substitution map
│   └── attack_generator.py  # Adversarial URL corpus generation
├── experiments/
│   ├── baseline_eval.py     # Clean-data evaluation (ROC-AUC, F1, FNR, FPR, 5-fold CV)
│   ├── adversarial_eval.py  # Homoglyph attack robustness evaluation
│   ├── explainability_bench.py  # Gini vs SHAP vs LIME latency comparison
│   └── generate_graphs.py   # Publication-quality chart generation
├── pipeline/
│   └── batch_evaluate.py    # Single-URL and CSV batch evaluation CLI
├── data/
│   ├── dataset.csv          # Training dataset (phishing + legitimate URLs)
│   └── build_dataset.py     # Dataset construction script (PhishTank + Tranco)
├── tests/
│   └── test_sentinEL.py     # 27-test pytest suite
├── docs/
│   └── SentinEL_Technical_Report.pdf
└── assets/
    ├── adversarial_detection.png
    ├── analytics_metrics.png
    └── ...
```

---

## 🔬 Key Research Findings

### Finding 1: Pattern-matching classifiers cannot generalize to adversarial inputs

The model learned that certain character sequences are associated with phishing. Cyrillic substitution changes those sequences while preserving visual appearance, bypassing learned patterns entirely. This is not a hyperparameter problem — it is a fundamental architectural limitation.

### Finding 2: SHAP and LIME are impractical for production SOC deployment

Both methods require inference times of 50–80ms per URL. A SOC processing 500 alerts/hour at this latency would spend 7–11 hours on explainability computation alone. Native Gini attribution at <2ms makes real-time deployment viable.

### Finding 3: Solving adversarial brittleness requires constraint-based reasoning, not better features

Adding more features trained on clean data will not close the robustness gap, because the attack operates at the Unicode level — below the abstraction layer the model reasons about. Closing this gap requires either constraint-based URL normalization at the feature extraction stage, or a reasoning layer that understands attacker intent.

This finding motivates the proposed MSc thesis direction: integrating human analyst-labeled adversarial examples into a Case-Based Reasoning system for adaptive phishing detection.

---

## 📖 Technical Report

A 70-page technical report covering full methodology, statistical analysis, adversarial attack taxonomy, and explainability framework evaluation is available in [`docs/SentinEL_Technical_Report.pdf`](docs/SentinEL_Technical_Report.pdf).

---

## 🔗 Related Project

[VaultZero](https://github.com/Maze-6/VaultZero-Core) — Fault-tolerant distributed storage with threshold cryptography

---

*Senior capstone research — Indrashil University*
[mouryaudumula@gmail.com](mailto:mouryaudumula@gmail.com)
