# Data

## Dataset
- **Source:** PhishTank (verified phishing URLs) + curated legitimate URL corpus
- **Size:** ~11,000 URLs, balanced split
- **Split:** 80% training / 20% test
- **Features:** 17 URL-derived features (see feature_extractor.py for full list)

## Files
- `dataset.csv` — raw labelled URL dataset
- `sentinel_ultima.pkl` — trained Random Forest model (serialised)

## Provenance
PhishTank data retrieved under open access terms. Legitimate URL corpus curated
from public web directories. No personally identifiable information present.

## Generated Files (Not Committed)

The following files are generated at runtime and are excluded from version control:

- `sentinel_ultima.pkl` — Trained Random Forest model. Generate by running:
  `python experiments/baseline_eval.py`

- `model_checksum.sha256` — SHA-256 integrity checksum for the model file.
  Generated automatically when baseline_eval.py trains or saves the model.

These files are excluded from git because:
1. Binary model files bloat repository history permanently
2. The model is fully reproducible from dataset.csv and baseline_eval.py
3. Committing the pkl alongside its checksum would defeat the integrity verification
