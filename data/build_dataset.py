"""
SentinEL — Dataset Builder
Reads the locally downloaded Kaggle "Phishing Site URLs" dataset.

Source : https://www.kaggle.com/datasets/taruntiwarihp/phishing-site-urls
File   : data/phishing_site_urls.csv
Columns: URL (bare URL string, no scheme), Label ("bad" = phishing, "good" = legitimate)

Outputs:
  data/phishing_dataset.csv  — 17-feature matrix used for model training
  data/url_list.csv          — raw URL + label list (for batch evaluator)

Run:
  python data/build_dataset.py
"""

import pandas as pd
import sys
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from feature_extractor import FeatureExtractor

RAW_FILE        = ROOT / "data" / "phishing_site_urls.csv"
OUT_FEATURES    = ROOT / "data" / "phishing_dataset.csv"
OUT_URLS        = ROOT / "data" / "url_list.csv"

TARGET_PER_CLASS = 5500


def is_clean_url(url: str) -> bool:
    if not isinstance(url, str):
        return False
    if len(url) < 6:
        return False
    try:
        url.encode("ascii")
    except UnicodeEncodeError:
        return False
    if "." not in url:
        return False
    return True


def add_scheme(url: str) -> str:
    url = url.strip()
    if url.startswith("http://") or url.startswith("https://"):
        return url
    return "http://" + url


# ── 1. Load raw CSV ──────────────────────────────────────────────────────────
if not RAW_FILE.exists():
    print(f"ERROR: File not found: {RAW_FILE}")
    print("Place phishing_site_urls.csv in the data/ folder.")
    sys.exit(1)

print(f"Loading {RAW_FILE.name}  ...")
raw = pd.read_csv(RAW_FILE, dtype=str)
print(f"  Raw rows   : {len(raw):,}")
print(f"  Columns    : {list(raw.columns)}")

# ── 2. Normalise column names ────────────────────────────────────────────────
raw.columns = [c.strip().lower() for c in raw.columns]

# ── 3. Normalise labels ───────────────────────────────────────────────────────
label_map = {
    "bad":        1,
    "phishing":   1,
    "malicious":  1,
    "good":       0,
    "legitimate": 0,
    "benign":     0,
}
raw["label"] = raw["label"].str.strip().str.lower().map(label_map)
raw = raw.dropna(subset=["label", "url"])
raw["label"] = raw["label"].astype(int)

print(f"\n  After label mapping:")
print(f"    Phishing (bad=1)   : {(raw.label==1).sum():,}")
print(f"    Legitimate (good=0): {(raw.label==0).sum():,}")

# ── 4. Filter out corrupt/garbage URLs ───────────────────────────────────────
raw["url"] = raw["url"].astype(str).str.strip()
clean_mask = raw["url"].apply(is_clean_url)
raw = raw[clean_mask].reset_index(drop=True)

print(f"\n  After removing garbage rows:")
print(f"    Phishing  : {(raw.label==1).sum():,}")
print(f"    Legitimate: {(raw.label==0).sum():,}")

# ── 5. Add http:// scheme ─────────────────────────────────────────────────────
raw["url"] = raw["url"].apply(add_scheme)

# ── 6. Sample balanced dataset ───────────────────────────────────────────────
n_phish = min(TARGET_PER_CLASS, (raw.label == 1).sum())
n_legit = min(TARGET_PER_CLASS, (raw.label == 0).sum())

phishing_df = raw[raw.label == 1].sample(n=n_phish, random_state=42)
legit_df    = raw[raw.label == 0].sample(n=n_legit, random_state=42)

combined = pd.concat([phishing_df, legit_df]) \
             .sample(frac=1, random_state=42) \
             .reset_index(drop=True)

print(f"\n  Sampled for training:")
print(f"    Phishing  : {(combined.label==1).sum():,}")
print(f"    Legitimate: {(combined.label==0).sum():,}")
print(f"    Total     : {len(combined):,}")

# ── 7. Save URL list ──────────────────────────────────────────────────────────
combined[["url", "label"]].to_csv(OUT_URLS, index=False)
print(f"\n  URL list saved → {OUT_URLS}")

# ── 8. Extract 17 features for every URL ─────────────────────────────────────
feature_names = FeatureExtractor.get_feature_names(mode="offline")
print(f"\n  Extracting {len(feature_names)} features for {len(combined):,} URLs ...")
print(f"  (Expected time: 5-15 minutes)")
print(f"  Progress updates every 500 URLs.\n")

rows   = []
failed = 0

for i, row in combined.iterrows():
    try:
        feats = FeatureExtractor.extract_features(row["url"], mode="offline")
        rows.append(list(feats) + [int(row["label"])])
    except Exception as e:
        failed += 1
        if failed <= 5:
            print(f"    ERROR on: {row['url'][:60]}")
            print(f"    {type(e).__name__}: {e}")
        continue

    count = len(rows) + failed
    if count % 500 == 0:
        pct = count / len(combined) * 100
        print(f"    {count:>6,} / {len(combined):,}  ({pct:5.1f}%)  |  "
              f"ok: {len(rows):,}  failed: {failed}")

print(f"\n  Feature extraction complete.")
print(f"    Successful : {len(rows):,}")
print(f"    Failed     : {failed}")

if len(rows) < 1000:
    print("\nERROR: Fewer than 1,000 URLs processed successfully.")
    print("Check the ERROR messages printed above for the root cause.")
    sys.exit(1)

# ── 9. Save feature matrix ────────────────────────────────────────────────────
columns    = feature_names + ["label"]
dataset_df = pd.DataFrame(rows, columns=columns)
dataset_df.to_csv(OUT_FEATURES, index=False)

# ── 10. Final summary ─────────────────────────────────────────────────────────
total     = len(dataset_df)
n_phish_f = int((dataset_df.label == 1).sum())
n_legit_f = int((dataset_df.label == 0).sum())
balance   = n_phish_f / total * 100

print(f"\n{'='*52}")
print(f"  DATASET BUILD COMPLETE")
print(f"{'='*52}")
print(f"  Total rows  : {total:,}")
print(f"  Phishing    : {n_phish_f:,}")
print(f"  Legitimate  : {n_legit_f:,}")
print(f"  Features    : {len(feature_names)}")
print(f"  Balance     : {balance:.1f}% phishing")
print(f"{'='*52}")
print(f"\n  Files saved:")
print(f"    {OUT_FEATURES}")
print(f"    {OUT_URLS}")
print(f"\n  Next step:")
print(f"    python experiments/baseline_eval.py --retrain")