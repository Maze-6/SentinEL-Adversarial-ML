import pandas as pd
import numpy as np
import pickle, os, datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from feature_extractor import FeatureExtractor
import config

class PhishModel:
    def __init__(self):
        self.model_file = "sentinel_ultima.pkl"
        self.feature_names = FeatureExtractor.get_feature_names()
        self.loaded_data = self.load_existing_model()
        self.model = self.loaded_data.get('model') if self.loaded_data else None
        
        self.f1, self.precision, self.recall = 0.0, 0.0, 0.0
        self.cm = None
        self.roc_fpr, self.roc_tpr, self.roc_auc = None, None, 0.0

    def load_existing_model(self):
        if os.path.exists(self.model_file):
            try:
                data = pickle.load(open(self.model_file, "rb"))
                if 'feature_count' in data and data['feature_count'] == len(self.feature_names):
                    return data
            except: pass
        return None

    def train_model(self, csv_path: str):
        try:
            df = pd.read_csv(csv_path)
            if len(df) < 50: df = pd.concat([df] * 5, ignore_index=True)
            X = df.iloc[:, 2:19]
            X.columns = self.feature_names
            y = df['label']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('clf', RandomForestClassifier(n_estimators=150, max_depth=12, random_state=42))
            ])
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            y_probs = pipeline.predict_proba(X_test)[:, 1]
            self.f1 = f1_score(y_test, y_pred)
            self.precision = precision_score(y_test, y_pred, zero_division=0)
            self.recall = recall_score(y_test, y_pred)
            self.cm = confusion_matrix(y_test, y_pred)
            self.roc_fpr, self.roc_tpr, _ = roc_curve(y_test, y_probs)
            self.roc_auc = auc(self.roc_fpr, self.roc_tpr)
            pipeline.fit(X, y)
            payload = {'model': pipeline, 'feature_count': len(self.feature_names), 'trained_at': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'dataset_size': len(df)}
            pickle.dump(payload, open(self.model_file, "wb"))
            self.model = pipeline
            self.loaded_data = payload
            return True, f"Calibration Complete. AUC: {self.roc_auc:.2f}"
        except Exception as e: return False, str(e)

    def predict_vanguard(self, url, threshold):
        reasons = []
        
        # 1. ALLOWLIST CHECK
        clean_url = url.lower().replace("https://","").replace("http://","").replace("www.","").split('/')[0]
        if clean_url in config.TRUSTED_DOMAINS or any(clean_url.endswith("." + trusted) for trusted in config.TRUSTED_DOMAINS):
             return "LEGITIMATE", 0.01, ["Verified Global Authority (Allowlist)"]

        # --- ADVERSARIAL HOMOGLYPH CHECK ---
        is_homoglyph = any(ord(c) > 127 for c in url)
        if is_homoglyph:
            reasons.append("Adversarial Risk: Unicode/Homoglyph Perturbation Detected")

        # 2. FEATURE EXTRACTION
        feats = FeatureExtractor.extract_features(url)
        
        # 3. DNS VERIFICATION
        if feats[8] == 0: 
            reasons.append("CRITICAL: Domain Resolution Failed")
            # If it's a homoglyph or bad DNS, it's highly likely phishing
            return "PHISHING", 1.0, reasons 

        # 4. REPUTATION OVERRIDE
        if feats[9] > 365:
            return "LEGITIMATE", 0.15, [f"Trust: Established Domain ({feats[9]} days old)"]

        # 5. ML PREDICTION
        if not self.model: return "Offline", 0.0, ["Model Calibration Required"]
        input_df = pd.DataFrame([feats], columns=self.feature_names)
        prob = self.model.predict_proba(input_df)[0][1]

        if prob >= threshold: v = "PHISHING"
        elif prob >= (threshold - 0.20): v = "SUSPICIOUS"
        else: v = "LEGITIMATE"
            
        # 6. ADD HEURISTIC ATTRIBUTION
        if feats[11] == 1: reasons.append("Behavioral: Data Capture Form Detected")
        if feats[16] == 1: reasons.append("Cryptographic: SSL Certificate Risk")
        if feats[3] == 1: reasons.append("Lexical: High-Risk TLD detected")
        if prob > threshold and not reasons: reasons.append("Heuristic: Statistical Pattern Match")
            
        return v, prob, reasons