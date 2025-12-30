🛡️ SentinEL: Ultima Intelligence Engine
Next-Gen Phishing Detection using Explainable AI (XAI)
Architects: Mourya R. Udumula & Jeet Upadhyaya | Institution: Indrashil University

🚀 Project Overview
SentinEL Ultima is a Hybrid Threat Intelligence Engine designed to detect sophisticated phishing attacks that bypass traditional blacklists. It combines a high-speed Random Forest classifier with real-time Forensic Analysis (WHOIS, DNS, SSL) to deliver verdicts with <150ms latency.

The system features Active Learning, allowing security analysts to flag false positives and retrain the decision boundary in real-time (Session Scope).

🛠️ Technical Stack
ML Core: Scikit-Learn (Random Forest, Pipelines), Pandas
Forensics: python-whois, dnspython, SSL Socket Inspection
Concurrency: concurrent.futures for high-velocity batch triage
Interface: Streamlit (Dynamic Dashboard)
⚡ Key Capabilities
Hybrid Engine: Combines Allowlisting (O(1) lookup) with ML Heuristics.
Explainable AI (XAI): Provides human-readable reasons for every verdict (e.g., "High Entropy", "Expired SSL").
Resilience: Self-healing CSV parsing logic for handling dirty IOC feeds.
Adversarial Defense: Detects DGA (Domain Generation Algorithms) via Shannon Entropy analysis.
🔧 Installation & Usage
# Clone the repository
git clone https://github.com/CassianLee14/SentinEL-Adversarial-ML.git

# Install dependencies
pip install -r requirements.txt

# Launch the Dashboard
streamlit run app.py
