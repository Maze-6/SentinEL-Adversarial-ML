import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import concurrent.futures
import datetime
import time
from ml_engine import PhishModel
import config

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="SentinEL Ultima", page_icon="🛡️", layout="wide", initial_sidebar_state="expanded")

# --- 2. CUSTOM CSS ---
st.markdown("""
<style>
    .title-text { font-size: 2.2rem; font-weight: 800; color: #0f172a; margin-bottom: 0px; }
    .subtitle-text { font-size: 1.0rem; color: #475569; font-style: italic; margin-bottom: 20px; }
    .stMetric { background-color: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 15px; }
    .badge-safe { background-color: #dcfce7; color: #166534; padding: 5px 10px; border-radius: 12px; font-weight: bold; border: 1px solid #bbf7d0; }
    .badge-risk { background-color: #fee2e2; color: #991b1b; padding: 5px 10px; border-radius: 12px; font-weight: bold; border: 1px solid #fecaca; }
    .badge-manual { background-color: #e0f2fe; color: #0369a1; padding: 5px 10px; border-radius: 12px; font-weight: bold; border: 1px solid #bae6fd; }
    .block-container { padding-top: 2rem; }
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# --- 3. SYSTEM CORE ---
@st.cache_resource
def load_engine():
    return PhishModel()

model = load_engine()

# --- 4. STATE MANAGEMENT ---
if 'logs' not in st.session_state: st.session_state['logs'] = []
if 'scan_data' not in st.session_state: st.session_state['scan_data'] = None
if 'batch_results' not in st.session_state: st.session_state['batch_results'] = None
if 'manual_whitelist' not in st.session_state: st.session_state['manual_whitelist'] = []

def log_event(category, details):
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    st.session_state['logs'].insert(0, {"Time": ts, "Event": category, "Details": details})

def run_calibration():
    success, msg = model.train_model("dataset.csv")
    if success: log_event("SYSTEM", f"Recalibration: {msg}")

# --- 5. SIDEBAR ---
with st.sidebar:
    st.markdown("# 🛡️ SentinEL Ops") 
    st.caption("👨‍💻 **Mourya Reddy Udumula**")
    st.caption("👨‍💻 **Jeet Anand Upadhyaya**")
    st.divider()
    threshold = st.slider("Sensitivity Threshold", 0.0, 1.0, 0.65, 0.05)
    col1, col2 = st.columns(2)
    with col1:
        if st.button("⚡ Calibrate"):
            run_calibration()
            time.sleep(0.5)
            st.rerun()
    with col2:
        if st.button("🔄 Reset"):
            st.session_state.clear()
            st.rerun()
    st.divider()
    with st.expander("ℹ️ About the Algorithm"):
        st.info("Hybrid Engine: Allowlist + WHOIS Forensics + Random Forest + XAI.")

# --- 6. MAIN LAYOUT ---
st.markdown('<div class="title-text">SentinEL: Ultima Intelligence Engine</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle-text">Next-Gen Phishing Detection using Explainable AI (XAI)</div>', unsafe_allow_html=True)

t1, t2, t3, t4 = st.tabs(["🔍 Forensic Scanner", "📦 Batch Triage", "📊 Analytics", "📝 Audit Logs"])

with t1:
    st.markdown("##### Live Endpoint Diagnostics")
    with st.container():
        with st.form(key='scanner_form'):
            c1, c2 = st.columns([4, 1])
            u_in = c1.text_input("Enter Target URL", "http://crunchyroll.com")
            submit_btn = c2.form_submit_button("🛡️ Scan Target", use_container_width=True)
        
        if submit_btn:
            if u_in in st.session_state['manual_whitelist']:
                st.session_state['scan_data'] = {'url': u_in, 'raw_prob': 0.0, 'reasons': ["Analyst Override: Manually Whitelisted"]}
                log_event("OVERRIDE", f"Allowed {u_in}")
            else:
                with st.spinner("Querying DNS & Parsing DOM..."):
                    _, p, r = model.predict_vanguard(u_in, threshold)
                    st.session_state['scan_data'] = {'url': u_in, 'raw_prob': p, 'reasons': r}
                    log_event("SCAN", f"Scanned {u_in}")

        if st.session_state['scan_data']:
            data = st.session_state['scan_data']
            prob = data['raw_prob']
            if prob >= threshold: current_verdict = "PHISHING"
            elif prob >= (threshold - 0.20) and prob > 0: current_verdict = "SUSPICIOUS"
            else: current_verdict = "LEGITIMATE"
            st.divider()
            m1, m2 = st.columns([1, 2])
            with m1:
                st.metric("Probability", f"{prob*100:.1f}%")
                if current_verdict == "PHISHING": st.error(f"⛔ {current_verdict}")
                elif current_verdict == "SUSPICIOUS": st.warning(f"⚠️ {current_verdict}")
                else: st.success(f"✅ {current_verdict}")
                if data['url'] not in st.session_state['manual_whitelist'] and current_verdict != "LEGITIMATE":
                    if st.button("🚩 Flag as False Positive"):
                        st.session_state['manual_whitelist'].append(data['url'])
                        log_event("FEEDBACK", f"User marked {data['url']} as Safe")
                        st.success("System Updated.")
                        time.sleep(0.5); st.rerun()
            with m2:
                st.markdown("#### XAI Analysis Report")
                if not data['reasons']: st.markdown("<div class='badge-safe'>✅ Verified Legitimate</div>", unsafe_allow_html=True)
                for x in data['reasons']: 
                    if "Analyst" in x: st.markdown(f"<div class='badge-manual'>👨‍💻 {x}</div>", unsafe_allow_html=True)
                    elif "Trust:" in x or "Allowlist" in x: st.markdown(f"<div class='badge-safe'>🛡️ {x}</div>", unsafe_allow_html=True)
                    else: st.markdown(f"<div class='badge-risk'>⚠️ {x}</div>", unsafe_allow_html=True)

with t2:
    st.markdown("##### High-Velocity Threat Pipeline")
    feed = st.file_uploader("Upload IOC Feed (CSV)", type=['csv'])
    if feed and model.model:
        df_feed = pd.read_csv(feed)
        target_col = None
        standard_headers = ['url', 'urls', 'link', 'links', 'website', 'domain', 'target']
        for col in df_feed.columns:
            if str(col).lower().strip() in standard_headers:
                target_col = col
                break
        urls_to_scan = []
        if target_col: urls_to_scan = df_feed[target_col].dropna().astype(str).tolist()
        else:
            target_col = df_feed.columns[0]; urls_to_scan = df_feed[target_col].dropna().astype(str).tolist()
            if "http" in str(target_col) or "www" in str(target_col): urls_to_scan.insert(0, str(target_col))
        if st.button(f"🚀 Process {len(urls_to_scan)} URLs"):
            results = []
            bar = st.progress(0)
            def task(u):
                v, p, _ = model.predict_vanguard(u, threshold)
                return {"URL": u, "Verdict": v, "Risk": round(p, 4)}
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                futures = {executor.submit(task, u): u for u in urls_to_scan}
                for i, future in enumerate(concurrent.futures.as_completed(futures)):
                    results.append(future.result())
                    bar.progress((i+1)/len(urls_to_scan))
            st.session_state['batch_results'] = pd.DataFrame(results)
            log_event("BATCH", f"Processed {len(results)} URLs")
            bar.empty()
    if st.session_state['batch_results'] is not None:
        res_df = st.session_state['batch_results']
        c_a, c_b, c_c = st.columns(3)
        c_a.metric("Total IOCs", len(res_df))
        c_b.metric("Threats Found", len(res_df[res_df['Verdict'] == 'PHISHING']), delta_color="inverse")
        c_c.metric("Clean URLs", len(res_df[res_df['Verdict'] == 'LEGITIMATE']))
        st.divider()
        v1, v2 = st.columns([1, 2])
        with v1:
            fig1, ax1 = plt.subplots(figsize=(4, 4))
            counts = res_df['Verdict'].value_counts()
            colors = {'PHISHING': '#ef4444', 'LEGITIMATE': '#22c55e', 'SUSPICIOUS': '#f59e0b'}
            ax1.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=90, colors=[colors.get(x, '#94a3b8') for x in counts.index])
            st.pyplot(fig1)
        with v2:
            def highlight(val):
                if val == 'PHISHING': return 'background-color: #fee2e2; color: #991b1b'
                elif val == 'LEGITIMATE': return 'background-color: #dcfce7; color: #166534'
                return 'background-color: #fef3c7; color: #b45309'
            st.dataframe(res_df.style.applymap(highlight, subset=['Verdict']), use_container_width=True)

with t3:
    if model.model:
        m_col1, m_col2, m_col3 = st.columns(3)
        m_col1.metric("F1-Score", f"{model.f1:.3f}")
        m_col2.metric("Precision", f"{model.precision:.3f}")
        m_col3.metric("Recall", f"{model.recall:.3f}")
        st.divider()
        g1, g2 = st.columns(2)
        with g1:
            if model.cm is not None:
                fig2, ax2 = plt.subplots(); sns.heatmap(model.cm, annot=True, fmt='d', cmap='Blues', ax=ax2)
                ax2.set_xlabel('Predicted'); ax2.set_ylabel('Actual'); st.pyplot(fig2)
        with g2:
            if model.roc_fpr is not None:
                fig3, ax3 = plt.subplots(); ax3.plot(model.roc_fpr, model.roc_tpr, color='darkorange', lw=2, label=f'AUC = {model.roc_auc:.2f}')
                ax3.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--'); ax3.legend(); st.pyplot(fig3)

with t4:
    if st.session_state['logs']:
        log_df = pd.DataFrame(st.session_state['logs'])
        st.dataframe(log_df, use_container_width=True)
        st.download_button("⬇️ Download Logs (CSV)", log_df.to_csv(index=False).encode('utf-8'), "sentinel_audit.csv", "text/csv")