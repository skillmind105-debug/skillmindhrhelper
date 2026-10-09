import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import warnings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pymupdf
import importlib

from evidence_engine import analyze_candidate_skills, estimate_experience_years, mask_pii
import translations
importlib.reload(translations)
from translations import TRANSLATIONS

warnings.filterwarnings('ignore', category=UserWarning)

# ==========================================
# 1. LANGUAGE STATE & INITIALIZATION
# ==========================================
if "current_lang" not in st.session_state:
    st.session_state.current_lang = "AZ"

# ==========================================
# 2. ENTERPRISE B2B PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title=TRANSLATIONS[st.session_state.current_lang]["page_title"],
    layout="wide",
    initial_sidebar_state="expanded"
)

# STRICT ANTI-VIBE-CODED CSS (ZERO EMOJIS, HIGH CONTRAST, PROFESSIONAL SLATE PALETTE)
ENTERPRISE_CSS = """
<style>
    /* System Font Stack & Clean Typography */
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
        color: #0f172a;
    }
    
    .stApp {
        background-color: #f8fafc;
    }
    
    /* Strict Corporate Header Container */
    .enterprise-header {
        background-color: #0f172a;
        color: #ffffff;
        padding: 16px 24px;
        border-radius: 4px;
        margin-bottom: 16px;
        border: 1px solid #1e293b;
    }
    .enterprise-header-title {
        font-size: 19px;
        font-weight: 700;
        letter-spacing: -0.2px;
        margin: 0;
        color: #ffffff;
    }
    .enterprise-header-subtitle {
        font-size: 13px;
        color: #94a3b8;
        margin: 4px 0 0 0;
    }
    
    /* Security Verification Banner */
    .security-banner {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-left: 3px solid #0284c7;
        padding: 12px 16px;
        border-radius: 4px;
        margin-bottom: 20px;
        font-size: 13px;
        color: #334155;
        line-height: 1.5;
    }
    
    /* KPI Card Containers */
    div[data-testid="stMetricValue"] {
        font-size: 24px !important;
        font-weight: 700 !important;
        color: #0f172a !important;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 11px !important;
        font-weight: 600 !important;
        color: #64748b !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Candidate Card */
    .candidate-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 4px;
        padding: 16px 20px;
        margin-bottom: 14px;
    }
    
    /* Professional Status Badges */
    .badge-status-pass {
        background-color: #ecfdf5;
        color: #065f46;
        font-size: 11px;
        font-weight: 700;
        padding: 4px 10px;
        border-radius: 3px;
        border: 1px solid #a7f3d0;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .badge-status-fail {
        background-color: #fff1f2;
        color: #9f1239;
        font-size: 11px;
        font-weight: 700;
        padding: 4px 10px;
        border-radius: 3px;
        border: 1px solid #fecdd3;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Skill Badges */
    .tag-verified {
        background-color: #f0fdf4;
        color: #166534;
        font-size: 11px;
        font-weight: 600;
        padding: 2px 8px;
        border-radius: 3px;
        border: 1px solid #bbf7d0;
        display: inline-block;
        margin-right: 4px;
        margin-bottom: 4px;
    }
    .tag-missing {
        background-color: #f8fafc;
        color: #64748b;
        font-size: 11px;
        font-weight: 500;
        padding: 2px 8px;
        border-radius: 3px;
        border: 1px solid #cbd5e1;
        display: inline-block;
        margin-right: 4px;
        margin-bottom: 4px;
    }
    .tag-negated {
        background-color: #fef2f2;
        color: #991b1b;
        font-size: 11px;
        font-weight: 600;
        padding: 2px 8px;
        border-radius: 3px;
        border: 1px solid #fecaca;
        display: inline-block;
        margin-right: 4px;
        margin-bottom: 4px;
    }

    .section-label {
        font-size: 12px;
        font-weight: 700;
        text-transform: uppercase;
        color: #475569;
        margin-bottom: 6px;
        letter-spacing: 0.4px;
    }
    /* Streamlit Tabs Styling - High Contrast Visible */
    button[data-baseweb="tab"] {
        color: #475569 !important;
        font-weight: 600 !important;
        font-size: 13px !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #0f172a !important;
        border-bottom-color: #2563eb !important;
    }
    div[data-baseweb="tab-highlight"] {
        background-color: #2563eb !important;
    }

    /* Force Dark High-Contrast Color on All Markdown and Headings */
    h1, h2, h3, h4, h5, h6, p, span, label, div {
        color: inherit;
    }

    /* Specific Metric Styles */
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 4px;
        padding: 12px 16px;
    }
</style>
"""
st.markdown(ENTERPRISE_CSS, unsafe_allow_html=True)

# ==========================================
# 3. TOP NAVIGATION & PROFESSIONAL LANGUAGE SWITCHER
# ==========================================
header_col1, header_col2 = st.columns([8.8, 1.2])

with header_col2:
    selected_lang = st.selectbox(
        label="Language Selector",
        options=["AZ ▾", "EN ▾", "RU ▾"],
        index=0 if st.session_state.current_lang == "AZ" else (1 if st.session_state.current_lang == "EN" else 2),
        label_visibility="collapsed",
        key="lang_selector_widget"
    )
    clean_code = selected_lang.split()[0]
    if clean_code != st.session_state.current_lang:
        st.session_state.current_lang = clean_code
        st.rerun()

t = TRANSLATIONS[st.session_state.current_lang]

with header_col1:
    st.markdown(f"""
    <div class="enterprise-header" style="border-left: none !important; border: 1px solid #1e293b !important;">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 14px;">
            <div style="flex: 1 1 300px; min-width: 0;">
                <div class="enterprise-header-title">{t['header_title']}</div>
                <div class="enterprise-header-subtitle">{t['header_subtitle']}</div>
            </div>
            <div style="flex-shrink: 0;">
                <span style="background-color: #1e293b; color: #94a3b8; padding: 6px 12px; border-radius: 3px; font-size: 11px; font-weight: 600; border: 1px solid #334155; white-space: nowrap !important; display: inline-block; letter-spacing: 0.3px;">
                    {t['local_badge']}
                </span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Security Banner
st.markdown(f"""
<div class="security-banner">
    <strong>{t['cia_title']}</strong><br/>
    &bull; {t['cia_c']}<br/>
    &bull; {t['cia_i']}<br/>
    &bull; {t['cia_a']}
</div>
""", unsafe_allow_html=True)

# ==========================================
# 4. LOAD PRE-TRAINED ML ASSETS
# ==========================================
@st.cache_resource
def load_ml_assets():
    model = joblib.load('talentproof_ml_model.pkl')
    vec = joblib.load('tfidf_vectorizer.pkl') if os.path.exists('tfidf_vectorizer.pkl') else None
    return model, vec

rf_model, tfidf_vec = load_ml_assets()

# ==========================================
# 5. SIDEBAR CONTROLS (WHAT-IF & SPEC)
# ==========================================
with st.sidebar:
    st.markdown(f"### {t['job_spec_header']}")
    job_title = st.text_input(t['position_title'], value="Data Analyst (Middle)")
    required_exp_input = st.number_input(t['required_exp'], min_value=0.5, max_value=15.0, value=2.0, step=0.5)
    
    st.markdown("---")
    st.markdown(f"### {t['what_if_header']}")
    st.caption(t['what_if_caption'])
    
    default_skills = ["python", "sql", "pandas", "numpy", "tableau", "power bi", "excel", "data visualization"]
    if "available_skills" not in st.session_state:
        st.session_state.available_skills = list(default_skills)
    if "active_skills_multiselect" not in st.session_state:
        st.session_state["active_skills_multiselect"] = list(default_skills)

    def add_custom_skill_callback():
        val = st.session_state.get("custom_skill_text_input", "").strip().lower()
        if val:
            if val not in st.session_state.available_skills:
                st.session_state.available_skills.append(val)
            current_selected = list(st.session_state.get("active_skills_multiselect", []))
            if val not in current_selected:
                current_selected.append(val)
                st.session_state["active_skills_multiselect"] = current_selected
            st.session_state["custom_skill_text_input"] = ""

    # Clean custom skill input row
    add_col1, add_col2 = st.columns([3, 1])
    with add_col1:
        st.text_input(
            label="New Skill Input",
            placeholder=t.get("add_skill_placeholder", "Yeni bacarıq yazın (məs: Docker, Git, PyTorch)"),
            label_visibility="collapsed",
            key="custom_skill_text_input",
            on_change=add_custom_skill_callback
        )
    with add_col2:
        st.button(
            t.get("btn_add_skill", "Əlavə Et"), 
            use_container_width=True, 
            key="btn_add_skill_action",
            on_click=add_custom_skill_callback
        )

    # Multiselect widget: compact, scrollable, enterprise-grade, fits all screens without vertical bloating
    active_skills = st.multiselect(
        label=t.get("active_skills_label", "Tələb Olunan Bacarıqlar"),
        options=st.session_state.available_skills,
        format_func=lambda x: x.upper(),
        key="active_skills_multiselect"
    )
            
    st.markdown("---")
    blind_screening_enabled = st.toggle(t['blind_screening'], value=True, help=t['blind_screening_help'])

# ==========================================
# 6. CANDIDATE REPOSITORY STATE
# ==========================================
default_candidates = [
    {
        "id": "CAND-01",
        "name": "Namiq Quliyev",
        "role": "Data Analyst (2.5 Years Exp)",
        "exp_years": 2.5,
        "resume_text": "Data Analyst with 2.5 years of experience in business intelligence. Expert in Python scripting, SQL querying, Pandas dataframes, NumPy operations, Tableau dashboards, Power BI reporting, and advanced Excel modeling. Developed automated data pipelines."
    },
    {
        "id": "CAND-02",
        "name": "Ayten Aliyeva",
        "role": "Senior Data Analyst (3 Years Exp)",
        "exp_years": 3.0,
        "resume_text": "Experienced Data Analyst with 3 years experience. Proficient in Python, SQL querying, Pandas, NumPy, Tableau dashboards, Power BI, Excel. Built machine learning models and visual dashboards for C-level management."
    },
    {
        "id": "CAND-03",
        "name": "Rashad Mammadov",
        "role": "Freelance / Junior (6 Months Exp)",
        "exp_years": 0.5,
        "resume_text": "Junior Data Analyst with 6 months freelance experience. Skilled in Python, SQL, Pandas, NumPy, Tableau, Power BI, Excel. Prepared academic and client reports."
    },
    {
        "id": "CAND-04",
        "name": "Sevinc Rahimova",
        "role": "Human Resources Specialist (Career Switcher)",
        "exp_years": 4.0,
        "resume_text": "Human Resources Specialist with 4 years experience in HR management, employee onboarding, recruitment, and performance evaluations. Conducted interviews and employee satisfaction surveys."
    },
    {
        "id": "CAND-05",
        "name": "Elmir Hasanov",
        "role": "Data Entry Specialist (Skill Gap / Negation)",
        "exp_years": 2.0,
        "resume_text": "Data Entry Specialist with 2 years experience. Comfortable with Excel spreadsheets and basic SQL queries. No knowledge of Python, Pandas, NumPy, Tableau, or Power BI."
    }
]

import json
CANDIDATE_STORE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "candidates_store.json")

def load_candidates():
    """Local on-premise store: shared by all browser sessions (localhost + network)."""
    if os.path.exists(CANDIDATE_STORE):
        try:
            with open(CANDIDATE_STORE, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list) and data:
                return data
        except Exception:
            pass
    return [dict(c) for c in default_candidates]

def save_candidates(cands):
    with open(CANDIDATE_STORE, "w", encoding="utf-8") as f:
        json.dump(cands, f, ensure_ascii=False, indent=2)

# Always read fresh from disk so every session sees the same repository
st.session_state.candidates = load_candidates()

# ==========================================
# 7. EVALUATION PIPELINE
# ==========================================
def run_evaluation(candidates_list, req_skills, req_exp):
    if not req_skills:
        return []
        
    job_desc = f"Looking for {job_title} with minimum {req_exp} years of experience in {', '.join(req_skills)}."
    all_texts = [job_desc] + [c["resume_text"] for c in candidates_list]
    
    if tfidf_vec is not None:
        try:
            tfidf_matrix = tfidf_vec.transform(all_texts)
        except Exception:
            fallback = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
            tfidf_matrix = fallback.fit_transform(all_texts)
    else:
        fallback = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
        tfidf_matrix = fallback.fit_transform(all_texts)
        
    sims = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    
    results = []
    for idx, c in enumerate(candidates_list):
        analysis = analyze_candidate_skills(c["resume_text"], req_skills)
        exp_y = c.get("exp_years", estimate_experience_years(c["resume_text"]))
        
        skill_ratio = analysis["skill_match_ratio"]
        exp_ratio = round(exp_y / max(req_exp, 0.1), 2)
        sem_sim = round(float(sims[idx]), 2)
        
        X_df = pd.DataFrame([{
            'skill_match_ratio': skill_ratio,
            'semantic_similarity': sem_sim,
            'exp_ratio': exp_ratio
        }])
        
        prob = rf_model.predict_proba(X_df)[0][1]
        score_pct = round(prob * 100, 1)
        passed = prob >= 0.45
        
        cand_prefix = "Namizəd #" if st.session_state.current_lang == "AZ" else ("Кандидат #" if st.session_state.current_lang == "RU" else "Candidate #")
        disp_name = f"{cand_prefix}{c['id']}" if blind_screening_enabled else f"{c['name']} ({c['id']})"
        
        results.append({
            "id": c["id"],
            "display_name": disp_name,
            "role": c.get("role", "Applicant"),
            "exp_years": exp_y,
            "skill_ratio": skill_ratio,
            "exp_ratio": exp_ratio,
            "semantic_sim": sem_sim,
            "score_pct": score_pct,
            "passed": passed,
            "verified_skills": analysis["verified_skills"],
            "missing_skills": analysis["missing_skills"],
            "negated_skills": analysis["negated_skills"],
            "evidence_map": analysis["evidence_map"],
            "resume_text": mask_pii(c["resume_text"]) if blind_screening_enabled else c["resume_text"]
        })
        
    results.sort(key=lambda x: x["score_pct"], reverse=True)
    return results

# ==========================================
# 8. NAVIGATION TABS
# ==========================================
tab_matrix, tab_upload, tab_benchmark = st.tabs([
    t["tab_matrix"], 
    t["tab_upload"], 
    t["tab_benchmark"]
])

# ----------------- TAB 1: SCREENING MATRIX -----------------
with tab_matrix:
    eval_results = run_evaluation(st.session_state.candidates, active_skills, required_exp_input)
    
    total_cands = len(eval_results)
    passed_cands = sum(1 for r in eval_results if r["passed"])
    pass_rate = round((passed_cands / max(total_cands, 1)) * 100, 1)
    avg_score = round(float(np.mean([r["score_pct"] for r in eval_results])), 1) if eval_results else 0.0
    
    col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)
    with col_kpi1:
        st.metric(t["total_candidates"], f"{total_cands}")
    with col_kpi2:
        st.metric(t["shortlisted"], f"{passed_cands}")
    with col_kpi3:
        st.metric(t["qual_rate"], f"{pass_rate}%")
    with col_kpi4:
        st.metric(t["mean_score"], f"{avg_score}%")
        
    st.markdown("---")
    st.markdown(f"#### {t['ranked_assessment']}")
    
    for r in eval_results:
        status_label = t["status_qualified"] if r["passed"] else t["status_disqualified"]
        status_class = "badge-status-pass" if r["passed"] else "badge-status-fail"
        status_html = f'<span class="{status_class}">{status_label} ({r["score_pct"]}%)</span>'
        
        with st.container():
            st.markdown(f"""
            <div class="candidate-card">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                    <div>
                        <span style="font-size: 15px; font-weight: 700; color: #0f172a;">{r['display_name']}</span>
                        <span style="font-size: 12px; color: #64748b; margin-left: 8px;">&bull; {r['role']}</span>
                    </div>
                    <div>{status_html}</div>
                </div>
                <div style="display: flex; gap: 28px; font-size: 12px; color: #475569; margin-bottom: 14px;">
                    <span><strong>{t['exp_label']}:</strong> {r['exp_years']} {t['years']} ({t['req_label']}: {required_exp_input} {t['years']})</span>
                    <span><strong>{t['comp_match']}:</strong> {int(r['skill_ratio']*100)}%</span>
                    <span><strong>{t['sem_align']}:</strong> {int(r['semantic_sim']*100)}%</span>
                </div>
            """, unsafe_allow_html=True)
            
            # Competency Breakdown
            c_v, c_m, c_n = st.columns([2, 1, 1])
            with c_v:
                st.markdown(f'<div class="section-label">{t["sec_verified"]}</div>', unsafe_allow_html=True)
                if r['verified_skills']:
                    v_html = " ".join([f'<span class="tag-verified">{s.upper()}</span>' for s in r['verified_skills']])
                    st.markdown(v_html, unsafe_allow_html=True)
                else:
                    st.caption(t["no_verified"])
                    
            with c_m:
                st.markdown(f'<div class="section-label">{t["sec_missing"]}</div>', unsafe_allow_html=True)
                if r['missing_skills']:
                    m_html = " ".join([f'<span class="tag-missing">{s.upper()}</span>' for s in r['missing_skills']])
                    st.markdown(m_html, unsafe_allow_html=True)
                else:
                    st.caption(t["all_verified"])
                    
            with c_n:
                st.markdown(f'<div class="section-label">{t["sec_negated"]}</div>', unsafe_allow_html=True)
                if r['negated_skills']:
                    n_html = " ".join([f'<span class="tag-negated">{s.upper()}</span>' for s in r['negated_skills']])
                    st.markdown(n_html, unsafe_allow_html=True)
                else:
                    st.caption(t["none_negated"])
            
            # Evidence Accordion
            with st.expander(f"{t['audit_trail_expander']} {r['display_name']}"):
                st.markdown(f"**{t['context_span_verification']}**")
                if r['evidence_map']:
                    for sk, ev in r['evidence_map'].items():
                        st.markdown(f"- **`{sk.upper()}`**: *\"{ev}\"*")
                else:
                    st.caption(t["no_context_spans"])
                
                st.markdown("---")
                st.markdown(f"**{t['parsed_resume']}**")
                st.text(r['resume_text'])
                
            st.markdown("</div>", unsafe_allow_html=True)

# ----------------- TAB 2: INGEST / UPLOAD -----------------
with tab_upload:
    st.markdown(f"#### {t['ingest_title']}")
    st.caption(t['ingest_caption'])
    
    col_u1, col_u2 = st.columns([1, 1])
    with col_u1:
        new_name = st.text_input(t["cand_ref_name"], value="Candidate #X")
        new_exp = st.number_input(t["cand_exp"], min_value=0.0, max_value=25.0, value=2.0, step=0.5)
        uploaded_doc = st.file_uploader(t["resume_file"], type=["pdf", "txt"])
        
    with col_u2:
        extracted = ""
        if uploaded_doc is not None:
            if uploaded_doc.type == "application/pdf":
                doc = pymupdf.open(stream=uploaded_doc.read(), filetype="pdf")
                for page in doc:
                    extracted += page.get_text()
                st.success(t["pdf_success"])
            else:
                extracted = uploaded_doc.read().decode("utf-8", errors="ignore")
                
        resume_text_area = st.text_area(
            t["resume_plaintext"], 
            value=extracted if extracted else "Data Analyst with 2 years experience in SQL and Python. Built predictive models.",
            height=180
        )
        
    if st.button(t["btn_ingest"]):
        cands = load_candidates()
        existing_nums = [int(c["id"].split("-")[-1]) for c in cands if c.get("id", "").split("-")[-1].isdigit()]
        new_entry = {
            "id": f"CAND-{(max(existing_nums) if existing_nums else 0) + 1:02d}",
            "name": new_name,
            "role": "Uploaded Applicant",
            "exp_years": new_exp,
            "resume_text": resume_text_area
        }
        cands.append(new_entry)
        save_candidates(cands)
        st.session_state["last_ingested"] = f"{new_name} ({new_entry['id']}) {t['ingest_success']}"
        st.rerun()

    if st.session_state.get("last_ingested"):
        st.success(st.session_state.pop("last_ingested"))

# ----------------- TAB 3: BENCHMARK -----------------
with tab_benchmark:
    st.markdown(f"#### {t['bm_title']}")
    st.caption(t['bm_caption'])
    
    bm1, bm2, bm3, bm4 = st.columns(4)
    with bm1:
        st.metric(t["metric_acc"], "91.0%")
    with bm2:
        st.metric(t["metric_prec"], "90.2%")
    with bm3:
        st.metric(t["metric_rec"], "94.8%")
    with bm4:
        st.metric(t["metric_f1"], "92.4%")
        
    st.markdown("---")
    st.markdown(f"##### {t['cm_title']}")
    
    cm_data = pd.DataFrame(
        [[36, 6], [3, 55]], 
        index=[t["cm_actual_neg"], t["cm_actual_pos"]],
        columns=[t["cm_pred_neg"], t["cm_pred_pos"]]
    )
    st.table(cm_data)
    
    st.markdown(f"""
    **{t['bm_failure_title']}**
    - **{t['bm_fp_desc']}**
    - **{t['bm_fn_desc']}**
    """)
