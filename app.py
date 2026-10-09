# Streamlit Enterprise Dashboard (Slate Theme Updated)
# pyrefly: ignore [missing-import]
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
        background-color: #f1f5f9;
        color: #0f172a;
        padding: 16px 24px;
        border-radius: 4px;
        margin-bottom: 16px;
        border: 1px solid #cbd5e1;
    }
    .enterprise-header-title {
        font-size: 19px;
        font-weight: 700;
        letter-spacing: -0.2px;
        margin: 0;
        color: #0f172a;
    }
    .enterprise-header-subtitle {
        font-size: 13px;
        color: #475569;
        margin: 4px 0 0 0;
    }
    
    /* Security Verification Banner */
    .security-banner {
        background-color: #ffffff;
        border: 1px solid #e2e8f0 !important;
        border-left: none !important;
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
        border-bottom-color: #cbd5e1 !important;
    }
    div[data-baseweb="tab-highlight"] {
        background-color: #cbd5e1 !important;
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

    /* Skill row with hover-reveal transparent delete 'x' */
    div[data-testid="stHorizontalBlock"]:has(.delete-skill-btn) .delete-skill-btn button {
        opacity: 0 !important;
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        color: #94a3b8 !important;
        padding: 0 !important;
        font-size: 13px !important;
        font-weight: 700 !important;
        line-height: 1 !important;
        min-height: 24px !important;
        height: 24px !important;
        transition: opacity 0.15s ease, color 0.15s ease !important;
    }
    div[data-testid="stHorizontalBlock"]:has(.delete-skill-btn):hover .delete-skill-btn button {
        opacity: 0.8 !important;
    }
    div[data-testid="stHorizontalBlock"]:has(.delete-skill-btn) .delete-skill-btn button:hover {
        opacity: 1 !important;
        color: #e11d48 !important;
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
    <div class="enterprise-header" style="background-color: #f1f5f9 !important; border: 1px solid #cbd5e1 !important;">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 14px;">
            <div style="flex: 1 1 300px; min-width: 0;">
                <div class="enterprise-header-title" style="color: #0f172a !important;">{t['header_title']}</div>
                <div class="enterprise-header-subtitle" style="color: #475569 !important;">{t['header_subtitle']}</div>
            </div>
            <div style="flex-shrink: 0;">
                <span style="background-color: #ffffff; color: #334155; padding: 6px 12px; border-radius: 3px; font-size: 11px; font-weight: 600; border: 1px solid #cbd5e1; white-space: nowrap !important; display: inline-block; letter-spacing: 0.3px;">
                    {t['local_badge']}
                </span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Security Banner
st.markdown(f"""
<div class="security-banner" style="border: 1px solid #e2e8f0 !important; border-left: none !important;">
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
    def add_custom_skill_callback():
        val = st.session_state.get("custom_skill_text_input", "").strip().lower()
        if val:
            if val not in st.session_state.available_skills:
                st.session_state.available_skills.append(val)
            if "active_skills_set" not in st.session_state:
                st.session_state.active_skills_set = set(default_skills)
            st.session_state.active_skills_set.add(val)
            st.session_state["custom_skill_text_input"] = ""

    if "active_skills_set" not in st.session_state:
        st.session_state.active_skills_set = set(default_skills)

    # Clean input row with full width
    st.markdown(f'<div style="font-size:12px; font-weight:600; color:#475569; margin-bottom:4px;">{t.get("add_skill_label", "YENİ BACARIQ ƏLAVƏ ET")}</div>', unsafe_allow_html=True)
    add_col1, add_col2 = st.columns([3, 1])
    with add_col1:
        st.text_input(
            label="New Skill Input",
            placeholder=t.get("add_skill_placeholder", "Məs: Docker, Git, PyTorch..."),
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

    def remove_skill_callback(skill_to_remove):
        if skill_to_remove in st.session_state.available_skills:
            st.session_state.available_skills.remove(skill_to_remove)
        if "active_skills_set" in st.session_state:
            st.session_state.active_skills_set.discard(skill_to_remove)

    # Clean Pill-based Skill Selector (Zero confusing search box, click to toggle on/off)
    st.markdown(f'<div style="font-size:12px; font-weight:600; color:#475569; margin-top:12px; margin-bottom:6px;">{t.get("active_skills_label", "TƏLƏB OLUNAN BACARIQLAR")}</div>', unsafe_allow_html=True)
    
    # Render interactive skill toggles grid (2 columns) with hover-revealed delete 'x'
    skill_cols = st.columns(2)
    for idx, sk in enumerate(list(st.session_state.available_skills)):
        col_target = skill_cols[idx % 2]
        is_checked = sk in st.session_state.active_skills_set
        with col_target:
            row_c1, row_c2 = st.columns([5, 1])
            with row_c1:
                if st.checkbox(sk.upper(), value=is_checked, key=f"pill_skill_{sk}"):
                    st.session_state.active_skills_set.add(sk)
                else:
                    st.session_state.active_skills_set.discard(sk)
            with row_c2:
                st.markdown('<div class="delete-skill-btn">', unsafe_allow_html=True)
                st.button("✕", key=f"del_sk_{sk}", on_click=remove_skill_callback, args=(sk,), help=f"{sk.upper()} bacarığını sil")
                st.markdown('</div>', unsafe_allow_html=True)

    active_skills = [s for s in st.session_state.available_skills if s in st.session_state.active_skills_set]
            
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

if "candidates" not in st.session_state:
    st.session_state.candidates = default_candidates

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
        new_entry = {
            "id": f"CAND-{len(st.session_state.candidates)+1:02d}",
            "name": new_name,
            "role": "Uploaded Applicant",
            "exp_years": new_exp,
            "resume_text": resume_text_area
        }
        st.session_state.candidates.append(new_entry)
        st.success(f"{new_name} {t['ingest_success']}")
        st.rerun()

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
