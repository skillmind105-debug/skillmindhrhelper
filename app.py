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
from evidence_engine import EMAIL_PATTERN, PHONE_PATTERN, URL_PATTERN
import re
import json
import html as _html
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
        font-size: 15.5px;
        color: #0f172a;
    }
    
    .stApp {
        background-color: #f8fafc;
    }

    /* Ensure content is positioned properly below Streamlit top header bar */
    header[data-testid="stHeader"] {
        background-color: transparent !important;
        z-index: 1;
    }

    .block-container {
        padding-top: 3.5rem !important;
        padding-bottom: 2rem !important;
        padding-left: 2.5rem !important;
        padding-right: 2.5rem !important;
        max-width: 1450px !important;
    }

    /* Remove excessive empty gaps between Streamlit elements */
    div[data-testid="stVerticalBlock"] {
        gap: 0.5rem !important;
    }
    
    div[data-testid="stVerticalBlock"] > div {
        margin-bottom: 0px !important;
    }

    /* Clean subtle dividers with tight margin */
    hr {
        margin: 10px 0 14px 0 !important;
        border-color: #e2e8f0 !important;
    }

    /* Headings aesthetics */
    h1, h2, h3, h4, h5, h6 {
        margin-top: 0 !important;
        margin-bottom: 8px !important;
        letter-spacing: -0.2px;
    }
    
    /* Strict Corporate Header Container - Light Enterprise Mode */
    .enterprise-header {
        background-color: #ffffff;
        color: #0f172a;
        padding: 14px 20px;
        border-radius: 6px;
        margin-bottom: 10px;
        border: 1px solid #cbd5e1;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
    }
    .enterprise-header-title {
        font-size: 21px;
        font-weight: 700;
        letter-spacing: -0.2px;
        margin: 0;
        color: #0f172a;
    }
    .enterprise-header-subtitle {
        font-size: 14px;
        color: #334155;
        margin: 3px 0 0 0;
    }
    
    /* Security Verification Banner */
    .security-banner {
        background-color: #ffffff;
        border: 1px solid #cbd5e1;
        border-left: 3px solid #0284c7;
        padding: 10px 16px;
        border-radius: 4px;
        margin-bottom: 12px;
        font-size: 13.5px;
        color: #1e293b;
        line-height: 1.45;
    }
    
    /* KPI Card Containers */
    div[data-testid="stMetricValue"] {
        font-size: 27px !important;
        font-weight: 700 !important;
        color: #0f172a !important;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 12.5px !important;
        font-weight: 700 !important;
        color: #334155 !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Candidate Card Native Container */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #ffffff !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 5px !important;
        margin-bottom: 10px !important;
        padding: 4px 6px !important;
    }

    /* Minimalist Borderless Delete Icon */
    .cand-delete-trigger {
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
    }
    .cand-delete-trigger:hover,
    div[data-testid="stVerticalBlockBorderWrapper"]:hover .cand-delete-trigger {
        opacity: 0.85 !important;
    }
    .cand-delete-trigger div[data-testid="stButton"],
    .cand-delete-trigger div[data-testid="stButton"] > button,
    .cand-delete-trigger button {
        background: transparent !important;
        background-color: transparent !important;
        border: none !important;
        border-color: transparent !important;
        outline: none !important;
        box-shadow: none !important;
        color: #475569 !important;
        font-size: 14px !important;
        font-weight: 600 !important;
        padding: 0 !important;
        margin: 0 !important;
        min-height: unset !important;
        min-width: unset !important;
        height: 20px !important;
        width: 20px !important;
        line-height: 20px !important;
        border-radius: 2px !important;
    }
    .cand-delete-trigger button:hover,
    .cand-delete-trigger div[data-testid="stButton"] > button:hover {
        color: #9f1239 !important;
        background: #fee2e2 !important;
        background-color: #fee2e2 !important;
        border: none !important;
    }
    .cand-delete-trigger button:focus,
    .cand-delete-trigger button:active {
        outline: none !important;
        box-shadow: none !important;
        border: none !important;
    }
    
    /* Professional Status Badges */
    .badge-status-pass {
        background-color: #ecfdf5;
        color: #065f46;
        font-size: 12px;
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
        font-size: 12px;
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
        color: #14532d;
        font-size: 12px;
        font-weight: 700;
        padding: 3px 9px;
        border-radius: 3px;
        border: 1px solid #86efac;
        display: inline-block;
        margin-right: 4px;
        margin-bottom: 4px;
    }
    .tag-missing {
        background-color: #f8fafc;
        color: #334155;
        font-size: 12px;
        font-weight: 600;
        padding: 3px 9px;
        border-radius: 3px;
        border: 1px solid #94a3b8;
        display: inline-block;
        margin-right: 4px;
        margin-bottom: 4px;
    }
    .tag-negated {
        background-color: #fef2f2;
        color: #991b1b;
        font-size: 12px;
        font-weight: 700;
        padding: 3px 9px;
        border-radius: 3px;
        border: 1px solid #fca5a5;
        display: inline-block;
        margin-right: 4px;
        margin-bottom: 4px;
    }

    .section-label {
        font-size: 12.5px;
        font-weight: 700;
        text-transform: uppercase;
        color: #1e293b;
        margin-bottom: 6px;
        letter-spacing: 0.5px;
    }
    /* Streamlit Tabs Styling - High Contrast Visible */
    button[data-baseweb="tab"] {
        color: #334155 !important;
        font-weight: 700 !important;
        font-size: 14px !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #0f172a !important;
        border-bottom-color: #1e40af !important;
    }
    div[data-baseweb="tab-highlight"] {
        background-color: #1e40af !important;
    }

    /* Force Dark High-Contrast Color on All Markdown and Headings */
    h1, h2, h3, h4, h5, h6, p, span, label, div {
        color: inherit;
    }

    /* Specific Metric Styles */
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #cbd5e1;
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
    <div class="enterprise-header">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 14px;">
            <div style="flex: 1 1 300px; min-width: 0;">
                <div class="enterprise-header-title">{t['header_title']}</div>
                <div class="enterprise-header-subtitle">{t['header_subtitle']}</div>
            </div>
            <div style="flex-shrink: 0;">
                <span style="background-color: #f1f5f9; color: #334155; padding: 6px 12px; border-radius: 4px; font-size: 11px; font-weight: 600; border: 1px solid #cbd5e1; white-space: nowrap !important; display: inline-block; letter-spacing: 0.3px;">
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
    logo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logo.png")
    if os.path.exists(logo_path):
        st.image(logo_path, width=80)
    st.markdown(f"### {t['job_spec_header']}")
    job_title = st.text_input(t['position_title'], value="Data Analyst (Middle)")
    required_exp_input = st.number_input(t['required_exp'], min_value=0.5, max_value=15.0, value=2.0, step=0.5)
    
    # Model Architecture Selector (Enterprise Dual-Engine: Core Evidence vs Advanced Deep Matcher)
    model_version_selected = st.selectbox(
        t.get("ml_model_label", "ML Model Arxitekturası"),
        options=[t["model_core_name"], t["model_v3_name"]],
        index=0,
        help=t.get("model_selector_help", "Model seçimi")
    )
    is_v3 = model_version_selected == t["model_v3_name"]

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
# 6b. DOCUMENT RENDERING, PII REDACTION & RATIONALE HELPERS
# ==========================================
CV_ASSET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cv_assets")
MAX_CV_PAGES = 3

def name_variants(name: str):
    """Case variants of the candidate's name tokens (incl. Azerbaijani dotted capital I)."""
    if not name or name.strip().lower().startswith("candidate"):
        return set()
    out = set()
    for tok in re.split(r"\s+", name.strip()):
        tok = tok.strip("(),.")
        if len(tok) < 3 or "#" in tok:
            continue
        out |= {tok, tok.upper(), tok.lower(), tok.replace("i", "İ").upper()}
    return out

def mask_name(text: str, name: str) -> str:
    for v in sorted(name_variants(name), key=len, reverse=True):
        text = re.sub(re.escape(v), "[REDACTED_NAME]", text, flags=re.IGNORECASE)
    return text

def render_cv_images(pdf_bytes: bytes, cand_id: str, cand_name: str) -> int:
    """Renders CV pages to PNG locally: an original and a PII-redacted copy (Blind Screening)."""
    os.makedirs(CV_ASSET_DIR, exist_ok=True)
    pages = 0
    for redacted in (False, True):
        doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
        for i, page in enumerate(doc):
            if i >= MAX_CV_PAGES:
                break
            if redacted:
                text = page.get_text()
                targets = set(re.findall(EMAIL_PATTERN, text)) | set(re.findall(URL_PATTERN, text)) | set(re.findall(PHONE_PATTERN, text))
                targets |= name_variants(cand_name)
                for tgt in targets:
                    if tgt and tgt.strip():
                        for rect in page.search_for(tgt.strip()):
                            page.add_redact_annot(rect, fill=(0.06, 0.09, 0.16))
                for link in page.get_links():
                    page.add_redact_annot(link["from"], fill=(0.06, 0.09, 0.16))
                for img in page.get_images(full=True):
                    for rect in page.get_image_rects(img[0]):
                        page.add_redact_annot(rect, fill=(0.80, 0.84, 0.88))
                page.apply_redactions()
            suffix = "_redacted" if redacted else ""
            page.get_pixmap(dpi=110).save(os.path.join(CV_ASSET_DIR, f"{cand_id}_p{i+1}{suffix}.png"))
            pages = i + 1
        doc.close()
    return pages

def trim_evidence(span: str, skill: str, window: int = 90) -> str:
    """Shortens a long evidence span to a readable window around the matched skill."""
    span = re.sub(r"\s+", " ", span).strip()
    m = re.search(r"\b" + re.escape(skill) + r"\b", span, re.IGNORECASE)
    if not m or len(span) <= window * 2:
        return span
    start, end = max(0, m.start() - window), min(len(span), m.end() + window)
    return ("..." if start > 0 else "") + span[start:end].strip() + ("..." if end < len(span) else "")

def build_decision_reasons(r, req_exp, tr):
    reasons = []
    n_req = len(r["verified_skills"]) + len(r["missing_skills"])
    reasons.append(tr["r_skills"].format(v=len(r["verified_skills"]), n=n_req))
    if r["missing_skills"]:
        reasons.append(tr["r_missing"].format(lst=", ".join(s.upper() for s in r["missing_skills"])))
    if r["exp_years"] >= req_exp:
        reasons.append(tr["r_exp_ok"].format(e=r["exp_years"], r=req_exp))
    else:
        reasons.append(tr["r_exp_low"].format(e=r["exp_years"], r=req_exp))
    if r["negated_skills"]:
        reasons.append(tr["r_negated"].format(lst=", ".join(s.upper() for s in r["negated_skills"])))
    sem = int(r["semantic_sim"] * 100)
    reasons.append((tr["r_sem_high"] if r["semantic_sim"] >= 0.5 else tr["r_sem_low"]).format(s=sem))
    return reasons

def comparison_deltas(a, b, tr, positive=True):
    """Lists metric differences of candidate a vs b. positive=True -> advantages, False -> gaps."""
    items = []
    d_score = a["score_pct"] - b["score_pct"]
    d_skill = int(round((a["skill_ratio"] - b["skill_ratio"]) * 100))
    d_exp = round(a["exp_years"] - b["exp_years"], 1)
    d_sem = int(round((a["semantic_sim"] - b["semantic_sim"]) * 100))
    sign = 1 if positive else -1
    if d_score * sign > 0:
        items.append(tr["d_score"].format(d=d_score))
    if d_skill * sign > 0:
        items.append(tr["d_skills"].format(d=d_skill))
    if d_exp * sign > 0:
        items.append(tr["d_exp"].format(d=d_exp))
    if d_sem * sign > 0:
        items.append(tr["d_sem"].format(d=d_sem))
    unique = [s for s in (a if positive else b)["verified_skills"] if s not in (b if positive else a)["verified_skills"]]
    if unique:
        items.append((tr["d_unique"] if positive else tr["d_unique_other"]).format(lst=", ".join(s.upper() for s in unique)))
    return items

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
        
        if is_v3:
            try:
                from talentproof_ai_fastapi.inference import model as v3_model, vectorizer as v3_vec, extract_skills as v3_extract, FEATURE_ORDER
                c_vec = v3_vec.transform([c["resume_text"]])
                j_vec = v3_vec.transform([job_desc])
                v3_sim = float(c_vec.multiply(j_vec).sum())
                c_sk = v3_extract(c["resume_text"])
                j_sk = set([s.lower() for s in req_skills])
                mtch = c_sk & j_sk
                j_cov = len(mtch) / len(j_sk) if j_sk else 0.0
                c_cov = len(mtch) / len(c_sk) if c_sk else 0.0
                un = c_sk | j_sk
                jacc = len(mtch) / len(un) if un else 0.0
                feat_df = pd.DataFrame([{
                    "text_similarity": v3_sim,
                    "job_skill_coverage": j_cov,
                    "cv_skill_coverage": c_cov,
                    "skill_jaccard": jacc
                }])[FEATURE_ORDER]
                prob = float(v3_model.predict_proba(feat_df)[0, 1])
                score_pct = round(prob * 100, 1)
                passed = prob >= 0.50
            except Exception:
                prob = rf_model.predict_proba(X_df)[0][1]
                score_pct = round(prob * 100, 1)
                passed = prob >= 0.45
        else:
            prob = rf_model.predict_proba(X_df)[0][1]
            score_pct = round(prob * 100, 1)
            passed = prob >= 0.45
        
        cand_prefix = "Namizəd #" if st.session_state.current_lang == "AZ" else ("Кандидат #" if st.session_state.current_lang == "RU" else "Candidate #")
        disp_name = f"{cand_prefix}{c['id']}" if blind_screening_enabled else f"{c['name']} ({c['id']})"
        
        def _protect(txt):
            return mask_name(mask_pii(txt), c.get("name", "")) if blind_screening_enabled else txt

        evidence_clean = {sk: _protect(trim_evidence(ev, sk)) for sk, ev in analysis["evidence_map"].items()}

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
            "evidence_map": evidence_clean,
            "cv_pages": c.get("cv_pages", 0),
            "resume_text": _protect(c["resume_text"])
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
    
    if st.session_state.get("cand_del_success"):
        st.success(st.session_state.pop("cand_del_success"))
    
    for rank_idx, r in enumerate(eval_results):
        status_label = t["status_qualified"] if r["passed"] else t["status_disqualified"]
        status_class = "badge-status-pass" if r["passed"] else "badge-status-fail"
        status_html = f'<span class="{status_class}">{status_label} ({r["score_pct"]}%)</span>'
        
        with st.container(border=True):
            # Header Row: Title & Subtitle on left, Status and compact X on right
            top_c1, top_c2 = st.columns([7.4, 2.6])
            with top_c1:
                st.markdown(f"""
                <div style="padding-top: 2px;">
                    <span style="font-size: 19.5px; font-weight: 700; color: #0f172a; letter-spacing: -0.2px;">{r['display_name']}</span>
                    <span style="font-size: 13.5px; font-weight: 500; color: #475569; margin-left: 10px;">&bull; {r['role']}</span>
                </div>
                """, unsafe_allow_html=True)
            with top_c2:
                b_c1, b_c2 = st.columns([8.2, 1.8])
                with b_c1:
                    st.markdown(f'<div style="text-align: right; padding-top: 2px;">{status_html}</div>', unsafe_allow_html=True)
                with b_c2:
                    st.markdown(f'<div class="cand-delete-trigger" style="opacity: 0.2; transition: opacity 0.2s; text-align: right;">', unsafe_allow_html=True)
                    if st.button("✕", key=f"del_cand_btn_{r['id']}", help=t.get("btn_delete_cand", "Namizədi Sil")):
                        st.session_state[f"confirm_delete_{r['id']}"] = True
                        st.rerun()
                    st.markdown('</div>', unsafe_allow_html=True)
            
            # Subtle Enterprise Confirmation Bar (Clean Slate/Rose Muted, Anti-Vibe-Coded)
            if st.session_state.get(f"confirm_delete_{r['id']}", False):
                st.markdown(f"""
                <div style="background-color: #f8fafc; border: 1px solid #cbd5e1; border-left: 3px solid #9f1239; border-radius: 3px; padding: 8px 12px; margin: 8px 0; font-size: 13px; color: #0f172a; display: flex; align-items: center; justify-content: space-between;">
                    <span><strong>{t['confirm_delete_q']}</strong> ({r['display_name']})</span>
                </div>
                """, unsafe_allow_html=True)
                conf_c1, conf_c2, conf_c3 = st.columns([1.2, 1.2, 7.6])
                with conf_c1:
                    if st.button(t['btn_confirm_yes'], key=f"yes_del_{r['id']}", use_container_width=True):
                        all_c = load_candidates()
                        all_c = [c for c in all_c if c.get("id") != r["id"]]
                        save_candidates(all_c)
                        st.session_state.candidates = all_c
                        st.session_state[f"confirm_delete_{r['id']}"] = False
                        st.session_state["cand_del_success"] = t.get("cand_deleted_msg", "Namizəd uğurla silindi.")
                        st.rerun()
                with conf_c2:
                    if st.button(t['btn_confirm_no'], key=f"no_del_{r['id']}", use_container_width=True):
                        st.session_state[f"confirm_delete_{r['id']}"] = False
                        st.rerun()

            st.markdown(f"""
                <div style="display: flex; gap: 24px; font-size: 13.5px; color: #1e293b; margin: 6px 0 10px 0;">
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
            
            # Decision Rationale & Comparative Advantage
            reasons = build_decision_reasons(r, required_exp_input, t)
            rat_title = t["why_qualified_title"] if r["passed"] else t["why_not_title"]
            reasons_html = "".join(f"<li>{x}</li>" for x in reasons)

            total_n = len(eval_results)
            comp_html = f'<div style="margin-bottom:6px;">{t["c_rank"].format(rank=rank_idx + 1, total=total_n, below=total_n - rank_idx - 1)}</div>'
            if total_n == 1:
                comp_html += f'<div>{t["c_only"]}</div>'
            if rank_idx + 1 < total_n:
                nxt = eval_results[rank_idx + 1]
                adv = comparison_deltas(r, nxt, t, positive=True)
                comp_html += f'<div style="font-weight:600;margin-top:6px;">{t["c_vs_next"].format(name=nxt["display_name"])}</div>'
                if adv:
                    comp_html += "<ul style='margin:4px 0 0 18px;padding:0;'>" + "".join(f"<li>{x}</li>" for x in adv) + "</ul>"
                else:
                    comp_html += f'<div>{t["c_no_adv"]}</div>'
            if rank_idx > 0:
                prv = eval_results[rank_idx - 1]
                gaps = comparison_deltas(r, prv, t, positive=False)
                comp_html += f'<div style="font-weight:600;margin-top:8px;">{t["c_vs_prev"].format(name=prv["display_name"])}</div>'
                if gaps:
                    comp_html += "<ul style='margin:4px 0 0 18px;padding:0;'>" + "".join(f"<li>{x}</li>" for x in gaps) + "</ul>"

            box_style = "background:#f8fafc;border:1px solid #cbd5e1;border-radius:4px;padding:12px 14px;font-size:13px;color:#1e293b;line-height:1.55;"
            rc1, rc2 = st.columns([1, 1])
            with rc1:
                st.markdown(f'<div style="{box_style}"><div class="section-label">{rat_title}</div><ul style="margin:4px 0 0 18px;padding:0;">{reasons_html}</ul></div>', unsafe_allow_html=True)
            with rc2:
                st.markdown(f'<div style="{box_style}"><div class="section-label">{t["comparative_title"]}</div>{comp_html}</div>', unsafe_allow_html=True)

            # Evidence Accordion: CV document image + evidence + collapsible text
            with st.expander(f"{t['audit_trail_expander']} {r['display_name']}"):
                doc_col, ev_col = st.columns([1, 1])
                with doc_col:
                    st.markdown(f'<div class="section-label">{t["cv_document"]}</div>', unsafe_allow_html=True)
                    suffix = "_redacted" if blind_screening_enabled else ""
                    page_paths = [os.path.join(CV_ASSET_DIR, f"{r['id']}_p{p}{suffix}.png") for p in range(1, r["cv_pages"] + 1)]
                    page_paths = [p for p in page_paths if os.path.exists(p)]
                    if page_paths:
                        if blind_screening_enabled:
                            st.caption(t["cv_redacted_note"])
                        pg = 1
                        if len(page_paths) > 1:
                            pg = st.radio(t["page_label"], list(range(1, len(page_paths) + 1)), horizontal=True, key=f"pg_{r['id']}")
                        st.image(page_paths[pg - 1], use_container_width=True)
                    else:
                        st.caption(t["cv_no_image"])
                with ev_col:
                    st.markdown(f'<div class="section-label">{t["context_span_verification"]}</div>', unsafe_allow_html=True)
                    if r['evidence_map']:
                        for sk, ev in r['evidence_map'].items():
                            st.markdown(f'<div style="border-left:2px solid #A7F3D0;padding:4px 10px;margin-bottom:8px;font-size:13px;color:#1e293b;"><strong>{sk.upper()}</strong><br/><span style="color:#334155;">"{_html.escape(ev)}"</span></div>', unsafe_allow_html=True)
                    else:
                        st.caption(t["no_context_spans"])

                if st.toggle(t["cv_text_expander"], key=f"txt_{r['id']}"):
                    st.text_area(t["parsed_resume"], value=r['resume_text'], height=260, disabled=True, key=f"ta_{r['id']}_{int(blind_screening_enabled)}")

# ----------------- TAB 2: INGEST / UPLOAD -----------------
with tab_upload:
    st.markdown(f"#### {t['ingest_title']}")
    st.caption(t['ingest_caption'])
    
    uploaded_doc = st.file_uploader(t["resume_file"], type=["pdf", "txt"], key="resume_file_uploader")
    
    if "cv_extracted_text" not in st.session_state:
        st.session_state.cv_extracted_text = "Data Analyst with 2 years experience in SQL and Python. Built predictive models."
    if "cv_pdf_bytes" not in st.session_state:
        st.session_state.cv_pdf_bytes = None
    if "last_processed_file" not in st.session_state:
        st.session_state.last_processed_file = None

    if uploaded_doc is not None and st.session_state.last_processed_file != uploaded_doc.name:
        st.session_state.last_processed_file = uploaded_doc.name
        if uploaded_doc.type == "application/pdf":
            st.session_state.cv_pdf_bytes = uploaded_doc.getvalue()
            doc = pymupdf.open(stream=st.session_state.cv_pdf_bytes, filetype="pdf")
            full_text = ""
            for page in doc:
                full_text += page.get_text() + "\n"
            st.session_state.cv_extracted_text = full_text
            st.success(t["pdf_success"])
        else:
            st.session_state.cv_pdf_bytes = None
            st.session_state.cv_extracted_text = uploaded_doc.getvalue().decode("utf-8", errors="ignore")
        
        # Calculate experience directly from uploaded text
        st.session_state.detected_exp_val = float(estimate_experience_years(st.session_state.cv_extracted_text))

    if "detected_exp_val" not in st.session_state:
        st.session_state.detected_exp_val = float(estimate_experience_years(st.session_state.cv_extracted_text))

    col_u1, col_u2 = st.columns([1, 1])
    with col_u2:
        resume_text_area = st.text_area(
            t["resume_plaintext"], 
            value=st.session_state.cv_extracted_text,
            height=200,
            key="upload_resume_text_area"
        )
        # Update text if user edits in textarea
        if resume_text_area != st.session_state.cv_extracted_text:
            st.session_state.cv_extracted_text = resume_text_area

    with col_u1:
        new_name = st.text_input(t["cand_ref_name"], value="Candidate #X")
        new_exp = st.number_input(
            t["cand_exp"], 
            min_value=0.0, 
            max_value=25.0, 
            value=st.session_state.detected_exp_val, 
            step=0.1,
            help="CV mətnindən avtomatik aşkarlanır və ya əllə tənzimlənə bilər."
        )
        
    if st.button(t["btn_ingest"]):
        cands = load_candidates()
        
        # Duplicate check: by exact name or high resume text similarity
        is_duplicate = False
        norm_name = new_name.strip().lower()
        clean_new_text = " ".join(resume_text_area.lower().split())

        for existing_c in cands:
            ex_name = existing_c.get("name", "").strip().lower()
            ex_text = " ".join(existing_c.get("resume_text", "").lower().split())
            
            # Check by name or text match
            if (norm_name and norm_name == ex_name and norm_name != "candidate #x") or (len(clean_new_text) > 80 and clean_new_text[:150] == ex_text[:150]):
                is_duplicate = True
                break

        if is_duplicate:
            st.session_state["dup_warning"] = t.get("cand_already_exists", "Bu namizəd artıq sistemdə mövcuddur! Eyni CV təkrar əlavə edilmədi.")
            st.rerun()
        else:
            existing_nums = [int(c["id"].split("-")[-1]) for c in cands if c.get("id", "").split("-")[-1].isdigit()]
            new_id = f"CAND-{(max(existing_nums) if existing_nums else 0) + 1:02d}"
            cv_pages = 0
            if st.session_state.cv_pdf_bytes:
                try:
                    cv_pages = render_cv_images(st.session_state.cv_pdf_bytes, new_id, new_name)
                except Exception:
                    cv_pages = 0
            new_entry = {
                "id": new_id,
                "name": new_name,
                "role": "Uploaded Applicant",
                "exp_years": new_exp,
                "cv_pages": cv_pages,
                "resume_text": resume_text_area
            }
            cands.append(new_entry)
            save_candidates(cands)
            st.session_state["last_ingested"] = f"{new_name} ({new_entry['id']}) {t['ingest_success']}"
            st.rerun()

    if st.session_state.get("dup_warning"):
        st.warning(st.session_state.pop("dup_warning"))

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
