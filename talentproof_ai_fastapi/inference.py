
from pathlib import Path
import re
import joblib
import pandas as pd

BASE = Path(__file__).resolve().parent

model = joblib.load(BASE / "match_model.joblib")
vectorizer = joblib.load(BASE / "tfidf_vectorizer.joblib")
skill_patterns = joblib.load(BASE / "skill_patterns.joblib")

FEATURE_ORDER = [
    "text_similarity",
    "job_skill_coverage",
    "cv_skill_coverage",
    "skill_jaccard"
]

def extract_skills(text):
    text = str(text).lower()
    return {
        skill for skill, pattern in skill_patterns.items()
        if re.search(pattern, text)
    }

def predict_cv_job_match(cv_text: str, job_text: str):
    if not cv_text or not cv_text.strip():
        raise ValueError("CV text cannot be empty")
    if not job_text or not job_text.strip():
        raise ValueError("Job description cannot be empty")

    cv_vec = vectorizer.transform([cv_text])
    job_vec = vectorizer.transform([job_text])
    similarity = float(cv_vec.multiply(job_vec).sum())

    cv_skills = extract_skills(cv_text)
    job_skills = extract_skills(job_text)
    matched = cv_skills & job_skills
    missing = job_skills - cv_skills

    job_coverage = len(matched) / len(job_skills) if job_skills else 0.0
    cv_coverage = len(matched) / len(cv_skills) if cv_skills else 0.0
    union = cv_skills | job_skills
    jaccard = len(matched) / len(union) if union else 0.0

    features = pd.DataFrame([{
        "text_similarity": similarity,
        "job_skill_coverage": job_coverage,
        "cv_skill_coverage": cv_coverage,
        "skill_jaccard": jaccard
    }])[FEATURE_ORDER]

    score = float(model.predict_proba(features)[0, 1])

    if score >= 0.70 and job_coverage >= 0.70:
        decision = "NÖVBƏTİ MƏRHƏLƏYƏ KEÇİRİLSİN"
    elif score < 0.45 and job_coverage < 0.30:
        decision = "HAZIRDA UYĞUN GÖRÜNMÜR"
    else:
        decision = "ƏLAVƏ İNSAN YOXLAMASI"

    return {
        "model_version": "TalentProof AI V3",
        "match_score": round(score * 100, 2),
        "decision": decision,
        "text_similarity_pct": round(similarity * 100, 2),
        "job_skill_coverage_pct": round(job_coverage * 100, 1),
        "cv_skill_coverage_pct": round(cv_coverage * 100, 1),
        "matched_skills": sorted(matched),
        "missing_job_skills": sorted(missing),
        "warning": "Prototype score; not a validated hiring probability."
    }
