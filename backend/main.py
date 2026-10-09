from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
import sys
import joblib
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pymupdf

# Layihə kök qovluğunu yola əlavə edirik
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evidence_engine import analyze_candidate_skills, estimate_experience_years, mask_pii
from backend.schemas import ScreeningRequest, ScreeningResponse, CandidateResult, WhatIfRequest

app = FastAPI(
    title="TalentProof AI — Core Screening Engine",
    description="Enterprise On-Premise ML & Evidence Verification REST API",
    version="1.0.0"
)

# Frontend komandası üçün CORS icazələri
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model və Vektorizerin yüklənməsi
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(ROOT_DIR, "talentproof_ml_model.pkl")
VEC_PATH = os.path.join(ROOT_DIR, "tfidf_vectorizer.pkl")

ml_model = joblib.load(MODEL_PATH)
tfidf_vec = joblib.load(VEC_PATH) if os.path.exists(VEC_PATH) else None

@app.get("/")
def health_check():
    return {
        "status": "OPERATIONAL",
        "service": "TalentProof AI Core API",
        "model_loaded": ml_model is not None,
        "vectorizer_loaded": tfidf_vec is not None,
        "cia_triad_compliance": True
    }

def run_screening_logic(
    job_title: str,
    required_exp: float,
    required_skills: list,
    candidates: list,
    blind_screening: bool = True
) -> ScreeningResponse:
    if not required_skills:
        raise HTTPException(status_code=400, detail="Required skills list cannot be empty.")

    job_description = f"Looking for {job_title} with minimum {required_exp} years of experience in {', '.join(required_skills)}."
    all_texts = [job_description] + [c.resume_text for c in candidates]

    # TF-IDF Cosine Similarity
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
    for idx, cand in enumerate(candidates):
        analysis = analyze_candidate_skills(cand.resume_text, required_skills)
        exp_y = cand.exp_years if cand.exp_years > 0 else estimate_experience_years(cand.resume_text)

        skill_ratio = analysis["skill_match_ratio"]
        exp_ratio = round(exp_y / max(required_exp, 0.1), 2)
        sem_sim = round(float(sims[idx]), 2)

        # ML Feature Matrix
        X_feat = pd.DataFrame([{
            'skill_match_ratio': skill_ratio,
            'semantic_similarity': sem_sim,
            'exp_ratio': exp_ratio
        }])

        prob = ml_model.predict_proba(X_feat)[0][1]
        score_pct = round(prob * 100, 1)
        is_qualified = prob >= 0.45

        disp_name = f"Candidate #{cand.id}" if blind_screening else f"{cand.name} ({cand.id})"

        results.append(CandidateResult(
            id=cand.id,
            display_name=disp_name,
            role=cand.role or "Applicant",
            exp_years=exp_y,
            skill_match_ratio=skill_ratio,
            semantic_similarity=sem_sim,
            exp_ratio=exp_ratio,
            suitability_score=score_pct,
            status="QUALIFIED" if is_qualified else "NOT_QUALIFIED",
            verified_skills=analysis["verified_skills"],
            missing_skills=analysis["missing_skills"],
            negated_skills=analysis["negated_skills"],
            evidence_breakdown=analysis["evidence_map"],
            masked_resume_text=mask_pii(cand.resume_text) if blind_screening else cand.resume_text
        ))

    # Yüksək baldan aşağıya sıralanma
    results.sort(key=lambda x: x.suitability_score, reverse=True)

    qualified_count = sum(1 for r in results if r.status == "QUALIFIED")
    total_count = len(results)

    return ScreeningResponse(
        total_candidates=total_count,
        qualified_count=qualified_count,
        qualification_rate_pct=round((qualified_count / max(total_count, 1)) * 100, 1),
        average_score_pct=round(float(np.mean([r.suitability_score for r in results])), 1) if results else 0.0,
        results=results
    )

@app.post("/api/screen", response_model=ScreeningResponse)
def screen_candidates(req: ScreeningRequest):
    return run_screening_logic(
        job_title=req.job_title,
        required_exp=req.required_exp_years,
        required_skills=req.required_skills,
        candidates=req.candidates,
        blind_screening=req.blind_screening
    )

@app.post("/api/what-if", response_model=ScreeningResponse)
def simulate_what_if(req: WhatIfRequest):
    return run_screening_logic(
        job_title="Simulation Job",
        required_exp=req.required_exp_years,
        required_skills=req.required_skills,
        candidates=req.candidates,
        blind_screening=True
    )

@app.post("/api/parse-cv")
async def parse_cv_file(file: UploadFile = File(...)):
    """PDF və ya TXT faylından mətni çıxarır və PII maskalayır"""
    content = await file.read()
    extracted_text = ""
    if file.filename.lower().endswith(".pdf"):
        doc = pymupdf.open(stream=content, filetype="pdf")
        for page in doc:
            extracted_text += page.get_text()
    else:
        extracted_text = content.decode("utf-8", errors="ignore")

    return {
        "filename": file.filename,
        "extracted_text": extracted_text,
        "estimated_experience_years": estimate_experience_years(extracted_text),
        "masked_preview": mask_pii(extracted_text[:400])
    }
