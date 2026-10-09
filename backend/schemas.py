from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any

class CandidateInput(BaseModel):
    id: str
    name: str
    role: Optional[str] = "Applicant"
    exp_years: float = Field(default=1.0, ge=0.0)
    resume_text: str

class ScreeningRequest(BaseModel):
    job_title: str = "Data Analyst"
    required_exp_years: float = 2.0
    required_skills: List[str]
    candidates: List[CandidateInput]
    blind_screening: bool = True

class WhatIfRequest(BaseModel):
    required_exp_years: float
    required_skills: List[str]
    candidates: List[CandidateInput]

class EvidenceItem(BaseModel):
    skill: str
    quote: str

class CandidateResult(BaseModel):
    id: str
    display_name: str
    role: str
    exp_years: float
    skill_match_ratio: float
    semantic_similarity: float
    exp_ratio: float
    suitability_score: float
    status: str  # QUALIFIED / NOT_QUALIFIED
    verified_skills: List[str]
    missing_skills: List[str]
    negated_skills: List[str]
    evidence_breakdown: Dict[str, str]
    masked_resume_text: Optional[str] = None

class ScreeningResponse(BaseModel):
    total_candidates: int
    qualified_count: int
    qualification_rate_pct: float
    average_score_pct: float
    results: List[CandidateResult]
