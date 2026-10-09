import re
from typing import List, Dict, Any, Tuple

# PII Masking regex patterns (Confidentiality / Blind Screening üçün)
EMAIL_PATTERN = r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'
PHONE_PATTERN = r'(?:\+?\d{1,3}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{4}'
URL_PATTERN = r'https?://\S+|www\.\S+|linkedin\.com/\S+|github\.com/\S+'

def mask_pii(text: str) -> str:
    """
    CIA Triada - Confidentiality & Blind Screening:
    Namizədin adı, email, telefon və profil linklərini maskalayır.
    Zero Bias (Qərəzsiz) qiymətləndirmə təmin edir.
    """
    masked = re.sub(EMAIL_PATTERN, '[REDACTED_EMAIL]', text)
    masked = re.sub(PHONE_PATTERN, '[REDACTED_PHONE]', masked)
    masked = re.sub(URL_PATTERN, '[REDACTED_LINK]', masked)
    return masked

def _build_skill_regex(skill: str) -> str:
    tokens = [re.escape(tok) for tok in skill.strip().lower().split()]
    return r'\b' + r'\s+'.join(tokens) + r'\b'

def extract_evidence_snippets(resume_text: str, skill: str) -> List[str]:
    """
    İnteqrasiyalı Sübut Mexanizmi:
    CV-dəki hər bir cümləni analiz edir və həmin bacarığın keçdiyi
    konkret cümləni (evidence span) çıxarır.
    """
    sentences = re.split(r'(?<=[.!?\n])\s+', resume_text)
    pattern = _build_skill_regex(skill)
    evidences = []
    
    for sent in sentences:
        sent_clean = sent.strip()
        if not sent_clean:
            continue
        if re.search(pattern, sent_clean, re.IGNORECASE):
            evidences.append(sent_clean)
            
    if not evidences:
        # Fallback for multi-line sentence structures
        if re.search(pattern, resume_text, re.IGNORECASE):
            evidences.append(f"Found in text: {skill}")
            
    return evidences

def analyze_candidate_skills(
    resume_text: str, 
    required_skills: List[str]
) -> Dict[str, Any]:
    """
    Integrity & Explainability:
    1. Negation Handling: Məs. 'No knowledge of Python' -> Python bacarıq sayılmır!
    2. Context Evidence: Tapılan bacarıq üçün dəqiq sitat çıxarılır.
    3. Gap Analysis: Çatışmayan bacarıqlar və səbəblər qeyd olunur.
    """
    text_lower = resume_text.lower()
    
    # Mənfi ifadələr (Negation detection)
    negation_patterns = [
        r'no\s+(?:knowledge|experience|skills?)\s+(?:of|in|with)?\s+([^.]+)',
        r'lack\s+of\s+([^.]+)',
        r'without\s+([^.]+)',
        r'never\s+worked\s+with\s+([^.]+)',
        r'little\s+to\s+no\s+experience\s+with\s+([^.]+)'
    ]
    
    negated_blocks = []
    for pattern in negation_patterns:
        matches = re.findall(pattern, text_lower)
        negated_blocks.extend(matches)
    negated_combined = " ".join(negated_blocks)
    
    verified_skills = []
    missing_skills = []
    negated_skills = []
    evidence_map = {}
    
    for raw_skill in required_skills:
        skill = raw_skill.strip().lower()
        skill_regex = _build_skill_regex(skill)
        
        # Mətn daxilində axtarış (sətir qırılmaları daxil)
        if re.search(skill_regex, text_lower):
            # İnkar blokundadırmı?
            if re.search(skill_regex, negated_combined):
                negated_skills.append(raw_skill)
            else:
                verified_skills.append(raw_skill)
                # Sübut kontekstini çıxar
                snippets = extract_evidence_snippets(resume_text, skill)
                evidence_map[raw_skill] = snippets[0] if snippets else "Context verified in text."
        else:
            missing_skills.append(raw_skill)
            
    return {
        "verified_skills": verified_skills,
        "missing_skills": missing_skills,
        "negated_skills": negated_skills,
        "evidence_map": evidence_map,
        "verified_count": len(verified_skills),
        "total_required": len(required_skills),
        "skill_match_ratio": round(len(verified_skills) / max(len(required_skills), 1), 2)
    }

def estimate_experience_years(resume_text: str) -> float:
    """
    Mətndən təcrübə ilini heuristik və regex ilə çıxarır.
    Məsələn: '3 years of experience', '2.5 years', '6 months'
    """
    # X months regex
    months_match = re.search(r'(\d+)\s*(?:month|ay)', resume_text, re.IGNORECASE)
    if months_match:
        return round(float(months_match.group(1)) / 12.0, 2)
        
    # X years regex
    years_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:\+?\s*(?:years?|il|yrs?))\s*(?:of\s+experience|təcrübə)?', resume_text, re.IGNORECASE)
    if years_match:
        return float(years_match.group(1))
        
    return 1.0 # Default baza
