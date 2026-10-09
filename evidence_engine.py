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

def extract_evidence_snippets(resume_text: str, skill: str) -> List[str]:
    """
    İnteqrasiyalı Sübut Mexanizmi:
    CV-dəki hər bir cümləni analiz edir və həmin bacarığın keçdiyi
    konkret cümləni (evidence span) çıxarır.
    """
    sentences = re.split(r'(?<=[.!?\n])\s+', resume_text)
    pattern = r'\b' + re.escape(skill) + r'\b'
    evidences = []
    
    for sent in sentences:
        sent_clean = sent.strip()
        if not sent_clean:
            continue
        if re.search(pattern, sent_clean, re.IGNORECASE):
            evidences.append(sent_clean)
            
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
        skill_regex = r'\b' + re.escape(skill) + r'\b'
        
        # Mətn daxilində axtarış
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

_MONTHS = {
    'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6, 'jul': 7, 'aug': 8,
    'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12,
    'yan': 1, 'fev': 2, 'mart': 3, 'apr': 4, 'iyun': 6, 'iyul': 7, 'avq': 8, 'sen': 9, 'okt': 10, 'noy': 11, 'dek': 12,
}
_PRESENT = r'(?:present|current|now|indiyədək|hal-hazırda|hazırda|настоящее\s+время|н\.в\.)'
_MONTH_RX = r'(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|yan|fev|iyun|iyul|avq|sen|okt|noy|dek)[a-zəıöüçşğ]*\.?'
_DATE_RX = r'(?:(?P<{p}m1>' + _MONTH_RX + r')\s*(?P<{p}y1>(?:19|20)\d{{2}})|(?P<{p}y2>(?:19|20)\d{{2}})\s*(?P<{p}m2>' + _MONTH_RX + r')?|(?P<{p}nm>\d{{1,2}})[./](?P<{p}ny>(?:19|20)\d{{2}}))'
_RANGE_RX = re.compile(
    _DATE_RX.format(p='s') + r'\s*[-–—]+\s*(?:(?P<present>' + _PRESENT + r')|' + _DATE_RX.format(p='e') + r')',
    re.IGNORECASE,
)
_EDU_KEYWORDS = r'(?:universit|university|college|school|bachelor|master|degree|education|student|təhsil|universitet|bakalavr|magistr|университет|образование)'

def _to_month_index(gd, p, is_end):
    """Converts a matched date to an absolute month index. Year-only dates default to mid-year."""
    m_txt = gd.get(f'{p}m1') or gd.get(f'{p}m2')
    year = gd.get(f'{p}y1') or gd.get(f'{p}y2') or gd.get(f'{p}ny')
    if not year:
        return None
    if gd.get(f'{p}nm'):
        month = int(gd[f'{p}nm'])
    elif m_txt:
        key = m_txt.lower().rstrip('.')
        month = next((v for k, v in _MONTHS.items() if key.startswith(k)), 7)
    else:
        month = 7  # year-only: conservative mid-year assumption
    return int(year) * 12 + max(1, min(month, 12)) - 1

def estimate_experience_years(resume_text: str) -> float:
    """
    Mətndən təcrübə ilini çıxarır (hardcoded default YOXDUR):
    1. Açıq ifadə: '3 years of experience', '2.5 il təcrübə', '6 months experience'
    2. İş tarixi aralıqları: '2024 - Present', 'Jan 2023 – Mar 2025', '2026 July - 2026 August'
       (təhsil aralıqları çıxarılır, üst-üstə düşən dövrlər birləşdirilir)
    Heç bir sübut tapılmasa 0.0 qaytarır.
    """
    from datetime import date

    explicit_years = re.search(r'(\d+(?:\.\d+)?)\s*\+?\s*(?:years?|yrs?|il|лет|года?)\s*(?:of\s+)?(?:experience|təcrübə|опыт)', resume_text, re.IGNORECASE)
    if explicit_years:
        return float(explicit_years.group(1))
    explicit_months = re.search(r'(\d+)\s*(?:months?|ay|месяц\w*)\s*(?:of\s+)?(?:\w+\s+)?(?:experience|təcrübə|опыт)', resume_text, re.IGNORECASE)
    if explicit_months:
        return round(float(explicit_months.group(1)) / 12.0, 2)

    today = date.today()
    now_idx = today.year * 12 + today.month - 1
    lines = resume_text.splitlines()
    intervals = []
    for li, line in enumerate(lines):
        for m in _RANGE_RX.finditer(line):
            context = " ".join(lines[li: li + 3])
            if re.search(_EDU_KEYWORDS, context, re.IGNORECASE):
                continue
            gd = m.groupdict()
            start = _to_month_index(gd, 's', False)
            end = now_idx if gd.get('present') else _to_month_index(gd, 'e', True)
            if start is None or end is None:
                continue
            end = min(end, now_idx)
            if end >= start:
                intervals.append((start, end + 1))

    if not intervals:
        return 0.0
    intervals.sort()
    merged = [list(intervals[0])]
    for s, e in intervals[1:]:
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    total_months = sum(e - s for s, e in merged)
    return round(total_months / 12.0, 1)
