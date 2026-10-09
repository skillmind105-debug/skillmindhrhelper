# TALENTPROOF AI — ARCHITECTURAL GOVERNANCE & AI AGENT CONSTRAINTS
> **CRITICAL DIRECTIVE FOR ANY AI ASSISTANT WORKING ON THIS REPOSITORY:**
> You must strictly respect the existing architecture, security contracts, and non-negotiable guidelines defined below. DO NOT refactor or delete the Core ML, CIA Triad rules, or Enterprise design decisions.

---

## 1. Project Overview & Core Mission
**TalentProof AI** is an on-premise, privacy-preserving, and explainable ML recruitment screening system.
Unlike traditional keyword-matching ATS and unlike cloud LLMs (OpenAI, Gemini), TalentProof:
- Runs **100% locally** (Zero external API dependencies).
- Analyzes candidate CVs based on **EVIDENCE** (Context spans) rather than pure keywords.
- Implements strict **Negation Handling** (e.g., *"no experience in Docker"* is disqualified as a skill).
- Adheres to the **CIA Triad** (Confidentiality via PII Masking, Integrity via Evidence Verification, Availability via local low-latency inference).

---

## 2. Non-Negotiable Frontend Rules (ANTI-"VIBE-CODED")
**Strict Design System Guidelines:**
- **NO EMOJIS:** Absolutely no emojis (no 🎯, 🚀, 🤖, 📊, 🔍, ⚠️, ✅, ❌, etc.) in the user interface, buttons, titles, headers, badges, or logs. Emojis look amateurish and unprofessional to enterprise judges.
- **NO VIBE-CODED EFFECTS:** 
  - Strictly **NO** `backdrop-filter: blur(...)` or frosted glass effects.
  - Strictly **NO** glowing neon shadows, purple/cyan gradients, or playful bounce animations.
  - Strictly **NO** toy/cartoon icon packs. Use plain text labels or standard enterprise vector iconography.
- **Enterprise Visual Identity:**
  - Color palette: Slate / Charcoal / Deep Navy (`#0F172A`, `#1E293B`, `#334155`), Neutral Off-White (`#F8FAFC`, `#FFFFFF`), and Muted Corporate Status colors:
    - Qualified / Pass: Emerald Muted (`#065F46` text, `#ECFDF5` background, `#A7F3D0` border).
    - Unqualified / Disqualified: Rose Muted (`#9F1239` text, `#FFF1F2` background, `#FECDD3` border).
    - Neutral / Information: Slate Muted (`#334155` text, `#F1F5F9` background, `#CBD5E1` border).
  - High contrast typography, clear 1px borders, data grid tables, and distinct metrics.

---

## 3. Strict Internationalization (i18n) Rules
- **Languages Supported:** Azerbaijani (`AZ`), English (`EN`), Russian (`RU`).
- **Language Switcher UI:** Must be positioned cleanly in the top navigation/header. It should display clean professional text labels (e.g. `AZ ▾`, `EN ▾`, `RU ▾`) — strictly **NO flags or flag emojis**.
- **Translation Completeness:** Every user-facing UI text, label, metric, button, header, error message, and audit log MUST have high-quality, professional corporate translations across all 3 languages.
- **Agent Self-Check Requirement:** When adding new UI components or text elements, any AI assistant MUST update the translation dictionary for `AZ`, `EN`, and `RU` simultaneously. Incomplete or hardcoded single-language strings are strictly forbidden.

---

## 4. Security & CIA Triad Contracts
- **Confidentiality:**
  - `evidence_engine.mask_pii()` must be preserved. When Blind Screening is toggled ON, no emails, phones, URLs, or candidate names should ever be displayed in the review cards.
  - No candidate resume or profile data must be transmitted over the internet or sent to external LLMs.
- **Integrity:**
  - Decisions must be backed by `evidence_map` (exact sentence quotes from the CV).
  - Negation rules (`negation_patterns`) must never be stripped away.
- **Availability:**
  - Candidate parsing, feature engineering, and inference must execute in under 50ms per candidate.

---

## 5. Machine Learning & Backend Contracts
- **Core Model Files:**
  - `talentproof_ml_model.pkl`: Pre-trained Random Forest Classifier. Expects features: `['skill_match_ratio', 'semantic_similarity', 'exp_ratio']`.
  - `tfidf_vectorizer.pkl`: Pre-trained TF-IDF vectorizer for semantic cosine similarity.
- **FastAPI Backend (`backend/`):**
  - All ML and evidence parsing logic is exposed via clean REST endpoints (`/api/screen`, `/api/what-if`, `/api/upload-cv`, `/api/benchmark`).
  - Input and output data models must strictly follow Pydantic schemas.

---

## 6. File Structure Reference
```
talentproof/
├── app.py                  # Corporate Streamlit Dashboard (Clean, Zero-Emoji, Multi-language)
├── evidence_engine.py      # Core NLP, Regex Negation Engine, PII Masking, Span Extractor
├── backend/
│   ├── main.py             # Enterprise FastAPI Server
│   └── schemas.py          # Strict Pydantic Data Contracts
├── talentproof_ml_model.pkl# Pre-trained Random Forest (Do not overwrite with dummy code)
├── tfidf_vectorizer.pkl    # Pre-trained TF-IDF Vectorizer
├── evaluate_model.py       # Benchmark evaluation script (100 candidate benchmark)
├── BENCHMARK_REPORT.md     # Audit report for judges (Accuracy, Precision, Recall, F1)
└── AGENTS.md               # THIS FILE — System Governance
```
