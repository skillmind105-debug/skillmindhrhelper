# TalentProof AI — Enterprise Recruitment Screening Intelligence

> **Privacy-Preserving, Evidence-Based, On-Premise ML Candidate Screening Engine**  
> Built for Enterprise HR Teams & Recruitment Audits. Adheres strictly to the **CIA Triad** (Zero Cloud LLM dependencies).

---

## 1. Live Endpoints & Deployments

- **Production REST API (Vercel):** [https://skillmindhrhelper.vercel.app](https://skillmindhrhelper.vercel.app)
- **Interactive Swagger Documentation:** [https://skillmindhrhelper.vercel.app/docs](https://skillmindhrhelper.vercel.app/docs)
- **Local / On-Premise UI Dashboard:** Streamlit Dashboard via `streamlit run app.py`

---

## 2. Core Architectural Pillars (CIA Triad)

1. **Confidentiality (Məxfilik):**
   - **Zero Cloud Transmission:** Resumes, PII, and evaluations are processed 100% locally. No data is sent to external LLM providers (OpenAI, Gemini, Anthropic).
   - **Blind Screening Engine:** Automatically detects and redacts candidate names, emails, phones, URLs, and profile photos from review cards and audit trails.

2. **Integrity (Bütövlük & İzaholunma):**
   - **Context Span Verification:** Every matched competency is mapped directly to exact quoted sentences in the resume.
   - **Negation Handling:** Rigorous regex-based negation detector disqualifies skills mentioned in negative context (e.g., *"no knowledge of Docker"*).
   - **Explainable Decision Rationale:** Generates explicit reasons for qualification/disqualification and comparative ranking deltas against competing candidates.

3. **Availability (Əlçatanlıq):**
   - Pre-trained Random Forest ML model (`talentproof_ml_model.pkl`) and TF-IDF semantic vectorizer execute inference in **<15ms per candidate**.

---

## 3. Benchmark & Quality Metrics (100 Sample Validation)

| Metric | Score | Industry ATS Baseline |
| :--- | :--- | :--- |
| **Accuracy** | **91.0%** | ~65.0% |
| **Precision** | **90.2%** | ~60.0% |
| **Recall** | **94.8%** | ~72.0% |
| **F1 Score** | **92.4%** | ~65.0% |

- **Confusion Matrix:** `[[TN: 36, FP: 6], [FN: 3, TP: 55]]`
- **Inference Latency:** `< 15ms` per resume on standard CPU hardware.

---

## 4. Internationalization (i18n)

Full corporate tripartite language localization without external translation APIs:
- **Azerbaijani (AZ)**
- **English (EN)**
- **Russian (RU)**

---

## 5. Local Setup & Quickstart

```bash
# Clone the repository
git clone https://github.com/skillmind105-debug/skillmindhrhelper.git
cd skillmindhrhelper

# Install dependencies
pip install -r requirements.txt

# Launch Enterprise Streamlit Dashboard
streamlit run app.py

# Launch FastAPI REST Backend
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```
