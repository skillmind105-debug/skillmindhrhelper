# TalentProof AI V3

## Files
- match_model.joblib: trained V3 model
- tfidf_vectorizer.joblib: vectorizer fitted for V3
- skill_patterns.joblib: skill extraction patterns
- inference.py: prediction function
- evaluation_metrics.json: V3 evaluation metrics
- requirements.txt: dependencies

## Usage
from inference import predict_cv_job_match

result = predict_cv_job_match(
    cv_text="Python, SQL, Excel, Power BI...",
    job_text="Data Analyst requiring Python, SQL..."
)
print(result)

The model is a prototype trained using weak labels.
Its score is not a calibrated hiring probability.
Use human review for final hiring decisions.
