import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

def generate_training_data(n_samples=600, random_state=42):
    np.random.seed(random_state)
    
    # 3 əsas xüsusiyyət (Feature Engine):
    # 1. skill_match_ratio: 0.0 - 1.0 (Tapılan sübutlu bacarıqların nisbəti)
    # 2. semantic_similarity: 0.0 - 1.0 (TF-IDF kosinus oxşarlığı)
    # 3. exp_ratio: 0.0 - 2.5 (Namizədin təcrübəsi / Tələb olunan təcrübə)
    
    skill_match = np.random.beta(a=2, b=2, size=n_samples)
    semantic_sim = np.clip(skill_match * 0.6 + np.random.normal(0.2, 0.1, size=n_samples), 0.0, 1.0)
    exp_ratio = np.random.gamma(shape=2.0, scale=0.5, size=n_samples) # 0.0 - ~3.0 aralığı
    
    # Qərar funksiyası:
    # Namizəd o zaman uyğun sayılır ki, əsas bacarıqların əhəmiyyətli hissəsi olsun,
    # təcrübəsi minimum 0.7x yaxınlığında olsun və semantik oxşarlıq təmin edilsin.
    score = (
        0.50 * skill_match + 
        0.25 * np.clip(exp_ratio / 1.0, 0.0, 1.2) / 1.2 + 
        0.25 * semantic_sim +
        np.random.normal(0, 0.05, size=n_samples) # Real dünyadakı küy (noise)
    )
    
    # Ehtimal həddi: 0.55+ uyğundur (1), əks halda uyğun deyil (0)
    labels = (score >= 0.55).astype(int)
    
    df = pd.DataFrame({
        'skill_match_ratio': np.round(skill_match, 2),
        'semantic_similarity': np.round(semantic_sim, 2),
        'exp_ratio': np.round(exp_ratio, 2),
        'suitable': labels
    })
    return df

def train_and_export_model():
    df = generate_training_data(n_samples=1000)
    X = df[['skill_match_ratio', 'semantic_similarity', 'exp_ratio']]
    y = df['suitable']
    
    # Random Forest modeli - explainable və qərarlı ağac strukturu
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        min_samples_split=4,
        random_state=42
    )
    rf.fit(X, y)
    
    # Modelin saxlanması
    model_path = 'talentproof_ml_model.pkl'
    joblib.dump(rf, model_path)
    print(f"[+] Model muveffeqiyyetle telim edildi ve yadda saxlanildi: {model_path}")
    
    # Feature importances
    for feature, importance in zip(X.columns, rf.feature_importances_):
        print(f"    - {feature}: {importance:.3f}")
        
    return rf

if __name__ == "__main__":
    train_and_export_model()
