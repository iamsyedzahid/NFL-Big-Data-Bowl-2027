import os
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from src.feature_engineering import build_features

ARTIFACT_DIR = "artifacts"

def get_fallback_data():
    np.random.seed(42)
    n = 500
    return pd.DataFrame({
        'combine_height': np.random.uniform(68, 78, n),
        'combine_weight': np.random.uniform(180, 320, n),
        'hand_size': np.random.uniform(8.5, 10.5, n),
        'arm_length': np.random.uniform(30.0, 35.0, n),
        'ten_yd_split': np.random.uniform(1.45, 1.75, n),
        'forty': np.random.uniform(4.30, 5.20, n),
        'vertical': np.random.uniform(28.0, 42.0, n),
        'broad_jump': np.random.uniform(100.0, 135.0, n),
        'three_cone': np.random.uniform(6.60, 7.80, n),
        'short_shuttle': np.random.uniform(4.00, 4.60, n),
        'bench_reps': np.random.randint(10, 35, n),
        'ngs_athleticism_score': np.random.uniform(50, 99, n),
        'ngs_college_production_score': np.random.uniform(50, 99, n),
        'combine_max_speed': np.random.uniform(5.0, 12.0, n),
        'combine_max_accel': np.random.uniform(2.0, 6.0, n),
        'avg_jerk_ratio': np.random.uniform(0.1, 1.5, n),
        'target_high_impact': np.random.choice([0, 1], size=n, p=[0.7, 0.3])
    })

def train_and_export():
    df_polars = build_features()

    if len(df_polars) > 0:
        print("📂 Building model from real CSV datasets in data/...")
        df = df_polars.to_pandas()
    else:
        print("⚠️ Data CSVs missing in data/ directory. Training using synthetic baseline data...")
        df = get_fallback_data()

    features = [
        'combine_height', 'combine_weight', 'hand_size', 'arm_length',
        'ten_yd_split', 'forty', 'vertical', 'broad_jump', 'three_cone',
        'short_shuttle', 'bench_reps', 'ngs_athleticism_score',
        'ngs_college_production_score', 'combine_max_speed',
        'combine_max_accel', 'avg_jerk_ratio'
    ]

    for feat in features:
        if feat not in df.columns:
            df[feat] = 0.0

    X = df[features].fillna(0)
    y = df['target_high_impact'].fillna(0)

    clf = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05, random_state=42)
    clf.fit(X, y)

    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    output_path = os.path.join(ARTIFACT_DIR, "nfl_prospect_model.joblib")
    
    artifact = {
        "model": clf,
        "features": features
    }
    
    joblib.dump(artifact, output_path)
    print(f"✅ Baseline model saved successfully to {output_path}")

if __name__ == "__main__":
    train_and_export()