import os
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from pandas.api.types import is_numeric_dtype  # <--- Added import
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score
from src.feature_engineering import build_features

ARTIFACTS_DIR = "artifacts"
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "nfl_prospect_model.joblib")

def train_and_export():
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    print("📂 Ingesting datasets and building features...")
    df_polars = build_features()

    if df_polars.is_empty():
        print("⚠️ No data loaded. Check files in data/ directory.")
        return

    # Convert to Pandas for scikit-learn / LightGBM
    df = df_polars.to_pandas()

    # Define target and feature set
    target_col = "target_high_impact"
    drop_cols = ["nfl_id", "player_name", "first_name", "last_name", "birth_date", 
                 "college_name", "position", "draft_round", target_col]
    
    # Filter features to numeric columns safely using pandas API
    features = [c for c in df.columns if c not in drop_cols and is_numeric_dtype(df[c])]
    
    X = df[features]
    y = df[target_col].fillna(0).astype(int)

    # Calculate class imbalance ratio for scale_pos_weight
    neg_count = (y == 0).sum()
    pos_count = (y == 1).sum()
    scale_weight = neg_count / max(pos_count, 1)

    print(f"📊 Dataset Size: {len(df)} samples | High-Impact: {pos_count} | Baseline: {neg_count}")
    print(f"⚖️ Applied Class Imbalance Weight (scale_pos_weight): {scale_weight:.2f}\n")

    # 5-Fold Stratified Cross-Validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    acc_scores, auc_scores, recall_scores = [], [], []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = lgb.LGBMClassifier(
            n_estimators=100,
            learning_rate=0.05,
            scale_pos_weight=scale_weight,
            random_state=42,
            verbose=-1
        )
        model.fit(X_train, y_train)

        preds = model.predict(X_val)
        probs = model.predict_proba(X_val)[:, 1]

        acc_scores.append(accuracy_score(y_val, preds))
        auc_scores.append(roc_auc_score(y_val, probs))
        recall_scores.append(recall_score(y_val, preds))

    print("--- 5-Fold Cross-Validation Results ---")
    print(f"Mean Accuracy: {np.mean(acc_scores):.4f}")
    print(f"Mean ROC-AUC:  {np.mean(auc_scores):.4f}")
    print(f"Mean Recall:   {np.mean(recall_scores):.4f}\n")

    # Final Model Training on full dataset
    final_model = lgb.LGBMClassifier(
        n_estimators=100,
        learning_rate=0.05,
        scale_pos_weight=scale_weight,
        random_state=42,
        verbose=-1
    )
    final_model.fit(X, y)

    # Export model artifact
    joblib.dump(final_model, MODEL_PATH)
    print(f"✅ Trained model successfully saved to {MODEL_PATH}")

if __name__ == "__main__":
    train_and_export()