import os
import time
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import xgboost as xgb
import lightgbm as lgb
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def ndcg_at_k(y_true, y_proba, k=5):
    """Kaggle NDCG@5: measures quality of top-k ranked predictions."""
    top_k = np.argsort(-y_proba, axis=1)[:, :k]
    scores = []
    for i in range(len(y_true)):
        if y_true[i] in top_k[i]:
            rank = int(np.where(top_k[i] == y_true[i])[0][0])
            scores.append(1.0 / np.log2(rank + 2))
        else:
            scores.append(0.0)
    return float(np.mean(scores))

def evaluate(name, model, X_v, y_v, fit_time):
    """Evaluate a fitted model and return result dict."""
    proba = model.predict_proba(X_v)
    preds = model.predict(X_v)
    acc   = accuracy_score(y_v, preds)
    f1    = f1_score(y_v, preds, average='macro', zero_division=0)
    ndcg  = ndcg_at_k(y_v, proba)
    logging.info('  {:<22} | Acc: {:.4f} | F1: {:.4f} | NDCG@5: {:.4f} | Time: {:.1f}s'.format(
        name, acc, f1, ndcg, fit_time))
    return {'Model': name, 'Accuracy': acc, 'Macro F1': f1, 'NDCG@5': ndcg,
            'Fit Time (s)': fit_time}

def main():
    data_path = 'data/processed/final_processed_data.csv'
    logging.info(f"Loading processed data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Check if target column is present and not null
    target_col = 'country_destination'
    if target_col not in df.columns:
        raise ValueError(f"Target column {target_col} not found in data.")
        
    y = df[target_col].copy()
    
    # Target encoding
    le_target = LabelEncoder()
    y_encoded = le_target.fit_transform(y)
    n_classes = len(le_target.classes_)
    logging.info(f"Target classes ({n_classes}): {list(le_target.classes_)}")
    
    drop_cols = ['id', 'user_id', target_col]
    feature_cols = [c for c in df.columns if c not in drop_cols and df[c].dtype != object]
    
    X = df[feature_cols].fillna(0).values.astype(np.float32)
    logging.info(f"Feature matrix shape: {X.shape}")
    
    # Train/Val Split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    logging.info(f"Split Train: {X_train.shape} | Val: {X_val.shape}")
    
    # Identify Imbalance
    logging.info("=== Identifying Class Imbalance ===")
    counts = Counter(y_train)
    total = sum(counts.values())
    for class_idx in sorted(counts.keys()):
        class_name = le_target.inverse_transform([class_idx])[0]
        cnt = counts[class_idx]
        logging.info(f"Class {class_name:5} | Count: {cnt:6d} | Proportion: {cnt/total*100:.2f}%")
        
    results = {}
    
    # Train XGBoost
    logging.info("=== Training XGBoost (with SMOTE Pipeline) ===")
    t0 = time.time()
    xgb_classifier = xgb.XGBClassifier(
        n_estimators     = 300,
        max_depth        = 6,
        learning_rate    = 0.1,
        subsample        = 0.8,
        colsample_bytree = 0.8,
        min_child_weight = 5,
        gamma            = 0.1,
        reg_alpha        = 0.1,
        reg_lambda       = 1.0,
        objective        = 'multi:softprob',
        num_class        = n_classes,
        eval_metric      = 'mlogloss',
        random_state     = 42,
        n_jobs           = -1,
        verbosity        = 0
    )
    xgb_pipeline = ImbPipeline([
        ('smote', SMOTE(random_state=42)),
        ('classifier', xgb_classifier)
    ])
    xgb_pipeline.fit(
        X_train, y_train,
        classifier__eval_set=[(X_val, y_val)],
        classifier__verbose=False
    )
    results['XGBoost'] = evaluate('XGBoost Pipeline', xgb_pipeline, X_val, y_val, time.time()-t0)
    
    # Train LightGBM
    logging.info("=== Training LightGBM (with SMOTE Pipeline) ===")
    t0 = time.time()
    lgb_classifier = lgb.LGBMClassifier(
        n_estimators     = 300,
        max_depth        = 7,
        learning_rate    = 0.08,
        subsample        = 0.8,
        colsample_bytree = 0.8,
        min_child_samples= 20,
        reg_alpha        = 0.1,
        reg_lambda       = 1.0,
        objective        = 'multiclass',
        num_class        = n_classes,
        random_state     = 42,
        n_jobs           = -1,
        verbose          = -1
    )
    lgb_pipeline = ImbPipeline([
        ('smote', SMOTE(random_state=42)),
        ('classifier', lgb_classifier)
    ])
    lgb_pipeline.fit(
        X_train, y_train, 
        classifier__eval_set=[(X_val, y_val)]
    )
    results['LightGBM'] = evaluate('LightGBM Pipeline', lgb_pipeline, X_val, y_val, time.time()-t0)

if __name__ == "__main__":
    main()
