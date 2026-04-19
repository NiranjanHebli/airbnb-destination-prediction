import os
import sys
import pickle
import warnings
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import xgboost as xgb
import sklearn

print(f"scikit-learn  : {sklearn.__version__}")
print(f"xgboost       : {xgb.__version__}")
print(f"numpy         : {np.__version__}")

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
RESEARCH_DIR = os.path.join(
    os.path.dirname(__file__),
    "..", "..", "airbnb-recruiting-new-user-bookings"
)
TRAIN_CSV    = os.path.join(RESEARCH_DIR, "train_users_2.csv")
SESSIONS_CSV = os.path.join(RESEARCH_DIR, "sessions.csv")

for p in [TRAIN_CSV, SESSIONS_CSV]:
    if not os.path.exists(p):
        sys.exit(f"  File not found: {p}\nMake sure the research directory is at {RESEARCH_DIR}")

# ── Load data ──────────────────────────────────────────────────────────────────
print("\n📥  Loading training data…")
train = pd.read_csv(TRAIN_CSV)
sessions = pd.read_csv(SESSIONS_CSV, usecols=["user_id", "action", "secs_elapsed"])

# ── Session aggregations (maps to total_actions & total_time in the API) ───────
print("⚙️   Aggregating sessions…")
sess_agg = (
    sessions.groupby("user_id")
    .agg(
        total_actions=("action", "count"),
        total_time=("secs_elapsed", "sum"),
    )
    .reset_index()
)

# ── User feature engineering ───────────────────────────────────────────────────
print("🔧  Engineering features…")
df = train.copy()

# Age: clip to valid range, fill outliers with median
df["age"] = pd.to_numeric(df["age"], errors="coerce")
df.loc[df["age"] > 100, "age"] = np.nan
df.loc[df["age"] < 15,  "age"] = np.nan
age_median = df["age"].median()
df["age"] = df["age"].fillna(age_median)

# Gender: normalise
df["gender"] = df["gender"].str.upper().fillna("-unknown-")

# Signup method: keep only known values
valid_signup = {"basic", "facebook", "google"}
df["signup_method"] = df["signup_method"].str.lower().where(
    df["signup_method"].str.lower().isin(valid_signup), "basic"
)

# Device type: rename first_device_type → device_type
df = df.rename(columns={"first_device_type": "device_type"})
df["device_type"] = df["device_type"].fillna("-unknown-")

# Merge session aggregations
df = df.merge(sess_agg, left_on="id", right_on="user_id", how="left")
df["total_actions"] = df["total_actions"].fillna(0).astype(int)
df["total_time"]    = df["total_time"].fillna(0.0)

# Drop rows without a label
df = df.dropna(subset=["country_destination"])

# ── Feature / target split ─────────────────────────────────────────────────────
FEATURE_COLS = ["age", "total_actions", "total_time", "gender", "signup_method", "device_type"]
NUM_COLS = ["age", "total_actions", "total_time"]
CAT_COLS = ["gender", "signup_method", "device_type"]

X = df[FEATURE_COLS].copy()
y_raw = df["country_destination"].copy()

print(f"   Training rows : {len(X):,}")
print(f"   Classes       : {sorted(y_raw.unique())}")

# ── Label encode target ────────────────────────────────────────────────────────
le = LabelEncoder()
y = le.fit_transform(y_raw)

# ── Preprocessor ──────────────────────────────────────────────────────────────
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), NUM_COLS),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_COLS),
    ]
)

# ── Train / val split ──────────────────────────────────────────────────────────
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)

print("\n🚀  Fitting preprocessor…")
X_train_t = preprocessor.fit_transform(X_train)
X_val_t   = preprocessor.transform(X_val)

print("🚀  Training XGBoost with SMOTE…")
model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    objective="multi:softprob",
    num_class=len(le.classes_),
    eval_metric="mlogloss",
    use_label_encoder=False,
    random_state=42,
    n_jobs=-1,
)
imb_pipeline = ImbPipeline([
    ('smote', SMOTE(random_state=42)),
    ('classifier', model)
])
imb_pipeline.fit(
    X_train_t, y_train,
    classifier__eval_set=[(X_val_t, y_val)],
    classifier__verbose=50,
)

# ── Quick NDCG@5 on val ───────────────────────────────────────────────────────
def ndcg5(y_true, proba):
    score = 0.0
    for i, true_cls in enumerate(y_true):
        top5 = np.argsort(proba[i])[::-1][:5]
        if true_cls in top5:
            rank = np.where(top5 == true_cls)[0][0]
            score += 1.0 / np.log2(rank + 2)
    return score / len(y_true)

proba_val = imb_pipeline.predict_proba(X_val_t)
score = ndcg5(y_val, proba_val)
print(f"\n  Validation NDCG@5 : {score:.4f}")

# ── Save pkl files ────────────────────────────────────────────────────────────
print("\n  Saving pkl files…")

with open("production_model.pkl", "wb") as f:
    pickle.dump(imb_pipeline, f)

with open("preprocessor.pkl", "wb") as f:
    pickle.dump(preprocessor, f)

with open("label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

print("Done!")
print("   production_model.pkl —", os.path.getsize("production_model.pkl") // 1024, "KB")
print("   preprocessor.pkl     —", os.path.getsize("preprocessor.pkl") // 1024, "KB")
print("   label_encoder.pkl    —", os.path.getsize("label_encoder.pkl") // 1024, "KB")
print("\nRebuild the Docker image to pick up the new pkl files:")
print("   docker compose up --build ml-service")
