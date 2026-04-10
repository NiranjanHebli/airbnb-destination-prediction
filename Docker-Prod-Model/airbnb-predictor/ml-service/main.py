import os
import pickle
import warnings
import numpy as np
import pandas as pd
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

warnings.filterwarnings("ignore")

# ── Patch for sklearn version mismatch ──────────────────────────────────────
import sklearn.compose._column_transformer as _ct
if not hasattr(_ct, "_RemainderColsList"):
    class _RemainderColsList(list):
        pass
    _ct._RemainderColsList = _RemainderColsList
# ────────────────────────────────────────────────────────────────────────────

MODEL_PATH      = os.getenv("MODEL_PATH",      "production_model.pkl")
PREPROCESSOR_PATH = os.getenv("PREPROCESSOR_PATH", "preprocessor.pkl")
ENCODER_PATH    = os.getenv("ENCODER_PATH",    "label_encoder.pkl")

# Global model state
state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all artefacts once at startup."""
    with open(MODEL_PATH, "rb") as f:
        bundle = pickle.load(f)

    # Support two pkl formats:
    # 1. Bundle dict: {'pipeline': <Pipeline>, 'label_encoder': <LabelEncoder>}
    # 2. Legacy: raw XGBoost model (no preprocessor bundled)
    if isinstance(bundle, dict) and "pipeline" in bundle:
        state["pipeline"]      = bundle["pipeline"]
        state["label_encoder"] = bundle["label_encoder"]
        state["bundled"]       = True
        print("✅  Loaded bundled pipeline (preprocessor + model + label encoder).")
    else:
        # Legacy format: separate preprocessor.pkl and label_encoder.pkl
        state["model"] = bundle
        with open(PREPROCESSOR_PATH, "rb") as f:
            state["preprocessor"] = pickle.load(f)
        with open(ENCODER_PATH, "rb") as f:
            state["label_encoder"] = pickle.load(f)
        state["bundled"] = False
        print("✅  Loaded legacy model + separate preprocessor and label encoder.")
    yield
    state.clear()


app = FastAPI(
    title="Airbnb Country Predictor — ML Service",
    description="Returns top-5 destination country predictions for a new Airbnb user.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Log every 422 validation error in full so it's visible in docker logs
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    import json
    body = None
    try:
        body = await request.body()
        body = body.decode()
    except Exception:
        pass
    print(f"❌ 422 Validation error | body={body} | errors={exc.errors()}")
    return JSONResponse(status_code=422, content={"detail": exc.errors()})



# ── Input / Output schemas ───────────────────────────────────────────────────

VALID_GENDERS        = {"-unknown-", "FEMALE", "MALE", "OTHER"}
VALID_SIGNUP_METHODS = {"basic", "facebook", "google"}
VALID_DEVICE_TYPES   = {
    "Android App Unknown Phone/Tablet", "Android Phone", "Android Tablet",
    "Chromebook", "Connected TV", "Desktop (Other)", "iPad", "iPhone",
    "Linux Desktop", "Mac Desktop", "SmartTV", "Tablet", "Windows Desktop",
    "Windows Phone", "-unknown-",
}


class PredictRequest(BaseModel):
    age:           float = Field(..., ge=15, le=90, description="User age (15–90)")
    gender:        str   = Field(..., description="User gender")
    signup_method: str   = Field(..., description="Signup method")
    device_type:   str   = Field(..., description="Device type")
    total_actions: int   = Field(..., ge=0, description="Total session actions")
    total_time:    float = Field(..., ge=0, description="Total time spent in seconds")

    @field_validator("gender")
    @classmethod
    def validate_gender(cls, v: str) -> str:
        normalised = v.strip().upper()
        mapping = {"MALE": "MALE", "FEMALE": "FEMALE", "OTHER": "OTHER"}
        if normalised in mapping:
            return mapping[normalised]
        return "-unknown-"

    @field_validator("signup_method")
    @classmethod
    def validate_signup_method(cls, v: str) -> str:
        v = v.strip().lower()
        if v not in VALID_SIGNUP_METHODS:
            raise ValueError(f"signup_method must be one of {sorted(VALID_SIGNUP_METHODS)}")
        return v

    @field_validator("device_type")
    @classmethod
    def validate_device_type(cls, v: str) -> str:
        # Accept exact match or case-insensitive match
        for valid in VALID_DEVICE_TYPES:
            if v.strip().lower() == valid.lower():
                return valid
        return "-unknown-"


class CountryPrediction(BaseModel):
    country:     str
    probability: float


class PredictResponse(BaseModel):
    top5: list[CountryPrediction]


# ── Routes ───────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    loaded = "pipeline" in state or "model" in state
    return {"status": "ok", "model_loaded": loaded}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    is_bundled = state.get("bundled", False)
    if is_bundled and "pipeline" not in state:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")
    if not is_bundled and "model" not in state:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    row = {
        "age":           req.age,
        "total_actions": req.total_actions,
        "total_time":    req.total_time,
        "gender":        req.gender,
        "signup_method": req.signup_method,
        "device_type":   req.device_type,
    }

    if is_bundled:
        # Bundled pipeline already includes the preprocessor — feed raw DataFrame directly
        pipeline = state["pipeline"]
        feature_order = list(pipeline.named_steps["preprocessor"].feature_names_in_)

        # The bundled pipeline was trained with richer column names (e.g. first_device_type,
        # total_secs). Map the 6 API fields to their pipeline equivalents where they differ,
        # then fill every remaining expected column with NaN so SimpleImputer handles them.
        col_map = {
            "device_type":   "first_device_type",
            "total_time":    "total_secs",
            "total_actions": "session_count",
        }
        mapped_row: dict = {}
        for api_col, val in row.items():
            target_col = col_map.get(api_col, api_col)
            mapped_row[target_col] = val

        df = pd.DataFrame([mapped_row])
        # Fill every column the pipeline expects that isn't in our row with NaN
        # SimpleImputer will substitute medians/modes learned during training
        for col in feature_order:
            if col not in df.columns:
                df[col] = np.nan
        df = df[feature_order]   # ensure correct column order

        try:
            proba = pipeline.predict_proba(df)[0]        # shape: (n_classes,)
        except Exception as exc:
            raise HTTPException(status_code=422, detail=f"Prediction error: {exc}")
        classes = state["label_encoder"].classes_
    else:
        # Legacy: separate preprocessor + model
        feature_order = list(state["preprocessor"].feature_names_in_)
        df = pd.DataFrame([row])[feature_order]
        try:
            X = state["preprocessor"].transform(df)
        except Exception as exc:
            raise HTTPException(status_code=422, detail=f"Preprocessing error: {exc}")
        proba = state["model"].predict_proba(X)[0]       # shape: (n_classes,)
        classes = state["label_encoder"].classes_

    top5_idx = np.argsort(proba)[::-1][:5]
    top5 = [
        CountryPrediction(
            country=classes[i],
            probability=round(float(proba[i]), 6),
        )
        for i in top5_idx
    ]

    return PredictResponse(top5=top5)
