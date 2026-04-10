"""
Airbnb New User Bookings — Production-Grade ML Preprocessing Pipeline v2
=========================================================================
Author  : Senior ML Engineer / Data Pipeline Architect
Version : 2.0  (upgraded from v1)
Purpose : Clean, aggregate, merge, encode, and validate user + session data
          into a single ML-ready feature table — safe for both TRAINING and
          INFERENCE without refit.

Key upgrades over v1
--------------------
  [U-01]  OrdinalEncoder replaces LabelEncoder → handles unseen categories at
          inference without crashing (handle_unknown='use_encoded_value', -1)
  [U-02]  Fit / Transform split → AirbnbPipeline.fit() learns all statistics
          from training data; .transform() applies them deterministically at
          inference time with zero data leakage
  [U-03]  country_destination NEVER encoded or used as a feature — guaranteed
          by explicit exclusion list at the start of encoding
  [U-04]  Data-leakage assertion: date_first_booking + timestamp_first_active
          are hard-checked absent before any feature step
  [U-05]  Frequency encoding for high-cardinality columns (top_action,
          first_action, last_action, most_used_device) learned from train set
          and applied at transform time — no collision from unseen tokens
  [U-06]  Action entropy (scipy-free, fully vectorised) added as a feature
  [U-07]  session_count_per_day, device_diversity_ratio, recency_gap, and
          time-of-day bucket features added
  [U-08]  gc.collect() after every large intermediate DataFrame deletion
  [U-09]  Structured Python logging replaces bare print() calls — severity
          levels, timestamps, easy redirection to file
  [U-10]  Schema + dtype validation at pipeline entry and exit
  [U-11]  Defensive guards: empty DataFrames, missing columns, bad dtypes
  [U-12]  np.maximum() replaces +1 denominator hacks for numerical stability
  [U-13]  top_action lambda replaced with fully vectorised idxmax on
          pre-computed pivot to eliminate Python-level group loop
  [U-14]  Deterministic sort (kind='mergesort') everywhere — no quicksort
          non-determinism on tied keys
  [U-15]  Single joblib dump of ALL artifacts: encoders, freq maps, stats,
          feature list → one file to version and ship to inference servers
  [U-16]  run_pipeline() returns a named tuple with df, feature_list,
          encoders, and a printed summary report

Assumptions
-----------
  • 'id' column in users CSV is the primary key (string or int — normalised)
  • 'user_id' in sessions CSV joins to users 'id'
  • Target column name is 'country_destination' (string labels kept as-is)
  • SESSION_TIMESTAMP_COL may be None if sessions have no timestamp
  • secs_elapsed is seconds (not milliseconds)
  • Age range [15, 90] is the agreed valid band
"""

# ══════════════════════════════════════════════════════════════════════════════
# IMPORTS
# ══════════════════════════════════════════════════════════════════════════════
import gc
import logging
import os
import time
from collections import namedtuple
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder   # [U-01]

# ══════════════════════════════════════════════════════════════════════════════
# LOGGING  [U-09]
# ══════════════════════════════════════════════════════════════════════════════
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("airbnb_pipeline")


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════
TRAIN_USERS_PATH      = "train_users_2.csv"
SESSIONS_PATH         = "sessions.csv"
OUTPUT_PATH           = "final_processed_data.csv"
ARTIFACTS_PATH        = "pipeline_artifacts.pkl"   # [U-15] single artifact bundle

SESSION_TIMESTAMP_COL = "timestamp"   # set None if absent in your data

# Columns that must NEVER appear in the feature matrix  [U-03, U-04]
LEAKAGE_COLUMNS = {"date_first_booking", "timestamp_first_active"}
TARGET_COLUMN   = "country_destination"

# High-cardinality columns that get frequency encoding instead of ordinal  [U-05]
FREQ_ENCODE_COLS = ["top_action", "first_action", "last_action", "most_used_device"]

# Low-cardinality columns that get OrdinalEncoder  [U-01]
ORDINAL_ENCODE_COLS = [
    "gender", "signup_method", "signup_flow", "language",
    "affiliate_channel", "affiliate_provider", "first_affiliate_tracked",
    "signup_app", "first_device_type", "first_browser",
]

# Named tuple for pipeline output  [U-16]
PipelineResult = namedtuple(
    "PipelineResult", ["df", "feature_list", "artifacts", "summary"]
)


# ══════════════════════════════════════════════════════════════════════════════
# SCHEMA & DTYPE VALIDATION  [U-10]
# ══════════════════════════════════════════════════════════════════════════════

REQUIRED_USER_COLS    = {"id", "age", "gender"}
REQUIRED_SESSION_COLS = {"user_id", "action"}


def _validate_dataframe(df: pd.DataFrame, required_cols: set, name: str) -> None:
    """Raise ValueError if required columns are missing or df is empty.  [U-11]"""
    if df.empty:
        raise ValueError(f"[VALIDATION] '{name}' DataFrame is empty — aborting.")
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"[VALIDATION] '{name}' is missing required columns: {missing}"
        )
    log.info("Schema OK for '%s'  (%d rows × %d cols)", name, *df.shape)


def _assert_no_leakage(df: pd.DataFrame) -> None:
    """Hard-fail if any known leakage column is present.  [U-04]"""
    found = LEAKAGE_COLUMNS & set(df.columns)
    assert not found, (
        f"DATA LEAKAGE DETECTED — remove columns before training: {found}"
    )
    log.info("Leakage check PASSED — no forbidden columns detected.")


def _assert_no_nulls(df: pd.DataFrame, context: str = "") -> None:
    """Hard-fail if any NaN remains.  [U-10]"""
    n = df.isnull().sum().sum()
    assert n == 0, f"FATAL: {n:,} NaN values remain after preprocessing ({context})"
    log.info("Null check PASSED — zero NaNs (%s).", context)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════

def load_users(path: str) -> pd.DataFrame:
    """
    Load train_users.csv with explicit low_memory=False.
    Validates schema immediately after load.  [U-10, U-11]
    """
    log.info("[STEP 1] Loading user data from '%s' …", path)
    df = pd.read_csv(path, low_memory=False)
    _validate_dataframe(df, REQUIRED_USER_COLS, "users")
    log.info("         Missing per column: %s",
             df.isnull().sum()[df.isnull().sum() > 0].to_dict())
    return df


def load_sessions(path: str, chunksize: int = 500_000) -> pd.DataFrame:
    """
    Load sessions.csv in chunks (handles 10 M+ rows without OOM).

    v1 issue: category dtype is lost after pd.concat because chunk-level
    category sets differ → object dtype returned.
    Fix: re-cast AFTER concat (retained from v1) AND use observed=True in
    downstream groupby calls to skip unused categories.  [U-11]
    """
    log.info("[STEP 1] Loading sessions from '%s' in %d-row chunks …",
             path, chunksize)
    cat_cols = ["action", "action_type", "action_detail", "device_type"]
    chunks   = []

    for i, chunk in enumerate(pd.read_csv(path, low_memory=False,
                                           chunksize=chunksize)):
        for col in cat_cols:
            if col in chunk.columns:
                chunk[col] = chunk[col].astype("category")
        chunks.append(chunk)
        log.debug("  chunk %d: %d rows", i + 1, len(chunk))

    df = pd.concat(chunks, ignore_index=True)
    del chunks
    gc.collect()   # [U-08] free chunk list immediately

    # Re-cast after concat (category sets differ across chunks)
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype("category")

    _validate_dataframe(df, REQUIRED_SESSION_COLS, "sessions")
    log.info("         → %d total rows | %d columns", *df.shape)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — CLEAN USER DATA
# ══════════════════════════════════════════════════════════════════════════════

def clean_users(df: pd.DataFrame, age_median: Optional[float] = None) -> Tuple[pd.DataFrame, float]:
    """
    Clean user features.  Returns (cleaned_df, age_median_used).

    Signature change vs v1: returns age_median so fit() can store it and
    transform() can reuse the TRAINING median at inference time — preventing
    test-set statistics from contaminating the imputation.  [U-02]

    Parameters
    ----------
    df         : raw users DataFrame
    age_median : if provided (inference mode), use this value instead of
                 computing from df (which would be test-set leakage).
    """
    log.info("[STEP 2] Cleaning user data …")

    # [U-11] Defensive: work on a copy only for users (small table — acceptable)
    df = df.copy()

    # ── ID normalisation ─────────────────────────────────────────────────────
    df["id"] = df["id"].astype(str).str.strip()

    # ── Leakage columns ───────────────────────────────────────────────────────
    # Drop known leakage + redundant date columns first so they never feed into
    # any downstream step — even accidentally.  [U-04]
    drop_early = list(LEAKAGE_COLUMNS) + [TARGET_COLUMN]
    # Keep target column separate; we'll re-attach it after feature processing
    target_series = df[TARGET_COLUMN].copy() if TARGET_COLUMN in df.columns else None
    cols_to_drop  = [c for c in drop_early if c in df.columns and c != TARGET_COLUMN]
    if cols_to_drop:
        df.drop(columns=cols_to_drop, inplace=True)
        log.info("         Dropped leakage/redundant columns: %s", cols_to_drop)

    # ── Age ───────────────────────────────────────────────────────────────────
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df.loc[(df["age"] < 15) | (df["age"] > 90), "age"] = np.nan

    if age_median is None:
        # TRAIN mode: learn from data
        age_median = float(df["age"].median())
    # INFERENCE mode: use stored training median — no leakage  [U-02]
    df["age"] = df["age"].fillna(age_median).astype("float32")
    log.info("         age → clipped & imputed; median=%.1f", age_median)

    # ── Gender / categoricals ─────────────────────────────────────────────────
    if "gender" in df.columns:
        df["gender"] = df["gender"].replace("-unknown-", "Unknown").fillna("Unknown")

    for col in ["first_browser", "language", "affiliate_channel",
                "signup_method", "affiliate_provider", "first_affiliate_tracked",
                "signup_app", "first_device_type"]:
        if col in df.columns:
            df[col] = (df[col].fillna("Unknown")
                               .replace("-unknown-", "Unknown")
                               .astype(str))

    # ── Date account created → temporal features ──────────────────────────────
    date_col = "date_account_created"
    if date_col in df.columns:
        dt = pd.to_datetime(df[date_col], errors="coerce")
        df["dac_year"]    = dt.dt.year.astype("Int16")
        df["dac_month"]   = dt.dt.month.astype("Int8")
        df["dac_day"]     = dt.dt.day.astype("Int8")
        df["dac_weekday"] = dt.dt.dayofweek.astype("Int8")
        # NEW: quarter (seasonal booking signal)
        df["dac_quarter"] = dt.dt.quarter.astype("Int8")
        df.drop(columns=[date_col], inplace=True)
        log.info("         '%s' expanded to year/month/day/weekday/quarter", date_col)

    # ── Re-attach target (never used as feature) ──────────────────────────────
    if target_series is not None:
        df[TARGET_COLUMN] = target_series.values

    log.info("         → users shape after cleaning: %s", df.shape)
    return df, age_median


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — CLEAN SESSION DATA
# ══════════════════════════════════════════════════════════════════════════════

def clean_sessions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fills NaN, drops duplicates, parses timestamps, sorts.

    v1 had a Python loop to handle category→fillna.  [U-11]
    v2: convert all category cols to str at once with .astype(), then batch fillna.
    """
    log.info("[STEP 3] Cleaning session data …")

    # [U-11] Guard: empty sessions
    if df.empty:
        log.warning("         Sessions DataFrame is empty — returning as-is.")
        return df

    # ── Fill missing — vectorised batch ───────────────────────────────────────
    # Convert category cols to str first (avoids per-column loop)
    cat_cols = df.select_dtypes(include=["category"]).columns.tolist()
    if cat_cols:
        df[cat_cols] = df[cat_cols].astype(str)

    obj_cols = df.select_dtypes(include=["object"]).columns.tolist()
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Single vectorised fillna per dtype group — no Python loop  [U-04]
    df[obj_cols] = df[obj_cols].fillna("Unknown").replace("nan", "Unknown")
    df[num_cols] = df[num_cols].fillna(0)

    # ── Remove duplicates ─────────────────────────────────────────────────────
    before = len(df)
    df.drop_duplicates(inplace=True)
    log.info("         Duplicates removed: %d", before - len(df))

    # ── Timestamp parsing + deterministic sort  [U-14] ────────────────────────
    has_ts = SESSION_TIMESTAMP_COL and SESSION_TIMESTAMP_COL in df.columns
    if has_ts:
        df[SESSION_TIMESTAMP_COL] = pd.to_datetime(
            df[SESSION_TIMESTAMP_COL], errors="coerce"
        )
        # [U-14] kind='mergesort' → stable sort → deterministic on ties
        df.sort_values(
            ["user_id", SESSION_TIMESTAMP_COL],
            inplace=True, kind="mergesort"
        )
        log.info("         Timestamp parsed; sorted by user_id + %s", SESSION_TIMESTAMP_COL)
    else:
        df.sort_values("user_id", inplace=True, kind="mergesort")
        log.info("         No timestamp column — sorted by user_id only")

    df.reset_index(drop=True, inplace=True)
    log.info("         → sessions shape after cleaning: %s", df.shape)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — AGGREGATE SESSION DATA
# ══════════════════════════════════════════════════════════════════════════════

def _action_entropy_vectorised(df: pd.DataFrame) -> pd.Series:
    """
    Compute per-user action entropy (Shannon) fully vectorised.  [U-06]

    H(u) = -Σ p_i * log2(p_i)   where p_i = count(action_i) / total_actions

    Entropy is high when a user spreads actions evenly across many types
    (exploratory behaviour) and 0 when they repeat a single action.

    v1 had no entropy feature.
    v2: pivot_table + log2 — no per-group Python call.
    """
    # Count occurrences of each action per user
    counts = (
        df.groupby(["user_id", "action"], observed=True)["action"]
        .count()
        .rename("cnt")
        .reset_index()
    )
    totals = counts.groupby("user_id")["cnt"].transform("sum")
    p      = counts["cnt"] / totals
    counts["h"] = -p * np.log2(p.clip(lower=1e-9))   # clip avoids log2(0)
    entropy = counts.groupby("user_id")["h"].sum().rename("action_entropy")
    return entropy


def aggregate_sessions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group by user_id and compute a rich set of behavioural features.

    New features vs v1
    ------------------
    action_entropy        [U-06] : behaviour randomness (Shannon entropy)
    session_count_per_day [U-07] : normalised session density
    device_diversity_ratio[U-07] : unique_devices / total_actions
    recency_gap           [U-07] : seconds since last activity (relative to max ts)
    tod_morning / _afternoon / _evening / _night  [U-07] : time-of-day buckets
    action_diversity_ratio        : unique_actions / (total_actions + 1)
    actions_per_second            : np.maximum denominator fix  [U-12]

    Performance fixes vs v1
    -----------------------
    [U-13] top_action: replaced lambda+value_counts (Python group loop) with
           fully vectorised groupby→size→idxmax on a two-column frame.
    [U-14] All sorts use kind='mergesort' for determinism.
    """
    log.info("[STEP 4] Aggregating session data …")
    t0 = time.time()

    # [U-11] Guard: empty or missing column
    if df.empty or "user_id" not in df.columns:
        log.warning("         Empty/invalid sessions — returning empty aggregation.")
        return pd.DataFrame(columns=["user_id"])

    # observed=True skips unused category levels → faster groupby  [U-13]
    grp = df.groupby("user_id", sort=False, observed=True)

    # ── Basic counts ──────────────────────────────────────────────────────────
    agg_kwargs = {
        "total_actions" : ("action", "count"),
        "unique_actions": ("action", "nunique"),
    }
    if "action_type" in df.columns:
        agg_kwargs["unique_action_types"] = ("action_type", "nunique")
    if "device_type" in df.columns:
        agg_kwargs["unique_devices"] = ("device_type", "nunique")

    agg_basic = grp.agg(**agg_kwargs).reset_index()

    # ── top_action — fully vectorised  [U-13] ────────────────────────────────
    # v1 used: .agg(lambda x: x.value_counts().idxmax()) — Python loop per group
    # v2: groupby size on (user_id, action) pair then idxmax on action axis
    if "action" in df.columns:
        action_counts = (
            df.groupby(["user_id", "action"], sort=False, observed=True)
            .size()
            .rename("cnt")
            .reset_index()
        )
        # For each user pick the action with the highest count (stable: first max)
        top_idx = action_counts.groupby("user_id")["cnt"].idxmax()
        top_action = (
            action_counts.loc[top_idx, ["user_id", "action"]]
            .rename(columns={"action": "top_action"})
        )
    else:
        top_action = pd.DataFrame({"user_id": df["user_id"].unique(),
                                   "top_action": "Unknown"})

    # ── first / last action ───────────────────────────────────────────────────
    first_last = grp["action"].agg(
        first_action="first",
        last_action="last",
    ).reset_index()

    # ── most used device ──────────────────────────────────────────────────────
    if "device_type" in df.columns:
        dev_counts = (
            df.groupby(["user_id", "device_type"], sort=False, observed=True)
            .size()
            .rename("cnt")
            .reset_index()
        )
        dev_idx        = dev_counts.groupby("user_id")["cnt"].idxmax()
        most_used_dev  = (
            dev_counts.loc[dev_idx, ["user_id", "device_type"]]
            .rename(columns={"device_type": "most_used_device"})
        )
    else:
        most_used_dev = pd.DataFrame({
            "user_id": df["user_id"].unique(), "most_used_device": "Unknown"
        })

    # ── Time-based features ───────────────────────────────────────────────────
    has_ts = SESSION_TIMESTAMP_COL and SESSION_TIMESTAMP_COL in df.columns

    if has_ts:
        ts = SESSION_TIMESTAMP_COL

        # session_start / session_end / session_duration
        time_agg = grp[ts].agg(
            session_start="min",
            session_end="max",
        ).reset_index()
        time_agg["session_duration"] = (
            (time_agg["session_end"] - time_agg["session_start"])
            .dt.total_seconds()
            .astype("float32")
        )

        # avg_time_between_actions — vectorised diff (no lambda)
        df_ts = df[["user_id", ts]].copy()
        df_ts["ts_diff"] = (
            df_ts.groupby("user_id", sort=False)[ts]
            .diff()
            .dt.total_seconds()
        )
        avg_tba = (
            df_ts.groupby("user_id")["ts_diff"]
            .mean()
            .reset_index()
            .rename(columns={"ts_diff": "avg_time_between_actions"})
        )
        time_agg = time_agg.merge(avg_tba, on="user_id", how="left")
        del df_ts, avg_tba
        gc.collect()

        # [U-07] recency_gap: seconds between session_end and the dataset-wide
        # max timestamp — proxy for "how long ago was this user last active"
        global_max_ts = time_agg["session_end"].max()
        time_agg["recency_gap"] = (
            (global_max_ts - time_agg["session_end"])
            .dt.total_seconds()
            .astype("float32")
        )

        # [U-07] session_count_per_day
        span_days = time_agg["session_duration"] / 86_400.0   # seconds → days
        time_agg["session_count_per_day"] = (
            agg_basic.set_index("user_id")["total_actions"]
            .reindex(time_agg["user_id"].values)
            .values
            / np.maximum(span_days, 1)
        ).astype("float32")

        # [U-07] Time-of-day action buckets (hour of session_start)
        hour = time_agg["session_start"].dt.hour
        time_agg["tod_morning"]   = ((hour >= 6)  & (hour < 12)).astype("int8")
        time_agg["tod_afternoon"] = ((hour >= 12) & (hour < 18)).astype("int8")
        time_agg["tod_evening"]   = ((hour >= 18) & (hour < 22)).astype("int8")
        time_agg["tod_night"]     = ((hour >= 22) | (hour < 6) ).astype("int8")

        # Stringify timestamps for CSV cleanliness
        time_agg["session_start"] = time_agg["session_start"].astype(str)
        time_agg["session_end"]   = time_agg["session_end"].astype(str)

    else:
        # No timestamp — populate placeholder schema so merge works  [U-11]
        unique_users = df["user_id"].unique()
        n            = len(unique_users)
        time_agg = pd.DataFrame({
            "user_id"               : unique_users,
            "session_start"         : "Unknown",
            "session_end"           : "Unknown",
            "session_duration"      : np.zeros(n, dtype="float32"),
            "avg_time_between_actions": np.zeros(n, dtype="float32"),
            "recency_gap"           : np.zeros(n, dtype="float32"),
            "session_count_per_day" : np.zeros(n, dtype="float32"),
            "tod_morning"           : np.zeros(n, dtype="int8"),
            "tod_afternoon"         : np.zeros(n, dtype="int8"),
            "tod_evening"           : np.zeros(n, dtype="int8"),
            "tod_night"             : np.zeros(n, dtype="int8"),
        })

    # ── secs_elapsed aggregate ────────────────────────────────────────────────
    if "secs_elapsed" in df.columns:
        df["secs_elapsed"] = df["secs_elapsed"].astype("float32")
        secs_agg = grp["secs_elapsed"].agg(
            total_secs_elapsed="sum",
            avg_secs_elapsed="mean",
        ).reset_index()
    else:
        secs_agg = None

    # ── Action entropy  [U-06] ────────────────────────────────────────────────
    entropy_series = _action_entropy_vectorised(df)

    # ── Combine all aggregations ──────────────────────────────────────────────
    result = (
        agg_basic
        .merge(top_action,    on="user_id", how="left")
        .merge(first_last,    on="user_id", how="left")
        .merge(most_used_dev, on="user_id", how="left")
        .merge(time_agg,      on="user_id", how="left")
    )
    if secs_agg is not None:
        result = result.merge(secs_agg, on="user_id", how="left")

    result = result.merge(
        entropy_series.reset_index(), on="user_id", how="left"
    )

    # ── Derived ratio features ─────────────────────────────────────────────────
    # [U-12] np.maximum() for numerical stability (no +1 inflation at large N)
    result["actions_per_second"] = (
        result["total_actions"] /
        np.maximum(result["session_duration"].fillna(0), 1)
    ).astype("float32")

    result["action_diversity_ratio"] = (
        result["unique_actions"] / (result["total_actions"] + 1)
    ).astype("float32")

    # [U-07] device_diversity_ratio: unique devices / total actions
    if "unique_devices" in result.columns:
        result["device_diversity_ratio"] = (
            result["unique_devices"] /
            np.maximum(result["total_actions"], 1)
        ).astype("float32")

    # high_activity_flag: binary power-user signal
    result["high_activity_flag"] = (result["total_actions"] > 50).astype("int8")

    elapsed = time.time() - t0
    log.info("         → aggregated shape: %s  (%.1fs)", result.shape, elapsed)
    return result


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — MERGE
# ══════════════════════════════════════════════════════════════════════════════

def merge_data(users: pd.DataFrame, sessions_agg: pd.DataFrame) -> pd.DataFrame:
    """LEFT JOIN users ← sessions_agg on id = user_id."""
    log.info("[STEP 5] Merging users with session features …")

    # [U-11] guard: both sides non-empty
    if users.empty:
        raise ValueError("users DataFrame is empty — cannot merge.")

    merged = users.merge(
        sessions_agg,
        left_on="id",
        right_on="user_id",
        how="left",
    )
    if "user_id" in merged.columns:
        merged.drop(columns=["user_id"], inplace=True)

    log.info("         → merged shape: %s", merged.shape)
    return merged


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 — HANDLE USERS WITH NO SESSION DATA
# ══════════════════════════════════════════════════════════════════════════════

def handle_no_session(df: pd.DataFrame, session_cols: List[str]) -> pd.DataFrame:
    """
    Flag and fill users with no session rows.

    v1 used df.copy() on the already-merged ~200 k-row DataFrame.
    v2 operates inplace where safe to keep peak RAM low.  [U-08]
    """
    log.info("[STEP 6] Handling users with no session data …")

    # Identify session-less users before filling (total_actions NaN → no join)
    df["no_session"] = df["total_actions"].isna().astype("int8")
    n_no_session = int(df["no_session"].sum())
    log.info("         Users with no session: %d", n_no_session)

    # Split session cols into numeric vs categorical
    existing_sess_cols = [c for c in session_cols if c in df.columns]
    num_sess  = df[existing_sess_cols].select_dtypes(include=[np.number]).columns.tolist()
    cat_sess  = [c for c in existing_sess_cols if c not in num_sess]

    df[num_sess] = df[num_sess].fillna(0)
    df[cat_sess] = df[cat_sess].fillna("None")

    # Downcast numeric session features to float32 / int32  [memory]
    for col in num_sess:
        if col in df.columns:
            if df[col].dtype == np.float64:
                df[col] = df[col].astype("float32")
            elif df[col].dtype == np.int64:
                df[col] = df[col].astype("int32")

    log.info("         → shape after no-session fill: %s", df.shape)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# STEP 7 — ENCODING
# ══════════════════════════════════════════════════════════════════════════════

def _build_frequency_map(series: pd.Series) -> Dict[str, float]:
    """
    Build a normalised frequency map (value → frequency ratio) from a Series.
    Used for high-cardinality columns.  [U-05]
    """
    freq = series.value_counts(normalize=True).to_dict()
    return freq


def _apply_frequency_encoding(
    df: pd.DataFrame,
    col: str,
    freq_map: Dict[str, float],
) -> pd.Series:
    """
    Map values to their training-set frequency.
    Unseen values (at inference time) map to 0.0 — never crashes.  [U-01, U-05]
    """
    return df[col].astype(str).map(freq_map).fillna(0.0).astype("float32")


def fit_encoders(df: pd.DataFrame) -> Tuple[Dict, Dict]:
    """
    Learn all encoding mappings from the TRAINING dataframe.  [U-02]

    Returns
    -------
    ordinal_encoders : col → fitted OrdinalEncoder
    freq_maps        : col → {value: frequency}
    """
    log.info("[STEP 7-FIT] Learning encoders from training data …")

    # ── OrdinalEncoder for low-cardinality columns  [U-01] ───────────────────
    ordinal_encoders: Dict[str, OrdinalEncoder] = {}
    for col in ORDINAL_ENCODE_COLS:
        if col not in df.columns or col == TARGET_COLUMN:
            continue
        df[col] = df[col].astype(str)
        enc = OrdinalEncoder(
            handle_unknown="use_encoded_value",   # [U-01] returns -1 for unseen
            unknown_value=-1,
            dtype=np.float32,
        )
        enc.fit(df[[col]])
        ordinal_encoders[col] = enc

    # ── Frequency maps for high-cardinality columns  [U-05] ──────────────────
    freq_maps: Dict[str, Dict] = {}
    for col in FREQ_ENCODE_COLS:
        if col not in df.columns or col == TARGET_COLUMN:
            continue
        freq_maps[col] = _build_frequency_map(df[col].astype(str))

    log.info("         OrdinalEncoder fitted for %d columns: %s",
             len(ordinal_encoders), list(ordinal_encoders.keys()))
    log.info("         Frequency maps built for %d columns: %s",
             len(freq_maps), list(freq_maps.keys()))
    return ordinal_encoders, freq_maps


def apply_encoders(
    df: pd.DataFrame,
    ordinal_encoders: Dict,
    freq_maps: Dict,
) -> pd.DataFrame:
    """
    Apply pre-fitted encoders to df.  Safe for inference data.  [U-02]

    [U-03] country_destination is NEVER touched — excluded explicitly.
    [U-01] OrdinalEncoder returns -1 for any category unseen at fit time.
    [U-05] Frequency encoder returns 0.0 for unseen categories.
    """
    log.info("[STEP 7-TRANSFORM] Applying encoders …")

    # ── OrdinalEncoder ────────────────────────────────────────────────────────
    for col, enc in ordinal_encoders.items():
        if col not in df.columns:
            log.warning("         Column '%s' missing at transform — skipping.", col)
            continue
        df[col] = enc.transform(df[[col]].astype(str)).flatten()

    # ── Frequency encoding ────────────────────────────────────────────────────
    for col, freq_map in freq_maps.items():
        if col not in df.columns:
            log.warning("         Column '%s' missing at transform — skipping.", col)
            continue
        df[col] = _apply_frequency_encoding(df, col, freq_map)

    # ── Safety: never encode target  [U-03] ──────────────────────────────────
    assert TARGET_COLUMN not in ordinal_encoders, \
        f"BUG: '{TARGET_COLUMN}' found in ordinal_encoders — remove it."
    assert TARGET_COLUMN not in freq_maps, \
        f"BUG: '{TARGET_COLUMN}' found in freq_maps — remove it."

    log.info("         Encoding complete. '%s' left as original string labels.",
             TARGET_COLUMN)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# STEP 8 — FINAL VALIDATION & SAVE
# ══════════════════════════════════════════════════════════════════════════════

def validate_and_save(
    df: pd.DataFrame,
    output_path: str,
    feature_list: List[str],
) -> pd.DataFrame:
    """
    Validates, sorts deterministically, and saves.  [U-10, U-14, U-16]
    """
    log.info("[STEP 8] Final validation …")

    # [U-04] Hard leakage check — abort if any forbidden column slipped through
    _assert_no_leakage(df)

    # [U-03] Confirm target is present as string and not encoded
    if TARGET_COLUMN in df.columns:
        assert df[TARGET_COLUMN].dtype == object, \
            f"'{TARGET_COLUMN}' dtype is {df[TARGET_COLUMN].dtype} — expected object (string)."
        log.info("         ✓ '%s' dtype is object (strings preserved).", TARGET_COLUMN)

    # ── Missing value check and fill ─────────────────────────────────────────
    missing = df.isnull().sum()
    remaining = missing[missing > 0]
    if not remaining.empty:
        pct    = (remaining / len(df) * 100).round(2)
        report = pd.DataFrame({"count": remaining, "%": pct})
        log.warning("Columns with NaN (auto-filling):\n%s", report.to_string())
        df[df.select_dtypes(include=[np.number]).columns] = \
            df.select_dtypes(include=[np.number]).fillna(0)
        df[df.select_dtypes(include=["object"]).columns] = \
            df.select_dtypes(include=["object"]).fillna("None")

    _assert_no_nulls(df, "post-preprocessing")   # [U-10]

    # ── Row uniqueness ────────────────────────────────────────────────────────
    total_rows   = len(df)
    unique_users = df["id"].nunique()
    assert unique_users == total_rows, \
        f"Duplicate IDs detected: {total_rows - unique_users:,} duplicates."
    log.info("         ✓ One row per user confirmed (%d users).", total_rows)

    # ── Deterministic sort  [U-14] ────────────────────────────────────────────
    df.sort_values("id", inplace=True, kind="mergesort")
    df.reset_index(drop=True, inplace=True)

    # ── Feature summary  [U-16] ──────────────────────────────────────────────
    n_features = len(feature_list)
    log.info("         Final feature count: %d", n_features)
    numeric_desc = df[feature_list].select_dtypes(include=[np.number]).describe().T
    log.info("         Numeric feature stats (first 10):\n%s",
             numeric_desc.head(10).to_string())

    # ── Save ──────────────────────────────────────────────────────────────────
    df.to_csv(output_path, index=False)
    log.info("         ✓ Saved → '%s'  (%d rows × %d cols)",
             output_path, df.shape[0], df.shape[1])
    return df


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE CLASS  [U-02]  fit() / transform() interface
# ══════════════════════════════════════════════════════════════════════════════

class AirbnbPipeline:
    """
    Stateful pipeline with a sklearn-style fit / transform API.

    Usage (training)
    ----------------
        pipe = AirbnbPipeline()
        result = pipe.fit_transform(
            users_path="train_users.csv",
            sessions_path="sessions.csv",
        )

    Usage (inference)
    -----------------
        pipe = AirbnbPipeline.load("pipeline_artifacts.pkl")
        X_new = pipe.transform(new_users_df, new_sessions_df)
    """

    def __init__(self) -> None:
        self.age_median_       : Optional[float]       = None
        self.ordinal_encoders_ : Optional[Dict]        = None
        self.freq_maps_        : Optional[Dict]        = None
        self.feature_list_     : Optional[List[str]]   = None
        self._is_fitted        : bool                  = False

    # ── Public API ────────────────────────────────────────────────────────────

    def fit(
        self,
        users_path:    str = TRAIN_USERS_PATH,
        sessions_path: str = SESSIONS_PATH,
    ) -> "AirbnbPipeline":
        """Learn statistics & encoders from training data."""
        t_start = time.time()
        self._print_banner("FIT")

        users    = load_users(users_path)
        sessions = load_sessions(sessions_path)

        users_clean, self.age_median_ = clean_users(users)
        del users; gc.collect()   # [U-08]

        sessions_clean = clean_sessions(sessions)
        del sessions;  gc.collect()

        sessions_agg = aggregate_sessions(sessions_clean)
        session_feature_cols = [c for c in sessions_agg.columns if c != "user_id"]
        del sessions_clean; gc.collect()

        merged = merge_data(users_clean, sessions_agg)
        merged = handle_no_session(merged, session_feature_cols)

        # Learn encoders from training data ONLY  [U-02]
        self.ordinal_encoders_, self.freq_maps_ = fit_encoders(merged)

        self._is_fitted = True
        log.info("Pipeline fitted in %.1fs", time.time() - t_start)
        return self

    def transform(
        self,
        users_df:    pd.DataFrame,
        sessions_df: pd.DataFrame,
        output_path: str = OUTPUT_PATH,
    ) -> PipelineResult:
        """
        Apply fitted transformations.  Safe for inference data.  [U-02]
        Raises RuntimeError if fit() has not been called first.
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before transform().")

        t_start = time.time()
        self._print_banner("TRANSFORM")

        # Clean — use TRAINING age_median (not test-set)  [U-02]
        users_clean, _ = clean_users(users_df, age_median=self.age_median_)
        sessions_clean = clean_sessions(sessions_df)
        sessions_agg   = aggregate_sessions(sessions_clean)
        session_feature_cols = [c for c in sessions_agg.columns if c != "user_id"]
        del sessions_clean; gc.collect()

        merged  = merge_data(users_clean, sessions_agg)
        merged  = handle_no_session(merged, session_feature_cols)
        encoded = apply_encoders(merged, self.ordinal_encoders_, self.freq_maps_)

        # Feature list: everything except id and target  [U-16]
        self.feature_list_ = [
            c for c in encoded.columns
            if c not in {"id", TARGET_COLUMN}
        ]

        final = validate_and_save(encoded, output_path, self.feature_list_)

        summary = self._build_summary(final)

        elapsed = time.time() - t_start
        log.info("Pipeline transform complete in %.1fs", elapsed)
        log.info("=" * 70)

        return PipelineResult(
            df           = final,
            feature_list = self.feature_list_,
            artifacts    = self._artifact_bundle(),
            summary      = summary,
        )

    def fit_transform(
        self,
        users_path:    str = TRAIN_USERS_PATH,
        sessions_path: str = SESSIONS_PATH,
        output_path:   str = OUTPUT_PATH,
    ) -> PipelineResult:
        """Convenience: fit on and transform the SAME dataset (training path)."""
        self.fit(users_path, sessions_path)

        users_df    = load_users(users_path)
        sessions_df = load_sessions(sessions_path)
        result      = self.transform(users_df, sessions_df, output_path)

        # Persist all artifacts  [U-15]
        joblib.dump(result.artifacts, ARTIFACTS_PATH)
        log.info("✓ Artifacts saved → '%s'", ARTIFACTS_PATH)
        return result

    def save(self, path: str = ARTIFACTS_PATH) -> None:
        """Persist fitted pipeline to disk."""
        joblib.dump(self._artifact_bundle(), path)
        log.info("Pipeline saved → '%s'", path)

    @classmethod
    def load(cls, path: str = ARTIFACTS_PATH) -> "AirbnbPipeline":
        """Restore a fitted pipeline from disk."""
        bundle = joblib.load(path)
        pipe   = cls()
        pipe.age_median_       = bundle["age_median"]
        pipe.ordinal_encoders_ = bundle["ordinal_encoders"]
        pipe.freq_maps_        = bundle["freq_maps"]
        pipe.feature_list_     = bundle["feature_list"]
        pipe._is_fitted        = True
        log.info("Pipeline loaded from '%s'", path)
        return pipe

    # ── Internals ─────────────────────────────────────────────────────────────

    def _artifact_bundle(self) -> Dict:
        """All fitted state in one serialisable dict.  [U-15]"""
        return {
            "age_median"       : self.age_median_,
            "ordinal_encoders" : self.ordinal_encoders_,
            "freq_maps"        : self.freq_maps_,
            "feature_list"     : self.feature_list_,
        }

    @staticmethod
    def _print_banner(mode: str) -> None:
        log.info("=" * 70)
        log.info("  AIRBNB NEW USER BOOKINGS — ML PIPELINE  [%s]", mode)
        log.info("=" * 70)

    @staticmethod
    def _build_summary(df: pd.DataFrame) -> str:
        """Human-readable summary for reporting.  [U-16]"""
        lines = [
            "=" * 60,
            "  PIPELINE SUMMARY",
            "=" * 60,
            f"  Rows          : {len(df):,}",
            f"  Columns       : {df.shape[1]}",
            f"  Target present: {TARGET_COLUMN in df.columns}",
            f"  Target dtype  : {df[TARGET_COLUMN].dtype if TARGET_COLUMN in df.columns else 'N/A'}",
            f"  Leakage cols  : {list(LEAKAGE_COLUMNS & set(df.columns))} (should be empty)",
            "",
            "  Numeric feature ranges (sample):",
        ]
        num_cols = df.select_dtypes(include=[np.number]).columns[:8]
        for col in num_cols:
            lines.append(
                f"    {col:<35} min={df[col].min():.3f}  max={df[col].max():.3f}"
            )
        lines.append("=" * 60)
        summary = "\n".join(lines)
        print(summary)
        return summary


# ══════════════════════════════════════════════════════════════════════════════
# BACKWARDS-COMPATIBLE run_pipeline() WRAPPER
# ══════════════════════════════════════════════════════════════════════════════

def run_pipeline(
    users_path:    str = TRAIN_USERS_PATH,
    sessions_path: str = SESSIONS_PATH,
    output_path:   str = OUTPUT_PATH,
) -> PipelineResult:
    """
    Drop-in replacement for v1 run_pipeline().
    Internally delegates to AirbnbPipeline.fit_transform().  [U-16]
    """
    pipe = AirbnbPipeline()
    return pipe.fit_transform(users_path, sessions_path, output_path)


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # ── Training run ─────────────────────────────────────────────────────────
    result = run_pipeline(
        users_path    = TRAIN_USERS_PATH,
        sessions_path = SESSIONS_PATH,
        output_path   = OUTPUT_PATH,
    )
    print(f"\nFeature list ({len(result.feature_list)} features):")
    print(result.feature_list)
    print("\nSample output (first 3 rows):")
    print(result.df.head(3).to_string())

    # ── Inference example ────────────────────────────────────────────────────
    # pipe   = AirbnbPipeline.load(ARTIFACTS_PATH)
    # result = pipe.transform(new_users_df, new_sessions_df, "inference_out.csv")
