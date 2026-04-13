import logging
import numpy as np
import pandas as pd
from typing import Optional, Tuple
from config import LEAKAGE_COLUMNS, TARGET_COLUMN, SESSION_TIMESTAMP_COL

log = logging.getLogger("airbnb_pipeline")

class DataCleaner:
    """Class containing static methods for data cleaning."""

    @staticmethod
    def clean_users(df: pd.DataFrame, age_median: Optional[float] = None) -> Tuple[pd.DataFrame, float]:
        """
        Clean user features. Returns (cleaned_df, age_median_used).
        """
        log.info("[STEP 2] Cleaning user data …")

        df = df.copy()

        # ID normalisation
        df["id"] = df["id"].astype(str).str.strip()

        # Leakage columns
        drop_early = list(LEAKAGE_COLUMNS) + [TARGET_COLUMN]
        target_series = df[TARGET_COLUMN].copy() if TARGET_COLUMN in df.columns else None
        cols_to_drop  = [c for c in drop_early if c in df.columns and c != TARGET_COLUMN]
        if cols_to_drop:
            df.drop(columns=cols_to_drop, inplace=True)
            log.info("         Dropped leakage/redundant columns: %s", cols_to_drop)

        # Age
        df["age"] = pd.to_numeric(df["age"], errors="coerce")
        df.loc[(df["age"] < 15) | (df["age"] > 90), "age"] = np.nan

        if age_median is None:
            age_median = float(df["age"].median())
        df["age"] = df["age"].fillna(age_median).astype("float32")
        log.info("         age → clipped & imputed; median=%.1f", age_median)

        # Gender / categoricals
        if "gender" in df.columns:
            df["gender"] = df["gender"].replace("-unknown-", "Unknown").fillna("Unknown")

        for col in ["first_browser", "language", "affiliate_channel",
                    "signup_method", "affiliate_provider", "first_affiliate_tracked",
                    "signup_app", "first_device_type"]:
            if col in df.columns:
                df[col] = (df[col].fillna("Unknown")
                                   .replace("-unknown-", "Unknown")
                                   .astype(str))

        # Date account created
        date_col = "date_account_created"
        if date_col in df.columns:
            dt = pd.to_datetime(df[date_col], errors="coerce")
            df["dac_year"]    = dt.dt.year.astype("Int16")
            df["dac_month"]   = dt.dt.month.astype("Int8")
            df["dac_day"]     = dt.dt.day.astype("Int8")
            df["dac_weekday"] = dt.dt.dayofweek.astype("Int8")
            df["dac_quarter"] = dt.dt.quarter.astype("Int8")
            df.drop(columns=[date_col], inplace=True)
            log.info("         '%s' expanded to year/month/day/weekday/quarter", date_col)

        # Re-attach target
        if target_series is not None:
            df[TARGET_COLUMN] = target_series.values

        log.info("         → users shape after cleaning: %s", df.shape)
        return df, age_median

    @staticmethod
    def clean_sessions(df: pd.DataFrame) -> pd.DataFrame:
        """
        Fills NaN, drops duplicates, parses timestamps, sorts.
        """
        log.info("[STEP 3] Cleaning session data …")

        if df.empty:
            log.warning("         Sessions DataFrame is empty — returning as-is.")
            return df

        # Fill missing
        cat_cols = df.select_dtypes(include=["category"]).columns.tolist()
        if cat_cols:
            df[cat_cols] = df[cat_cols].astype(str)

        obj_cols = df.select_dtypes(include=["object"]).columns.tolist()
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        df[obj_cols] = df[obj_cols].fillna("Unknown").replace("nan", "Unknown")
        df[num_cols] = df[num_cols].fillna(0)

        # Remove duplicates
        before = len(df)
        df.drop_duplicates(inplace=True)
        log.info("         Duplicates removed: %d", before - len(df))

        # Timestamp parsing + deterministic sort
        has_ts = SESSION_TIMESTAMP_COL and SESSION_TIMESTAMP_COL in df.columns
        if has_ts:
            df[SESSION_TIMESTAMP_COL] = pd.to_datetime(
                df[SESSION_TIMESTAMP_COL], errors="coerce"
            )
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
