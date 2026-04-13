import logging
import numpy as np
import pandas as pd
from typing import List

log = logging.getLogger("airbnb_pipeline")

class DataMerger:
    """Class containing static methods for merging user and session data."""

    @staticmethod
    def merge_data(users: pd.DataFrame, sessions_agg: pd.DataFrame) -> pd.DataFrame:
        """LEFT JOIN users ← sessions_agg on id = user_id."""
        log.info("[STEP 5] Merging users with session features …")

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

    @staticmethod
    def handle_no_session(df: pd.DataFrame, session_cols: List[str]) -> pd.DataFrame:
        """Flag and fill users with no session rows."""
        log.info("[STEP 6] Handling users with no session data …")

        df["no_session"] = df["total_actions"].isna().astype("int8")
        n_no_session = int(df["no_session"].sum())
        log.info("         Users with no session: %d", n_no_session)

        existing_sess_cols = [c for c in session_cols if c in df.columns]
        num_sess  = df[existing_sess_cols].select_dtypes(include=[np.number]).columns.tolist()
        cat_sess  = [c for c in existing_sess_cols if c not in num_sess]

        df[num_sess] = df[num_sess].fillna(0)
        df[cat_sess] = df[cat_sess].fillna("None")

        for col in num_sess:
            if col in df.columns:
                if df[col].dtype == np.float64:
                    df[col] = df[col].astype("float32")
                elif df[col].dtype == np.int64:
                    df[col] = df[col].astype("int32")

        log.info("         → shape after no-session fill: %s", df.shape)
        return df
