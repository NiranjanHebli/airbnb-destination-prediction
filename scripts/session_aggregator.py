import gc
import logging
import time
import numpy as np
import pandas as pd
from config import SESSION_TIMESTAMP_COL

log = logging.getLogger("airbnb_pipeline")

class SessionAggregator:
    """Class containing static methods for aggregating session data."""

    @staticmethod
    def _action_entropy_vectorised(df: pd.DataFrame) -> pd.Series:
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

    @staticmethod
    def aggregate_sessions(df: pd.DataFrame) -> pd.DataFrame:
        log.info("[STEP 4] Aggregating session data …")
        t0 = time.time()

        if df.empty or "user_id" not in df.columns:
            log.warning("         Empty/invalid sessions — returning empty aggregation.")
            return pd.DataFrame(columns=["user_id"])

        grp = df.groupby("user_id", sort=False, observed=True)

        # Basic counts
        agg_kwargs = {
            "total_actions" : ("action", "count"),
            "unique_actions": ("action", "nunique"),
        }
        if "action_type" in df.columns:
            agg_kwargs["unique_action_types"] = ("action_type", "nunique")
        if "device_type" in df.columns:
            agg_kwargs["unique_devices"] = ("device_type", "nunique")

        agg_basic = grp.agg(**agg_kwargs).reset_index()

        # top_action
        if "action" in df.columns:
            action_counts = (
                df.groupby(["user_id", "action"], sort=False, observed=True)
                .size()
                .rename("cnt")
                .reset_index()
            )
            top_idx = action_counts.groupby("user_id")["cnt"].idxmax()
            top_action = (
                action_counts.loc[top_idx, ["user_id", "action"]]
                .rename(columns={"action": "top_action"})
            )
        else:
            top_action = pd.DataFrame({"user_id": df["user_id"].unique(),
                                       "top_action": "Unknown"})

        # first / last action
        first_last = grp["action"].agg(
            first_action="first",
            last_action="last",
        ).reset_index()

        # most used device
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

        # Time-based features
        has_ts = SESSION_TIMESTAMP_COL and SESSION_TIMESTAMP_COL in df.columns

        if has_ts:
            ts = SESSION_TIMESTAMP_COL

            time_agg = grp[ts].agg(
                session_start="min",
                session_end="max",
            ).reset_index()
            time_agg["session_duration"] = (
                (time_agg["session_end"] - time_agg["session_start"])
                .dt.total_seconds()
                .astype("float32")
            )

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

            global_max_ts = time_agg["session_end"].max()
            time_agg["recency_gap"] = (
                (global_max_ts - time_agg["session_end"])
                .dt.total_seconds()
                .astype("float32")
            )

            span_days = time_agg["session_duration"] / 86_400.0   # seconds → days
            time_agg["session_count_per_day"] = (
                agg_basic.set_index("user_id")["total_actions"]
                .reindex(time_agg["user_id"].values)
                .values
                / np.maximum(span_days, 1)
            ).astype("float32")

            hour = time_agg["session_start"].dt.hour
            time_agg["tod_morning"]   = ((hour >= 6)  & (hour < 12)).astype("int8")
            time_agg["tod_afternoon"] = ((hour >= 12) & (hour < 18)).astype("int8")
            time_agg["tod_evening"]   = ((hour >= 18) & (hour < 22)).astype("int8")
            time_agg["tod_night"]     = ((hour >= 22) | (hour < 6) ).astype("int8")

            time_agg["session_start"] = time_agg["session_start"].astype(str)
            time_agg["session_end"]   = time_agg["session_end"].astype(str)

        else:
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

        # secs_elapsed aggregate
        if "secs_elapsed" in df.columns:
            df["secs_elapsed"] = df["secs_elapsed"].astype("float32")
            secs_agg = grp["secs_elapsed"].agg(
                total_secs_elapsed="sum",
                avg_secs_elapsed="mean",
            ).reset_index()
        else:
            secs_agg = None

        # Action entropy
        entropy_series = SessionAggregator._action_entropy_vectorised(df)

        # Combine all aggregations
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

        # Derived ratio features
        result["actions_per_second"] = (
            result["total_actions"] /
            np.maximum(result["session_duration"].fillna(0), 1)
        ).astype("float32")

        result["action_diversity_ratio"] = (
            result["unique_actions"] / (result["total_actions"] + 1)
        ).astype("float32")

        if "unique_devices" in result.columns:
            result["device_diversity_ratio"] = (
                result["unique_devices"] /
                np.maximum(result["total_actions"], 1)
            ).astype("float32")

        result["high_activity_flag"] = (result["total_actions"] > 50).astype("int8")

        elapsed = time.time() - t0
        log.info("         → aggregated shape: %s  (%.1fs)", result.shape, elapsed)
        return result
