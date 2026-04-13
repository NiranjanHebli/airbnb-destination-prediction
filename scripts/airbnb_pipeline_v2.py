
import gc
import logging
import time
from collections import namedtuple
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

# Import modular components
from config import (
    TRAIN_USERS_PATH, SESSIONS_PATH, OUTPUT_PATH, ARTIFACTS_PATH,
    TARGET_COLUMN, LEAKAGE_COLUMNS, REQUIRED_USER_COLS, REQUIRED_SESSION_COLS
)
from data_validator import DataValidator
from data_loader import DataLoader
from data_cleaner import DataCleaner
from session_aggregator import SessionAggregator
from data_merger import DataMerger
from feature_encoder import FeatureEncoder

# Setup Logging
log = logging.getLogger("airbnb_pipeline")
if not log.hasHandlers():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

PipelineResult = namedtuple(
    "PipelineResult", ["df", "feature_list", "artifacts", "summary"]
)

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
        self.encoder_          : FeatureEncoder        = FeatureEncoder()
        self.feature_list_     : Optional[List[str]]   = None
        self._is_fitted        : bool                  = False

    def fit(
        self,
        users_path:    str = TRAIN_USERS_PATH,
        sessions_path: str = SESSIONS_PATH,
    ) -> "AirbnbPipeline":
        """Learn statistics & encoders from training data."""
        t_start = time.time()
        self._print_banner("FIT")

        users    = DataLoader.load_users(users_path, REQUIRED_USER_COLS)
        sessions = DataLoader.load_sessions(sessions_path, REQUIRED_SESSION_COLS)

        users_clean, self.age_median_ = DataCleaner.clean_users(users)
        del users; gc.collect()

        sessions_clean = DataCleaner.clean_sessions(sessions)
        del sessions;  gc.collect()

        sessions_agg = SessionAggregator.aggregate_sessions(sessions_clean)
        session_feature_cols = [c for c in sessions_agg.columns if c != "user_id"]
        del sessions_clean; gc.collect()

        merged = DataMerger.merge_data(users_clean, sessions_agg)
        merged = DataMerger.handle_no_session(merged, session_feature_cols)

        self.encoder_.fit(merged)

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
        Apply fitted transformations.  Safe for inference data.
        Raises RuntimeError if fit() has not been called first.
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before transform().")

        t_start = time.time()
        self._print_banner("TRANSFORM")

        # Clean — use TRAINING age_median (not test-set)
        users_clean, _ = DataCleaner.clean_users(users_df, age_median=self.age_median_)
        sessions_clean = DataCleaner.clean_sessions(sessions_df)
        sessions_agg   = SessionAggregator.aggregate_sessions(sessions_clean)
        session_feature_cols = [c for c in sessions_agg.columns if c != "user_id"]
        del sessions_clean; gc.collect()

        merged  = DataMerger.merge_data(users_clean, sessions_agg)
        merged  = DataMerger.handle_no_session(merged, session_feature_cols)
        encoded = self.encoder_.transform(merged)

        # Feature list: everything except id and target
        self.feature_list_ = [
            c for c in encoded.columns
            if c not in {"id", TARGET_COLUMN}
        ]

        final = self._validate_and_save(encoded, output_path, self.feature_list_)
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

    def _validate_and_save(
        self,
        df: pd.DataFrame,
        output_path: str,
        feature_list: List[str],
    ) -> pd.DataFrame:
        """Validates, sorts deterministically, and saves."""
        log.info("[STEP 8] Final validation …")

        DataValidator.assert_no_leakage(df, LEAKAGE_COLUMNS)

        if TARGET_COLUMN in df.columns:
            assert df[TARGET_COLUMN].dtype == object, \
                f"'{TARGET_COLUMN}' dtype is {df[TARGET_COLUMN].dtype} — expected object (string)."
            log.info("         ✓ '%s' dtype is object (strings preserved).", TARGET_COLUMN)

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

        DataValidator.assert_no_nulls(df, "post-preprocessing")

        total_rows   = len(df)
        unique_users = df["id"].nunique()
        assert unique_users == total_rows, \
            f"Duplicate IDs detected: {total_rows - unique_users:,} duplicates."
        log.info("         ✓ One row per user confirmed (%d users).", total_rows)

        df.sort_values("id", inplace=True, kind="mergesort")
        df.reset_index(drop=True, inplace=True)

        n_features = len(feature_list)
        log.info("         Final feature count: %d", n_features)
        numeric_desc = df[feature_list].select_dtypes(include=[np.number]).describe().T
        log.info("         Numeric feature stats (first 10):\n%s",
                 numeric_desc.head(10).to_string())

        df.to_csv(output_path, index=False)
        log.info("         ✓ Saved → '%s'  (%d rows × %d cols)",
                 output_path, df.shape[0], df.shape[1])
        return df

    def fit_transform(
        self,
        users_path:    str = TRAIN_USERS_PATH,
        sessions_path: str = SESSIONS_PATH,
        output_path:   str = OUTPUT_PATH,
    ) -> PipelineResult:
        """Convenience: fit on and transform the SAME dataset (training path)."""
        self.fit(users_path, sessions_path)

        users_df    = DataLoader.load_users(users_path, REQUIRED_USER_COLS)
        sessions_df = DataLoader.load_sessions(sessions_path, REQUIRED_SESSION_COLS)
        result      = self.transform(users_df, sessions_df, output_path)

        # Persist all artifacts
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
        pipe.encoder_.set_state(bundle["ordinal_encoders"], bundle["freq_maps"])
        pipe.feature_list_     = bundle["feature_list"]
        pipe._is_fitted        = True
        log.info("Pipeline loaded from '%s'", path)
        return pipe

    def _artifact_bundle(self) -> Dict:
        """All fitted state in one serialisable dict."""
        oe, fm = self.encoder_.get_state()
        return {
            "age_median"       : self.age_median_,
            "ordinal_encoders" : oe,
            "freq_maps"        : fm,
            "feature_list"     : self.feature_list_,
        }

    @staticmethod
    def _print_banner(mode: str) -> None:
        log.info("=" * 70)
        log.info("  AIRBNB NEW USER BOOKINGS — ML PIPELINE  [%s]", mode)
        log.info("=" * 70)

    @staticmethod
    def _build_summary(df: pd.DataFrame) -> str:
        """Human-readable summary for reporting."""
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


def run_pipeline(
    users_path:    str = TRAIN_USERS_PATH,
    sessions_path: str = SESSIONS_PATH,
    output_path:   str = OUTPUT_PATH,
) -> PipelineResult:
    """Drop-in replacement for v1 run_pipeline()."""
    pipe = AirbnbPipeline()
    return pipe.fit_transform(users_path, sessions_path, output_path)

if __name__ == "__main__":
    result = run_pipeline(
        users_path    = TRAIN_USERS_PATH,
        sessions_path = SESSIONS_PATH,
        output_path   = OUTPUT_PATH,
    )
    print(f"\nFeature list ({len(result.feature_list)} features):")
    print(result.feature_list)
    print("\nSample output (first 3 rows):")
    print(result.df.head(3).to_string())
