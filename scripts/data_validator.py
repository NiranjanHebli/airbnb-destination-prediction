import logging
import pandas as pd

log = logging.getLogger("airbnb_pipeline")

class DataValidator:
    """Class containing static methods for dataframe validation."""
    
    @staticmethod
    def validate_dataframe(df: pd.DataFrame, required_cols: set, name: str) -> None:
        """Raise ValueError if required columns are missing or df is empty."""
        if df.empty:
            raise ValueError(f"[VALIDATION] '{name}' DataFrame is empty — aborting.")
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(
                f"[VALIDATION] '{name}' is missing required columns: {missing}"
            )
        log.info("Schema OK for '%s'  (%d rows × %d cols)", name, *df.shape)

    @staticmethod
    def assert_no_leakage(df: pd.DataFrame, leakage_columns: set) -> None:
        """Hard-fail if any known leakage column is present."""
        found = leakage_columns & set(df.columns)
        assert not found, (
            f"DATA LEAKAGE DETECTED — remove columns before training: {found}"
        )
        log.info("Leakage check PASSED — no forbidden columns detected.")

    @staticmethod
    def assert_no_nulls(df: pd.DataFrame, context: str = "") -> None:
        """Hard-fail if any NaN remains."""
        n = df.isnull().sum().sum()
        assert n == 0, f"FATAL: {n:,} NaN values remain after preprocessing ({context})"
        log.info("Null check PASSED — zero NaNs (%s).", context)
