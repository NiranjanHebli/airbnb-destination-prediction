import gc
import logging
import pandas as pd
from typing import Set
from data_validator import DataValidator

log = logging.getLogger("airbnb_pipeline")

class DataLoader:
    """Class containing static methods for data loading."""

    @staticmethod
    def load_users(path: str, required_cols: Set[str]) -> pd.DataFrame:
        """
        Load train_users.csv with explicit low_memory=False.
        Validates schema immediately after load.
        """
        log.info("[STEP 1] Loading user data from '%s' …", path)
        df = pd.read_csv(path, low_memory=False)
        DataValidator.validate_dataframe(df, required_cols, "users")
        log.info("         Missing per column: %s",
                 df.isnull().sum()[df.isnull().sum() > 0].to_dict())
        return df

    @staticmethod
    def load_sessions(path: str, required_cols: Set[str], chunksize: int = 500_000) -> pd.DataFrame:
        """
        Load sessions.csv in chunks (handles 10 M+ rows without OOM).
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
        gc.collect()

        for col in cat_cols:
            if col in df.columns:
                df[col] = df[col].astype("category")

        DataValidator.validate_dataframe(df, required_cols, "sessions")
        log.info("         → %d total rows | %d columns", *df.shape)
        return df
