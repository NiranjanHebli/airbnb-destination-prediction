import logging
import numpy as np
import pandas as pd
from typing import Dict, Tuple
from sklearn.preprocessing import OrdinalEncoder
from config import ORDINAL_ENCODE_COLS, FREQ_ENCODE_COLS, TARGET_COLUMN

log = logging.getLogger("airbnb_pipeline")

class FeatureEncoder:
    """Class to manage and apply encodings for model features."""

    def __init__(self):
        self.ordinal_encoders_ = None
        self.freq_maps_ = None

    @staticmethod
    def _build_frequency_map(series: pd.Series) -> Dict[str, float]:
        """Build a normalised frequency map (value → frequency ratio)."""
        return series.value_counts(normalize=True).to_dict()

    @staticmethod
    def _apply_frequency_encoding(
        df: pd.DataFrame, col: str, freq_map: Dict[str, float]
    ) -> pd.Series:
        """Map values to their training-set frequency."""
        return df[col].astype(str).map(freq_map).fillna(0.0).astype("float32")

    def fit(self, df: pd.DataFrame) -> "FeatureEncoder":
        """Learn all encoding mappings from the TRAINING dataframe."""
        log.info("[STEP 7-FIT] Learning encoders from training data …")

        # OrdinalEncoder for low-cardinality columns
        self.ordinal_encoders_ = {}
        for col in ORDINAL_ENCODE_COLS:
            if col not in df.columns or col == TARGET_COLUMN:
                continue
            df[col] = df[col].astype(str)
            enc = OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1,
                dtype=np.float32,
            )
            enc.fit(df[[col]])
            self.ordinal_encoders_[col] = enc

        # Frequency maps for high-cardinality columns
        self.freq_maps_ = {}
        for col in FREQ_ENCODE_COLS:
            if col not in df.columns or col == TARGET_COLUMN:
                continue
            self.freq_maps_[col] = self._build_frequency_map(df[col].astype(str))

        log.info("         OrdinalEncoder fitted for %d columns", len(self.ordinal_encoders_))
        log.info("         Frequency maps built for %d columns", len(self.freq_maps_))
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply pre-fitted encoders to df."""
        log.info("[STEP 7-TRANSFORM] Applying encoders …")

        if self.ordinal_encoders_ is None or self.freq_maps_ is None:
            raise RuntimeError("FeatureEncoder must be fitted before transform.")

        # OrdinalEncoder
        for col, enc in self.ordinal_encoders_.items():
            if col not in df.columns:
                log.warning("         Column '%s' missing at transform — skipping.", col)
                continue
            df[col] = enc.transform(df[[col]].astype(str)).flatten()

        # Frequency encoding
        for col, freq_map in self.freq_maps_.items():
            if col not in df.columns:
                log.warning("         Column '%s' missing at transform — skipping.", col)
                continue
            df[col] = self._apply_frequency_encoding(df, col, freq_map)

        # Safety: never encode target
        assert TARGET_COLUMN not in self.ordinal_encoders_, \
            f"BUG: '{TARGET_COLUMN}' found in ordinal_encoders — remove it."
        assert TARGET_COLUMN not in self.freq_maps_, \
            f"BUG: '{TARGET_COLUMN}' found in freq_maps — remove it."

        log.info("         Encoding complete. '%s' left as original string labels.", TARGET_COLUMN)
        return df

    def set_state(self, ordinal_encoders: Dict, freq_maps: Dict):
        """Restore state from disk."""
        self.ordinal_encoders_ = ordinal_encoders
        self.freq_maps_ = freq_maps

    def get_state(self) -> Tuple[Dict, Dict]:
        """Extract state for serialisation."""
        return self.ordinal_encoders_, self.freq_maps_
