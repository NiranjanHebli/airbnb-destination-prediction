TRAIN_USERS_PATH      = "../data/raw/train_users_2.csv"
SESSIONS_PATH         = "../data/raw/sessions.csv"
OUTPUT_PATH           = "../data/processed/final_processed_data.csv"
ARTIFACTS_PATH        = "../models/pipeline_artifacts.pkl"

SESSION_TIMESTAMP_COL = "timestamp"

LEAKAGE_COLUMNS = {"date_first_booking", "timestamp_first_active"}
TARGET_COLUMN   = "country_destination"

FREQ_ENCODE_COLS = ["top_action", "first_action", "last_action", "most_used_device"]

ORDINAL_ENCODE_COLS = [
    "gender", "signup_method", "signup_flow", "language",
    "affiliate_channel", "affiliate_provider", "first_affiliate_tracked",
    "signup_app", "first_device_type", "first_browser",
]

REQUIRED_USER_COLS    = {"id", "age", "gender"}
REQUIRED_SESSION_COLS = {"user_id", "action"}
