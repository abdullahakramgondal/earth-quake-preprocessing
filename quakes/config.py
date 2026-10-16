from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FALLBACK_PATH = ROOT / "data" / "fallback" / "all_week.geojson"
STREAM_DIR = ROOT / "data" / "stream"
PROCESSED_DIR = ROOT / "data" / "processed"

SEED = 42
TARGET = "big_quake"

# TODO (Task 4 and 5): fill these in after you have explored the data.
NUMERIC: list[str] = ["lon", "lat", "depth_km", "nst", "dmin", "rms", "gap",
                      "hour", "dayofweek", "update_lag_hours", "is_reviewed",
                      "nst_missing", "abs_lat", "is_shallow"]   # numeric feature columns
NOMINAL: list[str] = ["region", "magType"]   # categorical feature columns (one-hot encoded)
LEAKY: list[str] = ["title", "sig", "mmi", "cdi", "felt", "alert", "tsunami"]     # columns that encode the magnitude: must be dropped
