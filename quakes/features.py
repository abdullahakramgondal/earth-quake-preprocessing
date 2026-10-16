"""Task 4: feature engineering."""
import pandas as pd

from quakes import config


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add 'hour' and 'dayofweek' (UTC) from 'time'."""
    out = df.copy()
    out["hour"] = out["time"].dt.hour
    out["dayofweek"] = out["time"].dt.dayofweek
    return out


def add_quality_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add:
    update_lag_hours = hours between 'time' and 'updated'
    is_reviewed      = 1 if status == 'reviewed' else 0
    nst_missing      = 1 if 'nst' is missing else 0
    """
    out = df.copy()
    out["update_lag_hours"] = (out["updated"] - out["time"]).dt.total_seconds() / 3600
    out["is_reviewed"] = (out["status"] == "reviewed").astype(int)
    out["nst_missing"] = out["nst"].isna().astype(int)
    return out


def add_location_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add 'abs_lat' and 'is_shallow' (1 if depth_km < 70 else 0)."""
    out = df.copy()
    out["abs_lat"] = out["lat"].abs()
    out["is_shallow"] = (out["depth_km"] < 70).astype(int)
    return out


def group_rare(s: pd.Series, top_k: int = 15) -> pd.Series:
    """Keep the top_k most frequent values; replace all others with 'Other'."""
    top = s.value_counts().nlargest(top_k).index
    return s.where(s.isin(top), "Other")


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """Add 'big_quake' = 1 if mag >= 4.5 else 0."""
    out = df.copy()
    out["big_quake"] = (out["mag"] >= 4.5).astype(int)
    return out


def drop_leaky_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop every column listed in config.LEAKY (ignore ones that are absent)."""
    return df.drop(columns=[c for c in config.LEAKY if c in df.columns])