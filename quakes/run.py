"""Task 5: run the whole pipeline end to end.

Steps (see LAB_GUIDE.md):
 1. fetch all_week (fall back to config.FALLBACK_PATH if the network fails)
 2. merge any data/stream/*.jsonl rows, then dedupe_latest
 3. clean, engineer features, drop leaky columns
 4. split, fit_transform on train only, transform test
 5. print the report, save train.csv, test.csv, preprocessor.joblib
"""
import argparse
import json

import joblib
import pandas as pd

from . import config
from .cleaning import clean, dedupe_latest
from .features import (add_location_features, add_quality_features, add_target,
                       add_time_features, drop_leaky_columns, group_rare)
from .fetch import fetch_feed, geojson_to_df
from .transform import build_preprocessor, split_data


def load_raw() -> pd.DataFrame:
    try:
        return geojson_to_df(fetch_feed("all_week"))
    except Exception as exc:
        print(f"Network failed ({exc}); using fallback file.")
        with open(config.FALLBACK_PATH, encoding="utf-8") as f:
            return geojson_to_df(json.load(f))


def load_stream() -> pd.DataFrame:
    files = sorted(config.STREAM_DIR.glob("*.jsonl"))
    if not files:
        return pd.DataFrame()
    parts = [pd.read_json(f, lines=True, convert_dates=False) for f in files]
    return pd.concat(parts, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the quake pipeline.")
    parser.add_argument("--scaler", default="robust",
                        choices=["standard", "minmax", "robust"])
    args = parser.parse_args()

    raw = load_raw()
    stream = load_stream()
    combined = pd.concat([raw, stream], ignore_index=True) if len(stream) else raw
    n_changed = int((combined.groupby("id")["updated"].nunique() > 1).sum())
    merged = dedupe_latest(combined)

    df = clean(merged)
    n_clean = len(df)
    df = add_time_features(df)
    df = add_quality_features(df)
    df = add_location_features(df)
    df = add_target(df)
    df["region"] = group_rare(df["region"])
    df = drop_leaky_columns(df)

    X_train, X_test, y_train, y_test = split_data(df)
    pre = build_preprocessor(args.scaler)
    Xtr = pre.fit_transform(X_train)
    Xte = pre.transform(X_test)

    print("---- REPORT ----")
    print(f"raw rows (batch):            {len(raw)}")
    print(f"stream rows merged:          {len(stream)}")
    print(f"events with newer updated:   {n_changed}")
    print(f"rows after cleaning:         {n_clean}")
    print(f"train shape: {Xtr.shape}   test shape: {Xte.shape}")
    print(f"positive rate train: {y_train.mean():.4f}   test: {y_test.mean():.4f}")

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    cols = list(pre.get_feature_names_out())
    train = pd.DataFrame(Xtr, columns=cols)
    train[config.TARGET] = y_train.to_numpy()
    test = pd.DataFrame(Xte, columns=cols)
    test[config.TARGET] = y_test.to_numpy()
    train.to_csv(config.PROCESSED_DIR / "train.csv", index=False)
    test.to_csv(config.PROCESSED_DIR / "test.csv", index=False)
    joblib.dump(pre, config.PROCESSED_DIR / "preprocessor.joblib")
    print(f"saved to {config.PROCESSED_DIR}")


if __name__ == "__main__":
    main()