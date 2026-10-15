"""Part 2, Task A: extract, transform, load, with logging."""
import argparse
import json
import logging
import sqlite3

import pandas as pd

from . import config
from .cleaning import clean
from .features import (add_location_features, add_quality_features, add_target,
                       add_time_features, drop_leaky_columns, group_rare)
from .fetch import fetch_feed, geojson_to_df

log = logging.getLogger("quakes.etl")
DB_PATH = config.PROCESSED_DIR / "quakes.db"
LOG_PATH = config.ROOT / "logs" / "etl.log"


def extract(feed: str = "all_week") -> pd.DataFrame:
    """Fetch the feed as a DataFrame. If the network fails, log a warning and
    use config.FALLBACK_PATH instead."""
    try:
        df = geojson_to_df(fetch_feed(feed))
        log.info("extract: fetched %d rows from live feed '%s'", len(df), feed)
    except Exception as exc:
        log.warning("extract: network failed (%s); using fallback file", exc)
        with open(config.FALLBACK_PATH, encoding="utf-8") as f:
            df = geojson_to_df(json.load(f))
        log.info("extract: read %d rows from fallback file", len(df))
    return df


def read_stream_files() -> pd.DataFrame:
    """Concatenate every data/stream/*.jsonl file (empty DataFrame if none)."""
    files = sorted(config.STREAM_DIR.glob("*.jsonl"))
    if not files:
        return pd.DataFrame()
    parts = [pd.read_json(f, lines=True, convert_dates=False) for f in files]
    return pd.concat(parts, ignore_index=True)


def transform(raw: pd.DataFrame) -> pd.DataFrame:
    """Day 1 steps in one call: clean, time/quality/location features, target,
    group_rare on 'region', drop leaky columns."""
    df = clean(raw)
    df = add_time_features(df)
    df = add_quality_features(df)
    df = add_location_features(df)
    df = add_target(df)
    df["region"] = group_rare(df["region"])
    df = drop_leaky_columns(df)
    return df


def load_sqlite(df: pd.DataFrame, db_path, table: str = "quakes") -> None:
    """Upsert on 'id': running twice must not duplicate rows, and a row with an
    existing id must be replaced by the new version.
    Hints: sqlite3, df.to_sql, DELETE before append. Datetime columns need to
    be converted to text first."""
    out = df.copy()
    for col in out.select_dtypes(include=["datetime", "datetimetz"]).columns:
        out[col] = out[col].astype(str)

    db_path = str(db_path)
    con = sqlite3.connect(db_path)
    try:
        exists = con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table,)).fetchone()
        if exists:
            con.executemany(f'DELETE FROM "{table}" WHERE id = ?',
                            [(i,) for i in out["id"].tolist()])
        out.to_sql(table, con, if_exists="append", index=False)
        con.commit()
    finally:
        con.close()


def run_etl(mode: str = "batch") -> None:
    """extract -> (merge stream files if mode == 'merge') -> transform -> load.
    Log row counts at every stage with the logging module."""
    raw = extract()
    log.info("after extract: %d rows", len(raw))
    if mode == "merge":
        stream = read_stream_files()
        log.info("stream files: %d rows", len(stream))
        if len(stream):
            raw = pd.concat([raw, stream], ignore_index=True)
        log.info("after merge: %d rows", len(raw))
    df = transform(raw)
    log.info("after transform: %d rows, %d columns", len(df), df.shape[1])
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    load_sqlite(df, DB_PATH, "quakes")
    log.info("load: wrote %d rows to %s", len(df), DB_PATH)


def main() -> None:
    """argparse: --mode {batch, merge}. Configure logging to LOG_PATH and the
    console, then call run_etl."""
    parser = argparse.ArgumentParser(description="Run the quake ETL.")
    parser.add_argument("--mode", choices=["batch", "merge"], default="batch")
    args = parser.parse_args()

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(LOG_PATH, encoding="utf-8"),
                  logging.StreamHandler()])
    try:
        run_etl(args.mode)
    except Exception:
        log.exception("ETL failed")
        raise


if __name__ == "__main__":
    main()