"""Task 2: simulate streaming by polling a feed on a timer."""
import argparse
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

from . import config
from .fetch import fetch_feed, geojson_to_df


def filter_unseen(df: pd.DataFrame, seen: set) -> pd.DataFrame:
    """Return only rows whose (id, updated) pair is not in `seen`.

    Add the new pairs to `seen` (modify the set in place).
    """
    if df.empty:
        return df
    pairs = list(zip(df["id"].tolist(), df["updated"].tolist()))
    mask = [pair not in seen for pair in pairs]
    new_rows = df[mask]
    seen.update(pair for pair, is_new in zip(pairs, mask) if is_new)
    return new_rows


def run_stream(feed: str, interval_s: int, duration_s: int, out_path: str) -> None:
    """Poll `feed` every `interval_s` seconds for `duration_s` seconds.

    Each poll: fetch, parse, keep unseen rows, append them to `out_path`
    as JSON lines, and print e.g. "[14:02:11] 3 new / 12 fetched".
    A failed poll must not stop the loop.
    """
    seen: set = set()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    end_time = time.time() + duration_s

    while time.time() < end_time:
        stamp = datetime.now().strftime("%H:%M:%S")
        try:
            df = geojson_to_df(fetch_feed(feed))
            new_rows = filter_unseen(df, seen)
            if not new_rows.empty:
                text = new_rows.to_json(orient="records", lines=True)
                if not text.endswith("\n"):
                    text += "\n"
                with open(out, "a", encoding="utf-8") as f:
                    f.write(text)
            print(f"[{stamp}] {len(new_rows)} new / {len(df)} fetched", flush=True)
        except (requests.RequestException, ValueError) as exc:
            print(f"[{stamp}] poll failed: {exc}", flush=True)
        time.sleep(interval_s)


def main() -> None:
    """argparse: --feed (default all_hour), --interval (60), --duration (2400),
    --out (data/stream/stream.jsonl), then call run_stream."""
    parser = argparse.ArgumentParser(description="Poll a USGS feed and save new events.")
    parser.add_argument("--feed", default="all_hour")
    parser.add_argument("--interval", type=int, default=60)
    parser.add_argument("--duration", type=int, default=2400)
    parser.add_argument("--out", default=str(config.STREAM_DIR / "stream.jsonl"))
    args = parser.parse_args()
    run_stream(args.feed, args.interval, args.duration, args.out)


if __name__ == "__main__":
    main()