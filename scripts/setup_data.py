# --- inside setup_data.py ---

import json
import subprocess
from datetime import datetime, timedelta, timezone
import pandas as pd
from pathlib import Path

BINANCE = "https://data-api.binance.vision/api/v3/klines"


def ms(dt):  # datetime -> milliseconds
    return int(dt.timestamp() * 1000)


def fetch_klines(symbol, interval, start_ms, end_ms, limit=1000):
    assert " " not in symbol and "+" not in symbol, f"Invalid symbol: {symbol!r}"
    url = (
        f"{BINANCE}?symbol={symbol}&interval={interval}"
        f"&startTime={start_ms}&endTime={end_ms}&limit={limit}"
    )
    result = subprocess.run(
        ["curl", "-s", "--connect-timeout", "15", "--retry", "3", url],
        capture_output=True, text=True, timeout=60,
    )
    if result.returncode != 0:
        raise RuntimeError(f"curl failed for {symbol}: {result.stderr.strip()}")
    data = json.loads(result.stdout)
    if isinstance(data, dict) and "code" in data:
        raise RuntimeError(f"API error for {symbol}: {data}")
    return data


def download_symbol(symbol, interval, t0, t1, outdir: Path):
    start = t0
    rows = []
    chunk = 0
    while True:
        chunk += 1
        data = fetch_klines(symbol, interval, ms(start), ms(t1), limit=1000)
        if not data:
            break
        rows.extend(data)
        last_open_time = data[-1][0] / 1000.0
        next_start = datetime.fromtimestamp(last_open_time, tz=timezone.utc) + timedelta(seconds=1)
        pct = (next_start - t0).total_seconds() / (t1 - t0).total_seconds() * 100
        print(f"  {symbol}  chunk {chunk:3d}  rows so far: {len(rows):6,d}  ({pct:.0f}%)", flush=True)
        if next_start >= t1 or len(data) < 1000:
            break
        start = next_start

    if not rows:
        print(f"No data for {symbol} in range.")
        return

    df = pd.DataFrame(rows, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "qav", "num_trades", "taker_base", "taker_quote", "ignore",
    ])
    df["symbol"] = symbol
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    df[["open", "high", "low", "close", "volume"]] = (
        df[["open", "high", "low", "close", "volume"]].astype(float)
    )

    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / f"{symbol}_{interval}.csv"
    df.to_csv(out, index=False)
    print(f"Saved {len(df):,} rows → {out}")


def parse_args():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--symbols", nargs="+", required=True)
    p.add_argument("--interval", default="1m")
    p.add_argument("--days", type=int, default=3)
    p.add_argument("--out", default="data/raw")
    return p.parse_args()


def main():
    args = parse_args()
    outdir = Path(args.out)
    t1 = datetime.now(timezone.utc)
    t0 = t1 - timedelta(days=args.days)
    print(f"Fetching {args.days}d of {args.interval} data from {t0:%Y-%m-%d} to {t1:%Y-%m-%d}")
    for sym in args.symbols:
        print(f"\n── {sym} ──")
        download_symbol(sym, args.interval, t0, t1, outdir)


if __name__ == "__main__":
    main()
