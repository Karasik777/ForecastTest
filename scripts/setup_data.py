#!/usr/bin/env python3
"""
Fetch historical candles via Binance REST and save CSVs.

Examples:
  python scripts/setup_data.py --symbols BTCUSDT ETHUSDT --interval 1m --days 3
  python scripts/setup_data.py --symbols BTCUSDT ETHUSDT --interval 5m --limit 1500
"""
import argparse
import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path

import aiohttp
import pandas as pd

BASE_REST = "https://api.binance.com"

def ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)

async def fetch_klines(session, symbol: str, interval: str, start_ms: int, end_ms: int, limit: int = 1000):
    url = f"{BASE_REST}/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "startTime": start_ms, "endTime": end_ms, "limit": limit}
    async with session.get(url, params=params, timeout=30) as r:
        r.raise_for_status()
        return await r.json()

async def download_symbol(symbol: str, interval: str, out_csv: Path, days: int = None, limit: int = None):
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    async with aiohttp.ClientSession() as session:
        rows = []
        if days is not None:
            # walk back in time by chunks
            now = datetime.now(timezone.utc)
            start = now - timedelta(days=days)
            step = timedelta(hours=24)
            t0 = start
            while t0 < now:
                t1 = min(t0 + step, now)
                data = await fetch_klines(session, symbol, interval, ms(t0), ms(t1))
                rows.extend(data)
                t0 = t1
        else:
            # single call with "limit"
            if limit is None:
                limit = 1000
            url = f"{BASE_REST}/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
            async with session.get(url, timeout=30) as r:
                r.raise_for_status()
                rows = await r.json()

    cols = ["open_time","open","high","low","close","volume","close_time","qav","num_trades","taker_base","taker_quote","ignore"]
    if not rows:
        raise RuntimeError(f"No data returned for {symbol} {interval}")
    df = pd.DataFrame(rows, columns=cols)
    df["symbol"] = symbol
    df["interval"] = interval
    df["open_time"]  = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    for c in ["open","high","low","close","volume","qav","taker_base","taker_quote"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    keep = ["symbol","interval","open_time","open","high","low","close","volume","num_trades"]
    df = df[keep].sort_values("open_time").drop_duplicates(subset=["symbol","open_time"])

    df.to_csv(out_csv, index=False)
    print(f"Saved {len(df)} rows to {out_csv}")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--symbols", nargs="+", required=True, help="e.g., BTCUSDT ETHUSDT")
    p.add_argument("--interval", default="1m", help="Binance interval (e.g., 1m,5m,15m,1h)")
    p.add_argument("--days", type=int, help="Fetch this many past days (chunks)")
    p.add_argument("--limit", type=int, help="Fetch 'limit' most recent candles if --days not set")
    p.add_argument("--outdir", default="data/raw", help="Output dir for CSVs")
    return p.parse_args()

def main():
    args = parse_args()
    outdir = Path(args.outdir)

    async def runner():
        tasks = []
        for sym in args.symbols:
            out_csv = outdir / f"{sym}_{args.interval}.csv"
            tasks.append(download_symbol(sym, args.interval, out_csv, args.days, args.limit))
        await asyncio.gather(*tasks)

    asyncio.run(runner())

if __name__ == "__main__":
    main()