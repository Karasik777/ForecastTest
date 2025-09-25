# --- inside setup_data.py ---

import asyncio
from datetime import datetime, timedelta, timezone
import aiohttp
import pandas as pd
from pathlib import Path

BINANCE = "https://api.binance.com/api/v3/klines"

def ms(dt):  # datetime -> milliseconds
    return int(dt.timestamp() * 1000)

async def fetch_klines(session, symbol, interval, start_ms, end_ms, limit=1000):
    assert " " not in symbol and "+" not in symbol, f"Invalid symbol: {symbol!r}"
    params = {
        "symbol": symbol,
        "interval": interval,
        "startTime": start_ms,
        "endTime": end_ms,
        "limit": limit,
    }
    async with session.get(BINANCE, params=params) as r:
        r.raise_for_status()
        return await r.json()

async def download_symbol(session, symbol, interval, t0, t1, outdir: Path):
    # fetch in chunks of <= 1000 candles if needed
    start = t0
    rows = []
    while True:
        data = await fetch_klines(session, symbol, interval, ms(start), ms(t1), limit=1000)
        if not data:
            break
        rows.extend(data)
        last_open_time = data[-1][0] / 1000.0
        next_start = datetime.fromtimestamp(last_open_time, tz=timezone.utc) + timedelta(seconds=1)
        if next_start >= t1 or len(data) < 1000:
            break
        start = next_start

    if not rows:
        print(f"No data for {symbol} in range.")
        return

    df = pd.DataFrame(rows, columns=[
        "open_time","open","high","low","close","volume",
        "close_time","qav","num_trades","taker_base","taker_quote","ignore"
    ])
    df["symbol"] = symbol
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    df[["open","high","low","close","volume"]] = df[["open","high","low","close","volume"]].astype(float)

    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / f"{symbol}_{interval}.csv"
    df.to_csv(out, index=False)
    print(f"Saved {len(df)} rows to {out}")

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

    async def runner():
        async with aiohttp.ClientSession() as session:
            tasks = [
                download_symbol(session, sym, args.interval, t0, t1, outdir)
                for sym in args.symbols
            ]
            await asyncio.gather(*tasks)

    asyncio.run(runner())

if __name__ == "__main__":
    main()