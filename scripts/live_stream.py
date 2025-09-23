#!/usr/bin/env python3
"""
Stream live ticks from Binance and plot price in real time (rudimentary).

Example:
  python scripts/live_stream.py --symbol btcusdt --max-msgs 200
"""
import argparse
import asyncio
import json
from typing import List

import pandas as pd
import matplotlib.pyplot as plt
import websockets

BASE_WS = "wss://stream.binance.com:9443/ws"

async def stream_trades(symbol: str, max_msgs: int):
    uri = f"{BASE_WS}/{symbol.lower()}@trade"
    prices: List[float] = []
    times:  List[pd.Timestamp] = []
    async with websockets.connect(uri, ping_interval=15, close_timeout=5) as ws:
        for _ in range(max_msgs):
            msg = await ws.recv()
            d = json.loads(msg)
            prices.append(float(d["p"]))
            times.append(pd.to_datetime(d["T"], unit="ms", utc=True))
    return pd.DataFrame({"t": times, "price": prices})

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default="btcusdt")
    p.add_argument("--max-msgs", type=int, default=200)
    return p.parse_args()

def main():
    args = parse_args()
    df = asyncio.run(stream_trades(args.symbol, args.max_msgs))
    print(df.tail())

    plt.figure()
    plt.plot(df["t"], df["price"])
    plt.title(f"{args.symbol.upper()} — last {len(df)} ticks")
    plt.xlabel("time"); plt.ylabel("price")
    plt.tight_layout(); plt.show()

if __name__ == "__main__":
    main()