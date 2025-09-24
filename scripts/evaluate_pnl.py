#!/usr/bin/env python3
"""
evaluate_pnl.py
---------------
Evaluate the *realized* PnL of a previously saved mock decision (artifacts/mock_trade.json).
This script fetches actual minute candles for the decision window and simulates a fill.

Assumptions:
- BUY fills at the 'close' of the decision minute (conservative).
- SELL fills at the 'close' of the chosen target minute (or the last minute if target not reached yet).
- Commission: $2 per BUY + $2 per SELL (flat) by default (read from artifact config).

Usage:
  python scripts/evaluate_pnl.py --artifact artifacts/mock_trade.json

If run before the forecast horizon has elapsed, it computes realized PnL on the data available
and marks the result as a partial window.
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import requests

BINANCE_BASE = "https://api.binance.com"
KLINES_EP    = "/api/v3/klines"


def ensure_utc(ts_like) -> pd.Timestamp:
    """Return a pandas Timestamp in UTC regardless of input tz-naive/aware."""
    t = pd.Timestamp(ts_like)
    if t.tzinfo is None:
        return t.tz_localize("UTC")
    return t.tz_convert("UTC")


def iso_to_ms(s: str) -> int:
    """ISO8601 → epoch milliseconds (UTC)."""
    t_utc = ensure_utc(s)
    return int(t_utc.timestamp() * 1000)


def fetch_range_klines(symbol: str, interval: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    """
    Fetch klines between start_ms and end_ms (UTC ms).
    For our 60-minute window the single call (limit=1000) is sufficient.
    """
    url = BINANCE_BASE + KLINES_EP
    params = {
        "symbol": symbol,
        "interval": interval,
        "startTime": start_ms,
        "endTime": end_ms,
        "limit": 1000
    }
    headers = {"User-Agent": "mock-consensus/1.0"}
    resp = requests.get(url, params=params, headers=headers, timeout=30)
    resp.raise_for_status()
    raw = resp.json()

    cols = ["open_time","open","high","low","close","volume",
            "close_time","qav","num_trades","taker_base","taker_quote","ignore"]
    df = pd.DataFrame(raw, columns=cols)
    if df.empty:
        return df

    df["open_time"]  = pd.to_datetime(df["open_time"],  unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    for c in ["open","high","low","close","volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", default="artifacts/mock_trade.json",
                    help="Path to decision artifact produced by consensus_live.py")
    args = ap.parse_args()

    # Load artifact
    with open(args.artifact, "r") as f:
        art = json.load(f)

    cfg = art.get("config", {})
    dec = art.get("decision", {})
    fc  = art.get("forecast", {})

    if not dec or dec.get("action") != "BUY":
        print("No trade to evaluate (decision was HOLD or artifact missing).")
        return

    symbol   = cfg.get("symbol", "BTCUSDT")
    interval = cfg.get("interval", "1m")
    cash     = float(cfg.get("start_cash_usd", 100.0))
    comm     = float(cfg.get("commission_per_side_usd", 2.0))

    entry_iso  = dec["entry_time"]
    target_iso = dec["target_time"]

    # Define evaluation window: from entry minute to the last forecast minute (+ 60s buffer)
    last_fc_iso = fc["times"][-1] if fc.get("times") else target_iso
    start_ms = iso_to_ms(entry_iso) - 1000    # small nudge
    end_ms   = iso_to_ms(last_fc_iso) + 60_000

    mkt = fetch_range_klines(symbol, interval, start_ms, end_ms)
    if mkt.empty:
        raise RuntimeError("No market data returned for evaluation window. Try again later.")

    # Entry fill = first candle with close_time >= entry_time
    entry_ts = ensure_utc(entry_iso)
    entry_row = mkt.loc[mkt["close_time"] >= entry_ts].head(1)
    if entry_row.empty:
        raise RuntimeError("Could not locate entry candle in fetched window.")
    entry_fill = float(entry_row["close"].iloc[0])

    # Target fill = first candle with close_time >= target_time
    target_ts = ensure_utc(target_iso)
    sell_row = mkt.loc[mkt["close_time"] >= target_ts].head(1)

    partial = False
    if sell_row.empty:
        # Not enough data to reach target; use last available (partial evaluation)
        sell_row = mkt.tail(1)
        partial = True
    else:
        # Even if we found a sell row, if our fetched data hasn't reached the end of the forecast window,
        # mark as partial to be explicit.
        have_until = mkt["close_time"].max()
        last_fc_ts = ensure_utc(last_fc_iso)
        if have_until < last_fc_ts:
            partial = True

    sell_fill = float(sell_row["close"].iloc[0])

    # Position sizing & PnL (pay buy commission up-front; sell commission on exit)
    units = (cash - comm) / entry_fill
    pnl   = units * (sell_fill - entry_fill) - comm
    rr    = (pnl / cash) * 100.0

    out = {
        "evaluated_at": pd.Timestamp.utcnow().tz_localize("UTC").isoformat(),
        "symbol": symbol,
        "interval": interval,
        "entry_time": entry_ts.isoformat(),
        "entry_fill": entry_fill,
        "sell_time": sell_row["close_time"].iloc[0].isoformat(),
        "sell_fill": sell_fill,
        "commission_total_usd": 2 * comm,
        "units_bought": units,
        "realized_pnl_usd": pnl,
        "realized_return_pct": rr,
        "was_partial_window": partial
    }

    Path("artifacts").mkdir(parents=True, exist_ok=True)
    with open("artifacts/mock_eval.json", "w") as f:
        json.dump(out, f, indent=2)

    print("[OK] Evaluation saved -> artifacts/mock_eval.json")
    print(f"Realized PnL: ${pnl:.2f} ({rr:.2f}%) | Partial window: {partial}")
    print(f"Entry @ {entry_fill:.2f} (UTC close) | Exit @ {sell_fill:.2f} (UTC close)")


if __name__ == "__main__":
    main()