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
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from forecast import ensure_utc, iso_to_ms, fetch_range_klines


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

    if not dec:
        raise RuntimeError("Decision block missing in artifact; run consensus_live first.")

    symbol   = cfg.get("symbol", "BTCUSDT")
    interval = cfg.get("interval", "1m")
    cash     = float(cfg.get("start_cash_usd", 100.0))
    comm     = float(cfg.get("commission_per_side_usd", 2.0))

    Path("artifacts").mkdir(parents=True, exist_ok=True)

    if dec.get("action") != "BUY":
        msg = dec.get("reason", "Decision did not trigger a BUY.")
        placeholder = {
            "evaluated_at": pd.Timestamp.utcnow().isoformat(),
            "symbol": symbol,
            "interval": interval,
            "action": dec.get("action", "HOLD"),
            "message": msg,
            "was_partial_window": False,
            "realized_pnl_usd": 0.0,
            "realized_return_pct": 0.0,
            "commission_total_usd": 0.0,
        }
        with open("artifacts/mock_eval.json", "w") as f:
            json.dump(placeholder, f, indent=2)
        print(f"[OK] Decision was {dec.get('action', 'HOLD')}; wrote placeholder → artifacts/mock_eval.json")
        return

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
        "evaluated_at": pd.Timestamp.utcnow().isoformat(),
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
