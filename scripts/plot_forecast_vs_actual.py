#!/usr/bin/env python3
"""
plot_forecast_vs_actual.py
--------------------------
Plot stored forecast vs. realized market closes and (optionally) export an
alignment CSV: time, forecast_price, actual_price.

Usage:
  python scripts/plot_forecast_vs_actual.py \
    --artifact artifacts/mock_trade.json \
    --forecast-csv artifacts/forecast_path.csv \
    --outdir artifacts \
    --pad-min 5 \
    --write-csv

Outputs:
  - artifacts/forecast_vs_actual_plot.png
  - artifacts/forecast_vs_actual_plot.pdf
  - artifacts/forecast_vs_actual.csv          (only if --write-csv)
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests


BINANCE_BASE = "https://api.binance.com"
KLINES_EP    = "/api/v3/klines"


def ensure_utc(ts_like) -> pd.Timestamp:
    t = pd.Timestamp(ts_like)
    if t.tzinfo is None:
        return t.tz_localize("UTC")
    return t.tz_convert("UTC")


def iso_to_ms(s: str) -> int:
    return int(ensure_utc(s).timestamp() * 1000)


def fetch_range_klines(symbol: str, interval: str, start_ms: int, end_ms: int) -> pd.DataFrame:
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

    df["open_time"]  = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    for c in ["open","high","low","close","volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def load_forecast(artifact_path: str, forecast_csv_path: str | None):
    # Load mock_trade.json
    with open(artifact_path, "r") as f:
        art = json.load(f)
    cfg = art.get("config", {})
    fc  = art.get("forecast", {})
    dec = art.get("decision", {})

    # Preferred: load CSV produced by consensus_live.py
    fc_df = None
    if forecast_csv_path and Path(forecast_csv_path).exists():
        fc_df = pd.read_csv(forecast_csv_path)
        # Expect "time","price_fc"
        if not {"time","price_fc"}.issubset(fc_df.columns):
            raise ValueError("forecast CSV must have columns: time, price_fc")
        fc_df["time"] = pd.to_datetime(fc_df["time"], utc=True)

    # Fallback if CSV missing: rebuild from JSON
    if fc_df is None:
        times = pd.to_datetime(fc.get("times", []), utc=True)
        prices = fc.get("price_forecast", [])
        fc_df = pd.DataFrame({"time": times, "price_fc": prices})

    # Basic sanity
    if fc_df.empty:
        raise RuntimeError("No forecast data found.")
    return cfg, fc, dec, fc_df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", default="artifacts/mock_trade.json")
    ap.add_argument("--forecast-csv", default="artifacts/forecast_path.csv")
    ap.add_argument("--outdir", default="artifacts")
    ap.add_argument("--pad-min", type=int, default=5, help="minutes padding for the fetch window")
    ap.add_argument("--write-csv", action="store_true", help="write forecast_vs_actual.csv")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    cfg, fc, dec, fc_df = load_forecast(args.artifact, args.forecast_csv)

    symbol   = cfg.get("symbol", "BTCUSDT")
    interval = cfg.get("interval", "1m")

    # Window bounds: t0 .. last forecast time (± pad)
    t0 = ensure_utc(fc.get("t0_iso"))
    if fc.get("times"):
        t_last = ensure_utc(fc["times"][-1])
    else:
        # assume 60-min horizon if missing
        t_last = t0 + pd.Timedelta(minutes=int(fc.get("horizon_m", 60)))

    start = (t0 - pd.Timedelta(minutes=args.pad_min)).isoformat()
    end   = (t_last + pd.Timedelta(minutes=args.pad_min)).isoformat()

    # Fetch actuals
    actual = fetch_range_klines(
        symbol, interval,
        iso_to_ms(start), iso_to_ms(end)
    )
    if actual.empty:
        raise RuntimeError("No actual market data returned for the plotting window.")

    # Build an "actuals" series at minute-close timestamps
    actual_series = actual[["close_time","close"]].rename(
        columns={"close_time":"time", "close":"actual_price"}
    )

    # Align forecast vs actual by timestamp
    # (Forecast 'time' are exact minute ends per consensus_live; actuals 'time' are close_time)
    merged = pd.merge(fc_df, actual_series, on="time", how="left").sort_values("time")

    # Plot
    plt.figure(figsize=(10.5, 6.0), dpi=150)
    ax = plt.gca()

    # Actuals for the whole window
    ax.plot(actual_series["time"], actual_series["actual_price"], linewidth=1.5, label=f"{symbol} close (actual)")

    # Forecast path
    ax.plot(merged["time"], merged["price_fc"], linestyle="--", linewidth=1.6, label="Forecast path")

    # Mark t0 and forecast end
    ax.axvline(ensure_utc(t0), linestyle=":", linewidth=1.0)
    ax.axvline(ensure_utc(t_last), linestyle=":", linewidth=1.0)

    # If decision had BUY, show markers
    if dec.get("action") == "BUY":
        try:
            entry_ts = ensure_utc(dec["entry_time"])
            ax.scatter([entry_ts], [float(dec["entry_price"])], marker="^", s=60, label="Entry (forecast)")
        except Exception:
            pass
        try:
            tgt_ts = ensure_utc(dec["target_time"])
            ax.scatter([tgt_ts], [float(dec["target_price"])], marker="v", s=60, label="Target (forecast)")
        except Exception:
            pass

    ax.set_title(f"Forecast vs Actual — {symbol} ({interval})")
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("Price (close)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    plt.tight_layout()

    png = outdir / "forecast_vs_actual_plot.png"
    pdf = outdir / "forecast_vs_actual_plot.pdf"
    plt.savefig(png)
    plt.savefig(pdf)

    print(f"[OK] Wrote {png}")
    print(f"[OK] Wrote {pdf}")

    # Optional alignment CSV
    if args.write_csv:
        out_csv = outdir / "forecast_vs_actual.csv"
        merged.to_csv(out_csv, index=False)
        print(f"[OK] Wrote {out_csv}")


if __name__ == "__main__":
    main()