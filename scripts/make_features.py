#!/usr/bin/env python3
"""
Build engineered features from raw candles and produce a merged parquet:
  data/processed/merged.parquet

Example:
  python scripts/make_features.py --src data/raw --dest data/processed --interval 1m
"""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
    loss = -delta.clip(upper=0).ewm(alpha=1/period, adjust=False).mean()
    rs = gain / (loss + 1e-12)
    return 100 - (100 / (1 + rs))


def add_calendar(df: pd.DataFrame) -> pd.DataFrame:
    # Ensure tz-aware UTC and add calendar features
    ts = pd.to_datetime(df["open_time"], utc=True, errors="coerce")
    df["open_time"] = ts  # keep in UTC
    df["hour"] = ts.dt.hour.astype("int16")
    df["dow"]  = ts.dt.dayofweek.astype("int16")
    df["dom"]  = ts.dt.day.astype("int16")
    return df


def build_features(path: Path, interval: str) -> pd.DataFrame:
    files = list(path.glob(f"*_{interval}.csv"))
    if not files:
        raise FileNotFoundError(f"No CSVs found under {path} matching '*_{interval}.csv'")

    frames = []
    for f in files:
        cur = pd.read_csv(f, parse_dates=["open_time"])
        # Normalize/ensure required columns exist
        if "symbol" not in cur.columns:
            # infer from filename like BTCUSDT_1m.csv
            sym = f.name.split("_")[0]
            cur["symbol"] = sym

        # Sort and enforce UTC time
        cur["open_time"] = pd.to_datetime(cur["open_time"], utc=True, errors="coerce")
        cur = cur.sort_values("open_time").reset_index(drop=True)

        # Target variable — use close price
        cur["target"] = cur["close"].astype("float64")

        # Log returns
        cur["ret_1"] = np.log(cur["close"]).diff()
        cur["ret_5"] = np.log(cur["close"]).diff(5)

        # Rolling volatility
        cur["vol_20"] = cur["ret_1"].rolling(20, min_periods=20).std()

        # RSI
        cur["rsi_14"] = rsi(cur["close"], 14)

        # Volume normalization (z-score over 48)
        vol_mean = cur["volume"].rolling(48, min_periods=24).mean()
        vol_std  = cur["volume"].rolling(48, min_periods=24).std()
        cur["vol_norm"] = (cur["volume"] - vol_mean) / (vol_std + 1e-6)

        # Lags of target
        for L in [1, 2, 3, 6, 12, 24]:
            cur[f"lag_{L}"] = cur["target"].shift(L)

        # Calendar features
        cur = add_calendar(cur)

        frames.append(cur)

    df = pd.concat(frames, ignore_index=True)

    # Sort globally for stable time_idx creation; drop initial NaNs (feature warmup)
    df = df.sort_values(["symbol", "open_time"]).reset_index(drop=True)
    # Warmup window: ensure enough history for lags/vol/RSI (e.g., 50)
    df = df[df.groupby("symbol").cumcount() >= 50].reset_index(drop=True)

    # ---- NEW: time_idx per symbol (kept for plotting/label maps) ----
    df["time_idx"] = df.groupby("symbol").cumcount().astype("int32")

    # Keep only the columns needed downstream (and keep open_time/time_idx!)
    lag_cols = [c for c in df.columns if c.startswith("lag_")]
    keep_cols = [
        "symbol", "open_time", "time_idx", "target",
        "hour", "dow", "dom", "ret_1", "ret_5", "vol_20", "rsi_14", "vol_norm",
        *lag_cols,
    ]
    df = df[keep_cols].sort_values(["symbol", "open_time"]).reset_index(drop=True)

    return df


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--src", default="data/raw")
    p.add_argument("--dest", default="data/processed")
    p.add_argument("--interval", default="1m")
    return p.parse_args()


def main():
    args = parse_args()
    src = Path(args.src)
    dest = Path(args.dest); dest.mkdir(parents=True, exist_ok=True)

    df = build_features(src, args.interval)

    out = dest / "merged.parquet"
    df.to_parquet(out, index=False)
    print(f"Saved features to {out} with shape {df.shape}")


if __name__ == "__main__":
    main()