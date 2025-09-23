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
    ts = df["open_time"].dt.tz_convert("UTC")
    df["hour"] = ts.dt.hour
    df["dow"]  = ts.dt.dayofweek
    df["dom"]  = ts.dt.day
    return df

def build_features(path: Path, interval: str) -> pd.DataFrame:
    files = list(path.glob(f"*_{interval}.csv"))
    if not files:
        raise FileNotFoundError(f"No CSVs found under {path} matching '*_{interval}.csv'")
    frames = []
    for f in files:
        cur = pd.read_csv(f, parse_dates=["open_time"])
        cur["open_time"] = pd.to_datetime(cur["open_time"], utc=True)
        cur = cur.sort_values("open_time")
        # target variable — we use close price
        cur["target"] = cur["close"]
        # log returns
        cur["ret_1"] = np.log(cur["close"]).diff()
        cur["ret_5"] = np.log(cur["close"]).diff(5)
        # rolling volatility
        cur["vol_20"] = cur["ret_1"].rolling(20).std()
        # RSI
        cur["rsi_14"] = rsi(cur["close"], 14)
        # volume features
        cur["vol_norm"] = (cur["volume"] - cur["volume"].rolling(48).mean()) / (cur["volume"].rolling(48).std() + 1e-6)
        # lags of target
        for L in [1,2,3,6,12,24]:
            cur[f"lag_{L}"] = cur["target"].shift(L)
        # calendar
        cur = add_calendar(cur)
        frames.append(cur)

    df = pd.concat(frames, ignore_index=True)
    # drop initial NaNs from features
    df = df.sort_values(["symbol", "open_time"])
    df = df[df.groupby("symbol").cumcount() >= 50].reset_index(drop=True)
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