#!/usr/bin/env python3
"""
Rolling-origin evaluation comparing TFT vs naive persistence (last value carry-forward).

Example:
  python scripts/evaluate.py --data data/processed/merged.parquet --lookback 168 --horizon 24 --folds 3
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import NaNLabelEncoder
from pytorch_forecasting.models import TemporalFusionTransformer
from pytorch_lightning import seed_everything

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/processed/merged.parquet")
    p.add_argument("--lookback", type=int, default=168)
    p.add_argument("--horizon", type=int, default=24)
    p.add_argument("--folds", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--artifacts", default="artifacts")
    p.add_argument("--checkpoint", help="Optional path to trained TFT checkpoint")
    return p.parse_args()

def naive_persistence(y_enc_last, horizon):
    return np.repeat(y_enc_last, horizon)

def main():
    args = parse_args()
    seed_everything(args.seed)

    df = pd.read_parquet(args.data).rename(columns={"symbol":"group_id"})
    known_reals = ["hour","dow","dom","ret_1","ret_5","vol_20","rsi_14","vol_norm"] + \
                  [c for c in df.columns if c.startswith("lag_")]
    keep = list(set(["group_id","time_idx","target"] + known_reals))
    df = df[keep].dropna().reset_index(drop=True)

    tmax = df["time_idx"].max()
    fold_starts = [tmax - (i+1)*args.horizon for i in range(args.folds)][::-1]

    rows = []
    for fs in fold_starts:
        training = TimeSeriesDataSet(
            df[df.time_idx <= fs],
            time_idx="time_idx",
            target="target",
            group_ids=["group_id"],
            max_encoder_length=args.lookback,
            max_prediction_length=args.horizon,
            time_varying_unknown_reals=["target"],
            time_varying_known_reals=known_reals,
            categorical_encoders={"group_id": NaNLabelEncoder().fit(df.group_id)},
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
        )
        validation = TimeSeriesDataSet.from_dataset(training, df, predict=True, stop_randomization=True)
        val_pd = validation.to_pandas()

        # naive baseline metrics
        naive_preds = []
        true_vals = []
        for gid, g in val_pd.groupby("group_id"):
            window = g.iloc[-(args.lookback + args.horizon):]
            enc = window.iloc[:args.lookback]
            fut = window.iloc[args.lookback:]
            if enc.empty or fut.empty:
                continue
            last_val = enc["target"].iloc[-1]
            naive_preds.append(naive_persistence(last_val, len(fut)))
            true_vals.append(fut["target"].values)

        if not true_vals:
            continue
        naive_preds = np.concatenate(naive_preds)
        true_vals   = np.concatenate(true_vals)

        mape_naive = float(np.mean(np.abs((true_vals - naive_preds) / (true_vals + 1e-8))) * 100)
        rmse_naive = float(np.sqrt(np.mean((true_vals - naive_preds)**2)))

        # TFT predictions (optional)
        tft_mape = None; tft_rmse = None
        if args.checkpoint:
            tft = TemporalFusionTransformer.load_from_checkpoint(args.checkpoint)
            preds = tft.predict(validation, mode="prediction")
            median = preds[..., 3].reshape(-1)
            tft_mape = float(np.mean(np.abs((true_vals - median) / (true_vals + 1e-8))) * 100)
            tft_rmse = float(np.sqrt(np.mean((true_vals - median)**2)))

        rows.append({
            "fold_start": int(fs),
            "mape_naive": mape_naive,
            "rmse_naive": rmse_naive,
            "mape_tft": tft_mape,
            "rmse_tft": tft_rmse,
        })

    out = pd.DataFrame(rows)
    out_path = Path(args.artifacts) / "evaluation.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"Saved evaluation to {out_path}")
    print(out)

if __name__ == "__main__":
    main()