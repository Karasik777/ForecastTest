#!/usr/bin/env python3
"""
Temporal Fusion Transformer — Multivariate (covariate-aware) training script
-----------------------------------------------------------------------------
Creates a multivariate-ready TFT that can forecast a single target per series
(e.g., BTC close) while using many known/unknown covariates (e.g., ETH close,
volume, technical indicators, calendar features) across one or more symbols.

Key ideas
- Each *series* is identified by `--group-id-col` (e.g., SYMBOL)
- Time column is `--time-col` (e.g., timestamp). We auto-build an integer index
  per group (`time_idx`).
- Target column is `--target-col` (e.g., close or return).
- Covariates:
  * `--known-reals` and `--known-categoricals` are known in the future (e.g.,
     time features, planned events)
  * `--unknown-reals` and `--unknown-categoricals` are only known up to `now`
     (e.g., other assets' prices, realized volumes). During training they appear
     in encoder+decoder; at prediction time decoder contains only *known* vars.
- Handles multiple symbols natively via the group id.

Example
-------
python train_tft_multi.py \
  --data data/market_1m.csv \
  --time-col timestamp --group-id-col SYMBOL --target-col BTC_close \
  --known-reals time_idx hour_of_day day_of_week \
  --unknown-reals BTC_close BTC_volume ETH_close ETH_volume \
  --lookback 168 --horizon 24 --epochs 15 --batch-size 64 \
  --output-dir artifacts --save-top-k 3

Notes
-----
- Input CSV needs at least: [time-col, group-id-col, target-col], plus any
  covariates you list. Rows per group must be time-ordered (we enforce sort).
- If your project already materializes a training parquet/csv with engineered
  features, simply point `--data` to it and enumerate columns in the flags.
- If you prefer, use `--add-time-features` to automatically add hour/day/month
  features from the timestamp.
"""
from __future__ import annotations

import argparse
import os
from typing import List, Optional

import pandas as pd
import numpy as np

import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor

from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import NaNLabelEncoder
from pytorch_forecasting.models import TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss


# ----------------------------
# Utilities
# ----------------------------

def parse_list(items: List[str]) -> List[str]:
    """Flatten space/comma separated argparse lists into a list of strings."""
    flat: List[str] = []
    for it in items or []:
        flat.extend([tok for tok in it.split(",") if tok])
    return flat


def add_calendar_features(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    ts = pd.to_datetime(df[time_col], utc=True)
    df["hour_of_day"] = ts.dt.hour.astype("int16")
    df["day_of_week"] = ts.dt.weekday.astype("int16")
    df["day_of_month"] = ts.dt.day.astype("int16")
    df["week_of_year"] = ts.dt.isocalendar().week.astype("int16")
    df["month"] = ts.dt.month.astype("int16")
    return df


def ensure_time_idx(df: pd.DataFrame, time_col: str, group_col: str) -> pd.DataFrame:
    df = df.sort_values([group_col, time_col]).copy()
    # build integer index per group
    df["time_idx"] = (
        df.groupby(group_col)[time_col]
        .rank(method="first")
        .astype(int)
        - 1
    )
    return df


# ----------------------------
# Main
# ----------------------------

def main():
    parser = argparse.ArgumentParser(description="Train multivariate TFT")
    parser.add_argument("--data", required=True, help="Path to CSV/Parquet with features")
    parser.add_argument("--time-col", default="timestamp")
    parser.add_argument("--group-id-col", default="SYMBOL")
    parser.add_argument("--target-col", required=True)

    parser.add_argument("--known-reals", nargs="*", default=[], help="Known real covariates (space/comma sep)")
    parser.add_argument("--known-categoricals", nargs="*", default=[], help="Known categorical covariates")
    parser.add_argument("--unknown-reals", nargs="*", default=[], help="Unknown real covariates")
    parser.add_argument("--unknown-categoricals", nargs="*", default=[], help="Unknown categorical covariates")

    parser.add_argument("--lookback", type=int, default=168, help="Encoder length")
    parser.add_argument("--horizon", type=int, default=24, help="Prediction length")

    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-3)

    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--lstm-layers", type=int, default=2)
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)

    parser.add_argument("--accelerator", default="auto", choices=["auto", "cpu", "gpu", "mps"])
    parser.add_argument("--devices", type=int, default=1)

    parser.add_argument("--output-dir", default="artifacts")
    parser.add_argument("--save-top-k", type=int, default=2)

    parser.add_argument("--val-size", type=int, default=0, help="Optional explicit validation size (per group)")
    parser.add_argument("--add-time-features", action="store_true", help="Add hour/day/week/month features")

    args = parser.parse_args()
    seed_everything(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    # ----------------------------
    # Load data
    # ----------------------------
    path = args.data
    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    # Basic checks
    if not {args.time_col, args.group_id_col, args.target_col}.issubset(df.columns):
        missing = {args.time_col, args.group_id_col, args.target_col} - set(df.columns)
        raise ValueError(f"Missing required columns: {missing}")

    # Coerce time col to datetime
    df[args.time_col] = pd.to_datetime(df[args.time_col], utc=True, errors="coerce")
    if df[args.time_col].isna().any():
        raise ValueError("Some rows have invalid timestamps after parsing. Fix input data.")

    if args.add_time_features:
        df = add_calendar_features(df, args.time_col)

    df = ensure_time_idx(df, args.time_col, args.group_id_col)

    known_reals = parse_list(args.known_reals)
    unknown_reals = parse_list(args.unknown_reals)
    known_cats = parse_list(args.known_categoricals)
    unknown_cats = parse_list(args.unknown_categoricals)

    # Ensure target included in reals (if not already in unknown_reals)
    if args.target_col not in unknown_reals:
        unknown_reals = [args.target_col] + unknown_reals

    # Remove potential duplicates
    def dedupe(seq):
        seen = set()
        out = []
        for x in seq:
            if x not in seen:
                out.append(x)
                seen.add(x)
        return out

    known_reals = dedupe(known_reals)
    unknown_reals = dedupe(unknown_reals)
    known_cats = dedupe(known_cats)
    unknown_cats = dedupe(unknown_cats)

    # Sanity: ensure covariate columns exist
    for col in known_reals + unknown_reals + known_cats + unknown_cats:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' listed as covariate but not found in data.")

    # ----------------------------
    # Build datasets
    # ----------------------------
    max_encoder_length = args.lookback
    max_prediction_length = args.horizon

    training_cutoff = df["time_idx"].max() - max_prediction_length
    if args.val_size and args.val_size > 0:
        # Keep a per-group tail for validation
        # Mark last `val_size + horizon` points per group as validation region
        df = df.copy()
        df["is_val"] = False
        def mark_val(g):
            # last val_size + horizon indices in group
            last_idx = g["time_idx"].max()
            cutoff = last_idx - (args.val_size + max_prediction_length) + 1
            g.loc[g["time_idx"] >= cutoff, "is_val"] = True
            return g
        df = df.groupby(args.group_id_col, group_keys=False).apply(mark_val)
        train_df = df[df["is_val"] == False]
        val_df = df[df["is_val"] == True]
    else:
        train_df = df[df["time_idx"] <= training_cutoff]
        val_df = df[df["time_idx"] > training_cutoff]

    # Define dataset spec
    dataset_common_kwargs = dict(
        time_idx="time_idx",
        target=args.target_col,
        group_ids=[args.group_id_col],
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        time_varying_known_reals=["time_idx"] + known_reals,
        time_varying_unknown_reals=unknown_reals,
        time_varying_known_categoricals=known_cats,
        time_varying_unknown_categoricals=unknown_cats,
        static_categoricals=[args.group_id_col],
        categorical_encoders={args.group_id_col: NaNLabelEncoder(add_nan=True)},
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )

    training = TimeSeriesDataSet(train_df, **dataset_common_kwargs)
    validation = TimeSeriesDataSet.from_dataset(training, val_df, stop_randomization=True)

    train_loader = training.to_dataloader(train=True, batch_size=args.batch_size, num_workers=args.num_workers)
    val_loader = validation.to_dataloader(train=False, batch_size=args.batch_size, num_workers=args.num_workers)

    # ----------------------------
    # Model
    # ----------------------------
    loss = QuantileLoss()

    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=args.lr,
        hidden_size=args.hidden_size,
        attention_head_size=args.attention_heads,
        dropout=args.dropout,
        hidden_continuous_size=args.hidden_size,
        lstm_layers=args.lstm_layers,
        loss=loss,
        optimizer="adamw",
        reduce_on_plateau_patience=4,
    )

    # ----------------------------
    # Trainer & Callbacks
    # ----------------------------
    checkpoint_cb = ModelCheckpoint(
        dirpath=args.output_dir,
        filename="tft-multi-epoch={epoch}-val_loss={val_loss:.4f}",
        monitor="val_loss",
        save_top_k=args.save_top_k,
        mode="min",
        save_last=True,
        auto_insert_metric_name=False,
    )

    early_stop_cb = EarlyStopping(monitor="val_loss", patience=8, mode="min")
    lr_cb = LearningRateMonitor(logging_interval="epoch")

    trainer = Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        max_epochs=args.epochs,
        callbacks=[checkpoint_cb, early_stop_cb, lr_cb],
        gradient_clip_val=0.01,
        enable_progress_bar=True,
        default_root_dir=args.output_dir,
        deterministic=True,
        log_every_n_steps=50,
    )

    # ----------------------------
    # Fit
    # ----------------------------
    trainer.fit(tft, train_loader, val_loader)

    # Save best path for your shell pipeline to pick up
    best_path = checkpoint_cb.best_model_path or checkpoint_cb.last_model_path
    marker = os.path.join(args.output_dir, "tft_multi_best_ckpt.txt")
    with open(marker, "w") as f:
        f.write(best_path + "\n")
    print(f"Saved best checkpoint path to: {marker}\n -> {best_path}")


if __name__ == "__main__":
    main()
