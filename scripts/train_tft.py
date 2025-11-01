#!/usr/bin/env python3
"""
Train a small Temporal Fusion Transformer on processed panel data.

Example:
  python scripts/train_tft.py --data data/processed/merged.parquet --lookback 168 --horizon 24 --epochs 15
"""
import argparse
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt

# Modern Lightning (Option A)
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

# PyTorch Forecasting
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import NaNLabelEncoder
from pytorch_forecasting.metrics import QuantileLoss
from pytorch_forecasting.models import TemporalFusionTransformer

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/processed/merged.parquet")
    p.add_argument("--lookback", type=int, default=168)
    p.add_argument("--horizon", type=int, default=24)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=0,
                   help="Number of worker processes for PyTorch dataloaders")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--artifacts", default="artifacts")
    p.add_argument("--device", choices=["auto", "cpu", "mps", "gpu"], default="auto")
    return p.parse_args()

def accel_from_arg(device: str):
    if device == "cpu":
        return dict(accelerator="cpu", devices=1)
    if device == "mps":
        return dict(accelerator="mps", devices=1)
    if device == "gpu":
        return dict(accelerator="gpu", devices=1)
    if torch.cuda.is_available():
        return dict(accelerator="gpu", devices=1)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return dict(accelerator="mps", devices=1)
    return dict(accelerator="cpu", devices=1)

def select_device(accel_kwargs: dict) -> torch.device:
    """Resolve the torch.device used for manual prediction."""
    accel = accel_kwargs.get("accelerator", "cpu")
    if accel == "gpu" and torch.cuda.is_available():
        return torch.device("cuda")
    if accel == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def main():
    args = parse_args()
    seed_everything(args.seed, workers=True)

    df = pd.read_parquet(args.data)

    # Ensure required columns exist
    required_anyway = ["symbol", "open_time", "target"]
    missing = [c for c in required_anyway if c not in df.columns]
    if missing:
        raise ValueError(
            f"Processed data is missing required columns: {missing}. "
            "Rebuild features with 'python scripts/exec.py --features --interval 1m'."
        )

    # ---- Ensure datetime & numeric index coexist ----
    # Convert open_time to datetime; keep a copy for plotting
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True, errors="coerce")
    if df["open_time"].isna().any():
        raise ValueError("Found invalid timestamps in 'open_time' after conversion to datetime.")

    # Sort for consistent indexing
    df = df.sort_values(["symbol", "open_time"]).reset_index(drop=True)

    # Create/ensure integer time_idx for the model
    if "time_idx" not in df.columns:
        df["time_idx"] = df.groupby("symbol").cumcount()

    # Keep real datetime for plotting/interpretation
    df["real_time"] = df["open_time"]

    # Rename for forecasting dataset
    df = df.rename(columns={"symbol": "group_id"})

    # ---- Ensure calendar features exist (hour/dow/dom) based on real_time ----
    # If your make_features.py already created these, this will just overwrite with same values.
    df["hour"] = df["real_time"].dt.hour.astype("int16")
    df["dow"]  = df["real_time"].dt.dayofweek.astype("int16")   # 0=Mon
    df["dom"]  = df["real_time"].dt.day.astype("int16")

    # FEATURES used by the model
    known_reals = [
        "hour", "dow", "dom", "ret_1", "ret_5", "vol_20", "rsi_14", "vol_norm",
        *[c for c in df.columns if c.startswith("lag_")]
    ]
    keep = list(set(["group_id", "time_idx", "target", "real_time"] + known_reals))
    df = df[keep].dropna().reset_index(drop=True)
    if df.empty:
        raise ValueError(
            "No usable rows remain after dropping NaNs. "
            "Regenerate features with a longer history or review preprocessing."
        )

    training_cutoff = df["time_idx"].max() - args.horizon
    if training_cutoff <= args.lookback:
        raise ValueError(
            f"Insufficient history for lookback={args.lookback} and horizon={args.horizon}. "
            "Collect more data or reduce lookback/horizon."
        )

    training = TimeSeriesDataSet(
        df[df.time_idx <= training_cutoff],
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

    train_loader = training.to_dataloader(train=True, batch_size=args.batch_size, num_workers=args.num_workers)
    val_loader   = validation.to_dataloader(train=False, batch_size=args.batch_size, num_workers=args.num_workers)

    loss = QuantileLoss()
    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=1e-3,
        hidden_size=32,
        attention_head_size=4,
        dropout=0.1,
        hidden_continuous_size=16,
        loss=loss,
        output_size=len(loss.quantiles),
        log_interval=50,
        reduce_on_plateau_patience=3,
    )

    es = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=3, mode="min")
    Path(args.artifacts).mkdir(parents=True, exist_ok=True)
    ck = ModelCheckpoint(
        monitor="val_loss", save_top_k=1, mode="min",
        dirpath=args.artifacts, filename="tft-{epoch:02d}-{val_loss:.4f}"
    )

    # Trainer for training (on chosen device)
    accel_kwargs = accel_from_arg(args.device)
    trainer = Trainer(
        max_epochs=args.epochs,
        **accel_kwargs,
        callbacks=[es, ck],
        gradient_clip_val=0.1,
        deterministic=True,
        log_every_n_steps=10,
    )

    trainer.fit(tft, train_dataloaders=train_loader, val_dataloaders=val_loader)
    best_path = ck.best_model_path
    print(f"Best checkpoint: {best_path}")
    if best_path:
        pointer = Path(args.artifacts) / "last_checkpoint.txt"
        pointer.write_text(str(best_path) + "\n")

    # --------- PREDICT (manual, no PredictCallback / trainer.predict) ----------
    best = TemporalFusionTransformer.load_from_checkpoint(best_path)

    # pick device consistent with training
    device = select_device(accel_kwargs)
    best.to(device)
    best.eval()

    preds_chunks = []
    with torch.no_grad():
        for batch in val_loader:
            # val_loader yields (x, y) or (x, y, weight); we only need x for inference
            if isinstance(batch, (list, tuple)):
                x = batch[0]
            else:
                x = batch

            # move batch inputs to device
            for k, v in list(x.items()):
                if torch.is_tensor(v):
                    x[k] = v.to(device)

            out = best(x)  # forward; returns dict with "prediction"
            preds = out["prediction"].detach().to("cpu")  # [B, horizon, n_quantiles]
            preds_chunks.append(preds)

    if not preds_chunks:
        print("Validation dataloader returned no batches; skipping qualitative plot.")
        return

    preds_tensor = torch.cat(preds_chunks, dim=0)  # [N, H, Q]
    quantiles = getattr(best.loss, "quantiles", None)
    if quantiles is None:
        raise RuntimeError("Loaded model does not expose quantile information.")
    quantiles = np.asarray([float(q) for q in quantiles])
    median_idx = int(np.argmin(np.abs(quantiles - 0.5)))
    if median_idx >= preds_tensor.shape[-1]:
        raise RuntimeError(
            f"Median quantile index {median_idx} invalid for prediction tensor with "
            f"last dimension {preds_tensor.shape[-1]}."
        )
    median = preds_tensor[0, :, median_idx].numpy()

    # --- Plot using real timestamps (real_time) from the original df ---
    first_gid = df["group_id"].value_counts().idxmax()
    g = df[df["group_id"] == first_gid].sort_values("time_idx")

    # encoder history = last lookback points up to cutoff
    enc = g[g["time_idx"] <= training_cutoff].tail(args.lookback)

    # future window = next horizon points after cutoff
    fut = g[(g["time_idx"] > training_cutoff) &
            (g["time_idx"] <= training_cutoff + args.horizon)]

    # align lengths just in case
    L = min(len(fut), len(median))
    fut = fut.head(L)
    pred_median = np.asarray(median[:L])

    plt.figure()
    plt.plot(enc["real_time"], enc["target"], label="history")
    if len(fut):
        plt.plot(fut["real_time"], fut["target"], label="true_future")
        plt.plot(fut["real_time"], pred_median, label="pred_median")
    plt.title(f"TFT demo — group {first_gid}")
    plt.xlabel("time"); plt.ylabel("target")
    plt.xticks(rotation=30)
    plt.legend(); plt.grid(True); plt.tight_layout()
    fig_path = Path(args.artifacts) / "qualitative_forecast.png"
    plt.savefig(fig_path, dpi=150)
    print(f"Saved plot: {fig_path}")
    # --------------------------------------------------------------------------

if __name__ == "__main__":
    main()
