#!/usr/bin/env python3
"""
Rolling-origin evaluation comparing TFT vs naive persistence (last value carry-forward).

Example:
  python scripts/evaluate.py --data data/processed/merged.parquet --lookback 168 --horizon 24 --folds 3 \
    --checkpoint artifacts/tft-epoch=11-val_loss=4.5663.ckpt --device cpu --batch-size 64
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import warnings

from lightning.pytorch import seed_everything
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import NaNLabelEncoder
from pytorch_forecasting.models import TemporalFusionTransformer

# silence harmless sklearn scaler warning seen during PF transforms
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but StandardScaler was fitted",
)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/processed/merged.parquet")
    p.add_argument("--lookback", type=int, default=168)
    p.add_argument("--horizon", type=int, default=24)
    p.add_argument("--folds", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--artifacts", default="artifacts")
    p.add_argument("--checkpoint", help="Optional path to trained TFT checkpoint (.ckpt)")
    p.add_argument("--device", choices=["cpu", "mps", "gpu"], default="cpu")
    p.add_argument("--batch-size", type=int, default=64)
    return p.parse_args()

def pick_device(name: str) -> torch.device:
    if name == "gpu" and torch.cuda.is_available():
        return torch.device("cuda")
    if name == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def mape(y_true, y_pred):
    # avoid division by zero on tiny magnitudes
    denom = np.clip(np.abs(y_true), 1e-8, None)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)

def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def main():
    args = parse_args()
    seed_everything(args.seed)

    # ---------- Load & basic checks ----------
    df0 = pd.read_parquet(args.data).rename(columns={"symbol": "group_id"})
    required = ["group_id", "open_time", "target"]
    missing = [c for c in required if c not in df0.columns]
    if missing:
        raise ValueError(f"Processed data missing columns {missing}. Rebuild features first.")

    # ensure datetime + global sort; keep time_idx per group if missing
    df0["open_time"] = pd.to_datetime(df0["open_time"], utc=True, errors="coerce")
    if df0["open_time"].isna().any():
        raise ValueError("Invalid timestamps in open_time after conversion.")

    df0 = df0.sort_values(["group_id", "open_time"]).reset_index(drop=True)
    if "time_idx" not in df0.columns:
        df0["time_idx"] = df0.groupby("group_id").cumcount()

    # ---------- NEW: emit time_idx -> datetime label map for plotting ----------
    dominant_gid = df0["group_id"].value_counts().idxmax()
    label_map = (
        df0.loc[df0["group_id"] == dominant_gid, ["time_idx", "open_time"]]
        .dropna()
        .drop_duplicates("time_idx")
        .sort_values("time_idx")
    )
    label_map_path = Path(args.artifacts) / "time_labels.csv"
    label_map_path.parent.mkdir(parents=True, exist_ok=True)
    label_map.to_csv(label_map_path, index=False)
    print(f"Saved time labels to {label_map_path}")
    # --------------------------------------------------------------------------

    # ---------- Build modeling dataframe (drop unused columns) ----------
    known_reals = [
        "hour", "dow", "dom", "ret_1", "ret_5", "vol_20", "rsi_14", "vol_norm",
        *[c for c in df0.columns if c.startswith("lag_")],
    ]
    keep = list(set(["group_id", "time_idx", "target"] + known_reals))
    df = df0[keep].dropna().reset_index(drop=True)
    if df.empty:
        raise ValueError(
            "No usable rows remain after dropping NaNs. "
            "Ensure features were generated with sufficient history."
        )
    label_encoder = NaNLabelEncoder().fit(df.group_id)

    tmax = int(df["time_idx"].max())
    raw_starts = [tmax - (i + 1) * args.horizon for i in range(args.folds)]
    fold_starts = sorted(fs for fs in raw_starts if fs >= args.lookback)
    if not fold_starts:
        raise ValueError(
            "Not enough history for the requested folds/lookback/horizon combination. "
            "Reduce --folds or horizon, or gather more data."
        )

    rows = []
    device = pick_device(args.device)

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
            categorical_encoders={"group_id": label_encoder},
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
        )
        validation = TimeSeriesDataSet.from_dataset(training, df, predict=True, stop_randomization=True)
        val_loader = validation.to_dataloader(train=False, batch_size=args.batch_size, num_workers=0)

        # --- Collect true futures & naive baseline ---
        all_true = []
        all_naive = []

        for batch in val_loader:
            # batch can be (x, y) or (x, y, weight)
            if isinstance(batch, (list, tuple)):
                if len(batch) == 3:
                    x, y, _w = batch
                else:
                    x, y = batch
            else:
                raise RuntimeError("Unexpected batch structure from dataloader")

            # y can itself be a tuple, e.g., (y, weight) in some PF versions
            if isinstance(y, (list, tuple)):
                y = y[0]

            # y shape: [B, H] or [B, H, 1] -> squeeze to [B, H]
            if hasattr(y, "dim") and y.dim() == 3 and y.size(-1) == 1:
                y = y.squeeze(-1)

            y_np = y.detach().cpu().numpy()  # [B, H]
            all_true.append(y_np)

            # naive = last encoder target repeated across horizon
            if "encoder_target" in x:
                enc_last = x["encoder_target"][:, -1]
                if hasattr(enc_last, "dim") and enc_last.dim() > 1:
                    enc_last = enc_last.squeeze(-1)
                enc_last_np = enc_last.detach().cpu().numpy()  # [B]
                naive_batch = np.repeat(enc_last_np[:, None], y_np.shape[1], axis=1)
            else:
                # fallback: repeat first future value (keeps code running, but baseline is weaker)
                naive_batch = np.repeat(y_np[:, :1], y_np.shape[1], axis=1)
            all_naive.append(naive_batch)

        if not all_true:
            continue

        true_vals = np.concatenate(all_true, axis=0).reshape(-1)
        naive_vals = np.concatenate(all_naive, axis=0).reshape(-1)

        mape_naive = mape(true_vals, naive_vals)
        rmse_naive = rmse(true_vals, naive_vals)

        # --- TFT predictions (optional) ---
        tft_mape = None
        tft_rmse = None
        if args.checkpoint:
            if not args.checkpoint.endswith(".ckpt"):
                print("⚠️  Note: your --checkpoint usually needs a .ckpt file extension.")
            model = TemporalFusionTransformer.load_from_checkpoint(args.checkpoint, map_location=device)
            model.eval()

            quantiles = getattr(model.loss, "quantiles", None)
            if quantiles is None:
                raise RuntimeError("Checkpoint loss function does not expose quantiles.")
            quantiles = np.asarray([float(q) for q in quantiles])
            median_idx = int(np.argmin(np.abs(quantiles - 0.5)))

            pred_chunks = []
            with torch.no_grad():
                for batch in val_loader:
                    if isinstance(batch, (list, tuple)):
                        if len(batch) == 3:
                            x, y, _w = batch
                        else:
                            x, y = batch
                    else:
                        raise RuntimeError("Unexpected batch structure from dataloader")

                    # move tensors to device
                    for k, v in list(x.items()):
                        if torch.is_tensor(v):
                            x[k] = v.to(device)

                    # forward pass; returns dict with "prediction": [B, H, Q]
                    out = model(x)
                    pred_chunks.append(out["prediction"].detach().cpu())

            if not pred_chunks:
                raise RuntimeError("No validation batches produced while loading checkpoint.")

            preds = torch.cat(pred_chunks, dim=0).numpy()  # [N, H, Q]
            if median_idx >= preds.shape[-1]:
                raise RuntimeError(
                    f"Median quantile index {median_idx} invalid for prediction tensor "
                    f"with last dimension {preds.shape[-1]}."
                )
            median = preds[:, :, median_idx].reshape(-1)
            tft_mape = mape(true_vals, median)
            tft_rmse = rmse(true_vals, median)

        rows.append({
            "fold_start": int(fs),
            "mape_naive": mape_naive,
            "rmse_naive": rmse_naive,
            "mape_tft": tft_mape,
            "rmse_tft": tft_rmse,
        })

    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError("Evaluation produced no rows. Adjust folds/horizon or check data availability.")
    out_path = Path(args.artifacts) / "evaluation.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"Saved evaluation to {out_path}")
    print(out.to_string(index=False))

if __name__ == "__main__":
    main()
