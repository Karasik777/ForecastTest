#!/usr/bin/env python3
"""
backtest_consensus.py
---------------------
Simulate TFT-driven consensus decisions across historical data and measure:
  - Win rate, total P&L, per-trade P&L
  - Directional accuracy (did price move in the forecast direction?)
  - Forecast MAPE / RMSE vs actuals
  - Comparison against EWMA+Drift baseline on the same windows

Outputs:
  artifacts/backtest_trades.csv    — per-window trade details (TFT + EWMA columns)
  artifacts/backtest_summary.json  — aggregate metrics for both models
  artifacts/backtest_pnl.png       — cumulative P&L curves (TFT vs EWMA)

Usage:
  python scripts/backtest_consensus.py \\
    --data data/processed/merged.parquet \\
    --checkpoint artifacts/tft-epoch=08-val_loss=4.5859.ckpt \\
    --lookback 168 --horizon 24 --n-windows 20 \\
    --cash 100 --commission 2 --edge-pct 0.5
"""

import argparse
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from lightning.pytorch import seed_everything
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import NaNLabelEncoder
from pytorch_forecasting.models import TemporalFusionTransformer

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data",       default="data/processed/merged.parquet")
    p.add_argument("--checkpoint", default="",
                   help="TFT .ckpt path. Auto-detected from artifacts/last_checkpoint.txt if omitted.")
    p.add_argument("--lookback",   type=int,   default=168,  help="Encoder length (steps)")
    p.add_argument("--horizon",    type=int,   default=24,   help="Decoder length (steps)")
    p.add_argument("--n-windows",  type=int,   default=20,   help="Number of backtest windows")
    p.add_argument("--cash",       type=float, default=100.0, help="Mock capital per trade (USD)")
    p.add_argument("--commission", type=float, default=2.0,  help="Commission per side (USD)")
    p.add_argument("--edge-pct",   type=float, default=0.5,  help="Extra edge % over breakeven to BUY")
    p.add_argument("--device",     choices=["cpu","mps","gpu","auto"], default="auto")
    p.add_argument("--batch-size", type=int,   default=64)
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--artifacts",  default="artifacts")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def pick_device(name: str) -> torch.device:
    if name in ("gpu", "auto") and torch.cuda.is_available():
        return torch.device("cuda")
    if name in ("mps", "auto") and getattr(torch.backends, "mps", None) \
            and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _mape(y_true, y_pred):
    denom = np.clip(np.abs(y_true), 1e-8, None)
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100.0)


def _rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def ewma_forecast(prices: np.ndarray, horizon: int) -> np.ndarray:
    """EWMA + linear-drift baseline (identical to consensus_live.py)."""
    if len(prices) < 5:
        return np.full(horizon, float(prices[-1]) if len(prices) else np.nan)
    alpha = 2 / (min(60, max(5, len(prices))) + 1)
    s = float(prices[0])
    for p in prices:
        s = alpha * float(p) + (1 - alpha) * s
    last = s
    window = min(30, max(10, len(prices)))
    y = prices[-window:].astype(float)
    x = np.arange(window, dtype=float)
    slope = float(np.polyfit(x, y, 1)[0]) if window >= 2 else 0.0
    return last + slope * np.arange(1, horizon + 1, dtype=float)


def simulate_trade(entry_price: float, forecast: np.ndarray,
                   actual_future: np.ndarray,
                   cash: float, commission: float, edge_pct: float) -> dict:
    """
    Apply the BUY/HOLD consensus rule and compute realised P&L.

    Exit strategy: sell at the minute the forecast predicts the highest price.
    """
    breakeven_pct = (2 * commission / cash) * 100.0
    threshold     = breakeven_pct + edge_pct

    best_fc_idx   = int(np.argmax(forecast))
    best_fc_price = float(forecast[best_fc_idx])
    gross_ret_pct = (best_fc_price - entry_price) / entry_price * 100.0

    action = "BUY" if gross_ret_pct >= threshold else "HOLD"

    actual_at_exit = float(actual_future[best_fc_idx])
    dir_correct    = (actual_at_exit > entry_price) == (best_fc_price > entry_price)

    pnl_usd            = 0.0
    actual_return_pct  = 0.0
    won                = None

    if action == "BUY":
        units             = (cash - commission) / entry_price
        pnl_usd           = units * (actual_at_exit - entry_price) - commission
        actual_return_pct = (actual_at_exit - entry_price) / entry_price * 100.0
        won               = pnl_usd > 0

    return {
        "action":            action,
        "fc_best_idx":       best_fc_idx,
        "fc_best_price":     best_fc_price,
        "fc_gross_ret_pct":  gross_ret_pct,
        "actual_exit_price": actual_at_exit,
        "actual_return_pct": actual_return_pct,
        "pnl_usd":           pnl_usd,
        "won":               won,
        "dir_correct":       dir_correct,
        "breakeven_pct":     breakeven_pct,
        "threshold_pct":     threshold,
    }


def tft_predict_window(model, df_model, fold_start, lookback, horizon,
                       batch_size, label_encoder, device, known_reals):
    """
    Build a TimeSeriesDataSet for one fold, run TFT inference,
    return the median forecast array [horizon] or None on failure.
    """
    try:
        training_ds = TimeSeriesDataSet(
            df_model[df_model.time_idx <= fold_start],
            time_idx="time_idx",
            target="target",
            group_ids=["group_id"],
            max_encoder_length=lookback,
            max_prediction_length=horizon,
            time_varying_unknown_reals=["target"],
            time_varying_known_reals=known_reals,
            categorical_encoders={"group_id": label_encoder},
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
        )
        val_ds = TimeSeriesDataSet.from_dataset(
            training_ds, df_model, predict=True, stop_randomization=True
        )
        loader = val_ds.to_dataloader(
            train=False, batch_size=batch_size, num_workers=0
        )

        quantiles  = np.array([float(q) for q in model.loss.quantiles])
        median_idx = int(np.argmin(np.abs(quantiles - 0.5)))

        preds = []
        with torch.no_grad():
            for batch in loader:
                x = batch[0] if isinstance(batch, (list, tuple)) else batch
                for k, v in list(x.items()):
                    if torch.is_tensor(v):
                        x[k] = v.to(device)
                preds.append(model(x)["prediction"].cpu())

        if not preds:
            return None

        pred_tensor = torch.cat(preds, dim=0).numpy()   # [N, H, Q]
        return pred_tensor[-1, :, median_idx]            # last seq, median [H]

    except Exception as e:
        print(f" [TFT err: {e}]", end="", flush=True)
        return None


def aggregate_metrics(df: pd.DataFrame, label: str) -> dict:
    """Compute summary stats from a per-window DataFrame (TFT or EWMA columns)."""
    n_total = len(df)
    buys    = df[df.action == "BUY"]
    n_buy   = len(buys)
    n_wins  = int((buys.won == True).sum()) if n_buy > 0 else 0
    n_loss  = n_buy - n_wins

    win_rate     = (n_wins / n_buy * 100.0) if n_buy > 0 else None
    total_pnl    = float(df.pnl_usd.sum())
    avg_pnl_buy  = float(buys.pnl_usd.mean()) if n_buy > 0 else None
    dir_acc      = float(df.dir_correct.mean() * 100.0)
    avg_mape     = float(df.fc_mape.mean())
    avg_rmse_val = float(df.fc_rmse.mean())

    sharpe = None
    if n_buy >= 5:
        rets   = buys.actual_return_pct.values
        sharpe = float(np.mean(rets) / (np.std(rets) + 1e-8))

    avg_exp_ret = float(buys.fc_gross_ret_pct.mean()) if n_buy > 0 else None
    avg_act_ret = float(buys.actual_return_pct.mean()) if n_buy > 0 else None

    return {
        "model":                     label,
        "n_windows":                 n_total,
        "n_buy":                     n_buy,
        "n_hold":                    n_total - n_buy,
        "n_wins":                    n_wins,
        "n_losses":                  n_loss,
        "win_rate_pct":              round(win_rate,  2) if win_rate  is not None else None,
        "total_pnl_usd":             round(total_pnl, 4),
        "avg_pnl_per_buy_usd":       round(avg_pnl_buy, 4) if avg_pnl_buy is not None else None,
        "avg_expected_return_pct":   round(avg_exp_ret, 4) if avg_exp_ret is not None else None,
        "avg_actual_return_pct":     round(avg_act_ret, 4) if avg_act_ret is not None else None,
        "directional_accuracy_pct":  round(dir_acc,  2),
        "avg_fc_mape_pct":           round(avg_mape, 4),
        "avg_fc_rmse":               round(avg_rmse_val, 4),
        "sharpe_ratio":              round(sharpe, 4) if sharpe is not None else None,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    seed_everything(args.seed)
    device = pick_device(args.device)
    Path(args.artifacts).mkdir(parents=True, exist_ok=True)

    # ---------- Load & prepare data ----------
    df0 = pd.read_parquet(args.data)
    df0["open_time"] = pd.to_datetime(df0["open_time"], utc=True, errors="coerce")
    df0 = df0.sort_values(["symbol", "open_time"]).reset_index(drop=True)
    df0 = df0.rename(columns={"symbol": "group_id"})

    dominant = df0["group_id"].value_counts().idxmax()
    df_full  = df0[df0["group_id"] == dominant].copy()

    known_reals = [
        "hour", "dow", "dom", "ret_1", "ret_5", "vol_20", "rsi_14", "vol_norm",
        *[c for c in df_full.columns if c.startswith("lag_")],
    ]
    # Keep open_time separately for display; exclude from TFT inputs
    keep_display = list(set(["group_id", "time_idx", "target", "open_time"] + known_reals))
    df_full = df_full[[c for c in keep_display if c in df_full.columns]].dropna().reset_index(drop=True)
    df_full["time_idx"] = range(len(df_full))

    keep_model = list(set(["group_id", "time_idx", "target"] + known_reals))
    df_model   = df_full[[c for c in keep_model if c in df_full.columns]].copy()

    label_encoder = NaNLabelEncoder().fit(df_full.group_id)

    tmax  = int(df_full["time_idx"].max())
    t_min = args.lookback
    t_max = tmax - args.horizon
    if t_max <= t_min:
        raise ValueError("Insufficient data for the configured lookback/horizon.")

    decision_points = sorted(set(
        np.linspace(t_min, t_max, args.n_windows, dtype=int).tolist()
    ))
    print(f"Backtesting {len(decision_points)} windows on {dominant} "
          f"(t={t_min}..{t_max}, lookback={args.lookback}, horizon={args.horizon})")

    # ---------- Load TFT model ----------
    model = None
    ckpt  = args.checkpoint
    if not ckpt:
        ptr = Path(args.artifacts) / "last_checkpoint.txt"
        if ptr.exists():
            ckpt = ptr.read_text().strip()
    if ckpt and Path(ckpt).exists():
        print(f"Loading TFT: {ckpt}")
        model = TemporalFusionTransformer.load_from_checkpoint(ckpt, map_location=device)
        model.eval()
    else:
        print("No TFT checkpoint found — running EWMA-only backtest (TFT columns will be null).")

    breakeven_pct = (2 * args.commission / args.cash) * 100.0

    # ---------- Simulate windows ----------
    records = []
    for i, t in enumerate(decision_points):
        entry_row = df_full[df_full.time_idx == t]
        if entry_row.empty:
            continue
        entry_price = float(entry_row["target"].iloc[0])

        future_rows  = df_full[(df_full.time_idx > t) & (df_full.time_idx <= t + args.horizon)]
        if len(future_rows) < args.horizon:
            continue
        actual_future = future_rows["target"].values[:args.horizon]

        history_prices = df_full[df_full.time_idx <= t]["target"].values
        open_time      = entry_row["open_time"].iloc[0] \
                         if "open_time" in df_full.columns else t

        print(f"  [{i+1:2d}/{len(decision_points)}] t={t:4d}  entry={entry_price:,.2f}", end="", flush=True)

        # ---- TFT forecast ----
        tft_fc = None
        if model is not None:
            tft_fc = tft_predict_window(
                model, df_model, t, args.lookback, args.horizon,
                args.batch_size, label_encoder, device, known_reals
            )

        # ---- EWMA forecast ----
        ewma_fc = ewma_forecast(history_prices[-720:], args.horizon)

        # ---- Build record ----
        rec = {
            "window":      i,
            "time_idx":    t,
            "entry_time":  str(open_time),
            "entry_price": entry_price,
        }

        for prefix, fc in [("tft", tft_fc), ("ewma", ewma_fc)]:
            if fc is None:
                for key in ("action","fc_best_idx","fc_best_price","fc_gross_ret_pct",
                            "actual_exit_price","actual_return_pct","pnl_usd","won",
                            "dir_correct","breakeven_pct","threshold_pct","fc_mape","fc_rmse"):
                    rec[f"{prefix}_{key}"] = None
                rec[f"{prefix}_pnl_usd"] = 0.0
                continue

            trade = simulate_trade(
                entry_price, fc, actual_future,
                args.cash, args.commission, args.edge_pct
            )
            for k, v in trade.items():
                rec[f"{prefix}_{k}"] = v
            rec[f"{prefix}_fc_mape"] = _mape(actual_future, fc)
            rec[f"{prefix}_fc_rmse"] = _rmse(actual_future, fc)

        # ---- Console output ----
        parts = []
        for prefix, label in [("tft", "TFT"), ("ewma", "EWMA")]:
            act = rec.get(f"{prefix}_action")
            if act in ("BUY", "HOLD"):
                pnl = rec[f"{prefix}_pnl_usd"]
                parts.append(f"{label}={act}(${pnl:+.2f})")
        print("  " + "  ".join(parts) if parts else "  (skipped)")
        records.append(rec)

    if not records:
        print("No records produced — check data availability.")
        return

    df_out = pd.DataFrame(records)
    trades_path = Path(args.artifacts) / "backtest_trades.csv"
    df_out.to_csv(trades_path, index=False)
    print(f"\nSaved {len(df_out)} windows → {trades_path}")

    # ---------- Aggregate metrics ----------
    summaries = []
    for prefix, label in [("tft", "TFT"), ("ewma", "EWMA+Drift")]:
        req = [f"{prefix}_action", f"{prefix}_pnl_usd", f"{prefix}_won",
               f"{prefix}_dir_correct", f"{prefix}_fc_mape", f"{prefix}_fc_rmse",
               f"{prefix}_actual_return_pct", f"{prefix}_fc_gross_ret_pct"]
        if not all(c in df_out.columns for c in req):
            continue
        tmp = df_out[req].copy()
        tmp.columns = [c.replace(f"{prefix}_", "") for c in tmp.columns]
        tmp = tmp[tmp.action.isin(["BUY", "HOLD"])].dropna(subset=["fc_mape"])
        if tmp.empty:
            continue
        summaries.append(aggregate_metrics(tmp, label))

    summary_path = Path(args.artifacts) / "backtest_summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "config": {
                "symbol":                  dominant,
                "lookback":                args.lookback,
                "horizon":                 args.horizon,
                "n_windows":               len(records),
                "cash_usd":                args.cash,
                "commission_per_side_usd": args.commission,
                "edge_pct":                args.edge_pct,
                "breakeven_pct":           round(breakeven_pct, 4),
            },
            "results": summaries,
        }, f, indent=2)
    print(f"Saved summary → {summary_path}")

    # ---------- Console summary ----------
    print("\n" + "=" * 62)
    print("BACKTEST SUMMARY")
    print("=" * 62)
    for s in summaries:
        wr  = f"{s['win_rate_pct']:.1f}%" if s["win_rate_pct"] is not None else "n/a"
        sp  = f"{s['sharpe_ratio']:.3f}"  if s["sharpe_ratio"] is not None else "n/a"
        apr = f"{s['avg_pnl_per_buy_usd']:+.4f}" if s["avg_pnl_per_buy_usd"] is not None else "n/a"
        er  = f"{s['avg_expected_return_pct']:+.2f}%" if s["avg_expected_return_pct"] is not None else "n/a"
        ar  = f"{s['avg_actual_return_pct']:+.2f}%"  if s["avg_actual_return_pct"]  is not None else "n/a"
        print(f"\n  {s['model']}:")
        print(f"    Windows:          {s['n_windows']}  (BUY={s['n_buy']}, HOLD={s['n_hold']})")
        print(f"    Win rate:         {wr}  ({s['n_wins']} W / {s['n_losses']} L)")
        print(f"    Total P&L:        ${s['total_pnl_usd']:+.4f}")
        print(f"    Avg P&L / BUY:    {apr}")
        print(f"    Exp vs Act ret:   {er} → {ar}")
        print(f"    Dir. accuracy:    {s['directional_accuracy_pct']:.1f}%")
        print(f"    Avg MAPE:         {s['avg_fc_mape_pct']:.2f}%")
        print(f"    Avg RMSE:         {s['avg_fc_rmse']:.4f}")
        print(f"    Sharpe ratio:     {sp}")
    print("=" * 62)

    # ---------- Plot: cumulative P&L (TFT vs EWMA) ----------
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
    windows = np.arange(len(df_out))

    for ax, (prefix, label, colour) in zip(
        axes,
        [("tft", "TFT", "steelblue"), ("ewma", "EWMA+Drift", "darkorange")]
    ):
        pnl_col = f"{prefix}_pnl_usd"
        act_col = f"{prefix}_action"
        won_col = f"{prefix}_won"

        if pnl_col not in df_out.columns:
            ax.text(0.5, 0.5, f"{label} not available",
                    ha="center", va="center", transform=ax.transAxes, fontsize=11)
            continue

        cum_pnl = df_out[pnl_col].fillna(0.0).cumsum().values

        ax.plot(windows, cum_pnl, color=colour, linewidth=2, zorder=3)
        ax.fill_between(windows, cum_pnl, 0,
                        where=(cum_pnl >= 0), alpha=0.12, color="green", zorder=2)
        ax.fill_between(windows, cum_pnl, 0,
                        where=(cum_pnl < 0),  alpha=0.12, color="red",   zorder=2)
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)

        buy_mask  = df_out[act_col] == "BUY"
        win_mask  = df_out[won_col] == True
        loss_mask = df_out[won_col] == False
        if buy_mask.any():
            ax.scatter(windows[buy_mask & win_mask],  cum_pnl[buy_mask & win_mask],
                       color="green", s=55, zorder=5, marker="^", label="Win")
            ax.scatter(windows[buy_mask & loss_mask], cum_pnl[buy_mask & loss_mask],
                       color="red",   s=55, zorder=5, marker="v", label="Loss")

        n_buy  = int(buy_mask.sum())
        n_wins = int((df_out[won_col] == True).sum())
        wr_str = f"{n_wins/n_buy*100:.1f}%" if n_buy > 0 else "n/a"
        mape_v = df_out[f"{prefix}_fc_mape"].mean()
        ax.set_title(
            f"{label}  |  Win rate: {wr_str} ({n_wins}/{n_buy} trades)  |  "
            f"Total P&L: ${cum_pnl[-1]:+.2f}  |  Avg MAPE: {mape_v:.2f}%",
            fontsize=10
        )
        ax.set_ylabel("Cumulative P&L (USD)")
        if buy_mask.any():
            ax.legend(fontsize=8, loc="upper left")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Window #")
    fig.suptitle(
        f"Consensus Backtest — {dominant}  |  lookback={args.lookback}  "
        f"|  horizon={args.horizon}  |  commission=${args.commission}/side  "
        f"|  edge={args.edge_pct}%",
        fontsize=11, fontweight="bold"
    )
    plt.tight_layout()
    plot_path = Path(args.artifacts) / "backtest_pnl.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot → {plot_path}")


if __name__ == "__main__":
    main()
