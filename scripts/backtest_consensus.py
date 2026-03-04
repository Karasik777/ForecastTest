#!/usr/bin/env python3
"""
backtest_consensus.py — v2
--------------------------
Simulate consensus decisions across historical data.  Three improvements over v1:

  1. Trend filter  (--trend-filter N)
     Only issue a BUY when the linear slope of the last N candles is positive.
     Prevents buying into confirmed downtrends.

  2. Better exit  (--exit-strategy [peak|end])
     peak : sell at the minute the forecast peaks (original, cherry-picks the top)
     end  : sell at the final minute of the horizon (no look-ahead bias on timing)

  3. Generic model loader
     Auto-detects TFT or N-HiTS from the checkpoint file.

  4. Grid search  (--grid-search)
     Sweeps over a lookback × horizon grid using the fast EWMA baseline.
     Prints a ranked table and marks the recommended (lookback, horizon) config.

Outputs (per run):
  artifacts/backtest_trades.csv    — per-window detail (TFT + EWMA columns)
  artifacts/backtest_summary.json  — aggregate metrics
  artifacts/backtest_pnl.png       — cumulative P&L curves
"""

import argparse
import json
import time
import warnings
from pathlib import Path
from itertools import product

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from lightning.pytorch import seed_everything
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import NaNLabelEncoder
from pytorch_forecasting.models import TemporalFusionTransformer, NHiTS

warnings.filterwarnings("ignore")


# ───────────────────────────── CLI ──────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data",          default="data/processed/merged.parquet")
    p.add_argument("--checkpoint",    default="",
                   help="Model .ckpt path. Auto-detected from artifacts/last_checkpoint.txt.")
    p.add_argument("--lookback",      type=int,   default=168)
    p.add_argument("--horizon",       type=int,   default=24)
    p.add_argument("--n-windows",     type=int,   default=30,
                   help="Number of evenly-spaced backtest windows")
    p.add_argument("--cash",          type=float, default=100.0)
    p.add_argument("--commission",    type=float, default=0.1,
                   help="Commission per side in USD (0.1 ≈ Binance 0.1%% on $100)")
    p.add_argument("--edge-pct",      type=float, default=0.3,
                   help="Extra edge %% over breakeven required to trigger BUY")
    # Step 1 — trend filter
    p.add_argument("--trend-filter",  type=int,   default=30,
                   help="Lookback steps for trend slope filter (0 = disabled). "
                        "BUY only when slope > 0 over this many candles.")
    # Step 2 — exit strategy
    p.add_argument("--exit-strategy", choices=["peak", "end"], default="end",
                   help="peak: exit at forecast-peak minute; end: exit at horizon end")
    # Grid search
    p.add_argument("--grid-search",   action="store_true",
                   help="Sweep lookback × horizon grid using EWMA baseline, "
                        "then recommend the best config.")
    p.add_argument("--device",        choices=["cpu","mps","gpu","auto"], default="auto")
    p.add_argument("--batch-size",    type=int,   default=64)
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--artifacts",     default="artifacts")
    return p.parse_args()


# ───────────────────────────── Helpers ──────────────────────────────────────

def pick_device(name: str) -> torch.device:
    if name in ("gpu", "auto") and torch.cuda.is_available():
        return torch.device("cuda")
    if name in ("mps", "auto") and getattr(torch.backends, "mps", None) \
            and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _mape(y_true, y_pred):
    d = np.clip(np.abs(y_true), 1e-8, None)
    return float(np.mean(np.abs(y_true - y_pred) / d) * 100.0)


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


def trend_slope(prices: np.ndarray, n: int) -> float:
    """Linear slope of the last n price candles (price-per-step)."""
    if n <= 0 or len(prices) < n:
        return 0.0
    y = prices[-n:].astype(float)
    x = np.arange(n, dtype=float)
    return float(np.polyfit(x, y, 1)[0])


def load_model(ckpt_path: str, device: torch.device):
    """Auto-detect TFT or N-HiTS from checkpoint."""
    for cls in [TemporalFusionTransformer, NHiTS]:
        try:
            m = cls.load_from_checkpoint(ckpt_path, map_location=device)
            print(f"  Loaded {cls.__name__} from {Path(ckpt_path).name}")
            return m
        except Exception:
            continue
    raise RuntimeError(f"Cannot load model from {ckpt_path!r}")


# ───────────────── Trade simulation (Steps 1 & 2) ───────────────────────────

def simulate_trade(
    entry_price: float,
    forecast: np.ndarray,
    actual_future: np.ndarray,
    history_prices: np.ndarray,
    cash: float,
    commission: float,
    edge_pct: float,
    trend_filter: int,
    exit_strategy: str,
) -> dict:
    """
    Apply BUY/HOLD decision with optional trend filter and configurable exit.

    Step 1 — Trend filter: if trend_filter > 0 and slope of last N prices < 0,
      force HOLD regardless of forecast.
    Step 2 — Exit strategy:
      'end'  → sell at the last step of the horizon (unbiased)
      'peak' → sell at the forecast-peak step (legacy, introduces selection bias)
    """
    breakeven_pct = (2 * commission / cash) * 100.0
    threshold     = breakeven_pct + edge_pct

    # Step 1: trend filter
    slope = trend_slope(history_prices, trend_filter)
    if trend_filter > 0 and slope < 0:
        return {
            "action": "HOLD", "trend_blocked": True,
            "fc_best_idx": None, "fc_best_price": None,
            "fc_gross_ret_pct": (float(forecast.max()) - entry_price) / entry_price * 100.0,
            "actual_exit_price": float(actual_future[-1]),
            "actual_return_pct": (float(actual_future[-1]) - entry_price) / entry_price * 100.0,
            "pnl_usd": 0.0, "won": None, "dir_correct": None,
            "breakeven_pct": breakeven_pct, "threshold_pct": threshold,
        }

    best_fc_idx   = int(np.argmax(forecast))
    best_fc_price = float(forecast[best_fc_idx])
    gross_ret_pct = (best_fc_price - entry_price) / entry_price * 100.0
    action        = "BUY" if gross_ret_pct >= threshold else "HOLD"

    # Step 2: pick exit index
    if exit_strategy == "end":
        exit_idx = len(actual_future) - 1
    else:   # "peak" — legacy
        exit_idx = best_fc_idx

    actual_at_exit    = float(actual_future[exit_idx])
    dir_correct       = (actual_at_exit > entry_price) == (best_fc_price > entry_price)
    pnl_usd           = 0.0
    actual_return_pct = 0.0
    won               = None

    if action == "BUY":
        units             = (cash - commission) / entry_price
        pnl_usd           = units * (actual_at_exit - entry_price) - commission
        actual_return_pct = (actual_at_exit - entry_price) / entry_price * 100.0
        won               = pnl_usd > 0

    return {
        "action":            action,
        "trend_blocked":     False,
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


# ───────────────────────── TFT/NHiTS inference ──────────────────────────────

def model_predict(model, df_model, fold_start, lookback, horizon,
                  batch_size, label_encoder, device, known_reals):
    """Run one-fold model inference. Returns median forecast [horizon] or None."""
    try:
        training_ds = TimeSeriesDataSet(
            df_model[df_model.time_idx <= fold_start],
            time_idx="time_idx", target="target",
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
        loader = val_ds.to_dataloader(train=False, batch_size=batch_size, num_workers=0)

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
        return torch.cat(preds, dim=0).numpy()[-1, :, median_idx]

    except Exception as e:
        print(f" [model err: {e}]", end="", flush=True)
        return None


# ──────────────────────────── Aggregation ───────────────────────────────────

def aggregate(df: pd.DataFrame, label: str) -> dict:
    n       = len(df)
    buys    = df[df.action == "BUY"]
    nb      = len(buys)
    nw      = int((buys.won == True).sum())  if nb > 0 else 0
    nl      = nb - nw
    wr      = nw / nb * 100.0 if nb > 0 else None
    total   = float(df.pnl_usd.sum())
    avg_buy = float(buys.pnl_usd.mean()) if nb > 0 else None
    dir_a   = float(df.dir_correct.dropna().mean() * 100.0) if df.dir_correct.notna().any() else None
    avg_m   = float(df.fc_mape.mean())    if "fc_mape" in df else None
    avg_r   = float(df.fc_rmse.mean())    if "fc_rmse" in df else None
    sharpe  = None
    if nb >= 5:
        rets   = buys.actual_return_pct.values
        sharpe = float(np.mean(rets) / (np.std(rets) + 1e-8))
    trend_b = int(df.get("trend_blocked", pd.Series([False]*n)).sum())
    return dict(
        model=label, n_windows=n, n_buy=nb, n_hold=n-nb,
        n_trend_blocked=trend_b,
        n_wins=nw, n_losses=nl,
        win_rate_pct      = round(wr,  2) if wr  is not None else None,
        total_pnl_usd     = round(total, 4),
        avg_pnl_per_buy   = round(avg_buy, 4) if avg_buy is not None else None,
        directional_acc   = round(dir_a, 2)  if dir_a  is not None else None,
        avg_mape_pct      = round(avg_m, 4)  if avg_m  is not None else None,
        avg_rmse          = round(avg_r, 4)  if avg_r  is not None else None,
        sharpe_ratio      = round(sharpe,4)  if sharpe is not None else None,
    )


# ─────────────────── Grid search (Step 4) ───────────────────────────────────

def run_grid_search(df_full, known_reals, cash, commission, edge_pct,
                    trend_filter, exit_strategy, n_windows, artifacts):
    """
    Sweep lookback × horizon using the fast EWMA baseline.
    Ranks configs by Sharpe ratio (or win-rate when trades < 5).
    Saves grid_search_results.csv and prints a ranked table.
    """
    lookbacks = [60, 120, 168, 240]
    horizons  = [12, 24, 48]
    rows = []
    t_max_global = int(df_full.time_idx.max())

    print("\n── Grid search: EWMA baseline across lookback × horizon ──")
    print(f"  {'lookback':>8} {'horizon':>8} {'windows':>8} {'buys':>6} "
          f"{'win%':>7} {'total_pnl':>10} {'dir_acc':>8} {'sharpe':>8}")
    print("  " + "─" * 64)

    for lb, hz in product(lookbacks, horizons):
        t_min = lb
        t_max = t_max_global - hz
        if t_max <= t_min:
            continue
        dpts = sorted(set(np.linspace(t_min, t_max, n_windows, dtype=int)))

        trade_rows = []
        for t in dpts:
            ep = float(df_full.loc[df_full.time_idx == t, "target"].iloc[0])
            fut = df_full[(df_full.time_idx > t) & (df_full.time_idx <= t + hz)]
            if len(fut) < hz:
                continue
            af = fut["target"].values[:hz]
            hp = df_full[df_full.time_idx <= t]["target"].values
            fc = ewma_forecast(hp[-720:], hz)
            tr = simulate_trade(ep, fc, af, hp, cash, commission, edge_pct,
                                trend_filter, exit_strategy)
            tr["fc_mape"] = _mape(af, fc)
            tr["fc_rmse"] = _rmse(af, fc)
            trade_rows.append(tr)

        if not trade_rows:
            continue
        tmp = pd.DataFrame(trade_rows)
        s   = aggregate(tmp, f"EWMA(lb={lb},hz={hz})")
        s.update({"lookback": lb, "horizon": hz})
        rows.append(s)

        wr = f"{s['win_rate_pct']:.1f}%" if s["win_rate_pct"] is not None else " n/a"
        sp = f"{s['sharpe_ratio']:.3f}"  if s["sharpe_ratio"] is not None else "  n/a"
        da = f"{s['directional_acc']:.1f}%" if s["directional_acc"] is not None else " n/a"
        print(f"  {lb:>8} {hz:>8} {s['n_windows']:>8} {s['n_buy']:>6} "
              f"{wr:>7} {s['total_pnl_usd']:>10.4f} {da:>8} {sp:>8}")

    if not rows:
        print("  No valid configs found.")
        return None, None

    grid_df = pd.DataFrame(rows).sort_values(
        by=["sharpe_ratio", "win_rate_pct", "total_pnl_usd"],
        ascending=False, na_position="last"
    )
    grid_path = Path(artifacts) / "grid_search_results.csv"
    grid_df.to_csv(grid_path, index=False)

    best = grid_df.iloc[0]
    print(f"\n  ★ Best config: lookback={int(best.lookback)}, horizon={int(best.horizon)}")
    print(f"    win_rate={best.win_rate_pct}%, total_pnl=${best.total_pnl_usd:.4f}, "
          f"sharpe={best.sharpe_ratio}")
    print(f"  Saved grid results → {grid_path}")
    return int(best.lookback), int(best.horizon)


# ────────────────────────────── Main ────────────────────────────────────────

def main():
    args = parse_args()
    seed_everything(args.seed)
    device = pick_device(args.device)
    Path(args.artifacts).mkdir(parents=True, exist_ok=True)

    # ── Load & prepare data ──────────────────────────────────────────────────
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
    keep_disp  = list(set(["group_id","time_idx","target","open_time"] + known_reals))
    df_full    = df_full[[c for c in keep_disp if c in df_full.columns]].dropna().reset_index(drop=True)
    df_full["time_idx"] = range(len(df_full))

    keep_model = list(set(["group_id","time_idx","target"] + known_reals))
    df_model   = df_full[[c for c in keep_model if c in df_full.columns]].copy()

    label_encoder = NaNLabelEncoder().fit(df_full.group_id)

    # ── Grid search (optional) ───────────────────────────────────────────────
    best_lb, best_hz = args.lookback, args.horizon
    if args.grid_search:
        best_lb, best_hz = run_grid_search(
            df_full, known_reals, args.cash, args.commission, args.edge_pct,
            args.trend_filter, args.exit_strategy, args.n_windows, args.artifacts,
        )
        if best_lb is None:
            best_lb, best_hz = args.lookback, args.horizon
        print(f"\nUsing best config for deep backtest: lookback={best_lb}, horizon={best_hz}\n")

    lookback = best_lb
    horizon  = best_hz

    # ── Decision points ──────────────────────────────────────────────────────
    tmax  = int(df_full["time_idx"].max())
    t_min = lookback
    t_max = tmax - horizon
    if t_max <= t_min:
        raise ValueError("Insufficient data for the configured lookback/horizon.")

    decision_points = sorted(set(
        np.linspace(t_min, t_max, args.n_windows, dtype=int).tolist()
    ))
    print(f"Backtesting {len(decision_points)} windows on {dominant} "
          f"(lookback={lookback}, horizon={horizon}, "
          f"exit={args.exit_strategy}, trend_filter={args.trend_filter})")

    # ── Load deep model ──────────────────────────────────────────────────────
    model = None
    ckpt  = args.checkpoint
    if not ckpt:
        ptr = Path(args.artifacts) / "last_checkpoint.txt"
        if ptr.exists():
            ckpt = ptr.read_text().strip()
    if ckpt and Path(ckpt).exists():
        model = load_model(ckpt, device)
        model.eval()
    else:
        print("No model checkpoint found — running EWMA-only backtest.")

    breakeven_pct = (2 * args.commission / args.cash) * 100.0

    # ── Simulate windows ─────────────────────────────────────────────────────
    records = []
    t0_total = time.time()
    for i, t in enumerate(decision_points):
        ep_row = df_full[df_full.time_idx == t]
        if ep_row.empty:
            continue
        entry_price = float(ep_row["target"].iloc[0])
        future_rows = df_full[(df_full.time_idx > t) & (df_full.time_idx <= t + horizon)]
        if len(future_rows) < horizon:
            continue
        actual_future  = future_rows["target"].values[:horizon]
        history_prices = df_full[df_full.time_idx <= t]["target"].values
        open_time      = ep_row["open_time"].iloc[0] if "open_time" in df_full.columns else t

        print(f"  [{i+1:2d}/{len(decision_points)}] t={t:4d}  "
              f"entry={entry_price:,.2f}", end="", flush=True)

        # Deep model forecast
        model_fc = None
        if model is not None:
            t_inf = time.time()
            model_fc = model_predict(
                model, df_model, t, lookback, horizon,
                args.batch_size, label_encoder, device, known_reals
            )
            elapsed = time.time() - t_inf
            print(f" [inf {elapsed:.1f}s]", end="", flush=True)

        # EWMA forecast
        ewma_fc = ewma_forecast(history_prices[-720:], horizon)

        rec = {
            "window":      i,
            "time_idx":    t,
            "entry_time":  str(open_time),
            "entry_price": entry_price,
        }

        for prefix, fc in [("model", model_fc), ("ewma", ewma_fc)]:
            null_rec = {f"{prefix}_{k}": None for k in (
                "action","trend_blocked","fc_best_idx","fc_best_price",
                "fc_gross_ret_pct","actual_exit_price","actual_return_pct",
                "pnl_usd","won","dir_correct","breakeven_pct","threshold_pct",
                "fc_mape","fc_rmse")}
            null_rec[f"{prefix}_pnl_usd"] = 0.0
            if fc is None:
                rec.update(null_rec)
                continue

            trade = simulate_trade(
                entry_price, fc, actual_future, history_prices,
                args.cash, args.commission, args.edge_pct,
                args.trend_filter, args.exit_strategy,
            )
            for k, v in trade.items():
                rec[f"{prefix}_{k}"] = v
            rec[f"{prefix}_fc_mape"] = _mape(actual_future, fc)
            rec[f"{prefix}_fc_rmse"] = _rmse(actual_future, fc)

        parts = []
        for pfx, lbl in [("model", "Model"), ("ewma", "EWMA")]:
            act = rec.get(f"{pfx}_action")
            if act in ("BUY", "HOLD"):
                tb  = rec.get(f"{pfx}_trend_blocked", False)
                pnl = rec.get(f"{pfx}_pnl_usd", 0.0) or 0.0
                tag = "[trend↓]" if tb else f"(${pnl:+.2f})"
                parts.append(f"{lbl}={act}{tag}")
        print("  " + "  ".join(parts))
        records.append(rec)

    elapsed_total = time.time() - t0_total
    print(f"\nBacktest completed in {elapsed_total:.1f}s")

    if not records:
        print("No records produced.")
        return

    df_out     = pd.DataFrame(records)
    trades_path = Path(args.artifacts) / "backtest_trades.csv"
    df_out.to_csv(trades_path, index=False)
    print(f"Saved {len(df_out)} windows → {trades_path}")

    # ── Aggregate ────────────────────────────────────────────────────────────
    summaries = []
    for prefix, label in [("model", "Deep Model (TFT/NHiTS)"), ("ewma", "EWMA+Drift")]:
        req = [f"{prefix}_action", f"{prefix}_pnl_usd", f"{prefix}_won",
               f"{prefix}_dir_correct", f"{prefix}_fc_mape", f"{prefix}_fc_rmse",
               f"{prefix}_actual_return_pct"]
        if not all(c in df_out.columns for c in req):
            continue
        tmp = df_out[[c for c in req + [f"{prefix}_trend_blocked",
                                         f"{prefix}_fc_gross_ret_pct"]
                      if c in df_out.columns]].copy()
        tmp.columns = [c.replace(f"{prefix}_", "") for c in tmp.columns]
        tmp = tmp[tmp.action.isin(["BUY","HOLD"])].dropna(subset=["fc_mape"])
        if tmp.empty:
            continue
        summaries.append(aggregate(tmp, label))

    config_meta = {
        "symbol":              dominant,
        "lookback":            lookback,
        "horizon":             horizon,
        "n_windows":           len(records),
        "exit_strategy":       args.exit_strategy,
        "trend_filter":        args.trend_filter,
        "cash_usd":            args.cash,
        "commission_per_side": args.commission,
        "edge_pct":            args.edge_pct,
        "breakeven_pct":       round(breakeven_pct, 4),
    }
    summary_path = Path(args.artifacts) / "backtest_summary.json"
    with open(summary_path, "w") as f:
        json.dump({"config": config_meta, "results": summaries}, f, indent=2)
    print(f"Saved summary → {summary_path}")

    # ── Console report ───────────────────────────────────────────────────────
    print("\n" + "═" * 66)
    print("BACKTEST SUMMARY")
    print(f"  exit={args.exit_strategy}  trend_filter={args.trend_filter}  "
          f"lookback={lookback}  horizon={horizon}")
    print("═" * 66)
    for s in summaries:
        wr  = f"{s['win_rate_pct']:.1f}%"  if s["win_rate_pct"]    is not None else "n/a"
        sp  = f"{s['sharpe_ratio']:.3f}"   if s["sharpe_ratio"]    is not None else "n/a"
        ab  = f"${s['avg_pnl_per_buy']:+.4f}" if s["avg_pnl_per_buy"] is not None else "n/a"
        da  = f"{s['directional_acc']:.1f}%" if s["directional_acc"] is not None else "n/a"
        m   = f"{s['avg_mape_pct']:.2f}%"  if s["avg_mape_pct"]    is not None else "n/a"
        tb  = s.get("n_trend_blocked", 0)
        print(f"\n  {s['model']}:")
        print(f"    Windows:         {s['n_windows']}  "
              f"(BUY={s['n_buy']}, HOLD={s['n_hold']}, trend-blocked={tb})")
        print(f"    Win rate:        {wr}  ({s['n_wins']}W / {s['n_losses']}L)")
        print(f"    Total P&L:       ${s['total_pnl_usd']:+.4f}")
        print(f"    Avg P&L / BUY:   {ab}")
        print(f"    Dir. accuracy:   {da}")
        print(f"    Avg MAPE:        {m}")
        print(f"    Sharpe ratio:    {sp}")
    print("═" * 66)

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    windows   = np.arange(len(df_out))

    for ax, (pfx, lbl, col) in zip(
        axes,
        [("model", "Deep Model (TFT / N-HiTS)", "steelblue"),
         ("ewma",  "EWMA + Drift baseline",     "darkorange")]
    ):
        pnl_col = f"{pfx}_pnl_usd"
        act_col = f"{pfx}_action"
        won_col = f"{pfx}_won"
        if pnl_col not in df_out.columns:
            ax.text(0.5, 0.5, f"{lbl} — no data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=11)
            continue

        cum = df_out[pnl_col].fillna(0.0).cumsum().values
        ax.plot(windows, cum, color=col, lw=2, zorder=3)
        ax.fill_between(windows, cum, 0, where=(cum >= 0), alpha=0.12, color="green", zorder=2)
        ax.fill_between(windows, cum, 0, where=(cum < 0),  alpha=0.12, color="red",   zorder=2)
        ax.axhline(0, color="black", lw=0.8, ls="--", alpha=0.5)

        buy_m  = df_out[act_col] == "BUY"
        win_m  = df_out[won_col] == True
        loss_m = df_out[won_col] == False
        if buy_m.any():
            ax.scatter(windows[buy_m & win_m],  cum[buy_m & win_m],
                       color="green", s=60, zorder=5, marker="^", label="Win")
            ax.scatter(windows[buy_m & loss_m], cum[buy_m & loss_m],
                       color="red",   s=60, zorder=5, marker="v", label="Loss")

        nb = int(buy_m.sum())
        nw = int((df_out[won_col] == True).sum())
        wr = f"{nw/nb*100:.1f}%" if nb > 0 else "n/a"
        mp = df_out.get(f"{pfx}_fc_mape", pd.Series([np.nan])).mean()
        ax.set_title(
            f"{lbl}  |  Win rate: {wr} ({nw}/{nb})  |  "
            f"Total P&L: ${cum[-1]:+.2f}  |  Avg MAPE: {mp:.2f}%",
            fontsize=10
        )
        ax.set_ylabel("Cumulative P&L (USD)")
        if buy_m.any():
            ax.legend(fontsize=8, loc="upper left")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Window #")
    fig.suptitle(
        f"Consensus Backtest — {dominant}  |  "
        f"lookback={lookback}  horizon={horizon}  "
        f"exit={args.exit_strategy}  trend_filter={args.trend_filter}  "
        f"comm=${args.commission}/side  edge={args.edge_pct}%",
        fontsize=10, fontweight="bold"
    )
    plt.tight_layout()
    plot_path = Path(args.artifacts) / "backtest_pnl.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot → {plot_path}")


if __name__ == "__main__":
    main()
