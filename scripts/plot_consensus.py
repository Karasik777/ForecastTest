#!/usr/bin/env python3
"""
plot_consensus.py
-----------------
Create publication-ready plots for the mock consensus trade:
- Actual price (close) over the evaluation window (+ a little padding).
- Forecast path (from artifacts/forecast_path.csv) aligned to its t0.
- Entry/Exit markers if a BUY trade occurred.
- Annotations: realized PnL, return %, commissions, partial-window flag.

Usage:
  python scripts/plot_consensus.py \
    --artifact artifacts/mock_trade.json \
    --eval artifacts/mock_eval.json \
    --outdir artifacts \
    --pad-min 10

Outputs:
  - artifacts/consensus_plot.png
  - artifacts/consensus_plot.pdf

Requires:
  - matplotlib, pandas, numpy, requests installed (run.sh env already installs requests, pandas, matplotlib).
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from forecast import ensure_utc, iso_to_ms, fetch_range_klines


def load_artifacts(artifact_path: str, eval_path: str | None):
    with open(artifact_path, "r") as f:
        trade = json.load(f)

    cfg = trade["config"]
    fc  = trade["forecast"]
    dec = trade["decision"]

    evald = None
    if eval_path is not None and Path(eval_path).exists():
        with open(eval_path, "r") as f:
            evald = json.load(f)

    # Forecast path CSV (optional but recommended)
    fc_csv = Path("artifacts/forecast_path.csv")
    fc_df = pd.read_csv(fc_csv) if fc_csv.exists() else None
    if fc_df is not None:
        fc_df["time"] = pd.to_datetime(fc_df["time"], utc=True)

    return cfg, fc, dec, evald, fc_df


def build_context_times(fc, dec, pad_min: int):
    t0  = ensure_utc(fc["t0_iso"])
    t_end = ensure_utc(fc["times"][-1]) if fc.get("times") else t0 + pd.Timedelta(minutes=60)
    # Window padding
    start = (ensure_utc(dec["entry_time"]) if dec["entry_time"] else t0) - pd.Timedelta(minutes=pad_min)
    end   = t_end + pd.Timedelta(minutes=pad_min)
    return start, end, t0, t_end


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", default="artifacts/mock_trade.json")
    ap.add_argument("--eval", default="artifacts/mock_eval.json")
    ap.add_argument("--outdir", default="artifacts")
    ap.add_argument("--pad-min", type=int, default=10, help="minutes padding around the forecast window")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    cfg, fc, dec, evald, fc_df = load_artifacts(args.artifact, args.eval)

    symbol   = cfg.get("symbol", "BTCUSDT")
    interval = cfg.get("interval", "1m")
    cash     = float(cfg.get("start_cash_usd", 100.0))
    comm     = float(cfg.get("commission_per_side_usd", 2.0))

    start, end, t0, t_end = build_context_times(fc, dec, args.pad_min)

    # Fetch market closes for context
    mkt = fetch_range_klines(symbol, interval, iso_to_ms(start.isoformat()), iso_to_ms(end.isoformat()))
    if mkt.empty:
        raise RuntimeError("No market data returned for plotting window.")

    # Prepare series
    t_close = mkt["close_time"]
    p_close = mkt["close"]

    # Prepare forecast series (aligned to its own time index)
    if fc_df is not None and not fc_df.empty:
        t_fc = fc_df["time"]
        p_fc = fc_df["price_fc"]
    else:
        # Fallback: reconstruct from forecast dict if CSV missing
        t_fc = pd.to_datetime(fc["times"], utc=True) if fc.get("times") else pd.DatetimeIndex([])
        p_fc = pd.Series(fc["price_forecast"]) if fc.get("price_forecast") else pd.Series(dtype=float)

    # Start plotting
    plt.figure(figsize=(10.5, 6.0), dpi=150)
    ax = plt.gca()

    ax.plot(t_close, p_close, linewidth=1.5, label=f"{symbol} close (actual)")
    if len(t_fc) > 0:
        ax.plot(t_fc, p_fc, linewidth=1.5, linestyle="--", label="Forecast path (next hour)")

    # Mark t0 and forecast end
    ax.axvline(t0, linestyle=":", linewidth=1.0)
    ax.axvline(t_end, linestyle=":", linewidth=1.0)

    # Entry/Exit markers if BUY
    trade_text_lines = []
    if dec.get("action") == "BUY":
        entry_ts = ensure_utc(dec["entry_time"])
        ax.scatter([entry_ts], [dec["entry_price"]], marker="^", s=60, label="Entry (forecast)")
        ax.annotate("Entry", (entry_ts, dec["entry_price"]), xytext=(5, 10), textcoords="offset points")

        tgt_ts = ensure_utc(dec["target_time"])
        ax.scatter([tgt_ts], [dec["target_price"]], marker="v", s=60, label="Target (forecast)")
        ax.annotate("Target", (tgt_ts, dec["target_price"]), xytext=(5, -15), textcoords="offset points")

    # If evaluation exists, plot realized exit
    if evald is not None and "sell_time" in evald:
        sell_ts = ensure_utc(evald["sell_time"])
        sell_fill = float(evald["sell_fill"])
        ax.scatter([sell_ts], [sell_fill], marker="o", s=52, label="Exit (realized)")
        trade_text_lines.append(f"Realized PnL: ${evald['realized_pnl_usd']:.2f} ({evald['realized_return_pct']:.2f}%)")
        trade_text_lines.append(f"Units: {evald['units_bought']:.8f} | Commissions: ${evald['commission_total_usd']:.2f}")
        trade_text_lines.append(f"Partial window: {evald.get('was_partial_window', False)}")

    # Add summary box (even for HOLD)
    if dec.get("action") == "BUY":
        trade_text_lines.insert(0, "Decision: BUY")
    else:
        trade_text_lines.insert(0, f"Decision: HOLD — {dec.get('reason','')}")

    trade_text_lines.append(f"Breakeven ≈ {(2*comm/cash)*100:.2f}%  | Safety +{cfg.get('min_edge_pct',0.5):.2f}%")
    textstr = "\n".join(trade_text_lines)

    bbox_props = dict(boxstyle="round,pad=0.4", alpha=0.12, linewidth=0.8)
    ax.text(0.01, 0.99, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment="top", bbox=bbox_props)

    ax.set_title(f"Mock Consensus — {symbol} ({interval})")
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("Price (close)")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.25)

    plt.tight_layout()
    png = outdir / "consensus_plot.png"
    pdf = outdir / "consensus_plot.pdf"
    plt.savefig(png)
    plt.savefig(pdf)
    print(f"[OK] Wrote {png}")
    print(f"[OK] Wrote {pdf}")


if __name__ == "__main__":
    main()
