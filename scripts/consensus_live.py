#!/usr/bin/env python3
"""
End-to-end "mock consensus" trading decision from *now*:
1) Collect recent 1m OHLCV
2) Make a short-horizon forecast (60 min by default)
   - Uses a simple fallback model (EWMA + drift).
   - If you have your TFT pipeline, you can plug it in (see hook).
3) Decide whether to buy with $100 mock cash, accounting for $2 commission per buy and per sell.
4) Save a decision file artifacts/mock_trade.json with all details.

Usage:
  python scripts/consensus_live.py --symbol BTCUSDT --interval 1m --lookback_m 720 --horizon_m 60

Notes:
- Commission model is fixed: $2 per BUY and $2 per SELL (flat).
- Single-shot strategy: either (a) buy now and plan to sell at the forecasted best minute, or (b) do nothing.
- This is a mock/sim only. Not financial advice.
"""

import argparse
import json
from dataclasses import dataclass, asdict
from typing import List, Tuple

import numpy as np
import pandas as pd
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from forecast import ensure_utc, fetch_recent_klines


@dataclass
class Config:
    symbol: str = "BTCUSDT"
    interval: str = "1m"
    lookback_m: int = 720
    horizon_m: int = 60
    start_cash_usd: float = 100.0
    commission_per_side_usd: float = 2.0
    tft_hook_cmd: str = ""  # e.g., 'python scripts/exec.py --full --symbols BTCUSDT ...'
    min_edge_pct: float = 0.5  # extra safety over breakeven (%)


@dataclass
class ForecastResult:
    t0_iso: str
    horizon_m: int
    times: List[str]  # ISO minute stamps
    price_forecast: List[float]  # forecasted close prices
    model_name: str


@dataclass
class Decision:
    action: str  # "BUY" or "HOLD"
    reason: str
    entry_time: str
    entry_price: float
    target_time: str
    target_price: float
    expected_return_pct: float
    expected_pnl_usd: float
    commission_total_usd: float


def ewma_forecast(prices: np.ndarray, horizon: int) -> np.ndarray:
    """
    Super-simple fallback forecaster:
    - EWMA to smooth noise
    - Linear drift using last 30m momentum (if available)
    Returns horizon-step ahead path forecast (minute-by-minute).
    """
    if len(prices) < 5:
        return np.full(horizon, prices[-1] if len(prices) else np.nan)
    # EWMA smoothing
    alpha = 2/(min(60, max(5, len(prices)))+1)  # ~60-min EWMA cap
    ewma = []
    s = prices[0]
    for p in prices:
        s = alpha*p + (1-alpha)*s
        ewma.append(s)
    ewma = np.array(ewma)
    last = ewma[-1]
    # Drift: slope over last 30 points (or all, min 10)
    window = min(30, max(10, len(ewma)))
    y = ewma[-window:]
    x = np.arange(window)
    slope = np.polyfit(x, y, 1)[0] if window >= 2 else 0.0  # price per minute
    path = last + slope * (np.arange(1, horizon+1))
    return path


def tft_hook_forecast(cfg: Config, history: pd.DataFrame) -> Tuple[str, np.ndarray]:
    """
    If you have a TFT pipeline that outputs a per-minute forecast vector,
    integrate it here. For now we return None to use fallback.
    """
    # Example stub: run external command that writes artifacts/forecast.csv with a 'close_pred' column
    # import subprocess, shlex
    # subprocess.run(shlex.split(cfg.tft_hook_cmd), check=True)
    # art = pd.read_csv("artifacts/forecast.csv")
    # preds = art["close_pred"].to_numpy()[: cfg.horizon_m]
    # return "TFT", preds
    return "EWMA+Drift", None


def decide_trade(cfg: Config, now_price: float, forecast: np.ndarray,
                 start_ts: pd.Timestamp) -> Decision:
    """
    Decide: buy now and sell at predicted best minute if the *net* edge beats commissions.
    Breakeven % = 2*commission / cash. Add cfg.min_edge_pct safety.
    """
    if np.isnan(forecast).any():
        return Decision("HOLD", "Forecast invalid (NaN).",
                        start_ts.isoformat(), now_price, "", float("nan"), 0.0, 0.0, 0.0)

    best_idx = int(np.argmax(forecast))
    best_price = float(forecast[best_idx])
    gross_ret = (best_price - now_price)/now_price * 100.0

    breakeven_pct = (2*cfg.commission_per_side_usd / cfg.start_cash_usd) * 100.0  # e.g., $4 on $100 = 4%
    needed = breakeven_pct + cfg.min_edge_pct

    if gross_ret >= needed:
        exp_pnl = (cfg.start_cash_usd * (best_price/now_price - 1.0)) - 2*cfg.commission_per_side_usd
        tgt_time = (start_ts + pd.Timedelta(minutes=best_idx+1)).isoformat()
        return Decision(
            action="BUY",
            reason=f"Forecasted peak +{gross_ret:.2f}% >= breakeven {breakeven_pct:.2f}% + safety {cfg.min_edge_pct:.2f}%.",
            entry_time=start_ts.isoformat(),
            entry_price=float(now_price),
            target_time=tgt_time,
            target_price=best_price,
            expected_return_pct=gross_ret - breakeven_pct,
            expected_pnl_usd=float(exp_pnl),
            commission_total_usd=2*cfg.commission_per_side_usd
        )
    else:
        return Decision("HOLD",
                        f"Forecasted peak +{gross_ret:.2f}% < threshold {needed:.2f}% (breakeven {breakeven_pct:.2f}% + safety).",
                        start_ts.isoformat(), now_price, "", float("nan"), 0.0, 0.0, 0.0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument("--interval", default="1m")
    p.add_argument("--lookback_m", type=int, default=720)
    p.add_argument("--horizon_m", type=int, default=60)
    p.add_argument("--start_cash_usd", type=float, default=100.0)
    p.add_argument("--commission_per_side_usd", type=float, default=2.0)
    p.add_argument("--min_edge_pct", type=float, default=0.5)
    p.add_argument("--tft_hook_cmd", default="")
    args = p.parse_args()

    cfg = Config(
        symbol=args.symbol,
        interval=args.interval,
        lookback_m=args.lookback_m,
        horizon_m=args.horizon_m,
        start_cash_usd=args.start_cash_usd,
        commission_per_side_usd=args.commission_per_side_usd,
        tft_hook_cmd=args.tft_hook_cmd,
        min_edge_pct=args.min_edge_pct
    )

    Path("artifacts").mkdir(parents=True, exist_ok=True)

    # 1) Fetch history
    limit = max(3, min(1000, cfg.lookback_m + 1))
    df = fetch_recent_klines(cfg.symbol, cfg.interval, limit=limit)
    if len(df) < 10:
        raise RuntimeError("Insufficient data returned.")

    now_row = df.iloc[-1]
    now_price = float(now_row["close"])
    start_ts = ensure_utc(now_row["close_time"])

    # 2) Forecast
    model_name, preds = tft_hook_forecast(cfg, df)
    if preds is None or len(preds) < cfg.horizon_m:
        preds = ewma_forecast(df["close"].to_numpy(), cfg.horizon_m)
        model_name = "EWMA+Drift"

    # 3) Decision
    dec = decide_trade(cfg, now_price, preds[:cfg.horizon_m], start_ts)

    # Save forecast and decision
    fr = ForecastResult(
        t0_iso=start_ts.isoformat(),
        horizon_m=cfg.horizon_m,
        times=[(start_ts + pd.Timedelta(minutes=i+1)).isoformat() for i in range(cfg.horizon_m)],
        price_forecast=[float(x) for x in preds[:cfg.horizon_m]],
        model_name=model_name
    )

    out = {
        "config": asdict(cfg),
        "forecast": asdict(fr),
        "decision": asdict(dec)
    }
    with open("artifacts/mock_trade.json", "w") as f:
        json.dump(out, f, indent=2)

    # Also save CSV of forecast
    pd.DataFrame({"time": fr.times, "price_fc": fr.price_forecast}).to_csv("artifacts/forecast_path.csv", index=False)

    print(f"[OK] Decision saved -> artifacts/mock_trade.json")
    print(f"Action: {dec.action} | Reason: {dec.reason}")
    if dec.action == "BUY":
        print(f"Entry @ {dec.entry_price:.2f} at {dec.entry_time} (UTC)")
        print(f"Target @ {dec.target_price:.2f} at {dec.target_time} (UTC)")
        print(f"Expected PnL: ${dec.expected_pnl_usd:.2f} (net of ${dec.commission_total_usd:.2f} commissions)")


if __name__ == "__main__":
    main()
