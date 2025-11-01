#!/usr/bin/env python3
"""
Executive launcher for the crypto forecasting project.

- Presents a simple menu to run the whole pipeline or individual steps.
- Offers sensible defaults and lets you override interactively.
- Also supports non-interactive flags for automation (see --help).

Usage (interactive):
    python scripts/exec.py

Usage (non-interactive one-liners):
    python scripts/exec.py --full
    python scripts/exec.py --fetch --symbols BTCUSDT ETHUSDT --interval 1m --days 3
    python scripts/exec.py --features --interval 1m
    python scripts/exec.py --train --lookback 168 --horizon 24 --epochs 15
    python scripts/exec.py --eval --lookback 168 --horizon 24 --folds 3
    python scripts/exec.py --live --symbol btcusdt --max-msgs 200
"""
import argparse
import os
import sys
import subprocess
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT_ROOT / "scripts"
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
ARTIFACTS = PROJECT_ROOT / "artifacts"

PY = sys.executable  # current Python interpreter

def run(cmd: List[str]) -> int:
    print("\n=> Running:", " ".join(cmd))
    try:
        rc = subprocess.call(cmd, cwd=PROJECT_ROOT)
        if rc != 0:
            print(f"\nStep failed with exit code {rc}. Aborting pipeline.")
            sys.exit(rc)
        return rc
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(130)

def auto_checkpoint() -> str:
    """Pick latest checkpoint in artifacts/ if available."""
    if not ARTIFACTS.exists():
        return ""
    ckpts = sorted(ARTIFACTS.glob("*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return str(ckpts[0]) if ckpts else ""

def do_fetch(symbols: List[str], interval: str, days: int = None, limit: int = None):
    cmd = [PY, str(SCRIPTS / "setup_data.py"), "--symbols", *symbols, "--interval", interval]
    if days is not None:
        cmd += ["--days", str(days)]
    if limit is not None:
        cmd += ["--limit", str(limit)]
    return run(cmd)

def do_features(interval: str):
    cmd = [PY, str(SCRIPTS / "make_features.py"), "--src", "data/raw", "--dest", "data/processed", "--interval", interval]
    return run(cmd)

def do_train(lookback: int, horizon: int, epochs: int, batch_size: int = 128, device: str = "auto"):
    cmd = [PY, str(SCRIPTS / "train_tft.py"),
           "--data", "data/processed/merged.parquet",
           "--lookback", str(lookback),
           "--horizon", str(horizon),
           "--epochs", str(epochs),
           "--batch-size", str(batch_size),
           "--device", device]
    return run(cmd)

def do_eval(lookback: int, horizon: int, folds: int, checkpoint: str = ""):
    ck = checkpoint or auto_checkpoint()
    if not ck:
        print("⚠️  No checkpoint found; evaluation will run naive-only metrics.")
    cmd = [PY, str(SCRIPTS / "evaluate.py"),
           "--data", "data/processed/merged.parquet",
           "--lookback", str(lookback),
           "--horizon", str(horizon),
           "--folds", str(folds)]
    if ck:
        cmd += ["--checkpoint", ck]
    return run(cmd)

def do_live(symbol: str, max_msgs: int):
    cmd = [PY, str(SCRIPTS / "live_stream.py"), "--symbol", symbol, "--max-msgs", str(max_msgs)]
    return run(cmd)

def interactive():
    print("=== Crypto Forecasting Executive ===")
    print("1) Full pipeline (fetch → features → train → evaluate)")
    print("2) Fetch data")
    print("3) Build features")
    print("4) Train TFT")
    print("5) Evaluate (rolling-origin, naive vs TFT)")
    print("6) Live stream ticks")
    print("7) Show latest checkpoint path")
    print("0) Exit")
    choice = input("Select an option: ").strip()

    if choice == "1":
        syms = input("Symbols [BTCUSDT ETHUSDT]: ").strip() or "BTCUSDT ETHUSDT"
        interval = input("Interval [1m]: ").strip() or "1m"
        days = input("Days of history [3]: ").strip() or "3"
        lookback = input("Lookback (encoder length) [168]: ").strip() or "168"
        horizon = input("Horizon (forecast steps) [24]: ").strip() or "24"
        epochs = input("Epochs [15]: ").strip() or "15"
        folds = input("Eval folds [3]: ").strip() or "3"

        do_fetch(syms.split(), interval, int(days), None)
        do_features(interval)
        do_train(int(lookback), int(horizon), int(epochs))
        ck = auto_checkpoint()
        if not ck:
            print("No checkpoint found; evaluation will run without TFT comparisons.")
        else:
            print(f"Using checkpoint: {ck}")
        do_eval(int(lookback), int(horizon), int(folds), ck)
        print("\nDone.")
        return

    if choice == "2":
        syms = input("Symbols [BTCUSDT ETHUSDT]: ").strip() or "BTCUSDT ETHUSDT"
        interval = input("Interval [1m]: ").strip() or "1m"
        days = input("Days (leave blank to use --limit): ").strip()
        limit = None
        if days == "":
            limit = input("Limit (most recent candles) [1500]: ").strip() or "1500"
            do_fetch(syms.split(), interval, None, int(limit))
        else:
            do_fetch(syms.split(), interval, int(days), None)
        return

    if choice == "3":
        interval = input("Interval [1m]: ").strip() or "1m"
        do_features(interval)
        return

    if choice == "4":
        lookback = input("Lookback [168]: ").strip() or "168"
        horizon  = input("Horizon [24]: ").strip() or "24"
        epochs   = input("Epochs [15]: ").strip() or "15"
        batch    = input("Batch size [128]: ").strip() or "128"
        do_train(int(lookback), int(horizon), int(epochs), int(batch))
        return

    if choice == "5":
        lookback = input("Lookback [168]: ").strip() or "168"
        horizon  = input("Horizon [24]: ").strip() or "24"
        folds    = input("Folds [3]: ").strip() or "3"
        ck = input(f"Checkpoint path (blank to auto) [{auto_checkpoint()}]: ").strip() or auto_checkpoint()
        do_eval(int(lookback), int(horizon), int(folds), ck)
        return

    if choice == "6":
        symbol   = input("Symbol [btcusdt]: ").strip() or "btcusdt"
        max_msgs = input("Max messages [200]: ").strip() or "200"
        do_live(symbol, int(max_msgs))
        return

    if choice == "7":
        ck = auto_checkpoint()
        print("Latest checkpoint:", ck if ck else "(none)")
        return

    if choice == "0":
        print("Bye.")
        return

    print("Invalid selection.")

def main():
    parser = argparse.ArgumentParser()
    g = parser.add_mutually_exclusive_group()
    g.add_argument("--full", action="store_true", help="Run full pipeline")
    g.add_argument("--fetch", action="store_true", help="Only fetch data")
    g.add_argument("--features", action="store_true", help="Only build features")
    g.add_argument("--train", action="store_true", help="Only train TFT")
    g.add_argument("--eval", action="store_true", help="Only evaluate")
    g.add_argument("--live", action="store_true", help="Only live stream")

    # shared / step-specific params
    parser.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT"])
    parser.add_argument("--interval", default="1m")
    parser.add_argument("--days", type=int)
    parser.add_argument("--limit", type=int)

    parser.add_argument("--lookback", type=int, default=168)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--checkpoint")

    parser.add_argument("--symbol", default="btcusdt")
    parser.add_argument("--max-msgs", type=int, default=200)
    parser.add_argument("--device", choices=["auto","cpu","mps","gpu"], default="auto",
                    help="Which device to use for training (passed to train_tft.py)")

    args = parser.parse_args()

    # Interactive if no flags
    if not any([args.full, args.fetch, args.features, args.train, args.eval, args.live]):
        return interactive()

    if args.full:
        do_fetch(args.symbols, args.interval, args.days, args.limit)
        do_features(args.interval)
        do_train(args.lookback, args.horizon, args.epochs, args.batch_size, args.device)
        do_eval(args.lookback, args.horizon, args.folds, args.checkpoint)

    if args.fetch:
        return do_fetch(args.symbols, args.interval, args.days, args.limit)

    if args.features:
        return do_features(args.interval)

    if args.train:
        do_train(args.lookback, args.horizon, args.epochs, args.batch_size, args.device)

    if args.eval:
        ck = args.checkpoint or auto_checkpoint()
        return do_eval(args.lookback, args.horizon, args.folds, ck)

    if args.live:
        return do_live(args.symbol, args.max_msgs)

if __name__ == "__main__":
    main()
