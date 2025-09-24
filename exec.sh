#!/usr/bin/env bash
set -euo pipefail

# =========================
# Defaults (override via CLI)
# =========================
SYMBOLS=("BTCUSDT" "ETHUSDT")
INTERVAL="1m"
DAYS=3
LOOKBACK=168
HORIZON=24
EPOCHS=15
FOLDS=3
BATCH_SIZE=64             # smaller default for mac stability
TRAIN_DEVICE="cpu"        # cpu | mps | gpu | auto
CHECKPOINT=""             # path to .ckpt for --eval stage
TIMEFMT="%Y-%m-%d %H:%M"  # for plot_evaluation datetime ticks
SKIP_VENV=0               # set to 1 to skip venv step
PYTHON="${PYTHON:-python3}"

# <<< consensus: defaults for live consensus
CONS_SYMBOL=""            # if empty, will use first of SYMBOLS[]
CONS_HORIZON_M=60         # minutes ahead for consensus forecast
CONS_CASH=100             # USD mock capital
CONS_COMM=2               # USD commission per side
CONS_EDGE=0.5             # extra % over breakeven
TFT_HOOK_CMD=""           # optional: hook to your TFT pipeline command

# <<< consensus-plot: defaults for plotting
CONS_PAD_MIN=10           # minutes padding around forecast window for plotting

# Stages toggles (default: full if none given)
DO_FETCH=0
DO_FEATURES=0
DO_TRAINTEE=0 # (typo guard not used)
DO_TRAIN=0
DO_EVAL=0
DO_PLOTS=0
DO_CONSENSUS=0           # <<< consensus
DO_CONS_EVAL=0           # <<< consensus
DO_CONS_PLOT=0           # <<< consensus-plot

# =========================
# Helpers
# =========================
usage() {
  cat <<'USAGE'
Usage: ./run.sh [options]

Stages (pick any; default is --full if none chosen):
  --full                 Run fetch -> features -> train -> eval -> plots
  --fetch                Only fetch raw data (setup_data.py)
  --features             Only build features (make_features.py)
  --train                Only train TFT (train_tft.py)
  --eval                 Only evaluate (evaluate.py)
  --plots                Only make evaluation plots (plot_evaluation.py)
  --consensus-live       Run mock consensus decision now (scripts/consensus_live.py)          # <<< consensus
  --consensus-eval       Evaluate realized PnL for last consensus (scripts/evaluate_pnl.py)   # <<< consensus
  --consensus-plot       Plot consensus forecast/actuals & PnL (scripts/plot_consensus.py)    # <<< consensus-plot

Knobs (override defaults):
  --symbols "BTCUSDT ETHUSDT"   Symbols (space-separated, quote the list)
  --interval 1m                 Candle interval
  --days 3                      Days of history to fetch
  --lookback 168                Encoder length (hours)
  --horizon 24                  Prediction length (hours)
  --epochs 15                   Training epochs
  --folds 3                     Rolling folds for evaluation
  --batch-size 64               Batch size for train/eval
  --device cpu|mps|gpu|auto     Device for training/prediction
  --checkpoint path.ckpt        Checkpoint for evaluation plots (optional for --eval)
  --timefmt "%Y-%m-%d %H:%M"    Datetime format for plots
  --skip-venv                   Skip venv creation/activation (use current env)

Consensus knobs (for --consensus-live):                                                     # <<< consensus
  --cons-symbol BTCUSDT          Override symbol (defaults to first of --symbols)
  --cons-horizon-m 60            Forecast horizon in minutes
  --cons-cash 100                Mock capital in USD
  --cons-comm 2                  Commission per side in USD
  --cons-edge 0.5                Extra edge (%) over breakeven required to trade
  --tft-hook-cmd "python ..."    Optional TFT pipeline command to produce preds

Plotting knobs (for --consensus-plot):                                                      # <<< consensus-plot
  --cons-pad-min 10              Minutes of padding around forecast window on the chart

Other:
  -h, --help                    Show this help

Examples:
  ./run.sh --full --device cpu
  ./run.sh --fetch --symbols "BTCUSDT ETHUSDT" --interval 1m --days 3
  ./run.sh --features
  ./run.sh --train --epochs 10 --batch-size 128 --device mps
  ./run.sh --eval --folds 5 --checkpoint artifacts/tft-epoch=08-val_loss=3.0475.ckpt --device cpu
  ./run.sh --plots --timefmt "%Y-%m-%d %H:%M"
  ./run.sh --consensus-live --cons-symbol BTCUSDT --cons-horizon-m 60 --cons-edge 0.5          # <<< consensus
  ./run.sh --consensus-eval                                                                     # <<< consensus
  ./run.sh --consensus-plot --cons-pad-min 12                                                   # <<< consensus-plot
USAGE
}

# join array by space (for exec.py)
join_by() { local IFS="$1"; shift; echo "$*"; }

# =========================
# Parse CLI
# =========================
while [[ $# -gt 0 ]]; do
  case "$1" in
    --full)      DO_FETCH=1; DO_FEATURES=1; DO_TRAIN=1; DO_EVAL=1; DO_PLOTS=1; shift ;;
    --fetch)     DO_FETCH=1; shift ;;
    --features)  DO_FEATURES=1; shift ;;
    --train)     DO_TRAIN=1; shift ;;
    --eval)      DO_EVAL=1; shift ;;
    --plots)     DO_PLOTS=1; shift ;;

    --consensus-live) DO_CONSENSUS=1; shift ;;            # <<< consensus
    --consensus-eval) DO_CONS_EVAL=1; shift ;;            # <<< consensus
    --consensus-plot) DO_CONS_PLOT=1; shift ;;            # <<< consensus-plot

    --symbols)   read -r -a SYMBOLS <<< "$2"; shift 2 ;;
    --interval)  INTERVAL="$2"; shift 2 ;;
    --days)      DAYS="$2"; shift 2 ;;
    --lookback)  LOOKBACK="$2"; shift 2 ;;
    --horizon)   HORIZON="$2"; shift 2 ;;
    --epochs)    EPOCHS="$2"; shift 2 ;;
    --folds)     FOLDS="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --device)    TRAIN_DEVICE="$2"; shift 2 ;;
    --checkpoint) CHECKPOINT="$2"; shift 2 ;;
    --timefmt)   TIMEFMT="$2"; shift 2 ;;
    --skip-venv) SKIP_VENV=1; shift ;;

    # <<< consensus: parse knobs
    --cons-symbol)    CONS_SYMBOL="$2"; shift 2 ;;
    --cons-horizon-m) CONS_HORIZON_M="$2"; shift 2 ;;
    --cons-cash)      CONS_CASH="$2"; shift 2 ;;
    --cons-comm)      CONS_COMM="$2"; shift 2 ;;
    --cons-edge)      CONS_EDGE="$2"; shift 2 ;;
    --tft-hook-cmd)   TFT_HOOK_CMD="$2"; shift 2 ;;

    # <<< consensus-plot: parse knob
    --cons-pad-min)   CONS_PAD_MIN="$2"; shift 2 ;;

    -h|--help)   usage; exit 0 ;;

    *) echo "Unknown arg: $1"; usage; exit 1 ;;
  esac
done

# If no stage was specified, default to full
if [[ $DO_FETCH -eq 0 && $DO_FEATURES -eq 0 && $DO_TRAIN -eq 0 && $DO_EVAL -eq 0 && $DO_PLOTS -eq 0 && $DO_CONSENSUS -eq 0 && $DO_CONS_EVAL -eq 0 && $DO_CONS_PLOT -eq 0 ]]; then
  DO_FETCH=1; DO_FEATURES=1; DO_TRAIN=1; DO_EVAL=1; DO_PLOTS=1
fi

# =========================
# Environment setup
# =========================
if [[ $SKIP_VENV -eq 0 ]]; then
  "$PYTHON" -m venv .venv
  # shellcheck disable=SC1091
  source .venv/bin/activate

  pip install --upgrade pip
  # remove legacy lightning to avoid clashes
  pip uninstall -y pytorch-lightning || true
  # core deps
  pip install "lightning>=2.2" "pandas>=2.0" "pyarrow" "numpy" "matplotlib" "requests"
  # pytorch-forecasting (pick a version compatible with your Python)
  pip install "pytorch-forecasting>=1.3.0" || true
  # optional project extras
  if [[ -f requirements.txt ]]; then
    pip install -r requirements.txt || true
  fi

  # Apple Silicon: safer MPS fallback
  export PYTORCH_ENABLE_MPS_FALLBACK=1
fi

# =========================
# Stage functions
# =========================
run_fetch() {
  echo "==> Fetching raw data"
  python scripts/setup_data.py \
    --symbols "$(join_by ' ' "${SYMBOLS[@]}")" \
    --interval "$INTERVAL" \
    --days "$DAYS"
}

run_features() {
  echo "==> Building features"
  python scripts/make_features.py \
    --src data/raw \
    --dest data/processed \
    --interval "$INTERVAL"
}

run_train() {
  echo "==> Training TFT"
  python scripts/train_tft.py \
    --data data/processed/merged.parquet \
    --lookback "$LOOKBACK" \
    --horizon "$HORIZON" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --device "$TRAIN_DEVICE"
}

run_eval() {
  echo "==> Evaluating (rolling-origin)"
  if [[ -n "$CHECKPOINT" ]]; then
    python scripts/evaluate.py \
      --data data/processed/merged.parquet \
      --lookback "$LOOKBACK" \
      --horizon "$HORIZON" \
      --folds "$FOLDS" \
      --device "$TRAIN_DEVICE" \
      --batch-size "$BATCH_SIZE" \
      --checkpoint "$CHECKPOINT"
  else
    python scripts/evaluate.py \
      --data data/processed/merged.parquet \
      --lookback "$LOOKBACK" \
      --horizon "$HORIZON" \
      --folds "$FOLDS" \
      --device "$TRAIN_DEVICE" \
      --batch-size "$BATCH_SIZE"
  fi
}

run_plots() {
  echo "==> Plotting evaluation"
  if [[ -f artifacts/time_labels.csv ]]; then
    python scripts/plot_evaluation.py \
      --csv artifacts/evaluation.csv \
      --outdir artifacts \
      --data data/processed/merged.parquet \
      --labels artifacts/time_labels.csv \
      --timefmt "$TIMEFMT"
  else
    python scripts/plot_evaluation.py \
      --csv artifacts/evaluation.csv \
      --outdir artifacts \
      --data data/processed/merged.parquet \
      --timefmt "$TIMEFMT"
  fi
}

# <<< consensus
run_consensus_live() {
  echo "==> Running consensus_live (single-shot decision)"
  local sym="${CONS_SYMBOL:-${SYMBOLS[0]}}"
  local lookback_m=$(( LOOKBACK * 60 ))   # map hours -> minutes to align with consensus script
  if [[ -n "$TFT_HOOK_CMD" ]]; then
    python scripts/consensus_live.py \
      --symbol "$sym" \
      --interval "$INTERVAL" \
      --lookback_m "$lookback_m" \
      --horizon_m "$CONS_HORIZON_M" \
      --start_cash_usd "$CONS_CASH" \
      --commission_per_side_usd "$CONS_COMM" \
      --min_edge_pct "$CONS_EDGE" \
      --tft_hook_cmd "$TFT_HOOK_CMD"
  else
    python scripts/consensus_live.py \
      --symbol "$sym" \
      --interval "$INTERVAL" \
      --lookback_m "$lookback_m" \
      --horizon_m "$CONS_HORIZON_M" \
      --start_cash_usd "$CONS_CASH" \
      --commission_per_side_usd "$CONS_COMM" \
      --min_edge_pct "$CONS_EDGE"
  fi
  echo "==> Wrote artifacts/mock_trade.json and artifacts/forecast_path.csv"
}

run_consensus_eval() {
  echo "==> Evaluating consensus decision"
  python scripts/evaluate_pnl.py --artifact artifacts/mock_trade.json
  echo "==> Wrote artifacts/mock_eval.json"
}

# <<< consensus-plot
run_consensus_plot() {
  echo "==> Plotting consensus trade"
  python scripts/plot_consensus.py \
    --artifact artifacts/mock_trade.json \
    --eval artifacts/mock_eval.json \
    --outdir artifacts \
    --pad-min "$CONS_PAD_MIN"
  echo "==> Wrote artifacts/consensus_plot.png and artifacts/consensus_plot.pdf"
}

# =========================
# Execute selected stages
# =========================
[[ $DO_FETCH      -eq 1 ]] && run_fetch
[[ $DO_FEATURES   -eq 1 ]] && run_features
[[ $DO_TRAIN      -eq 1 ]] && run_train
[[ $DO_EVAL       -eq 1 ]] && run_eval
[[ $DO_PLOTS      -eq 1 ]] && run_plots
[[ $DO_CONSENSUS  -eq 1 ]] && run_consensus_live      # <<< consensus
[[ $DO_CONS_EVAL  -eq 1 ]] && run_consensus_eval      # <<< consensus
[[ $DO_CONS_PLOT  -eq 1 ]] && run_consensus_plot      # <<< consensus-plot

echo
echo "Artifacts:"
echo "  - Checkpoints: artifacts/*.ckpt"
echo "  - Metrics:     artifacts/evaluation.csv"
echo "  - Plots:       artifacts/qualitative_forecast.png, artifacts/eval_*.{png,pdf}"
echo "  - Data:        data/raw/, data/processed/merged.parquet"
echo "  - Consensus:   artifacts/mock_trade.json, artifacts/mock_eval.json, artifacts/consensus_plot.{png,pdf}"   # <<< consensus-plot