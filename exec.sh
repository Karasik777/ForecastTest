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

# Stages toggles (default: full if none given)
DO_FETCH=0
DO_FEATURES=0
DO_TRAIN=0
DO_EVAL=0
DO_PLOTS=0

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

Knobs (override defaults):
  --symbols "BTCUSDT ETHUSDT"   Symbols (space-separated, quote the list)
  --interval 1m                 Candle interval
  --days 3                      Days of history to fetch
  --lookback 168                Encoder length
  --horizon 24                  Prediction length
  --epochs 15                   Training epochs
  --folds 3                     Rolling folds for evaluation
  --batch-size 64               Batch size for train/eval
  --device cpu|mps|gpu|auto     Device for training/prediction
  --checkpoint path.ckpt        Checkpoint for evaluation plots (optional for --eval)
  --timefmt "%Y-%m-%d %H:%M"    Datetime format for plots
  --skip-venv                   Skip venv creation/activation (use current env)

Other:
  -h, --help                    Show this help

Examples:
  ./run.sh --full --device cpu
  ./run.sh --fetch --symbols "BTCUSDT ETHUSDT" --interval 1m --days 3
  ./run.sh --features
  ./run.sh --train --epochs 10 --batch-size 128 --device mps
  ./run.sh --eval --folds 5 --checkpoint artifacts/tft-epoch=08-val_loss=3.0475.ckpt --device cpu
  ./run.sh --plots --timefmt "%Y-%m-%d %H:%M"
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
    -h|--help)   usage; exit 0 ;;

    *) echo "Unknown arg: $1"; usage; exit 1 ;;
  esac
done

# If no stage was specified, default to full
if [[ $DO_FETCH -eq 0 && $DO_FEATURES -eq 0 && $DO_TRAIN -eq 0 && $DO_EVAL -eq 0 && $DO_PLOTS -eq 0 ]]; then
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
  pip install "lightning>=2.2" "pandas>=2.0" "pyarrow" "numpy" "matplotlib"
  # pytorch-forecasting (pick a version compatible with your Python)
  # If you already installed it in your env, the next line can be removed.
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
  # checkpoint is optional; if empty, you'll only get naive baseline
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
  # plot_evaluation can accept labels/timefmt to show datetime on x-axis
  if [[ -f artifacts/time_labels.csv ]]; then
    python scripts/plot_evaluation.py \
      --csv artifacts/evaluation.csv \
      --outdir artifacts \
      --data data/processed/merged.parquet \
      --labels artifacts/time_labels.csv \
      --timefmt "$TIMEFMT"
  else
    # will still plot with time_idx integers if labels not present
    python scripts/plot_evaluation.py \
      --csv artifacts/evaluation.csv \
      --outdir artifacts \
      --data data/processed/merged.parquet \
      --timefmt "$TIMEFMT"
  fi
}

# =========================
# Execute selected stages
# =========================
[[ $DO_FETCH    -eq 1 ]] && run_fetch
[[ $DO_FEATURES -eq 1 ]] && run_features
[[ $DO_TRAIN    -eq 1 ]] && run_train
[[ $DO_EVAL     -eq 1 ]] && run_eval
[[ $DO_PLOTS    -eq 1 ]] && run_plots

echo
echo "Artifacts:"
echo "  - Checkpoints: artifacts/*.ckpt"
echo "  - Metrics:     artifacts/evaluation.csv"
echo "  - Plots:       artifacts/qualitative_forecast.png, artifacts/eval_*.{png,pdf}"
echo "  - Data:        data/raw/, data/processed/merged.parquet"