#!/usr/bin/env bash
set -euo pipefail

# =========================
# Pretty logging & traps
# =========================
TS() { date +"%Y-%m-%d %H:%M:%S"; }
log()  { printf "\033[1;34m[%s] %s\033[0m\n" "$(TS)" "$*"; }
ok()   { printf "\033[1;32m[%s] %s\033[0m\n" "$(TS)" "$*"; }
warn() { printf "\033[1;33m[%s] %s\033[0m\n" "$(TS)" "$*"; }
err()  { printf "\033[1;31m[%s] %s\033[0m\n" "$(TS)" "$*"; }
on_err(){ err "Failed at line ${BASH_LINENO[0]}: ${BASH_COMMAND}"; exit 1; }
trap on_err ERR

# =========================
# Defaults (override via CLI)
# =========================
SYMBOLS=("BTCUSDT" "ETHUSDT")
INTERVAL="1m"
DAYS=3
LOOKBACK=168              # hours
HORIZON=24                # hours
EPOCHS=15
FOLDS=3
BATCH_SIZE=64
TRAIN_DEVICE="cpu"        # cpu | mps | gpu | auto
CHECKPOINT=""             # path to .ckpt for --eval stage
TIMEFMT="%Y-%m-%d %H:%M"  # matplotlib ticks
SKIP_VENV=0
PYTHON="${PYTHON:-python3}"
SEED=42
DEBUG=0

# Dirs
DATA_DIR="data"
RAW_DIR="${DATA_DIR}/raw"
PROC_DIR="${DATA_DIR}/processed"
ARTIFACTS_DIR="artifacts"

# <<< consensus
CONS_SYMBOL=""
CONS_HORIZON_M=60
CONS_CASH=100
CONS_COMM=2
CONS_EDGE=0.5
TFT_HOOK_CMD=""
CONS_PAD_MIN=10

# Clean toggles / knobs
DO_CLEAN=0
CLEAN_KEEP_METRICS=0
CLEAN_DRY_RUN=0
CLEAN_YES=0

# Stages toggles (default: full if none given)
DO_FETCH=0
DO_FEATURES=0
DO_TRAINTEE=0 # guard
DO_TRAIN=0
DO_EVAL=0
DO_PLOTS=0
DO_CONSENSUS=0
DO_CONS_EVAL=0
DO_CONS_PLOT=0

# Per-stage extra args (quoted strings passed through)
FETCH_ARGS=""
FEATURES_ARGS=""
TRAIN_ARGS=""
EVAL_ARGS=""
PLOTS_ARGS=""
CONS_LIVE_ARGS=""
CONS_EVAL_ARGS=""
CONS_PLOT_ARGS=""

# =========================
# Helpers
# =========================
usage() {
  cat <<'USAGE'
Usage: ./exec.sh [stages] [knobs] [extras]

Stages (pick any; default is --full if none chosen):
  --clean                 Clean data & checkpoints (keep plots); calls scripts/clean_workspace.sh
  --full                  Run fetch -> features -> train -> eval -> plots
  --fetch                 Only fetch raw data (setup_data.py)
  --features              Only build features (make_features.py)
  --train                 Only train TFT (train_tft.py)
  --eval                  Only evaluate (evaluate.py)
  --plots                 Only make evaluation plots (plot_evaluation.py)
  --consensus-live        Run mock consensus decision now (scripts/consensus_live.py)
  --consensus-eval        Evaluate realized PnL for last consensus (scripts/evaluate_pnl.py)
  --consensus-plot        Plot consensus forecast/actuals & PnL (scripts/plot_consensus.py)

Knobs (override defaults):
  --symbols "BTCUSDT ETHUSDT"   Symbols (space-separated, quote the list)
  --interval 1m                 Candle interval
  --days 3                      Days of history to fetch
  --lookback 168                Encoder length (hours)
  --horizon 24                  Prediction length (hours)
  --epochs 15                   Training epochs
  --folds 3                     Rolling folds for evaluation
  --batch-size 64               Batch size for train/eval
  --device cpu|mps|gpu|auto     Device for training/prediction (auto = CUDA→MPS→CPU)
  --checkpoint path.ckpt        Checkpoint for evaluation plots (optional for --eval)
  --timefmt "%Y-%m-%d %H:%M"    Datetime format for plots
  --seed 42                     Global RNG seed for train/eval
  --skip-venv                   Skip venv creation/activation (use current env)
  --data-dir data               Base data dir (contains raw/ and processed/)
  --artifacts-dir artifacts     Output artifacts dir
  --debug                       Enable bash xtrace (set -x)

Clean options (forwarded to scripts/clean_workspace.sh):
  --clean-keep-metrics          Preserve artifacts/*.csv (e.g., evaluation.csv)
  --clean-dry-run               Show what would be deleted without deleting
  --clean-yes                   Skip confirmation prompt

Consensus knobs:
  --cons-symbol BTCUSDT         Override symbol (defaults to first of --symbols)
  --cons-horizon-m 60           Forecast horizon in minutes
  --cons-cash 100               Mock capital in USD
  --cons-comm 2                 Commission per side in USD
  --cons-edge 0.5               Extra edge (%) over breakeven required to trade
  --tft-hook-cmd "python ..."   Optional TFT pipeline command to produce preds
  --cons-pad-min 10             Minutes of padding around forecast window (plot)

Per-stage passthrough (quoted):
  --fetch-args "--api-key XXX --rate-limit 10"
  --features-args "--n-workers 4"
  --train-args "--dropout 0.2"
  --eval-args "--stride 6"
  --plots-args "--style dark"
  --cons-live-args "--verbose"
  --cons-eval-args "--tz UTC"
  --cons-plot-args "--dpi 200"

Other:
  -h, --help                    Show this help

Examples:
  ./exec.sh --clean --clean-yes
  ./exec.sh --full --device auto --seed 123
  ./exec.sh --fetch --symbols "BTCUSDT ETHUSDT" --interval 1m --days 3
  ./exec.sh --train --epochs 10 --batch-size 128 --device mps --train-args "--lr 1e-3"
  ./exec.sh --eval --folds 5 --checkpoint artifacts/tft.ckpt --device cpu
  ./exec.sh --plots --timefmt "%Y-%m-%d %H:%M"
  ./exec.sh --consensus-live --cons-symbol BTCUSDT --cons-horizon-m 60 --cons-edge 0.5
  ./exec.sh --consensus-eval --cons-eval-args "--tz UTC"
  ./exec.sh --consensus-plot --cons-pad-min 12 --cons-plot-args "--dpi 200"
USAGE
}

have() { command -v "$1" >/dev/null 2>&1; }

pycheck() {
  "$PYTHON" - <<'PY' || { err "Python sanity check failed"; exit 1; }
import sys; import platform
print("Python:", sys.version.replace("\n"," "))
print("Platform:", platform.platform())
PY
}

autodev() {
  # Decide device if TRAIN_DEVICE=auto (CUDA → MPS → CPU)
  local dev="$TRAIN_DEVICE"
  if [[ "$TRAIN_DEVICE" == "auto" ]]; then
    dev="$("$PYTHON" - <<'PY'
try:
    import torch
    if torch.cuda.is_available(): print("gpu")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available(): print("mps")
    else: print("cpu")
except Exception: print("cpu")
PY
)"
  fi
  echo "$dev"
}

assert_file() {
  local f="$1" msg="${2:-"Missing required file: $1"}"
  [[ -f "$f" ]] || { err "$msg"; exit 1; }
}

latest_checkpoint() {
  shopt -s nullglob
  local files=("${ARTIFACTS_DIR}"/*.ckpt)
  shopt -u nullglob
  if (( ${#files[@]} == 0 )); then
    echo ""
    return 0
  fi
  local newest="${files[0]}"
  for f in "${files[@]}"; do
    if [[ "$f" -nt "$newest" ]]; then
      newest="$f"
    fi
  done
  echo "$newest"
}

mkdirs() {
  mkdir -p "$RAW_DIR" "$PROC_DIR" "$ARTIFACTS_DIR"
}

# =========================
# Parse CLI
# =========================
while [[ $# -gt 0 ]]; do
  case "$1" in
    # stages
    --clean) DO_CLEAN=1; shift ;;
    --full) DO_FETCH=1; DO_FEATURES=1; DO_TRAIN=1; DO_EVAL=1; DO_PLOTS=1; shift ;;
    --fetch) DO_FETCH=1; shift ;;
    --features) DO_FEATURES=1; shift ;;
    --train) DO_TRAIN=1; shift ;;
    --eval) DO_EVAL=1; shift ;;
    --plots) DO_PLOTS=1; shift ;;
    --consensus-live) DO_CONSENSUS=1; shift ;;
    --consensus-eval) DO_CONS_EVAL=1; shift ;;
    --consensus-plot) DO_CONS_PLOT=1; shift ;;

    # knobs
    --symbols) read -r -a SYMBOLS <<< "$2"; shift 2 ;;
    --interval) INTERVAL="$2"; shift 2 ;;
    --days) DAYS="$2"; shift 2 ;;
    --lookback) LOOKBACK="$2"; shift 2 ;;
    --horizon) HORIZON="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --folds) FOLDS="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --device) TRAIN_DEVICE="$2"; shift 2 ;;
    --checkpoint) CHECKPOINT="$2"; shift 2 ;;
    --timefmt) TIMEFMT="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --skip-venv) SKIP_VENV=1; shift ;;
    --data-dir) DATA_DIR="$2"; RAW_DIR="$2/raw"; PROC_DIR="$2/processed"; shift 2 ;;
    --artifacts-dir) ARTIFACTS_DIR="$2"; shift 2 ;;
    --debug) DEBUG=1; shift ;;

    # clean flags
    --clean-keep-metrics) CLEAN_KEEP_METRICS=1; shift ;;
    --clean-dry-run) CLEAN_DRY_RUN=1; shift ;;
    --clean-yes) CLEAN_YES=1; shift ;;

    # consensus knobs
    --cons-symbol) CONS_SYMBOL="$2"; shift 2 ;;
    --cons-horizon-m) CONS_HORIZON_M="$2"; shift 2 ;;
    --cons-cash) CONS_CASH="$2"; shift 2 ;;
    --cons-comm) CONS_COMM="$2"; shift 2 ;;
    --cons-edge) CONS_EDGE="$2"; shift 2 ;;
    --tft-hook-cmd) TFT_HOOK_CMD="$2"; shift 2 ;;
    --cons-pad-min) CONS_PAD_MIN="$2"; shift 2 ;;

    # passthrough
    --fetch-args) FETCH_ARGS="$2"; shift 2 ;;
    --features-args) FEATURES_ARGS="$2"; shift 2 ;;
    --train-args) TRAIN_ARGS="$2"; shift 2 ;;
    --eval-args) EVAL_ARGS="$2"; shift 2 ;;
    --plots-args) PLOTS_ARGS="$2"; shift 2 ;;
    --cons-live-args) CONS_LIVE_ARGS="$2"; shift 2 ;;
    --cons-eval-args) CONS_EVAL_ARGS="$2"; shift 2 ;;
    --cons-plot-args) CONS_PLOT_ARGS="$2"; shift 2 ;;

    -h|--help) usage; exit 0 ;;
    *) err "Unknown arg: $1"; usage; exit 1 ;;
  esac
done

# If no stage specified → full
if [[ $DO_CLEAN -eq 0 && $DO_FETCH -eq 0 && $DO_FEATURES -eq 0 && $DO_TRAIN -eq 0 && $DO_EVAL -eq 0 && $DO_PLOTS -eq 0 && $DO_CONSENSUS -eq 0 && $DO_CONS_EVAL -eq 0 && $DO_CONS_PLOT -eq 0 ]]; then
  DO_FETCH=1; DO_FEATURES=1; DO_TRAIN=1; DO_EVAL=1; DO_PLOTS=1
fi

[[ $DEBUG -eq 1 ]] && set -x

# =========================
# Environment setup
# =========================
if [[ $SKIP_VENV -eq 0 ]]; then
  # inside managed environments (e.g. Docker) users can set SKIP_VENV=1
  log "Creating/activating venv and installing deps"
  if [[ ! -d .venv ]]; then
    "$PYTHON" -m venv .venv
  fi
  # shellcheck disable=SC1091
  source .venv/bin/activate
  pip install --upgrade pip
  pip uninstall -y pytorch-lightning || true
  pip install "lightning>=2.2" "pandas>=2.0" "pyarrow" "numpy" "matplotlib" "requests"
  pip install "pytorch-forecasting>=1.3.0" || true
  [[ -f requirements.txt ]] && pip install -r requirements.txt || true
  export PYTORCH_ENABLE_MPS_FALLBACK=1
else
  log "SKIP_VENV=1 — using existing interpreter (${PYTHON})"
fi

have "$PYTHON" || { err "Python not found: $PYTHON"; exit 1; }
pycheck

# =========================
# Stage functions
# =========================
run_clean() {
  log "Cleaning workspace (data & checkpoints; keeping plots)"
  local cleaner="scripts/clean_workspace.sh"
  [[ -x "$cleaner" ]] || { err "$cleaner not found or not executable. Make it executable: chmod +x $cleaner"; exit 1; }
  local args=()
  [[ $CLEAN_KEEP_METRICS -eq 1 ]] && args+=("--keep-metrics")
  [[ $CLEAN_DRY_RUN -eq 1 ]]      && args+=("--dry-run")
  [[ $CLEAN_YES -eq 1 ]]          && args+=("--yes")
  "$cleaner" "${args[@]+"${args[@]}"}"
  ok "Clean done"
}

run_fetch() {
  mkdirs
  log "Fetching raw data"
  "$PYTHON" scripts/setup_data.py \
    --symbols "${SYMBOLS[@]}" \
    --interval "$INTERVAL" \
    --days "$DAYS" \
    ${FETCH_ARGS:+$FETCH_ARGS}
  ok "Fetch completed → ${RAW_DIR}"
}

run_features() {
  mkdirs
  log "Building features"
  "$PYTHON" scripts/make_features.py \
    --src "$RAW_DIR" \
    --dest "$PROC_DIR" \
    --interval "$INTERVAL" \
    ${FEATURES_ARGS:+$FEATURES_ARGS}
  assert_file "${PROC_DIR}/merged.parquet" "Expected ${PROC_DIR}/merged.parquet after features."
  ok "Features ready → ${PROC_DIR}/merged.parquet"
}

run_train() {
  mkdirs
  local dev; dev="$(autodev)"
  log "Training TFT (device=${dev}, seed=${SEED})"
  /usr/bin/env SEED="$SEED" "$PYTHON" scripts/train_tft.py \
    --data "${PROC_DIR}/merged.parquet" \
    --lookback "$LOOKBACK" \
    --horizon "$HORIZON" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --device "$dev" \
    ${TRAIN_ARGS:+$TRAIN_ARGS}
  local new_ckpt
  new_ckpt="$(latest_checkpoint)"
  if [[ -n "$new_ckpt" ]]; then
    CHECKPOINT="$new_ckpt"
    ok "Training done. Latest checkpoint → ${CHECKPOINT}"
  else
    warn "Training finished but no checkpoint detected in ${ARTIFACTS_DIR}."
  fi
}

run_eval() {
  mkdirs
  local dev; dev="$(autodev)"
  log "Evaluating (folds=${FOLDS}, device=${dev}, seed=${SEED})"
  if [[ -z "$CHECKPOINT" ]]; then
    CHECKPOINT="$(latest_checkpoint)"
    [[ -z "$CHECKPOINT" ]] && warn "No checkpoint available; TFT metrics will be omitted."
  fi
  local base_args=(
    --data "${PROC_DIR}/merged.parquet"
    --lookback "$LOOKBACK"
    --horizon "$HORIZON"
    --folds "$FOLDS"
    --device "$dev"
    --batch-size "$BATCH_SIZE"
  )
  if [[ -n "$CHECKPOINT" ]]; then
    log "Using checkpoint ${CHECKPOINT}"
    base_args+=(--checkpoint "$CHECKPOINT")
  fi
  /usr/bin/env SEED="$SEED" "$PYTHON" scripts/evaluate.py \
    "${base_args[@]}" \
    ${EVAL_ARGS:+$EVAL_ARGS}
  assert_file "${ARTIFACTS_DIR}/evaluation.csv" "Expected ${ARTIFACTS_DIR}/evaluation.csv after eval."
  ok "Eval done → ${ARTIFACTS_DIR}/evaluation.csv"
}

run_plots() {
  mkdirs
  log "Plotting evaluation"
  local args=(
    --csv "${ARTIFACTS_DIR}/evaluation.csv"
    --outdir "$ARTIFACTS_DIR"
    --data "${PROC_DIR}/merged.parquet"
    --timefmt "$TIMEFMT"
  )
  [[ -f "${ARTIFACTS_DIR}/time_labels.csv" ]] && args+=(--labels "${ARTIFACTS_DIR}/time_labels.csv")
  "$PYTHON" scripts/plot_evaluation.py \
    "${args[@]}" \
    ${PLOTS_ARGS:+$PLOTS_ARGS}
  ok "Plots written → ${ARTIFACTS_DIR}"
}

run_consensus_live() {
  mkdirs
  log "Running consensus_live (single-shot decision)"
  local sym="${CONS_SYMBOL:-${SYMBOLS[0]}}"
  local lookback_m=$(( LOOKBACK * 60 ))
  local args=(
    --symbol "$sym"
    --interval "$INTERVAL"
    --lookback_m "$lookback_m"
    --horizon_m "$CONS_HORIZON_M"
    --start_cash_usd "$CONS_CASH"
    --commission_per_side_usd "$CONS_COMM"
    --min_edge_pct "$CONS_EDGE"
  )
  [[ -n "$TFT_HOOK_CMD" ]] && args+=(--tft_hook_cmd "$TFT_HOOK_CMD")
  "$PYTHON" scripts/consensus_live.py \
    "${args[@]}" \
    ${CONS_LIVE_ARGS:+$CONS_LIVE_ARGS}
  assert_file "${ARTIFACTS_DIR}/mock_trade.json" "Expected ${ARTIFACTS_DIR}/mock_trade.json after consensus-live."
  # forecast path is optional but common:
  [[ -f "${ARTIFACTS_DIR}/forecast_path.csv" ]] || warn "No forecast_path.csv found (this may be fine)."
  ok "Consensus-live done → ${ARTIFACTS_DIR}/mock_trade.json"
}

run_consensus_eval() {
  mkdirs
  log "Evaluating consensus decision"
  assert_file "${ARTIFACTS_DIR}/mock_trade.json" "mock_trade.json missing. Run --consensus-live first."
  # Tip: if your evaluate_pnl.py had tz_localize error, update it to tz_convert(…)
  "$PYTHON" scripts/evaluate_pnl.py \
    --artifact "${ARTIFACTS_DIR}/mock_trade.json" \
    ${CONS_EVAL_ARGS:+$CONS_EVAL_ARGS}
  assert_file "${ARTIFACTS_DIR}/mock_eval.json" "Expected ${ARTIFACTS_DIR}/mock_eval.json after consensus-eval."
  ok "Consensus-eval done → ${ARTIFACTS_DIR}/mock_eval.json"
}

run_consensus_plot() {
  mkdirs
  log "Plotting consensus trade"
  assert_file "${ARTIFACTS_DIR}/mock_trade.json" "mock_trade.json missing. Run --consensus-live."
  assert_file "${ARTIFACTS_DIR}/mock_eval.json"  "mock_eval.json missing. Run --consensus-eval."
  "$PYTHON" scripts/plot_consensus.py \
    --artifact "${ARTIFACTS_DIR}/mock_trade.json" \
    --eval "${ARTIFACTS_DIR}/mock_eval.json" \
    --outdir "$ARTIFACTS_DIR" \
    --pad-min "$CONS_PAD_MIN" \
    ${CONS_PLOT_ARGS:+$CONS_PLOT_ARGS}
  ok "Consensus plots → ${ARTIFACTS_DIR}/consensus_plot.{png,pdf}"
}

print_config() {
  log "===== CONFIG SUMMARY ====="
  echo "Symbols:          ${SYMBOLS[*]}"
  echo "Interval:         ${INTERVAL}"
  echo "Days:             ${DAYS}"
  echo "Lookback (h):     ${LOOKBACK}"
  echo "Horizon (h):      ${HORIZON}"
  echo "Epochs:           ${EPOCHS}"
  echo "Folds:            ${FOLDS}"
  echo "Batch size:       ${BATCH_SIZE}"
  echo "Device:           ${TRAIN_DEVICE}"
  echo "Checkpoint:       ${CHECKPOINT:-<none>}"
  echo "Seed:             ${SEED}"
  echo "Data dir:         ${DATA_DIR}  (raw=${RAW_DIR}, processed=${PROC_DIR})"
  echo "Artifacts dir:    ${ARTIFACTS_DIR}"
  echo "Timefmt:          ${TIMEFMT}"
  echo "Consensus:        sym=${CONS_SYMBOL:-<auto-first>}, horizon_m=${CONS_HORIZON_M}, cash=${CONS_CASH}, comm=${CONS_COMM}, edge%=${CONS_EDGE}"
  echo "Passthrough args: fetch='${FETCH_ARGS}', features='${FEATURES_ARGS}', train='${TRAIN_ARGS}', eval='${EVAL_ARGS}', plots='${PLOTS_ARGS}', cons_live='${CONS_LIVE_ARGS}', cons_eval='${CONS_EVAL_ARGS}', cons_plot='${CONS_PLOT_ARGS}'"
  log "=========================="
}

# =========================
# Execute selected stages
# =========================
print_config
[[ $DO_CLEAN     -eq 1 ]] && run_clean
[[ $DO_FETCH     -eq 1 ]] && run_fetch
[[ $DO_FEATURES  -eq 1 ]] && run_features
[[ $DO_TRAIN     -eq 1 ]] && run_train
[[ $DO_EVAL      -eq 1 ]] && run_eval
[[ $DO_PLOTS     -eq 1 ]] && run_plots
[[ $DO_CONSENSUS -eq 1 ]] && run_consensus_live
[[ $DO_CONS_EVAL -eq 1 ]] && run_consensus_eval
[[ $DO_CONS_PLOT -eq 1 ]] && run_consensus_plot

echo
ok "Artifacts:"
echo "  - Checkpoints: ${ARTIFACTS_DIR}/*.ckpt"
echo "  - Metrics:     ${ARTIFACTS_DIR}/evaluation.csv"
echo "  - Plots:       ${ARTIFACTS_DIR}/qualitative_forecast.png, ${ARTIFACTS_DIR}/eval_*.{png,pdf}"
echo "  - Data:        ${RAW_DIR}/, ${PROC_DIR}/merged.parquet"
if [[ -f "${ARTIFACTS_DIR}/mock_trade.json" && -f "${ARTIFACTS_DIR}/mock_eval.json" && -f "${ARTIFACTS_DIR}/consensus_plot.png" ]]; then
  echo "  - Consensus:   ${ARTIFACTS_DIR}/mock_trade.json, ${ARTIFACTS_DIR}/mock_eval.json, ${ARTIFACTS_DIR}/consensus_plot.{png,pdf}"
else
  echo "  - Consensus:   (run --consensus-live / --consensus-eval / --consensus-plot to generate)"
fi
