#!/usr/bin/env bash
set -euo pipefail



# =========================
# Multivariate TFT Training
# =========================
# This script runs only the multivariate TFT training pipeline.
# Edit defaults here or override via CLI.
# Example:
#   ./run_tft_multi.sh --epochs 30 --horizon 48
# ./scripts/run_tft_multi.sh
# ./scripts/run_tft_multi.sh --horizon 48 --epochs 30

# -------------------------
# Defaults
# -------------------------
DATA="data/market_1m.csv"         # path to prepared dataset
TIME_COL="timestamp"
GROUP_COL="SYMBOL"
TARGET_COL="BTC_close"

LOOKBACK=168                      # encoder length
HORIZON=24                        # decoder length
BATCH_SIZE=64
EPOCHS=15
LR=1e-3
HIDDEN_SIZE=64
LSTM_LAYERS=2
ATTN_HEADS=4
DROPOUT=0.1

OUTPUT_DIR="artifacts"
SAVE_TOP_K=3

# Example covariates
KNOWN_REALS="time_idx hour_of_day day_of_week"
UNKNOWN_REALS="BTC_close BTC_volume ETH_close ETH_volume"

# -------------------------
# Parse overrides
# -------------------------
ARGS=("$@")

# -------------------------
# Run training
# -------------------------
echo ">>> Running multivariate TFT training..."
python3 train_tft_multi.py \
  --data "$DATA" \
  --time-col "$TIME_COL" \
  --group-id-col "$GROUP_COL" \
  --target-col "$TARGET_COL" \
  --lookback "$LOOKBACK" \
  --horizon "$HORIZON" \
  --batch-size "$BATCH_SIZE" \
  --epochs "$EPOCHS" \
  --lr "$LR" \
  --hidden-size "$HIDDEN_SIZE" \
  --lstm-layers "$LSTM_LAYERS" \
  --attention-heads "$ATTN_HEADS" \
  --dropout "$DROPOUT" \
  --output-dir "$OUTPUT_DIR" \
  --save-top-k "$SAVE_TOP_K" \
  --known-reals $KNOWN_REALS \
  --unknown-reals $UNKNOWN_REALS \
  --add-time-features \
  "${ARGS[@]}"

echo ">>> Training complete. Best checkpoint path saved in $OUTPUT_DIR/tft_multi_best_ckpt.txt"