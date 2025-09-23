#!/usr/bin/env bash
set -euo pipefail

# -------- configurable knobs --------
SYMBOLS=("BTCUSDT" "ETHUSDT")
INTERVAL="1m"
DAYS=3
LOOKBACK=168
HORIZON=24
EPOCHS=15
FOLDS=3
BATCH_SIZE=64             # smaller default for mac stability
TRAIN_DEVICE="cpu"        # cpu | mps | gpu | auto  (train_tft.py consumes this)
PYTHON="${PYTHON:-python3}"
# ------------------------------------

# create & activate venv
$PYTHON -m venv .venv
source .venv/bin/activate

# deps: modern Lightning + Forecasting + parquet engine
pip install --upgrade pip
pip uninstall -y pytorch-lightning || true           # remove legacy package to avoid clashes
pip install "lightning>=2.2" "pytorch-forecasting>=1.0.0" pyarrow
# if you already have a requirements.txt for the rest of the stack, install it too:
if [[ -f requirements.txt ]]; then
  pip install -r requirements.txt
fi

# Apple Silicon: if you *do* try MPS later, fallback helps prevent crashes
export PYTORCH_ENABLE_MPS_FALLBACK=1

# run full non-interactive pipeline (exec.py must forward --device into train_tft.py)
python scripts/exec.py --full \
  --symbols "${SYMBOLS[@]}" \
  --interval "$INTERVAL" \
  --days "$DAYS" \
  --lookback "$LOOKBACK" \
  --horizon "$HORIZON" \
  --epochs "$EPOCHS" \
  --folds "$FOLDS" \
  --batch-size "$BATCH_SIZE" \
  --device "$TRAIN_DEVICE"

echo
echo "Artifacts:"
echo "  - Checkpoints: artifacts/*.ckpt"
echo "  - Metrics:     artifacts/evaluation.csv"
echo "  - Plot:        artifacts/qualitative_forecast.png"
echo "  - Data:        data/raw/, data/processed/merged.parquet"