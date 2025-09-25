#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------
# Clean the workspace while preserving plots.
# ------------------------------------------------------------

KEEP_METRICS=0
DRY_RUN=0
ASSUME_YES=0

usage() {
  cat <<'USAGE'
Usage:
  scripts/clean_workspace.sh [--keep-metrics] [--dry-run] [-y|--yes]

What it does by default:
  - Deletes:   data/raw/, data/processed/
  - Deletes:   artifacts/*.ckpt (model checkpoints)
  - Deletes:   artifacts/*.csv  (metrics) unless --keep-metrics is set
  - Keeps:     artifacts/*.png, artifacts/*.pdf (plots)
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --keep-metrics) KEEP_METRICS=1; shift ;;
    --dry-run)      DRY_RUN=1; shift ;;
    -y|--yes)       ASSUME_YES=1; shift ;;
    -h|--help)      usage; exit 0 ;;
    *) echo "Unknown argument: $1"; usage; exit 1 ;;
  esac
done

ROOT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
DATA_RAW="$ROOT_DIR/data/raw"
DATA_PROCESSED="$ROOT_DIR/data/processed"
ARTIFACTS="$ROOT_DIR/artifacts"

echo "Workspace root: $ROOT_DIR"
echo "Will remove:"
echo "  - $DATA_RAW"
echo "  - $DATA_PROCESSED"
echo "  - $ARTIFACTS/*.ckpt"
if [[ $KEEP_METRICS -eq 0 ]]; then
  echo "  - $ARTIFACTS/*.csv (metrics)"
else
  echo "  - Preserving metrics (*.csv)"
fi
echo "Will keep:"
echo "  - $ARTIFACTS/*.png, $ARTIFACTS/*.pdf (plots)"

if [[ $ASSUME_YES -eq 0 ]]; then
  read -rp "Proceed? [y/N]: " ans
  case "$ans" in
    y|Y|yes|YES) ;;
    *) echo "Aborted."; exit 0 ;;
  esac
fi

rm_target() {
  local path="$1"
  if [[ $DRY_RUN -eq 1 ]]; then
    echo "[dry-run] rm -rf -- $path"
  else
    rm -rf -- "$path"
  fi
}

rm_glob() {
  local pattern=$1
  shopt -s nullglob
  local matches=($pattern)   # expands pattern into actual files
  shopt -u nullglob
  if (( ${#matches[@]} )); then
    for f in "${matches[@]}"; do
      if [[ $DRY_RUN -eq 1 ]]; then
        echo "[dry-run] rm -f -- $f"
      else
        rm -f -- "$f"
      fi
    done
  fi
}

# 1) Remove data directories
[[ -d "$DATA_RAW" ]] && rm_target "$DATA_RAW" || echo "Skip: $DATA_RAW (not found)"
[[ -d "$DATA_PROCESSED" ]] && rm_target "$DATA_PROCESSED" || echo "Skip: $DATA_PROCESSED (not found)"

# 2) Remove checkpoints (unquoted glob fix)
[[ -d "$ARTIFACTS" ]] && rm_glob $ARTIFACTS/*.ckpt || echo "Skip: $ARTIFACTS (not found)"

# 3) Remove metrics (CSV) unless preserving (unquoted glob fix)
if [[ -d "$ARTIFACTS" && $KEEP_METRICS -eq 0 ]]; then
  rm_glob $ARTIFACTS/*.csv
fi

echo
echo "Done."
if [[ $DRY_RUN -eq 1 ]]; then
  echo "(dry-run) Nothing was actually deleted."
else
  echo "Kept plots in $ARTIFACTS/*.png and $ARTIFACTS/*.pdf"
fi