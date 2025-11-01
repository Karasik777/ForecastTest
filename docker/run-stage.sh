#!/usr/bin/env bash
set -euo pipefail

cd /workspace

stage="${1:-}"
shift || true

case "$stage" in
  fetch)
    exec ./exec.sh --skip-venv --fetch "$@"
    ;;
  process)
    ./exec.sh --skip-venv --features "$@"
    exec ./exec.sh --skip-venv --train "$@"
    ;;
  predict)
    exec ./exec.sh --skip-venv --consensus-live "$@"
    ;;
  evaluate)
    exec ./exec.sh --skip-venv --eval "$@"
    ;;
  consensus)
    exec ./exec.sh --skip-venv --consensus-eval "$@"
    ;;
  *)
    echo "Unknown stage: ${stage}" >&2
    exit 1
    ;;
esac
