#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

COMPOSE=${COMPOSE:-docker compose}

run_stage() {
  local service="$1"; shift || true
  echo ">>> Running stage: ${service}"
  ${COMPOSE} run --rm "${service}" "$@"
}

run_stage stage_fetch "$@"
run_stage stage_process "$@"
run_stage stage_predict "$@"
run_stage stage_evaluate "$@"
run_stage stage_consensus "$@"

echo ">>> Pipeline completed."
