#!/usr/bin/env bash
set -euo pipefail

cd /workspace

if [[ $# -eq 0 ]]; then
  exec bash
fi

if [[ "$1" == -* ]]; then
  set -- ./exec.sh "$@"
fi

exec "$@"
