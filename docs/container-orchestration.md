## Container & Orchestration Overview

This repository now ships with a light-weight container architecture that
keeps every heavy stage isolated while still writing outputs to your local
workspace. The guiding principles are:

- **Stages in containers:** data fetch, feature engineering, TFT training,
  evaluation, and consensus logic run inside Docker so you can scale or
  parallelise without mutating the host environment.
- **Local storage:** `data/`, `artifacts/`, and `lightning_logs/` are mounted
  as bind volumes, so containers write directly into the project tree. The
  files stay on disk for subsequent stages or local inspection.
- **Local plotting:** any Matplotlib-based plotting (evaluation figures,
  qualitative plots, consensus plots) is expected to run on the host to avoid
  X11/GUI headaches. Use the generated artifacts after containerised stages
  finish.

### Components

| Item | Path | Purpose |
| ---- | ---- | ------- |
| `Dockerfile` | root | Builds a reusable Python image with all project dependencies. |
| `docker/entrypoint.sh` | docker/ | Ensures every container session starts inside `/workspace` and executes the passed command. |
| `docker-compose.yml` | root | Provides stage-specific services (`fetch`, `features`, `train`, `eval`, `full`, `consensus_*`) plus an interactive shell (`pipeline`). |
| `docs/container-orchestration.md` | docs/ | This file; central reference for the workflow. |

### Typical Workflow

1. **Build the base image**

   ```bash
   docker compose build
   ```

2. **Run staged containers (sequential helper)**

   ```bash
   ./docker/pipeline.sh --symbols "BTCUSDT ETHUSDT" --interval 1m
   ```

   The helper chains the following services (all on the `forecast_net` network):

   | Service | Responsibilities |
   | --- | --- |
   | `stage_fetch` | Raw data acquisition |
   | `stage_process` | Feature engineering + TFT training |
   | `stage_predict` | Inference / consensus_live forecast |
   | `stage_evaluate` | Rolling evaluation + metrics |
| `stage_consensus` | Trade evaluation (plot locally afterwards) |

   Each service can also run ad-hoc:

   ```bash
   docker compose run --rm stage_process --env SKIP_VENV=1 --epochs 5
   docker compose run --rm stage_evaluate --env SKIP_VENV=1 --folds 5
   ```

   Override commands or add flags as needed (ideal for Swarm overrides), e.g.
   ```bash
   docker compose run --rm stage_predict --env SKIP_VENV=1 ./exec.sh --skip-venv --consensus-live --cons-symbol BTCUSDT
   ```

3. **Run consensus utilities**

   ```bash
   # Live decision (produces artifacts/mock_trade.json + forecast_path.csv)
   docker compose run --rm consensus_live

   # PnL evaluation (writes artifacts/mock_eval.json — even for HOLD decisions)
 docker compose run --rm consensus_eval
  ```

4. **Generate plots locally**

   ```bash
   ./exec.sh --plots --timefmt "%Y-%m-%d %H:%M"
   ./exec.sh --consensus-plot --cons-pad-min 10
   ```

   All plotting scripts read from the `artifacts/` folder populated by the
   containerised stages.

### Scaling Notes

- **Parallel stages:** `docker compose run --rm --name train_run1 train ...`
  combined with different mounts or output directories allows side-by-side
  experiments without conflicting Python environments.
- **GPU readiness:** switch the base image to `pytorch/pytorch` or add CUDA
access, then set `--device gpu` when invoking `train`/`eval`. Docker Compose
will pick up GPU-capable runtimes automatically (`docker compose run --gpus all train`).
- **Environment overrides:** the container honours every CLI flag in
  `exec.sh`. For repeated configurations, create shell aliases or wrapper
  scripts that call `docker compose run` with the desired flags.
- **Networks:** the default `forecast_net` uses the bridge driver. When
  deploying with Docker Swarm or Kubernetes, swap it for an overlay network to
  enable multi-host coordination.

### Local Plotting Reminder

Because figures rely on Matplotlib backends and often need host display access,
keep `--plots` and `--consensus-plot` outside Docker. The pipeline still writes
raw metrics and checkpoints into `artifacts/`, so the plotting scripts have
everything they need locally.

### Convenience Makefile

The repository includes a `Makefile` wrapper. Append `ARGS='--symbols BTCUSDT'`
or similar to forward extra flags:

```bash
make docker-build
make docker-fetch
make docker-process ARGS='--epochs 5'
make docker-predict
make docker-pipeline ARGS='--symbols "BTCUSDT ETHUSDT" --interval 1m'
make plots ARGS='--timefmt "%Y-%m-%d %H:%M"'
```

The Make targets mirror the Docker services and keep plotting commands on the
host by default.
