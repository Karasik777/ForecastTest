# =========================
# Build args (override at build time)
# =========================
ARG PYTHON_VERSION=3.11
ARG DEBIAN_FRONTEND=noninteractive
ARG TZ=Australia/Brisbane

# ---------- Base image (CPU) ----------
FROM python:${PYTHON_VERSION}-slim AS base

# ---------- System deps ----------
ARG DEBIAN_FRONTEND
ARG TZ
ENV TZ=${TZ} \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MPLBACKEND=Agg

# Avoid running as root in final image
ARG APP_USER=app
ARG APP_UID=10001
ARG APP_GID=10001

# Common packages for data/ML builds + plotting headless
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    ca-certificates \
    tzdata \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip tooling
RUN python -m pip install --upgrade pip wheel setuptools

# Create user and working directory
RUN groupadd -g ${APP_GID} ${APP_USER} \
    && useradd -u ${APP_UID} -g ${APP_GID} -m -s /bin/bash ${APP_USER}
WORKDIR /app

# =========================
# Dependency layer
# Copies only lockfiles first for better caching
# =========================
FROM base AS deps

# Copy whichever dependency files exist (both is fine)
# - requirements.txt (pip)
# - pyproject.toml + (optional) poetry.lock/uv.lock
COPY requirements.txt* /tmp/requirements.txt
COPY pyproject.toml* poetry.lock* uv.lock* /tmp/

# Choose installer automatically:
# 1) If requirements.txt exists -> pip install -r
# 2) Else if pyproject exists -> pip install . (PEP 517)
RUN bash -lc '\
  set -euo pipefail; \
  if [ -s /tmp/requirements.txt ]; then \
      echo "[deps] Installing from requirements.txt"; \
      python -m pip install -r /tmp/requirements.txt; \
  elif [ -s /tmp/pyproject.toml ]; then \
      echo "[deps] Installing from pyproject.toml (PEP 517)"; \
      python -m pip install . --no-build-isolation -C--global-option=; \
      # If your project isn\'t a buildable package, consider adding a \
      # tool like uv/poetry here or generate a requirements.txt first. \
  else \
      echo "[deps] No dependency files found; skipping."; \
  fi'

# =========================
# Runtime image
# =========================
FROM base AS runtime

# Copy installed packages from deps layer
COPY --from=deps /usr/local/lib/python*/site-packages /usr/local/lib/python*/site-packages
COPY --from=deps /usr/local/bin /usr/local/bin

# Copy the rest of the application source
COPY . /app

# Ensure ownership for non-root user
RUN chown -R ${APP_USER}:${APP_USER} /app
USER ${APP_USER}

# Default environment knobs (override at runtime with -e)
ENV TRAIN_DEVICE=cpu \
    MPLCONFIGDIR=/tmp/mpl

# Create writable dirs for artifacts/logs
RUN mkdir -p /app/artifacts /app/logs

# Healthcheck (customise if you expose an API)
# HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD python -c "import sys; sys.exit(0)"

# Entry points:
# If you have a top-level launcher script, use it; otherwise run a sane default.
# You can override CMD in `docker run ...` easily.
# Example tries: scripts/main.sh -> python -m app.main -> python main.py
ENTRYPOINT ["/bin/bash","-lc"]
CMD '\
  if [ -x scripts/main.sh ]; then \
      echo "[run] scripts/main.sh"; exec scripts/main.sh; \
  elif [ -f app/main.py ]; then \
      echo "[run] python app/main.py"; exec python app/main.py; \
  elif [ -f main.py ]; then \
      echo "[run] python main.py"; exec python main.py; \
  else \
      echo "[run] No obvious entrypoint; starting bash."; exec bash; \
  fi'