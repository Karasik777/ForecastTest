FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /workspace

RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY docker/entrypoint.sh /workspace/docker/entrypoint.sh
COPY docker/run-stage.sh /workspace/docker/run-stage.sh
RUN chmod +x /workspace/docker/entrypoint.sh /workspace/docker/run-stage.sh

ENV PYTHONPATH=/workspace

ENTRYPOINT ["/workspace/docker/entrypoint.sh"]
CMD ["bash"]
