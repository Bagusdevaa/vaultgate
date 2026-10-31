# One image for the api, inspector and indexer services (see docker-compose.yml).
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.12.1 /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/models \
    HF_HUB_OFFLINE=1

WORKDIR /app
# Dependencies first, so code changes do not reinstall torch.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project
COPY config.yaml ./
COPY src ./src
COPY scripts ./scripts
COPY inspector ./inspector
RUN uv sync --frozen --no-dev

RUN useradd --uid 10001 --no-create-home app && mkdir -p /app/data/index && chown app /app/data/index
USER app
