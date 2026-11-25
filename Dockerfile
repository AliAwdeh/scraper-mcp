# syntax=docker/dockerfile:1.7
FROM python:3.13-slim-bookworm AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PLAYWRIGHT_BROWSERS_PATH=/ms-playwright \
    MPLCONFIGDIR=/tmp/matplotlib \
    DATA_DIR=/data \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN --mount=from=ghcr.io/astral-sh/uv:0.11.29,source=/uv,target=/bin/uv \
    --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

RUN playwright install --with-deps --only-shell chromium \
 && rm -rf /var/lib/apt/lists/* /tmp/*

FROM base AS test
COPY README.md ./
COPY src ./src
COPY tests ./tests
RUN --mount=from=ghcr.io/astral-sh/uv:0.11.29,source=/uv,target=/bin/uv \
    --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen && python -m pytest -q -p no:cacheprovider

FROM base AS runtime
COPY README.md ./
COPY src ./src
RUN --mount=from=ghcr.io/astral-sh/uv:0.11.29,source=/uv,target=/bin/uv \
    --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev \
 && useradd --create-home --uid 1000 app \
 && mkdir -p /data /tmp/matplotlib \
 && chown -R app:app /data /tmp/matplotlib
USER app

VOLUME ["/data"]
EXPOSE 8000
ENTRYPOINT ["scraper-mcp"]
CMD ["--transport", "stdio"]
