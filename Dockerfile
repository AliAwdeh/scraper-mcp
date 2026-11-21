# syntax=docker/dockerfile:1.7
# ---------------------------------------------------------------------------- base
# Python + locked dependencies + headless Chromium. Rebuilt only when the lockfile changes.
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
# uv is bind-mounted only for the build step and its download cache lives in a BuildKit
# cache mount, so neither ends up in the image.
RUN --mount=from=ghcr.io/astral-sh/uv:0.11.29,source=/uv,target=/bin/uv \
    --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

# Headless Chromium and the system libraries it needs (for JavaScript-rendered pages).
RUN playwright install --with-deps --only-shell chromium \
 && rm -rf /var/lib/apt/lists/* /tmp/*

# ---------------------------------------------------------------------------- test
# docker build --target test .   -> runs the whole suite, including the real-browser test.
FROM base AS test
COPY README.md ./
COPY src ./src
COPY tests ./tests
RUN --mount=from=ghcr.io/astral-sh/uv:0.11.29,source=/uv,target=/bin/uv \
    --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen && python -m pytest -q -p no:cacheprovider

# ---------------------------------------------------------------------------- runtime
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
# stdio by default (Claude Desktop / Claude Code launch the container per session);
# docker-compose.yml overrides this with --transport http.
CMD ["--transport", "stdio"]
