FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

# Keep the image resolver aligned with the pinned CI toolchain.
RUN python -m pip install --no-cache-dir "uv==0.11.29" \
    && useradd --create-home --uid 10001 --shell /usr/sbin/nologin robata

# Application layers are root-owned and readable; only explicit state mounts are writable.
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY scripts ./scripts
COPY schemas ./schemas
COPY config ./config

RUN uv sync --locked --no-dev --extra mcap --extra r2 --extra pgvector --extra redis

RUN install -d -o robata -g robata -m 0750 /var/lib/robata

USER robata

ENV PATH="/app/.venv/bin:${PATH}"

# This image is a command worker, not an HTTP service. Compose supplies the
# concrete one-shot canonical command; callers can replace the command safely.
ENTRYPOINT ["python"]
CMD ["scripts/run_canonical_fixture.py", "--help"]
