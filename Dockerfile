# One image, two roles. Which role it plays is decided by the command, so the
# API and the UI can never drift out of sync with each other or with the data.
FROM python:3.11-slim

# curl is here for the container healthcheck, nothing else.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first, so editing code does not re-install them every build.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY data/ ./data/
COPY evals/ ./evals/
COPY docker/ ./docker/
RUN chmod +x docker/start.sh

ENV PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Build the processed table at image build time. The app must never depend on
# a build step happening at start-up, and on a free host the first request is
# already slow enough without parsing four PDFs first.
RUN python -m copilot.data.ingest && python -m copilot.data.validate

# Run as a normal user, not root.
RUN useradd --create-home app && chown -R app:app /app
USER app

EXPOSE 8000 8501

# The API is internal, so the healthcheck asks IT, not the UI: if the thing
# that actually holds the data is alive, the service is alive.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -fsS http://127.0.0.1:${API_PORT:-8000}/health || exit 1

# Default: both processes in one container, which is what Render runs on its
# single public port. docker-compose overrides this to run them as two
# services locally, so both shapes stay exercised.
CMD ["./docker/start.sh"]
