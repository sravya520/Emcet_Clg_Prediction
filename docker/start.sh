#!/bin/sh
# Start the API and the UI in one container.
#
# Render's free plan gives a service exactly one public port, so splitting the
# API and the UI across two services would mean two free services, two cold
# starts, and a user waiting for both. One container keeps it to one cold start
# and still deploys the API for real: the UI talks to it over HTTP on
# 127.0.0.1, the same way any other client would.
#
#   $PORT      the public port Render assigns. The UI listens here.
#   $API_PORT  internal only, never exposed.

set -eu

API_PORT="${API_PORT:-8000}"
PUBLIC_PORT="${PORT:-8501}"

echo "starting API on 127.0.0.1:${API_PORT} (internal)"
uvicorn copilot.api:app --host 127.0.0.1 --port "${API_PORT}" --log-level warning &
API_PID=$!

# If the API dies, take the whole container down rather than serving a UI that
# cannot answer anything. A half-working service is harder to diagnose than a
# restart loop.
trap 'kill -TERM "${API_PID}" 2>/dev/null || true' TERM INT

echo "waiting for the API to answer /health"
i=0
while [ "${i}" -lt 60 ]; do
    if curl -fsS "http://127.0.0.1:${API_PORT}/health" >/dev/null 2>&1; then
        echo "API is up after ${i}s"
        break
    fi
    if ! kill -0 "${API_PID}" 2>/dev/null; then
        echo "API process died during startup" >&2
        exit 1
    fi
    i=$((i + 1))
    sleep 1
done

echo "starting UI on 0.0.0.0:${PUBLIC_PORT} (public)"
exec streamlit run src/copilot/ui.py \
    --server.port "${PUBLIC_PORT}" \
    --server.address 0.0.0.0 \
    --server.headless true \
    --server.enableCORS false \
    --server.enableXsrfProtection true \
    --browser.gatherUsageStats false
