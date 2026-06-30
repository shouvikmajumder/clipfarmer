#!/usr/bin/env bash
#
# Starts the clipfarmer.bot worker (datapipeline/main.py, no-arg mode) and the
# web UI (datapipeline/interfaces/web/server.py) concurrently, regardless of the
# caller's current working directory.
#
# Usage: ./start.sh
#
# Ctrl+C (SIGINT) or SIGTERM cleanly kills both child processes.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATAPIPELINE_DIR="${SCRIPT_DIR}/datapipeline"

cleanup() {
    echo ""
    echo "Shutting down worker (PID ${WORKER_PID:-?}) and web UI (PID ${WEB_PID:-?})..."
    [[ -n "${WORKER_PID:-}" ]] && kill "${WORKER_PID}" 2>/dev/null
    [[ -n "${WEB_PID:-}" ]] && kill "${WEB_PID}" 2>/dev/null
    wait "${WORKER_PID:-}" 2>/dev/null
    wait "${WEB_PID:-}" 2>/dev/null
    echo "Stopped."
}

trap cleanup SIGINT SIGTERM

(cd "${DATAPIPELINE_DIR}" && python main.py) &
WORKER_PID=$!

(cd "${DATAPIPELINE_DIR}" && python interfaces/web/server.py) &
WEB_PID=$!

echo "clipfarmer.bot started:"
echo "  worker  -> PID ${WORKER_PID}"
echo "  web UI  -> PID ${WEB_PID} (http://localhost:5050)"
echo "Press Ctrl+C to stop both."

wait "${WORKER_PID}" "${WEB_PID}"
