#!/usr/bin/env bash
# Sobe a API localmente em http://localhost:8000/docs (SQLite).
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
