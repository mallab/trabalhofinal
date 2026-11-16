#!/usr/bin/env bash
# Sobe API + PostgreSQL com Docker Compose.
set -euo pipefail
cd "$(dirname "$0")/.."
docker compose up --build -d
echo "Aguardando a API ficar saudável..."
for _ in $(seq 1 60); do
  if curl -fs http://localhost:8000/health > /dev/null; then
    echo "API no ar: http://localhost:8000/docs"; exit 0
  fi
  sleep 2
done
echo "A API não respondeu. Veja os logs: docker compose logs api"; exit 1
