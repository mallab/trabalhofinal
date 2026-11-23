#!/usr/bin/env bash
# Instala o projeto localmente (sem Docker): cria venv, instala dependências
# e baixa o MovieLens.
set -euo pipefail
cd "$(dirname "$0")/.."

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements-dev.txt
python -m scripts.download_data || echo "Sem internet? A API vai usar a amostra em data/sample."
echo "Pronto! Rode ./scripts/run.sh"
