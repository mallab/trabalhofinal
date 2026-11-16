# Sobe a API localmente no Windows em http://localhost:8000/docs (SQLite).
Set-Location (Join-Path $PSScriptRoot "..")
.\.venv\Scripts\Activate.ps1
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
