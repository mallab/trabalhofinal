# Roda os testes no Windows.
Set-Location (Join-Path $PSScriptRoot "..")
if (Test-Path .venv) { .\.venv\Scripts\Activate.ps1 }
python -m pytest -v @args
