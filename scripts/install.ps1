# Instala o projeto localmente no Windows (PowerShell), sem Docker.
Set-Location (Join-Path $PSScriptRoot "..")
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
python -m scripts.download_data
if ($LASTEXITCODE -ne 0) { Write-Host "Sem internet? A API vai usar a amostra em data/sample." }
Write-Host "Pronto! Rode .\scripts\run.ps1"
