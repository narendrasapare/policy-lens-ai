$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "../backend")
& ./.venv/Scripts/python.exe -m uvicorn policylens.main:app --host 127.0.0.1 --port 8000 --workers 1
exit $LASTEXITCODE
