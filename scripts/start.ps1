param([int]$Port = 8000)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
if (Test-Path -LiteralPath '.venv/Scripts/python.exe') {
    & '.venv/Scripts/python.exe' -m uvicorn backend.app.main:app --host 127.0.0.1 --port $Port
} elseif (Test-Path -LiteralPath '.tools/python/python.exe') {
    & '.tools/python/python.exe' -m uvicorn backend.app.main:app --host 127.0.0.1 --port $Port
} else {
    & python -m uvicorn backend.app.main:app --host 127.0.0.1 --port $Port
}
exit $LASTEXITCODE
