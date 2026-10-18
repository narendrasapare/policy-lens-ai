$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "../frontend")
& npm.cmd exec vite -- --host 127.0.0.1
exit $LASTEXITCODE
