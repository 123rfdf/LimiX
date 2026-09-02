param([string]$CondaEnvironment = "limix")

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$RunScript = Join-Path $PSScriptRoot "run.ps1"
if (-not (Test-Path -LiteralPath (Join-Path $ProjectRoot ".env"))) { throw "Copy .env.example to .env first." }

$BackendLog = Join-Path $ProjectRoot "artifacts\backend-dev.log"
$BackendErrorLog = Join-Path $ProjectRoot "artifacts\backend-dev-error.log"
$Backend = Start-Process -FilePath "powershell.exe" -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $RunScript, "-CondaEnvironment", $CondaEnvironment) -RedirectStandardOutput $BackendLog -RedirectStandardError $BackendErrorLog -WindowStyle Hidden -PassThru
Write-Host "Backend starting at http://127.0.0.1:8000 (log: $BackendLog)"
Write-Host "Frontend starting at http://127.0.0.1:5173"
Push-Location (Join-Path $ProjectRoot "frontend")
try { & npm.cmd run dev }
finally {
    Pop-Location
    if (-not $Backend.HasExited) { Stop-Process -Id $Backend.Id }
}
