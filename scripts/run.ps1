param([string]$CondaEnvironment = "limix")

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot

function Import-DotEnv([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path)) { throw "Missing .env. Copy .env.example to .env first." }
    foreach ($Line in Get-Content -LiteralPath $Path) {
        $Trimmed = $Line.Trim()
        if (-not $Trimmed -or $Trimmed.StartsWith("#") -or -not $Trimmed.Contains("=")) { continue }
        $Name, $Value = $Trimmed.Split("=", 2)
        [Environment]::SetEnvironmentVariable($Name.Trim(), $Value.Trim(), "Process")
    }
}

function Resolve-Python {
    if ($env:LIMIX_PYTHON -and (Test-Path -LiteralPath $env:LIMIX_PYTHON)) { return $env:LIMIX_PYTHON }
    $Candidate = Join-Path $env:USERPROFILE "anaconda3\envs\$CondaEnvironment\python.exe"
    if (Test-Path -LiteralPath $Candidate) { return $Candidate }
    throw "Python for conda environment '$CondaEnvironment' was not found. Set LIMIX_PYTHON in .env."
}

Import-DotEnv (Join-Path $ProjectRoot ".env")
$RequiredFiles = @($env:LIMIX_MODEL_PATH, $env:LIMIX_CLASSIFICATION_CONFIG, $env:LIMIX_REGRESSION_CONFIG)
foreach ($Required in $RequiredFiles) {
    if (-not $Required -or -not (Test-Path -LiteralPath $Required -PathType Leaf)) { throw "Required LimiX file is missing: $Required" }
}
if (-not $env:LIMIX_SOURCE_DIR -or -not (Test-Path -LiteralPath $env:LIMIX_SOURCE_DIR -PathType Container)) {
    throw "LIMIX_SOURCE_DIR is missing or invalid."
}

$Python = Resolve-Python
& $Python -c "import fastapi, pandas, sklearn, torch; print('Python dependencies OK; CUDA:', torch.cuda.is_available())"
if ($LASTEXITCODE -ne 0) { throw "Python dependency check failed. Run scripts/setup.ps1." }

$Frontend = Join-Path $ProjectRoot "frontend"
if (-not (Test-Path -LiteralPath (Join-Path $Frontend "dist\index.html"))) {
    $Npm = (Get-Command npm.cmd -ErrorAction SilentlyContinue).Source
    if (-not $Npm) { throw "npm was not found and the frontend has not been built." }
    Push-Location $Frontend
    try {
        if (-not (Test-Path -LiteralPath "node_modules")) { & $Npm ci }
        & $Npm run build
        if ($LASTEXITCODE -ne 0) { throw "Frontend build failed." }
    } finally { Pop-Location }
}

Write-Host "LimiX Workbench: http://127.0.0.1:8000" -ForegroundColor Green
Push-Location (Join-Path $ProjectRoot "backend")
try { & $Python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 }
finally { Pop-Location }

