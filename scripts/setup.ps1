param([string]$CondaEnvironment = "limix")

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot

function Resolve-Conda {
    $Command = Get-Command conda -ErrorAction SilentlyContinue
    if ($Command) { return $Command.Source }
    $Candidate = Join-Path $env:USERPROFILE "anaconda3\Scripts\conda.exe"
    if (Test-Path -LiteralPath $Candidate) { return $Candidate }
    throw "Conda was not found. Install Anaconda/Miniconda or add conda to PATH."
}

$Conda = Resolve-Conda
$Npm = (Get-Command npm.cmd -ErrorAction SilentlyContinue).Source
if (-not $Npm) { throw "Node.js/npm was not found. Install Node.js 22 or newer." }

Write-Host "Installing Python development dependencies into '$CondaEnvironment'..."
& $Conda run -n $CondaEnvironment python -m pip install -r (Join-Path $ProjectRoot "requirements-dev.txt")
if ($LASTEXITCODE -ne 0) { throw "Python dependency installation failed." }

Write-Host "Installing locked frontend dependencies..."
Push-Location (Join-Path $ProjectRoot "frontend")
try {
    & $Npm ci
    if ($LASTEXITCODE -ne 0) { throw "Frontend dependency installation failed." }
} finally { Pop-Location }

if (-not (Test-Path -LiteralPath (Join-Path $ProjectRoot ".env"))) {
    Copy-Item -LiteralPath (Join-Path $ProjectRoot ".env.example") -Destination (Join-Path $ProjectRoot ".env")
    Write-Host "Created .env. Edit the LimiX paths before running the application." -ForegroundColor Yellow
}
Write-Host "Setup complete." -ForegroundColor Green

