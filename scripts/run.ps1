$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$PythonWindowed = Join-Path $ProjectRoot ".venv\Scripts\pythonw.exe"
$Installer = Join-Path $ProjectRoot "install.bat"

function Test-VenvPython {
    if (-not (Test-Path -LiteralPath $Python)) {
        return $false
    }
    try {
        & $Python -c "import sys; raise SystemExit(0 if sys.prefix != sys.base_prefix else 1)"
        return $LASTEXITCODE -eq 0
    } catch {
        return $false
    }
}

if (-not (Test-VenvPython)) {
    $Answer = Read-Host "Lecture Scribe is not installed. Run the installer now? (Y/N)"
    if ($Answer -notmatch "^[Yy]") {
        exit 1
    }
    & $Installer
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}

Set-Location -LiteralPath $ProjectRoot
$AppPath = Join-Path $ProjectRoot "app.py"
Start-Process -FilePath $PythonWindowed -ArgumentList @($AppPath) -WorkingDirectory $ProjectRoot
