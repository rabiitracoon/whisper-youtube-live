$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$PythonWindowed = Join-Path $ProjectRoot ".venv\Scripts\pythonw.exe"
$Installer = Join-Path $ProjectRoot "install.bat"

if (-not (Test-Path -LiteralPath $PythonWindowed)) {
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
