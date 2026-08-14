param(
    [switch]$SkipDesktopShortcut
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$VenvRoot = Join-Path $ProjectRoot ".venv"
$VenvPython = Join-Path $VenvRoot "Scripts\python.exe"
$ToolsRoot = Join-Path $ProjectRoot "tools\codex-cli"

function Write-Step([string]$Message) {
    Write-Host ""
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Resolve-Python {
    $KnownPython = Join-Path $env:LOCALAPPDATA "Programs\Python\Python312\python.exe"
    if (Test-Path -LiteralPath $KnownPython) {
        return $KnownPython
    }
    try {
        $Resolved = & py.exe -3.12 -c "import sys; print(sys.executable)" 2>$null
        if ($LASTEXITCODE -eq 0 -and $Resolved) {
            return $Resolved.Trim()
        }
    } catch {}
    try {
        $Resolved = & py.exe -3 -c "import sys; print(sys.executable)" 2>$null
        if ($LASTEXITCODE -eq 0 -and $Resolved) {
            return $Resolved.Trim()
        }
    } catch {}
    try {
        $Resolved = & python.exe -c "import sys; print(sys.executable)" 2>$null
        if ($LASTEXITCODE -eq 0 -and $Resolved) {
            return $Resolved.Trim()
        }
    } catch {}
    return $null
}

Write-Host "Lecture Scribe installer for Windows NVIDIA" -ForegroundColor White
Write-Host "This installs Python packages, CUDA PyTorch, and Codex CLI locally." -ForegroundColor DarkGray

$PythonExe = Resolve-Python
if (-not $PythonExe) {
    Write-Step "Python 3.11+ installation"
    $Winget = Get-Command winget.exe -ErrorAction SilentlyContinue
    if (-not $Winget) {
        throw "Python 3.11 or newer is not installed and winget is unavailable. Install Python 3.12 from python.org, then run install.bat again."
    }
    & winget.exe install --exact --id Python.Python.3.12 --scope user --accept-package-agreements --accept-source-agreements --silent
    if ($LASTEXITCODE -ne 0) {
        throw "Python installation failed with exit code $LASTEXITCODE."
    }
    $PythonExe = Resolve-Python
    if (-not $PythonExe) {
        throw "Python was installed but could not be found. Sign out and back in, then run install.bat again."
    }
}
Write-Host "Python: $PythonExe" -ForegroundColor Green

Write-Step "Creating the private Python environment"
$VenvHealthy = $false
if (Test-Path -LiteralPath $VenvPython) {
    try {
        & $VenvPython -c "import sys; raise SystemExit(0 if sys.prefix != sys.base_prefix else 1)"
        $VenvHealthy = $LASTEXITCODE -eq 0
    } catch {
        $VenvHealthy = $false
    }
}
if (-not $VenvHealthy) {
    if (Test-Path -LiteralPath $VenvRoot) {
        Write-Host "The existing .venv is broken or belongs to another Python installation. Recreating it." -ForegroundColor Yellow
        Remove-Item -LiteralPath $VenvRoot -Recurse -Force
    }
    & $PythonExe -m venv $VenvRoot
    if ($LASTEXITCODE -ne 0) { throw "Failed to create .venv." }
}
& $VenvPython -m pip install --upgrade pip setuptools wheel
if ($LASTEXITCODE -ne 0) { throw "Failed to update pip." }

Write-Step "Installing NVIDIA CUDA 12.8 runtime"
& $VenvPython -m pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu128
if ($LASTEXITCODE -ne 0) { throw "GPU PyTorch installation failed." }

Write-Step "Installing Lecture Scribe dependencies"
& $VenvPython -m pip install --upgrade -r (Join-Path $ProjectRoot "requirements.txt")
if ($LASTEXITCODE -ne 0) { throw "Python dependency installation failed." }

Write-Step "Installing the latest official Codex CLI locally"
$Npm = Get-Command npm.cmd -ErrorAction SilentlyContinue
if (-not $Npm) {
    $Npm = Get-Command npm.exe -ErrorAction SilentlyContinue
}
if (-not $Npm) {
    Write-Host "Node.js/npm was not found, so Codex CLI installation was skipped." -ForegroundColor Yellow
    Write-Host "Transcription still works. Install Node.js LTS and rerun install.bat for AI notes or automatic terminology." -ForegroundColor Yellow
} else {
    New-Item -ItemType Directory -Force -Path $ToolsRoot | Out-Null
    & $Npm.Source install --prefix $ToolsRoot --no-audit --no-fund "@openai/codex@latest"
    if ($LASTEXITCODE -ne 0) { throw "Codex CLI installation failed." }
}

Write-Step "Verifying GPU and app dependencies"
& $VenvPython (Join-Path $ProjectRoot "scripts\verify_install.py")
if ($LASTEXITCODE -ne 0) { throw "Installation verification failed." }

if (-not $SkipDesktopShortcut) {
    Write-Step "Creating a desktop shortcut"
    $Desktop = [Environment]::GetFolderPath("Desktop")
    $ShortcutPath = Join-Path $Desktop "Lecture Scribe.lnk"
    $Shell = New-Object -ComObject WScript.Shell
    $Shortcut = $Shell.CreateShortcut($ShortcutPath)
    $Shortcut.TargetPath = Join-Path $ProjectRoot "run.bat"
    $Shortcut.WorkingDirectory = $ProjectRoot
    $Shortcut.Description = "YouTube lecture GPU transcription and Markdown notes"
    $Shortcut.Save()
    Write-Host "Shortcut: $ShortcutPath" -ForegroundColor Green
}

Write-Host ""
Write-Host "Ready. Start with run.bat, then sign in from Tools > ChatGPT OAuth." -ForegroundColor Green
Write-Host "The Whisper large-v3 model downloads on the first transcription." -ForegroundColor Yellow
