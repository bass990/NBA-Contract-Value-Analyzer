# Windows setup script — run from PowerShell in the project root
# Usage:  .\setup.ps1
#
# What this does:
# 1. Creates a Python virtual environment in .venv\
# 2. Activates it
# 3. Installs all dependencies from requirements.txt

$ErrorActionPreference = "Stop"

Write-Host "NBA Contract Value — Windows setup" -ForegroundColor Cyan
Write-Host "────────────────────────────────────`n"

# Check Python is available
try {
    $pyVersion = python --version 2>&1
    Write-Host "Found Python: $pyVersion"
} catch {
    Write-Host "Python not found in PATH. Install Python 3.11+ from python.org first." -ForegroundColor Red
    exit 1
}

# Create venv if missing
if (-not (Test-Path ".venv")) {
    Write-Host "`nCreating virtual environment in .venv\..." -ForegroundColor Yellow
    python -m venv .venv
    Write-Host "Virtual environment created."
} else {
    Write-Host "Virtual environment .venv already exists."
}

# Activate
Write-Host "`nActivating virtual environment..." -ForegroundColor Yellow
& .\.venv\Scripts\Activate.ps1

# Upgrade pip
Write-Host "Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip --quiet

# Install requirements
Write-Host "Installing dependencies (this takes ~2 min)..." -ForegroundColor Yellow
pip install -r requirements.txt
pip install --quiet jupyter nbformat

Write-Host "`nSetup complete.`n" -ForegroundColor Green
Write-Host "Next steps:`n"
Write-Host "  1. Activate the venv in any new PowerShell window:"
Write-Host "     .\.venv\Scripts\Activate.ps1`n"
Write-Host "  2. Run the notebook:"
Write-Host "     jupyter notebook notebooks\NBA_Contract_Value_END_TO_END.ipynb`n"
Write-Host "  3. After the notebook runs, launch the dashboard:"
Write-Host "     streamlit run src\app\dashboard.py`n"
