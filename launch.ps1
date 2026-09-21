<#
.SYNOPSIS
    Launches the real-time fraud detection pipeline end-to-end on Windows.

.DESCRIPTION
    1. Starts Kafka via docker compose and waits for it to be healthy.
    2. Finds a Python 3.10-3.12 interpreter (PySpark 3.5.0 needs numpy<2.0,
       which has no Python 3.13 wheel), creates/reuses a virtual environment
       with it, and installs requirements.txt.
    3. Sets PYSPARK_PYTHON / PYSPARK_DRIVER_PYTHON to the venv's python.exe
       so the Spark driver and workers agree on which interpreter to use -
       this is required whenever more than one Python is installed, since
       PySpark otherwise falls back to whatever "python" resolves to on
       PATH, which can silently be a different version.
    4. Trains the model (train_model.py) if saved_fraud_rf_model/ doesn't exist yet.
    5. Opens the consumer and producer each in their own PowerShell window,
       with the same PYSPARK_PYTHON / PYSPARK_DRIVER_PYTHON env vars set.

.PARAMETER SkipTraining
    Skip model training even if saved_fraud_rf_model/ is missing.

.PARAMETER Retrain
    Force retraining even if saved_fraud_rf_model/ already exists.

.PARAMETER VenvPath
    Path to the virtual environment to create/use. Defaults to .\.venv

.EXAMPLE
    .\launch.ps1

.EXAMPLE
    .\launch.ps1 -Retrain
#>

[CmdletBinding()]
param(
    [switch]$SkipTraining,
    [switch]$Retrain,
    [string]$VenvPath = ".\.venv"
)

$ErrorActionPreference = "Stop"
# PowerShell 7.3+ turns a non-zero exit code from a native command (like
# `py -3.12 ...` when that version isn't registered) into a terminating
# error whenever $ErrorActionPreference is "Stop", even with stderr
# redirected to $null. Disable that so $LASTEXITCODE checks throughout
# this script behave as written. Harmless no-op on Windows PowerShell 5.1,
# which doesn't have this preference variable.
$PSNativeCommandUseErrorActionPreference = $false

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

function Write-Step {
    param([string]$Message)
    Write-Host ""
    Write-Host ">> $Message" -ForegroundColor Cyan
}

function Test-CommandExists {
    param([string]$Name)
    return [bool](Get-Command $Name -ErrorAction SilentlyContinue)
}

function Get-CompatiblePython {
    <#
    PySpark 3.5.0 needs numpy<2.0 (SPARK-48710), and numpy 1.26.x has no
    Python 3.13 wheel - so this project needs Python 3.10, 3.11, or 3.12
    specifically. Check the default `python` on PATH first (fast path when
    it's already a supported version), then fall back to the Windows `py`
    launcher to find a supported version if `python` itself is 3.13+ or
    missing.
    #>
    if (Test-CommandExists "python") {
        $verString = $null
        try { $verString = & python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>$null } catch {}
        if ($verString -match "^3\.(10|11|12)$") {
            return @{ Cmd = "python"; Arg = $null; Version = $verString }
        }
    }

    if (Test-CommandExists "py") {
        foreach ($ver in @("3.12", "3.11", "3.10")) {
            $probe = $null
            try { $probe = & py "-$ver" -c "print('ok')" 2>$null } catch { $probe = $null }
            if ($LASTEXITCODE -eq 0 -and $probe -eq "ok") {
                return @{ Cmd = "py"; Arg = "-$ver"; Version = $ver }
            }
        }
    }

    if (Test-CommandExists "python") {
        $verString = $null
        try { $verString = & python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>$null } catch {}
        return @{ Cmd = "python"; Arg = $null; Version = $verString; Incompatible = $true }
    }

    return $null
}

# ---------------------------------------------------------------------------
# 0. Pre-flight checks
# ---------------------------------------------------------------------------
Write-Step "Checking prerequisites..."

if (-not (Test-CommandExists "docker")) {
    Write-Error "Docker was not found on PATH. Install Docker Desktop and try again."
    exit 1
}

$pythonInfo = Get-CompatiblePython
if ($null -eq $pythonInfo) {
    Write-Error "Python was not found on PATH. Install Python 3.10-3.12 (see README) and try again."
    exit 1
}
if ($pythonInfo.Incompatible) {
    Write-Error (
        "Found Python $($pythonInfo.Version) on PATH, but this project needs Python " +
        "3.10, 3.11, or 3.12 - PySpark 3.5.0 is incompatible with NumPy 2.0+ " +
        "(required by Python 3.13+). Install Python 3.12 from python.org, then " +
        "either put it first on PATH or install the 'py' launcher so this script " +
        "can find it automatically."
    )
    exit 1
}
Write-Host "Using Python $($pythonInfo.Version) ($($pythonInfo.Cmd) $($pythonInfo.Arg))" -ForegroundColor Green

function Invoke-Python {
    # Runs the selected interpreter with the given args, whether that's
    # `python <args>` or `py -3.12 <args>`.
    param([string[]]$ArgumentList)
    if ($pythonInfo.Arg) {
        & $pythonInfo.Cmd $pythonInfo.Arg @ArgumentList
    } else {
        & $pythonInfo.Cmd @ArgumentList
    }
}

try {
    java -version 2>$null | Out-Null
} catch {
    Write-Warning "Java was not found on PATH. PySpark requires Java 8/11/17 - install it if train_model.py or stream_consumer.py fail."
}

if (-not (Test-Path ".\docker-compose.yml")) {
    Write-Error "docker-compose.yml not found in $ScriptDir. Run this script from the project folder."
    exit 1
}
if (-not (Test-Path ".\creditcard.csv") -and -not $SkipTraining) {
    Write-Warning "creditcard.csv not found in $ScriptDir. train_model.py will fail without it."
}

# ---------------------------------------------------------------------------
# 1. Start Kafka
# ---------------------------------------------------------------------------
Write-Step "Starting Kafka (docker compose up -d)..."
docker compose up -d
if ($LASTEXITCODE -ne 0) {
    Write-Error "docker compose up failed. Is Docker Desktop running?"
    exit 1
}

Write-Step "Waiting for Kafka to report healthy..."
$maxWaitSeconds = 120
$elapsed = 0
$pollInterval = 5
$healthy = $false

while ($elapsed -lt $maxWaitSeconds) {
    $kafkaHealth = docker inspect --format='{{.State.Health.Status}}' kafka 2>$null

    Write-Host "  kafka: $kafkaHealth"

    if ($kafkaHealth -eq "healthy") {
        $healthy = $true
        break
    }

    Start-Sleep -Seconds $pollInterval
    $elapsed += $pollInterval
}

if (-not $healthy) {
    Write-Warning "Kafka did not report healthy within $maxWaitSeconds seconds. Check 'docker compose logs kafka' - continuing anyway."
} else {
    Write-Host "Kafka is healthy." -ForegroundColor Green
}

# ---------------------------------------------------------------------------
# 2. Python virtual environment + dependencies
# ---------------------------------------------------------------------------
Write-Step "Setting up Python virtual environment at $VenvPath..."

if (-not (Test-Path $VenvPath)) {
    Invoke-Python -ArgumentList @("-m", "venv", $VenvPath)
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to create virtual environment."
        exit 1
    }
}

$VenvPython = Join-Path (Resolve-Path $VenvPath) "Scripts\python.exe"

if (-not (Test-Path $VenvPython)) {
    Write-Error "Virtual environment python.exe not found at $VenvPython."
    exit 1
}

# PySpark needs the driver and every worker subprocess to agree on which
# Python interpreter to use. With more than one Python installed, PySpark
# can otherwise silently fall back to whatever "python" resolves to on
# PATH (which may not even be a 3.10-3.12 interpreter), causing version
# mismatch errors between driver and workers. Pointing both at the venv's
# own python.exe removes that ambiguity - it's the same interpreter that
# just had requirements.txt installed into it.
$env:PYSPARK_PYTHON = $VenvPython
$env:PYSPARK_DRIVER_PYTHON = $VenvPython
Write-Host "PYSPARK_PYTHON / PYSPARK_DRIVER_PYTHON set to: $VenvPython" -ForegroundColor Green

Write-Step "Installing requirements.txt (this can take a few minutes for pyspark)..."
& $VenvPython -m pip install --upgrade pip --quiet
& $VenvPython -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) {
    Write-Error "pip install -r requirements.txt failed."
    exit 1
}

# ---------------------------------------------------------------------------
# 3. Train the model if needed
# ---------------------------------------------------------------------------
$ModelPath = ".\saved_fraud_rf_model"

if ($Retrain -and (Test-Path $ModelPath)) {
    Write-Step "Removing existing model at $ModelPath (Retrain requested)..."
    Remove-Item -Recurse -Force $ModelPath
}

if ($SkipTraining) {
    Write-Step "Skipping training (SkipTraining flag set)."
} elseif (Test-Path $ModelPath) {
    Write-Step "Model already exists at $ModelPath - skipping training. Use -Retrain to force."
} else {
    Write-Step "Training model (python train_model.py)..."
    & $VenvPython train_model.py
    if ($LASTEXITCODE -ne 0) {
        Write-Error "train_model.py failed. See output above."
        exit 1
    }
}

# ---------------------------------------------------------------------------
# 4. Launch consumer and producer in their own windows
# ---------------------------------------------------------------------------
# Each spawned window inherits this process's environment, including the
# PYSPARK_PYTHON / PYSPARK_DRIVER_PYTHON vars set above.
Write-Step "Launching stream_consumer.py in a new window..."
Start-Process powershell -ArgumentList @(
    "-NoExit",
    "-Command",
    "Set-Location '$ScriptDir'; " +
    "`$env:PYSPARK_PYTHON = '$VenvPython'; " +
    "`$env:PYSPARK_DRIVER_PYTHON = '$VenvPython'; " +
    "& '$VenvPython' stream_consumer.py"
)

Write-Host "Waiting a few seconds for the Spark streaming query to initialize..."
Start-Sleep -Seconds 8

Write-Step "Launching kafka_producer.py in a new window..."
Start-Process powershell -ArgumentList @(
    "-NoExit",
    "-Command",
    "Set-Location '$ScriptDir'; & '$VenvPython' kafka_producer.py"
)

Write-Step "Pipeline launched."
Write-Host "  - Kafka           : running in Docker (docker compose ps to check)" -ForegroundColor Green
Write-Host "  - Consumer        : running in its own window" -ForegroundColor Green
Write-Host "  - Producer        : running in its own window" -ForegroundColor Green
Write-Host ""
Write-Host "To stop everything: close both windows (Ctrl+C in each), then run 'docker compose down'." -ForegroundColor Yellow