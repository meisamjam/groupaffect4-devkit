# Master BIDS Processing Pipeline Launcher for PowerShell
# Quick start script with preset configurations

param(
    [string]$DataDir = "..\affectai-data-processing-seed\data",
    [string]$OutputDir = "E:\processed_data",
    [string]$Preset = "standard",
    [switch]$Verbose,
    [switch]$ListPresets
)

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "====================================================================" -ForegroundColor Cyan
Write-Host "        MASTER BIDS PIPELINE - WINDOWS POWERSHELL LAUNCHER" -ForegroundColor Cyan
Write-Host "====================================================================" -ForegroundColor Cyan
Write-Host ""

$presets = @{
    "quick" = @{
        description = "Quick BIDS-only processing (no 3D pose/face-hand)"
        args = @("--max-workers", "4", "--gpu-devices", "0")
    }
    "standard" = @{
        description = "Standard processing with 3D pose (recommended)"
        args = @("--max-workers", "4", "--gpu-devices", "0", "--enable-3d-pose")
    }
    "full" = @{
        description = "Full processing with 3D pose and face/hand landmarks"
        args = @("--max-workers", "4", "--gpu-devices", "0", "--enable-3d-pose", "--enable-face-hand")
    }
    "dual_gpu" = @{
        description = "Full processing with dual GPU acceleration"
        args = @("--max-workers", "8", "--gpu-devices", "0", "1", "--enable-3d-pose", "--enable-face-hand")
    }
    "single_session" = @{
        description = "Process single session with full features"
        args = @("--max-workers", "1", "--gpu-devices", "0", "--enable-3d-pose", "--enable-face-hand")
    }
}

if ($ListPresets) {
    Write-Host "Available Presets:" -ForegroundColor Yellow
    Write-Host "-" * 70
    foreach ($name in $presets.Keys) {
        $config = $presets[$name]
        Write-Host ""
        Write-Host $name.ToUpper() -ForegroundColor Green
        Write-Host "  $($config.description)"
        Write-Host "  Args: $($config.args -join ' ')"
    }
    Write-Host ""
    exit 0
}

# Verify Python
try {
    $PythonVersion = python.exe --version 2>&1
    Write-Host "Python found: $PythonVersion" -ForegroundColor Green
}
catch {
    Write-Host "ERROR: Python is not installed or not in PATH" -ForegroundColor Red
    exit 1
}

# Verify data directory
if (-not (Test-Path $DataDir)) {
    Write-Host "ERROR: Data directory not found: $DataDir" -ForegroundColor Red
    exit 1
}

# Create output directory
if (-not (Test-Path $OutputDir)) {
    Write-Host "Creating output directory: $OutputDir"
    New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null
}

# Display configuration
Write-Host "Configuration:" -ForegroundColor Yellow
Write-Host "  Data directory: $DataDir"
Write-Host "  Output directory: $OutputDir"
Write-Host "  Preset: $Preset"

if ($Preset -notin $presets.Keys) {
    Write-Host "ERROR: Unknown preset: $Preset" -ForegroundColor Red
    Write-Host "Available presets: $($presets.Keys -join ', ')"
    exit 1
}

Write-Host ""
Write-Host "Description: $($presets[$Preset].description)" -ForegroundColor Cyan
Write-Host ""

# Build command
$cmdArgs = @(
    "tools\master_bids_pipeline.py",
    "--data-dir", $DataDir,
    "--output-dir", $OutputDir
)

# Add preset args
$cmdArgs += $presets[$Preset].args

# Add verbose flag if specified
if ($Verbose) {
    $cmdArgs += "--verbose"
}

# Display command
$cmdString = "python.exe " + ($cmdArgs -join " ")
Write-Host "Running: $cmdString" -ForegroundColor Cyan
Write-Host ""

# Run pipeline
& python.exe @cmdArgs

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "ERROR: Pipeline execution failed with exit code $LASTEXITCODE" -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host ""
Write-Host "====================================================================" -ForegroundColor Green
Write-Host "Pipeline completed successfully!" -ForegroundColor Green
Write-Host "Results saved to: $OutputDir" -ForegroundColor Green
Write-Host "====================================================================" -ForegroundColor Green
Write-Host ""
