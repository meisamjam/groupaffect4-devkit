@echo off
REM Master BIDS Processing Pipeline Launcher for Windows
REM Quick start script with preset configurations

setlocal enabledelayedexpansion

echo.
echo ====================================================================
echo                 MASTER BIDS PIPELINE - WINDOWS LAUNCHER
echo ====================================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    exit /b 1
)

REM Default values
set DATA_DIR=..\affectai-data-processing-seed\data
set OUTPUT_DIR=E:\processed_data
set PRESET=standard
set VERBOSE=

REM Parse arguments
:parse_args
if "%1"=="" goto args_end
if "%1"=="--data-dir" (
    set DATA_DIR=%2
    shift
    shift
    goto parse_args
)
if "%1"=="--output-dir" (
    set OUTPUT_DIR=%2
    shift
    shift
    goto parse_args
)
if "%1"=="--preset" (
    set PRESET=%2
    shift
    shift
    goto parse_args
)
if "%1"=="--verbose" (
    set VERBOSE=--verbose
    shift
    goto parse_args
)
if "%1"=="--help" (
    echo Usage: run_pipeline.bat [OPTIONS]
    echo.
    echo Options:
    echo   --data-dir DIR         Root data directory ^(default: ..\affectai-data-processing-seed\data^)
    echo   --output-dir DIR       Output directory ^(default: E:\processed_data^)
    echo   --preset PRESET        Preset config: quick, standard, full, dual_gpu, single_session
    echo   --verbose              Enable verbose logging
    echo   --help                 Show this help message
    echo.
    echo Presets:
    echo   quick       BIDS-only processing ^(no 3D pose/face-hand^)
    echo   standard    Standard with 3D pose ^(recommended^)
    echo   full        Full with 3D pose and face/hand
    echo   dual_gpu    Full processing with dual GPU
    echo   single_session  Single session with full features
    echo.
    exit /b 0
)
shift
goto parse_args

:args_end

REM Verify data directory
if not exist "%DATA_DIR%" (
    echo ERROR: Data directory not found: %DATA_DIR%
    exit /b 1
)

REM Create output directory
if not exist "%OUTPUT_DIR%" (
    echo Creating output directory: %OUTPUT_DIR%
    mkdir "%OUTPUT_DIR%"
)

REM Display configuration
echo Configuration:
echo   Data directory: %DATA_DIR%
echo   Output directory: %OUTPUT_DIR%
echo   Preset: %PRESET%
echo.

REM Run pipeline
python tools\run_pipeline.py ^
    --data-dir "%DATA_DIR%" ^
    --output-dir "%OUTPUT_DIR%" ^
    --preset %PRESET% ^
    %VERBOSE%

if errorlevel 1 (
    echo.
    echo ERROR: Pipeline execution failed
    exit /b 1
)

echo.
echo ====================================================================
echo Pipeline completed successfully!
echo Results saved to: %OUTPUT_DIR%
echo ====================================================================
echo.

endlocal
