@echo off
REM Change directory to the location of this .bat file
cd /d "%~dp0"

REM Check if Python is installed
python --version >nul 2>&1
IF ERRORLEVEL 1 (
    echo Python is not installed or not in PATH.
    echo Please install Python from https://www.python.org/downloads/ and try again.
    pause
    exit /b
)

REM Run the Python script inside source
python src\main.py

pause
