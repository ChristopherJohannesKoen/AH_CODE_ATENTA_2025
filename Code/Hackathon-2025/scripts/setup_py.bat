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

python -m pip install --upgrade pip
python -m pip install fastapi uvicorn python-multipart
python -m pip install openai-whisper torch torchaudio numpy tqdm
python -m pip install ffmpeg-python pyannote.audio deepfilternet

pause
