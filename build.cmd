@echo off
rem Baut Silberkorn als Programmordner und ZIP:
rem   dist\Silberkorn\                      (Silberkorn.exe starten)
rem   dist\Silberkorn-<Version>-win64.zip   (zum Verteilen, mit .sha256)
rem
rem Gebaut wird in der sauberen Umgebung .venv-build - ohne NVIDIA-Pakete,
rem TensorRT und pillow-heif (siehe requirements-build.txt).
rem
rem Licensed under MIT License
rem Copyright 2026 Alexander Unverhau
rem Created with assistance of Claude AI

setlocal
cd /d "%~dp0"

if not exist .venv-build\Scripts\python.exe (
    echo Lege .venv-build an ...
    python -m venv .venv-build || exit /b 1
)
.venv-build\Scripts\python.exe -m pip install --disable-pip-version-check -r requirements-build.txt || exit /b 1
.venv-build\Scripts\python.exe werkzeuge\exe_bauen.py || exit /b 1
