@echo off
setlocal
chcp 65001 >nul

if not defined AEGIS_PYTHON (
    echo       [ERROR] CLI environment is not initialized. Run start-cli.bat first.
    exit /b 1
)
if not exist "%AEGIS_PYTHON%" (
    echo       [ERROR] Python interpreter not found: %AEGIS_PYTHON%
    exit /b 1
)

set "PYTHONDONTWRITEBYTECODE=1"
set "PYTHONPATH=%AEGIS_PROJECT_ROOT%;%PYTHONPATH%"

"%AEGIS_PYTHON%" -B -m launcher.cli %*
exit /b %ERRORLEVEL%
