@echo off
setlocal
chcp 65001 >nul

set "PROJECT_ROOT=%~dp0"
set "LAUNCHER=%PROJECT_ROOT%launcher\start.ps1"

if not exist "%LAUNCHER%" (
    echo       [ERROR] Shared launcher not found: %LAUNCHER%
    pause
    exit /b 1
)

where pwsh.exe >nul 2>nul
if not errorlevel 1 (
    set "POWERSHELL_EXE=pwsh.exe"
) else (
    where powershell.exe >nul 2>nul
    if errorlevel 1 (
        echo       [ERROR] PowerShell 5.1 or PowerShell 7 is required.
        pause
        exit /b 1
    )
    set "POWERSHELL_EXE=powershell.exe"
)

"%POWERSHELL_EXE%" -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%LAUNCHER%" -Mode gui %*
set "EXIT_CODE=%ERRORLEVEL%"

if not "%EXIT_CODE%"=="0" pause
exit /b %EXIT_CODE%
