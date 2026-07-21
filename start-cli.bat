@echo off
setlocal
chcp 65001 >nul

set "PAUSE_ON_EXIT="
if "%~1"=="" set "PAUSE_ON_EXIT=1"

set "PROJECT_ROOT=%~dp0"
set "LAUNCHER=%PROJECT_ROOT%launcher\start.ps1"

if not exist "%LAUNCHER%" (
    echo       [错误] 启动器缺失: %LAUNCHER%
    pause
    exit /b 1
)

where pwsh.exe >nul 2>nul
if not errorlevel 1 (
    set "POWERSHELL_EXE=pwsh.exe"
) else (
    where powershell.exe >nul 2>nul
    if errorlevel 1 (
        echo       [错误] 未找到 PowerShell 5.1 或 PowerShell 7。
        pause
        exit /b 1
    )
    set "POWERSHELL_EXE=powershell.exe"
)

"%POWERSHELL_EXE%" -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%LAUNCHER%" -Mode cli %*
set "EXIT_CODE=%ERRORLEVEL%"

if not "%EXIT_CODE%"=="0" set "PAUSE_ON_EXIT=1"
if defined PAUSE_ON_EXIT pause
exit /b %EXIT_CODE%
