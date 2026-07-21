@echo off
rem Aegis-LoRA WebUI 的 Windows 一键启动入口。
setlocal
chcp 65001 >nul

rem 始终基于脚本位置定位项目，支持从任意目录启动。
set "PROJECT_ROOT=%~dp0"
set "LAUNCHER=%PROJECT_ROOT%launcher\start.ps1"

rem 在调用 PowerShell 前确认共用启动器存在。
if not exist "%LAUNCHER%" (
    echo       [错误] 启动器缺失: %LAUNCHER%
    pause
    exit /b 1
)

rem 优先使用 PowerShell 7，不可用时回退到 Windows PowerShell 5.1。
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

rem 将 Torch 等参数原样交给共用启动器，并保留其退出码。
"%POWERSHELL_EXE%" -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%LAUNCHER%" -Mode gui %*
set "EXIT_CODE=%ERRORLEVEL%"

rem 仅在失败时暂停窗口，便于查看错误信息。
if not "%EXIT_CODE%"=="0" pause
exit /b %EXIT_CODE%
