@echo off
rem CLI 会话内的 aegis 命令包装器，将参数转交给 Python/Typer 客户端。
setlocal

rem Python 路径由 start.ps1 注入，避免依赖当前激活的系统环境。
if not defined AEGIS_PYTHON (
    echo       [错误] CLI 环境尚未初始化，请先运行 start-cli.bat。
    exit /b 1
)
if not exist "%AEGIS_PYTHON%" (
    echo       [错误] Python 解释器不存在: %AEGIS_PYTHON%
    exit /b 1
)

rem 禁止字节码写入，并通过 PYTHONPATH 保证从任意目录都能导入项目模块。
set "PYTHONDONTWRITEBYTECODE=1"
set "PYTHONPATH=%AEGIS_PROJECT_ROOT%;%PYTHONPATH%"

rem 原样转发 aegis 后的命令参数，并返回 CLI 的退出码。
"%AEGIS_PYTHON%" -B -m launcher.cli %*
exit /b %ERRORLEVEL%
