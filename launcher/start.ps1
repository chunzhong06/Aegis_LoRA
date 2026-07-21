#Requires -Version 5.1

[CmdletBinding(PositionalBinding = $false)]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("gui", "cli")]
    [string]$Mode,

    [ValidateSet("auto", "cu130", "cpu")]
    [string]$Torch = "auto",

    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ApplicationArgs
)

$ErrorActionPreference = "Stop"
[Console]::InputEncoding = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$OutputEncoding = [Console]::OutputEncoding

# =====================================================================
# 启动显示
# =====================================================================
function Show-Banner {
    Write-Host ""
    Write-Host "    _    _____ ____ ___ ____    _     ___  ____      _    " -ForegroundColor Cyan
    Write-Host "   / \  | ____/ ___|_ _/ ___|  | |   / _ \|  _ \    / \   " -ForegroundColor Cyan
    Write-Host "  / _ \ |  _|| |  _ | |\___ \  | |  | | | | |_) |  / _ \  " -ForegroundColor Cyan
    Write-Host " / ___ \| |__| |_| || | ___) | | |__| |_| |  _ <  / ___ \ " -ForegroundColor Cyan
    Write-Host "/_/   \_\_____\____|___|____/  |_____\___/|_| \_\/_/   \_\" -ForegroundColor Cyan
    Write-Host ""
}

function Write-Stage([string]$Name, [string]$Message) {
    Write-Host ""
    Write-Host ">>> [$Name] $Message" -ForegroundColor Cyan
}

function Stop-Launcher([string]$Message, [int]$Code = 1) {
    Write-Host "      [错误] $Message" -ForegroundColor Red
    exit $Code
}

# =====================================================================
# 工具与环境
# =====================================================================
function Find-Uv {
    $command = Get-Command uv.exe -ErrorAction SilentlyContinue
    if ($command) {
        return $command.Source
    }

    $candidate = Join-Path $env:USERPROFILE ".local\bin\uv.exe"
    if (Test-Path -LiteralPath $candidate -PathType Leaf) {
        return $candidate
    }

    return $null
}

function Install-Uv {
    Write-Host "      [-] 未检测到 uv，正在安装..."
    try {
        [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
        $installScript = Invoke-RestMethod "https://astral.sh/uv/install.ps1"
        Invoke-Expression $installScript
    }
    catch {
        Stop-Launcher "uv 安装失败: $($_.Exception.Message)"
    }

    $uvPath = Find-Uv
    if (-not $uvPath) {
        Stop-Launcher "uv 已执行安装，但当前用户目录中仍未找到 uv.exe。"
    }
    return $uvPath
}

function Resolve-TorchProfile([string]$Requested) {
    if ($Requested -ne "auto") {
        return $Requested
    }

    $nvidiaSmi = Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue
    if (-not $nvidiaSmi) {
        Write-Host "      [警告] 未检测到 NVIDIA 驱动，使用 CPU 版 PyTorch。" -ForegroundColor Yellow
        return "cpu"
    }

    try {
        $gpuName = (& $nvidiaSmi.Source --query-gpu=name --format=csv,noheader 2>$null | Select-Object -First 1).Trim()
        if ($LASTEXITCODE -eq 0 -and $gpuName) {
            Write-Host "      [-] NVIDIA GPU: $gpuName"
            return "cu130"
        }
    }
    catch {
        Write-Host "      [警告] NVIDIA 驱动探测失败，使用 CPU 版 PyTorch。" -ForegroundColor Yellow
    }

    return "cpu"
}

function Invoke-Uv([string[]]$Arguments) {
    & $script:UvPath @Arguments
    if ($LASTEXITCODE -ne 0) {
        Stop-Launcher "uv 命令执行失败，退出码: $LASTEXITCODE。" $LASTEXITCODE
    }
}

# =====================================================================
# 启动流程
# =====================================================================
Show-Banner

$LauncherRoot = $PSScriptRoot
$ProjectRoot = Split-Path -Parent $LauncherRoot
$ProjectFile = Join-Path $LauncherRoot "pyproject.toml"
$LockFile = Join-Path $LauncherRoot "uv.lock"
$EntryName = if ($Mode -eq "gui") { "webui.py" } else { "cli.py" }
$EntryFile = Join-Path $LauncherRoot $EntryName
$EntryModule = if ($Mode -eq "gui") { "launcher.webui" } else { "launcher.cli" }

if (-not (Test-Path -LiteralPath $EntryFile -PathType Leaf)) {
    Stop-Launcher "无法定位项目入口: $EntryFile"
}
if (-not (Test-Path -LiteralPath $ProjectFile -PathType Leaf)) {
    Stop-Launcher "环境配置缺失: $ProjectFile"
}
if (-not (Test-Path -LiteralPath $LockFile -PathType Leaf)) {
    Stop-Launcher "依赖锁文件缺失: $LockFile"
}

Set-Location -LiteralPath $ProjectRoot
$env:UV_PROJECT_ENVIRONMENT = Join-Path $ProjectRoot ".venv"
$env:UV_PYTHON_INSTALL_DIR = Join-Path $ProjectRoot ".cache\python"
$env:UV_NO_CACHE = "1"
$env:PYTHONDONTWRITEBYTECODE = "1"

# Conda 可能留下只有占位文件的证书目录；仅在确认没有证书文件时忽略它。
if ($env:SSL_CERT_DIR -and (Test-Path -LiteralPath $env:SSL_CERT_DIR -PathType Container)) {
    $certificate = Get-ChildItem -LiteralPath $env:SSL_CERT_DIR -File -ErrorAction SilentlyContinue |
        Where-Object { $_.Extension -in ".pem", ".crt", ".cer" } |
        Select-Object -First 1
    if (-not $certificate) {
        Remove-Item Env:SSL_CERT_DIR
    }
}

Write-Stage "检查工具" "正在检查启动工具与运行设备..."
$script:UvPath = Find-Uv
if (-not $script:UvPath) {
    $script:UvPath = Install-Uv
}
$uvVersion = & $script:UvPath --version
Write-Host "      [-] $uvVersion"
Write-Host "      [-] 项目目录: $ProjectRoot"

$TorchProfile = $null
if ($Mode -eq "gui") {
    $TorchProfile = Resolve-TorchProfile $Torch
    Write-Host "      [-] PyTorch 配置: $TorchProfile"
}
else {
    Write-Host "      [-] CLI 使用轻量客户端环境"
}

Write-Stage "配置环境" "正在同步 Python 3.10 与锁定依赖..."
$syncArguments = @(
    "sync",
    "--project", $LauncherRoot,
    "--locked"
)
if ($Mode -eq "gui") {
    $syncArguments += @("--extra", "full", "--extra", $TorchProfile)
}
else {
    # CLI 只补齐客户端依赖，不移除已经由 WebUI 安装的完整算法环境。
    $syncArguments += "--inexact"
}
Invoke-Uv $syncArguments

$CheckMessage = if ($Mode -eq "gui") {
    "正在验证 Python、PyTorch 与项目资源..."
}
else {
    "正在验证 Python 与 CLI 依赖..."
}
Write-Stage "检查运行" $CheckMessage
if ($Mode -eq "gui") {
    Invoke-Uv @(
        "run",
        "--project", $LauncherRoot,
        "--locked",
        "--extra", "full",
        "--extra", $TorchProfile,
        "--no-sync",
        "python", "-B", "-c",
        "import sys, torch; print(f'      [-] Python: {sys.version.split()[0]}'); print(f'      [-] PyTorch: {torch.__version__}'); print(f'      [-] CUDA 可用: {torch.cuda.is_available()}')"
    )

    if ($TorchProfile -eq "cu130") {
        & $script:UvPath run --project $LauncherRoot --locked --extra full --extra $TorchProfile --no-sync python -B -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)"
        if ($LASTEXITCODE -ne 0) {
            Stop-Launcher "已安装 CUDA 版 PyTorch，但 CUDA 当前不可用。可使用 -Torch cpu 强制切换 CPU 环境。"
        }
    }
}
else {
    Invoke-Uv @(
        "run",
        "--project", $LauncherRoot,
        "--locked",
        "--no-sync",
        "python", "-B", "-c",
        "import sys, httpx, typer, rich; print(f'      [-] Python: {sys.version.split()[0]}'); print('      [-] CLI 依赖: 就绪')"
    )
}

if ($Mode -eq "gui") {
    $resources = @(
        @{ Path = "models\Qwen2.5-3B-Instruct\config.json"; Name = "Qwen 基础模型" },
        @{ Path = "models\Llama-3.2-3B-Instruct\config.json"; Name = "LLaMA 基础模型" },
        @{ Path = "models\DeepSeek-R1-Distill-Qwen-1.5B\config.json"; Name = "DeepSeek 基础模型" },
        @{ Path = "models\detectors\spectral_detector_llama.pkl"; Name = "静态检测器" },
        @{ Path = "datasets\clean_data_recovery.json"; Name = "恢复数据" },
        @{ Path = "datasets\clean_data_variants.json"; Name = "变体数据" },
        @{ Path = "datasets\qwen_multidomain_signatures.pt"; Name = "Qwen 离线签名" },
        @{ Path = "datasets\llama_multidomain_signatures.pt"; Name = "LLaMA 离线签名" },
        @{ Path = "datasets\deepseek_multidomain_signatures.pt"; Name = "DeepSeek 离线签名" }
    )
    foreach ($resource in $resources) {
        $resourcePath = Join-Path $ProjectRoot $resource.Path
        if (-not (Test-Path -LiteralPath $resourcePath -PathType Leaf)) {
            Write-Host "      [警告] $($resource.Name)缺失: $resourcePath" -ForegroundColor Yellow
        }
    }

    $portBusy = [System.Net.NetworkInformation.IPGlobalProperties]::GetIPGlobalProperties().GetActiveTcpListeners() |
        Where-Object { $_.Port -eq 7860 } |
        Select-Object -First 1
    if ($portBusy) {
        Stop-Launcher "WebUI 端口 7860 已被占用。"
    }
}

$ModeName = if ($Mode -eq "gui") { "WebUI" } else { "CLI" }
Write-Stage "启动程序" "正在启动 $ModeName..."

$runArguments = @(
    "run",
    "--project", $LauncherRoot,
    "--locked",
    "--no-sync",
    "python", "-B", "-m", $EntryModule
)
if ($Mode -eq "gui") {
    $runArguments = @(
        "run",
        "--project", $LauncherRoot,
        "--locked",
        "--extra", "full",
        "--extra", $TorchProfile,
        "--no-sync",
        "python", "-B", "-m", $EntryModule
    )
}

if ($Mode -eq "cli" -and $ApplicationArgs -and $ApplicationArgs[0] -eq "--") {
    $ApplicationArgs = $ApplicationArgs | Select-Object -Skip 1
}
if ($Mode -eq "cli" -and (-not $ApplicationArgs -or $ApplicationArgs.Count -eq 0)) {
    $ApplicationArgs = @("--help")
}

& $script:UvPath @runArguments @ApplicationArgs
$exitCode = $LASTEXITCODE
if ($exitCode -ne 0) {
    Write-Host "      [错误] $ModeName 已退出，退出码: $exitCode。" -ForegroundColor Red
}
exit $exitCode
