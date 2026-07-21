#Requires -Version 5.1
# Aegis-LoRA Windows 共用启动器：检查工具、配置环境并启动 GUI 或 CLI。

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
# 输出项目 ASCII 标题。
function Show-Banner {
    Write-Host ""
    Write-Host "    _    _____ ____ ___ ____       _     ___  ____      _    " -ForegroundColor Cyan
    Write-Host "   / \  | ____/ ___|_ _/ ___|     | |   / _ \|  _ \    / \   " -ForegroundColor Cyan
    Write-Host "  / _ \ |  _|| |  _ | |\___ \     | |  | | | | |_) |  / _ \  " -ForegroundColor Cyan
    Write-Host " / ___ \| |__| |_| || | ___) |    | |__| |_| |  _ <  / ___ \ " -ForegroundColor Cyan
    Write-Host "/_/   \_\_____\____|___|____/     |_____\___/|_| \_\/_/   \_\" -ForegroundColor Cyan
    Write-Host ""
}

# 使用统一格式显示当前执行阶段。
function Write-Stage([string]$Name, [string]$Message) {
    Write-Host ""
    Write-Host ">>> [$Name] $Message" -ForegroundColor Cyan
}

# 输出错误并使用指定退出码终止启动流程。
function Stop-Launcher([string]$Message, [int]$Code = 1) {
    Write-Host "      [错误] $Message" -ForegroundColor Red
    exit $Code
}

# =====================================================================
# 工具与环境
# =====================================================================
# 查找系统 PATH 或默认安装目录中的 uv。
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

# 查找可由当前终端直接调用的 Conda。
function Find-Conda {
    foreach ($name in @("conda.exe", "conda.bat", "conda")) {
        $command = Get-Command $name -ErrorAction SilentlyContinue
        if ($command) {
            return $command.Source
        }
    }

    return $null
}

# 在系统缺少 uv 时执行官方安装脚本。
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

# 根据显式参数或 NVIDIA 驱动选择 PyTorch 构建版本。
function Resolve-TorchProfile([string]$Requested) {
    if ($Requested -ne "auto") {
        return $Requested
    }

    $nvidiaSmi = Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue
    $nvidiaSmiPath = if ($nvidiaSmi) {
        $nvidiaSmi.Source
    }
    else {
        Join-Path ([Environment]::GetFolderPath("System")) "nvidia-smi.exe"
    }

    if (-not (Test-Path -LiteralPath $nvidiaSmiPath -PathType Leaf)) {
        Write-Host "      [警告] 未检测到 NVIDIA 驱动，使用 CPU 版 PyTorch。" -ForegroundColor Yellow
        return "cpu"
    }

    try {
        $gpuName = (& $nvidiaSmiPath --query-gpu=name --format=csv,noheader 2>$null | Select-Object -First 1).Trim()
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

# 执行 uv 命令并统一处理失败状态。
function Invoke-Uv([string[]]$Arguments) {
    & $script:UvPath @Arguments
    if ($LASTEXITCODE -ne 0) {
        Stop-Launcher "uv 命令执行失败，退出码: $LASTEXITCODE。" $LASTEXITCODE
    }
}

# 执行 Conda 命令并统一处理失败状态。
function Invoke-Conda([string[]]$Arguments) {
    & $script:CondaPath @Arguments
    if ($LASTEXITCODE -ne 0) {
        Stop-Launcher "Conda 命令执行失败，退出码: $LASTEXITCODE。" $LASTEXITCODE
    }
}

# 从 Conda 环境列表中定位指定名称的环境目录。
function Get-CondaEnvironmentPath([string]$Name) {
    $environmentJson = & $script:CondaPath env list --json
    if ($LASTEXITCODE -ne 0) {
        Stop-Launcher "无法读取 Conda 环境列表，退出码: $LASTEXITCODE。" $LASTEXITCODE
    }

    # PowerShell 5.1 会拒绝 envs_details 中仅大小写不同的重复路径键，因此只解析 envs 数组。
    $environmentMatch = [Regex]::Match(
        ($environmentJson -join "`n"),
        '(?s)"envs"\s*:\s*(\[[^\]]*\])'
    )
    if (-not $environmentMatch.Success) {
        Stop-Launcher "Conda 环境列表中缺少 envs 数组。"
    }

    try {
        $environmentPaths = $environmentMatch.Groups[1].Value | ConvertFrom-Json
    }
    catch {
        Stop-Launcher "Conda 环境列表解析失败: $($_.Exception.Message)"
    }

    foreach ($environmentPath in @($environmentPaths)) {
        if ((Split-Path -Leaf $environmentPath) -ieq $Name) {
            return $environmentPath
        }
    }

    return $null
}

# =====================================================================
# 启动流程
# =====================================================================
Show-Banner

# 统一解析项目入口、环境配置和依赖锁文件。
$LauncherRoot = $PSScriptRoot
$ProjectRoot = Split-Path -Parent $LauncherRoot
$ProjectFile = Join-Path $LauncherRoot "pyproject.toml"
$LockFile = Join-Path $LauncherRoot "uv.lock"
$CondaEnvName = "aegis_env"
$EntryName = if ($Mode -eq "gui") { "webui.py" } else { "cli.py" }
$EntryFile = Join-Path $LauncherRoot $EntryName
$EntryModule = if ($Mode -eq "gui") { "launcher.webui" } else { "launcher.cli" }
$CliCommandFile = Join-Path $LauncherRoot "aegis.cmd"

# 在配置环境前验证本次启动所需的项目文件。
if (-not (Test-Path -LiteralPath $EntryFile -PathType Leaf)) {
    Stop-Launcher "无法定位项目入口: $EntryFile"
}
if ($Mode -eq "cli" -and -not (Test-Path -LiteralPath $CliCommandFile -PathType Leaf)) {
    Stop-Launcher "CLI 命令入口缺失: $CliCommandFile"
}
if (-not (Test-Path -LiteralPath $ProjectFile -PathType Leaf)) {
    Stop-Launcher "环境配置缺失: $ProjectFile"
}
if (-not (Test-Path -LiteralPath $LockFile -PathType Leaf)) {
    Stop-Launcher "依赖锁文件缺失: $LockFile"
}

# 固定工作目录和运行时环境变量，避免依赖调用者所在目录。
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

# 检测 Conda、uv 和 GPU，并确定本次使用的运行配置。
Write-Stage "检查工具" "正在检查启动工具与运行设备..."
$script:CondaPath = Find-Conda
if ($script:CondaPath) {
    $condaVersion = & $script:CondaPath --version
    if ($LASTEXITCODE -ne 0) {
        Stop-Launcher "检测到 Conda，但无法正常执行。"
    }
    Write-Host "      [-] $condaVersion"
}
else {
    Write-Host "      [-] 未检测到 Conda，将使用 uv 创建 .venv" -ForegroundColor Yellow
}

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

# 优先复用 Conda 环境；环境不存在时创建 aegis_env。
if ($script:CondaPath) {
    Write-Stage "配置环境" "正在配置 Conda 环境 $CondaEnvName..."
    $condaEnvironmentPath = Get-CondaEnvironmentPath $CondaEnvName
    if (-not $condaEnvironmentPath) {
        Write-Host "      [-] 环境不存在，正在创建 Python 3.10 环境..."
        Invoke-Conda @("create", "--name", $CondaEnvName, "python=3.10", "--yes")
        $condaEnvironmentPath = Get-CondaEnvironmentPath $CondaEnvName
    }

    if (-not $condaEnvironmentPath) {
        Stop-Launcher "Conda 环境创建完成，但无法定位环境: $CondaEnvName"
    }

    $script:PythonPath = Join-Path $condaEnvironmentPath "python.exe"
    if (-not (Test-Path -LiteralPath $script:PythonPath -PathType Leaf)) {
        Stop-Launcher "Conda 环境中未找到 Python: $script:PythonPath"
    }

    $pythonMinorOutput = & $script:PythonPath -B -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
    if ($LASTEXITCODE -ne 0 -or -not $pythonMinorOutput) {
        Stop-Launcher "无法读取 Conda 环境 $CondaEnvName 的 Python 版本。"
    }
    $pythonMinor = ($pythonMinorOutput | Select-Object -First 1).Trim()
    if ($pythonMinor -ne "3.10") {
        Stop-Launcher "Conda 环境 $CondaEnvName 必须使用 Python 3.10，当前版本: $pythonMinor"
    }
    Write-Host "      [-] Conda 环境: $CondaEnvName ($condaEnvironmentPath)"

    # Conda 负责隔离 Python；uv 根据同一锁文件向该环境补齐依赖。
    $requirementsFile = Join-Path ([System.IO.Path]::GetTempPath()) "aegis_lora_requirements_$PID.txt"
    try {
        $exportArguments = @(
            "export",
            "--project", $LauncherRoot,
            "--locked",
            "--no-dev",
            "--no-emit-project",
            "--no-hashes",
            "--quiet",
            "--python", $script:PythonPath,
            "--output-file", $requirementsFile
        )
        if ($Mode -eq "gui") {
            $exportArguments += @("--extra", "full", "--extra", $TorchProfile)
        }
        Invoke-Uv $exportArguments

        $installArguments = @(
            "pip", "install",
            "--python", $script:PythonPath,
            "--requirements", $requirementsFile
        )
        if ($Mode -eq "gui") {
            $torchIndex = if ($TorchProfile -eq "cu130") {
                "https://download.pytorch.org/whl/cu130"
            }
            else {
                "https://download.pytorch.org/whl/cpu"
            }
            # 所有版本均由 uv.lock 固定；允许在 PyPI 与官方 PyTorch 索引间选择匹配版本。
            $installArguments += @(
                "--index", $torchIndex,
                "--index-strategy", "unsafe-best-match"
            )
        }
        Invoke-Uv $installArguments
    }
    finally {
        Remove-Item -LiteralPath $requirementsFile -Force -ErrorAction SilentlyContinue
    }
}
else {
    # 没有 Conda 时由 uv 创建并维护项目内的 .venv。
    Write-Stage "配置环境" "正在使用 uv 配置 Python 3.10 环境..."
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
    $script:PythonPath = Join-Path $env:UV_PROJECT_ENVIRONMENT "Scripts\python.exe"
}

# 后续检查和程序启动统一使用已确定的 Python 解释器。
if (-not (Test-Path -LiteralPath $script:PythonPath -PathType Leaf)) {
    Stop-Launcher "Python 环境配置完成，但未找到解释器: $script:PythonPath"
}

$CheckMessage = if ($Mode -eq "gui") {
    "正在验证 Python、PyTorch 与项目资源..."
}
else {
    "正在验证 Python 与 CLI 依赖..."
}
Write-Stage "检查运行" $CheckMessage

# 通过最小导入检查确认当前模式所需依赖可用。
if ($Mode -eq "gui") {
    & $script:PythonPath -B -c "import sys, torch; print(f'      [-] Python: {sys.version.split()[0]}'); print(f'      [-] PyTorch: {torch.__version__}'); print(f'      [-] CUDA 可用: {torch.cuda.is_available()}')"
    if ($LASTEXITCODE -ne 0) {
        Stop-Launcher "Python 或 PyTorch 运行检查失败，退出码: $LASTEXITCODE。" $LASTEXITCODE
    }

    if ($TorchProfile -eq "cu130") {
        & $script:PythonPath -B -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)"
        if ($LASTEXITCODE -ne 0) {
            Stop-Launcher "已安装 CUDA 版 PyTorch，但 CUDA 当前不可用。可使用 -Torch cpu 强制切换 CPU 环境。"
        }
    }
}
else {
    & $script:PythonPath -B -c "import sys, httpx, typer, rich; print(f'      [-] Python: {sys.version.split()[0]}'); print('      [-] CLI 依赖: 就绪')"
    if ($LASTEXITCODE -ne 0) {
        Stop-Launcher "Python 或 CLI 依赖检查失败，退出码: $LASTEXITCODE。" $LASTEXITCODE
    }
}

# GUI 启动前检查本地资源完整性和默认端口占用情况。
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

# 去除批处理转发参数时可能产生的分隔符。
if ($Mode -eq "cli" -and $ApplicationArgs -and $ApplicationArgs[0] -eq "--") {
    $ApplicationArgs = $ApplicationArgs | Select-Object -Skip 1
}

# 无参数 CLI 进入持续命令会话，其余情况直接启动目标程序。
$interactiveCli = $Mode -eq "cli" -and (-not $ApplicationArgs -or $ApplicationArgs.Count -eq 0)
if ($interactiveCli) {
    $env:AEGIS_PYTHON = $script:PythonPath
    $env:AEGIS_PROJECT_ROOT = $ProjectRoot
    $env:PATH = "$LauncherRoot;$env:PATH"

    & $script:PythonPath -B -m $EntryModule --help
    if ($LASTEXITCODE -ne 0) {
        Stop-Launcher "CLI 帮助加载失败，退出码: $LASTEXITCODE。" $LASTEXITCODE
    }

    Write-Host ""
    Write-Host "      输入 aegis <命令> 开始操作，输入 exit 退出。" -ForegroundColor Green
    Write-Host "      示例: aegis health" -ForegroundColor DarkGray
    Write-Host ""
    & $env:ComSpec /D /K 'title Aegis LoRA CLI & prompt AEGIS$G'
    $exitCode = $LASTEXITCODE
}
else {
    & $script:PythonPath -B -m $EntryModule @ApplicationArgs
    $exitCode = $LASTEXITCODE
}

if ($exitCode -ne 0) {
    Write-Host "      [错误] $ModeName 已退出，退出码: $exitCode。" -ForegroundColor Red
}
exit $exitCode
