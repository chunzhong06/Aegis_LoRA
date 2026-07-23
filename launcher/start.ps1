#Requires -Version 5.1
# Aegis-LoRA Windows 共用启动器：配置运行环境并启动 GUI 或 CLI。

[CmdletBinding(PositionalBinding = $false)]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("gui", "cli")]
    [string]$Mode,

    [ValidateSet("auto", "cu118", "cu121", "cu124", "cu126", "cu128", "cu130", "cpu")]
    [string]$Torch = "auto",

    [ValidateSet("direct", "local", "ssh")]
    [string]$ConnectionMode,

    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ApplicationArgs
)

# 统一错误处理与控制台编码，使异常和中文输出可被外层入口正确接收。
$ErrorActionPreference = "Stop"
[Console]::InputEncoding = [Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = [Text.UTF8Encoding]::new($false)
$OutputEncoding = [Console]::OutputEncoding

# 显示当前执行阶段；启动流程中的所有阶段共用这一格式。
function Write-Stage([string]$Name, [string]$Message) {
    Write-Host ""
    Write-Host ">>> [$Name] $Message" -ForegroundColor Cyan
}

# 输出启动错误并将退出码传递给批处理入口。
function Stop-Launcher([string]$Message, [int]$Code = 1) {
    Write-Host "      [错误] $Message" -ForegroundColor Red
    exit $Code
}

# 执行 uv 并统一终止失败的依赖操作。
function Invoke-Uv([string[]]$Arguments) {
    & $script:UvPath @Arguments
    if ($LASTEXITCODE -ne 0) {
        Stop-Launcher "uv 命令执行失败，退出码: $LASTEXITCODE。" $LASTEXITCODE
    }
}

# 启动时保留项目 ASCII 标题。
Write-Host ""
Write-Host "    _    _____ ____ ___ ____       _     ___  ____      _    " -ForegroundColor Cyan
Write-Host "   / \  | ____/ ___|_ _/ ___|     | |   / _ \|  _ \    / \   " -ForegroundColor Cyan
Write-Host "  / _ \ |  _|| |  _ | |\___ \     | |  | | | | |_) |  / _ \  " -ForegroundColor Cyan
Write-Host " / ___ \| |__| |_| || | ___) |    | |__| |_| |  _ <  / ___ \ " -ForegroundColor Cyan
Write-Host "/_/   \_\_____\____|___|____/     |_____\___/|_| \_\/_/   \_\" -ForegroundColor Cyan
Write-Host ""

# 解析固定路径与入口模块；缺失的项目元数据由 uv --locked 或 Python 入口报告。
$LauncherRoot = $PSScriptRoot
$ProjectRoot = Split-Path -Parent $LauncherRoot
$CondaEnvName = "aegis_env"
$EntryModule = if ($Mode -eq "gui") { "launcher.webui" } else { "launcher.cli" }
$CliCommandFile = Join-Path $LauncherRoot "aegis.cmd"
$ConnectionScript = Join-Path $LauncherRoot "connect.ps1"
$CliStateRoot = Join-Path $ProjectRoot ".cache\cli"
$ConnectionConfig = $null

# 批处理入口可能传入参数分隔符；移除后才能准确识别无参数交互 CLI。
if ($Mode -eq "cli" -and $ApplicationArgs -and $ApplicationArgs[0] -eq "--") {
    $ApplicationArgs = @($ApplicationArgs | Select-Object -Skip 1)
}
$interactiveCli = $Mode -eq "cli" -and (-not $ApplicationArgs -or $ApplicationArgs.Count -eq 0)

# GUI 不处理连接参数；CLI 在安装 Python 环境前完成模式选择和本次连接配置。
if ($Mode -eq "gui") {
    if ($ConnectionMode) {
        Stop-Launcher "-ConnectionMode 仅适用于 CLI。"
    }
}
else {
    if (-not (Test-Path -LiteralPath $ConnectionScript -PathType Leaf)) {
        Stop-Launcher "连接管理脚本缺失: $ConnectionScript"
    }
    . $ConnectionScript

    # ConnectionMode 是命令行上的显式选择；没有传入时必须在本次启动中重新选择。
    if ($ConnectionMode) {
        $selectedMode = $ConnectionMode
    }
    else {
        Write-Stage "选择模式" "请选择本次 CLI 使用的 API 方式..."
        Write-Host "      [1] direct  连接已有 API"
        Write-Host "      [2] local   启动本次会话的本地 API"
        Write-Host "      [3] ssh     创建本次会话的一次性 SSH 隧道"
        do {
            $choice = (Read-Host "连接模式").Trim().ToLowerInvariant()
            if ($choice -in @("1", "direct")) {
                $selectedMode = "direct"
            }
            elseif ($choice -in @("2", "local")) {
                $selectedMode = "local"
            }
            elseif ($choice -in @("3", "ssh")) {
                $selectedMode = "ssh"
            }
            else {
                $selectedMode = $null
                Write-Host "      [错误] 必须输入 1、2、3 或对应的模式名称。" -ForegroundColor Red
            }
        } while (-not $selectedMode)
    }

    # configure 在内存中生成本次连接对象；持久文件只更新允许保存的 API 默认值。
    Write-Stage "配置连接" "正在配置本次 $selectedMode 连接..."
    try {
        $ConnectionConfig = New-AegisConnectionConfig -Mode $selectedMode `
            -StateRoot $CliStateRoot
    }
    catch {
        Stop-Launcher $_.Exception.Message
    }
    Write-Host "      [-] 连接模式: $($ConnectionConfig.Mode)"
    Write-Host "      [-] 配置目录: $CliStateRoot"
}
# GUI 与 local CLI 需要算法环境；direct/ssh CLI 只需要轻量客户端环境。
$NeedsFullRuntime = $Mode -eq "gui" -or (
    $Mode -eq "cli" -and $ConnectionConfig.Mode -eq "local"
)
if ($Mode -eq "cli" -and -not (Test-Path -LiteralPath $CliCommandFile -PathType Leaf)) {
    Stop-Launcher "CLI 命令入口缺失: $CliCommandFile"
}

# 固定项目目录和 uv 环境位置，并禁止 Python 在源码目录生成字节码缓存。
Set-Location -LiteralPath $ProjectRoot
$env:UV_PROJECT_ENVIRONMENT = Join-Path $ProjectRoot ".venv"
$env:UV_PYTHON_INSTALL_DIR = Join-Path $ProjectRoot ".cache\python"
$env:PYTHONDONTWRITEBYTECODE = "1"

# Conda 偶尔留下无证书文件的占位目录；这种目录会干扰后续 HTTPS 请求。
if ($env:SSL_CERT_DIR -and (Test-Path -LiteralPath $env:SSL_CERT_DIR -PathType Container)) {
    $certificate = Get-ChildItem -LiteralPath $env:SSL_CERT_DIR -File -ErrorAction SilentlyContinue |
        Where-Object { $_.Extension -in ".pem", ".crt", ".cer" } |
        Select-Object -First 1
    if (-not $certificate) { Remove-Item Env:SSL_CERT_DIR }
}

# Conda 可选：检测到时复用命名环境，否则由 uv 维护项目内 .venv。
Write-Stage "检查工具" "正在检查启动工具与运行设备..."
$condaCommand = Get-Command conda.exe, conda.bat, conda -ErrorAction SilentlyContinue |
    Select-Object -First 1
$script:CondaPath = if ($condaCommand) { $condaCommand.Source } else { $null }
if (-not $script:CondaPath) {
    Write-Host "      [-] 未检测到 Conda，将使用 uv 创建 .venv" -ForegroundColor Yellow
}

# uv 优先使用 PATH 和默认用户目录；均不存在时调用官方安装器。
$uvCommand = Get-Command uv.exe -ErrorAction SilentlyContinue
$uvCandidate = Join-Path $env:USERPROFILE ".local\bin\uv.exe"
$script:UvPath = if ($uvCommand) {
    $uvCommand.Source
}
elseif (Test-Path -LiteralPath $uvCandidate -PathType Leaf) {
    $uvCandidate
}
else {
    $null
}
if (-not $script:UvPath) {
    Write-Host "      [-] 未检测到 uv，正在安装..."
    try {
        [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
        Invoke-Expression (Invoke-RestMethod "https://astral.sh/uv/install.ps1")
    }
    catch {
        Stop-Launcher "uv 安装失败: $($_.Exception.Message)"
    }
    if (-not (Test-Path -LiteralPath $uvCandidate -PathType Leaf)) {
        Stop-Launcher "uv 安装完成，但未找到 uv.exe。"
    }
    $script:UvPath = $uvCandidate
}
Write-Host "      [-] 项目目录: $ProjectRoot"

# 完整运行时按显式参数或 NVIDIA 驱动能力选择锁文件提供的 PyTorch 构建。
$TorchProfile = $null
if ($NeedsFullRuntime) {
    $TorchProfile = $Torch
    if ($TorchProfile -eq "auto") {
        $TorchProfile = "cpu"
        $nvidiaSmi = Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue
        $nvidiaSmiPath = if ($nvidiaSmi) {
            $nvidiaSmi.Source
        }
        else {
            Join-Path ([Environment]::GetFolderPath("System")) "nvidia-smi.exe"
        }

        if (Test-Path -LiteralPath $nvidiaSmiPath -PathType Leaf) {
            try {
                $gpuOutput = @(& $nvidiaSmiPath --query-gpu=name,compute_cap --format=csv,noheader 2>$null)
                $gpuExitCode = $LASTEXITCODE
                $driverOutput = @(& $nvidiaSmiPath 2>$null)
                if ($gpuExitCode -ne 0 -or $LASTEXITCODE -ne 0 -or $gpuOutput.Count -eq 0) {
                    throw "nvidia-smi 返回异常结果。"
                }

                $gpuFields = $gpuOutput[0] -split ",\s*", 2
                $gpuName = $gpuFields[0].Trim()
                $computeCapability = if ($gpuFields.Count -gt 1) {
                    [Version]$gpuFields[1].Trim()
                }
                else {
                    $null
                }
                $cudaMatch = [Regex]::Match(
                    ($driverOutput -join [Environment]::NewLine),
                    'CUDA(?: UMD)? Version:\s*(\d+\.\d+)'
                )
                if (-not $cudaMatch.Success) {
                    Write-Host "      [警告] 无法读取驱动 CUDA 上限，使用 CPU 版 PyTorch。" -ForegroundColor Yellow
                }
                elseif ($computeCapability -and $computeCapability -lt [Version]"6.0") {
                    Write-Host "      [警告] GPU 计算能力低于 6.0，使用 CPU 版 PyTorch。" -ForegroundColor Yellow
                }
                else {
                    $cudaVersion = [Version]$cudaMatch.Groups[1].Value
                    $profiles = @(
                        @{ Minimum = [Version]"13.0"; Name = "cu130" }
                        @{ Minimum = [Version]"12.8"; Name = "cu128" }
                        @{ Minimum = [Version]"12.6"; Name = "cu126" }
                        @{ Minimum = [Version]"12.4"; Name = "cu124" }
                        @{ Minimum = [Version]"12.1"; Name = "cu121" }
                        @{ Minimum = [Version]"11.8"; Name = "cu118" }
                    )
                    $profile = $profiles |
                        Where-Object { $cudaVersion -ge $_.Minimum } |
                        Select-Object -First 1
                    if ($profile) {
                        $TorchProfile = $profile.Name
                        if ($computeCapability -and $computeCapability -lt [Version]"7.0" -and $TorchProfile -in @("cu128", "cu130")) {
                            $TorchProfile = "cu126"
                        }
                        elseif ($computeCapability -and $computeCapability -lt [Version]"7.5" -and $TorchProfile -eq "cu130") {
                            $TorchProfile = "cu128"
                        }
                        Write-Host "      [-] NVIDIA GPU: $gpuName"
                        Write-Host "      [-] 驱动 CUDA 上限: $cudaVersion"
                    }
                    else {
                        Write-Host "      [警告] CUDA 低于 11.8，使用 CPU 版 PyTorch。" -ForegroundColor Yellow
                    }
                }
            }
            catch {
                Write-Host "      [警告] NVIDIA 驱动探测失败: $($_.Exception.Message)" -ForegroundColor Yellow
            }
        }
        else {
            Write-Host "      [警告] 未检测到 NVIDIA 驱动，使用 CPU 版 PyTorch。" -ForegroundColor Yellow
        }
    }

    Write-Host "      [-] PyTorch 配置: $TorchProfile"
    if ($Mode -eq "gui" -and $TorchProfile -eq "cpu") {
        Write-Host "      [提示] CPU 本地推理较慢；已有远程 API 时可使用 start-cli.bat。" -ForegroundColor Yellow
    }
    elseif ($Mode -eq "cli") {
        Write-Host "      [-] CLI 本地 API 使用完整算法环境"
    }
}
else {
    Write-Host "      [-] CLI 使用轻量客户端环境"
}

# Conda 管理解释器时，uv 从锁文件导出依赖并安装到同一 Python。
if ($script:CondaPath) {
    Write-Stage "配置环境" "正在配置 Conda 环境 $CondaEnvName..."
    $pythonOutput = @(& $script:CondaPath run -n $CondaEnvName python -B -c "import sys; print(sys.executable)" 2>$null)
    if ($LASTEXITCODE -ne 0) {
        Write-Host "      [-] 环境不可用，正在创建 Python 3.10 环境..."
        & $script:CondaPath create --name $CondaEnvName "python=3.10" --yes
        if ($LASTEXITCODE -ne 0) {
            Stop-Launcher "Conda 环境创建失败，退出码: $LASTEXITCODE。" $LASTEXITCODE
        }
        $pythonOutput = @(& $script:CondaPath run -n $CondaEnvName python -B -c "import sys; print(sys.executable)" 2>$null)
    }
    if ($LASTEXITCODE -ne 0) {
        Stop-Launcher "无法定位 Conda 环境 $CondaEnvName 的 Python。"
    }
    $script:PythonPath = $pythonOutput |
        ForEach-Object { ([string]$_).Trim() } |
        Where-Object { $_ -and (Test-Path -LiteralPath $_ -PathType Leaf) } |
        Select-Object -Last 1
    if (-not $script:PythonPath) {
        Stop-Launcher "Conda 环境中未找到 Python 解释器。"
    }

    # 项目固定使用 Python 3.10，防止复用同名但版本不兼容的环境。
    $pythonMinor = @(& $script:PythonPath -B -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    $pythonVersion = $pythonMinor | Select-Object -First 1
    if ($LASTEXITCODE -ne 0 -or -not $pythonVersion -or $pythonVersion.Trim() -ne "3.10") {
        Stop-Launcher "Conda 环境 $CondaEnvName 必须使用 Python 3.10。"
    }
    Write-Host "      [-] Conda 环境: $CondaEnvName"

    # 临时 requirements 仅用于在 Conda Python 中复现 uv.lock，离开本块即删除。
    $requirementsFile = Join-Path ([IO.Path]::GetTempPath()) "aegis_lora_requirements_$PID.txt"
    try {
        $exportArguments = @(
            "export", "--project", $LauncherRoot, "--locked", "--no-dev",
            "--no-emit-project", "--no-hashes", "--quiet",
            "--python", $script:PythonPath, "--output-file", $requirementsFile
        )
        if ($NeedsFullRuntime) {
            $exportArguments += @("--extra", "full", "--extra", $TorchProfile)
        }
        Invoke-Uv $exportArguments

        $installArguments = @(
            "pip", "install", "--python", $script:PythonPath,
            "--requirements", $requirementsFile
        )
        if ($NeedsFullRuntime) {
            $installArguments += @(
                "--index", "https://download.pytorch.org/whl/$TorchProfile",
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
    # 没有 Conda 时由 uv 创建 .venv；轻量同步不移除已存在的完整算法依赖。
    Write-Stage "配置环境" "正在使用 uv 配置 Python 3.10 环境..."
    $syncArguments = @("sync", "--project", $LauncherRoot, "--locked")
    if ($NeedsFullRuntime) {
        $syncArguments += @("--extra", "full", "--extra", $TorchProfile)
    }
    else {
        $syncArguments += "--inexact"
    }
    Invoke-Uv $syncArguments
    $script:PythonPath = Join-Path $env:UV_PROJECT_ENVIRONMENT "Scripts\python.exe"
}
if (-not (Test-Path -LiteralPath $script:PythonPath -PathType Leaf)) {
    Stop-Launcher "环境配置完成，但未找到 Python: $script:PythonPath"
}

# 非 CPU 构建必须能访问 CUDA；其他依赖由随后真实入口加载并给出原始错误。
if ($NeedsFullRuntime -and $TorchProfile -ne "cpu") {
    Write-Stage "验证运行" "正在验证 CUDA 运行环境..."
    & $script:PythonPath -B -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)"
    if ($LASTEXITCODE -ne 0) {
        Stop-Launcher "CUDA 版 PyTorch 已安装，但 CUDA 当前不可用；可使用 -Torch cpu。"
    }
}

# Python 环境就绪后建立本次连接；direct 返回 null，local/ssh 返回本次会话的托管对象。
$ManagedConnection = $null
if ($Mode -eq "cli") {
    try {
        $ManagedConnection = Open-AegisConnection -Config $ConnectionConfig `
            -StateRoot $CliStateRoot -PythonPath $script:PythonPath -ProjectRoot $ProjectRoot
    }
    catch {
        Stop-Launcher $_.Exception.Message
    }
}

# 当前 API 地址和 Token 只注入本次 PowerShell 及其子进程；aegis 子命令不会读取旧 SSH 配置。
$serverWasSet = Test-Path Env:AEGIS_API_SERVER
$previousServer = $env:AEGIS_API_SERVER
$tokenWasSet = Test-Path Env:AEGIS_API_TOKEN
$previousToken = $env:AEGIS_API_TOKEN
$configFileWasSet = Test-Path Env:AEGIS_CONFIG_FILE
$previousConfigFile = $env:AEGIS_CONFIG_FILE
if ($Mode -eq "cli") {
    $env:AEGIS_API_SERVER = $ConnectionConfig.Server
    $env:AEGIS_API_TOKEN = $ConnectionConfig.Token
    $env:AEGIS_CONFIG_FILE = Join-Path $CliStateRoot "config.json"
}
# 主程序退出码原样返回，finally 仅回收当前会话创建的托管连接。
$exitCode = 0
try {
    $ModeName = if ($Mode -eq "gui") { "WebUI" } else { "CLI" }
    Write-Stage "启动程序" "正在启动 $ModeName..."
    if ($interactiveCli) {
        $env:AEGIS_PYTHON = $script:PythonPath
        $env:AEGIS_PROJECT_ROOT = $ProjectRoot
        $env:PATH = "$LauncherRoot;$env:PATH"

        # 先显示一次帮助，再进入保留 Aegis 环境变量的持续命令会话。
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
}
finally {
    # local/ssh 只关闭本次启动返回的托管对象；direct 没有子进程，因此无需回收。
    if ($Mode -eq "cli" -and $null -ne $ManagedConnection) {
        Close-AegisConnection -StateRoot $CliStateRoot -ManagedConnection $ManagedConnection
    }

    # 恢复启动器进入前的环境，避免被其他调用方式点引入时污染父 PowerShell。
    if ($Mode -eq "cli") {
        if ($serverWasSet) { $env:AEGIS_API_SERVER = $previousServer }
        else { Remove-Item Env:AEGIS_API_SERVER -ErrorAction SilentlyContinue }
        if ($tokenWasSet) { $env:AEGIS_API_TOKEN = $previousToken }
        else { Remove-Item Env:AEGIS_API_TOKEN -ErrorAction SilentlyContinue }
        if ($configFileWasSet) { $env:AEGIS_CONFIG_FILE = $previousConfigFile }
        else { Remove-Item Env:AEGIS_CONFIG_FILE -ErrorAction SilentlyContinue }
    }
}
exit $exitCode
