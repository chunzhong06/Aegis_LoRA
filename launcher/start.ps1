#Requires -Version 5.1
# Aegis-LoRA - Windows 共用启动器
# 本脚本负责解析入口参数、准备 Python 环境、建立 CLI 连接，并在退出时回收本次会话资源。

# Mode 决定 Python 入口；Torch 只影响需要模型运行时的 GUI / local CLI。
# ConnectionMode 可跳过 CLI 连接菜单；ApplicationArgs 原样传给最终 Python 模块。
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

# =====================================================================
# 控制台与公共输出
# =====================================================================
# 文件 BOM 保证 PowerShell 5.1 正确解析源码；以下设置统一当前进程与子进程输出。
$ErrorActionPreference = "Stop"
$Utf8Encoding = [Text.UTF8Encoding]::new($false)
[Console]::InputEncoding = $Utf8Encoding
[Console]::OutputEncoding = $Utf8Encoding
$OutputEncoding = [Console]::OutputEncoding

# Python 输出不依赖系统活动代码页，CLI、uvicorn 和依赖工具统一使用 UTF-8。
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

# 输出统一的启动阶段标题。
function Write-Stage([string]$Name, [string]$Message) {
    Write-Host ""
    Write-Host ">>> [$Name] $Message" -ForegroundColor Cyan
}

# 输出最终错误并把退出码传递给 start-cli.bat / start-gui.bat。
function Stop-Launcher([string]$Message, [int]$Code = 1) {
    Write-Host "      [错误] $Message" -ForegroundColor Red
    exit $Code
}

# 执行 uv；任何非零退出码都终止启动，避免在不完整环境中继续运行。
function Invoke-Uv([string[]]$Arguments) {
    & $script:UvPath @Arguments
    if ($LASTEXITCODE -ne 0) {
        Stop-Launcher "uv 命令执行失败，退出码: $LASTEXITCODE。" $LASTEXITCODE
    }
}

# 启动横幅只负责标识当前项目，不参与流程状态判断。
Write-Host ""
Write-Host "    _    _____ ____ ___ ____       _     ___  ____      _    " -ForegroundColor Cyan
Write-Host "   / \  | ____/ ___|_ _/ ___|     | |   / _ \|  _ \    / \   " -ForegroundColor Cyan
Write-Host "  / _ \ |  _|| |  _ | |\___ \     | |  | | | | |_) |  / _ \  " -ForegroundColor Cyan
Write-Host " / ___ \| |__| |_| || | ___) |    | |__| |_| |  _ <  / ___ \ " -ForegroundColor Cyan
Write-Host "/_/   \_\_____\____|___|____/     |_____\___/|_| \_\/_/   \_\" -ForegroundColor Cyan
Write-Host ""

# =====================================================================
# 路径与入口
# =====================================================================
# LauncherRoot 包含锁文件和入口脚本；ProjectRoot 是模型、数据与统一缓存的根目录。
$LauncherRoot = $PSScriptRoot
$ProjectRoot = Split-Path -Parent $LauncherRoot

# CondaEnvName 是可复用的命名环境；EntryModule 根据 Mode 选择 GUI 或 CLI。
$CondaEnvName = "aegis_env"
$EntryModule = if ($Mode -eq "gui") { "launcher.webui" } else { "launcher.cli" }

# CLI 专用路径：aegis.cmd 提供交互命令，connect.ps1 管理连接，CliStateRoot 统一保存连接文件。
$CliCommandFile = Join-Path $LauncherRoot "aegis.cmd"
$ConnectionScript = Join-Path $LauncherRoot "connect.ps1"
$CliStateRoot = Join-Path $ProjectRoot ".cache\cli"

# ConnectionConfig 只保存本次 CLI 的 API 参数；GUI 模式保持 null。
$ConnectionConfig = $null

# 批处理入口可能保留参数分隔符 `--`；移除后才能准确识别无参数交互 CLI。
if ($Mode -eq "cli" -and $ApplicationArgs -and $ApplicationArgs[0] -eq "--") {
    $ApplicationArgs = @($ApplicationArgs | Select-Object -Skip 1)
}
# interactiveCli 为 true 时进入持续 cmd 会话，否则只执行一次 Python 命令。
$interactiveCli = $Mode -eq "cli" -and (-not $ApplicationArgs -or $ApplicationArgs.Count -eq 0)

# =====================================================================
# CLI 连接模式
# =====================================================================
# GUI 不接受连接参数；CLI 在环境安装前确定 direct / local / ssh 及本次 API 参数。
if ($Mode -eq "gui") {
    if ($ConnectionMode) {
        Stop-Launcher "-ConnectionMode 仅适用于 CLI。"
    }
}
else {
    # 点引 connect.ps1 只导入连接函数，不会在此时创建进程或 SSH 文件。
    if (-not (Test-Path -LiteralPath $ConnectionScript -PathType Leaf)) {
        Stop-Launcher "连接管理脚本缺失: $ConnectionScript"
    }
    . $ConnectionScript

    # 显式 ConnectionMode 直接使用；省略时每次重新询问，不复用上次模式。
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

    # New-AegisConnectionConfig 在内存中生成本次连接对象，并只更新允许复用的 API 默认值。
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

# NeedsFullRuntime 决定是否安装模型、训练和 PyTorch 依赖。
# GUI 与 local CLI 需要完整环境；direct / ssh 只运行轻量 HTTP 客户端。
$NeedsFullRuntime = $Mode -eq "gui" -or (
    $Mode -eq "cli" -and $ConnectionConfig.Mode -eq "local"
)
# 交互 CLI 依赖 aegis.cmd；单次模块调用也保留该入口完整性检查。
if ($Mode -eq "cli" -and -not (Test-Path -LiteralPath $CliCommandFile -PathType Leaf)) {
    Stop-Launcher "CLI 命令入口缺失: $CliCommandFile"
}

# =====================================================================
# Python 与工具位置
# =====================================================================
# 所有相对路径从 ProjectRoot 解析；uv 环境和下载的 Python 均限制在项目目录中。
Set-Location -LiteralPath $ProjectRoot
$env:UV_PROJECT_ENVIRONMENT = Join-Path $ProjectRoot ".venv"
$env:UV_PYTHON_INSTALL_DIR = Join-Path $ProjectRoot ".cache\python"

# 启动器执行的 Python 不在源码目录生成 __pycache__ / .pyc。
$env:PYTHONDONTWRITEBYTECODE = "1"

# Conda 偶尔留下没有证书文件的 SSL_CERT_DIR；仅在确认目录为空时移除该变量。
if ($env:SSL_CERT_DIR -and (Test-Path -LiteralPath $env:SSL_CERT_DIR -PathType Container)) {
    $certificate = Get-ChildItem -LiteralPath $env:SSL_CERT_DIR -File -ErrorAction SilentlyContinue |
        Where-Object { $_.Extension -in ".pem", ".crt", ".cer" } |
        Select-Object -First 1
    if (-not $certificate) { Remove-Item Env:SSL_CERT_DIR }
}

# Conda 存在时复用 aegis_env；不存在时由 uv 维护项目内 .venv。
Write-Stage "检查工具" "正在检查启动工具与运行设备..."
$condaCommand = Get-Command conda.exe, conda.bat, conda -ErrorAction SilentlyContinue |
    Select-Object -First 1
$script:CondaPath = if ($condaCommand) { $condaCommand.Source } else { $null }
if (-not $script:CondaPath) {
    Write-Host "      [-] 未检测到 Conda，将使用 uv 创建 .venv" -ForegroundColor Yellow
}

# UvPath 按 PATH、默认用户安装目录的顺序解析；两处都不存在时调用官方安装器。
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
    # 安装器只在 uv 确实缺失时执行；TLS 1.2 兼容旧版 Windows PowerShell。
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

# =====================================================================
# PyTorch 构建选择
# =====================================================================
# TorchProfile 是最终传给 uv extra 和 PyTorch 索引的构建名称。
# 轻量 CLI 不需要 PyTorch，因此保持 null 并跳过整个硬件探测。
$TorchProfile = $null
if ($NeedsFullRuntime) {
    $TorchProfile = $Torch
    if ($TorchProfile -eq "auto") {
        # 自动探测以 CPU 为保底；只有驱动和计算能力都满足要求时才切换 CUDA 构建。
        $TorchProfile = "cpu"
        $nvidiaSmi = Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue

        # nvidiaSmiPath 优先使用 PATH，回退到 Windows 系统目录中的驱动工具。
        $nvidiaSmiPath = if ($nvidiaSmi) {
            $nvidiaSmi.Source
        }
        else {
            Join-Path ([Environment]::GetFolderPath("System")) "nvidia-smi.exe"
        }

        if (Test-Path -LiteralPath $nvidiaSmiPath -PathType Leaf) {
            try {
                # gpuOutput 提供名称和计算能力；driverOutput 提供驱动支持的最高 CUDA 版本。
                $gpuOutput = @(& $nvidiaSmiPath --query-gpu=name,compute_cap --format=csv,noheader 2>$null)
                $gpuExitCode = $LASTEXITCODE
                $driverOutput = @(& $nvidiaSmiPath 2>$null)
                if ($gpuExitCode -ne 0 -or $LASTEXITCODE -ne 0 -or $gpuOutput.Count -eq 0) {
                    throw "nvidia-smi 返回异常结果。"
                }

                # computeCapability 用于排除新版 PyTorch 已不支持的旧 GPU 架构。
                $gpuFields = $gpuOutput[0] -split ",\s*", 2
                $gpuName = $gpuFields[0].Trim()
                $computeCapability = if ($gpuFields.Count -gt 1) {
                    [Version]$gpuFields[1].Trim()
                }
                else {
                    $null
                }
                # cudaMatch 从完整驱动信息中提取 CUDA 上限，例如 12.8。
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

                    # profiles 按最低驱动版本从高到低排列，选择驱动能够支持的最新锁定构建。
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

                        # 较旧计算能力对 CUDA 12.8 / 13.0 轮子兼容有限，向下选择可用构建。
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
                # 硬件探测异常不阻断启动，保留先前设置的 CPU 安全回退。
                Write-Host "      [警告] NVIDIA 驱动探测失败: $($_.Exception.Message)" -ForegroundColor Yellow
            }
        }
        else {
            Write-Host "      [警告] 未检测到 NVIDIA 驱动，使用 CPU 版 PyTorch。" -ForegroundColor Yellow
        }
    }

    # 输出最终选择；用户显式指定的 Torch 参数同样会到达这里。
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

# =====================================================================
# Python 环境同步
# =====================================================================
# Conda 分支复用命名环境；uv 仍以 uv.lock 为唯一依赖来源。
if ($script:CondaPath) {
    Write-Stage "配置环境" "正在配置 Conda 环境 $CondaEnvName..."

    # pythonOutput 用于从 conda run 输出中定位环境解释器；不存在时创建固定 Python 3.10 环境。
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
    # PythonPath 是后续依赖安装、CUDA 验证和最终入口共用的解释器绝对路径。
    $script:PythonPath = $pythonOutput |
        ForEach-Object { ([string]$_).Trim() } |
        Where-Object { $_ -and (Test-Path -LiteralPath $_ -PathType Leaf) } |
        Select-Object -Last 1
    if (-not $script:PythonPath) {
        Stop-Launcher "Conda 环境中未找到 Python 解释器。"
    }

    # 同名 Conda 环境也必须验证版本，防止复用不兼容解释器。
    $pythonMinor = @(& $script:PythonPath -B -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    $pythonVersion = $pythonMinor | Select-Object -First 1
    if ($LASTEXITCODE -ne 0 -or -not $pythonVersion -or $pythonVersion.Trim() -ne "3.10") {
        Stop-Launcher "Conda 环境 $CondaEnvName 必须使用 Python 3.10。"
    }
    Write-Host "      [-] Conda 环境: $CondaEnvName"

    # requirementsFile 是 uv.lock 到 conda pip 安装之间的一次性桥接文件。
    # 文件位于系统临时目录，并由 finally 无条件删除。
    $requirementsFile = Join-Path ([IO.Path]::GetTempPath()) "aegis_lora_requirements_$PID.txt"
    try {
        # exportArguments 只导出运行依赖；完整环境额外加入 full 和选定 TorchProfile。
        $exportArguments = @(
            "export", "--project", $LauncherRoot, "--locked", "--no-dev",
            "--no-emit-project", "--no-hashes", "--quiet",
            "--python", $script:PythonPath, "--output-file", $requirementsFile
        )
        if ($NeedsFullRuntime) {
            $exportArguments += @("--extra", "full", "--extra", $TorchProfile)
        }
        Invoke-Uv $exportArguments

        # installArguments 把锁定依赖安装进同一个 Conda Python，不创建第二套环境。
        $installArguments = @(
            "pip", "install", "--python", $script:PythonPath,
            "--requirements", $requirementsFile
        )
        if ($NeedsFullRuntime) {
            # PyTorch 官方索引提供对应 CUDA 构建，unsafe-best-match 允许其余依赖回到默认索引解析。
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
    # 无 Conda 时由 uv 直接创建项目 .venv；锁文件和 Python 版本仍来自 launcher 项目。
    Write-Stage "配置环境" "正在使用 uv 配置 Python 3.10 环境..."

    # 轻量同步使用 --inexact，避免 direct / ssh 启动时卸载已有完整算法依赖。
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
# 环境同步完成后必须得到可执行 Python，后续连接和入口都依赖此路径。
if (-not (Test-Path -LiteralPath $script:PythonPath -PathType Leaf)) {
    Stop-Launcher "环境配置完成，但未找到 Python: $script:PythonPath"
}

# 非 CPU 构建必须在当前解释器中实际访问 CUDA，避免安装成功但运行设备不可用。
# 其他依赖由真实入口加载，以保留原始导入错误。
if ($NeedsFullRuntime -and $TorchProfile -ne "cpu") {
    Write-Stage "验证运行" "正在验证 CUDA 运行环境..."
    & $script:PythonPath -B -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)"
    if ($LASTEXITCODE -ne 0) {
        Stop-Launcher "CUDA 版 PyTorch 已安装，但 CUDA 当前不可用；可使用 -Torch cpu。"
    }
}

# =====================================================================
# 建立连接并启动应用
# =====================================================================
# ManagedConnection 只在 local / ssh 模式持有进程、Job 句柄和 SessionRoot。
# direct 不创建本地资源，因此 Open-AegisConnection 验证成功后返回 null。
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

# 保存调用者原有 API 环境，确保本脚本被点引或嵌套调用时可以完整恢复。
$serverWasSet = Test-Path Env:AEGIS_API_SERVER
$previousServer = $env:AEGIS_API_SERVER
$tokenWasSet = Test-Path Env:AEGIS_API_TOKEN
$previousToken = $env:AEGIS_API_TOKEN

# CLI 只消费这两个会话变量，不读取 config.json 或任何 SSH 配置。
if ($Mode -eq "cli") {
    $env:AEGIS_API_SERVER = $ConnectionConfig.Server
    $env:AEGIS_API_TOKEN = $ConnectionConfig.Token
}

# exitCode 保存 Python 模块或交互 cmd 的退出状态，最终原样返回批处理入口。
$exitCode = 0
try {
    # ModeName 只用于终端显示；EntryModule 已在路径阶段确定真实 Python 入口。
    $ModeName = if ($Mode -eq "gui") { "WebUI" } else { "CLI" }
    Write-Stage "启动程序" "正在启动 $ModeName..."
    if ($interactiveCli) {
        # aegis.cmd 通过 AEGIS_PYTHON 和 AEGIS_PROJECT_ROOT 定位当前环境与模块。
        # LauncherRoot 临时加入 PATH，使持续 cmd 会话可直接执行 aegis。
        $env:AEGIS_PYTHON = $script:PythonPath
        $env:AEGIS_PROJECT_ROOT = $ProjectRoot
        $env:PATH = "$LauncherRoot;$env:PATH"

        # 先验证 CLI 能够加载并显示帮助，再进入保留全部会话变量的 cmd。
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
        # 非交互模式只执行一次入口模块，ApplicationArgs 保持原始顺序传递。
        & $script:PythonPath -B -m $EntryModule @ApplicationArgs
        $exitCode = $LASTEXITCODE
    }
    if ($exitCode -ne 0) {
        Write-Host "      [错误] $ModeName 已退出，退出码: $exitCode。" -ForegroundColor Red
    }
}
finally {
    # -----------------------------------------------------------------
    # 回收本次启动资源
    # -----------------------------------------------------------------
    # 只关闭 Open-AegisConnection 返回的托管对象，不扫描或接管历史进程。
    if ($Mode -eq "cli" -and $null -ne $ManagedConnection) {
        Close-AegisConnection -StateRoot $CliStateRoot -ManagedConnection $ManagedConnection
    }

    # 根据 *WasSet 区分“原值为空”和“原变量不存在”，精确恢复调用前状态。
    if ($Mode -eq "cli") {
        if ($serverWasSet) { $env:AEGIS_API_SERVER = $previousServer }
        else { Remove-Item Env:AEGIS_API_SERVER -ErrorAction SilentlyContinue }
        if ($tokenWasSet) { $env:AEGIS_API_TOKEN = $previousToken }
        else { Remove-Item Env:AEGIS_API_TOKEN -ErrorAction SilentlyContinue }
    }
}
exit $exitCode
