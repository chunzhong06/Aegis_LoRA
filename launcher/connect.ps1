# Aegis-LoRA CLI 连接管理：配置 direct/local/ssh 模式并维护相关进程。
# 本文件由 start.ps1 点引入，只保留跨流程复用的探活、回收和统一动作入口。

# =====================================================================
# 复用逻辑
# =====================================================================
# 在限定时间内确认服务身份与 Token，并监控可选的托管进程。
function Wait-AegisApi {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)][string]$Server,
        [Parameter(Mandatory = $true)][string]$Token,
        [int]$TimeoutSeconds = 15,
        [Diagnostics.Process]$Process
    )

    # 截止时间控制整个探活过程，单次请求使用短超时以便及时响应进程退出。
    $deadline = [DateTime]::UtcNow.AddSeconds($TimeoutSeconds)
    $lastError = "API 未响应。"

    do {
        # 启动器创建的子进程若已退出，无需继续等待网络超时。
        if ($null -ne $Process) {
            $Process.Refresh()
            if ($Process.HasExited) {
                throw "连接进程已退出，退出码: $($Process.ExitCode)。"
            }
        }

        # 先用公开端点确认目标确为 Aegis，再向受保护端点发送 Token。
        try {
            $health = Invoke-RestMethod `
                -Uri "$Server/health" `
                -Method Get `
                -TimeoutSec 3 `
                -ErrorAction Stop
            if ($health.service -ne "Aegis-LoRA API") {
                throw "目标地址不是 Aegis-LoRA API。"
            }

            # 健康检查成功后只执行一次身份验证，错误 Token 不参与重试。
            try {
                $identity = Invoke-RestMethod `
                    -Uri "$Server/v1/me" `
                    -Method Get `
                    -Headers @{ Authorization = "Bearer $Token" } `
                    -TimeoutSec 3 `
                    -ErrorAction Stop
            }
            catch {
                throw [UnauthorizedAccessException]::new("API Token 验证失败: $($_.Exception.Message)")
            }
            if (-not $identity.authenticated -or $identity.service -ne "Aegis-LoRA API") {
                throw [UnauthorizedAccessException]::new("API Token 验证响应无效。")
            }
            return $true
        }
        # 认证错误立即返回；服务尚未监听等瞬时错误则等待后重试。
        catch {
            if ($_.Exception -is [UnauthorizedAccessException]) {
                throw
            }
            $lastError = $_.Exception.Message
        }

        if ([DateTime]::UtcNow -lt $deadline) {
            Start-Sleep -Seconds 1
        }
    } while ([DateTime]::UtcNow -lt $deadline)

    throw "API 连接失败: $lastError"
}

# 根据进程对象或持久化记录安全回收启动器托管的进程。
function Stop-AegisProcess {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)][string]$ProjectRoot,
        [Parameter(Mandatory = $true)][ValidateSet("local-api", "ssh-tunnel")][string]$Name,
        [Diagnostics.Process]$Process
    )

    # 每类托管进程使用独立记录，供后续终端恢复其 PID 与启动时间。
    $recordPath = Join-Path $ProjectRoot ".cache\connections\$Name.json"
    $processToStop = $Process

    # 没有直接传入进程时，从记录中恢复并核对启动时间，避免 PID 复用后误杀其他进程。
    if ($null -eq $processToStop -and (Test-Path -LiteralPath $recordPath -PathType Leaf)) {
        try {
            $record = Get-Content -Raw -LiteralPath $recordPath -Encoding UTF8 | ConvertFrom-Json
            $candidate = Get-Process -Id ([int]$record.pid) -ErrorAction SilentlyContinue
            if (
                $null -ne $candidate -and
                $candidate.ProcessName -eq [string]$record.process_name -and
                $candidate.StartTime.ToUniversalTime().Ticks -eq [Int64]$record.start_time_ticks
            ) {
                $processToStop = $candidate
            }
        }
        catch {
            Write-Host "      [警告] $Name 的进程记录无效: $($_.Exception.Message)" -ForegroundColor Yellow
        }
    }

    # 仅在确认进程已退出或记录已失效时删除记录，失败时保留重试依据。
    $stopped = $false
    $removeRecord = $null -eq $processToStop
    if ($null -ne $processToStop) {
        try {
            $processToStop.Refresh()
            if (-not $processToStop.HasExited) {
                Stop-Process -Id $processToStop.Id -Force -ErrorAction Stop
                if (-not $processToStop.WaitForExit(5000)) {
                    throw "等待进程退出超时。"
                }
                $stopped = $true
                Write-Host "      [-] 已停止 $Name。"
            }
            $processToStop.Refresh()
            $removeRecord = $processToStop.HasExited
        }
        catch {
            Write-Host "      [警告] 进程 $($processToStop.Id) 回收失败: $($_.Exception.Message)" -ForegroundColor Yellow
        }
    }

    if ($removeRecord) {
        Remove-Item -LiteralPath $recordPath -Force -ErrorAction SilentlyContinue
    }
    return $stopped
}

# =====================================================================
# 连接动作
# =====================================================================
# 统一处理配置、状态、启动与关闭动作，使 start.ps1 只负责流程编排。
function Invoke-AegisConnection {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)]
        [ValidateSet("read", "configure", "status", "stop", "open", "close")]
        [string]$Action,

        [ValidateSet("direct", "local", "ssh")]
        [string]$Mode,

        [string]$ConfigPath,
        $Config,
        [string]$PythonPath,
        [string]$ProjectRoot,
        $Connection
    )

    # -----------------------------------------------------------------
    # 读取配置
    # -----------------------------------------------------------------
    if ($Action -eq "read") {
        if (-not $ConfigPath) {
            if (-not $env:USERPROFILE) {
                throw "无法定位当前用户目录。"
            }
            $ConfigPath = Join-Path $env:USERPROFILE ".aegis\config.json"
        }

        # 首次运行没有配置时返回可用的默认对象，由调用方决定是否提示配置。
        if (-not (Test-Path -LiteralPath $ConfigPath -PathType Leaf)) {
            return [pscustomobject]@{
                ConfigPath = $ConfigPath
                IsConfigured = $false
                Mode = "direct"
                NeedsFullRuntime = $false
                Server = ""
                Token = ""
                ApiPort = 0
                Ssh = $null
            }
        }

        # 配置由 CLI 登录与启动器共同维护，读取后统一转换为运行时对象。
        try {
            $saved = Get-Content -Raw -LiteralPath $ConfigPath -Encoding UTF8 | ConvertFrom-Json
        }
        catch {
            throw "连接配置解析失败: $($_.Exception.Message)"
        }

        # 未包含 launcher 字段的旧登录配置按 direct 模式解释。
        $launcher = $saved.launcher
        $savedMode = if ($launcher -and $launcher.mode) {
            ([string]$launcher.mode).Trim().ToLowerInvariant()
        }
        else {
            "direct"
        }
        if ($savedMode -notin @("direct", "local", "ssh")) {
            throw "不支持的连接模式: $savedMode"
        }

        $server = ([string]$saved.server).Trim().TrimEnd("/")
        $token = [string]$saved.token
        $apiPort = 0
        $sshConfig = $null

        # 各模式在此归一化服务地址、端口和 SSH 字段，后续流程不再读取原始 JSON。
        if ($savedMode -eq "local") {
            $apiPort = if ($launcher.api_port) { [int]$launcher.api_port } else { 8000 }
            if ($apiPort -lt 1 -or $apiPort -gt 65535) {
                throw "本地 API 端口必须位于 1 到 65535 之间。"
            }
            $server = "http://127.0.0.1:$apiPort"
        }
        elseif ($savedMode -eq "ssh") {
            $savedSsh = $launcher.ssh
            $localPort = if ($launcher.local_port) { [int]$launcher.local_port } else { 18000 }
            $sshPort = if ($savedSsh.port) { [int]$savedSsh.port } else { 22 }
            $remoteApiPort = if ($savedSsh.remote_api_port) { [int]$savedSsh.remote_api_port } else { 8000 }
            if (
                $localPort -lt 1 -or $localPort -gt 65535 -or
                $sshPort -lt 1 -or $sshPort -gt 65535 -or
                $remoteApiPort -lt 1 -or $remoteApiPort -gt 65535
            ) {
                throw "SSH 连接端口必须位于 1 到 65535 之间。"
            }

            $identityFile = [Environment]::ExpandEnvironmentVariables(
                ([string]$savedSsh.identity_file).Trim()
            )
            if ($identityFile -eq "~") {
                $identityFile = $env:USERPROFILE
            }
            elseif ($identityFile.StartsWith("~\") -or $identityFile.StartsWith("~/")) {
                $identityFile = Join-Path $env:USERPROFILE $identityFile.Substring(2)
            }

            $server = "http://127.0.0.1:$localPort"
            $sshConfig = [pscustomobject]@{
                Target = ([string]$savedSsh.target).Trim()
                Port = $sshPort
                IdentityFile = $identityFile
                LocalPort = $localPort
                RemoteApiHost = if ($savedSsh.remote_api_host) {
                    ([string]$savedSsh.remote_api_host).Trim()
                }
                else {
                    "127.0.0.1"
                }
                RemoteApiPort = $remoteApiPort
                StartCommand = [string]$savedSsh.start_command
            }
        }
        elseif ($server -and $server -notmatch '^https?://') {
            throw "API 地址必须以 http:// 或 https:// 开头。"
        }

        $configured = [bool]($server -and $token)
        if ($savedMode -eq "ssh" -and -not $sshConfig.Target) {
            $configured = $false
        }

        # 返回固定结构，供 start.ps1 判断依赖规模并建立连接。
        return [pscustomobject]@{
            ConfigPath = $ConfigPath
            IsConfigured = $configured
            Mode = $savedMode
            NeedsFullRuntime = ($savedMode -eq "local")
            Server = $server
            Token = $token
            ApiPort = $apiPort
            Ssh = $sshConfig
        }
    }

    # -----------------------------------------------------------------
    # 交互配置
    # -----------------------------------------------------------------
    if ($Action -eq "configure") {
        if (-not $ConfigPath) {
            if (-not $env:USERPROFILE) {
                throw "无法定位当前用户目录。"
            }
            $ConfigPath = Join-Path $env:USERPROFILE ".aegis\config.json"
        }

        # 有效旧配置作为交互默认值；损坏配置只影响复用，不阻止重新创建。
        try {
            $current = Invoke-AegisConnection -Action read -ConfigPath $ConfigPath
        }
        catch {
            Write-Host "      [警告] 现有配置不可用，将重新创建: $($_.Exception.Message)" -ForegroundColor Yellow
            $current = [pscustomobject]@{ Mode = "direct"; Token = ""; Server = ""; ApiPort = 8000; Ssh = $null }
        }

        # 未通过参数指定模式时循环读取，直到得到 direct、local 或 ssh。
        if (-not $Mode) {
            do {
                $Mode = (Read-Host "连接模式 direct/local/ssh [$($current.Mode)]").Trim().ToLowerInvariant()
                if (-not $Mode) {
                    $Mode = $current.Mode
                }
                if ($Mode -notin @("direct", "local", "ssh")) {
                    Write-Host "      [错误] 连接模式只能是 direct、local 或 ssh。" -ForegroundColor Red
                }
            } while ($Mode -notin @("direct", "local", "ssh"))
        }

        # Token 使用安全输入；留空时保留已有值，避免普通文本回显。
        $tokenPrompt = if ($current.Token) { "API Token（留空保留当前值）" } else { "API Token" }
        $secureToken = Read-Host $tokenPrompt -AsSecureString
        $pointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secureToken)
        try {
            $token = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($pointer)
        }
        finally {
            [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($pointer)
        }
        if (-not $token) {
            $token = $current.Token
        }
        if (-not $token) {
            throw "API Token 不能为空。"
        }

        # 三种模式只收集各自实际需要的字段，并生成统一的 server 地址。
        $launcherConfig = [ordered]@{ version = 1; mode = $Mode }
        if ($Mode -eq "direct") {
            $defaultServer = if ($current.Mode -eq "direct") { $current.Server } else { "" }
            $server = (Read-Host "API 地址 [$defaultServer]").Trim()
            if (-not $server) {
                $server = $defaultServer
            }
            $server = $server.TrimEnd("/")
            if ($server -notmatch '^https?://') {
                throw "API 地址必须以 http:// 或 https:// 开头。"
            }
            $uri = [Uri]$server
            if ($uri.Scheme -eq "http" -and -not $uri.IsLoopback) {
                Write-Host "      [警告] 非本机直连地址正在使用 HTTP，建议通过 HTTPS 部署。" -ForegroundColor Yellow
            }
        }
        elseif ($Mode -eq "local") {
            $defaultPort = if ($current.Mode -eq "local") { $current.ApiPort } else { 8000 }
            $portValue = (Read-Host "本地 API 端口 [$defaultPort]").Trim()
            if (-not $portValue) {
                $portValue = [string]$defaultPort
            }
            $apiPort = 0
            if (-not [int]::TryParse($portValue, [ref]$apiPort) -or $apiPort -lt 1 -or $apiPort -gt 65535) {
                throw "本地 API 端口必须位于 1 到 65535 之间。"
            }
            $launcherConfig.api_port = $apiPort
            $server = "http://127.0.0.1:$apiPort"
        }
        else {
            $oldSsh = if ($current.Mode -eq "ssh") { $current.Ssh } else { $null }
            $defaultTarget = if ($oldSsh) { $oldSsh.Target } else { "" }
            $target = (Read-Host "SSH 目标 user@host [$defaultTarget]").Trim()
            if (-not $target) {
                $target = $defaultTarget
            }
            if ($target -notmatch '^[^\s@]+@[^\s@]+$') {
                throw "SSH 目标必须采用 user@host 格式。"
            }

            $defaultSshPort = if ($oldSsh) { $oldSsh.Port } else { 22 }
            $sshPortValue = (Read-Host "SSH 端口 [$defaultSshPort]").Trim()
            if (-not $sshPortValue) {
                $sshPortValue = [string]$defaultSshPort
            }
            $sshPort = 0
            if (-not [int]::TryParse($sshPortValue, [ref]$sshPort) -or $sshPort -lt 1 -or $sshPort -gt 65535) {
                throw "SSH 端口必须位于 1 到 65535 之间。"
            }

            $defaultIdentity = if ($oldSsh) { $oldSsh.IdentityFile } else { "" }
            $identityFile = (Read-Host "SSH 私钥路径（留空使用默认密钥） [$defaultIdentity]").Trim()
            if (-not $identityFile) {
                $identityFile = $defaultIdentity
            }
            $identityFile = [Environment]::ExpandEnvironmentVariables($identityFile)
            if ($identityFile -eq "~") {
                $identityFile = $env:USERPROFILE
            }
            elseif ($identityFile.StartsWith("~\") -or $identityFile.StartsWith("~/")) {
                $identityFile = Join-Path $env:USERPROFILE $identityFile.Substring(2)
            }
            if ($identityFile) {
                $identityFile = [System.IO.Path]::GetFullPath($identityFile)
            }

            $defaultLocalPort = if ($oldSsh) { $oldSsh.LocalPort } else { 18000 }
            $localPortValue = (Read-Host "本地转发端口 [$defaultLocalPort]").Trim()
            if (-not $localPortValue) {
                $localPortValue = [string]$defaultLocalPort
            }
            $localPort = 0
            if (-not [int]::TryParse($localPortValue, [ref]$localPort) -or $localPort -lt 1 -or $localPort -gt 65535) {
                throw "本地转发端口必须位于 1 到 65535 之间。"
            }

            $defaultRemoteHost = if ($oldSsh) { $oldSsh.RemoteApiHost } else { "127.0.0.1" }
            $remoteApiHost = (Read-Host "远端 API 主机 [$defaultRemoteHost]").Trim()
            if (-not $remoteApiHost) {
                $remoteApiHost = $defaultRemoteHost
            }

            $defaultRemotePort = if ($oldSsh) { $oldSsh.RemoteApiPort } else { 8000 }
            $remotePortValue = (Read-Host "远端 API 端口 [$defaultRemotePort]").Trim()
            if (-not $remotePortValue) {
                $remotePortValue = [string]$defaultRemotePort
            }
            $remoteApiPort = 0
            if (-not [int]::TryParse($remotePortValue, [ref]$remoteApiPort) -or $remoteApiPort -lt 1 -or $remoteApiPort -gt 65535) {
                throw "远端 API 端口必须位于 1 到 65535 之间。"
            }

            $defaultCommand = if ($oldSsh) { $oldSsh.StartCommand } else { "" }
            $startCommand = (Read-Host "远端 API 启动命令（留空表示服务已托管） [$defaultCommand]").Trim()
            if (-not $startCommand) {
                $startCommand = $defaultCommand
            }

            $launcherConfig.local_port = $localPort
            $launcherConfig.ssh = [ordered]@{
                target = $target
                port = $sshPort
                identity_file = $identityFile
                remote_api_host = $remoteApiHost
                remote_api_port = $remoteApiPort
                start_command = $startCommand
            }
            $server = "http://127.0.0.1:$localPort"
        }

        # 配置以无 BOM UTF-8 写入用户目录，保持与 Python CLI 的格式兼容。
        $parent = Split-Path -Parent $ConfigPath
        if (-not (Test-Path -LiteralPath $parent -PathType Container)) {
            $null = New-Item -ItemType Directory -Path $parent -Force
        }
        $saved = [ordered]@{ server = $server; token = $token; launcher = $launcherConfig }
        $json = $saved | ConvertTo-Json -Depth 6
        [System.IO.File]::WriteAllText(
            $ConfigPath,
            "$json`n",
            [System.Text.UTF8Encoding]::new($false)
        )

        Write-Host "      [-] 连接模式: $Mode"
        Write-Host "      [-] API 地址: $server"
        Write-Host "      [-] 配置已保存: $ConfigPath" -ForegroundColor Green
        return Invoke-AegisConnection -Action read -ConfigPath $ConfigPath
    }

    # -----------------------------------------------------------------
    # 状态与回收
    # -----------------------------------------------------------------
    # status 只读取当前配置并做短探活，不创建或回收任何进程。
    if ($Action -eq "status") {
        if ($null -eq $Config) {
            throw "状态检查缺少连接配置。"
        }
        Write-Host "      [-] 连接模式: $($Config.Mode)"
        if (-not $Config.IsConfigured) {
            Write-Host "      [警告] 尚未完成连接配置。" -ForegroundColor Yellow
            return $false
        }
        Write-Host "      [-] API 地址: $($Config.Server)"
        try {
            $null = Wait-AegisApi -Server $Config.Server -Token $Config.Token -TimeoutSeconds 3
            Write-Host "      [-] API 当前可用。" -ForegroundColor Green
            return $true
        }
        catch {
            Write-Host "      [错误] API 当前不可用: $($_.Exception.Message)" -ForegroundColor Red
            return $false
        }
    }

    # stop 回收两类持久化进程；重复执行保持幂等。
    if ($Action -eq "stop") {
        if (-not $ProjectRoot) {
            throw "连接回收缺少项目目录。"
        }
        $stoppedTunnel = Stop-AegisProcess -ProjectRoot $ProjectRoot -Name "ssh-tunnel"
        $stoppedApi = Stop-AegisProcess -ProjectRoot $ProjectRoot -Name "local-api"
        if (-not $stoppedTunnel -and -not $stoppedApi) {
            Write-Host "      [-] 没有由启动器管理的活动进程。"
        }
        return
    }

    # -----------------------------------------------------------------
    # 建立连接
    # -----------------------------------------------------------------
    if ($Action -eq "open") {
        if ($null -eq $Config -or -not $ProjectRoot) {
            throw "建立连接缺少配置或项目目录。"
        }

        # 会话对象只记录退出 CLI 时需要关闭的临时资源；本地 API 默认持续运行。
        $opened = [pscustomobject]@{
            Mode = $Config.Mode
            Server = $Config.Server
            TunnelProcess = $null
            TunnelStartedByLauncher = $false
        }
        if (-not $Config.IsConfigured) {
            Write-Host "      [警告] 尚未配置 API 连接；请先使用 -Action configure。" -ForegroundColor Yellow
            return $opened
        }

        # direct 不创建本地资源，只验证远端服务与 Token。
        if ($Config.Mode -eq "direct") {
            Write-Host ""
            Write-Host ">>> [验证服务] 正在验证直连 API..." -ForegroundColor Cyan
            $null = Wait-AegisApi -Server $Config.Server -Token $Config.Token -TimeoutSeconds 15
            Write-Host "      [-] API 地址: $($Config.Server)"
            return $opened
        }

        # local 和 ssh 都优先复用目标地址上已经可用的服务。
        try {
            $null = Wait-AegisApi -Server $Config.Server -Token $Config.Token -TimeoutSeconds 3
            Write-Host "      [-] 已复用现有 $($Config.Mode) 连接。"
            return $opened
        }
        catch {
            if ($Config.Mode -eq "ssh" -and $_.Exception -is [UnauthorizedAccessException]) {
                throw
            }
        }

        # 另一个终端可能已经启动了托管进程；先等待其就绪，再决定是否回收重建。
        $recordName = if ($Config.Mode -eq "local") { "local-api" } else { "ssh-tunnel" }
        $recordPath = Join-Path $ProjectRoot ".cache\connections\$recordName.json"
        $recordedProcess = $null
        if (Test-Path -LiteralPath $recordPath -PathType Leaf) {
            try {
                $record = Get-Content -Raw -LiteralPath $recordPath -Encoding UTF8 | ConvertFrom-Json
                $candidate = Get-Process -Id ([int]$record.pid) -ErrorAction SilentlyContinue
                if (
                    $null -ne $candidate -and
                    $candidate.ProcessName -eq [string]$record.process_name -and
                    $candidate.StartTime.ToUniversalTime().Ticks -eq [Int64]$record.start_time_ticks
                ) {
                    $recordedProcess = $candidate
                }
            }
            catch {
                # 无效记录由后面的统一回收逻辑清理。
            }
        }

        if ($null -ne $recordedProcess) {
            $startupTimeout = if ($Config.Mode -eq "local") { 30 } else { 60 }
            Write-Host "      [-] 正在等待已有 $recordName 就绪..."
            try {
                $null = Wait-AegisApi `
                    -Server $Config.Server `
                    -Token $Config.Token `
                    -TimeoutSeconds $startupTimeout `
                    -Process $recordedProcess
                Write-Host "      [-] 已复用启动中的 $recordName。"
                return $opened
            }
            catch {
                if ($Config.Mode -eq "ssh" -and $_.Exception -is [UnauthorizedAccessException]) {
                    throw
                }
                Write-Host "      [警告] 已有 $recordName 未能就绪，正在重建。" -ForegroundColor Yellow
            }
        }

        # local 与 ssh 的日志和进程记录统一保存在连接状态目录。
        $stateRoot = Join-Path $ProjectRoot ".cache\connections"
        if (-not (Test-Path -LiteralPath $stateRoot -PathType Container)) {
            $null = New-Item -ItemType Directory -Path $stateRoot -Force
        }

        # local 使用当前完整 Python 环境启动同仓库内的 FastAPI 服务。
        if ($Config.Mode -eq "local") {
            if (-not $PythonPath) {
                throw "启动本地 API 缺少 Python 路径。"
            }

            # 清理失效记录后再检查监听端口，避免覆盖其他程序。
            $null = Stop-AegisProcess -ProjectRoot $ProjectRoot -Name "local-api"
            $portBusy = [System.Net.NetworkInformation.IPGlobalProperties]::GetIPGlobalProperties().GetActiveTcpListeners() |
                Where-Object {
                    ($_.Address.IsLoopback -or
                        $_.Address.Equals([Net.IPAddress]::Any) -or
                        $_.Address.Equals([Net.IPAddress]::IPv6Any)) -and
                    $_.Port -eq $Config.ApiPort
                } |
                Select-Object -First 1
            if ($portBusy) {
                throw "本地端口 $($Config.ApiPort) 已被其他程序占用。"
            }

            Write-Host ""
            Write-Host ">>> [启动服务] 正在启动本地 API..." -ForegroundColor Cyan
            # Token 仅在创建子进程时注入，随后恢复当前终端原有环境。
            $tokenWasSet = Test-Path Env:AEGIS_API_TOKEN
            $previousToken = $env:AEGIS_API_TOKEN
            try {
                $env:AEGIS_API_TOKEN = $Config.Token
                $apiProcess = Start-Process `
                    -FilePath $PythonPath `
                    -ArgumentList @(
                        "-B", "-m", "uvicorn", "utils.api_server:app",
                        "--host", "127.0.0.1", "--port", [string]$Config.ApiPort
                    ) `
                    -WorkingDirectory $ProjectRoot `
                    -RedirectStandardOutput (Join-Path $stateRoot "local-api.stdout.log") `
                    -RedirectStandardError (Join-Path $stateRoot "local-api.stderr.log") `
                    -WindowStyle Hidden `
                    -PassThru
            }
            finally {
                if ($tokenWasSet) {
                    $env:AEGIS_API_TOKEN = $previousToken
                }
                else {
                    Remove-Item Env:AEGIS_API_TOKEN -ErrorAction SilentlyContinue
                }
            }

            # 先持久化进程身份，再等待 API 就绪；失败时统一回收进程与记录。
            try {
                $record = [ordered]@{
                    pid = $apiProcess.Id
                    process_name = $apiProcess.ProcessName
                    start_time_ticks = $apiProcess.StartTime.ToUniversalTime().Ticks
                }
                $recordJson = $record | ConvertTo-Json
                [System.IO.File]::WriteAllText(
                    (Join-Path $stateRoot "local-api.json"),
                    "$recordJson`n",
                    [System.Text.UTF8Encoding]::new($false)
                )

                Write-Host ""
                Write-Host ">>> [验证服务] 正在等待本地 API 就绪..." -ForegroundColor Cyan
                $null = Wait-AegisApi `
                    -Server $Config.Server `
                    -Token $Config.Token `
                    -TimeoutSeconds 90 `
                    -Process $apiProcess
                Write-Host "      [-] 本地 API 将保持运行；可使用 -Action stop 停止。"
                return $opened
            }
            catch {
                $null = Stop-AegisProcess -ProjectRoot $ProjectRoot -Name "local-api" -Process $apiProcess
                throw
            }
        }

        # ssh 使用本地端口转发访问远端 API，不在本机启动算法服务。
        if (-not $Config.Ssh.Target) {
            throw "SSH 目标不能为空。"
        }
        $null = Stop-AegisProcess -ProjectRoot $ProjectRoot -Name "ssh-tunnel"
        $portBusy = [System.Net.NetworkInformation.IPGlobalProperties]::GetIPGlobalProperties().GetActiveTcpListeners() |
            Where-Object {
                ($_.Address.IsLoopback -or
                    $_.Address.Equals([Net.IPAddress]::Any) -or
                    $_.Address.Equals([Net.IPAddress]::IPv6Any)) -and
                $_.Port -eq $Config.Ssh.LocalPort
            } |
            Select-Object -First 1
        if ($portBusy) {
            throw "本地端口 $($Config.Ssh.LocalPort) 已被其他程序占用。"
        }

        # 前台预检允许首次确认主机指纹，并可执行一次远端启动命令。
        $sshCommand = Get-Command ssh.exe -ErrorAction SilentlyContinue
        if (-not $sshCommand) {
            throw "未找到 Windows OpenSSH 客户端 ssh.exe。"
        }

        Write-Host ""
        Write-Host ">>> [建立连接] 正在连接 SSH 并创建端口转发..." -ForegroundColor Cyan
        $sshArguments = @("-T", "-o", "ConnectTimeout=10", "-p", [string]$Config.Ssh.Port)
        if ($Config.Ssh.IdentityFile) {
            $sshArguments += @("-i", $Config.Ssh.IdentityFile)
        }
        $sshArguments += $Config.Ssh.Target
        $sshArguments += if ($Config.Ssh.StartCommand) { $Config.Ssh.StartCommand } else { "exit 0" }
        & $sshCommand.Source @sshArguments
        if ($LASTEXITCODE -ne 0) {
            throw "SSH 预检或远端 API 启动命令失败，退出码: $LASTEXITCODE。"
        }

        # 后台隧道启用严格主机校验、转发失败退出和保活参数。
        $forward = "127.0.0.1:{0}:{1}:{2}" -f (
            $Config.Ssh.LocalPort,
            $Config.Ssh.RemoteApiHost,
            $Config.Ssh.RemoteApiPort
        )
        $tunnelArguments = @(
            "-N", "-T",
            "-o", "BatchMode=yes",
            "-o", "StrictHostKeyChecking=yes",
            "-o", "ExitOnForwardFailure=yes",
            "-o", "ServerAliveInterval=30",
            "-o", "ServerAliveCountMax=3",
            "-L", $forward,
            "-p", [string]$Config.Ssh.Port
        )
        if ($Config.Ssh.IdentityFile) {
            $tunnelArguments += @("-i", $Config.Ssh.IdentityFile)
        }
        $tunnelArguments += $Config.Ssh.Target
        # Start-Process 接收单个命令行，含空白的 SSH 参数需要显式引用。
        $commandLine = ($tunnelArguments | ForEach-Object {
            $value = [string]$_
            if ($value.Contains('"')) {
                throw "SSH 参数不能包含双引号: $value"
            }
            if ($value -match '\s') { '"' + $value + '"' } else { $value }
        }) -join " "

        # 隧道建立后记录进程身份并验证转发后的 Aegis API。
        try {
            $opened.TunnelProcess = Start-Process `
                -FilePath $sshCommand.Source `
                -ArgumentList $commandLine `
                -WorkingDirectory $ProjectRoot `
                -RedirectStandardOutput (Join-Path $stateRoot "ssh-tunnel.stdout.log") `
                -RedirectStandardError (Join-Path $stateRoot "ssh-tunnel.stderr.log") `
                -WindowStyle Hidden `
                -PassThru
            $opened.TunnelStartedByLauncher = $true

            $record = [ordered]@{
                pid = $opened.TunnelProcess.Id
                process_name = $opened.TunnelProcess.ProcessName
                start_time_ticks = $opened.TunnelProcess.StartTime.ToUniversalTime().Ticks
            }
            $recordJson = $record | ConvertTo-Json
            [System.IO.File]::WriteAllText(
                (Join-Path $stateRoot "ssh-tunnel.json"),
                "$recordJson`n",
                [System.Text.UTF8Encoding]::new($false)
            )

            Write-Host ""
            Write-Host ">>> [验证服务] 正在等待 SSH 隧道后的 API 就绪..." -ForegroundColor Cyan
            $null = Wait-AegisApi `
                -Server $Config.Server `
                -Token $Config.Token `
                -TimeoutSeconds 60 `
                -Process $opened.TunnelProcess
            Write-Host "      [-] SSH 目标: $($Config.Ssh.Target)"
            Write-Host "      [-] API 地址: $($Config.Server)"
            return $opened
        }
        catch {
            $null = Stop-AegisProcess `
                -ProjectRoot $ProjectRoot `
                -Name "ssh-tunnel" `
                -Process $opened.TunnelProcess
            throw
        }
    }

    # -----------------------------------------------------------------
    # 关闭当前会话创建的临时连接
    # -----------------------------------------------------------------
    # CLI 退出时只关闭本次会话创建的 SSH 隧道，不停止持久化本地 API。
    if ($Action -eq "close") {
        if (
            $null -ne $Connection -and
            $Connection.TunnelStartedByLauncher -and
            $null -ne $Connection.TunnelProcess
        ) {
            Write-Host ""
            Write-Host ">>> [回收连接] 正在关闭 SSH 隧道..." -ForegroundColor Cyan
            $null = Stop-AegisProcess `
                -ProjectRoot $ProjectRoot `
                -Name "ssh-tunnel" `
                -Process $Connection.TunnelProcess
        }
        return
    }
}
