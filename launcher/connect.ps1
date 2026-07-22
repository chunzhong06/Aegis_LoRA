# Aegis-LoRA CLI 连接管理：配置 direct/local/ssh 模式并维护托管进程。
# 本文件由 start.ps1 点引入，不单独处理 Python 环境与应用启动。

# =====================================================================
# 复用逻辑
# =====================================================================
# 读取带默认值的交互输入；端口及 SSH 配置会多次复用。
function Read-AegisValue([string]$Label, [string]$Default = "", [switch]$Required) {
    $prompt = if ($Default) { "$Label [$Default]" } else { $Label }
    $value = (Read-Host $prompt).Trim()
    if (-not $value) { $value = $Default }
    if ($Required -and -not $value) { throw "$Label 不能为空。" }
    return $value
}

# 读取并验证 TCP 端口，供 local 与 SSH 的四个端口字段共用。
function Read-AegisPort([string]$Label, [int]$Default) {
    $value = Read-AegisValue -Label $Label -Default ([string]$Default)
    $port = 0
    if (-not [int]::TryParse($value, [ref]$port) -or $port -lt 1 -or $port -gt 65535) {
        throw "$Label 必须位于 1 到 65535 之间。"
    }
    return $port
}

# 在限定时间内确认服务身份与 Token，并监控可选的托管进程。
function Wait-AegisApi {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)][string]$Server,
        [Parameter(Mandatory = $true)][string]$Token,
        [int]$TimeoutSeconds = 15,
        [Diagnostics.Process]$Process
    )

    $deadline = [DateTime]::UtcNow.AddSeconds($TimeoutSeconds)
    $lastError = "API 未响应。"
    do {
        if ($null -ne $Process) {
            $Process.Refresh()
            if ($Process.HasExited) { throw "连接进程已退出，退出码: $($Process.ExitCode)。" }
        }

        # 先确认公开端点属于 Aegis，避免向占用同一地址的其他服务发送 Token。
        try {
            $health = Invoke-RestMethod -Uri "$Server/health" -Method Get -TimeoutSec 3 -ErrorAction Stop
            if ($health.service -ne "Aegis-LoRA API") { throw "目标地址不是 Aegis-LoRA API。" }
        }
        catch {
            $lastError = $_.Exception.Message
            if ([DateTime]::UtcNow -lt $deadline) { Start-Sleep -Seconds 1 }
            continue
        }

        # 健康检查成功后验证受保护身份端点，认证类错误不再重复等待。
        try {
            $identity = Invoke-RestMethod -Uri "$Server/v1/me" -Method Get -Headers @{
                Authorization = "Bearer $Token"
            } -TimeoutSec 3 -ErrorAction Stop
        }
        catch {
            throw [UnauthorizedAccessException]::new("API 身份验证失败: $($_.Exception.Message)")
        }
        if (-not $identity.authenticated -or $identity.service -ne "Aegis-LoRA API") {
            throw [UnauthorizedAccessException]::new("API 身份验证响应无效。")
        }
        return $true
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

    $recordPath = Join-Path $ProjectRoot ".cache\connections\$Name.json"
    if ($null -eq $Process) {
        if (-not (Test-Path -LiteralPath $recordPath -PathType Leaf)) { return $false }
        try {
            $record = Get-Content -Raw -LiteralPath $recordPath -Encoding UTF8 | ConvertFrom-Json
            $candidate = Get-Process -Id ([int]$record.pid) -ErrorAction SilentlyContinue
            if (
                $null -ne $candidate -and
                $candidate.ProcessName -eq [string]$record.process_name -and
                $candidate.StartTime.ToUniversalTime().Ticks -eq [Int64]$record.start_time_ticks
            ) {
                $Process = $candidate
            }
        }
        catch {
            Write-Host "      [警告] $Name 的进程记录无效。" -ForegroundColor Yellow
        }
        if ($null -eq $Process) {
            Remove-Item -LiteralPath $recordPath -Force -ErrorAction SilentlyContinue
            return $false
        }
    }

    # 停止失败时保留记录，后续仍可依据 PID 与启动时间安全重试。
    try {
        $Process.Refresh()
        if ($Process.HasExited) {
            Remove-Item -LiteralPath $recordPath -Force -ErrorAction SilentlyContinue
            return $false
        }
        Stop-Process -Id $Process.Id -Force -ErrorAction Stop
        if (-not $Process.WaitForExit(5000)) { throw "等待进程退出超时。" }
        Remove-Item -LiteralPath $recordPath -Force -ErrorAction SilentlyContinue
        Write-Host "      [-] 已停止 $Name。"
        return $true
    }
    catch {
        Write-Host "      [警告] 进程 $($Process.Id) 回收失败: $($_.Exception.Message)" -ForegroundColor Yellow
        return $false
    }
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
        [ValidateSet("direct", "local", "ssh")][string]$Mode,
        [string]$ConfigPath,
        $Config,
        [string]$PythonPath,
        [string]$ProjectRoot,
        [Diagnostics.Process]$Process
    )

    if ($Action -in @("read", "configure") -and -not $ConfigPath) {
        $ConfigPath = Join-Path $env:USERPROFILE ".aegis\config.json"
    }

    # -----------------------------------------------------------------
    # 读取配置
    # -----------------------------------------------------------------
    if ($Action -eq "read") {
        if (-not (Test-Path -LiteralPath $ConfigPath -PathType Leaf)) {
            return [pscustomobject]@{
                IsConfigured = $false; Mode = "direct"; Server = ""; Token = ""; ApiPort = 0; Ssh = $null
            }
        }

        $saved = Get-Content -Raw -LiteralPath $ConfigPath -Encoding UTF8 | ConvertFrom-Json
        $launcher = $saved.launcher
        $savedMode = if ($launcher -and $launcher.mode) {
            ([string]$launcher.mode).Trim().ToLowerInvariant()
        } else { "direct" }
        if ($savedMode -notin @("direct", "local", "ssh")) { throw "不支持的连接模式: $savedMode" }

        $server = ([string]$saved.server).Trim().TrimEnd("/")
        $token = [string]$saved.token
        $apiPort = 0
        $sshConfig = $null

        if ($savedMode -eq "local") {
            $apiPort = if ($launcher.api_port) { [int]$launcher.api_port } else { 8000 }
            if ($apiPort -lt 1 -or $apiPort -gt 65535) { throw "本地 API 端口必须位于 1 到 65535 之间。" }
            $server = "http://127.0.0.1:$apiPort"
        }
        elseif ($savedMode -eq "ssh") {
            $savedSsh = $launcher.ssh
            $localPort = if ($launcher.local_port) { [int]$launcher.local_port } else { 18000 }
            $sshPort = if ($savedSsh.port) { [int]$savedSsh.port } else { 22 }
            $remotePort = if ($savedSsh.remote_api_port) { [int]$savedSsh.remote_api_port } else { 8000 }
            if (@($localPort, $sshPort, $remotePort) | Where-Object { $_ -lt 1 -or $_ -gt 65535 }) {
                throw "SSH 连接端口必须位于 1 到 65535 之间。"
            }

            $target = ([string]$savedSsh.target).Trim()
            if ($target -and ($target.StartsWith("-") -or $target -notmatch '^[^\s@]+@[^\s@]+$')) {
                throw "SSH target 必须采用 user@host 格式。"
            }
            $identity = [Environment]::ExpandEnvironmentVariables(([string]$savedSsh.identity_file).Trim())
            if ($identity -eq "~") { $identity = $env:USERPROFILE }
            elseif ($identity.StartsWith("~\") -or $identity.StartsWith("~/")) {
                $identity = Join-Path $env:USERPROFILE $identity.Substring(2)
            }
            if ($identity) { $identity = [IO.Path]::GetFullPath($identity) }

            $remoteHost = if ($savedSsh.remote_api_host) {
                ([string]$savedSsh.remote_api_host).Trim()
            } else { "127.0.0.1" }
            if (-not $remoteHost -or $remoteHost -match '\s') { throw "远端 API 主机无效。" }

            $server = "http://127.0.0.1:$localPort"
            $sshConfig = [pscustomobject]@{
                Target = $target; Port = $sshPort; IdentityFile = $identity; LocalPort = $localPort
                RemoteApiHost = $remoteHost; RemoteApiPort = $remotePort
                StartCommand = [string]$savedSsh.start_command
            }
        }
        elseif ($server -and $server -notmatch '^https?://') {
            throw "API 地址必须以 http:// 或 https:// 开头。"
        }

        $configured = [bool]($server -and $token -and ($savedMode -ne "ssh" -or $sshConfig.Target))
        return [pscustomobject]@{
            IsConfigured = $configured; Mode = $savedMode; Server = $server
            Token = $token; ApiPort = $apiPort; Ssh = $sshConfig
        }
    }

    # -----------------------------------------------------------------
    # 交互配置
    # -----------------------------------------------------------------
    if ($Action -eq "configure") {
        try { $current = Invoke-AegisConnection -Action read -ConfigPath $ConfigPath }
        catch {
            Write-Host "      [警告] 现有配置不可用，将重新创建。" -ForegroundColor Yellow
            $current = [pscustomobject]@{
                Mode = "direct"; Token = ""; Server = ""; ApiPort = 8000; Ssh = $null
            }
        }

        if (-not $Mode) {
            do {
                $Mode = (Read-AegisValue -Label "连接模式 direct/local/ssh" -Default $current.Mode).ToLowerInvariant()
                if ($Mode -notin @("direct", "local", "ssh")) {
                    Write-Host "      [错误] 连接模式只能是 direct、local 或 ssh。" -ForegroundColor Red
                }
            } while ($Mode -notin @("direct", "local", "ssh"))
        }

        # Token 使用安全输入，并在转换后立即清理非托管明文缓冲。
        $prompt = if ($current.Token) { "API Token（留空保留当前值）" } else { "API Token" }
        $secureToken = Read-Host $prompt -AsSecureString
        $pointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secureToken)
        try { $token = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($pointer) }
        finally { [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($pointer) }
        if (-not $token) { $token = $current.Token }
        if (-not $token) { throw "API Token 不能为空。" }

        $launcherConfig = [ordered]@{ version = 1; mode = $Mode }
        if ($Mode -eq "direct") {
            $defaultServer = if ($current.Mode -eq "direct") { $current.Server } else { "" }
            $server = (Read-AegisValue -Label "API 地址" -Default $defaultServer -Required).TrimEnd("/")
            if ($server -notmatch '^https?://') { throw "API 地址必须以 http:// 或 https:// 开头。" }
        }
        elseif ($Mode -eq "local") {
            $apiPort = Read-AegisPort "本地 API 端口" $(if ($current.Mode -eq "local") { $current.ApiPort } else { 8000 })
            $launcherConfig.api_port = $apiPort
            $server = "http://127.0.0.1:$apiPort"
        }
        else {
            $oldSsh = if ($current.Mode -eq "ssh") { $current.Ssh } else { $null }
            $target = Read-AegisValue "SSH 目标 user@host" $(if ($oldSsh) { $oldSsh.Target } else { "" }) -Required
            if ($target.StartsWith("-") -or $target -notmatch '^[^\s@]+@[^\s@]+$') {
                throw "SSH 目标必须采用 user@host 格式。"
            }
            $sshPort = Read-AegisPort "SSH 端口" $(if ($oldSsh) { $oldSsh.Port } else { 22 })
            $identity = Read-AegisValue "SSH 私钥路径（留空使用默认密钥）" $(if ($oldSsh) { $oldSsh.IdentityFile } else { "" })
            $localPort = Read-AegisPort "本地转发端口" $(if ($oldSsh) { $oldSsh.LocalPort } else { 18000 })
            $remoteHost = Read-AegisValue "远端 API 主机" $(if ($oldSsh) { $oldSsh.RemoteApiHost } else { "127.0.0.1" }) -Required
            if ($remoteHost -match '\s') { throw "远端 API 主机不能包含空白字符。" }
            $remotePort = Read-AegisPort "远端 API 端口" $(if ($oldSsh) { $oldSsh.RemoteApiPort } else { 8000 })
            $startCommand = Read-AegisValue "远端 API 启动命令（留空表示服务已托管）" $(if ($oldSsh) { $oldSsh.StartCommand } else { "" })

            $launcherConfig.local_port = $localPort
            $launcherConfig.ssh = [ordered]@{
                target = $target; port = $sshPort; identity_file = $identity
                remote_api_host = $remoteHost; remote_api_port = $remotePort
                start_command = $startCommand
            }
            $server = "http://127.0.0.1:$localPort"
        }

        $null = [IO.Directory]::CreateDirectory((Split-Path -Parent $ConfigPath))
        $json = [ordered]@{ server = $server; token = $token; launcher = $launcherConfig } |
            ConvertTo-Json -Depth 6
        [IO.File]::WriteAllText($ConfigPath, $json + [Environment]::NewLine, [Text.UTF8Encoding]::new($false))
        Write-Host "      [-] 连接模式: $Mode"
        Write-Host "      [-] API 地址: $server"
        Write-Host "      [-] 配置已保存: $ConfigPath" -ForegroundColor Green
        return Invoke-AegisConnection -Action read -ConfigPath $ConfigPath
    }

    # -----------------------------------------------------------------
    # 状态与回收
    # -----------------------------------------------------------------
    if ($Action -eq "status") {
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

    if ($Action -eq "stop") {
        $null = Stop-AegisProcess -ProjectRoot $ProjectRoot -Name "ssh-tunnel"
        $null = Stop-AegisProcess -ProjectRoot $ProjectRoot -Name "local-api"
        return
    }

    # -----------------------------------------------------------------
    # 建立连接
    # -----------------------------------------------------------------
    if ($Action -eq "open") {
        if (-not $Config.IsConfigured) {
            Write-Host "      [警告] 尚未配置 API 连接；请先选择连接模式。" -ForegroundColor Yellow
            return $null
        }
        if ($Config.Mode -eq "direct") {
            Write-Host ""
            Write-Host ">>> [验证服务] 正在验证直连 API..." -ForegroundColor Cyan
            $null = Wait-AegisApi -Server $Config.Server -Token $Config.Token -TimeoutSeconds 15
            Write-Host "      [-] API 地址: $($Config.Server)"
            return $null
        }

        # 已经就绪的 local/ssh 地址直接复用，不创建新的托管进程。
        try {
            $null = Wait-AegisApi -Server $Config.Server -Token $Config.Token -TimeoutSeconds 5
            Write-Host "      [-] 已复用现有 $($Config.Mode) 连接。"
            return $null
        }
        catch {
            if ($Config.Mode -eq "ssh" -and $_.Exception -is [UnauthorizedAccessException]) { throw }
        }

        $stateRoot = Join-Path $ProjectRoot ".cache\connections"
        $null = [IO.Directory]::CreateDirectory($stateRoot)
        $recordName = if ($Config.Mode -eq "local") { "local-api" } else { "ssh-tunnel" }
        $recordPath = Join-Path $stateRoot "$recordName.json"
        if (Test-Path -LiteralPath $recordPath -PathType Leaf) {
            throw "$recordName 正在启动或配置已经变化；请稍后重试，或先使用 -Action stop。"
        }

        $managedProcess = $null
        $timeout = if ($Config.Mode -eq "local") { 90 } else { 60 }
        try {
            if ($Config.Mode -eq "local") {
                Write-Host ""
                Write-Host ">>> [启动服务] 正在启动本地 API..." -ForegroundColor Cyan
                $tokenWasSet = Test-Path Env:AEGIS_API_TOKEN
                $previousToken = $env:AEGIS_API_TOKEN
                try {
                    $env:AEGIS_API_TOKEN = $Config.Token
                    $start = @{
                        FilePath = $PythonPath
                        ArgumentList = @("-B", "-m", "uvicorn", "utils.api_server:app", "--host", "127.0.0.1", "--port", [string]$Config.ApiPort)
                        WorkingDirectory = $ProjectRoot
                        RedirectStandardOutput = Join-Path $stateRoot "local-api.stdout.log"
                        RedirectStandardError = Join-Path $stateRoot "local-api.stderr.log"
                        WindowStyle = "Hidden"
                        PassThru = $true
                    }
                    $managedProcess = Start-Process @start
                }
                finally {
                    if ($tokenWasSet) { $env:AEGIS_API_TOKEN = $previousToken }
                    else { Remove-Item Env:AEGIS_API_TOKEN -ErrorAction SilentlyContinue }
                }
            }
            else {
                $ssh = Get-Command ssh.exe -ErrorAction Stop
                Write-Host ""
                Write-Host ">>> [建立连接] 正在连接 SSH 并创建端口转发..." -ForegroundColor Cyan

                # 前台预检允许首次确认主机指纹，并可执行一次远端启动命令。
                $preflight = @("-T", "-o", "ConnectTimeout=10", "-p", [string]$Config.Ssh.Port)
                if ($Config.Ssh.IdentityFile) { $preflight += @("-i", $Config.Ssh.IdentityFile) }
                $preflight += $Config.Ssh.Target
                $preflight += if ($Config.Ssh.StartCommand) { $Config.Ssh.StartCommand } else { "exit 0" }
                & $ssh.Source @preflight
                if ($LASTEXITCODE -ne 0) { throw "SSH 预检或远端 API 启动命令失败，退出码: $LASTEXITCODE。" }

                $forward = "127.0.0.1:{0}:{1}:{2}" -f $Config.Ssh.LocalPort, $Config.Ssh.RemoteApiHost, $Config.Ssh.RemoteApiPort
                $arguments = @(
                    "-N", "-T", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes"
                    "-o", "ExitOnForwardFailure=yes", "-L", $forward
                    "-p", [string]$Config.Ssh.Port
                )
                if ($Config.Ssh.IdentityFile) { $arguments += @("-i", $Config.Ssh.IdentityFile) }
                $arguments += $Config.Ssh.Target
                $commandLine = ($arguments | ForEach-Object {
                    $value = [string]$_
                    if ($value.Contains('"')) { throw "SSH 参数不能包含双引号: $value" }
                    if ($value -match '\s') { '"' + $value + '"' } else { $value }
                }) -join " "

                $start = @{
                    FilePath = $ssh.Source
                    ArgumentList = $commandLine
                    WorkingDirectory = $ProjectRoot
                    RedirectStandardOutput = Join-Path $stateRoot "ssh-tunnel.stdout.log"
                    RedirectStandardError = Join-Path $stateRoot "ssh-tunnel.stderr.log"
                    WindowStyle = "Hidden"
                    PassThru = $true
                }
                $managedProcess = Start-Process @start
            }

            # local 与 ssh 共用进程记录、就绪等待和失败回收流程。
            $record = [ordered]@{
                pid = $managedProcess.Id
                process_name = $managedProcess.ProcessName
                start_time_ticks = $managedProcess.StartTime.ToUniversalTime().Ticks
            } | ConvertTo-Json
            [IO.File]::WriteAllText($recordPath, $record + [Environment]::NewLine, [Text.UTF8Encoding]::new($false))
            Write-Host ""
            Write-Host ">>> [验证服务] 正在等待 $recordName 就绪..." -ForegroundColor Cyan
            $null = Wait-AegisApi -Server $Config.Server -Token $Config.Token -TimeoutSeconds $timeout -Process $managedProcess

            if ($Config.Mode -eq "local") {
                Write-Host "      [-] 本地 API 将保持运行；可使用 -Action stop 停止。"
                return $null
            }
            Write-Host "      [-] SSH 目标: $($Config.Ssh.Target)"
            Write-Host "      [-] API 地址: $($Config.Server)"
            return $managedProcess
        }
        catch {
            $null = Stop-AegisProcess -ProjectRoot $ProjectRoot -Name $recordName -Process $managedProcess
            throw
        }
    }

    # CLI 退出时只关闭本次会话创建的 SSH 隧道，本地 API 保持运行。
    if ($Action -eq "close") {
        if ($null -ne $Process) {
            Write-Host ""
            Write-Host ">>> [回收连接] 正在关闭 SSH 隧道..." -ForegroundColor Cyan
            $null = Stop-AegisProcess -ProjectRoot $ProjectRoot -Name "ssh-tunnel" -Process $Process
        }
        return
    }
}
