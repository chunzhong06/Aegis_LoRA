# Aegis-LoRA CLI 连接管理：配置本地/直连/SSH 模式并维护相关进程。
# 本文件由 start.ps1 点引入，不单独负责 Python 环境或 CLI 启动。

# =====================================================================
# 配置读写
# =====================================================================
function Get-AegisConfigPath {
    if (-not $env:USERPROFILE) {
        throw "无法定位当前用户目录。"
    }
    return Join-Path $env:USERPROFILE ".aegis\config.json"
}

function Get-AegisProperty($Object, [string]$Name, $Default = $null) {
    if ($null -ne $Object -and $null -ne $Object.PSObject.Properties[$Name]) {
        return $Object.PSObject.Properties[$Name].Value
    }
    return $Default
}

function Read-AegisConfigFile([string]$ConfigPath, [switch]$AllowMissing) {
    if (-not (Test-Path -LiteralPath $ConfigPath -PathType Leaf)) {
        if ($AllowMissing) {
            return $null
        }
        throw "连接配置不存在: $ConfigPath"
    }

    try {
        return Get-Content -Raw -LiteralPath $ConfigPath -Encoding UTF8 | ConvertFrom-Json
    }
    catch {
        throw "连接配置解析失败: $($_.Exception.Message)"
    }
}

function Write-AegisJsonFile([string]$Path, $Value) {
    $parent = Split-Path -Parent $Path
    if (-not (Test-Path -LiteralPath $parent -PathType Container)) {
        $null = New-Item -ItemType Directory -Path $parent -Force
    }

    $json = $Value | ConvertTo-Json -Depth 8
    [System.IO.File]::WriteAllText(
        $Path,
        "$json`n",
        [System.Text.UTF8Encoding]::new($false)
    )
}

function ConvertTo-AegisPort($Value, [string]$Name) {
    try {
        $port = [int]$Value
    }
    catch {
        throw "$Name 必须是整数。"
    }
    if ($port -lt 1 -or $port -gt 65535) {
        throw "$Name 必须位于 1 到 65535 之间。"
    }
    return $port
}

function ConvertTo-AegisServer([string]$Server) {
    $serverValue = $Server.Trim().TrimEnd("/")
    $uri = $null
    if (
        -not [Uri]::TryCreate($serverValue, [UriKind]::Absolute, [ref]$uri) -or
        $uri.Scheme -notin @("http", "https")
    ) {
        throw "API 地址必须是有效的 http:// 或 https:// 地址。"
    }
    return $serverValue
}

function ConvertFrom-AegisSecureString([Security.SecureString]$SecureValue) {
    $pointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($SecureValue)
    try {
        return [Runtime.InteropServices.Marshal]::PtrToStringBSTR($pointer)
    }
    finally {
        [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($pointer)
    }
}

function Read-AegisValue(
    [string]$Label,
    [string]$Default = "",
    [switch]$Required
) {
    $prompt = if ($Default) { "$Label [$Default]" } else { $Label }
    $value = (Read-Host $prompt).Trim()
    if (-not $value) {
        $value = $Default
    }
    if ($Required -and -not $value) {
        throw "$Label 不能为空。"
    }
    return $value
}

function Read-AegisPort([string]$Label, [int]$Default) {
    $value = Read-AegisValue -Label $Label -Default ([string]$Default)
    return ConvertTo-AegisPort $value $Label
}

function Resolve-AegisIdentityPath([string]$Path) {
    if (-not $Path) {
        return ""
    }

    $expanded = [Environment]::ExpandEnvironmentVariables($Path.Trim())
    if ($expanded -eq "~") {
        $expanded = $env:USERPROFILE
    }
    elseif ($expanded.StartsWith("~\") -or $expanded.StartsWith("~/")) {
        $expanded = Join-Path $env:USERPROFILE $expanded.Substring(2)
    }

    $expanded = [System.IO.Path]::GetFullPath($expanded)
    if (-not (Test-Path -LiteralPath $expanded -PathType Leaf)) {
        throw "SSH 私钥不存在: $expanded"
    }
    return $expanded
}

function Get-AegisConnectionConfig {
    [CmdletBinding()]
    param(
        [string]$ConfigPath = (Get-AegisConfigPath)
    )

    $config = Read-AegisConfigFile -ConfigPath $ConfigPath -AllowMissing
    if ($null -eq $config) {
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

    $launcher = Get-AegisProperty $config "launcher"
    $mode = [string](Get-AegisProperty $launcher "mode" "direct")
    $mode = $mode.Trim().ToLowerInvariant()
    if ($mode -notin @("direct", "local", "ssh")) {
        throw "不支持的连接模式: $mode"
    }

    $server = [string](Get-AegisProperty $config "server" "")
    $token = [string](Get-AegisProperty $config "token" "")
    $apiPort = 0
    $sshConfig = $null

    if ($mode -eq "local") {
        $apiPort = ConvertTo-AegisPort (Get-AegisProperty $launcher "api_port" 8000) "本地 API 端口"
        $server = "http://127.0.0.1:$apiPort"
    }
    elseif ($mode -eq "ssh") {
        $localPort = ConvertTo-AegisPort (Get-AegisProperty $launcher "local_port" 18000) "本地转发端口"
        $ssh = Get-AegisProperty $launcher "ssh"
        $target = [string](Get-AegisProperty $ssh "target" "")
        $sshPort = ConvertTo-AegisPort (Get-AegisProperty $ssh "port" 22) "SSH 端口"
        $identityFile = Resolve-AegisIdentityPath ([string](Get-AegisProperty $ssh "identity_file" ""))
        $remoteApiHost = [string](Get-AegisProperty $ssh "remote_api_host" "127.0.0.1")
        $remoteApiPort = ConvertTo-AegisPort (Get-AegisProperty $ssh "remote_api_port" 8000) "远端 API 端口"
        $startCommand = [string](Get-AegisProperty $ssh "start_command" "")

        if (-not $target -or $target -notmatch '^[^\s@]+@[^\s@]+$') {
            throw "SSH target 必须采用 user@host 格式。"
        }
        if (-not $remoteApiHost -or $remoteApiHost -match '\s') {
            throw "远端 API 主机无效。"
        }

        $server = "http://127.0.0.1:$localPort"
        $sshConfig = [pscustomobject]@{
            Target = $target
            Port = $sshPort
            IdentityFile = $identityFile
            LocalPort = $localPort
            RemoteApiHost = $remoteApiHost
            RemoteApiPort = $remoteApiPort
            StartCommand = $startCommand
        }
    }
    elseif ($server) {
        $server = ConvertTo-AegisServer $server
    }

    $isConfigured = [bool]($server -and $token)
    return [pscustomobject]@{
        ConfigPath = $ConfigPath
        IsConfigured = $isConfigured
        Mode = $mode
        NeedsFullRuntime = ($mode -eq "local")
        Server = $server
        Token = $token
        ApiPort = $apiPort
        Ssh = $sshConfig
    }
}

function Set-AegisConnectionConfig {
    [CmdletBinding()]
    param(
        [ValidateSet("direct", "local", "ssh")]
        [string]$Mode,

        [string]$ConfigPath = (Get-AegisConfigPath)
    )

    $existing = $null
    try {
        $existing = Read-AegisConfigFile -ConfigPath $ConfigPath -AllowMissing
    }
    catch {
        Write-Host "      [警告] 现有配置不可用，将重新创建: $($_.Exception.Message)" -ForegroundColor Yellow
    }

    $existingLauncher = Get-AegisProperty $existing "launcher"
    $currentMode = [string](Get-AegisProperty $existingLauncher "mode" "direct")
    if (-not $Mode) {
        while ($true) {
            $Mode = (Read-AegisValue -Label "连接模式 direct/local/ssh" -Default $currentMode).ToLowerInvariant()
            if ($Mode -in @("direct", "local", "ssh")) {
                break
            }
            Write-Host "      [错误] 连接模式只能是 direct、local 或 ssh。" -ForegroundColor Red
        }
    }

    $existingToken = [string](Get-AegisProperty $existing "token" "")
    $tokenPrompt = if ($existingToken) { "API Token（留空保留当前值）" } else { "API Token" }
    $secureToken = Read-Host $tokenPrompt -AsSecureString
    $token = ConvertFrom-AegisSecureString $secureToken
    if (-not $token) {
        $token = $existingToken
    }
    if (-not $token) {
        throw "API Token 不能为空。"
    }

    $launcherConfig = [ordered]@{ version = 1; mode = $Mode }
    $server = ""

    if ($Mode -eq "direct") {
        $defaultServer = if ($currentMode -eq "direct") {
            [string](Get-AegisProperty $existing "server" "")
        }
        else {
            ""
        }
        $server = ConvertTo-AegisServer (Read-AegisValue -Label "API 地址" -Default $defaultServer -Required)
        $uri = [Uri]$server
        if ($uri.Scheme -eq "http" -and -not $uri.IsLoopback) {
            Write-Host "      [警告] 非本机直连地址正在使用 HTTP，建议通过 HTTPS 部署。" -ForegroundColor Yellow
        }
    }
    elseif ($Mode -eq "local") {
        $defaultPort = if ($currentMode -eq "local") {
            [int](Get-AegisProperty $existingLauncher "api_port" 8000)
        }
        else {
            8000
        }
        $apiPort = Read-AegisPort -Label "本地 API 端口" -Default $defaultPort
        $launcherConfig.api_port = $apiPort
        $server = "http://127.0.0.1:$apiPort"
    }
    else {
        $existingSsh = Get-AegisProperty $existingLauncher "ssh"
        $defaultTarget = [string](Get-AegisProperty $existingSsh "target" "")
        $defaultSshPort = [int](Get-AegisProperty $existingSsh "port" 22)
        $defaultIdentity = [string](Get-AegisProperty $existingSsh "identity_file" "")
        $defaultLocalPort = [int](Get-AegisProperty $existingLauncher "local_port" 18000)
        $defaultRemoteHost = [string](Get-AegisProperty $existingSsh "remote_api_host" "127.0.0.1")
        $defaultRemotePort = [int](Get-AegisProperty $existingSsh "remote_api_port" 8000)
        $defaultStartCommand = [string](Get-AegisProperty $existingSsh "start_command" "")

        $target = Read-AegisValue -Label "SSH 目标 user@host" -Default $defaultTarget -Required
        if ($target -notmatch '^[^\s@]+@[^\s@]+$') {
            throw "SSH 目标必须采用 user@host 格式。"
        }
        $sshPort = Read-AegisPort -Label "SSH 端口" -Default $defaultSshPort
        $identityFile = Resolve-AegisIdentityPath (
            Read-AegisValue -Label "SSH 私钥路径（留空使用 ssh-agent 或默认密钥）" -Default $defaultIdentity
        )
        $localPort = Read-AegisPort -Label "本地转发端口" -Default $defaultLocalPort
        $remoteApiHost = Read-AegisValue -Label "远端 API 主机" -Default $defaultRemoteHost -Required
        if ($remoteApiHost -match '\s') {
            throw "远端 API 主机不能包含空白字符。"
        }
        $remoteApiPort = Read-AegisPort -Label "远端 API 端口" -Default $defaultRemotePort
        $startCommand = Read-AegisValue -Label "远端 API 启动命令（应立即返回；留空表示服务已托管）" -Default $defaultStartCommand

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

    $config = [ordered]@{
        server = $server
        token = $token
        launcher = $launcherConfig
    }
    Write-AegisJsonFile -Path $ConfigPath -Value $config
    Write-Host "      [-] 连接模式: $Mode"
    Write-Host "      [-] API 地址: $server"
    Write-Host "      [-] 配置已保存: $ConfigPath" -ForegroundColor Green

    return Get-AegisConnectionConfig -ConfigPath $ConfigPath
}

# =====================================================================
# 探活与进程记录
# =====================================================================
function Write-AegisConnectionStage([string]$Name, [string]$Message) {
    Write-Host ""
    Write-Host ">>> [$Name] $Message" -ForegroundColor Cyan
}

function Invoke-AegisHealthProbe([string]$Server, [int]$TimeoutSeconds = 3) {
    try {
        $health = Invoke-RestMethod `
            -Uri "$($Server.TrimEnd('/'))/health" `
            -Method Get `
            -TimeoutSec $TimeoutSeconds
        if ($health.service -ne "Aegis-LoRA API") {
            return [pscustomobject]@{
                Success = $false
                Health = $null
                Error = "目标返回的不是 Aegis-LoRA API。"
            }
        }
        return [pscustomobject]@{ Success = $true; Health = $health; Error = "" }
    }
    catch {
        return [pscustomobject]@{
            Success = $false
            Health = $null
            Error = $_.Exception.Message
        }
    }
}

function Wait-AegisApi(
    [string]$Server,
    [int]$TimeoutSeconds,
    [Diagnostics.Process]$Process = $null
) {
    $deadline = [DateTime]::UtcNow.AddSeconds($TimeoutSeconds)
    $lastError = "尚未收到健康检查响应。"
    do {
        if ($null -ne $Process) {
            $Process.Refresh()
            if ($Process.HasExited) {
                throw "后台进程已退出，退出码: $($Process.ExitCode)。"
            }
        }

        $probe = Invoke-AegisHealthProbe -Server $Server
        if ($probe.Success) {
            return $probe.Health
        }
        $lastError = $probe.Error
        Start-Sleep -Milliseconds 500
    } while ([DateTime]::UtcNow -lt $deadline)

    throw "API 在 $TimeoutSeconds 秒内未就绪: $lastError"
}

function Confirm-AegisApi(
    [string]$Server,
    [string]$Token,
    [int]$TimeoutSeconds,
    [Diagnostics.Process]$Process = $null
) {
    $health = Wait-AegisApi -Server $Server -TimeoutSeconds $TimeoutSeconds -Process $Process
    try {
        $headers = @{ Authorization = "Bearer $Token" }
        $me = Invoke-RestMethod `
            -Uri "$($Server.TrimEnd('/'))/v1/me" `
            -Method Get `
            -Headers $headers `
            -TimeoutSec 10
        if (-not $me.authenticated -or $me.service -ne "Aegis-LoRA API") {
            throw "登录验证响应无效。"
        }
    }
    catch {
        throw "API Token 验证失败: $($_.Exception.Message)"
    }

    if ($health.status -eq "ready") {
        Write-Host "      [-] API 状态: 就绪" -ForegroundColor Green
    }
    else {
        Write-Host "      [警告] API 已连接，但部分模型或算法资源未就绪。" -ForegroundColor Yellow
    }
    return $health
}

function Test-AegisLoopbackPort([int]$Port) {
    $client = [Net.Sockets.TcpClient]::new()
    try {
        $client.Connect("127.0.0.1", $Port)
        return $true
    }
    catch {
        return $false
    }
    finally {
        $client.Dispose()
    }
}

function Get-AegisConnectionStatePath([string]$ProjectRoot, [string]$Name) {
    $stateRoot = Join-Path $ProjectRoot ".cache\connections"
    if (-not (Test-Path -LiteralPath $stateRoot -PathType Container)) {
        $null = New-Item -ItemType Directory -Path $stateRoot -Force
    }
    return Join-Path $stateRoot "$Name.json"
}

function Write-AegisProcessRecord(
    [string]$ProjectRoot,
    [string]$Name,
    [Diagnostics.Process]$Process,
    [string]$Server
) {
    $record = [ordered]@{
        pid = $Process.Id
        process_name = $Process.ProcessName
        start_time_ticks = $Process.StartTime.ToUniversalTime().Ticks
        server = $Server
        created_at = [DateTime]::UtcNow.ToString("o")
    }
    Write-AegisJsonFile `
        -Path (Get-AegisConnectionStatePath -ProjectRoot $ProjectRoot -Name $Name) `
        -Value $record
}

function Get-AegisRecordedProcess([string]$ProjectRoot, [string]$Name) {
    $recordPath = Get-AegisConnectionStatePath -ProjectRoot $ProjectRoot -Name $Name
    if (-not (Test-Path -LiteralPath $recordPath -PathType Leaf)) {
        return $null
    }

    try {
        $record = Get-Content -Raw -LiteralPath $recordPath -Encoding UTF8 | ConvertFrom-Json
        $process = Get-Process -Id ([int]$record.pid) -ErrorAction SilentlyContinue
        if ($null -eq $process) {
            Remove-Item -LiteralPath $recordPath -Force -ErrorAction SilentlyContinue
            return $null
        }

        $sameName = $process.ProcessName -eq [string]$record.process_name
        $sameStart = $process.StartTime.ToUniversalTime().Ticks -eq [Int64]$record.start_time_ticks
        if (-not $sameName -or -not $sameStart) {
            Write-Host "      [警告] $Name 的进程记录已失效，不会结束当前 PID。" -ForegroundColor Yellow
            Remove-Item -LiteralPath $recordPath -Force -ErrorAction SilentlyContinue
            return $null
        }
        return [pscustomobject]@{ Process = $process; RecordPath = $recordPath; Record = $record }
    }
    catch {
        Write-Host "      [警告] 无法读取 $Name 进程记录: $($_.Exception.Message)" -ForegroundColor Yellow
        Remove-Item -LiteralPath $recordPath -Force -ErrorAction SilentlyContinue
        return $null
    }
}

function Stop-AegisProcessInstance([Diagnostics.Process]$Process) {
    if ($null -eq $Process) {
        return
    }
    try {
        $Process.Refresh()
        if (-not $Process.HasExited) {
            Stop-Process -Id $Process.Id -Force -ErrorAction Stop
            $null = $Process.WaitForExit(5000)
        }
    }
    catch {
        Write-Host "      [警告] 进程 $($Process.Id) 回收失败: $($_.Exception.Message)" -ForegroundColor Yellow
    }
}

function Stop-AegisRecordedProcess([string]$ProjectRoot, [string]$Name) {
    $recorded = Get-AegisRecordedProcess -ProjectRoot $ProjectRoot -Name $Name
    if ($null -eq $recorded) {
        return $false
    }

    Stop-AegisProcessInstance -Process $recorded.Process
    Remove-Item -LiteralPath $recorded.RecordPath -Force -ErrorAction SilentlyContinue
    Write-Host "      [-] 已停止 $Name。"
    return $true
}

# =====================================================================
# 本地 API 与 SSH 隧道
# =====================================================================
function Start-AegisLocalApi(
    [string]$PythonPath,
    [string]$ProjectRoot,
    [string]$Server,
    [string]$Token,
    [int]$Port
) {
    $logRoot = Join-Path $ProjectRoot ".cache\connections"
    if (-not (Test-Path -LiteralPath $logRoot -PathType Container)) {
        $null = New-Item -ItemType Directory -Path $logRoot -Force
    }

    $tokenWasSet = Test-Path Env:AEGIS_API_TOKEN
    $previousToken = $env:AEGIS_API_TOKEN
    try {
        $env:AEGIS_API_TOKEN = $Token
        $process = Start-Process `
            -FilePath $PythonPath `
            -ArgumentList @(
                "-B", "-m", "uvicorn", "utils.api_server:app",
                "--host", "127.0.0.1", "--port", [string]$Port
            ) `
            -WorkingDirectory $ProjectRoot `
            -RedirectStandardOutput (Join-Path $logRoot "local-api.stdout.log") `
            -RedirectStandardError (Join-Path $logRoot "local-api.stderr.log") `
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

    Write-AegisProcessRecord -ProjectRoot $ProjectRoot -Name "local-api" -Process $process -Server $Server
    return $process
}

function Get-AegisSshArguments($Ssh, [switch]$Background) {
    $arguments = @("-T", "-o", "ConnectTimeout=10")
    if ($Background) {
        $forward = "127.0.0.1:{0}:{1}:{2}" -f $Ssh.LocalPort, $Ssh.RemoteApiHost, $Ssh.RemoteApiPort
        $arguments = @(
            "-N", "-T",
            "-o", "BatchMode=yes",
            "-o", "StrictHostKeyChecking=yes",
            "-o", "ExitOnForwardFailure=yes",
            "-o", "ServerAliveInterval=30",
            "-o", "ServerAliveCountMax=3",
            "-L", $forward
        )
    }
    $arguments += @("-p", [string]$Ssh.Port)
    if ($Ssh.IdentityFile) {
        $arguments += @("-i", $Ssh.IdentityFile)
    }
    return $arguments
}

function ConvertTo-AegisProcessArgument([string]$Value) {
    if (-not $Value) {
        return '""'
    }
    if ($Value -notmatch '[\s"]') {
        return $Value
    }
    if ($Value.Contains('"')) {
        throw "进程参数不能包含双引号: $Value"
    }
    return '"' + $Value + '"'
}

function Start-AegisSshTunnel(
    [string]$SshPath,
    $Ssh,
    [string]$ProjectRoot,
    [string]$Server
) {
    $logRoot = Join-Path $ProjectRoot ".cache\connections"
    if (-not (Test-Path -LiteralPath $logRoot -PathType Container)) {
        $null = New-Item -ItemType Directory -Path $logRoot -Force
    }

    $arguments = @(Get-AegisSshArguments -Ssh $Ssh -Background)
    $arguments += $Ssh.Target
    $commandLine = ($arguments | ForEach-Object { ConvertTo-AegisProcessArgument ([string]$_) }) -join " "
    $process = Start-Process `
        -FilePath $SshPath `
        -ArgumentList $commandLine `
        -WorkingDirectory $ProjectRoot `
        -RedirectStandardOutput (Join-Path $logRoot "ssh-tunnel.stdout.log") `
        -RedirectStandardError (Join-Path $logRoot "ssh-tunnel.stderr.log") `
        -WindowStyle Hidden `
        -PassThru

    Write-AegisProcessRecord -ProjectRoot $ProjectRoot -Name "ssh-tunnel" -Process $process -Server $Server
    return $process
}

# =====================================================================
# 对外连接生命周期
# =====================================================================
function Open-AegisConnection {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)]$Config,
        [Parameter(Mandatory = $true)][string]$PythonPath,
        [Parameter(Mandatory = $true)][string]$ProjectRoot
    )

    $connection = [pscustomobject]@{
        Mode = $Config.Mode
        Server = $Config.Server
        ApiProcess = $null
        TunnelProcess = $null
        ApiStartedByLauncher = $false
        TunnelStartedByLauncher = $false
    }

    if (-not $Config.IsConfigured) {
        Write-Host "      [警告] 尚未配置 API 连接；可执行 start-cli.bat -Action configure。" -ForegroundColor Yellow
        return $connection
    }

    if ($Config.Mode -eq "direct") {
        Write-AegisConnectionStage "验证服务" "正在验证直连 API..."
        $null = Confirm-AegisApi -Server $Config.Server -Token $Config.Token -TimeoutSeconds 15
        Write-Host "      [-] 连接模式: direct"
        Write-Host "      [-] API 地址: $($Config.Server)"
        return $connection
    }

    $initialProbe = Invoke-AegisHealthProbe -Server $Config.Server
    if ($initialProbe.Success) {
        Write-AegisConnectionStage "验证服务" "正在复用已有 API 连接..."
        try {
            $null = Confirm-AegisApi -Server $Config.Server -Token $Config.Token -TimeoutSeconds 10
            Write-Host "      [-] 连接模式: $($Config.Mode)"
            Write-Host "      [-] API 地址: $($Config.Server)"
            return $connection
        }
        catch {
            if ($Config.Mode -ne "local") {
                throw
            }
            $recordedApi = Get-AegisRecordedProcess -ProjectRoot $ProjectRoot -Name "local-api"
            if ($null -eq $recordedApi) {
                throw
            }
            Write-Host "      [警告] 本地 API 配置已变化，正在重启托管进程。" -ForegroundColor Yellow
            $null = Stop-AegisRecordedProcess -ProjectRoot $ProjectRoot -Name "local-api"
        }
    }

    if ($Config.Mode -eq "local") {
        Write-AegisConnectionStage "准备服务" "正在启动本地 API..."

        $recordedApi = Get-AegisRecordedProcess -ProjectRoot $ProjectRoot -Name "local-api"
        if ($null -ne $recordedApi) {
            try {
                $null = Confirm-AegisApi `
                    -Server $Config.Server `
                    -Token $Config.Token `
                    -TimeoutSeconds 30 `
                    -Process $recordedApi.Process
                Write-Host "      [-] 已复用启动中的本地 API。"
                return $connection
            }
            catch {
                $null = Stop-AegisRecordedProcess -ProjectRoot $ProjectRoot -Name "local-api"
            }
        }

        if (Test-AegisLoopbackPort -Port $Config.ApiPort) {
            throw "本地端口 $($Config.ApiPort) 已被其他程序占用。"
        }

        try {
            $connection.ApiProcess = Start-AegisLocalApi `
                -PythonPath $PythonPath `
                -ProjectRoot $ProjectRoot `
                -Server $Config.Server `
                -Token $Config.Token `
                -Port $Config.ApiPort
            $connection.ApiStartedByLauncher = $true

            Write-AegisConnectionStage "验证服务" "正在等待本地 API 就绪..."
            $null = Confirm-AegisApi `
                -Server $Config.Server `
                -Token $Config.Token `
                -TimeoutSeconds 90 `
                -Process $connection.ApiProcess
            Write-Host "      [-] 本地 API 将保持运行；可使用 -Action stop 停止。"
            return $connection
        }
        catch {
            Close-AegisConnection -Connection $connection -ProjectRoot $ProjectRoot -StopApi
            throw
        }
    }

    Write-AegisConnectionStage "建立连接" "正在连接 SSH 并创建端口转发..."
    if (Test-AegisLoopbackPort -Port $Config.Ssh.LocalPort) {
        throw "本地端口 $($Config.Ssh.LocalPort) 已被其他程序占用。"
    }

    $sshCommand = Get-Command ssh.exe -ErrorAction SilentlyContinue
    if (-not $sshCommand) {
        throw "未找到 Windows OpenSSH 客户端 ssh.exe。"
    }

    # 前台预检允许首次连接时确认主机指纹；后台隧道随后强制使用 known_hosts。
    $sshArguments = @(Get-AegisSshArguments -Ssh $Config.Ssh)
    $sshArguments += $Config.Ssh.Target
    $remoteCommand = if ($Config.Ssh.StartCommand) { $Config.Ssh.StartCommand } else { "exit 0" }
    $sshArguments += $remoteCommand
    & $sshCommand.Source @sshArguments
    if ($LASTEXITCODE -ne 0) {
        throw "SSH 预检或远端 API 启动命令失败，退出码: $LASTEXITCODE。"
    }

    try {
        $connection.TunnelProcess = Start-AegisSshTunnel `
            -SshPath $sshCommand.Source `
            -Ssh $Config.Ssh `
            -ProjectRoot $ProjectRoot `
            -Server $Config.Server
        $connection.TunnelStartedByLauncher = $true

        Write-AegisConnectionStage "验证服务" "正在等待 SSH 隧道后的 API 就绪..."
        $null = Confirm-AegisApi `
            -Server $Config.Server `
            -Token $Config.Token `
            -TimeoutSeconds 60 `
            -Process $connection.TunnelProcess
        Write-Host "      [-] SSH 目标: $($Config.Ssh.Target)"
        Write-Host "      [-] API 地址: $($Config.Server)"
        return $connection
    }
    catch {
        Close-AegisConnection -Connection $connection -ProjectRoot $ProjectRoot
        throw
    }
}

function Close-AegisConnection {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)]$Connection,
        [Parameter(Mandatory = $true)][string]$ProjectRoot,
        [switch]$StopApi
    )

    if ($Connection.TunnelStartedByLauncher -and $null -ne $Connection.TunnelProcess) {
        Write-AegisConnectionStage "回收连接" "正在关闭 SSH 隧道..."
        Stop-AegisProcessInstance -Process $Connection.TunnelProcess
        $recordPath = Get-AegisConnectionStatePath -ProjectRoot $ProjectRoot -Name "ssh-tunnel"
        Remove-Item -LiteralPath $recordPath -Force -ErrorAction SilentlyContinue
    }

    if ($StopApi -and $Connection.ApiStartedByLauncher -and $null -ne $Connection.ApiProcess) {
        Stop-AegisProcessInstance -Process $Connection.ApiProcess
        $recordPath = Get-AegisConnectionStatePath -ProjectRoot $ProjectRoot -Name "local-api"
        Remove-Item -LiteralPath $recordPath -Force -ErrorAction SilentlyContinue
    }
}

function Stop-AegisManagedConnections {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)][string]$ProjectRoot
    )

    $stoppedTunnel = Stop-AegisRecordedProcess -ProjectRoot $ProjectRoot -Name "ssh-tunnel"
    $stoppedApi = Stop-AegisRecordedProcess -ProjectRoot $ProjectRoot -Name "local-api"
    if (-not $stoppedTunnel -and -not $stoppedApi) {
        Write-Host "      [-] 没有由启动器管理的活动进程。"
    }
}

function Show-AegisConnectionStatus {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)]$Config,
        [Parameter(Mandatory = $true)][string]$ProjectRoot
    )

    Write-Host "      [-] 连接模式: $($Config.Mode)"
    if (-not $Config.IsConfigured) {
        Write-Host "      [警告] 尚未完成连接配置。" -ForegroundColor Yellow
        return $false
    }
    Write-Host "      [-] API 地址: $($Config.Server)"

    $apiRecord = Get-AegisRecordedProcess -ProjectRoot $ProjectRoot -Name "local-api"
    $tunnelRecord = Get-AegisRecordedProcess -ProjectRoot $ProjectRoot -Name "ssh-tunnel"
    if ($null -ne $apiRecord) {
        Write-Host "      [-] 本地 API 进程: $($apiRecord.Process.Id)"
    }
    if ($null -ne $tunnelRecord) {
        Write-Host "      [-] SSH 隧道进程: $($tunnelRecord.Process.Id)"
    }

    try {
        $null = Confirm-AegisApi -Server $Config.Server -Token $Config.Token -TimeoutSeconds 3
        return $true
    }
    catch {
        Write-Host "      [错误] API 当前不可用: $($_.Exception.Message)" -ForegroundColor Red
        return $false
    }
}
