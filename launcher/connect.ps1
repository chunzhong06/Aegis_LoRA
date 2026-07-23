# Aegis-LoRA - CLI 连接模块
# 本模块负责生成本次连接参数、验证 API，并管理 local / ssh 子进程的完整生命周期。
# 持久化范围仅限 API 默认值；SSH 目标、密码、主机指纹和运行状态都属于一次性会话。

# =====================================================================
# 交互输入
# =====================================================================
# 读取普通文本参数；Default 提供可接受的建议值，Required 标记本次连接必需的字段。
function Read-AegisValue([string]$Label, [string]$Default = "", [switch]$Required) {
    # prompt 只负责显示默认值，最终返回值始终经过 Trim。
    $prompt = if ($Default) { "$Label [$Default]" } else { $Label }
    $value = (Read-Host $prompt).Trim()
    if (-not $value) { $value = $Default }
    if ($Required -and -not $value) { throw "$Label 不能为空。" }
    return $value
}

# 对所有历史默认值采用拒绝优先策略，只有明确输入 y / yes 才会复用。
function Read-AegisConfirmation([string]$Message) {
    return (Read-Host "$Message [y/N]").Trim().ToLowerInvariant() -in @("y", "yes")
}

# Token 允许保存在统一配置中；交互输入使用 SecureString，避免明文回显。
function Read-AegisToken([string]$Label, [string]$SavedToken = "") {
    if ($SavedToken) {
        # suffix 只用于帮助用户识别已保存 Token，不在终端显示完整内容。
        $suffix = if ($SavedToken.Length -gt 4) {
            $SavedToken.Substring($SavedToken.Length - 4)
        }
        else {
            $SavedToken
        }
        if (Read-AegisConfirmation "使用已保存的 $Label（尾号 $suffix）") {
            return $SavedToken
        }
    }

    # SecureString 需要短暂转换为普通字符串才能传给 API；BSTR 在 finally 中立即清零。
    $secureToken = Read-Host $Label -AsSecureString
    $pointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secureToken)
    try {
        $token = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($pointer)
    }
    finally {
        [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($pointer)
    }
    if (-not $token) { throw "$Label 不能为空。" }
    return $token
}

# 请求系统分配一个临时回环端口，避免 SSH 隧道复用固定端口或历史状态。
function Get-AegisFreeTcpPort {
    # 绑定端口 0 后，LocalEndpoint 会返回操作系统实际分配的端口。
    $listener = [Net.Sockets.TcpListener]::new([Net.IPAddress]::Loopback, 0)
    try {
        $listener.Start()
        return ([Net.IPEndPoint]$listener.LocalEndpoint).Port
    }
    finally {
        $listener.Stop()
    }
}

# =====================================================================
# API 就绪检查
# =====================================================================
# 先通过公开 health 确认服务身份，再向受保护接口发送 Token。
# local / ssh 额外传入 Process，以便在等待期间立即发现子进程提前退出。
function Wait-AegisApi {
    param(
        [Parameter(Mandatory = $true)][string]$Server,
        [Parameter(Mandatory = $true)][string]$Token,
        [int]$TimeoutSeconds = 15,
        [Diagnostics.Process]$Process
    )

    # deadline 限制整个等待窗口；lastError 保存最后一次连接错误供超时诊断。
    $deadline = [DateTime]::UtcNow.AddSeconds($TimeoutSeconds)
    $lastError = "API 未响应。"
    do {
        # 进程先于 API 就绪检查结束时直接失败，不继续无意义地等待超时。
        if ($null -ne $Process) {
            $Process.Refresh()
            if ($Process.HasExited) {
                throw "连接进程已退出，退出码: $($Process.ExitCode)。"
            }
        }

        # health 不携带 Token，只验证目标确实是 Aegis-LoRA API。
        try {
            $health = Invoke-RestMethod -Uri "$Server/health" -TimeoutSec 3 -ErrorAction Stop
            if ($health.service -ne "Aegis-LoRA API") {
                throw "目标地址不是 Aegis-LoRA API。"
            }
        }
        catch {
            $lastError = $_.Exception.Message
            if ([DateTime]::UtcNow -lt $deadline) { Start-Sleep -Seconds 1 }
            continue
        }

        # /v1/me 只验证当前 Token；认证失败不重试，避免把错误凭据误判为启动缓慢。
        try {
            $null = Invoke-RestMethod -Uri "$Server/v1/me" -Headers @{
                Authorization = "Bearer $Token"
            } -TimeoutSec 3 -ErrorAction Stop
            return
        }
        catch {
            throw [UnauthorizedAccessException]::new("API 身份验证失败: $($_.Exception.Message)")
        }
    } while ([DateTime]::UtcNow -lt $deadline)

    throw "API 连接失败: $lastError"
}

# =====================================================================
# 当前会话的进程边界
# =====================================================================
# local API 与 ssh.exe 都加入 KILL_ON_JOB_CLOSE Job Object。
# 正常退出由 finally 主动回收；窗口被直接关闭时由 Windows 随 Job 句柄销毁子进程。
# C# 仅封装创建 Job、绑定进程和关闭句柄三个 Win32 操作，不保存任何连接状态。
if (-not ([System.Management.Automation.PSTypeName]"AegisLauncher.JobObject").Type) {
    Add-Type -TypeDefinition @"
using System;
using System.ComponentModel;
using System.Runtime.InteropServices;

namespace AegisLauncher {
    public static class JobObject {
        // ExtendedLimitInformationClass selects the structure that carries KillOnJobClose.
        private const int ExtendedLimitInformationClass = 9;
        private const uint KillOnJobClose = 0x00002000;

        // These structures mirror JOBOBJECT_EXTENDED_LIMIT_INFORMATION in Win32.
        [StructLayout(LayoutKind.Sequential)]
        private struct BasicLimits {
            public long PerProcessUserTimeLimit, PerJobUserTimeLimit;
            public uint LimitFlags;
            public UIntPtr MinimumWorkingSetSize, MaximumWorkingSetSize;
            public uint ActiveProcessLimit;
            public UIntPtr Affinity;
            public uint PriorityClass, SchedulingClass;
        }

        // I/O counters are required for native layout even though the launcher does not read them.
        [StructLayout(LayoutKind.Sequential)]
        private struct IoCounters {
            public ulong ReadOperationCount, WriteOperationCount, OtherOperationCount;
            public ulong ReadTransferCount, WriteTransferCount, OtherTransferCount;
        }

        // Only BasicLimitInformation.LimitFlags is assigned; remaining limits keep system defaults.
        [StructLayout(LayoutKind.Sequential)]
        private struct ExtendedLimits {
            public BasicLimits BasicLimitInformation;
            public IoCounters IoInfo;
            public UIntPtr ProcessMemoryLimit, JobMemoryLimit;
            public UIntPtr PeakProcessMemoryUsed, PeakJobMemoryUsed;
        }

        // The wrapper owns each returned Job handle and closes it exactly once.
        [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)] private static extern IntPtr CreateJobObject(IntPtr attributes, string name);
        [DllImport("kernel32.dll", SetLastError = true)] private static extern bool SetInformationJobObject(IntPtr job, int kind, IntPtr info, uint size);
        [DllImport("kernel32.dll", SetLastError = true)] private static extern bool AssignProcessToJobObject(IntPtr job, IntPtr process);
        [DllImport("kernel32.dll", SetLastError = true)] private static extern bool CloseHandle(IntPtr handle);

        public static IntPtr CreateKillOnClose() {
            // Configure a new unnamed Job so closing its last handle terminates all assigned processes.
            IntPtr job = CreateJobObject(IntPtr.Zero, null);
            if (job == IntPtr.Zero) { throw new Win32Exception(Marshal.GetLastWin32Error()); }
            ExtendedLimits limits = new ExtendedLimits();
            limits.BasicLimitInformation.LimitFlags = KillOnJobClose;
            int size = Marshal.SizeOf(typeof(ExtendedLimits));

            // SetInformationJobObject expects unmanaged memory matching the native structure layout.
            IntPtr buffer = Marshal.AllocHGlobal(size);
            try {
                Marshal.StructureToPtr(limits, buffer, false);
                if (!SetInformationJobObject(job, ExtendedLimitInformationClass, buffer, (uint)size))
                    { throw new Win32Exception(Marshal.GetLastWin32Error()); }
                return job;
            }
            catch {
                CloseHandle(job);
                throw;
            }
            finally { Marshal.FreeHGlobal(buffer); }
        }

        public static void Assign(IntPtr job, IntPtr process)
            { if (!AssignProcessToJobObject(job, process)) { throw new Win32Exception(Marshal.GetLastWin32Error()); } }
        public static void Close(IntPtr job) { if (job != IntPtr.Zero) { CloseHandle(job); } }
    }
}
"@
}

# =====================================================================
# 本次连接参数
# =====================================================================
# 读取可复用默认值并生成本次连接对象。
# StateRoot\config.json 是唯一持久文件，不保存模式、SSH 目标、SSH 密码、PID 或会话状态。
function New-AegisConnectionConfig {
    param(
        [Parameter(Mandatory = $true)][ValidateSet("direct", "local", "ssh")][string]$Mode,
        [Parameter(Mandatory = $true)][string]$StateRoot
    )

    # defaults 是配置文件的内存表示；缺少文件时使用这些安全初值。
    $configPath = Join-Path $StateRoot "config.json"
    $defaults = [ordered]@{
        Direct = [ordered]@{ Server = ""; Token = "" }
        Local = [ordered]@{ ApiPort = 8000; Token = "" }
        Ssh = [ordered]@{ Token = "" }
    }

    # 当前版本只接受结构明确的 version 2，不兼容或迁移旧版配置。
    if (Test-Path -LiteralPath $configPath -PathType Leaf) {
        try {
            $saved = Get-Content -Raw -LiteralPath $configPath -Encoding UTF8 | ConvertFrom-Json
            if ([int]$saved.version -ne 2) { throw "配置版本必须为 2。" }
            $defaults.Direct.Server = ([string]$saved.defaults.direct.server).Trim().TrimEnd("/")
            $defaults.Direct.Token = [string]$saved.defaults.direct.token
            $defaults.Local.ApiPort = [int]$saved.defaults.local.api_port
            $defaults.Local.Token = [string]$saved.defaults.local.token
            $defaults.Ssh.Token = [string]$saved.defaults.ssh.token
        }
        catch {
            throw "CLI 配置无效，请删除 $configPath 后重新启动：$($_.Exception.Message)"
        }
    }

    # -----------------------------------------------------------------
    # 1. direct：连接用户提供的现有 API
    # -----------------------------------------------------------------
    if ($Mode -eq "direct") {
        # 服务器地址和 Token 必须成对复用，否则重新询问并覆盖 direct 默认值。
        $useSaved = $defaults.Direct.Server -and $defaults.Direct.Token -and
            (Read-AegisConfirmation "使用已保存的直连 API $($defaults.Direct.Server)")
        if ($useSaved) {
            $server = $defaults.Direct.Server
            $token = $defaults.Direct.Token
        }
        else {
            # server 是 CLI 最终使用的 API 根地址，统一去掉尾部斜杠。
            $server = (Read-AegisValue -Label "API 地址" -Required).TrimEnd("/")
            if ($server -notmatch '^https?://') {
                throw "API 地址必须以 http:// 或 https:// 开头。"
            }
            $token = Read-AegisToken -Label "API Token"
            $defaults.Direct.Server = $server
            $defaults.Direct.Token = $token
        }
        # connection 只描述当前启动，不包含进程或磁盘状态。
        $connection = [pscustomobject]@{
            Mode = "direct"; Server = $server; Token = $token; ApiPort = 0; Ssh = $null
        }
    }
    elseif ($Mode -eq "local") {
        # -----------------------------------------------------------------
        # 2. local：在本机启动仅监听回环地址的 API
        # -----------------------------------------------------------------
        # apiPort 同时用于 uvicorn 监听地址和 CLI 的 Server 地址。
        $portText = Read-AegisValue -Label "本地 API 端口" -Default ([string]$defaults.Local.ApiPort)
        $apiPort = 0
        if (-not [int]::TryParse($portText, [ref]$apiPort) -or $apiPort -lt 1 -or $apiPort -gt 65535) {
            throw "本地 API 端口必须位于 1 到 65535 之间。"
        }
        $token = Read-AegisToken -Label "本地 API Token" -SavedToken $defaults.Local.Token
        $defaults.Local.ApiPort = $apiPort
        $defaults.Local.Token = $token
        # local 不需要 SSH 参数；进程由 Open-AegisConnection 在环境就绪后创建。
        $connection = [pscustomobject]@{
            Mode = "local"; Server = "http://127.0.0.1:$apiPort"; Token = $token
            ApiPort = $apiPort; Ssh = $null
        }
    }
    else {
        # -----------------------------------------------------------------
        # 3. ssh：解析目标并构造一次性本地转发参数
        # -----------------------------------------------------------------
        # sshCommand 只接受可审计的 user@host 与可选端口，不执行用户输入的任意参数。
        $sshCommand = Read-AegisValue -Label "SSH 连接命令（例如 ssh -p 31544 root@host）" -Required
        $match = [Regex]::Match(
            $sshCommand,
            '^(?:ssh(?:\.exe)?\s+)?(?:-p\s+([0-9]+)\s+)?([^\s@]+@[^\s@]+)$',
            [Text.RegularExpressions.RegexOptions]::IgnoreCase
        )
        if (-not $match.Success) { throw "SSH 连接命令格式应为 ssh -p 端口 user@host。" }

        # sshPort 是 SSH 服务端口；remoteHost / remotePort 是服务器侧 API 地址。
        $sshPort = 22
        if ($match.Groups[1].Success) {
            $sshPort = [int]$match.Groups[1].Value
            if ($sshPort -lt 1 -or $sshPort -gt 65535) {
                throw "SSH 端口必须位于 1 到 65535 之间。"
            }
        }
        $remoteHost = Read-AegisValue -Label "远端 API 主机" -Default "127.0.0.1"
        if ($remoteHost -match '\s') { throw "远端 API 主机不能包含空白字符。" }
        $remotePortText = Read-AegisValue -Label "远端 API 端口" -Default "8000"
        $remotePort = 0
        if (-not [int]::TryParse($remotePortText, [ref]$remotePort) -or $remotePort -lt 1 -or $remotePort -gt 65535) {
            throw "远端 API 端口必须位于 1 到 65535 之间。"
        }

        # API Token 与 SSH 密码相互独立；SSH 密码始终由 ssh.exe 自己读取。
        $token = Read-AegisToken -Label "远端 Aegis API Token（不是 SSH 密码）" `
            -SavedToken $defaults.Ssh.Token
        $defaults.Ssh.Token = $token

        # localPort 只属于本次隧道，CLI 始终通过 127.0.0.1 访问远端 API。
        $localPort = Get-AegisFreeTcpPort
        $connection = [pscustomobject]@{
            Mode = "ssh"; Server = "http://127.0.0.1:$localPort"; Token = $token; ApiPort = 0
            Ssh = [pscustomobject]@{
                Target = $match.Groups[2].Value
                Port = $sshPort
                LocalPort = $localPort
                RemoteApiHost = $remoteHost
                RemoteApiPort = $remotePort
            }
        }
    }

    # -----------------------------------------------------------------
    # 4. 保存允许复用的默认值
    # -----------------------------------------------------------------
    # 三种模式共用一次原子替换；tempPath 防止中途中断留下半个 JSON。
    $null = [IO.Directory]::CreateDirectory($StateRoot)
    $tempPath = Join-Path $StateRoot ("config.{0}.{1}.tmp" -f $PID, [Guid]::NewGuid().ToString("N"))

    # document 是唯一落盘结构，明确排除 connection 中的 SSH 和运行时字段。
    $document = [ordered]@{
        version = 2
        defaults = [ordered]@{
            direct = [ordered]@{
                server = [string]$defaults.Direct.Server
                token = [string]$defaults.Direct.Token
            }
            local = [ordered]@{
                api_port = [int]$defaults.Local.ApiPort
                token = [string]$defaults.Local.Token
            }
            ssh = [ordered]@{ token = [string]$defaults.Ssh.Token }
        }
    }
    try {
        # 使用无 BOM UTF-8 写入临时文件，成功后一次性替换正式配置。
        $json = $document | ConvertTo-Json -Depth 5
        [IO.File]::WriteAllText($tempPath, $json + [Environment]::NewLine, [Text.UTF8Encoding]::new($false))
        Move-Item -LiteralPath $tempPath -Destination $configPath -Force
    }
    finally {
        # Move-Item 成功后临时文件已不存在；失败时这里负责清理残留。
        Remove-Item -LiteralPath $tempPath -Force -ErrorAction SilentlyContinue
    }

    # 返回值仅供 start.ps1 本次启动使用，不作为历史连接状态保存。
    return $connection
}

# =====================================================================
# 建立与关闭当前连接
# =====================================================================
# 根据连接对象建立可用 API：direct 只验证地址，local / ssh 创建受托管子进程。
# local / ssh 每次使用新的 SessionRoot，日志和 SSH 主机指纹不会被后续会话复用。
function Open-AegisConnection {
    param(
        [Parameter(Mandatory = $true)]$Config,
        [Parameter(Mandatory = $true)][string]$StateRoot,
        [Parameter(Mandatory = $true)][string]$PythonPath,
        [Parameter(Mandatory = $true)][string]$ProjectRoot
    )

    # -----------------------------------------------------------------
    # 1. direct：验证现有 API，不创建任何本地资源
    # -----------------------------------------------------------------
    if ($Config.Mode -eq "direct") {
        Write-Host ""
        Write-Host ">>> [验证服务] 正在验证直连 API..." -ForegroundColor Cyan
        Wait-AegisApi -Server $Config.Server -Token $Config.Token
        Write-Host "      [-] API 地址: $($Config.Server)"
        return $null
    }

    # -----------------------------------------------------------------
    # 2. local / ssh：创建本次会话的资源边界
    # -----------------------------------------------------------------
    # sessionId 保证并发启动互不覆盖；sessionRoot 只保存本次日志和 known_hosts。
    $sessionId = "{0}-{1}" -f [DateTime]::Now.ToString("yyyyMMdd-HHmmss"), [Guid]::NewGuid().ToString("N").Substring(0, 8)
    $sessionRoot = Join-Path (Join-Path $StateRoot "sessions") $sessionId
    $null = [IO.Directory]::CreateDirectory($sessionRoot)

    # process 是本次 API 或隧道进程；jobHandle 是窗口异常关闭时的系统级兜底。
    $process = $null
    $jobHandle = [IntPtr]::Zero

    try {
        $jobHandle = [AegisLauncher.JobObject]::CreateKillOnClose()
        if ($Config.Mode -eq "local") {
            # -------------------------------------------------------------
            # 2.1 local：启动隐藏的本地 uvicorn 进程
            # -------------------------------------------------------------
            Write-Host ""
            Write-Host ">>> [启动服务] 正在启动本次会话的本地 API..." -ForegroundColor Cyan

            # Token 只在 Start-Process 继承环境的瞬间注入，随后恢复调用者原值。
            $tokenWasSet = Test-Path Env:AEGIS_API_TOKEN
            $previousToken = $env:AEGIS_API_TOKEN
            try {
                $env:AEGIS_API_TOKEN = $Config.Token

                # stdout / stderr 进入 SessionRoot，正常关闭时随会话目录删除。
                $process = Start-Process -FilePath $PythonPath -ArgumentList @(
                    "-B", "-m", "uvicorn", "utils.api_server:app",
                    "--host", "127.0.0.1", "--port", [string]$Config.ApiPort
                ) -WorkingDirectory $ProjectRoot `
                    -RedirectStandardOutput (Join-Path $sessionRoot "stdout.log") `
                    -RedirectStandardError (Join-Path $sessionRoot "stderr.log") `
                    -WindowStyle Hidden -PassThru
            }
            finally {
                if ($tokenWasSet) { $env:AEGIS_API_TOKEN = $previousToken }
                else { Remove-Item Env:AEGIS_API_TOKEN -ErrorAction SilentlyContinue }
            }
        }
        else {
            # -------------------------------------------------------------
            # 2.2 ssh：启动不读取用户 SSH 配置的一次性端口转发
            # -------------------------------------------------------------
            # knownHostsPath 只记录本次主机指纹；forward 描述本地端口到远端 API 的映射。
            $ssh = Get-Command ssh.exe -ErrorAction Stop
            $knownHostsPath = Join-Path $sessionRoot "known_hosts"
            $forward = "127.0.0.1:{0}:{1}:{2}" -f $Config.Ssh.LocalPort, $Config.Ssh.RemoteApiHost, $Config.Ssh.RemoteApiPort
            Write-Host ""
            Write-Host ">>> [建立连接] 正在创建本次会话的一次性 SSH 隧道..." -ForegroundColor Cyan
            Write-Host "      请根据 SSH 提示输入服务器密码；密码不会保存。" -ForegroundColor Yellow

            # -F NUL 与独立 known_hosts 切断默认 .ssh；-N -T 表示只建立隧道，不启动远端 shell。
            # 密码由 ssh.exe 直接交互读取，PowerShell 不接触也不保存。
            $process = Start-Process -FilePath $ssh.Source -ArgumentList @(
                "-F", "NUL", "-N", "-T"
                "-o", "ConnectTimeout=15"
                "-o", "StrictHostKeyChecking=accept-new"
                "-o", "UserKnownHostsFile=`"$knownHostsPath`""
                "-o", "GlobalKnownHostsFile=NUL"
                "-o", "ExitOnForwardFailure=yes"
                "-o", "PreferredAuthentications=password,keyboard-interactive"
                "-o", "PubkeyAuthentication=no"
                "-L", $forward
                "-p", [string]$Config.Ssh.Port
                $Config.Ssh.Target
            ) -WorkingDirectory $ProjectRoot -NoNewWindow -PassThru
        }

        # -----------------------------------------------------------------
        # 3. 托管子进程并等待 API 可用
        # -----------------------------------------------------------------
        # 先加入 Job Object，再开始等待；此后任何失败路径都可可靠回收子进程。
        [AegisLauncher.JobObject]::Assign($jobHandle, $process.Handle)

        # 本地模型服务通常启动更快；SSH 为网络连接和密码输入预留更长时间。
        $timeout = if ($Config.Mode -eq "local") { 90 } else { 180 }
        Write-Host ""
        Write-Host ">>> [验证服务] 正在等待 $($Config.Mode) 连接就绪..." -ForegroundColor Cyan
        Wait-AegisApi -Server $Config.Server -Token $Config.Token -TimeoutSeconds $timeout -Process $process
        Write-Host "      [-] API 地址: $($Config.Server)"
        if ($Config.Mode -eq "ssh") {
            Write-Host "      [-] SSH 目标: $($Config.Ssh.Target)"
            Write-Host "      [-] 本地转发端口: $($Config.Ssh.LocalPort)"
        }
        Write-Host "      [-] 当前 CLI 退出时将关闭本次连接。"

        # ManagedConnection 只包含关闭阶段需要的资源句柄，不写入 config.json。
        return [pscustomobject]@{
            Mode = $Config.Mode
            Process = $process
            JobHandle = $jobHandle
            SessionRoot = $sessionRoot
        }
    }
    catch {
        # 创建过程失败时按 Job、进程、目录顺序兜底清理，再保留原始异常。
        if ($jobHandle -ne [IntPtr]::Zero) {
            [AegisLauncher.JobObject]::Close($jobHandle)
        }
        if ($null -ne $process) {
            try {
                $process.Refresh()
                if (-not $process.HasExited) { Stop-Process -Id $process.Id -Force }
            }
            catch { }
        }
        Remove-Item -LiteralPath $sessionRoot -Recurse -Force -ErrorAction SilentlyContinue
        throw
    }
}

# 关闭本次 ManagedConnection；顺序固定为进程、Job 句柄、会话目录。
# 独立 known_hosts 和日志随 SessionRoot 删除，不影响其他会话或用户 .ssh。
function Close-AegisConnection {
    param(
        [Parameter(Mandatory = $true)]$ManagedConnection,
        [Parameter(Mandatory = $true)][string]$StateRoot
    )

    # name 仅用于显示当前回收的是本地 API 还是 SSH 隧道。
    Write-Host ""
    $name = if ($ManagedConnection.Mode -eq "local") { "本地 API" } else { "SSH 隧道" }
    Write-Host ">>> [回收连接] 正在关闭 $name..." -ForegroundColor Cyan
    try {
        $ManagedConnection.Process.Refresh()
        if (-not $ManagedConnection.Process.HasExited) {
            Stop-Process -Id $ManagedConnection.Process.Id -Force -ErrorAction Stop
            $null = $ManagedConnection.Process.WaitForExit(5000)
        }
    }
    catch {
        # 主动停止失败时不立即退出；关闭 Job 句柄仍会触发 KILL_ON_JOB_CLOSE。
        Write-Host "      [警告] 进程回收失败，将由 Job Object 强制关闭。" -ForegroundColor Yellow
    }
    finally {
        if ($ManagedConnection.JobHandle -ne [IntPtr]::Zero) {
            [AegisLauncher.JobObject]::Close($ManagedConnection.JobHandle)
        }
        try { $null = $ManagedConnection.Process.WaitForExit(5000) } catch { }
    }

    # -----------------------------------------------------------------
    # 验证并删除本次会话目录
    # -----------------------------------------------------------------
    # sessionsRoot 与 sessionRoot 都转为带尾分隔符的绝对路径，避免前缀碰撞。
    # 只有 StateRoot\sessions 的后代路径允许进入递归删除。
    $separator = [IO.Path]::DirectorySeparatorChar
    $sessionsRoot = [IO.Path]::GetFullPath((Join-Path $StateRoot "sessions")).TrimEnd($separator) + $separator
    $sessionRoot = [IO.Path]::GetFullPath([string]$ManagedConnection.SessionRoot).TrimEnd($separator) + $separator
    if (-not $sessionRoot.StartsWith($sessionsRoot, [StringComparison]::OrdinalIgnoreCase)) {
        Write-Host "      [警告] 拒绝清理统一会话目录之外的路径。" -ForegroundColor Yellow
        return
    }
    try {
        # 此目录仅包含本次 stdout、stderr 和 SSH known_hosts。
        Remove-Item -LiteralPath $ManagedConnection.SessionRoot -Recurse -Force -ErrorAction Stop
    }
    catch {
        Write-Host "      [警告] 会话目录未能删除: $($_.Exception.Message)" -ForegroundColor Yellow
    }
    Write-Host "      [-] 当前连接已关闭。"
}
