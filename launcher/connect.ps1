# Aegis-LoRA CLI 连接模块：只管理当前启动会话，不读取或接管历史连接。
# start.ps1 负责环境与 CLI 入口；本文件负责参数、API 连接和子进程生命周期。

# =====================================================================
# 交互输入
# =====================================================================
# 普通字段允许提供默认值；Required 用于地址、SSH 目标等不能留空的字段。
function Read-AegisValue([string]$Label, [string]$Default = "", [switch]$Required) {
    $prompt = if ($Default) { "$Label [$Default]" } else { $Label }
    $value = (Read-Host $prompt).Trim()
    if (-not $value) { $value = $Default }
    if ($Required -and -not $value) { throw "$Label 不能为空。" }
    return $value
}

# 已保存值默认不复用，必须由用户明确输入 y/yes。
function Read-AegisConfirmation([string]$Message) {
    return (Read-Host "$Message [y/N]").Trim().ToLowerInvariant() -in @("y", "yes")
}

# Token 可以作为默认值长期保存；重新输入时使用 SecureString，避免显示在终端中。
function Read-AegisToken([string]$Label, [string]$SavedToken = "") {
    if ($SavedToken) {
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

# SSH 本地转发不使用固定端口，由系统选择当前可用的回环端口。
function Get-AegisFreeTcpPort {
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
# 先访问公开 health，确认目标服务身份后才发送 Token；local/ssh 等待时同时监控子进程。
function Wait-AegisApi {
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
            if ($Process.HasExited) {
                throw "连接进程已退出，退出码: $($Process.ExitCode)。"
            }
        }

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
# local API 与 ssh.exe 会加入 KILL_ON_JOB_CLOSE Job。PowerShell 正常退出时由 finally
# 主动关闭；窗口被直接关闭时，操作系统也会随 Job 句柄销毁子进程。
if (-not ([System.Management.Automation.PSTypeName]"AegisLauncher.JobObject").Type) {
    Add-Type -TypeDefinition @"
using System;
using System.ComponentModel;
using System.Runtime.InteropServices;

namespace AegisLauncher {
    public static class JobObject {
        private const int ExtendedLimitInformationClass = 9;
        private const uint KillOnJobClose = 0x00002000;

        [StructLayout(LayoutKind.Sequential)]
        private struct BasicLimits {
            public long PerProcessUserTimeLimit, PerJobUserTimeLimit;
            public uint LimitFlags;
            public UIntPtr MinimumWorkingSetSize, MaximumWorkingSetSize;
            public uint ActiveProcessLimit;
            public UIntPtr Affinity;
            public uint PriorityClass, SchedulingClass;
        }

        [StructLayout(LayoutKind.Sequential)]
        private struct IoCounters {
            public ulong ReadOperationCount, WriteOperationCount, OtherOperationCount;
            public ulong ReadTransferCount, WriteTransferCount, OtherTransferCount;
        }

        [StructLayout(LayoutKind.Sequential)]
        private struct ExtendedLimits {
            public BasicLimits BasicLimitInformation;
            public IoCounters IoInfo;
            public UIntPtr ProcessMemoryLimit, JobMemoryLimit;
            public UIntPtr PeakProcessMemoryUsed, PeakJobMemoryUsed;
        }

        [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)] private static extern IntPtr CreateJobObject(IntPtr attributes, string name);
        [DllImport("kernel32.dll", SetLastError = true)] private static extern bool SetInformationJobObject(IntPtr job, int kind, IntPtr info, uint size);
        [DllImport("kernel32.dll", SetLastError = true)] private static extern bool AssignProcessToJobObject(IntPtr job, IntPtr process);
        [DllImport("kernel32.dll", SetLastError = true)] private static extern bool CloseHandle(IntPtr handle);

        public static IntPtr CreateKillOnClose() {
            IntPtr job = CreateJobObject(IntPtr.Zero, null);
            if (job == IntPtr.Zero) { throw new Win32Exception(Marshal.GetLastWin32Error()); }
            ExtendedLimits limits = new ExtendedLimits();
            limits.BasicLimitInformation.LimitFlags = KillOnJobClose;
            int size = Marshal.SizeOf(typeof(ExtendedLimits));
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
# 唯一持久文件是 StateRoot\config.json，只保存 API 默认值和本地端口。
# 不读取旧版配置，不保存上次模式、SSH 目标、SSH 密码、PID 或历史会话状态。
function New-AegisConnectionConfig {
    param(
        [Parameter(Mandatory = $true)][ValidateSet("direct", "local", "ssh")][string]$Mode,
        [Parameter(Mandatory = $true)][string]$StateRoot
    )

    $configPath = Join-Path $StateRoot "config.json"
    $defaults = [ordered]@{
        Direct = [ordered]@{ Server = ""; Token = "" }
        Local = [ordered]@{ ApiPort = 8000; Token = "" }
        Ssh = [ordered]@{ Token = "" }
    }

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

    if ($Mode -eq "direct") {
        $useSaved = $defaults.Direct.Server -and $defaults.Direct.Token -and
            (Read-AegisConfirmation "使用已保存的直连 API $($defaults.Direct.Server)")
        if ($useSaved) {
            $server = $defaults.Direct.Server
            $token = $defaults.Direct.Token
        }
        else {
            $server = (Read-AegisValue -Label "API 地址" -Required).TrimEnd("/")
            if ($server -notmatch '^https?://') {
                throw "API 地址必须以 http:// 或 https:// 开头。"
            }
            $token = Read-AegisToken -Label "API Token"
            $defaults.Direct.Server = $server
            $defaults.Direct.Token = $token
        }
        $connection = [pscustomobject]@{
            Mode = "direct"; Server = $server; Token = $token; ApiPort = 0; Ssh = $null
        }
    }
    elseif ($Mode -eq "local") {
        $portText = Read-AegisValue -Label "本地 API 端口" -Default ([string]$defaults.Local.ApiPort)
        $apiPort = 0
        if (-not [int]::TryParse($portText, [ref]$apiPort) -or $apiPort -lt 1 -or $apiPort -gt 65535) {
            throw "本地 API 端口必须位于 1 到 65535 之间。"
        }
        $token = Read-AegisToken -Label "本地 API Token" -SavedToken $defaults.Local.Token
        $defaults.Local.ApiPort = $apiPort
        $defaults.Local.Token = $token
        $connection = [pscustomobject]@{
            Mode = "local"; Server = "http://127.0.0.1:$apiPort"; Token = $token
            ApiPort = $apiPort; Ssh = $null
        }
    }
    else {
        $sshCommand = Read-AegisValue -Label "SSH 连接命令（例如 ssh -p 31544 root@host）" -Required
        $match = [Regex]::Match(
            $sshCommand,
            '^(?:ssh(?:\.exe)?\s+)?(?:-p\s+([0-9]+)\s+)?([^\s@]+@[^\s@]+)$',
            [Text.RegularExpressions.RegexOptions]::IgnoreCase
        )
        if (-not $match.Success) { throw "SSH 连接命令格式应为 ssh -p 端口 user@host。" }

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

        $token = Read-AegisToken -Label "远端 Aegis API Token（不是 SSH 密码）" `
            -SavedToken $defaults.Ssh.Token
        $defaults.Ssh.Token = $token
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

    # 三种模式共用一次原子写入；中途中断不会留下半个 JSON。
    $null = [IO.Directory]::CreateDirectory($StateRoot)
    $tempPath = Join-Path $StateRoot ("config.{0}.{1}.tmp" -f $PID, [Guid]::NewGuid().ToString("N"))
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
        $json = $document | ConvertTo-Json -Depth 5
        [IO.File]::WriteAllText($tempPath, $json + [Environment]::NewLine, [Text.UTF8Encoding]::new($false))
        Move-Item -LiteralPath $tempPath -Destination $configPath -Force
    }
    finally {
        Remove-Item -LiteralPath $tempPath -Force -ErrorAction SilentlyContinue
    }
    return $connection
}

# =====================================================================
# 建立与关闭当前连接
# =====================================================================
# direct 只验证已有 API。local/ssh 每次创建新进程和独立会话目录，从不复用旧 SSH 配置。
function Open-AegisConnection {
    param(
        [Parameter(Mandatory = $true)]$Config,
        [Parameter(Mandatory = $true)][string]$StateRoot,
        [Parameter(Mandatory = $true)][string]$PythonPath,
        [Parameter(Mandatory = $true)][string]$ProjectRoot
    )

    if ($Config.Mode -eq "direct") {
        Write-Host ""
        Write-Host ">>> [验证服务] 正在验证直连 API..." -ForegroundColor Cyan
        Wait-AegisApi -Server $Config.Server -Token $Config.Token
        Write-Host "      [-] API 地址: $($Config.Server)"
        return $null
    }

    $sessionId = "{0}-{1}" -f [DateTime]::Now.ToString("yyyyMMdd-HHmmss"), [Guid]::NewGuid().ToString("N").Substring(0, 8)
    $sessionRoot = Join-Path (Join-Path $StateRoot "sessions") $sessionId
    $null = [IO.Directory]::CreateDirectory($sessionRoot)
    $process = $null
    $jobHandle = [IntPtr]::Zero

    try {
        $jobHandle = [AegisLauncher.JobObject]::CreateKillOnClose()
        if ($Config.Mode -eq "local") {
            Write-Host ""
            Write-Host ">>> [启动服务] 正在启动本次会话的本地 API..." -ForegroundColor Cyan
            $tokenWasSet = Test-Path Env:AEGIS_API_TOKEN
            $previousToken = $env:AEGIS_API_TOKEN
            try {
                $env:AEGIS_API_TOKEN = $Config.Token
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
            $ssh = Get-Command ssh.exe -ErrorAction Stop
            $knownHostsPath = Join-Path $sessionRoot "known_hosts"
            $forward = "127.0.0.1:{0}:{1}:{2}" -f $Config.Ssh.LocalPort, $Config.Ssh.RemoteApiHost, $Config.Ssh.RemoteApiPort
            Write-Host ""
            Write-Host ">>> [建立连接] 正在创建本次会话的一次性 SSH 隧道..." -ForegroundColor Cyan
            Write-Host "      请根据 SSH 提示输入服务器密码；密码不会保存。" -ForegroundColor Yellow
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

        [AegisLauncher.JobObject]::Assign($jobHandle, $process.Handle)
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
        return [pscustomobject]@{
            Mode = $Config.Mode
            Process = $process
            JobHandle = $jobHandle
            SessionRoot = $sessionRoot
        }
    }
    catch {
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

# 关闭顺序固定为“进程、Job、会话目录”；独立 known_hosts 和日志随本次会话一起删除。
function Close-AegisConnection {
    param(
        [Parameter(Mandatory = $true)]$ManagedConnection,
        [Parameter(Mandatory = $true)][string]$StateRoot
    )

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
        Write-Host "      [警告] 进程回收失败，将由 Job Object 强制关闭。" -ForegroundColor Yellow
    }
    finally {
        if ($ManagedConnection.JobHandle -ne [IntPtr]::Zero) {
            [AegisLauncher.JobObject]::Close($ManagedConnection.JobHandle)
        }
        try { $null = $ManagedConnection.Process.WaitForExit(5000) } catch { }
    }

    # SessionRoot 只接受 StateRoot\sessions 下的路径，避免递归删除越出统一目录。
    $separator = [IO.Path]::DirectorySeparatorChar
    $sessionsRoot = [IO.Path]::GetFullPath((Join-Path $StateRoot "sessions")).TrimEnd($separator) + $separator
    $sessionRoot = [IO.Path]::GetFullPath([string]$ManagedConnection.SessionRoot).TrimEnd($separator) + $separator
    if (-not $sessionRoot.StartsWith($sessionsRoot, [StringComparison]::OrdinalIgnoreCase)) {
        Write-Host "      [警告] 拒绝清理统一会话目录之外的路径。" -ForegroundColor Yellow
        return
    }
    try {
        Remove-Item -LiteralPath $ManagedConnection.SessionRoot -Recurse -Force -ErrorAction Stop
    }
    catch {
        Write-Host "      [警告] 会话目录未能删除: $($_.Exception.Message)" -ForegroundColor Yellow
    }
    Write-Host "      [-] 当前连接已关闭。"
}
