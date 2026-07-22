# Aegis-LoRA - 命令行客户端
import json
import time
from pathlib import Path
from typing import Literal

import httpx
import typer
from rich import box
from rich.console import Console
from rich.table import Table
from rich.text import Text

# =====================================================================
# 客户端配置
# =====================================================================
# Typer 负责命令注册、参数解析和帮助信息；未传命令时直接展示帮助。
app = typer.Typer(
    name="aegis",
    help="Aegis-LoRA 远程检测与清洗客户端",
    no_args_is_help=True,
)

# 登录信息固定保存在用户目录，不依赖工程位置和当前终端目录。
CONFIG_FILE = Path.home() / ".aegis" / "config.json"


# =====================================================================
# 核心逻辑
# =====================================================================
def _client() -> httpx.Client:
    """读取登录配置并创建适合长耗时任务的 HTTP 客户端。"""

    # -----------------------------------------------------------------
    # 1. 读取当前登录信息
    # -----------------------------------------------------------------
    if not CONFIG_FILE.is_file():
        typer.echo("尚未登录，请先执行 login。")
        raise typer.Exit(1)

    # config 只保存服务地址和 Token，是所有受保护命令的共同连接上下文。
    config = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))

    # 读取阶段不设总超时，避免深度清洗轮询被客户端提前中断。
    return httpx.Client(
        base_url=config["server"],
        headers={"Authorization": f"Bearer {config['token']}"},
        timeout=httpx.Timeout(None, connect=10.0),
    )


def _request(
    client: httpx.Client,
    method: str,
    url: str,
    *,
    output: Path | None = None,
    **kwargs,
):
    """统一处理 JSON 请求、API 错误和流式文件下载。"""

    # -----------------------------------------------------------------
    # 1. 发送普通请求或流式下载
    # -----------------------------------------------------------------
    try:
        if output is None:
            # 普通接口响应较小，直接请求并在末尾解析 JSON。
            response = client.request(method, url, **kwargs)
        else:
            # 报告和模型产物使用流式下载，避免大文件一次性进入内存。
            output = output.expanduser().resolve()
            output.parent.mkdir(parents=True, exist_ok=True)
            with client.stream(method, url, **kwargs) as response:
                if response.is_error:
                    # 错误响应先读取完整内容，退出流上下文后再提取 detail。
                    response.read()
                else:
                    # output 是用户最终下载路径，父目录按需创建。
                    with output.open("wb") as file:
                        for chunk in response.iter_bytes():
                            file.write(chunk)
    except httpx.RequestError as exc:
        # DNS、连接拒绝和传输中断统一视为服务器连接问题。
        typer.echo(f"无法连接服务器：{exc}")
        raise typer.Exit(1)

    # -----------------------------------------------------------------
    # 2. 转换 API 错误信息
    # -----------------------------------------------------------------
    if response.is_error:
        try:
            # FastAPI 错误优先读取 detail，非标准响应回退到原始正文。
            data = response.json()
            detail = data.get("detail", response.text)
        except (ValueError, AttributeError):
            detail = response.text
        typer.echo(f"请求失败 [{response.status_code}]：{detail}")
        raise typer.Exit(1)

    # -----------------------------------------------------------------
    # 3. 返回下载结果或 JSON 数据
    # -----------------------------------------------------------------
    if output is not None:
        typer.echo(f"已保存：{output}")
        return None
    return response.json()


def _display(view: str, data, json_output: bool = False):
    """将接口数据转换为统一的终端摘要，并按需保留原始 JSON。"""

    # -----------------------------------------------------------------
    # 1. 处理原始 JSON 模式
    # -----------------------------------------------------------------
    # JSON 模式只输出接口数据，便于重定向或交给其他脚本处理。
    if json_output:
        typer.echo(json.dumps(data, ensure_ascii=False, indent=2))
        return

    # -----------------------------------------------------------------
    # 2. 准备统一终端样式
    # -----------------------------------------------------------------
    # states 统一状态语义和颜色；modes 统一清洗模式的中文名称。
    console = Console(highlight=False)
    states = {
        "ready": Text("就绪", style="bold green"),
        "degraded": Text("降级", style="bold yellow"),
        "safe": Text("安全", style="bold green"),
        "poisoned": Text("疑似中毒", style="bold red"),
        "queued": Text("等待", style="yellow"),
        "running": Text("运行中", style="cyan"),
        "succeeded": Text("成功", style="bold green"),
        "failed": Text("失败", style="bold red"),
    }
    modes = {"fast": "快速", "deep": "深度"}

    # table_style 被本次页面中的所有表格复用，保持间距和边框一致。
    table_style = dict(box=box.SIMPLE_HEAD, header_style="bold", padding=(0, 1))
    console.print()

    # -----------------------------------------------------------------
    # 3. 按接口类型生成摘要
    # -----------------------------------------------------------------
    # 健康检查突出总体结论，并按类别列出所有运行能力。
    if view == "health":
        # status 是服务器总体结论，ready 之外均表示存在不可用能力。
        status = data.get("status", "unknown")
        console.print(
            "[bold]服务状态[/bold]  ",
            states.get(status, str(status)),
            f"  {data.get('service', '-')} v{data.get('version', '-')}",
        )
        # rows 将基础能力、模型、签名和运行目录统一为三列表格数据。
        rows = [
            ("基础能力", "认证", data.get("auth_ready")),
            ("基础能力", "静态检测器", data.get("detector_ready")),
            *(
                ("基础模型", name, ready)
                for name, ready in data.get("models_ready", {}).items()
            ),
            *(
                ("快速清洗", name, ready)
                for name, ready in data.get("fast_cleanse_ready", {}).items()
            ),
            ("深度清洗", "训练与恢复数据", data.get("deep_cleanse_ready")),
            ("运行环境", "存储目录", data.get("storage_ready")),
        ]
        # 同一状态列只使用“就绪/缺失”，缺失项以红色突出。
        table = Table(**table_style)
        table.add_column("类别", style="cyan", no_wrap=True)
        table.add_column("对象")
        table.add_column("状态", no_wrap=True)
        for category, name, ready in rows:
            table.add_row(
                category,
                name,
                Text("就绪" if ready else "缺失", style="green" if ready else "red"),
            )
        console.print(table)

    # 模型列表只展示选择模型时真正需要的编号、系列和状态。
    elif view == "models":
        # 每行对应一个服务端注册模型，ready 表示模型文件可以加载。
        table = Table(title="基础模型", **table_style)
        table.add_column("模型 ID", style="cyan")
        table.add_column("系列")
        table.add_column("状态")
        for model in data:
            table.add_row(
                model["model_id"],
                model["family"],
                Text(
                    "就绪" if model["ready"] else "缺失",
                    style="green" if model["ready"] else "red",
                ),
            )
        console.print(table)

    # 单独扫描先给出判定，再展示影响判定的核心指标。
    elif view == "scan":
        # risk 与 threshold 共同决定 verdict，差值用于直观看出越界程度。
        risk = float(data.get("risk_score", 0.0))
        threshold = float(data.get("threshold", 0.0))
        verdict = data.get("verdict", "unknown")
        console.print("[bold]检测结论[/bold]  ", states.get(verdict, str(verdict)))

        # 检测器和耗时作为审计信息保留，但不干扰首行判定结论。
        table = Table(**table_style)
        table.add_column("指标", style="cyan")
        table.add_column("结果")
        table.add_row("风险分数", f"{risk:.4f}")
        table.add_row("判定阈值", f"{threshold:.4f}")
        table.add_row("阈值差值", f"{risk - threshold:+.4f}")
        table.add_row("检测器", str(data.get("detector", "-")))
        table.add_row("耗时", f"{float(data.get('elapsed_seconds', 0.0)):.2f} 秒")
        console.print(table)
        console.print(
            "[yellow]建议：使用 audit 命令创建清洗任务。[/yellow]"
            if data.get("is_poisoned")
            else "[green]未发现超过判定阈值的风险特征。[/green]"
        )

    # 任务列表使用单行摘要，完整扫描和清洗信息交由 show 展开。
    elif view == "jobs":
        # items 是本次 limit 截取后的任务，total 是服务器全部任务数量。
        items = data.get("items", [])
        console.print(
            f"[bold]任务列表[/bold]  共 {data.get('total', len(items))} 个，"
            f"当前显示 {len(items)} 个"
        )
        if not items:
            console.print("[yellow]暂无审计任务。[/yellow]")
            return

        # 表格只保留定位任务及判断当前进度所需的字段。
        table = Table(**table_style)
        table.add_column("任务 ID", style="cyan", no_wrap=True)
        table.add_column("状态", no_wrap=True)
        table.add_column("阶段", no_wrap=True)
        table.add_column("模型")
        table.add_column("模式", no_wrap=True)
        table.add_column("提交时间", no_wrap=True)
        for job in items:
            status = job.get("status", "unknown")
            table.add_row(
                str(job.get("job_id", "-")),
                states.get(status, str(status)),
                str(job.get("stage", "-")),
                str(job.get("model_id", "-")),
                modes.get(job.get("cleanse_mode"), "-"),
                str(job.get("submitted_at") or "-").replace("T", " ")[:16],
            )
        console.print(table)
        console.print("使用 [cyan]show <任务 ID>[/cyan] 查看完整结果。")

    # 单任务详情根据终态展示检测、清洗、下载入口或失败原因。
    elif view == "show":
        # status 决定任务当前状态样式，job_id 同时用于后续下载命令。
        status = data.get("status", "unknown")
        job_id = str(data.get("job_id", "-"))
        console.print(
            f"[bold]任务详情[/bold]  [cyan]{job_id}[/cyan]  ",
            states.get(status, str(status)),
        )

        # 第一张表只描述任务上下文和执行时间，不混入算法结果。
        table = Table(**table_style)
        table.add_column("项目", style="cyan")
        table.add_column("内容")
        table.add_row("当前阶段", str(data.get("stage", "-")))
        table.add_row("基础模型", str(data.get("model_id", "-")))
        table.add_row("LoRA ID", str(data.get("lora_id", "-")))
        table.add_row("清洗模式", modes.get(data.get("cleanse_mode"), "-"))
        for label, key in (
            ("提交时间", "submitted_at"),
            ("开始时间", "started_at"),
            ("结束时间", "finished_at"),
        ):
            table.add_row(label, str(data.get(key) or "-").replace("T", " "))
        console.print(table)

        # result 只在成功终态出现；scan 与 cleanse 分别对应检测和清洗结果。
        result = data.get("result") or {}
        scan = result.get("scan") or {}
        if scan:
            # 检测结果采用与 scan 命令相同的结论、分数和阈值口径。
            scan_table = Table(title="检测结果", **table_style)
            scan_table.add_column("结论")
            scan_table.add_column("风险分数", justify="right")
            scan_table.add_column("阈值", justify="right")
            scan_table.add_column("检测器")
            verdict = scan.get("verdict", "unknown")
            scan_table.add_row(
                states.get(verdict, str(verdict)),
                f"{float(scan.get('risk_score', 0.0)):.4f}",
                f"{float(scan.get('threshold', 0.0)):.4f}",
                str(scan.get("detector", "-")),
            )
            console.print(scan_table)

        cleanse = result.get("cleanse") or {}

        # 失败、安全通过、清洗完成和活动任务分别输出对应的最终操作信息。
        if status == "failed":
            console.print(
                "[bold red]失败原因[/bold red]  ",
                Text(str(data.get("error") or "未知错误"), style="red"),
            )
        elif result.get("action") == "passed":
            console.print("[bold green]处理结果：安全通过，无需执行清洗。[/bold green]")
        elif cleanse:
            cleanse_table = Table(title="清洗结果", **table_style)
            cleanse_table.add_column("项目", style="cyan")
            cleanse_table.add_column("内容")
            cleanse_table.add_row("清洗模式", modes.get(cleanse.get("mode"), "-"))
            cleanse_table.add_row("签名", str(cleanse.get("signature") or "动态提取"))
            cleanse_table.add_row(
                "抑制参数",
                f"{int(cleanse.get('suppressed_count', 0)):,}",
            )
            cleanse_table.add_row("审计报告", f"report {job_id}")
            cleanse_table.add_row("清洗模型", f"artifact {job_id}")
            console.print(cleanse_table)
        else:
            console.print("[yellow]任务尚未结束，可稍后再次执行 show。[/yellow]")


# =====================================================================
# 登录
# =====================================================================
@app.command()
def login(
    server: str = typer.Argument(..., help="API 地址"),
    token: str = typer.Option(..., "--token", prompt=True, hide_input=True),
):
    """验证并保存服务器地址和 Token。"""

    # -----------------------------------------------------------------
    # 1. 使用受保护接口验证连接信息
    # -----------------------------------------------------------------
    # 去掉末尾斜杠，避免后续接口路径出现重复分隔符。
    server = server.rstrip("/")
    with httpx.Client(
        base_url=server,
        headers={"Authorization": f"Bearer {token}"},
        timeout=10.0,
    ) as client:
        _request(client, "GET", "/v1/me")

    # -----------------------------------------------------------------
    # 2. 保存已经验证通过的登录信息
    # -----------------------------------------------------------------
    # 同一地址重新登录时保留连接编排配置；切换地址则回到普通直连模式。
    saved_config = {}
    if CONFIG_FILE.is_file():
        try:
            current_config = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
            if isinstance(current_config, dict):
                saved_config = current_config
        except (OSError, TypeError, ValueError):
            saved_config = {}

    current_server = str(saved_config.get("server", "")).rstrip("/")
    launcher_config = saved_config.get("launcher")
    if current_server != server or not isinstance(launcher_config, dict):
        saved_config["launcher"] = {"version": 1, "mode": "direct"}
    saved_config.update(server=server, token=token)

    CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_FILE.write_text(
        json.dumps(saved_config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    typer.echo(f"登录成功：{server}")


@app.command()
def logout():
    """删除本地登录配置。"""
    # missing_ok 使重复退出保持幂等。
    CONFIG_FILE.unlink(missing_ok=True)
    typer.echo("已退出登录。")


# =====================================================================
# 服务器状态
# =====================================================================
@app.command()
def health(
    server: str | None = typer.Argument(None, help="API 地址；省略时使用登录配置"),
    json_output: bool = typer.Option(False, "--json", help="输出原始 JSON"),
):
    """检查服务器检测与清洗能力是否就绪。"""

    # -----------------------------------------------------------------
    # 1. 请求公开健康接口
    # -----------------------------------------------------------------
    # 显式地址用于未登录诊断；省略时复用当前登录服务器。
    if server:
        with httpx.Client(base_url=server.rstrip("/"), timeout=10.0) as client:
            result = _request(client, "GET", "/health")
    else:
        with _client() as client:
            result = _request(client, "GET", "/health")

    # -----------------------------------------------------------------
    # 2. 展示能力状态并返回探活退出码
    # -----------------------------------------------------------------
    _display("health", result, json_output)

    # degraded 使用非零退出码，便于部署脚本直接判断整体状态。
    if result.get("status") != "ready":
        raise typer.Exit(1)


# =====================================================================
# 模型与审计
# =====================================================================
@app.command("models")
def list_models():
    """查看服务器已经注册的基础模型。"""
    # 模型接口只返回稳定编号、系列和就绪状态，直接交给统一展示层。
    with _client() as client:
        _display("models", _request(client, "GET", "/v1/models"))


@app.command()
def scan(
    lora_path: Path,
    batch: bool = typer.Option(False, "--batch", help="批量扫描目录中的 LoRA"),
    output: Path | None = typer.Option(
        None, "--output", "-o", help="扫描结果 JSON 归档路径"
    ),
    json_output: bool = typer.Option(False, "--json", help="输出原始 JSON"),
):
    """上传并检测单个或一组 LoRA。"""

    # -----------------------------------------------------------------
    # 1. 定位待扫描 LoRA
    # -----------------------------------------------------------------
    # 单扫描直接使用目标目录；批量扫描根据标准权重文件递归定位各 LoRA。
    lora_path = lora_path.expanduser().resolve()
    lora_paths = (
        sorted({path.parent for path in lora_path.rglob("adapter_model.safetensors")})
        if batch
        else [lora_path]
    )
    if not lora_paths:
        typer.echo(f"目录中未发现 LoRA：{lora_path}")
        raise typer.Exit(1)

    # -----------------------------------------------------------------
    # 2. 依次上传并扫描 LoRA
    # -----------------------------------------------------------------
    # results 保存每个 LoRA 的独立结果；批量模式额外保留名称用于归档定位。
    results = []

    # 同一批次复用连接，LoRA 仍按顺序上传扫描，避免并发请求争用服务端检测器。
    with _client() as client:
        for index, current_path in enumerate(lora_paths, start=1):
            # JSON 模式保持标准输出纯净，其余模式展示当前扫描进度。
            if not json_output:
                typer.echo(
                    f"[{index}/{len(lora_paths)}] 正在扫描 {current_path.name}..."
                    if batch
                    else "正在上传 LoRA..."
                )

            try:
                # 每个 LoRA 由标准权重和 PEFT 配置共同组成。
                weights_path = current_path / "adapter_model.safetensors"
                config_path = current_path / "adapter_config.json"
                with (
                    weights_path.open("rb") as weights_file,
                    config_path.open("rb") as config_file,
                ):
                    # 上传接口返回一次性 lora_id，扫描接口使用后由服务端回收资源。
                    uploaded = _request(
                        client,
                        "POST",
                        "/v1/loras",
                        files={
                            "weights": (
                                weights_path.name,
                                weights_file,
                                "application/octet-stream",
                            ),
                            "config": (
                                config_path.name,
                                config_file,
                                "application/json",
                            ),
                        },
                    )
                    scan_result = _request(
                        client,
                        "POST",
                        "/v1/scan",
                        json={"lora_id": uploaded["lora_id"]},
                    )

                # 单扫描保持原始响应；批量归档在响应前增加 LoRA 名称。
                results.append(
                    {"name": current_path.name, **scan_result} if batch else scan_result
                )
            except (OSError, typer.Exit) as exc:
                # 单扫描沿用原有异常；批量扫描记录失败项后继续处理其余 LoRA。
                if not batch:
                    raise
                results.append(
                    {"name": current_path.name, "error": str(exc) or "扫描失败"}
                )

    # -----------------------------------------------------------------
    # 3. 汇总并归档扫描结果
    # -----------------------------------------------------------------
    if batch:
        # 批量归档保留四项计数和逐项结果，不额外生成报告或模型产物。
        result = {
            "total": len(results),
            "safe": sum(item.get("verdict") == "safe" for item in results),
            "poisoned": sum(item.get("verdict") == "poisoned" for item in results),
            "failed": sum("error" in item for item in results),
            "items": results,
        }

        # 未指定输出路径时，以当前时间生成不会重复的默认归档名。
        output = output or Path(f"scan_archive_{time.strftime('%Y%m%d_%H%M%S')}.json")
    else:
        # 单扫描继续返回服务端原始结果，兼容原有终端和 JSON 输出。
        result = results[0]

    # output 同时适用于单扫描和批量扫描，归档内容统一使用 UTF-8 JSON。
    if output is not None:
        output = output.expanduser().resolve()
        output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # 批量模式输出整体统计，单扫描继续复用现有详情视图。
    if batch:
        if json_output:
            typer.echo(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            typer.echo(
                f"批量扫描完成：共 {result['total']} 个，安全 {result['safe']} 个，"
                f"疑似中毒 {result['poisoned']} 个，失败 {result['failed']} 个。"
            )
    else:
        _display("scan", result, json_output)
    if output is not None and not json_output:
        typer.echo(f"扫描归档已保存：{output}")


@app.command()
def audit(
    lora_path: Path,
    model: str = typer.Option(..., "--model", "-m"),
    mode: Literal["fast", "deep"] = typer.Option("fast", "--mode"),
    wait: bool = typer.Option(True, "--wait/--no-wait"),
):
    """上传 LoRA，创建审计任务并按需等待完成。"""

    # -----------------------------------------------------------------
    # 1. 定位待审计 LoRA
    # -----------------------------------------------------------------
    lora_path = lora_path.expanduser().resolve()
    weights_path = lora_path / "adapter_model.safetensors"
    config_path = lora_path / "adapter_config.json"

    # -----------------------------------------------------------------
    # 2. 上传 LoRA 并创建后台任务
    # -----------------------------------------------------------------
    typer.echo("正在上传 LoRA...")
    with (
        _client() as client,
        weights_path.open("rb") as weights_file,
        config_path.open("rb") as config_file,
    ):
        # 上传结果中的 lora_id 是创建审计任务时使用的临时资源编号。
        uploaded = _request(
            client,
            "POST",
            "/v1/loras",
            files={
                "weights": (
                    weights_path.name,
                    weights_file,
                    "application/octet-stream",
                ),
                "config": (config_path.name, config_file, "application/json"),
            },
        )
        # job 保存最新任务快照，轮询期间会被服务器响应持续替换。
        job = _request(
            client,
            "POST",
            "/v1/jobs",
            json={
                "model_id": model,
                "lora_id": uploaded["lora_id"],
                "cleanse_mode": mode,
            },
        )
        typer.echo(f"任务已创建：{job['job_id']}")
        if not wait:
            return

        # -----------------------------------------------------------------
        # 3. 轮询任务并只输出变化的阶段
        # -----------------------------------------------------------------
        # last_stage 避免每两秒重复打印相同状态。
        last_stage = None
        while job["status"] in ("queued", "running"):
            job = _request(client, "GET", f"/v1/jobs/{job['job_id']}")
            if job.get("stage") != last_stage:
                typer.echo(f"当前阶段：{job.get('stage', '-')}")
                last_stage = job.get("stage")
            if job["status"] in ("queued", "running"):
                time.sleep(2)

    # -----------------------------------------------------------------
    # 4. 展示最终检测、清洗或失败结果
    # -----------------------------------------------------------------
    _display("show", job)


# =====================================================================
# 任务与产物
# =====================================================================
@app.command("jobs")
def list_jobs(
    limit: int = 20,
    json_output: bool = typer.Option(False, "--json", help="输出原始 JSON"),
):
    """查看最近提交的任务。"""
    # limit 控制本次返回数量，total 仍由服务端保留全部任务规模。
    with _client() as client:
        result = _request(client, "GET", "/v1/jobs", params={"limit": limit})

    # 默认输出任务摘要，--json 保留完整任务快照。
    _display("jobs", result, json_output)


@app.command()
def show(
    job_id: str,
    json_output: bool = typer.Option(False, "--json", help="输出原始 JSON"),
):
    """查看单个任务的完整状态和结果。"""
    # job_id 同时用于任务查询、报告下载和模型产物下载。
    with _client() as client:
        result = _request(client, "GET", f"/v1/jobs/{job_id}")

    # 统一展示层根据任务终态决定输出检测、清洗或错误信息。
    _display("show", result, json_output)


@app.command()
def report(
    job_id: str,
    report_format: Literal["html", "json"] = typer.Option("html", "--format"),
    output: Path | None = typer.Option(None, "--output", "-o"),
):
    """下载任务生成的 HTML 或 JSON 审计报告。"""
    # 未指定 output 时，使用任务编号生成可直接识别的默认文件名。
    output = output or Path(f"{job_id}_report.{report_format}")

    # output 参数使统一请求逻辑进入流式下载分支。
    with _client() as client:
        _request(
            client,
            "GET",
            f"/v1/jobs/{job_id}/report?report_format={report_format}",
            output=output,
        )


@app.command()
def artifact(
    job_id: str,
    output: Path | None = typer.Option(None, "--output", "-o"),
):
    """下载清洗后的 LoRA ZIP。"""
    # 清洗产物固定为 ZIP，默认名称与对应任务保持一致。
    output = output or Path(f"{job_id}_cleaned_lora.zip")
    with _client() as client:
        _request(client, "GET", f"/v1/jobs/{job_id}/artifact", output=output)


if __name__ == "__main__":
    # 无论由启动器还是模块方式运行，帮助信息统一展示 aegis 命令。
    app(prog_name="aegis")
