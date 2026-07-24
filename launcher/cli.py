# Aegis-LoRA - 命令行客户端
# 负责 API 会话、LoRA 上传、任务轮询、结果展示和产物下载。
import json
import os
import sys
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
# 命令注册
# =====================================================================
# app 统一注册命令和参数；未传命令时直接显示帮助。
app = typer.Typer(
    name="aegis",
    help="Aegis-LoRA 远程检测与清洗客户端",
    no_args_is_help=True,
)


# =====================================================================
# API 客户端与请求
# =====================================================================
def _client() -> httpx.Client:
    """创建使用当前环境配置的 Aegis API 客户端。"""
    # server 和 token 由启动脚本注入，不在 CLI 内持久化。
    server = os.getenv("AEGIS_API_SERVER", "").strip().rstrip("/")
    token = os.getenv("AEGIS_API_TOKEN", "")
    if not server or not token:
        typer.echo(
            "当前 CLI 会话缺少 API 地址或 Token；请使用 start-cli.bat，"
            "或设置 AEGIS_API_SERVER 和 AEGIS_API_TOKEN。"
        )
        raise typer.Exit(1)

    return httpx.Client(
        base_url=server,
        headers={"Authorization": f"Bearer {token}"},
        timeout=httpx.Timeout(None, connect=10.0),
    )


def _request(
    client: httpx.Client,
    method: str,
    url: str,
    *,
    output: Path | None = None,
    fatal: bool = True,
    **kwargs,
):
    """发送 API 请求并统一处理 JSON、下载及错误响应。"""
    # fatal 控制请求失败是否结束命令；批量扫描使用非致命错误继续后续项。
    # output 为空时返回 JSON，指定时流式写入目标文件。

    # -----------------------------------------------------------------
    # 步骤 1：发送普通请求或流式下载
    # -----------------------------------------------------------------
    try:
        if output is None:
            response = client.request(method, url, **kwargs)
        else:
            # 报告和模型产物使用流式下载，避免一次性读入内存。
            output = output.expanduser().resolve()
            output.parent.mkdir(parents=True, exist_ok=True)
            with client.stream(method, url, **kwargs) as response:
                if response.is_error:
                    # 错误响应需先读取正文，退出流上下文后仍可解析 detail。
                    response.read()
                else:
                    with output.open("wb") as file:
                        for chunk in response.iter_bytes():
                            file.write(chunk)
    except httpx.RequestError as exc:
        # DNS、连接拒绝和传输中断统一为连接错误。
        message = f"无法连接服务器：{exc}"
    else:
        if not response.is_error:
            if output is not None:
                typer.echo(f"已保存：{output}")
                return None
            return response.json()
        try:
            # FastAPI 响应优先读取 detail，非标准响应回退到原始正文。
            detail = response.json().get("detail", response.text)
        except (ValueError, AttributeError):
            detail = response.text
        message = f"请求失败 [{response.status_code}]：{detail}"

    # -----------------------------------------------------------------
    # 步骤 2：统一转换连接和 API 错误
    # -----------------------------------------------------------------
    if fatal:
        typer.echo(message)
        raise typer.Exit(1)
    raise RuntimeError(message)


def _upload_lora(
    client: httpx.Client,
    lora_path: Path,
    *,
    fatal: bool = True,
):
    """上传标准 PEFT LoRA 的权重和配置文件。"""
    # scan 与 audit 共用标准 PEFT 文件和 multipart 结构。
    weights_path = lora_path / "adapter_model.safetensors"
    config_path = lora_path / "adapter_config.json"
    try:
        with (
            weights_path.open("rb") as weights_file,
            config_path.open("rb") as config_file,
        ):
            files = {
                "weights": (
                    weights_path.name,
                    weights_file,
                    "application/octet-stream",
                ),
                "config": (config_path.name, config_file, "application/json"),
            }
            return _request(client, "POST", "/v1/loras", fatal=fatal, files=files)
    except OSError as exc:
        message = f"无法读取 LoRA 文件：{exc}"
        if fatal:
            typer.echo(message)
            raise typer.Exit(1)
        raise RuntimeError(message) from exc


def _display(view: str, data, json_output: bool = False):
    """按指定视图格式化并输出 API 响应。"""
    # view 选择展示布局；json_output 跳过 Rich 格式化并保留原始接口结构。

    # -----------------------------------------------------------------
    # 步骤 1：处理原始 JSON 模式
    # -----------------------------------------------------------------
    if json_output:
        typer.echo(json.dumps(data, ensure_ascii=False, indent=2))
        return

    # -----------------------------------------------------------------
    # 步骤 2：准备统一终端样式
    # -----------------------------------------------------------------
    # states 和 modes 是全部视图共用的状态与模式映射。
    console = Console(highlight=False)
    states = {
        "ready": Text("就绪", style="bold green"),
        "degraded": Text("降级", style="bold yellow"),
        "safe": Text("安全", style="bold green"),
        "poisoned": Text("疑似中毒", style="bold red"),
        "queued": Text("等待", style="yellow"),
        "running": Text("运行中", style="cyan"),
        "awaiting_confirmation": Text("待确认", style="bold yellow"),
        "succeeded": Text("成功", style="bold green"),
        "failed": Text("失败", style="bold red"),
    }
    modes = {"fast": "快速", "deep": "深度"}

    # table_style 保持所有表格的边框、表头和间距一致。
    table_style = dict(box=box.SIMPLE_HEAD, header_style="bold", padding=(0, 1))
    console.print()

    # -----------------------------------------------------------------
    # 步骤 3：按接口类型生成终端视图
    # -----------------------------------------------------------------
    # 健康检查按类别列出服务能力，总状态单独置顶。
    if view == "health":
        console.print(
            f"[bold]{data['service']} v{data['version']}[/bold]  ",
            states.get(data["status"], str(data["status"])),
        )
        # rows 将异构就绪项归一为“类别、对象、状态”。
        rows = [
            ("基础能力", "认证", data["auth_ready"]),
            ("基础能力", "静态检测器", data["detector_ready"]),
            *(("基础模型", name, ready) for name, ready in data["models_ready"].items()),
            *(("快速清洗", name, ready) for name, ready in data["fast_cleanse_ready"].items()),
            ("深度清洗", "训练与恢复数据", data["deep_cleanse_ready"]),
            ("运行环境", "存储目录", data["storage_ready"]),
        ]
        table = Table("类别", "对象", "状态", **table_style)
        previous_category = None
        for category, name, ready in rows:
            if previous_category is not None and category != previous_category:
                table.add_section()
            table.add_row(
                category,
                name,
                Text("就绪" if ready else "缺失", style="green" if ready else "red"),
            )
            previous_category = category
        console.print(table)

    # 模型列表只展示选择审计基座所需字段。
    elif view == "models":
        table = Table("模型 ID", "系列", "状态", title="基础模型", **table_style)
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

    # 单独扫描先展示判定，再列出风险分数、阈值和检测器。
    elif view == "scan":
        risk = float(data["risk_score"])
        threshold = float(data["threshold"])
        console.print(
            "[bold]检测结论[/bold]  ",
            states.get(data["verdict"], str(data["verdict"])),
        )

        table = Table("指标", "结果", **table_style)
        table.add_row("风险分数", f"{risk:.4f}")
        table.add_row("判定阈值", f"{threshold:.4f}")
        table.add_row("阈值差值", f"{risk - threshold:+.4f}")
        table.add_row("检测器", str(data["detector"]))
        table.add_row("耗时", f"{float(data['elapsed_seconds']):.2f} 秒")
        console.print(table)
        console.print(
            "[yellow]建议：使用 audit 命令创建清洗任务。[/yellow]"
            if data["is_poisoned"]
            else "[green]未发现超过判定阈值的风险特征。[/green]"
        )

    # jobs 只消费列表摘要，完整结果由 show 展开。
    elif view == "jobs":
        # items 受 limit 限制，total 表示服务器任务总数。
        items = data.get("items", [])
        console.print(
            f"[bold]任务列表[/bold]  共 {data.get('total', len(items))} 个，"
            f"当前显示 {len(items)} 个"
        )
        if not items:
            console.print("[yellow]暂无审计任务。[/yellow]")
            return

        table = Table(
            "任务 ID", "状态", "阶段", "模型", "模式", "提交时间", **table_style
        )
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

    # show 汇总任务上下文、检测结果、候选模型和最终操作。
    elif view == "show":
        # job_id 同时用于后续 report 和 artifact 命令。
        status = data.get("status", "unknown")
        job_id = str(data.get("job_id", "-"))
        console.print(
            f"[bold]任务详情[/bold]  [cyan]{job_id}[/cyan]  ",
            states.get(status, str(status)),
        )

        # 第一张表只展示任务上下文和执行时间。
        table = Table("项目", "内容", **table_style)
        table.add_row("当前阶段", str(data.get("stage", "-")))
        table.add_row("基础模型", str(data.get("model_id", "-")))
        table.add_row("LoRA ID", str(data.get("lora_id", "-")))
        table.add_row("清洗模式", modes.get(data.get("cleanse_mode"), "-"))

        # 候选预计大小是下载总量的唯一来源，任务只保存已下载字节。
        candidate = data.get("model_candidate") or {}
        downloaded = data.get("downloaded_bytes")
        total = int(candidate.get("estimated_size", 0))
        if downloaded is not None and total > 0:
            downloaded = int(downloaded)
            percent = max(0, min(100, downloaded * 100 // total))
            progress = f"{downloaded:,} / {total:,} 字节 ({percent}%)"
            table.add_row("模型下载", progress)
        table.add_row(
            "提交时间", str(data.get("submitted_at") or "-").replace("T", " ")
        )
        table.add_row("开始时间", str(data.get("started_at") or "-").replace("T", " "))
        table.add_row("结束时间", str(data.get("finished_at") or "-").replace("T", " "))
        console.print(table)

        # 等待阶段从 scan_result 读取检测结果，成功后改从 result.scan 读取。
        result = data.get("result") or {}
        scan = result.get("scan") or data.get("scan_result") or {}
        if scan:
            scan_table = Table(
                "结论", "风险分数", "阈值", "检测器", title="检测结果", **table_style
            )
            verdict = scan.get("verdict", "unknown")
            scan_table.add_row(
                states.get(verdict, str(verdict)),
                f"{float(scan.get('risk_score', 0.0)):.4f}",
                f"{float(scan.get('threshold', 0.0)):.4f}",
                str(scan.get("detector", "-")),
            )
            console.print(scan_table)

        if candidate:
            candidate_table = Table("项目", "内容", title="候选模型", **table_style)
            candidate_rows = {
                "名称": candidate.get("name"),
                "Repo ID": candidate.get("repo_id"),
                "Revision": candidate.get("revision"),
                "预计大小": f"{int(candidate.get('estimated_size', 0)):,} 字节",
                "LoRA 基座": candidate.get("lora_base_model") or "未声明",
                "有效模式": modes.get(candidate.get("cleanse_mode"), "-"),
            }
            for label, value in candidate_rows.items():
                candidate_table.add_row(label, str(value))
            console.print(candidate_table)
            if data.get("cleanse_mode") != candidate.get("cleanse_mode"):
                console.print(
                    "[yellow]原请求为快速清洗，确认后将改用深度清洗。[/yellow]"
                )

        cleanse = result.get("cleanse") or {}

        # 根据任务终态给出失败原因、通过结论或产物入口。
        if status == "failed":
            console.print(
                "[bold red]失败原因[/bold red]  ",
                Text(str(data.get("error") or "未知错误"), style="red"),
            )
        elif result.get("action") == "passed":
            console.print("[bold green]处理结果：安全通过，无需执行清洗。[/bold green]")
        elif cleanse:
            cleanse_table = Table("项目", "内容", title="清洗结果", **table_style)
            cleanse_table.add_row("清洗模式", modes.get(cleanse.get("mode"), "-"))
            cleanse_table.add_row("签名", str(cleanse.get("signature") or "动态提取"))
            cleanse_table.add_row(
                "抑制参数", f"{int(cleanse.get('suppressed_count', 0)):,}"
            )
            cleanse_table.add_row("审计报告", f"report {job_id}")
            cleanse_table.add_row("清洗模型", f"artifact {job_id}")
            console.print(cleanse_table)
        else:
            console.print("[yellow]任务尚未结束，可稍后再次执行 show。[/yellow]")


# =====================================================================
# 服务器状态
# =====================================================================
@app.command()
def health(
    server: str | None = typer.Argument(
        None, help="公开健康检查地址；省略时使用当前会话"
    ),
    json_output: bool = typer.Option(False, "--json", help="输出原始 JSON"),
):
    """检查服务器检测与清洗能力是否就绪。"""

    # -----------------------------------------------------------------
    # 步骤 1：请求公开健康接口
    # -----------------------------------------------------------------
    # 显式地址用于无 Token 探活，省略时复用当前会话。
    if server:
        with httpx.Client(base_url=server.rstrip("/"), timeout=10.0) as client:
            result = _request(client, "GET", "/health")
    else:
        with _client() as client:
            result = _request(client, "GET", "/health")

    # -----------------------------------------------------------------
    # 步骤 2：展示能力状态并返回探活退出码
    # -----------------------------------------------------------------
    _display("health", result, json_output)

    # degraded 使用非零退出码，便于脚本判断总体状态。
    if result.get("status") != "ready":
        raise typer.Exit(1)


# =====================================================================
# 模型与审计
# =====================================================================
@app.command("models")
def list_models():
    """查看服务器已经注册的基础模型。"""
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
    # 步骤 1：定位待扫描 LoRA
    # -----------------------------------------------------------------
    # 批量模式按标准权重文件递归识别 LoRA 目录。
    lora_path = lora_path.expanduser().resolve()
    lora_paths = [lora_path]
    if batch:
        lora_paths = sorted(
            {path.parent for path in lora_path.rglob("adapter_model.safetensors")}
        )
    if not lora_paths:
        typer.echo(f"目录中未发现 LoRA：{lora_path}")
        raise typer.Exit(1)

    # -----------------------------------------------------------------
    # 步骤 2：依次上传并扫描 LoRA
    # -----------------------------------------------------------------
    # results 保存每个 LoRA 的独立结果；批量项额外记录目录名。
    results = []

    # 同批次复用连接并顺序扫描，避免并发争用服务端检测器。
    with _client() as client:
        for index, current_path in enumerate(lora_paths, start=1):
            # JSON 模式保持标准输出纯净。
            if not json_output:
                typer.echo(
                    f"[{index}/{len(lora_paths)}] 正在扫描 {current_path.name}..."
                    if batch
                    else "正在上传 LoRA..."
                )

            try:
                # 批量请求使用非致命错误，单项失败不终止后续扫描。
                uploaded = _upload_lora(client, current_path, fatal=not batch)
                scan_result = _request(
                    client,
                    "POST",
                    "/v1/scan",
                    fatal=not batch,
                    json={"lora_id": uploaded["lora_id"]},
                )

                results.append(
                    {"name": current_path.name, **scan_result} if batch else scan_result
                )
            except RuntimeError as exc:
                results.append(
                    {"name": current_path.name, "error": str(exc) or "扫描失败"}
                )
                if not json_output:
                    typer.echo(f"扫描失败：{exc}")

    # -----------------------------------------------------------------
    # 步骤 3：汇总并归档扫描结果
    # -----------------------------------------------------------------
    if batch:
        # 批量归档包含总体计数和全部逐项结果。
        result = {
            "total": len(results),
            "safe": sum(item.get("verdict") == "safe" for item in results),
            "poisoned": sum(item.get("verdict") == "poisoned" for item in results),
            "failed": sum("error" in item for item in results),
            "items": results,
        }

        output = output or Path(f"scan_archive_{time.strftime('%Y%m%d_%H%M%S')}.json")
    else:
        result = results[0]

    # 单项和批量归档均使用 UTF-8 JSON。
    if output is not None:
        output = output.expanduser().resolve()
        output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )

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
    revision: str | None = typer.Option(None, "--revision"),
    mode: Literal["fast", "deep"] = typer.Option("fast", "--mode"),
    wait: bool = typer.Option(True, "--wait/--no-wait"),
):
    """上传 LoRA，创建审计任务并按需等待完成。"""

    # -----------------------------------------------------------------
    # 步骤 1：定位待审计 LoRA
    # -----------------------------------------------------------------
    lora_path = lora_path.expanduser().resolve()

    # -----------------------------------------------------------------
    # 步骤 2：上传 LoRA 并创建后台任务
    # -----------------------------------------------------------------
    typer.echo("正在上传 LoRA...")
    with _client() as client:
        uploaded = _upload_lora(client, lora_path)

        # revision 未提供时不发送 null，保持服务端默认解析语义。
        payload = {
            "model_id": model,
            "lora_id": uploaded["lora_id"],
            "cleanse_mode": mode,
        }
        if revision is not None:
            payload["model_revision"] = revision

        # job 始终保存同一任务的最新服务端快照。
        job = _request(client, "POST", "/v1/jobs", json=payload)
        typer.echo(f"任务已创建：{job['job_id']}")
        if not wait:
            return

        # -----------------------------------------------------------------
        # 步骤 3：轮询任务并只输出变化的阶段
        # -----------------------------------------------------------------
        # last_stage 避免每两秒重复打印相同状态。
        last_stage = None
        while job["status"] in ("queued", "running", "awaiting_confirmation"):
            job = _request(client, "GET", f"/v1/jobs/{job['job_id']}")
            if job.get("stage") != last_stage:
                typer.echo(f"当前阶段：{job.get('stage', '-')}")
                last_stage = job.get("stage")
            if job["status"] == "awaiting_confirmation":
                # 先展示候选信息；非交互终端只提示 confirm 命令，不自动接受。
                _display("show", job)
                if not sys.stdin.isatty():
                    typer.echo(
                        f"当前终端不可交互；请执行 aegis confirm {job['job_id']} --accept 或 --reject。"
                    )
                    return
                confirmed = typer.confirm("是否确认下载并使用该社区模型？")
                job = _request(
                    client,
                    "POST",
                    f"/v1/jobs/{job['job_id']}/model-confirmation",
                    json={"confirmed": confirmed},
                )
                if not confirmed:
                    break
            if job["status"] in ("queued", "running"):
                time.sleep(2)

    # -----------------------------------------------------------------
    # 步骤 4：展示最终检测、清洗或失败结果
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
    with _client() as client:
        result = _request(client, "GET", "/v1/jobs", params={"limit": limit})

    # 默认和 --json 都使用列表摘要；完整字段由 show 查询。
    _display("jobs", result, json_output)


@app.command()
def show(
    job_id: str,
    json_output: bool = typer.Option(False, "--json", help="输出原始 JSON"),
):
    """查看单个任务的完整状态和结果。"""
    with _client() as client:
        _display("show", _request(client, "GET", f"/v1/jobs/{job_id}"), json_output)


@app.command()
def confirm(
    job_id: str,
    accept: bool = typer.Option(False, "--accept"),
    reject: bool = typer.Option(False, "--reject"),
):
    """接受或拒绝待确认任务的社区模型。"""
    # accept/reject 必须互斥，拒绝也作用于原任务，不创建新任务。
    if accept == reject:
        raise typer.BadParameter("必须且只能指定 --accept 或 --reject。")
    with _client() as client:
        result = _request(
            client,
            "POST",
            f"/v1/jobs/{job_id}/model-confirmation",
            json={"confirmed": accept},
        )
    _display("show", result)


@app.command()
def report(
    job_id: str,
    report_format: Literal["html", "json"] = typer.Option("html", "--format"),
    output: Path | None = typer.Option(None, "--output", "-o"),
):
    """下载任务生成的 HTML 或 JSON 审计报告。"""
    # 默认文件名包含任务编号，output 传给统一请求层后启用流式下载。
    output = output or Path(f"{job_id}_report.{report_format}")

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
    # 清洗产物固定为 ZIP，默认名称包含对应任务编号。
    output = output or Path(f"{job_id}_cleaned_lora.zip")
    with _client() as client:
        _request(client, "GET", f"/v1/jobs/{job_id}/artifact", output=output)


if __name__ == "__main__":
    # 无论由启动器还是模块方式运行，帮助信息统一展示 aegis 命令。
    app(prog_name="aegis")
