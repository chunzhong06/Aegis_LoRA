# Aegis-LoRA - 命令行客户端
import json
import time
from pathlib import Path

import httpx
import typer

# =====================================================================
# 客户端配置
# =====================================================================
# Typer 负责命令注册、参数解析和帮助信息生成；未传命令时直接展示帮助。
app = typer.Typer(
    name="aegis",
    help="Aegis-LoRA 远程检测与清洗客户端",
    no_args_is_help=True,
)

# 登录配置保存在用户目录，与工程目录和当前终端位置无关。
CONFIG_FILE = Path.home() / ".aegis" / "config.json"


# =====================================================================
# 核心逻辑
# =====================================================================
def _client() -> httpx.Client:
    """读取本地登录配置并创建长耗时 HTTP 客户端。"""
    # -----------------------------------------------------------------
    # 步骤 1：确认客户端已经完成登录
    # -----------------------------------------------------------------
    if not CONFIG_FILE.is_file():
        typer.echo("尚未登录，请先执行 login。")
        raise typer.Exit(1)

    try:
        # -----------------------------------------------------------------
        # 步骤 2：读取并验证服务器地址与 Token
        # -----------------------------------------------------------------
        # 配置必须是包含 server 和 token 的 JSON 对象，缺失字段统一视为损坏。
        config = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
        server = config["server"]
        token = config["token"]

        # 空字符串或非字符串会导致 HTTP 客户端产生不明确错误，因此提前拒绝。
        if (
            not isinstance(server, str)
            or not server
            or not isinstance(token, str)
            or not token
        ):
            raise ValueError

        # -----------------------------------------------------------------
        # 步骤 3：创建适合长耗时审计任务的 HTTP 客户端
        # -----------------------------------------------------------------
        # 连接阶段限制为 10 秒，读取阶段不设总超时，避免深度清洗轮询被中断。
        return httpx.Client(
            base_url=server,
            headers={"Authorization": f"Bearer {token}"},
            timeout=httpx.Timeout(None, connect=10.0),
        )
    except (OSError, ValueError, KeyError, TypeError):
        # 文件读取、JSON 解析、字段缺失和字段类型错误使用同一修复提示。
        typer.echo("登录配置损坏，请重新登录。")
        raise typer.Exit(1)


def _request(
    client: httpx.Client,
    method: str,
    url: str,
    *,
    output: Path | None = None,
    **kwargs,
):
    """统一处理 JSON 请求、API 错误和流式文件下载。"""
    try:
        # -----------------------------------------------------------------
        # 步骤 1：根据 output 判断执行普通请求还是流式下载
        # -----------------------------------------------------------------
        if output is None:
            # 普通接口响应体较小，直接完成请求并在后续解析 JSON。
            response = client.request(method, url, **kwargs)
        else:
            # 下载路径先转换为绝对路径，并确保用户指定的父目录存在。
            output = output.expanduser().resolve()
            output.parent.mkdir(parents=True, exist_ok=True)

            # stream 模式逐块写入报告或 ZIP，避免大文件一次性进入内存。
            with client.stream(method, url, **kwargs) as response:
                if response.is_error:
                    # 错误响应需要先完整读取，退出流上下文后才能解析 detail。
                    response.read()
                else:
                    # 只有成功响应才创建文件，防止把 JSON 错误页保存成产物。
                    with output.open("wb") as file:
                        for chunk in response.iter_bytes():
                            file.write(chunk)
                    typer.echo(f"已保存：{output}")
                    return None
    except httpx.RequestError as exc:
        # DNS、连接拒绝、连接超时和传输中断都属于服务器连接问题。
        typer.echo(f"无法连接服务器：{exc}")
        raise typer.Exit(1)
    except OSError as exc:
        # 下载目录不可写或磁盘写入失败时给出本地文件错误。
        typer.echo(f"文件保存失败：{exc}")
        raise typer.Exit(1)

    # -----------------------------------------------------------------
    # 步骤 2：解析 API 返回的错误信息
    # -----------------------------------------------------------------
    if response.is_error:
        try:
            # FastAPI 标准错误使用 detail 字段；非标准响应回退到原始文本。
            data = response.json()
            detail = (
                data.get("detail", response.text)
                if isinstance(data, dict)
                else response.text
            )
        except ValueError:
            # 代理服务器等组件可能返回 HTML 或纯文本错误页。
            detail = response.text
        typer.echo(f"请求失败 [{response.status_code}]：{detail}")
        raise typer.Exit(1)

    # -----------------------------------------------------------------
    # 步骤 3：解析成功接口的 JSON 结果
    # -----------------------------------------------------------------
    try:
        return response.json()
    except ValueError:
        # 当前普通接口均约定返回 JSON，其他内容说明服务端契约异常。
        typer.echo("服务器返回了无效的 JSON 响应。")
        raise typer.Exit(1)


def _upload_lora(client: httpx.Client, lora_path: Path) -> str:
    """检查并上传标准 LoRA 权重与配置文件。"""
    # -----------------------------------------------------------------
    # 步骤 1：解析本地 LoRA 目录并检查标准文件
    # -----------------------------------------------------------------
    lora_path = lora_path.expanduser().resolve()
    weights_path = lora_path / "adapter_model.safetensors"
    config_path = lora_path / "adapter_config.json"

    # 权重和配置缺少任意一个都无法构成可加载的 PEFT 适配器。
    if not weights_path.is_file():
        typer.echo("缺少 adapter_model.safetensors。")
        raise typer.Exit(1)
    if not config_path.is_file():
        typer.echo("缺少 adapter_config.json。")
        raise typer.Exit(1)

    # -----------------------------------------------------------------
    # 步骤 2：以 multipart/form-data 上传权重和配置
    # -----------------------------------------------------------------
    typer.echo("正在上传 LoRA...")

    # 文件句柄只在请求期间保持打开，请求结束后由 with 自动关闭。
    with weights_path.open("rb") as weights_file, config_path.open("rb") as config_file:
        result = _request(
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

    # -----------------------------------------------------------------
    # 步骤 3：返回服务端生成的 LoRA 资源编号
    # -----------------------------------------------------------------
    # 后续扫描和审计只传递 lora_id，不再重复上传或暴露本地路径。
    typer.echo(f"上传完成：{result['lora_id']}")
    return result["lora_id"]


# =====================================================================
# 登录
# =====================================================================
@app.command()
def login(
    server: str = typer.Argument(..., help="API 地址"),
    token: str = typer.Option(
        ..., "--token", prompt=True, hide_input=True, help="访问令牌"
    ),
):
    """验证并保存服务器地址和 Token。"""
    # -----------------------------------------------------------------
    # 步骤 1：规范化并检查登录参数
    # -----------------------------------------------------------------
    # 去掉末尾斜杠，避免与各接口路径拼接后产生重复分隔符。
    server = server.rstrip("/")

    # 客户端只接受明确的 HTTP/HTTPS 地址，提前阻止无协议输入。
    if not server.startswith(("http://", "https://")):
        typer.echo("API 地址必须以 http:// 或 https:// 开头。")
        raise typer.Exit(1)
    if not token:
        typer.echo("访问令牌不能为空。")
        raise typer.Exit(1)

    # -----------------------------------------------------------------
    # 步骤 2：调用受保护接口验证服务器和 Token
    # -----------------------------------------------------------------
    # 此时配置尚未写入磁盘，验证失败不会覆盖原有登录信息。
    try:
        with httpx.Client(
            base_url=server,
            headers={"Authorization": f"Bearer {token}"},
            timeout=10.0,
        ) as client:
            _request(client, "GET", "/v1/me")
    except ValueError as exc:
        typer.echo(f"API 地址无效：{exc}")
        raise typer.Exit(1)

    # -----------------------------------------------------------------
    # 步骤 3：持久化已经验证通过的登录配置
    # -----------------------------------------------------------------
    try:
        # 首次登录时创建 ~/.aegis，配置使用 UTF-8 JSON 便于人工检查。
        CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
        CONFIG_FILE.write_text(
            json.dumps(
                {"server": server, "token": token}, ensure_ascii=False, indent=2
            ),
            encoding="utf-8",
        )
    except OSError as exc:
        # 目录权限或磁盘错误不会被误报为服务器登录失败。
        typer.echo(f"登录配置保存失败：{exc}")
        raise typer.Exit(1)

    typer.echo(f"登录成功：{server}")


@app.command()
def logout():
    """删除本地登录配置。"""
    try:
        # missing_ok 使重复退出保持幂等，不存在配置时仍视为成功。
        CONFIG_FILE.unlink(missing_ok=True)
    except OSError as exc:
        typer.echo(f"退出登录失败：{exc}")
        raise typer.Exit(1)
    typer.echo("已退出登录。")


# =====================================================================
# 模型与审计
# =====================================================================
@app.command("models")
def list_models():
    """查看服务器已经注册的基础模型。"""
    # 请求模型注册表；Token 和服务器地址统一由 _client 注入。
    with _client() as client:
        models = _request(client, "GET", "/v1/models")

    # 使用固定列宽输出稳定表格，ready 反映服务器模型文件是否完整。
    typer.echo(f"{'MODEL ID':<24} {'FAMILY':<12} STATUS")
    for model in models:
        typer.echo(
            f"{model['model_id']:<24} "
            f"{model['family']:<12} "
            f"{'ready' if model['ready'] else 'missing'}"
        )


@app.command()
def scan(lora_path: Path):
    """上传并单独检测 LoRA。"""
    # -----------------------------------------------------------------
    # 步骤 1：上传本地 LoRA 并取得服务器资源编号
    # -----------------------------------------------------------------
    with _client() as client:
        lora_id = _upload_lora(client, lora_path)

        # -----------------------------------------------------------------
        # 步骤 2：请求静态检测并读取结构化结果
        # -----------------------------------------------------------------
        result = _request(client, "POST", "/v1/scan", json={"lora_id": lora_id})

    # 保留完整风险分数、阈值和检测器信息，便于人工查看或复制。
    typer.echo(json.dumps(result, ensure_ascii=False, indent=2))


@app.command()
def audit(
    lora_path: Path,
    model: str = typer.Option(..., "--model", "-m"),
    mode: str = typer.Option("fast", "--mode"),
    wait: bool = typer.Option(True, "--wait/--no-wait"),
):
    """上传 LoRA，创建审计任务并按需等待完成。"""
    # -----------------------------------------------------------------
    # 步骤 1：检查清洗模式
    # -----------------------------------------------------------------
    # CLI 先提供直观错误，服务端仍会通过 Pydantic 再次校验该字段。
    if mode not in ("fast", "deep"):
        typer.echo("mode 只能是 fast 或 deep。")
        raise typer.Exit(1)

    # -----------------------------------------------------------------
    # 步骤 2：上传 LoRA 并创建后台审计任务
    # -----------------------------------------------------------------
    with _client() as client:
        lora_id = _upload_lora(client, lora_path)
        submitted = _request(
            client,
            "POST",
            "/v1/jobs",
            json={"model_id": model, "lora_id": lora_id, "cleanse_mode": mode},
        )
        job_id = submitted["job_id"]
        typer.echo(f"任务已创建：{job_id}")

        # --no-wait 只负责提交任务，用户可稍后通过 show 查询。
        if not wait:
            return

        # -----------------------------------------------------------------
        # 步骤 3：轮询任务并仅输出发生变化的阶段
        # -----------------------------------------------------------------
        last_stage = None
        while True:
            job = _request(client, "GET", f"/v1/jobs/{job_id}")

            # 阶段未变化时保持终端安静，避免每两秒重复打印相同状态。
            if job.get("stage") != last_stage:
                typer.echo(f"状态={job['status']} 阶段={job['stage']}")
                last_stage = job.get("stage")

            # queued 和 running 之外均为终态，成功或失败后停止轮询。
            if job["status"] not in ("queued", "running"):
                break

            # 两秒间隔兼顾状态响应速度和服务端查询压力。
            time.sleep(2)

    # -----------------------------------------------------------------
    # 步骤 4：输出最终任务结果
    # -----------------------------------------------------------------
    if job["status"] == "succeeded":
        typer.echo("任务执行完成。")
    else:
        typer.echo(f"任务失败：{job.get('error')}")


# =====================================================================
# 任务与产物
# =====================================================================
@app.command("jobs")
def list_jobs(limit: int = 20):
    """查看最近提交的任务。"""
    # limit 交由服务端限制在有效范围，并按提交时间倒序返回。
    with _client() as client:
        result = _request(client, "GET", "/v1/jobs", params={"limit": limit})

    # 列表只展示最常用字段，完整结果由 show 命令负责。
    typer.echo(f"{'JOB ID':<22} {'STATUS':<12} {'STAGE':<8} MODEL")
    for job in result["items"]:
        typer.echo(
            f"{job['job_id']:<22} "
            f"{job['status']:<12} "
            f"{job['stage']:<8} "
            f"{job.get('model_id', '-')}"
        )


@app.command()
def show(job_id: str):
    """查看单个任务的完整状态和结果。"""
    # 直接输出完整 JSON，保留 scan、cleanse、下载地址和错误信息。
    with _client() as client:
        job = _request(client, "GET", f"/v1/jobs/{job_id}")
    typer.echo(json.dumps(job, ensure_ascii=False, indent=2))


@app.command()
def report(
    job_id: str,
    report_format: str = typer.Option("html", "--format"),
    output: Path | None = typer.Option(None, "--output", "-o"),
):
    """下载任务生成的 HTML 或 JSON 审计报告。"""
    # CLI 提前限制格式，服务端仍会通过 Literal 再次校验查询参数。
    if report_format not in ("html", "json"):
        typer.echo("format 只能是 html 或 json。")
        raise typer.Exit(1)

    # 未指定路径时在当前目录生成带任务编号的默认文件名。
    output = output or Path(f"{job_id}_report.{report_format}")

    # output 参数触发 _request 的流式下载分支，不把报告整体读入内存。
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
    # 模型产物始终为 ZIP，默认名称包含 job_id 便于与任务对应。
    output = output or Path(f"{job_id}_cleaned_lora.zip")

    # 安全直通或失败任务没有产物，API 错误由 _request 统一显示。
    with _client() as client:
        _request(client, "GET", f"/v1/jobs/{job_id}/artifact", output=output)


if __name__ == "__main__":
    # 仅在直接运行 cli.py 时启动 Typer；作为模块导入时不会解析命令行。
    app()
