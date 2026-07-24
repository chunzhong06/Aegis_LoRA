# Aegis-LoRA - API 服务模块
# 负责服务初始化、远程鉴权、健康检查、LoRA 上传和任务接口编排。
import hmac
import json
import os
import shutil
from datetime import datetime, timezone
from typing import Literal
from uuid import uuid4

from fastapi import APIRouter, Depends, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from . import api_jobs as runtime
from .model_registry import is_community_model_id

# =====================================================================
# 配置与状态
# =====================================================================
# api_jobs 统一维护工程路径、模型注册表、任务状态和后台执行器。
ROOT = runtime.ROOT

# API 版本同时用于服务元信息和客户端登录响应。
VERSION = "0.3.0"


# =====================================================================
# 请求与响应
# =====================================================================
# 健康检查按认证、算法资源、模型和存储拆分就绪状态。
class HealthResponse(BaseModel):
    service: str
    version: str
    status: Literal["ready", "degraded"]
    auth_ready: bool
    detector_ready: bool
    models_ready: dict[str, bool]
    fast_cleanse_ready: dict[str, bool]
    deep_cleanse_ready: bool
    storage_ready: bool


# 模型列表只暴露稳定标识，不向客户端泄露服务器绝对路径。
class ModelResponse(BaseModel):
    model_id: str
    name: str
    family: str
    ready: bool


# 上传结果只返回后续扫描和审计使用的临时资源编号。
class LoraUploadResponse(BaseModel):
    lora_id: str


# 单独检测只接收已上传资源编号，不允许客户端传入任意服务器路径。
class ScanRequest(BaseModel):
    lora_id: str = Field(pattern=r"^lora-[0-9a-f]{12}$", description="LoRA 资源编号")


# 检测响应与 pipeline 的 return_details 结构保持一致。
class ScanResponse(BaseModel):
    verdict: Literal["safe", "poisoned"]
    is_poisoned: bool
    risk_score: float
    threshold: float
    detector: str
    elapsed_seconds: float


# 审计请求同时确定基础模型、上传资源、清洗模式和可选 revision。
class AuditRequest(BaseModel):
    model_id: str = Field(min_length=1, description="服务器基础模型编号")
    lora_id: str = Field(pattern=r"^lora-[0-9a-f]{12}$", description="LoRA 资源编号")
    cleanse_mode: Literal["fast", "deep"] = "fast"
    model_revision: str | None = Field(
        default=None,
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._/-]*$",
        description="ModelScope revision",
    )


class ModelConfirmationRequest(BaseModel):
    confirmed: bool


# 创建接口只确认任务进入队列，最终结果通过 status_url 轮询。
class JobCreatedResponse(BaseModel):
    job_id: str
    status: Literal["queued"]
    stage: Literal["等待"]
    status_url: str


# =====================================================================
# 核心逻辑
# =====================================================================
# 关闭 HTTPBearer 自动错误，由统一依赖区分服务未配置和凭证无效。
bearer = HTTPBearer(auto_error=False)


def _require_token(credentials: HTTPAuthorizationCredentials | None = Depends(bearer)):
    """验证业务接口使用的 Bearer Token。"""
    # Token 每次从环境读取，不进入模块状态、任务记录或日志。
    token = os.getenv("AEGIS_API_TOKEN", "").strip()

    if not token:
        raise HTTPException(status_code=503, detail="服务器尚未配置 AEGIS_API_TOKEN。")

    # compare_digest 避免普通字符串比较的时序差异。
    if (
        credentials is None
        or credentials.scheme.lower() != "bearer"
        or not hmac.compare_digest(credentials.credentials, token)
    ):
        raise HTTPException(
            status_code=401,
            detail="访问令牌无效。",
            headers={"WWW-Authenticate": "Bearer"},
        )


# =====================================================================
# API 接口
# =====================================================================
# FastAPI 实例保存服务元信息；业务接口统一注册到受保护的 v1 路由。
app = FastAPI(
    title="Aegis-LoRA API",
    version=VERSION,
    description="面向 LoRA 适配器的远程检测、清洗与审计服务。",
)

# /health 保持公开供部署探活，/v1 共享 Token 依赖。
v1 = APIRouter(prefix="/v1", dependencies=[Depends(_require_token)])


@app.get("/health", summary="健康检查", response_model=HealthResponse)
def health():
    """汇总认证、算法资源、模型与存储的服务就绪状态。"""
    # -----------------------------------------------------------------
    # 步骤 1：检查认证、算法资源和注册模型
    # -----------------------------------------------------------------
    auth_ready = bool(os.getenv("AEGIS_API_TOKEN", "").strip())
    detector_ready = (
        ROOT / "models" / "detectors" / "spectral_detector_llama.pkl"
    ).is_file()
    recovery_ready = (ROOT / "datasets" / "clean_data_recovery.json").is_file()

    # ready 不持久化，每次根据模型目录中的 config.json 动态计算。
    models_ready = {
        model_id: (model["path"] / "config.json").is_file()
        for model_id, model in runtime.MODELS.items()
    }

    # 快速清洗要求模型明确登记签名，family 仅作为能力展示分组。
    fast_ready = {}
    for model in runtime.MODELS.values():
        signature = model.get("signature")
        ready = signature is not None and signature.is_file() and recovery_ready
        fast_ready[model["family"]] = fast_ready.get(model["family"], False) or ready

    deep_ready = (
        ROOT / "datasets" / "clean_data_variants.json"
    ).is_file() and recovery_ready

    # -----------------------------------------------------------------
    # 步骤 2：检查运行目录并汇总服务状态
    # -----------------------------------------------------------------
    storage_ready = all(
        (ROOT / ".cache" / directory).is_dir()
        for directory in ("uploads", "api_reports", "artifacts")
    )

    ready = (
        auth_ready
        and detector_ready
        and all(models_ready.values())
        and all(fast_ready.values())
        and deep_ready
        and storage_ready
    )

    # 分项状态用于定位降级原因，ready 是部署探活的总判定。
    return {
        "service": "Aegis-LoRA API",
        "version": VERSION,
        "status": "ready" if ready else "degraded",
        "auth_ready": auth_ready,
        "detector_ready": detector_ready,
        "models_ready": models_ready,
        "fast_cleanse_ready": fast_ready,
        "deep_cleanse_ready": deep_ready,
        "storage_ready": storage_ready,
    }


@v1.get("/me", summary="验证登录")
def get_current_client():
    """返回已通过鉴权的当前服务身份。"""
    # 路由级依赖已完成鉴权，此处只返回当前服务身份。
    return {"authenticated": True, "service": "Aegis-LoRA API", "version": VERSION}


@v1.get("/models", summary="查看模型", response_model=list[ModelResponse])
def list_models():
    """列出已注册模型及其动态就绪状态。"""
    # 模型路径不对外暴露，ready 仍按磁盘状态动态计算。
    return [
        {
            "model_id": model_id,
            "name": model["name"],
            "family": model["family"],
            "ready": (model["path"] / "config.json").is_file(),
        }
        for model_id, model in runtime.MODELS.items()
    ]


@v1.post("/loras", summary="上传 LoRA", response_model=LoraUploadResponse)
async def upload_lora(
    weights: UploadFile = File(..., description="adapter_model.safetensors"),
    config: UploadFile = File(..., description="adapter_config.json"),
):
    """校验并保存上传的 LoRA 配置与 safetensors 权重。"""
    # -----------------------------------------------------------------
    # 步骤 1：创建本次上传的隔离目录
    # -----------------------------------------------------------------
    # lora_id 同时作为上传目录名和后续任务引用的临时资源编号。
    lora_id = f"lora-{uuid4().hex[:12]}"
    lora_dir = ROOT / ".cache" / "uploads" / lora_id
    lora_dir.mkdir(parents=True)

    # 上传上限就近定义：配置 1 MiB，权重 2 GiB。
    max_config = 1024 * 1024
    max_weights = 2 * 1024 * 1024 * 1024

    try:
        # -----------------------------------------------------------------
        # 步骤 2：限制并解析 adapter_config.json
        # -----------------------------------------------------------------
        # 多读一个字节即可判断配置是否超限。
        config_bytes = await config.read(max_config + 1)
        if len(config_bytes) > max_config:
            raise HTTPException(
                status_code=413, detail="adapter_config.json 体积过大。"
            )

        # 同时验证 UTF-8、JSON 语法和对象根节点。
        try:
            config_data = json.loads(config_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise HTTPException(
                status_code=400, detail="adapter_config.json 内容无效。"
            ) from exc

        if not isinstance(config_data, dict):
            raise HTTPException(
                status_code=400, detail="adapter_config.json 必须是对象。"
            )

        # -----------------------------------------------------------------
        # 步骤 3：限制体积并分块写入权重
        # -----------------------------------------------------------------
        # total_size 只统计已接收权重，超限分块不会写入磁盘。
        total_size = 0

        # 1 MiB 分块使大文件上传保持稳定内存占用。
        weights_path = lora_dir / "adapter_model.safetensors"
        with weights_path.open("wb") as file:
            while chunk := await weights.read(1024 * 1024):
                total_size += len(chunk)

                if total_size > max_weights:
                    raise HTTPException(
                        status_code=413, detail="LoRA 权重超过上传限制。"
                    )
                file.write(chunk)

        if total_size == 0:
            raise HTTPException(status_code=400, detail="LoRA 权重文件为空。")

        # -----------------------------------------------------------------
        # 步骤 4：验证 safetensors 实际内容
        # -----------------------------------------------------------------
        # 文件名不作为可信依据；safe_open 只解析文件头和张量索引。
        from safetensors import SafetensorError, safe_open

        try:
            with safe_open(str(weights_path), framework="pt", device="cpu") as tensors:
                if not tensors.keys():
                    raise ValueError("权重文件不包含张量。")
        except (SafetensorError, ValueError) as exc:
            raise HTTPException(
                status_code=400, detail="LoRA 权重不是有效的 safetensors 文件。"
            ) from exc

        # -----------------------------------------------------------------
        # 步骤 5：保存配置并返回 LoRA 资源信息
        # -----------------------------------------------------------------
        # 重新序列化已验证配置，统一使用服务端标准文件名。
        (lora_dir / "adapter_config.json").write_text(
            json.dumps(config_data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        # 客户端只取得 lora_id，服务器路径不进入响应。
        return {"lora_id": lora_id}

    # 任一失败都删除本次隔离目录；已知输入错误保留原状态码。
    except HTTPException:
        shutil.rmtree(lora_dir, ignore_errors=True)
        raise

    except Exception as exc:
        shutil.rmtree(lora_dir, ignore_errors=True)
        print(f"      [错误] LoRA 上传失败: {exc}")
        raise HTTPException(
            status_code=500, detail="LoRA 上传失败，请查看服务日志。"
        ) from exc


@v1.post("/scan", summary="单独检测", response_model=ScanResponse)
def scan(request: ScanRequest):
    """对已上传 LoRA 执行一次静态检测并回收资源。"""
    # -----------------------------------------------------------------
    # 步骤 1：解析本次扫描独占的上传资源
    # -----------------------------------------------------------------
    # 接口只接受服务器上传编号，不接受任意本地路径。
    lora_path = runtime.resolve_lora(request.lora_id)

    # -----------------------------------------------------------------
    # 步骤 2：执行检测并在响应生成前回收资源
    # -----------------------------------------------------------------
    try:
        return runtime.scan_lora(lora_path)
    finally:
        # 单独扫描一次性消费上传，结束后立即回收。
        shutil.rmtree(lora_path, ignore_errors=True)


@v1.post(
    "/jobs", summary="创建任务", status_code=202, response_model=JobCreatedResponse
)
def create_job(request: AuditRequest):
    """校验审计请求并创建持久化后台任务。"""
    # -----------------------------------------------------------------
    # 步骤 1：确认 LoRA、检测器和已注册模型资源可用
    # -----------------------------------------------------------------
    # 任务入队前接口拥有上传资源；准入失败由当前请求回收。
    lora_path = runtime.resolve_lora(request.lora_id)
    try:
        if request.model_revision and ".." in request.model_revision:
            raise HTTPException(status_code=400, detail="模型 revision 不合法。")

        # 社区模型只有 revision 匹配时才视为已注册。
        model = runtime.registered_model(
            runtime.MODELS,
            request.model_id,
            request.model_revision,
        )
        if model is None:
            if not is_community_model_id(request.model_id):
                raise HTTPException(
                    status_code=400,
                    detail="未注册模型必须使用精确的 ModelScope owner/model ID。",
                )

        # 未注册模型此时只检查 ID、上传和检测器，不访问 ModelScope。
        if not (
            ROOT / "models" / "detectors" / "spectral_detector_llama.pkl"
        ).is_file():
            raise HTTPException(status_code=503, detail="静态检测器尚未就绪。")

        if model is not None:
            # 已注册模型沿用现有清洗资源检查。
            if not (model["path"] / "config.json").is_file():
                raise HTTPException(status_code=503, detail="基础模型尚未就绪。")
            if not (ROOT / "datasets" / "clean_data_recovery.json").is_file():
                raise HTTPException(status_code=503, detail="清洗恢复数据尚未就绪。")
            signature = model.get("signature")
            if request.cleanse_mode == "fast" and (
                signature is None or not signature.is_file()
            ):
                detail = (
                    "该社区模型未登记专属快速签名，请显式使用 deep 模式。"
                    if model["source"] == "community"
                    else f"{model['family']} 快速清洗签名尚未就绪。"
                )
                raise HTTPException(status_code=503, detail=detail)
            if (
                request.cleanse_mode == "deep"
                and not (ROOT / "datasets" / "clean_data_variants.json").is_file()
            ):
                raise HTTPException(
                    status_code=503, detail="深度清洗变体数据尚未就绪。"
                )
    except HTTPException:
        shutil.rmtree(lora_path, ignore_errors=True)
        raise

    # -----------------------------------------------------------------
    # 步骤 2：创建可持久化的等待任务
    # -----------------------------------------------------------------
    # job_id 同时标识任务状态、归档报告和清洗产物。
    job_id = f"audit-{uuid4().hex[:12]}"

    # 初始任务只保存已产生字段，model_revision 仅在用户传入时写入。
    job = {
        "job_id": job_id,
        "status": "queued",
        "stage": "等待",
        "model_id": request.model_id,
        "lora_id": request.lora_id,
        "cleanse_mode": request.cleanse_mode,
        "submitted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    if request.model_revision is not None:
        job["model_revision"] = request.model_revision
    with runtime.job_lock:
        runtime.jobs[job_id] = job

    # -----------------------------------------------------------------
    # 步骤 3：先保存等待状态，再提交后台执行器
    # -----------------------------------------------------------------
    try:
        # 先落盘再入队，使意外退出后的 queued 状态可恢复为中断。
        runtime.sync_jobs()

        runtime.executor.submit(runtime.run_job, job_id)
    except (OSError, TypeError) as exc:
        # 状态未可靠保存时撤销任务和上传。
        with runtime.job_lock:
            runtime.jobs.pop(job_id, None)

        shutil.rmtree(lora_path, ignore_errors=True)
        raise HTTPException(status_code=500, detail="任务状态保存失败。") from exc
    except RuntimeError as exc:
        # 执行器不可用时撤销已持久化的排队任务。
        with runtime.job_lock:
            runtime.jobs.pop(job_id, None)

        try:
            runtime.sync_jobs()
        except (OSError, TypeError) as sync_exc:
            print(f"      [警告] 队列失败状态同步失败: {sync_exc}")

        shutil.rmtree(lora_path, ignore_errors=True)
        raise HTTPException(status_code=503, detail="后台任务队列不可用。") from exc

    # 202 只表示已入队，最终状态由 status_url 查询。
    return {
        "job_id": job_id,
        "status": "queued",
        "stage": "等待",
        "status_url": f"/v1/jobs/{job_id}",
    }


@v1.post("/jobs/{job_id}/model-confirmation", summary="确认社区模型")
def confirm_model(job_id: str, request: ModelConfirmationRequest):
    """确认或拒绝待下载的社区模型并推进任务状态。"""
    # -----------------------------------------------------------------
    # 步骤 1：兑现超时并校验待确认任务
    # -----------------------------------------------------------------
    if runtime.expire_confirmations():
        try:
            runtime.sync_jobs()
        except (OSError, TypeError) as exc:
            print(f"      [警告] 确认超时状态同步失败: {exc}")

    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with runtime.job_lock:
        job = runtime.jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="任务不存在。")
        if job.get("status") != "awaiting_confirmation":
            raise HTTPException(
                status_code=409,
                detail="任务不在待确认状态，可能已经确认、结束或超时。",
            )
        candidate = job.get("model_candidate")
        if not isinstance(candidate, dict):
            raise HTTPException(status_code=409, detail="任务缺少候选模型信息。")

        # previous 用于持久化失败时恢复完整待确认状态。
        previous = dict(job)
        if request.confirmed:
            job.update(
                status="queued",
                stage="等待",
                confirmed_at=now,
                cleanse_mode=candidate["cleanse_mode"],
            )
        else:
            job.update(
                status="failed",
                stage="未确认",
                finished_at=now,
                error="用户未确认社区模型下载。",
            )
        job.pop("confirmation_deadline", None)

    # -----------------------------------------------------------------
    # 步骤 2：持久化确认结果
    # -----------------------------------------------------------------
    try:
        runtime.sync_jobs()
    except (OSError, TypeError) as exc:
        with runtime.job_lock:
            runtime.jobs[job_id] = previous
        raise HTTPException(status_code=500, detail="任务状态保存失败。") from exc

    # 拒绝后直接终止并回收上传，不创建第二个任务。
    if not request.confirmed:
        shutil.rmtree(
            ROOT / ".cache" / "uploads" / previous["lora_id"],
            ignore_errors=True,
        )
        with runtime.job_lock:
            return dict(runtime.jobs[job_id])

    # -----------------------------------------------------------------
    # 步骤 3：接受后将同一任务重新提交单线程执行器
    # -----------------------------------------------------------------
    try:
        runtime.executor.submit(runtime.run_job, job_id)
    except RuntimeError as exc:
        with runtime.job_lock:
            runtime.jobs[job_id].update(
                status="failed",
                stage="失败",
                finished_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                error="后台任务队列不可用。",
            )
        shutil.rmtree(
            ROOT / ".cache" / "uploads" / previous["lora_id"],
            ignore_errors=True,
        )
        try:
            runtime.sync_jobs()
        except (OSError, TypeError) as sync_exc:
            print(f"      [警告] 队列失败状态同步失败: {sync_exc}")
        raise HTTPException(status_code=503, detail="后台任务队列不可用。") from exc

    with runtime.job_lock:
        return dict(runtime.jobs[job_id])


@v1.get("/jobs", summary="查看任务")
def list_jobs(limit: int = Query(default=20, ge=1, le=100)):
    """按提交时间倒序返回受限数量的任务摘要。"""
    if runtime.expire_confirmations():
        runtime.sync_jobs()

    # 列表只返回定位和判断状态所需字段，完整信息由详情接口提供。
    summary_fields = (
        "job_id",
        "status",
        "stage",
        "model_id",
        "cleanse_mode",
        "submitted_at",
    )
    with runtime.job_lock:
        job_items = [
            {field: job.get(field) for field in summary_fields}
            for job in runtime.jobs.values()
        ]

    # ISO 8601 字符串可直接按时间倒序。
    job_items.sort(key=lambda job: job.get("submitted_at") or "", reverse=True)

    # total 表示任务总数，items 受 limit 限制。
    return {"total": len(job_items), "items": job_items[:limit]}


@v1.get("/jobs/{job_id}", summary="查询任务")
def get_job(job_id: str):
    """返回指定任务的完整状态快照。"""
    if runtime.expire_confirmations():
        runtime.sync_jobs()

    # 在锁内复制完整快照，响应序列化不持有共享状态。
    with runtime.job_lock:
        job = runtime.jobs.get(job_id)

        if job is None:
            raise HTTPException(status_code=404, detail="任务不存在。")

        return dict(job)


@v1.get("/jobs/{job_id}/report", summary="下载报告")
def get_report(job_id: str, report_format: Literal["html", "json"] = "html"):
    """校验任务结果并返回指定格式的审计报告。"""
    # 先复制任务快照，再同时校验任务结果和实际归档文件。
    with runtime.job_lock:
        job = dict(runtime.jobs[job_id]) if job_id in runtime.jobs else None

    if job is None:
        raise HTTPException(status_code=404, detail="任务不存在。")

    if job.get("status") in ("queued", "running", "awaiting_confirmation"):
        raise HTTPException(status_code=409, detail="任务尚未完成。")

    # 安全直通和失败任务均不生成清洗报告。
    if (
        job.get("status") != "succeeded"
        or (job.get("result") or {}).get("action") != "cleaned"
    ):
        raise HTTPException(status_code=404, detail="该任务没有审计报告。")

    # report_format 已由 Literal 限制，只能定位 HTML 或 JSON。
    report_path = ROOT / ".cache" / "api_reports" / f"{job_id}.{report_format}"

    if not report_path.is_file():
        raise HTTPException(status_code=404, detail="审计报告文件不存在。")

    media_type = "text/html" if report_format == "html" else "application/json"
    return FileResponse(
        path=str(report_path),
        media_type=media_type,
        filename=f"{job_id}_audit_report.{report_format}",
    )


@v1.get("/jobs/{job_id}/artifact", summary="下载清洗模型")
def get_artifact(job_id: str):
    """校验任务结果并返回清洗后的 LoRA 压缩产物。"""
    # 任务状态和实际 ZIP 双重校验，避免返回过期或丢失产物。
    with runtime.job_lock:
        job = dict(runtime.jobs[job_id]) if job_id in runtime.jobs else None

    if job is None:
        raise HTTPException(status_code=404, detail="任务不存在。")

    # 安全直通和失败任务均没有清洗产物。
    if (
        job.get("status") != "succeeded"
        or (job.get("result") or {}).get("action") != "cleaned"
    ):
        raise HTTPException(status_code=404, detail="该任务没有清洗产物。")

    artifact_path = ROOT / ".cache" / "artifacts" / f"{job_id}.zip"

    if not artifact_path.is_file():
        raise HTTPException(status_code=404, detail="清洗产物不存在。")
    return FileResponse(
        path=str(artifact_path),
        media_type="application/zip",
        filename=f"{job_id}_cleaned_lora.zip",
    )


# 注册统一 Token 保护的 /v1 业务路由，公开健康检查不受影响。
app.include_router(v1)
