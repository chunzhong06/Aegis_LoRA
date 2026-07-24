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

# =====================================================================
# 配置与状态
# =====================================================================
# 运行模块集中维护工程路径、模型注册表、任务状态和后台执行器。
# API 层只保留稳定别名，使健康检查和路由代码不感知运行时内部结构。
ROOT = runtime.ROOT
MODELS = runtime.MODELS

# 版本号同时用于服务元信息和客户端登录响应。
VERSION = "0.3.0"


# =====================================================================
# 请求与响应
# =====================================================================
# 健康检查分别返回认证、模型、算法数据和存储目录的就绪状态。
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


# 审计请求同时确定基础模型、LoRA 资源和清洗强度。
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
# 关闭自动错误后，由 _require_token 统一区分未配置 Token 和凭证无效。
bearer = HTTPBearer(auto_error=False)


def _require_token(credentials: HTTPAuthorizationCredentials | None = Depends(bearer)):
    """验证业务接口使用的 Bearer Token。"""
    # 每次请求读取当前环境变量，避免为单个配置项保留模块级常量。
    token = os.getenv("AEGIS_API_TOKEN", "").strip()

    # 未配置 Token 时保持健康检查可用，但拒绝开放全部业务接口。
    if not token:
        raise HTTPException(status_code=503, detail="服务器尚未配置 AEGIS_API_TOKEN。")

    # compare_digest 避免普通字符串比较带来的时序差异。
    # 缺少凭证、认证方案错误或 Token 不匹配都统一返回 401。
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
# FastAPI 实例只保存服务元信息；具体业务接口统一注册到 v1 路由。
app = FastAPI(
    title="Aegis-LoRA API",
    version=VERSION,
    description="面向 LoRA 适配器的远程检测、清洗与审计服务。",
)

# /v1 下所有接口共享 Token 依赖，/health 则保持公开供部署系统探活。
v1 = APIRouter(prefix="/v1", dependencies=[Depends(_require_token)])


@app.get("/health", summary="健康检查", response_model=HealthResponse)
def health():
    # Token 在检查时直接读取环境变量，结果只保留为本次响应的局部状态。
    auth_ready = bool(os.getenv("AEGIS_API_TOKEN", "").strip())

    # 探测静态检测器文件，缺失时扫描接口无法工作。
    detector_ready = (
        ROOT / "models" / "detectors" / "spectral_detector_llama.pkl"
    ).is_file()

    # 恢复数据同时被快速和深度清洗使用，是两类清洗的共同依赖。
    recovery_ready = (ROOT / "datasets" / "clean_data_recovery.json").is_file()

    # 每个模型独立检查 config.json，客户端可据此选择实际可用的 model_id。
    models_ready = {
        model_id: (model["path"] / "config.json").is_file()
        for model_id, model in MODELS.items()
    }

    # 同系列只要至少一个模型登记了可用专属签名，即保留原有系列级能力展示。
    fast_ready = {}
    for model in MODELS.values():
        signature = model.get("signature")
        ready = signature is not None and signature.is_file() and recovery_ready
        fast_ready[model["family"]] = fast_ready.get(model["family"], False) or ready

    # 深度清洗不使用离线签名，但需要变体数据和恢复数据。
    deep_ready = (
        ROOT / "datasets" / "clean_data_variants.json"
    ).is_file() and recovery_ready

    # 三类运行目录都应在启动阶段创建成功，否则无法保存上传和产物。
    storage_ready = all(
        (ROOT / ".cache" / directory).is_dir()
        for directory in ("uploads", "api_reports", "artifacts")
    )

    # 当前部署要求所有注册资源就绪，任一缺失时服务状态降级。
    ready = (
        auth_ready
        and detector_ready
        and all(models_ready.values())
        and all(fast_ready.values())
        and deep_ready
        and storage_ready
    )

    # 保留各子项状态，不只返回总状态，便于调用方定位降级原因。
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
    # 能进入该函数说明路由级 Token 依赖已经验证通过。
    return {"authenticated": True, "service": "Aegis-LoRA API", "version": VERSION}


@v1.get("/models", summary="查看模型", response_model=list[ModelResponse])
def list_models():
    # 每次请求动态检查模型文件，不缓存可能已经变化的就绪状态。
    return [
        {
            "model_id": model_id,
            "name": model["name"],
            "family": model["family"],
            "ready": (model["path"] / "config.json").is_file(),
        }
        for model_id, model in MODELS.items()
    ]


@v1.post("/loras", summary="上传 LoRA", response_model=LoraUploadResponse)
async def upload_lora(
    weights: UploadFile = File(..., description="adapter_model.safetensors"),
    config: UploadFile = File(..., description="adapter_config.json"),
):
    # -----------------------------------------------------------------
    # 步骤 1：创建本次上传的隔离目录
    # -----------------------------------------------------------------
    # 随机资源编号同时作为目录名，避免不同客户端上传时互相覆盖。
    lora_id = f"lora-{uuid4().hex[:12]}"
    lora_dir = ROOT / ".cache" / "uploads" / lora_id
    lora_dir.mkdir(parents=True)

    # 配置限制为 1 MiB，LoRA 权重限制为 2 GiB；限制就近保留在上传流程。
    max_config = 1024 * 1024
    max_weights = 2 * 1024 * 1024 * 1024

    try:
        # -----------------------------------------------------------------
        # 步骤 2：限制并解析 adapter_config.json
        # -----------------------------------------------------------------
        # 多读取一个字节即可判断文件是否超限，无需把无界内容读入内存。
        config_bytes = await config.read(max_config + 1)
        if len(config_bytes) > max_config:
            raise HTTPException(
                status_code=413, detail="adapter_config.json 体积过大。"
            )

        # 同时验证 UTF-8 编码和 JSON 语法，异常统一映射为客户端输入错误。
        try:
            config_data = json.loads(config_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise HTTPException(
                status_code=400, detail="adapter_config.json 内容无效。"
            ) from exc

        # PEFT 配置的根节点必须是对象，数组和基础类型不具备配置语义。
        if not isinstance(config_data, dict):
            raise HTTPException(
                status_code=400, detail="adapter_config.json 必须是对象。"
            )

        # -----------------------------------------------------------------
        # 步骤 3：限制体积并分块写入权重
        # -----------------------------------------------------------------
        total_size = 0

        # 每次只读取 1 MiB，使大体积 LoRA 上传保持稳定内存占用。
        weights_path = lora_dir / "adapter_model.safetensors"
        with weights_path.open("wb") as file:
            while chunk := await weights.read(1024 * 1024):
                total_size += len(chunk)

                # 在写入越界分块前终止，异常出口会删除整个隔离目录。
                if total_size > max_weights:
                    raise HTTPException(
                        status_code=413, detail="LoRA 权重超过上传限制。"
                    )
                file.write(chunk)

        # 空权重不是有效适配器，即使文件可以正常写入也必须拒绝。
        if total_size == 0:
            raise HTTPException(status_code=400, detail="LoRA 权重文件为空。")

        # -----------------------------------------------------------------
        # 步骤 4：验证 safetensors 实际内容
        # -----------------------------------------------------------------
        # 文件名不参与可信判断，只解析文件头和张量索引，避免加载完整权重。
        from safetensors import SafetensorError, safe_open

        try:
            with safe_open(str(weights_path), framework="pt", device="cpu") as tensors:
                # 只有元数据而没有任何张量的文件不能构成有效 LoRA 权重。
                if not tensors.keys():
                    raise ValueError("权重文件不包含张量。")
        except (SafetensorError, ValueError) as exc:
            # 真实格式错误统一作为客户端输入问题返回，异常出口会删除上传目录。
            raise HTTPException(
                status_code=400, detail="LoRA 权重不是有效的 safetensors 文件。"
            ) from exc

        # -----------------------------------------------------------------
        # 步骤 5：保存配置并返回 LoRA 资源信息
        # -----------------------------------------------------------------
        # 重新序列化已验证配置，并统一使用服务端约定的标准文件名。
        (lora_dir / "adapter_config.json").write_text(
            json.dumps(config_data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        # 客户端后续只携带 lora_id，服务器路径不会出现在响应中。
        # 文件体积只参与本次上传限制，不再返回未被客户端消费的摘要信息。
        return {"lora_id": lora_id}
    # 已知 HTTP 错误保留原状态码，同时删除本次上传产生的不完整目录。
    except HTTPException:
        shutil.rmtree(lora_dir, ignore_errors=True)
        raise

    # 文件系统等未知异常记录服务日志，对外只返回统一错误信息。
    except Exception as exc:
        shutil.rmtree(lora_dir, ignore_errors=True)
        print(f"      [错误] LoRA 上传失败: {exc}")
        raise HTTPException(
            status_code=500, detail="LoRA 上传失败，请查看服务日志。"
        ) from exc


@v1.post("/scan", summary="单独检测", response_model=ScanResponse)
def scan(request: ScanRequest):
    # -----------------------------------------------------------------
    # 步骤 1：解析本次扫描独占的上传资源
    # -----------------------------------------------------------------
    # 解析服务器资源后复用统一检测逻辑，不直接接收客户端文件路径。
    lora_path = runtime.resolve_lora(request.lora_id)

    # -----------------------------------------------------------------
    # 步骤 2：执行检测并在响应生成前回收资源
    # -----------------------------------------------------------------
    try:
        # 检测结果已经转换为独立字典，删除磁盘文件不会影响响应内容。
        return runtime.scan_lora(lora_path)
    finally:
        # 单独扫描只消费一次上传资源，响应生成后立即回收服务器副本。
        shutil.rmtree(lora_path, ignore_errors=True)


@v1.post(
    "/jobs", summary="创建任务", status_code=202, response_model=JobCreatedResponse
)
def create_job(request: AuditRequest):
    # -----------------------------------------------------------------
    # 步骤 1：确认 LoRA、检测器和已注册模型资源可用
    # -----------------------------------------------------------------
    # 先取得本次任务独占的临时上传目录，准入失败时由当前接口负责回收。
    lora_path = runtime.resolve_lora(request.lora_id)
    try:
        if request.model_revision and ".." in request.model_revision:
            raise HTTPException(status_code=400, detail="模型 revision 不合法。")

        # 相同社区 ID 只有 revision 一致时才能直接复用，避免误用旧模型目录。
        model = runtime.registered_model(
            MODELS,
            request.model_id,
            request.model_revision,
        )
        if model is None:
            if not runtime.is_community_model_id(request.model_id):
                raise HTTPException(
                    status_code=400,
                    detail="未注册模型必须使用精确的 ModelScope owner/model ID。",
                )

        # 未注册社区模型此时只检查静态检测准入，不访问 ModelScope 或创建模型目录。
        if not (
            ROOT / "models" / "detectors" / "spectral_detector_llama.pkl"
        ).is_file():
            raise HTTPException(status_code=503, detail="静态检测器尚未就绪。")

        if model is not None:
            # 已注册模型继续执行原有模型、恢复数据和清洗模式就绪检查。
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
        # 任务未进入队列时由准入接口回收上传资源，客户端下次需重新上传。
        shutil.rmtree(lora_path, ignore_errors=True)
        raise

    # -----------------------------------------------------------------
    # 步骤 2：创建可持久化的等待任务
    # -----------------------------------------------------------------
    # job_id 是状态查询、报告归档和产物下载共同使用的稳定标识。
    job_id = f"audit-{uuid4().hex[:12]}"

    # 任务记录仅包含 JSON 可序列化字段，确保可以直接写入状态文件。
    with runtime.job_lock:
        runtime.jobs[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "stage": "等待",
            "model_id": request.model_id,
            "model_revision": request.model_revision,
            "lora_id": request.lora_id,
            "cleanse_mode": request.cleanse_mode,
            "submitted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "started_at": None,
            "finished_at": None,
            "scan_result": None,
            "model_candidate": None,
            "model_confirmed": False,
            "confirmation_deadline": None,
            "confirmed_at": None,
            "download": None,
            "result": None,
            "error": None,
        }

    # -----------------------------------------------------------------
    # 步骤 3：先保存等待状态，再提交后台执行器
    # -----------------------------------------------------------------
    try:
        # 先落盘再入队，服务意外退出时仍能把该任务恢复为中断状态。
        runtime.sync_jobs()

        # 后台函数只接收 job_id，执行参数统一从任务快照和模型注册表读取。
        runtime.executor.submit(runtime.run_job, job_id)
    except (OSError, TypeError) as exc:
        # 等待状态无法可靠保存时撤销内存记录，避免出现不可恢复任务。
        with runtime.job_lock:
            runtime.jobs.pop(job_id, None)

        # 任务尚未交给后台线程，原始上传必须由当前请求负责回收。
        shutil.rmtree(lora_path, ignore_errors=True)
        raise HTTPException(status_code=500, detail="任务状态保存失败。") from exc
    except RuntimeError as exc:
        # 执行器关闭或不可用时撤销任务，并把撤销结果重新写回状态文件。
        with runtime.job_lock:
            runtime.jobs.pop(job_id, None)

        # 撤销结果需要再次同步，确保磁盘中不保留无法执行的等待任务。
        try:
            runtime.sync_jobs()
        except (OSError, TypeError) as sync_exc:
            # 二次同步失败只记录日志，接口仍以队列不可用作为主要错误返回。
            print(f"      [警告] 队列失败状态同步失败: {sync_exc}")

        # 后台线程没有取得资源所有权，因此在返回错误前删除本次上传。
        shutil.rmtree(lora_path, ignore_errors=True)
        raise HTTPException(status_code=503, detail="后台任务队列不可用。") from exc

    # 202 响应只表示任务已入队，不代表检测或清洗已经完成。
    return {
        "job_id": job_id,
        "status": "queued",
        "stage": "等待",
        "status_url": f"/v1/jobs/{job_id}",
    }


@v1.post("/jobs/{job_id}/model-confirmation", summary="确认社区模型")
def confirm_model(job_id: str, request: ModelConfirmationRequest):
    # 查询或确认动作同时负责兑现已经到期的确认任务。
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
        previous = dict(job)
        if request.confirmed:
            job.update(
                status="queued",
                stage="等待",
                confirmed_at=now,
                model_confirmed=True,
                cleanse_mode=candidate["cleanse_mode"],
                finished_at=None,
                error=None,
            )
        else:
            job.update(
                status="failed",
                stage="未确认",
                confirmed_at=now,
                model_confirmed=False,
                finished_at=now,
                error="用户未确认社区模型下载。",
            )

    try:
        runtime.sync_jobs()
    except (OSError, TypeError) as exc:
        with runtime.job_lock:
            runtime.jobs[job_id] = previous
        raise HTTPException(status_code=500, detail="任务状态保存失败。") from exc

    if not request.confirmed:
        shutil.rmtree(
            ROOT / ".cache" / "uploads" / previous["lora_id"],
            ignore_errors=True,
        )
        with runtime.job_lock:
            return dict(runtime.jobs[job_id])

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
    if runtime.expire_confirmations():
        runtime.sync_jobs()

    # 锁内只复制快照，排序和截断在锁外完成。
    with runtime.job_lock:
        job_items = [dict(job) for job in runtime.jobs.values()]

    # ISO 8601 时间可以直接按字符串倒序，最新提交的任务排在最前面。
    job_items.sort(key=lambda job: job.get("submitted_at") or "", reverse=True)

    # total 返回全部任务数，items 只包含本次请求的数量上限。
    return {"total": len(job_items), "items": job_items[:limit]}


@v1.get("/jobs/{job_id}", summary="查询任务")
def get_job(job_id: str):
    if runtime.expire_confirmations():
        runtime.sync_jobs()

    # 返回副本，避免响应序列化过程持有或修改共享状态。
    with runtime.job_lock:
        job = runtime.jobs.get(job_id)

        # 查询不存在的任务时明确返回 404，而不是空对象。
        if job is None:
            raise HTTPException(status_code=404, detail="任务不存在。")

        # 副本包含完整阶段、时间、结果或错误信息，供 CLI show 和轮询使用。
        return dict(job)


@v1.get("/jobs/{job_id}/report", summary="下载报告")
def get_report(job_id: str, report_format: Literal["html", "json"] = "html"):
    # 先读取任务快照，再校验任务是否真正生成了清洗报告。
    with runtime.job_lock:
        job = dict(runtime.jobs[job_id]) if job_id in runtime.jobs else None

    # job_id 不存在和报告文件不存在分别处理，错误原因更明确。
    if job is None:
        raise HTTPException(status_code=404, detail="任务不存在。")

    # 正在执行的任务尚不能判断是否会生成报告，使用 409 表示状态冲突。
    if job.get("status") in ("queued", "running", "awaiting_confirmation"):
        raise HTTPException(status_code=409, detail="任务尚未完成。")
    # 只有实际执行过清洗且成功结束的任务才会生成报告。
    if (
        job.get("status") != "succeeded"
        or (job.get("result") or {}).get("action") != "cleaned"
    ):
        raise HTTPException(status_code=404, detail="该任务没有审计报告。")

    # report_format 已由 Literal 限制，只能定位同一任务的 HTML 或 JSON 文件。
    report_path = ROOT / ".cache" / "api_reports" / f"{job_id}.{report_format}"

    # 状态记录成功但归档文件丢失时，仍以实际磁盘状态为准。
    if not report_path.is_file():
        raise HTTPException(status_code=404, detail="审计报告文件不存在。")

    # 根据报告格式设置响应类型，使浏览器和 CLI 正确处理内容。
    media_type = "text/html" if report_format == "html" else "application/json"
    return FileResponse(
        path=str(report_path),
        media_type=media_type,
        filename=f"{job_id}_audit_report.{report_format}",
    )


@v1.get("/jobs/{job_id}/artifact", summary="下载清洗模型")
def get_artifact(job_id: str):
    # 只有成功完成清洗的任务才允许下载模型产物。
    with runtime.job_lock:
        job = dict(runtime.jobs[job_id]) if job_id in runtime.jobs else None

    # 先确认任务存在，再判断该任务是否具备清洗产物。
    if job is None:
        raise HTTPException(status_code=404, detail="任务不存在。")
    # 安全直通任务没有清洗产物，下载接口统一返回 404。
    if (
        job.get("status") != "succeeded"
        or (job.get("result") or {}).get("action") != "cleaned"
    ):
        raise HTTPException(status_code=404, detail="该任务没有清洗产物。")

    # 产物固定使用 job_id 命名，避免依赖算法层返回的临时路径。
    artifact_path = ROOT / ".cache" / "artifacts" / f"{job_id}.zip"

    # 数据库状态和实际文件双重校验，防止返回已经丢失的 ZIP。
    if not artifact_path.is_file():
        raise HTTPException(status_code=404, detail="清洗产物不存在。")
    return FileResponse(
        path=str(artifact_path),
        media_type="application/zip",
        filename=f"{job_id}_cleaned_lora.zip",
    )


# 注册统一 Token 保护的 /v1 业务路由，公开健康检查不受影响。
app.include_router(v1)
