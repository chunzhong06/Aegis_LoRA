# Aegis-LoRA - API 服务模块
# 负责服务初始化、远程鉴权、LoRA 上传、检测清洗、状态查询和产物下载。
import hashlib
import hmac
import json
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from shutil import copy2, make_archive, rmtree
from threading import Lock
from typing import Literal
from uuid import uuid4

from fastapi import APIRouter, Depends, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

# =====================================================================
# 配置与状态
# =====================================================================
# 工程根目录是所有服务端资源的统一定位基准，避免依赖启动命令所在目录。
ROOT = Path(__file__).resolve().parents[1]

# 基础模型由服务端统一注册。
# path 用于实际加载模型，signature 只在快速清洗模式下使用。
MODELS = {
    "llama-3.2-3b": {
        "name": "Llama 3.2 3B Instruct",
        "family": "llama",
        "path": ROOT / "models" / "Llama-3.2-3B-Instruct",
        "signature": ROOT / "datasets" / "llama_multidomain_signatures.pt",
    },
    "qwen2.5-3b": {
        "name": "Qwen 2.5 3B Instruct",
        "family": "qwen",
        "path": ROOT / "models" / "Qwen2.5-3B-Instruct",
        "signature": ROOT / "datasets" / "qwen_multidomain_signatures.pt",
    },
    "deepseek-r1-1.5b": {
        "name": "DeepSeek R1 Distill Qwen 1.5B",
        "family": "deepseek",
        "path": ROOT / "models" / "DeepSeek-R1-Distill-Qwen-1.5B",
        "signature": ROOT / "datasets" / "deepseek_multidomain_signatures.pt",
    },
}

# 版本号同时用于服务元信息和客户端登录响应。
VERSION = "0.3.0"

# API 运行目录统一放在 .cache，路径在使用位置由 ROOT 直接解析。
for directory in ("uploads", "api_reports", "artifacts"):
    (ROOT / ".cache" / directory).mkdir(parents=True, exist_ok=True)

# jobs 保存可序列化任务快照，供查询接口和服务重启恢复共同使用。
jobs = {}

# job_lock 保护任务状态；model_lock 串行化所有占用模型资源的操作。
job_lock = Lock()
model_lock = Lock()

# 审计任务只使用一个后台线程，避免多个清洗流程同时争抢显存。
executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="aegis-audit")


def _save_jobs():
    """使用临时文件原子保存任务状态。"""
    # 状态文件在保存时直接由工程根目录定位，不保留额外路径常量。
    jobs_file = ROOT / ".cache" / "api_jobs.json"

    # 序列化期间持有任务锁，确保内存快照不会被后台线程同时修改。
    with job_lock:
        content = json.dumps(jobs, ensure_ascii=False, indent=2)

        # 先写同目录临时文件，再替换正式文件，降低中途退出造成的 JSON 损坏风险。
        temp_path = jobs_file.with_suffix(".tmp")
        temp_path.write_text(content, encoding="utf-8")
        temp_path.replace(jobs_file)


# 启动时恢复历史任务；损坏记录不会阻止服务继续启动。
try:
    # 首次启动没有状态文件时使用空字典，后续启动读取上一次任务快照。
    loaded = (
        json.loads((ROOT / ".cache" / "api_jobs.json").read_text(encoding="utf-8"))
        if (ROOT / ".cache" / "api_jobs.json").is_file()
        else {}
    )

    # 根节点必须是 job_id 到任务记录的映射，其他结构直接视为损坏。
    if not isinstance(loaded, dict):
        raise ValueError("任务状态文件的根节点必须是 JSON 对象。")

    # 单条异常记录不会污染整个任务表，仅恢复值为字典的历史记录。
    jobs.update(
        {job_id: job for job_id, job in loaded.items() if isinstance(job, dict)}
    )
except (OSError, ValueError) as exc:
    print(f"      [警告] 历史任务读取失败: {exc}")

# 线程任务无法跨进程恢复，因此重启前未结束的任务统一标记为中断。
interrupted = False
interrupted_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
for job in jobs.values():
    if job.get("status") in ("queued", "running"):
        # 保留原任务参数，只覆盖执行状态、结束时间和中断原因。
        job.update(
            status="failed",
            stage="中断",
            finished_at=interrupted_at,
            error="服务重启，未完成的任务已经中断。",
        )
        interrupted = True

if interrupted:
    # 只有确实修正过历史状态时才触发一次磁盘写入。
    try:
        _save_jobs()
    except OSError as exc:
        print(f"      [警告] 中断任务状态保存失败: {exc}")


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


# 上传结果包含资源编号、实际字节数和内容摘要，便于客户端核对文件。
class LoraUploadResponse(BaseModel):
    lora_id: str
    size_bytes: int
    sha256: str
    status: Literal["ready"]


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


def _resolve_lora(lora_id: str) -> Path:
    """定位并检查服务器上的 LoRA 资源。"""
    # 上传根目录在资源解析时由 ROOT 进入，不保留模块级路径常量。
    uploads = (ROOT / ".cache" / "uploads").resolve()

    # resolve 后校验父目录，防止资源编号逃逸出 uploads 目录。
    lora_path = (uploads / lora_id).resolve()
    if lora_path.parent != uploads:
        raise HTTPException(status_code=400, detail="LoRA 资源编号不合法。")

    # 资源编号合法但目录不存在时，明确告知调用方该资源尚未上传。
    if not lora_path.is_dir():
        raise HTTPException(status_code=404, detail="LoRA 资源不存在。")

    # 只有权重和配置同时存在时，资源才允许进入检测流水线。
    if not (lora_path / "adapter_model.safetensors").is_file():
        raise HTTPException(status_code=409, detail="LoRA 权重文件不完整。")
    if not (lora_path / "adapter_config.json").is_file():
        raise HTTPException(status_code=409, detail="LoRA 配置文件不完整。")

    # 返回规范化后的服务端目录，算法层无需再次处理相对路径。
    return lora_path


def _scan_lora(lora_path: Path) -> dict:
    """串行调用静态探测流水线并转换算法异常。"""
    try:
        # 延迟导入算法模块，使健康检查无需加载完整模型依赖。
        from .pipeline import run_static_scan_pipeline

        # 独立扫描可能与后台清洗并发，统一模型锁用于避免争抢 GPU。
        with model_lock:
            return run_static_scan_pipeline(
                lora_path=str(lora_path),
                detector_path=str(
                    ROOT / "models" / "detectors" / "spectral_detector_llama.pkl"
                ),
                return_details=True,
            )

    # 探测器文件或必要算法资源缺失，表示服务端尚未就绪。
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    # 流水线拒绝当前 LoRA 时，将具体输入错误返回客户端。
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # 探测器参数或内部结构无效时按服务配置错误处理。
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    # 未知异常只写入服务日志，避免向外暴露实现细节和内部路径。
    except Exception as exc:
        print(f"      [错误] API 静态检测失败: {exc}")
        raise HTTPException(
            status_code=500, detail="静态检测失败，请查看服务日志。"
        ) from exc


def _run_job(job_id: str):
    """执行检测、清洗、归档和任务状态更新。"""
    # -----------------------------------------------------------------
    # 步骤 1：读取任务上下文并进入检测阶段
    # -----------------------------------------------------------------
    # 先复制任务快照，再更新共享状态，后续耗时算法不会长期占用任务锁。
    with job_lock:
        job = dict(jobs[job_id])
        jobs[job_id].update(
            status="running",
            stage="检测",
            started_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )

    try:
        # -----------------------------------------------------------------
        # 步骤 2：重新校验服务器资源并执行静态检测
        # -----------------------------------------------------------------
        # 后台线程只依赖 job_id，从注册表和任务快照重建全部执行上下文。
        model = MODELS[job["model_id"]]
        base_path = Path(model["path"]).resolve()
        lora_path = _resolve_lora(job["lora_id"])
        scan_result = _scan_lora(lora_path)

        # 安全 LoRA 直接通过，不生成清洗报告和模型压缩包。
        # 只有命中风险时才加载体积较大的清洗依赖。
        if not scan_result["is_poisoned"]:
            result = {"action": "passed", "scan": scan_result, "cleanse": None}
        else:
            with job_lock:
                # CLI 轮询依赖该字段展示任务已经进入清洗阶段。
                jobs[job_id]["stage"] = "清洗"

            # -----------------------------------------------------------------
            # 步骤 3：根据任务模式执行快速清洗或深度清洗
            # -----------------------------------------------------------------
            from .pipeline import run_fast_cleanse_pipeline, run_immunization_pipeline

            signature = None
            with model_lock:
                if job["cleanse_mode"] == "fast":
                    # 快速模式复用与基础模型系列匹配的离线签名，耗时较低。
                    signature = model["signature"]
                    if not signature.is_file():
                        raise FileNotFoundError(
                            f"缺少 {model['family']} 快速清洗签名：{signature}"
                        )
                    report_path, suppressed, output_path = run_fast_cleanse_pipeline(
                        base_model_path=str(base_path),
                        lora_path=str(lora_path),
                        signature_path=str(signature),
                        recovery_data_path=str(
                            ROOT / "datasets" / "clean_data_recovery.json"
                        ),
                    )
                else:
                    # 深度模式动态提取多域签名，不依赖预生成 signature 文件。
                    report_path, suppressed, output_path = run_immunization_pipeline(
                        base_model_path=str(base_path),
                        lora_path=str(lora_path),
                        variant_data_path=str(
                            ROOT / "datasets" / "clean_data_variants.json"
                        ),
                        recovery_data_path=str(
                            ROOT / "datasets" / "clean_data_recovery.json"
                        ),
                    )

            # -----------------------------------------------------------------
            # 步骤 4：归档流水线生成的 HTML 与 JSON 审计报告
            # -----------------------------------------------------------------
            source_html = Path(report_path).resolve()
            source_json = source_html.with_suffix(".json")

            # HTML 与 JSON 共同组成完整审计记录，缺少任意一个都视为任务失败。
            if not source_html.is_file() or not source_json.is_file():
                raise FileNotFoundError("清洗完成，但审计报告不完整。")

            # 以 job_id 归档报告，避免算法默认文件名覆盖其他历史任务。
            copy2(source_html, ROOT / ".cache" / "api_reports" / f"{job_id}.html")
            copy2(source_json, ROOT / ".cache" / "api_reports" / f"{job_id}.json")

            with job_lock:
                # 报告归档完成后再切换阶段，使 CLI 展示与实际进度一致。
                jobs[job_id]["stage"] = "打包"

            # -----------------------------------------------------------------
            # 步骤 5：将清洗后的标准 LoRA 目录打包为下载产物
            # -----------------------------------------------------------------
            output_path = Path(output_path).resolve()
            if not output_path.is_dir():
                raise FileNotFoundError("清洗完成，但没有找到输出 LoRA。")

            # ZIP 根目录直接指向清洗后的 LoRA，下载后即得到标准适配器文件。
            artifact_path = Path(
                make_archive(
                    str(ROOT / ".cache" / "artifacts" / job_id),
                    "zip",
                    root_dir=str(output_path),
                )
            )
            result = {
                "action": "cleaned",
                "scan": scan_result,
                "cleanse": {
                    "mode": job["cleanse_mode"],
                    "signature_family": model["family"] if signature else None,
                    "signature": signature.name if signature else None,
                    "suppressed_count": int(suppressed),
                    # 结果只暴露下载地址和文件名，不返回服务器绝对路径。
                    "report_urls": {
                        "html": f"/v1/jobs/{job_id}/report?report_format=html",
                        "json": f"/v1/jobs/{job_id}/report?report_format=json",
                    },
                    "artifact_url": f"/v1/jobs/{job_id}/artifact",
                    "artifact_name": artifact_path.name,
                },
            }

        # -----------------------------------------------------------------
        # 步骤 6：写入结构化结果并结束任务
        # -----------------------------------------------------------------
        # 成功状态与完整结果在同一锁区间写入，查询接口不会读到半成品。
        with job_lock:
            jobs[job_id].update(
                status="succeeded",
                stage="完成",
                finished_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                result=result,
                error=None,
            )
    except Exception as exc:
        # 已知算法和文件错误保留原因，未知错误只对外返回统一信息。
        if isinstance(exc, HTTPException):
            # HTTPException 已经经过接口语义转换，可直接使用 detail。
            error = str(exc.detail)
        elif isinstance(exc, (OSError, RuntimeError, ValueError)):
            # 文件、算法和参数错误保留可操作的诊断信息。
            error = str(exc)
        else:
            # 未知异常只写日志，任务记录不包含内部堆栈和敏感路径。
            print(f"      [错误] 审计任务失败: {exc}")
            error = "审计任务执行失败，请查看服务日志。"

        # 写入结束时间和失败原因，确保任务不会一直停留在 running。
        with job_lock:
            jobs[job_id].update(
                status="failed",
                stage="失败",
                finished_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                error=error,
            )
    finally:
        # 无论成功或失败，最终状态都必须落盘。
        try:
            _save_jobs()
        except (OSError, TypeError) as exc:
            print(f"      [警告] 任务状态保存失败: {exc}")


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

    # 快速清洗要求对应模型签名和恢复数据同时存在。
    fast_ready = {
        model["family"]: model["signature"].is_file() and recovery_ready
        for model in MODELS.values()
    }

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
    # 步骤 1：检查上传文件类型
    # -----------------------------------------------------------------
    # 文件名检查用于快速拒绝明显错误输入，实际内容仍由后续流水线验证。
    if not (weights.filename or "").lower().endswith(".safetensors"):
        raise HTTPException(status_code=400, detail="weights 必须是 safetensors 文件。")

    # 配置允许任意原始文件名，但必须具有 JSON 扩展名。
    if not (config.filename or "").lower().endswith(".json"):
        raise HTTPException(status_code=400, detail="config 必须是 JSON 文件。")

    # -----------------------------------------------------------------
    # 步骤 2：创建本次上传的隔离目录
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
        # 步骤 3：限制并解析 adapter_config.json
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
        # 步骤 4：分块写入权重并计算 SHA256
        # -----------------------------------------------------------------
        # 摘要随写入同步更新，避免为了计算哈希再次读取整个大文件。
        digest = hashlib.sha256()
        total_size = 0

        # 每次只读取 1 MiB，使大体积 LoRA 上传保持稳定内存占用。
        with (lora_dir / "adapter_model.safetensors").open("wb") as file:
            while chunk := await weights.read(1024 * 1024):
                total_size += len(chunk)

                # 在写入越界分块前终止，异常出口会删除整个隔离目录。
                if total_size > max_weights:
                    raise HTTPException(
                        status_code=413, detail="LoRA 权重超过上传限制。"
                    )
                digest.update(chunk)
                file.write(chunk)

        # 空权重不是有效适配器，即使文件名和扩展名正确也必须拒绝。
        if total_size == 0:
            raise HTTPException(status_code=400, detail="LoRA 权重文件为空。")

        # -----------------------------------------------------------------
        # 步骤 5：保存配置并返回 LoRA 资源信息
        # -----------------------------------------------------------------
        # 重新序列化已验证配置，并统一使用服务端约定的标准文件名。
        (lora_dir / "adapter_config.json").write_text(
            json.dumps(config_data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        # 客户端后续只携带 lora_id，服务器路径不会出现在响应中。
        return {
            "lora_id": lora_id,
            "size_bytes": total_size,
            "sha256": digest.hexdigest(),
            "status": "ready",
        }
    # 已知 HTTP 错误保留原状态码，同时删除本次上传产生的不完整目录。
    except HTTPException:
        rmtree(lora_dir, ignore_errors=True)
        raise

    # 文件系统等未知异常记录服务日志，对外只返回统一错误信息。
    except Exception as exc:
        rmtree(lora_dir, ignore_errors=True)
        print(f"      [错误] LoRA 上传失败: {exc}")
        raise HTTPException(
            status_code=500, detail="LoRA 上传失败，请查看服务日志。"
        ) from exc


@v1.post("/scan", summary="单独检测", response_model=ScanResponse)
def scan(request: ScanRequest):
    # 解析服务器资源后复用统一检测逻辑，不直接接收客户端文件路径。
    return _scan_lora(_resolve_lora(request.lora_id))


@v1.post(
    "/jobs", summary="创建任务", status_code=202, response_model=JobCreatedResponse
)
def create_job(request: AuditRequest):
    # -----------------------------------------------------------------
    # 步骤 1：确认基础模型与 LoRA 资源可用
    # -----------------------------------------------------------------
    # model_id 必须来自服务端注册表，客户端不能提交任意模型路径。
    model = MODELS.get(request.model_id)
    if model is None:
        raise HTTPException(status_code=404, detail="基础模型不存在。")

    # config.json 是本地基础模型完成下载和可加载的最小就绪标志。
    if not (model["path"] / "config.json").is_file():
        raise HTTPException(status_code=503, detail="基础模型尚未就绪。")

    # 入队前再次检查 LoRA 完整性，避免无效任务占用后台队列。
    _resolve_lora(request.lora_id)

    # -----------------------------------------------------------------
    # 步骤 2：创建可持久化的等待任务
    # -----------------------------------------------------------------
    # job_id 是状态查询、报告归档和产物下载共同使用的稳定标识。
    job_id = f"audit-{uuid4().hex[:12]}"

    # 任务记录仅包含 JSON 可序列化字段，确保可以直接写入状态文件。
    with job_lock:
        jobs[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "stage": "等待",
            "model_id": request.model_id,
            "lora_id": request.lora_id,
            "cleanse_mode": request.cleanse_mode,
            "submitted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "started_at": None,
            "finished_at": None,
            "result": None,
            "error": None,
        }

    # -----------------------------------------------------------------
    # 步骤 3：先保存等待状态，再提交后台执行器
    # -----------------------------------------------------------------
    try:
        # 先落盘再入队，服务意外退出时仍能把该任务恢复为中断状态。
        _save_jobs()

        # 后台函数只接收 job_id，执行参数统一从任务快照和模型注册表读取。
        executor.submit(_run_job, job_id)
    except (OSError, TypeError) as exc:
        # 等待状态无法可靠保存时撤销内存记录，避免出现不可恢复任务。
        with job_lock:
            jobs.pop(job_id, None)
        raise HTTPException(status_code=500, detail="任务状态保存失败。") from exc
    except RuntimeError as exc:
        # 执行器关闭或不可用时撤销任务，并把撤销结果重新写回状态文件。
        with job_lock:
            jobs.pop(job_id, None)
        try:
            _save_jobs()
        except OSError as save_exc:
            print(f"      [警告] 队列失败状态保存失败: {save_exc}")
        raise HTTPException(status_code=503, detail="后台任务队列不可用。") from exc

    # 202 响应只表示任务已入队，不代表检测或清洗已经完成。
    return {
        "job_id": job_id,
        "status": "queued",
        "stage": "等待",
        "status_url": f"/v1/jobs/{job_id}",
    }


@v1.get("/jobs", summary="查看任务")
def list_jobs(limit: int = Query(default=20, ge=1, le=100)):
    # 锁内只复制快照，排序和截断在锁外完成。
    with job_lock:
        job_items = [dict(job) for job in jobs.values()]

    # ISO 8601 时间可以直接按字符串倒序，最新提交的任务排在最前面。
    job_items.sort(key=lambda job: job.get("submitted_at") or "", reverse=True)

    # total 返回全部任务数，items 只包含本次请求的数量上限。
    return {"total": len(job_items), "items": job_items[:limit]}


@v1.get("/jobs/{job_id}", summary="查询任务")
def get_job(job_id: str):
    # 返回副本，避免响应序列化过程持有或修改共享状态。
    with job_lock:
        job = jobs.get(job_id)

        # 查询不存在的任务时明确返回 404，而不是空对象。
        if job is None:
            raise HTTPException(status_code=404, detail="任务不存在。")

        # 副本包含完整阶段、时间、结果或错误信息，供 CLI show 和轮询使用。
        return dict(job)


@v1.get("/jobs/{job_id}/report", summary="下载报告")
def get_report(job_id: str, report_format: Literal["html", "json"] = "html"):
    # 先读取任务快照，再校验任务是否真正生成了清洗报告。
    with job_lock:
        job = dict(jobs[job_id]) if job_id in jobs else None

    # job_id 不存在和报告文件不存在分别处理，错误原因更明确。
    if job is None:
        raise HTTPException(status_code=404, detail="任务不存在。")

    # 正在执行的任务尚不能判断是否会生成报告，使用 409 表示状态冲突。
    if job.get("status") in ("queued", "running"):
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
    with job_lock:
        job = dict(jobs[job_id]) if job_id in jobs else None

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
