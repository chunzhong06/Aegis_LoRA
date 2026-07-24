# Aegis-LoRA - API 任务运行模块
# 负责任务状态、资源定位、检测清洗、归档持久化和临时缓存回收。
import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Lock

from fastapi import HTTPException

from .model_registry import (
    download_and_register,
    refresh_registry,
    registered_model,
    resolve_candidate,
)

# =====================================================================
# 配置与状态
# =====================================================================
# 项目资源统一基于当前模块定位，避免依赖服务启动目录。
ROOT = Path(__file__).resolve().parents[1]

# MODELS 必须保持对象身份，社区模型注册后由 model_registry 原地更新。
MODELS = {}
refresh_registry(MODELS)

# API 上传、报告和产物统一写入 .cache。
for directory in ("uploads", "api_reports", "artifacts"):
    (ROOT / ".cache" / directory).mkdir(parents=True, exist_ok=True)

# jobs 保存可序列化任务状态，也是查询接口和重启恢复的共同数据源。
jobs = {}

# job_lock 保护任务状态；model_lock 串行化检测和清洗等模型操作。
job_lock = Lock()
model_lock = Lock()

# 单线程执行器避免多个清洗任务同时争抢显存。
executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="aegis-audit")


# =====================================================================
# 任务持久化与启动恢复
# =====================================================================
def sync_jobs(*, cleanup_orphans: bool = False):
    """裁剪历史任务并使用临时文件原子保存状态。"""
    jobs_file = ROOT / ".cache" / "api_jobs.json"

    # -----------------------------------------------------------------
    # 步骤 1：保留全部活动任务和最近一百条终态任务
    # -----------------------------------------------------------------
    # 活动任务全部保留，终态任务按结束时间只保留最近一百条。
    with job_lock:
        terminal = sorted(
            (
                (job_id, job)
                for job_id, job in jobs.items()
                if job.get("status")
                not in ("queued", "running", "awaiting_confirmation")
            ),
            key=lambda item: item[1].get("finished_at")
            or item[1].get("submitted_at")
            or "",
            reverse=True,
        )

        # expired 同时用于裁剪内存状态和对应归档产物。
        expired = {job_id for job_id, _ in terminal[100:]}

        # snapshot 是本次写入磁盘的完整任务视图。
        snapshot = {
            job_id: job for job_id, job in jobs.items() if job_id not in expired
        }

        # -----------------------------------------------------------------
        # 步骤 2：先落盘新快照，再同步更新内存任务表
        # -----------------------------------------------------------------
        # 写入失败时不裁剪内存，避免磁盘和运行时状态进一步分歧。
        content = json.dumps(snapshot, ensure_ascii=False, indent=2)

        # 临时文件同目录写入，完整后再原子替换。
        temp_path = jobs_file.with_suffix(".tmp")
        temp_path.write_text(content, encoding="utf-8")
        temp_path.replace(jobs_file)

        for job_id in expired:
            jobs.pop(job_id, None)

    # -----------------------------------------------------------------
    # 步骤 3：仅在启动或实际裁剪历史时扫描孤立产物
    # -----------------------------------------------------------------
    if not cleanup_orphans and not expired:
        return
    with job_lock:
        retained = set(jobs)

    # 报告与 ZIP 均以 job_id 命名，可直接判断是否仍有任务引用。
    for directory in ("api_reports", "artifacts"):
        for path in (ROOT / ".cache" / directory).iterdir():
            if path.is_file() and path.stem not in retained:
                try:
                    path.unlink()
                except OSError as exc:
                    # 单个清理失败不影响其他状态与产物。
                    print(f"      [警告] 历史产物清理失败: {path.name}: {exc}")


def expire_confirmations() -> int:
    """将已经超过确认截止时间的任务转为失败，并回收对应上传。"""
    now = datetime.now(timezone.utc)

    # expired_loras 延迟到锁外删除，避免文件系统操作长时间占用任务锁。
    expired_loras = []
    with job_lock:
        for job in jobs.values():
            if job.get("status") != "awaiting_confirmation":
                continue
            try:
                deadline = datetime.fromisoformat(job["confirmation_deadline"])
            except (KeyError, TypeError, ValueError):
                deadline = now
            if deadline.tzinfo is None:
                deadline = deadline.replace(tzinfo=timezone.utc)
            if deadline <= now:
                job.update(
                    status="failed",
                    stage="确认超时",
                    finished_at=now.isoformat(timespec="seconds"),
                    error="社区模型确认已超时。",
                )
                job.pop("confirmation_deadline", None)
                expired_loras.append(job.get("lora_id"))
    for lora_id in expired_loras:
        if isinstance(lora_id, str):
            shutil.rmtree(ROOT / ".cache" / "uploads" / lora_id, ignore_errors=True)
    return len(expired_loras)


# -----------------------------------------------------------------
# 启动恢复：读取任务、处理中断状态并回收孤立资源
# -----------------------------------------------------------------
# 损坏的历史状态只记录警告，不阻止服务启动。
try:
    loaded = (
        json.loads((ROOT / ".cache" / "api_jobs.json").read_text(encoding="utf-8"))
        if (ROOT / ".cache" / "api_jobs.json").is_file()
        else {}
    )

    if not isinstance(loaded, dict):
        raise ValueError("任务状态文件的根节点必须是 JSON 对象。")

    # 单条异常记录被跳过，避免污染整个任务表。
    jobs.update(
        {job_id: job for job_id, job in loaded.items() if isinstance(job, dict)}
    )
except (OSError, ValueError) as exc:
    print(f"      [警告] 历史任务读取失败: {exc}")

# 后台线程无法跨进程恢复，queued/running 统一标记为中断。
interrupted_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
for job in jobs.values():
    if job.get("status") in ("queued", "running"):
        job.update(
            status="failed",
            stage="中断",
            finished_at=interrupted_at,
            error="服务重启，未完成的任务已经中断。",
        )

# awaiting_confirmation 跨重启保留，过期任务立即转为确认超时。
expire_confirmations()

# 启动时执行一次历史裁剪和孤立归档清理。
try:
    sync_jobs(cleanup_orphans=True)
except (OSError, TypeError) as exc:
    print(f"      [警告] 启动任务状态同步失败: {exc}")

# 待确认任务需要保留上传，其余孤立或中断上传在启动时回收。
retained_uploads = {
    job.get("lora_id")
    for job in jobs.values()
    if job.get("status") == "awaiting_confirmation"
}
for path in (ROOT / ".cache" / "uploads").iterdir():
    if path.name in retained_uploads:
        continue
    try:
        shutil.rmtree(path) if path.is_dir() else path.unlink()
    except OSError as exc:
        print(f"      [警告] 启动上传缓存清理失败: {path.name}: {exc}")

# 服务异常退出可能留下未验证下载；稳定模型目录不在这里处理。
community_root = ROOT / "models" / "community"
if community_root.is_dir():
    for path in community_root.glob(".partial-*"):
        shutil.rmtree(path, ignore_errors=True)


def resolve_lora(lora_id: str) -> Path:
    """定位并检查服务器上的 LoRA 资源。"""
    uploads = (ROOT / ".cache" / "uploads").resolve()

    # lora_path 必须是 uploads 的直接子目录，防止路径穿越。
    lora_path = (uploads / lora_id).resolve()
    if lora_path.parent != uploads:
        raise HTTPException(status_code=400, detail="LoRA 资源编号不合法。")

    if not lora_path.is_dir():
        raise HTTPException(status_code=404, detail="LoRA 资源不存在。")

    # 权重和配置必须同时存在，避免后台任务接收半成品上传。
    if not (lora_path / "adapter_model.safetensors").is_file():
        raise HTTPException(status_code=409, detail="LoRA 权重文件不完整。")
    if not (lora_path / "adapter_config.json").is_file():
        raise HTTPException(status_code=409, detail="LoRA 配置文件不完整。")

    return lora_path


def scan_lora(lora_path: Path) -> dict:
    """串行调用静态探测流水线并转换算法异常。"""
    try:
        # 延迟导入，服务启动和健康检查不加载完整算法依赖。
        from .pipeline import run_static_scan_pipeline

        # 独立扫描与后台清洗共享 model_lock，避免争用模型资源。
        with model_lock:
            return run_static_scan_pipeline(
                lora_path=str(lora_path),
                detector_path=str(
                    ROOT / "models" / "detectors" / "spectral_detector_llama.pkl"
                ),
                return_details=True,
            )

    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    # 未知异常只写服务日志，不向客户端暴露内部路径和堆栈。
    except Exception as exc:
        print(f"      [错误] API 静态检测失败: {exc}")
        raise HTTPException(
            status_code=500, detail="静态检测失败，请查看服务日志。"
        ) from exc


# =====================================================================
# 审计任务状态机
# =====================================================================
def run_job(job_id: str):
    """执行检测、清洗、归档和任务状态更新。"""
    # -----------------------------------------------------------------
    # 步骤 1：领取排队任务并建立本次执行上下文
    # -----------------------------------------------------------------
    with job_lock:
        if job_id not in jobs or jobs[job_id].get("status") != "queued":
            return
        job = dict(jobs[job_id])
        jobs[job_id].update(
            status="running",
            stage="检测",
            started_at=jobs[job_id].get("started_at")
            or datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )

    # 中间路径用于 finally 统一回收；待确认分支通过 preserve_lora 保留上传。
    lora_path = None
    output_path = None
    source_html = None
    source_json = None
    preserve_lora = False
    sync_on_exit = True

    try:
        # -----------------------------------------------------------------
        # 步骤 2：解析 LoRA 并执行或复用静态检测
        # -----------------------------------------------------------------
        lora_path = resolve_lora(job["lora_id"])

        # 确认后的任务保留 scan_result，重新入队时不得重复检测。
        scan_result = job.get("scan_result")
        if scan_result is None:
            scan_result = scan_lora(lora_path)
            with job_lock:
                jobs[job_id]["scan_result"] = scan_result

        if not scan_result["is_poisoned"]:
            # 安全 LoRA 直接通过，不解析、下载或加载社区模型。
            result = {"action": "passed", "scan": scan_result, "cleanse": None}
        else:
            # -----------------------------------------------------------------
            # 步骤 3：中毒后解析已注册模型或社区模型候选
            # -----------------------------------------------------------------
            model = registered_model(
                MODELS,
                job["model_id"],
                job.get("model_revision"),
            )
            if model is None:
                # model_candidate 只在首次解析后保存，确认恢复时直接复用。
                candidate = job.get("model_candidate")
                if candidate is None:
                    try:
                        adapter_config = json.loads(
                            (lora_path / "adapter_config.json").read_text(
                                encoding="utf-8"
                            )
                        )
                    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                        raise ValueError("LoRA adapter_config.json 无效。") from exc
                    lora_base_model = (
                        adapter_config.get("base_model_name_or_path")
                        if isinstance(adapter_config, dict)
                        and isinstance(
                            adapter_config.get("base_model_name_or_path"), str
                        )
                        else None
                    )
                    candidate = resolve_candidate(
                        job["model_id"],
                        job.get("model_revision"),
                        lora_base_model,
                        job["cleanse_mode"],
                    )
                    deadline = datetime.now(timezone.utc) + timedelta(hours=24)
                    with job_lock:
                        jobs[job_id].update(
                            status="awaiting_confirmation",
                            stage="待确认",
                            model_candidate=candidate,
                            model_revision=candidate["revision"],
                            confirmation_deadline=deadline.isoformat(
                                timespec="seconds"
                            ),
                        )
                    preserve_lora = True
                    sync_jobs()
                    sync_on_exit = False

                    # 待确认状态已落盘，当前线程立即返回并释放执行器。
                    return

                # -----------------------------------------------------------------
                # 步骤 4：确认后下载并注册社区模型
                # -----------------------------------------------------------------
                if not job.get("confirmed_at"):
                    raise ValueError("社区模型尚未确认。")
                if candidate["cleanse_mode"] != "deep":
                    raise ValueError("未登记专属签名的社区模型只能使用深度清洗。")
                for path, message in (
                    (
                        ROOT / "datasets" / "clean_data_variants.json",
                        "深度清洗变体数据尚未就绪。",
                    ),
                    (
                        ROOT / "datasets" / "clean_data_recovery.json",
                        "清洗恢复数据尚未就绪。",
                    ),
                ):
                    if not path.is_file():
                        raise FileNotFoundError(message)
                with job_lock:
                    jobs[job_id].update(
                        stage="准备模型",
                        downloaded_bytes=0,
                    )
                model = download_and_register(
                    candidate,
                    job_id,
                    MODELS,
                    jobs,
                    job_lock,
                )
                with job_lock:
                    jobs[job_id]["downloaded_bytes"] = int(
                        candidate["estimated_size"]
                    )

            # -----------------------------------------------------------------
            # 步骤 5：复核清洗资源并调用对应流水线
            # -----------------------------------------------------------------
            base_path = Path(model["path"]).resolve()
            if not (base_path / "config.json").is_file():
                raise FileNotFoundError("基础模型尚未就绪。")
            if not (ROOT / "datasets" / "clean_data_recovery.json").is_file():
                raise FileNotFoundError("清洗恢复数据尚未就绪。")
            if (
                job["cleanse_mode"] == "deep"
                and not (ROOT / "datasets" / "clean_data_variants.json").is_file()
            ):
                raise FileNotFoundError("深度清洗变体数据尚未就绪。")
            with job_lock:
                jobs[job_id]["stage"] = "清洗"

            from .pipeline import run_fast_cleanse_pipeline, run_immunization_pipeline

            signature = None
            # output_path 是流水线约定的临时清洗目录，完成后统一打包并删除。
            suffix = (
                "_fast_immunized" if job["cleanse_mode"] == "fast" else "_immunized"
            )
            output_path = Path(f"{lora_path}{suffix}").resolve()

            with model_lock:
                if job["cleanse_mode"] == "fast":
                    signature = model["signature"]
                    if signature is None or not signature.is_file():
                        raise FileNotFoundError(
                            f"缺少 {model['family']} 快速清洗签名：{signature}"
                        )
                    report_path, suppressed, pipeline_output = (
                        run_fast_cleanse_pipeline(
                            base_model_path=str(base_path),
                            lora_path=str(lora_path),
                            signature_path=str(signature),
                            recovery_data_path=str(
                                ROOT / "datasets" / "clean_data_recovery.json"
                            ),
                            auto_batch_size=True,
                        )
                    )
                else:
                    report_path, suppressed, pipeline_output = (
                        run_immunization_pipeline(
                            base_model_path=str(base_path),
                            lora_path=str(lora_path),
                            variant_data_path=str(
                                ROOT / "datasets" / "clean_data_variants.json"
                            ),
                            recovery_data_path=str(
                                ROOT / "datasets" / "clean_data_recovery.json"
                            ),
                            auto_batch_size=True,
                        )
                    )

            # pipeline_output 必须与约定目录完全一致，防止归档任意路径。
            candidate = Path(pipeline_output).resolve()
            uploads = (ROOT / ".cache" / "uploads").resolve()
            if (
                candidate != output_path
                or candidate.parent != uploads
                or not candidate.is_dir()
            ):
                raise FileNotFoundError("清洗完成，但没有找到输出 LoRA。")

            source_html = Path(report_path).resolve()
            source_json = source_html.with_suffix(".json")

            if not source_html.is_file() or not source_json.is_file():
                raise FileNotFoundError("清洗完成，但审计报告不完整。")

            # -----------------------------------------------------------------
            # 步骤 6：归档报告并打包清洗后的 LoRA
            # -----------------------------------------------------------------
            source_html.replace(ROOT / ".cache" / "api_reports" / f"{job_id}.html")
            source_json.replace(ROOT / ".cache" / "api_reports" / f"{job_id}.json")

            with job_lock:
                jobs[job_id]["stage"] = "打包"

            artifact_path = Path(
                shutil.make_archive(
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
                    "report_urls": {
                        "html": f"/v1/jobs/{job_id}/report?report_format=html",
                        "json": f"/v1/jobs/{job_id}/report?report_format=json",
                    },
                    "artifact_url": f"/v1/jobs/{job_id}/artifact",
                    "artifact_name": artifact_path.name,
                },
            }

        # -----------------------------------------------------------------
        # 步骤 7：写入成功终态并移除执行期临时字段
        # -----------------------------------------------------------------
        with job_lock:
            completed = jobs[job_id]
            completed.update(
                status="succeeded",
                stage="完成",
                finished_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                result=result,
            )
            for field in (
                "scan_result",
                "model_candidate",
                "confirmation_deadline",
                "downloaded_bytes",
            ):
                completed.pop(field, None)
            completed.pop("error", None)
    except Exception as exc:
        # 异常任务不得保留不完整报告或产物。
        for path in (
            ROOT / ".cache" / "api_reports" / f"{job_id}.html",
            ROOT / ".cache" / "api_reports" / f"{job_id}.json",
            ROOT / ".cache" / "artifacts" / f"{job_id}.zip",
        ):
            try:
                path.unlink(missing_ok=True)
            except OSError as cleanup_exc:
                print(f"      [警告] 失败任务产物清理失败: {cleanup_exc}")

        if isinstance(exc, HTTPException):
            error = str(exc.detail)
        elif isinstance(exc, (OSError, RuntimeError, ValueError)):
            error = str(exc)
        else:
            print(f"      [错误] 审计任务失败: {exc}")
            error = "审计任务执行失败，请查看服务日志。"

        with job_lock:
            jobs[job_id].update(
                status="failed",
                stage="失败",
                finished_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                error=error,
            )
    finally:
        # 流水线原始报告已归档或失败，均可清理。
        for path in (source_html, source_json):
            if path is not None:
                try:
                    path.unlink(missing_ok=True)
                except OSError as exc:
                    print(f"      [警告] 流水线报告清理失败: {exc}")

        # 待确认任务只保留原始 LoRA，其余路径在任务退出时回收。
        for path in (output_path, None if preserve_lora else lora_path):
            if path is not None:
                shutil.rmtree(path, ignore_errors=True)

        # 待确认分支已主动同步，避免 finally 连续重复落盘。
        if sync_on_exit:
            try:
                sync_jobs()
            except (OSError, TypeError) as exc:
                print(f"      [警告] 任务状态同步失败: {exc}")
