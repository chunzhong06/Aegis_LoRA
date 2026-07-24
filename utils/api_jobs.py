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
    is_community_model_id,
    registered_model,
    resolve_candidate,
)

# =====================================================================
# 配置与状态
# =====================================================================
# 工程根目录是全部服务端资源的统一定位基准，避免依赖启动命令所在目录。
ROOT = Path(__file__).resolve().parents[1]

# MODELS 保持稳定对象身份，api_server 持有的别名可在注册表刷新后立即看到新模型。
MODELS = {}
refresh_registry(MODELS)

# API 运行目录统一放在 .cache，具体路径在使用位置由 ROOT 直接解析。
for directory in ("uploads", "api_reports", "artifacts"):
    (ROOT / ".cache" / directory).mkdir(parents=True, exist_ok=True)

# jobs 保存可序列化任务快照，供查询接口和服务重启恢复共同使用。
jobs = {}

# job_lock 保护任务状态；model_lock 串行化所有占用模型资源的操作。
job_lock = Lock()
model_lock = Lock()

# 审计任务只使用一个后台线程，避免多个清洗流程同时争抢显存。
executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="aegis-audit")


def sync_jobs():
    """裁剪历史任务并使用临时文件原子保存状态。"""
    # 状态文件在保存时直接由工程根目录定位，不保留额外路径常量。
    jobs_file = ROOT / ".cache" / "api_jobs.json"

    # -----------------------------------------------------------------
    # 步骤 1：保留全部活动任务和最近一百条终态任务
    # -----------------------------------------------------------------
    # 终态任务按结束时间倒序排列；活动任务不参与数量限制。
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

        # 排序后一百条之外的终态任务进入过期集合，等待从状态和产物中共同移除。
        expired = {job_id for job_id, _ in terminal[100:]}

        # 快照同时保留全部活动任务和有效终态任务，作为本次持久化的唯一数据源。
        snapshot = {
            job_id: job for job_id, job in jobs.items() if job_id not in expired
        }

        # -----------------------------------------------------------------
        # 步骤 2：先落盘新快照，再同步更新内存任务表
        # -----------------------------------------------------------------
        # 写入失败时不修改内存，避免磁盘状态和当前服务状态产生分歧。
        content = json.dumps(snapshot, ensure_ascii=False, indent=2)

        # 临时文件与正式文件位于同一目录，写入完整后再原子替换状态文件。
        temp_path = jobs_file.with_suffix(".tmp")
        temp_path.write_text(content, encoding="utf-8")
        temp_path.replace(jobs_file)

        # 磁盘替换成功后再裁剪内存，并保存仍允许访问产物的任务编号集合。
        for job_id in expired:
            jobs.pop(job_id, None)
        retained = set(jobs)

    # -----------------------------------------------------------------
    # 步骤 3：删除已经没有任务记录的报告和模型产物
    # -----------------------------------------------------------------
    # 两个目录只保存 API 归档文件，因此可以按文件名中的 job_id 清理孤立产物。
    for directory in ("api_reports", "artifacts"):
        # HTML、JSON 和 ZIP 都使用 job_id 作为文件主名，可直接对应任务记录。
        for path in (ROOT / ".cache" / directory).iterdir():
            if path.is_file() and path.stem not in retained:
                try:
                    # 只删除已经失去任务记录的普通文件，不处理目录或有效任务产物。
                    path.unlink()
                except OSError as exc:
                    # 单个历史文件删除失败不影响状态同步和其他产物清理。
                    print(f"      [警告] 历史产物清理失败: {path.name}: {exc}")


def expire_confirmations() -> int:
    """将已经超过确认截止时间的任务转为失败，并回收对应上传。"""
    now = datetime.now(timezone.utc)
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
                expired_loras.append(job.get("lora_id"))
    for lora_id in expired_loras:
        if isinstance(lora_id, str):
            shutil.rmtree(ROOT / ".cache" / "uploads" / lora_id, ignore_errors=True)
    return len(expired_loras)


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
# 同一启动时间用于本轮全部中断任务，避免逐条生成略有差异的结束时间。
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

# 待确认任务跨重启保留，但已过截止时间的记录立即转为确认超时。
expire_confirmations()

# 启动时统一裁剪历史任务并清理已经没有记录的报告和模型产物。
try:
    # 即使没有中断任务也执行一次同步，使历史上限和孤立产物立即生效。
    sync_jobs()
except (OSError, TypeError) as exc:
    # 启动同步失败只降低历史管理能力，不阻止健康检查和服务进程启动。
    print(f"      [警告] 启动任务状态同步失败: {exc}")

# 待确认任务仍需要原始 LoRA；其他孤立或中断任务上传在启动时回收。
retained_uploads = {
    job.get("lora_id")
    for job in jobs.values()
    if job.get("status") == "awaiting_confirmation"
}
for path in (ROOT / ".cache" / "uploads").iterdir():
    if path.name in retained_uploads:
        continue
    try:
        # 上传目录和可能遗留的临时文件统一删除，只跳过仍有活动任务引用的目录。
        shutil.rmtree(path) if path.is_dir() else path.unlink()
    except OSError as exc:
        # 单个缓存清理失败仅记录警告，其余上传资源继续处理。
        print(f"      [警告] 启动上传缓存清理失败: {path.name}: {exc}")

# 服务异常退出可能留下未验证下载；稳定模型目录不在这里处理。
community_root = ROOT / "models" / "community"
if community_root.is_dir():
    for path in community_root.glob(".partial-*"):
        shutil.rmtree(path, ignore_errors=True)


def resolve_lora(lora_id: str) -> Path:
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


def scan_lora(lora_path: Path) -> dict:
    """串行调用静态探测流水线并转换算法异常。"""
    try:
        # 延迟导入算法模块，使健康检查无需加载完整模型依赖。
        from .pipeline import run_static_scan_pipeline

        # 独立扫描可能与后台清洗并发，统一模型锁用于避免争抢模型资源。
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


def run_job(job_id: str):
    """执行检测、清洗、归档和任务状态更新。"""
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

    lora_path = None
    output_path = None
    source_html = None
    source_json = None
    preserve_lora = False

    try:
        lora_path = resolve_lora(job["lora_id"])
        scan_result = job.get("scan_result")
        if scan_result is None:
            scan_result = scan_lora(lora_path)
            with job_lock:
                jobs[job_id]["scan_result"] = scan_result

        if not scan_result["is_poisoned"]:
            result = {"action": "passed", "scan": scan_result, "cleanse": None}
        else:
            model = registered_model(
                MODELS,
                job["model_id"],
                job.get("model_revision"),
            )
            if model is None:
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
                            download=None,
                        )
                    preserve_lora = True
                    sync_jobs()
                    return

                if not job.get("model_confirmed"):
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
                        download={
                            "downloaded_bytes": 0,
                            "total_bytes": int(candidate["estimated_size"]),
                            "percent": 0,
                        },
                    )
                sync_jobs()
                model = download_and_register(
                    candidate,
                    job_id,
                    MODELS,
                    jobs,
                    job_lock,
                )
                with job_lock:
                    jobs[job_id]["download"] = {
                        "downloaded_bytes": int(candidate["estimated_size"]),
                        "total_bytes": int(candidate["estimated_size"]),
                        "percent": 100,
                    }
                sync_jobs()

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

        with job_lock:
            jobs[job_id].update(
                status="succeeded",
                stage="完成",
                finished_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                result=result,
                error=None,
            )
    except Exception as exc:
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
        for path in (source_html, source_json):
            if path is not None:
                try:
                    path.unlink(missing_ok=True)
                except OSError as exc:
                    print(f"      [警告] 流水线报告清理失败: {exc}")

        for path in (output_path, None if preserve_lora else lora_path):
            if path is not None:
                shutil.rmtree(path, ignore_errors=True)

        try:
            sync_jobs()
        except (OSError, TypeError) as exc:
            print(f"      [警告] 任务状态同步失败: {exc}")
