# Aegis-LoRA - API 任务运行模块
# 负责任务状态、资源定位、检测清洗、归档持久化和临时缓存回收。
import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

from fastapi import HTTPException

# =====================================================================
# 配置与状态
# =====================================================================
# 工程根目录是全部服务端资源的统一定位基准，避免依赖启动命令所在目录。
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
                if job.get("status") not in ("queued", "running")
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

# 启动时统一裁剪历史任务并清理已经没有记录的报告和模型产物。
try:
    # 即使没有中断任务也执行一次同步，使历史上限和孤立产物立即生效。
    sync_jobs()
except (OSError, TypeError) as exc:
    # 启动同步失败只降低历史管理能力，不阻止健康检查和服务进程启动。
    print(f"      [警告] 启动任务状态同步失败: {exc}")

# 上传资源只服务当前扫描或任务，不跨服务重启保留。
for path in (ROOT / ".cache" / "uploads").iterdir():
    try:
        # 上传目录和可能遗留的临时文件统一删除，重启后不恢复未消费资源。
        shutil.rmtree(path) if path.is_dir() else path.unlink()
    except OSError as exc:
        # 单个缓存清理失败仅记录警告，其余上传资源继续处理。
        print(f"      [警告] 启动上传缓存清理失败: {path.name}: {exc}")


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

    # 临时路径在 finally 中统一回收，任一步骤失败都不会遗留上传和流水线文件。
    # LoRA 与清洗目录使用目录删除，源报告则使用文件删除，因此分别保存引用。
    lora_path = None
    output_path = None
    source_html = None
    source_json = None

    try:
        # -----------------------------------------------------------------
        # 步骤 2：重新校验服务器资源并执行静态检测
        # -----------------------------------------------------------------
        # 后台线程只依赖 job_id，从注册表和任务快照重建全部执行上下文。
        model = MODELS[job["model_id"]]
        base_path = Path(model["path"]).resolve()
        lora_path = resolve_lora(job["lora_id"])
        scan_result = scan_lora(lora_path)

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

            # 快速与深度流水线使用不同输出后缀，提前计算可预知的工作目录。
            # 即使流水线中途抛出异常，finally 仍能定位并回收已经生成的目录。
            signature = None
            suffix = (
                "_fast_immunized" if job["cleanse_mode"] == "fast" else "_immunized"
            )
            output_path = Path(f"{lora_path}{suffix}").resolve()

            # 模型锁覆盖实际清洗阶段，防止扫描或其他任务同时占用模型资源。
            with model_lock:
                if job["cleanse_mode"] == "fast":
                    # 快速模式复用与基础模型系列匹配的离线签名，耗时较低。
                    signature = model["signature"]
                    if not signature.is_file():
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
                    # 深度模式动态提取多域签名，不依赖预生成 signature 文件。
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

            # 流水线输出必须位于上传缓存内，通过校验后才允许后续打包和回收。
            # 同时要求返回路径与预期目录完全一致，避免归档意外位置的模型文件。
            candidate = Path(pipeline_output).resolve()
            uploads = (ROOT / ".cache" / "uploads").resolve()
            if (
                candidate != output_path
                or candidate.parent != uploads
                or not candidate.is_dir()
            ):
                raise FileNotFoundError("清洗完成，但没有找到输出 LoRA。")

            # -----------------------------------------------------------------
            # 步骤 4：归档流水线生成的 HTML 与 JSON 审计报告
            # -----------------------------------------------------------------
            source_html = Path(report_path).resolve()
            source_json = source_html.with_suffix(".json")

            # HTML 与 JSON 共同组成完整审计记录，缺少任意一个都视为任务失败。
            if not source_html.is_file() or not source_json.is_file():
                raise FileNotFoundError("清洗完成，但审计报告不完整。")

            # 以 job_id 移动报告，不在流水线目录中保留内容相同的副本。
            # HTML 和 JSON 共享同一任务主名，下载接口可以直接由 job_id 定位。
            source_html.replace(ROOT / ".cache" / "api_reports" / f"{job_id}.html")
            source_json.replace(ROOT / ".cache" / "api_reports" / f"{job_id}.json")

            with job_lock:
                # 报告归档完成后再切换阶段，使 CLI 展示与实际进度一致。
                jobs[job_id]["stage"] = "打包"

            # -----------------------------------------------------------------
            # 步骤 5：将清洗后的标准 LoRA 目录打包为下载产物
            # -----------------------------------------------------------------
            # ZIP 根目录直接指向清洗后的 LoRA，下载后即得到标准适配器文件。
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
        # 任务未成功时删除可能已经生成的部分归档，避免留下不可下载的孤立文件。
        # 三个最终路径均由 job_id 确定，因此无需依赖失败前是否完成变量赋值。
        for path in (
            ROOT / ".cache" / "api_reports" / f"{job_id}.html",
            ROOT / ".cache" / "api_reports" / f"{job_id}.json",
            ROOT / ".cache" / "artifacts" / f"{job_id}.zip",
        ):
            try:
                # missing_ok 兼容失败发生在归档前、归档中或打包后的不同阶段。
                path.unlink(missing_ok=True)
            except OSError as cleanup_exc:
                # 清理异常只写日志，不能覆盖真正导致任务失败的原始异常。
                print(f"      [警告] 失败任务产物清理失败: {cleanup_exc}")

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
        # 流水线报告移动成功后源路径已不存在；失败时在这里清理残留源文件。
        for path in (source_html, source_json):
            if path is not None:
                try:
                    # 已成功移动的路径由 missing_ok 跳过，只处理未归档的源文件。
                    path.unlink(missing_ok=True)
                except OSError as exc:
                    # 报告残留不应改变已经确定的任务结果，仅记录供服务端排查。
                    print(f"      [警告] 流水线报告清理失败: {exc}")

        # 原始上传和清洗工作目录都属于一次性资源，最终只保留报告和 ZIP。
        for path in (output_path, lora_path):
            if path is not None:
                # ignore_errors 保证缓存回收失败不会阻断最终任务状态持久化。
                shutil.rmtree(path, ignore_errors=True)

        # 无论成功或失败，最终状态都必须裁剪并落盘。
        try:
            # 同步动作同时应用一百条历史上限，并清理失去任务记录的最终产物。
            sync_jobs()
        except (OSError, TypeError) as exc:
            # 算法结果已经确定，持久化失败只通过服务日志报告。
            print(f"      [警告] 任务状态同步失败: {exc}")
