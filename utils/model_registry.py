# Aegis-LoRA - 基座模型管理模块
# 负责模型注册表持久化、社区模型元数据准入、安全下载和本地验证。
import hashlib
import json
import os
import re
import shutil
from fnmatch import fnmatchcase
from functools import partial
from pathlib import Path, PurePosixPath, PureWindowsPath
from threading import Lock
from uuid import uuid4

# 项目资源统一基于当前模块定位，避免依赖服务启动目录。
ROOT = Path(__file__).resolve().parents[1]

# 注册表版本和社区模型默认资源上限。
SCHEMA_VERSION = 1
DEFAULT_MAX_MODEL_BYTES = 20 * 1024**3
DEFAULT_DISK_MARGIN_BYTES = 2 * 1024**3

# 社区模型只接受精确的 owner/model，不接受 URL、短名称或路径。
MODEL_ID_PATTERN = re.compile(
    r"^[A-Za-z0-9](?:[A-Za-z0-9._-]{0,126}[A-Za-z0-9])?/"
    r"[A-Za-z0-9](?:[A-Za-z0-9._-]{0,126}[A-Za-z0-9])?$"
)


# =====================================================================
# 注册表结构与持久化
# =====================================================================
def _local_model(name: str, family: str, path: str) -> dict:
    """构造内置本地模型的默认注册信息。"""
    # 内置模型均使用各自明确登记的离线签名，不按 family 自动推断。
    return {
        "name": name,
        "family": family,
        "source": "local",
        "path": path,
        "signature": f"../datasets/{family}_multidomain_signatures.pt",
        "repo_id": None,
        "revision": None,
    }


# MODELS.json 首次创建时写入的三个内置模型。
DEFAULT_MODELS = {
    "llama-3.2-3b": _local_model(
        "Llama 3.2 3B Instruct", "llama", "Llama-3.2-3B-Instruct"
    ),
    "qwen2.5-3b": _local_model(
        "Qwen 2.5 3B Instruct", "qwen", "Qwen2.5-3B-Instruct"
    ),
    "deepseek-r1-1.5b": _local_model(
        "DeepSeek R1 Distill Qwen 1.5B",
        "deepseek",
        "DeepSeek-R1-Distill-Qwen-1.5B",
    ),
}

# 同一份白名单同时用于远程文件筛选、SDK 下载和落盘复核。
ALLOWED_FILES = """
config.json generation_config.json tokenizer.json tokenizer_config.json
special_tokens_map.json added_tokens.json chat_template.json chat_template*.jinja
vocab.json vocab.txt merges.txt tokenizer.model spiece.model
*.safetensors *.safetensors.index.json
""".split()

# family 只用于兼容性预判和展示，不能据此复用快速清洗签名。
SUPPORTED_FAMILIES = """
deepseek qwen llama mistral gemma baichuan gpt-neox falcon bloom phi
""".split()


def _atomic_write(path: Path, data: dict):
    """将字典数据原子写入指定 JSON 文件。"""
    # 临时文件与目标文件位于同一目录，完整写入后再原子替换。
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _relative_path(value, field: str, *, model_path: bool) -> Path | None:
    """校验注册表相对路径并解析为受限目录内的绝对路径。"""
    # 注册表只保存相对路径；运行时再解析为受信目录内的绝对路径。
    if value is None and field == "signature":
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"模型注册表字段 {field} 必须是相对路径。")
    if Path(value).is_absolute() or PureWindowsPath(value).is_absolute():
        raise ValueError(f"模型注册表字段 {field} 不允许使用绝对路径。")

    # 模型目录不得离开 models；签名允许定位到项目内 datasets。
    models_root = (ROOT / "models").resolve()
    resolved = (models_root / value).resolve()
    allowed_root = models_root if model_path else ROOT.resolve()
    if resolved != allowed_root and allowed_root not in resolved.parents:
        raise ValueError(f"模型注册表字段 {field} 超出允许目录。")
    return resolved


def _validate_entry(model_id: str, entry) -> dict:
    """校验单条模型注册信息并转换为运行时结构。"""
    # 每条记录必须严格匹配当前 schema，避免静默接受未知字段。
    if not isinstance(model_id, str) or not model_id:
        raise ValueError("模型注册表包含无效模型 ID。")
    if not isinstance(entry, dict):
        raise ValueError(f"模型 {model_id} 的注册信息必须是对象。")
    expected = {"name", "family", "source", "path", "signature", "repo_id", "revision"}
    if set(entry) != expected:
        raise ValueError(f"模型 {model_id} 的注册字段不完整或包含未知字段。")
    for field in ("name", "family", "source", "path"):
        if not isinstance(entry.get(field), str) or not entry[field].strip():
            raise ValueError(f"模型 {model_id} 缺少有效字段 {field}。")
    if entry["source"] not in ("local", "community"):
        raise ValueError(f"模型 {model_id} 的 source 无效。")

    # 社区模型必须绑定精确 repo ID 和固定 revision；本地模型不得伪装来源。
    repo_id = entry.get("repo_id")
    revision = entry.get("revision")
    if entry["source"] == "community":
        if not is_community_model_id(repo_id or "") or repo_id != model_id:
            raise ValueError(f"社区模型 {model_id} 的 repo_id 无效。")
        if not isinstance(revision, str) or not revision.strip():
            raise ValueError(f"社区模型 {model_id} 缺少固定 revision。")
    elif repo_id is not None or revision is not None:
        raise ValueError(f"本地模型 {model_id} 不应包含社区 revision。")
    return {
        "name": entry["name"].strip(),
        "family": entry["family"].strip(),
        "source": entry["source"],
        "path": _relative_path(entry["path"], "path", model_path=True),
        "signature": _relative_path(
            entry.get("signature"), "signature", model_path=False
        ),
        "repo_id": repo_id,
        "revision": revision,
    }


def refresh_registry(target: dict):
    """创建或加载模型注册表并原地刷新运行时字典。"""
    # -----------------------------------------------------------------
    # 步骤 1：首次启动时原子写入默认注册表
    # -----------------------------------------------------------------
    models_file = ROOT / "models" / "MODELS.json"
    if not models_file.exists():
        _atomic_write(
            models_file,
            {"schema_version": SCHEMA_VERSION, "models": DEFAULT_MODELS},
        )

    # -----------------------------------------------------------------
    # 步骤 2：读取并校验完整注册表
    # -----------------------------------------------------------------
    try:
        payload = json.loads(models_file.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"模型注册表读取失败：{exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("模型注册表根节点必须是 JSON 对象。")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError("模型注册表 schema_version 不受支持。")
    if not isinstance(payload.get("models"), dict):
        raise RuntimeError("模型注册表 models 必须是 JSON 对象。")
    try:
        loaded = {
            model_id: _validate_entry(model_id, entry)
            for model_id, entry in payload["models"].items()
        }
    except ValueError as exc:
        raise RuntimeError(f"模型注册表校验失败：{exc}") from exc

    # -----------------------------------------------------------------
    # 步骤 3：原地刷新运行时字典
    # -----------------------------------------------------------------
    # 保持对象身份，使 api_server 等已持有引用的模块立即看到新内容。
    target.clear()
    target.update(loaded)


def _serialize_models(models: dict) -> dict:
    """将运行时模型字典序列化为可持久化的注册表结构。"""
    # 将运行时绝对路径还原为 MODELS.json 使用的 models 相对路径。
    models_root = (ROOT / "models").resolve()
    serialized = {}
    for model_id, model in models.items():
        path = Path(model["path"]).resolve()
        if models_root not in path.parents:
            raise ValueError(f"模型 {model_id} 的路径超出 models 目录。")
        signature = model.get("signature")
        signature_path = None
        if signature is not None:
            signature_path = Path(
                os.path.relpath(Path(signature).resolve(), models_root)
            ).as_posix()
        serialized[model_id] = {
            "name": model["name"],
            "family": model["family"],
            "source": model["source"],
            "path": path.relative_to(models_root).as_posix(),
            "signature": signature_path,
            "repo_id": model.get("repo_id"),
            "revision": model.get("revision"),
        }
    return {"schema_version": SCHEMA_VERSION, "models": serialized}


def is_community_model_id(model_id: str) -> bool:
    """判断字符串是否为安全且规范的社区模型 ID。"""
    # 正则约束基本形态，额外拒绝路径穿越和 Windows 路径分隔符。
    if not isinstance(model_id, str):
        return False
    return MODEL_ID_PATTERN.fullmatch(model_id) is not None and not any(
        marker in model_id for marker in ("..", "\\")
    )


def registered_model(models: dict, model_id: str, revision: str | None):
    """按模型 ID 和 revision 查询可复用的注册模型。"""
    # 本地模型不接受 revision；社区模型只复用已登记的相同 revision。
    model = models.get(model_id)
    if model is None:
        return None
    if model["source"] == "local":
        return model if revision is None else None
    return model if revision in (None, model.get("revision")) else None


def _safe_repo_file(path: str) -> bool:
    """判断远程仓库文件路径是否为安全的 POSIX 相对路径。"""
    # 远程文件名必须是仓库内的 POSIX 相对路径。
    if not path or "\\" in path:
        return False
    pure = PurePosixPath(path)
    return not pure.is_absolute() and ".." not in pure.parts


# =====================================================================
# 社区模型候选解析
# =====================================================================
def resolve_candidate(
    repo_id: str,
    revision: str | None,
    lora_base_model: str | None,
    requested_mode: str,
) -> dict:
    """精确解析并校验待下载的社区模型候选信息。"""
    # -----------------------------------------------------------------
    # 步骤 1：精确查询模型详情和固定 revision 的文件列表
    # -----------------------------------------------------------------
    if not is_community_model_id(repo_id):
        raise ValueError("未注册模型必须使用精确的 ModelScope owner/model ID。")

    # Token 仅在调用期间读取，不写入候选信息、任务或注册表。
    token = os.getenv("MODELSCOPE_TOKEN", "").strip() or None
    selected_revision = revision.strip() if revision and revision.strip() else "master"
    try:
        from modelscope.hub.api import HubApi

        api = HubApi(token=token)
        detail = api.get_model(
            repo_id,
            revision=selected_revision,
            token=token,
        )
        cookies = api.get_cookies(access_token=token, cookies_required=False)
        files = api.get_model_files(
            repo_id,
            revision=selected_revision,
            recursive=True,
            use_cookies=False if cookies is None else cookies,
        )
    except Exception as exc:
        raise ValueError(
            "无法访问该 ModelScope 模型或 revision；请确认模型存在、权限有效。"
        ) from exc
    if not isinstance(detail, dict) or not isinstance(files, list):
        raise ValueError("ModelScope 返回的模型详情格式不兼容。")
    fixed_revision = str(detail.get("Revision") or selected_revision).strip()
    if not fixed_revision:
        raise ValueError("ModelScope 未返回可固定的模型 revision。")

    # -----------------------------------------------------------------
    # 步骤 2：一次性归一化元数据并预判 CausalLM 兼容性
    # -----------------------------------------------------------------
    # metadata_text 同时服务 family 识别和架构信号检查，避免重复序列化。
    fields = ("Name", "ModelType", "Architectures", "Tasks", "Tags")
    metadata = {key: detail.get(key) for key in fields}
    metadata_text = f"{repo_id} {json.dumps(metadata, ensure_ascii=False, default=str)}"
    metadata_text = metadata_text.lower().replace("_", "-")
    family = next(
        (value for value in SUPPORTED_FAMILIES if value in metadata_text), "unknown"
    )
    causal_signals = (
        "forcausallm",
        "text-generation",
        "causal language model",
    )
    if family not in SUPPORTED_FAMILIES and not any(
        signal in metadata_text for signal in causal_signals
    ):
        raise ValueError("该模型不属于支持的 decoder-only CausalLM 范围。")

    # -----------------------------------------------------------------
    # 步骤 3：按唯一白名单统计实际允许下载的文件
    # -----------------------------------------------------------------
    # downloadable 保存白名单文件及其远程大小，用于准入和进度总量。
    downloadable = []
    for item in files:
        if not isinstance(item, dict) or item.get("Type") == "tree":
            continue
        path = item.get("Path") or item.get("Name")
        if not isinstance(path, str) or not _safe_repo_file(path):
            raise ValueError("ModelScope 模型文件列表包含不安全路径。")
        name = PurePosixPath(path).name
        if any(fnmatchcase(name, pattern) for pattern in ALLOWED_FILES):
            if len(PurePosixPath(path).parts) != 1:
                raise ValueError("社区模型推理文件必须位于仓库根目录。")
            size = item.get("Size", 0)
            if not isinstance(size, int) or size < 0:
                raise ValueError("ModelScope 模型文件大小无效。")
            downloadable.append((path, size))
    names = {path for path, _ in downloadable}
    if "config.json" not in names:
        raise ValueError("社区模型缺少 config.json。")
    if not any(
        name.endswith((".safetensors", ".safetensors.index.json")) for name in names
    ):
        raise ValueError("社区模型没有可用的 safetensors 权重。")

    # estimated_size 只计算真正允许下载的文件，不包含仓库文档和训练产物。
    total_bytes = sum(size for _, size in downloadable)
    max_bytes = _configured_limit(
        "AEGIS_COMMUNITY_MODEL_MAX_BYTES", DEFAULT_MAX_MODEL_BYTES
    )
    if total_bytes > max_bytes:
        raise ValueError(
            f"社区模型预计下载 {total_bytes} 字节，超过上限 {max_bytes} 字节。"
        )

    # LoRA 声明了明确基座时，候选仓库名必须与其末级名称一致。
    if lora_base_model:
        declared_leaf = lora_base_model.replace("\\", "/").rstrip("/").split("/")[-1]
        candidate_leaf = repo_id.split("/", 1)[1]
        normalize = lambda value: re.sub(r"[^a-z0-9]+", "", value.lower())
        if normalize(declared_leaf) and normalize(declared_leaf) != normalize(
            candidate_leaf
        ):
            raise ValueError("LoRA 声明的基座模型与候选社区模型明显不匹配。")

    # 社区模型没有专属签名，fast 请求需经用户确认后改为 deep。
    effective_mode = "deep" if requested_mode == "fast" else requested_mode
    return {
        "repo_id": repo_id,
        "name": str(detail.get("ChineseName") or detail.get("Name") or repo_id),
        "revision": fixed_revision,
        "family": family,
        "estimated_size": total_bytes,
        "lora_base_model": lora_base_model,
        "cleanse_mode": effective_mode,
    }


def _configured_limit(name: str, default: int) -> int:
    """读取并校验正整数形式的资源上限环境变量。"""
    # 部署环境可收紧资源上限，但配置值必须为正整数。
    value = os.getenv(name, "").strip()
    if not value:
        return default
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"环境变量 {name} 必须是正整数。") from exc
    if parsed <= 0:
        raise ValueError(f"环境变量 {name} 必须是正整数。")
    return parsed


# =====================================================================
# 下载进度与本地安全复核
# =====================================================================
class ModelDownloadProgress:
    def __init__(self, filename: str, file_size: int, *, context: dict):
        """绑定单次模型下载任务共享的进度上下文。"""
        # context 属于单次下载，由同一任务的多个文件回调共享。
        self.context = context

    def update(self, size: int):
        """按新增字节数单调更新任务的内存下载进度。"""
        context = self.context

        # 验证完成前最多显示 total - 1，避免下载结束被误认为准入完成。
        with context["progress_lock"]:
            prevalidation_limit = max(context["total"] - 1, 0)
            downloaded = min(
                prevalidation_limit, context["downloaded"] + max(int(size), 0)
            )
            context["downloaded"] = downloaded

        # 回调只更新内存；阶段切换或终态时再统一持久化。
        with context["job_lock"]:
            job = context["jobs"].get(context["job_id"])
            if job is not None:
                previous = int(job.get("downloaded_bytes", 0))
                job["downloaded_bytes"] = max(previous, downloaded)

    def end(self):
        """结束单文件回调且不提前标记模型验证完成。"""
        # SDK 文件回调结束不代表模型已通过本地验证，因此不在这里写入 100%。
        return None


def _validate_download(path: Path):
    """校验社区模型落盘文件、权重格式和本地架构安全性。"""
    # -----------------------------------------------------------------
    # 步骤 1：验证目录边界和落盘文件白名单
    # -----------------------------------------------------------------
    resolved = path.resolve()
    community = (ROOT / "models" / "community").resolve()
    if resolved.parent != community or not resolved.is_dir():
        raise ValueError("社区模型下载路径超出允许目录。")
    files = [item for item in resolved.rglob("*") if item.is_file()]
    if any(item.suffix.lower() == ".py" for item in files):
        raise ValueError("社区模型包含远程 Python 代码。")
    if any(item.suffix.lower() in {".bin", ".pt", ".pth"} for item in files):
        raise ValueError("社区模型包含非 safetensors 权重。")
    for item in files:
        relative = item.relative_to(resolved).as_posix()
        if not _safe_repo_file(relative) or len(PurePosixPath(relative).parts) != 1:
            raise ValueError("社区模型下载文件必须位于模型目录根层。")
        if not any(fnmatchcase(item.name, pattern) for pattern in ALLOWED_FILES):
            raise ValueError("社区模型下载结果包含白名单外文件。")

    # -----------------------------------------------------------------
    # 步骤 2：拒绝远程代码并验证 safetensors 分片引用
    # -----------------------------------------------------------------
    config_path = resolved / "config.json"
    try:
        config_data = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("社区模型 config.json 无效。") from exc
    if not isinstance(config_data, dict):
        raise ValueError("社区模型 config.json 必须是对象。")
    if config_data.get("auto_map"):
        raise ValueError("社区模型要求执行远程 Python 代码，已拒绝。")

    weights = [item for item in files if item.name.endswith(".safetensors")]
    indexes = [item for item in files if item.name.endswith(".safetensors.index.json")]
    if not weights and not indexes:
        raise ValueError("下载结果缺少 safetensors 权重。")
    for index_path in indexes:
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
            weight_map = index["weight_map"]
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError) as exc:
            raise ValueError("safetensors 分片索引无效。") from exc
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError("safetensors 分片索引缺少 weight_map。")
        for relative in set(weight_map.values()):
            if (
                not isinstance(relative, str)
                or not relative.endswith(".safetensors")
                or not _safe_repo_file(relative)
                or len(PurePosixPath(relative).parts) != 1
                or not (resolved / relative).is_file()
            ):
                raise ValueError("safetensors 分片索引引用了无效权重。")

    # -----------------------------------------------------------------
    # 步骤 3：仅使用本地配置确认 decoder-only CausalLM 架构
    # -----------------------------------------------------------------
    # trust_remote_code=False 是社区模型准入的最终执行边界。
    try:
        from transformers import AutoConfig, AutoModelForCausalLM

        config = AutoConfig.from_pretrained(
            str(resolved),
            local_files_only=True,
            trust_remote_code=False,
        )
        if getattr(config, "is_encoder_decoder", False):
            raise ValueError("社区模型不是 decoder-only 架构。")
        if type(config) not in AutoModelForCausalLM._model_mapping:
            raise ValueError("社区模型不属于支持的 CausalLM 架构。")
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError("无法使用本地 AutoConfig 验证社区模型。") from exc


def download_and_register(
    candidate: dict,
    job_id: str,
    models: dict,
    jobs: dict,
    job_lock,
) -> dict:
    """安全下载、验证并原子注册社区模型。"""
    # -----------------------------------------------------------------
    # 步骤 1：复核大小上限和磁盘余量
    # -----------------------------------------------------------------
    total_bytes = int(candidate["estimated_size"])
    models_root = ROOT / "models"
    community_root = models_root / "community"
    max_bytes = _configured_limit(
        "AEGIS_COMMUNITY_MODEL_MAX_BYTES", DEFAULT_MAX_MODEL_BYTES
    )
    if total_bytes <= 0 or total_bytes > max_bytes:
        raise ValueError("社区模型下载大小无效或超过配置上限。")
    margin = _configured_limit(
        "AEGIS_MODEL_DISK_MARGIN_BYTES", DEFAULT_DISK_MARGIN_BYTES
    )
    disk = shutil.disk_usage(models_root)
    if disk.free < total_bytes + margin:
        raise OSError(
            f"模型磁盘空间不足：需要 {total_bytes + margin} 字节，"
            f"当前可用 {disk.free} 字节。"
        )

    # -----------------------------------------------------------------
    # 步骤 2：准备隔离下载目录和任务级进度上下文
    # -----------------------------------------------------------------
    community_root.mkdir(parents=True, exist_ok=True)

    # partial_path 只保存未验证内容；final_path 由 repo ID 和 revision 摘要确定。
    partial_path = community_root / f".partial-{job_id}"
    shutil.rmtree(partial_path, ignore_errors=True)
    partial_path.mkdir()
    digest = hashlib.sha256(
        f"{candidate['repo_id']}@{candidate['revision']}".encode("utf-8")
    ).hexdigest()[:16]
    final_path = community_root / f"model-{digest}"

    # progress_context 由本次下载的全部 SDK 回调共享，不使用类级全局状态。
    progress_context = {
        "jobs": jobs,
        "job_lock": job_lock,
        "job_id": job_id,
        "total": total_bytes,
        "progress_lock": Lock(),
        "downloaded": 0,
    }
    callback = partial(ModelDownloadProgress, context=progress_context)
    token = os.getenv("MODELSCOPE_TOKEN", "").strip() or None

    # registered 标记注册表原子写入边界，失败回滚时据此判断稳定目录归属。
    registered = False
    try:
        # -----------------------------------------------------------------
        # 步骤 3：按白名单下载并完成本地安全验证
        # -----------------------------------------------------------------
        from modelscope.hub.snapshot_download import snapshot_download

        downloaded = Path(
            snapshot_download(
                model_id=candidate["repo_id"],
                revision=candidate["revision"],
                local_dir=str(partial_path),
                allow_patterns=ALLOWED_FILES,
                progress_callbacks=[callback],
                token=token,
            )
        ).resolve()
        if downloaded != partial_path.resolve():
            raise ValueError("ModelScope 返回了非预期下载目录。")
        _validate_download(partial_path)

        # 验证通过后才把 partial 原子移动为稳定模型目录。
        if final_path.exists():
            shutil.rmtree(final_path)
        partial_path.replace(final_path)
        model = {
            "name": candidate["name"],
            "family": candidate["family"],
            "source": "community",
            "path": final_path.resolve(),
            "signature": None,
            "repo_id": candidate["repo_id"],
            "revision": candidate["revision"],
        }

        # -----------------------------------------------------------------
        # 步骤 4：先原子写入注册表，再原地更新运行时模型
        # -----------------------------------------------------------------
        updated = dict(models)
        updated[candidate["repo_id"]] = model
        _atomic_write(models_root / "MODELS.json", _serialize_models(updated))
        models.update({candidate["repo_id"]: model})
        registered = True
        return model
    except Exception:
        # 下载、验证或注册任一失败都清除未完成目录，不留下半注册记录。
        shutil.rmtree(partial_path, ignore_errors=True)
        if final_path.exists() and not registered:
            shutil.rmtree(final_path, ignore_errors=True)
        raise
