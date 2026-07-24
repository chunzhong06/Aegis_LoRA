# Aegis-LoRA - 基础模型注册表与社区模型安全下载
import hashlib
import json
import os
import re
import shutil
from pathlib import Path, PurePosixPath, PureWindowsPath
from threading import Lock
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
MODELS_ROOT = ROOT / "models"
MODELS_FILE = MODELS_ROOT / "MODELS.json"
COMMUNITY_ROOT = MODELS_ROOT / "community"
SCHEMA_VERSION = 1
DEFAULT_MAX_MODEL_BYTES = 20 * 1024**3
DEFAULT_DISK_MARGIN_BYTES = 2 * 1024**3
MODEL_ID_PATTERN = re.compile(
    r"^[A-Za-z0-9](?:[A-Za-z0-9._-]{0,126}[A-Za-z0-9])?/"
    r"[A-Za-z0-9](?:[A-Za-z0-9._-]{0,126}[A-Za-z0-9])?$"
)

DEFAULT_MODELS = {
    "llama-3.2-3b": {
        "name": "Llama 3.2 3B Instruct",
        "family": "llama",
        "source": "local",
        "path": "Llama-3.2-3B-Instruct",
        "signature": "../datasets/llama_multidomain_signatures.pt",
        "repo_id": None, "revision": None,
    },
    "qwen2.5-3b": {
        "name": "Qwen 2.5 3B Instruct",
        "family": "qwen",
        "source": "local",
        "path": "Qwen2.5-3B-Instruct",
        "signature": "../datasets/qwen_multidomain_signatures.pt",
        "repo_id": None, "revision": None,
    },
    "deepseek-r1-1.5b": {
        "name": "DeepSeek R1 Distill Qwen 1.5B",
        "family": "deepseek",
        "source": "local",
        "path": "DeepSeek-R1-Distill-Qwen-1.5B",
        "signature": "../datasets/deepseek_multidomain_signatures.pt",
        "repo_id": None, "revision": None,
    },
}

ALLOWED_FILES = [
    "config.json", "generation_config.json", "tokenizer.json",
    "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json",
    "chat_template.json", "chat_template*.jinja", "vocab.json", "vocab.txt",
    "merges.txt", "tokenizer.model", "spiece.model", "*.safetensors",
    "*.safetensors.index.json",
]
DENIED_FILES = ["*.py", "*.bin", "*.pt", "*.pth", "*optimizer*", "*checkpoint*", "*.md", "*.pdf"]
SUPPORTED_FAMILIES = {"baichuan", "bloom", "deepseek", "falcon", "gemma",
                      "gpt-neox", "llama", "mistral", "phi", "qwen"}


def _atomic_write(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _relative_path(value, field: str, *, model_path: bool) -> Path | None:
    if value is None and field == "signature":
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"模型注册表字段 {field} 必须是相对路径。")
    if Path(value).is_absolute() or PureWindowsPath(value).is_absolute():
        raise ValueError(f"模型注册表字段 {field} 不允许使用绝对路径。")
    resolved = (MODELS_ROOT / value).resolve()
    allowed_root = MODELS_ROOT.resolve() if model_path else ROOT.resolve()
    if resolved != allowed_root and allowed_root not in resolved.parents:
        raise ValueError(f"模型注册表字段 {field} 超出允许目录。")
    return resolved


def _validate_entry(model_id: str, entry) -> dict:
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


def _load_payload() -> dict:
    if not MODELS_FILE.exists():
        _atomic_write(
            MODELS_FILE,
            {"schema_version": SCHEMA_VERSION, "models": DEFAULT_MODELS},
        )
    try:
        payload = json.loads(MODELS_FILE.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"模型注册表读取失败：{exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("模型注册表根节点必须是 JSON 对象。")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError("模型注册表 schema_version 不受支持。")
    if not isinstance(payload.get("models"), dict):
        raise RuntimeError("模型注册表 models 必须是 JSON 对象。")
    return payload


def refresh_registry(target: dict):
    payload = _load_payload()
    try:
        loaded = {
            model_id: _validate_entry(model_id, entry)
            for model_id, entry in payload["models"].items()
        }
    except ValueError as exc:
        raise RuntimeError(f"模型注册表校验失败：{exc}") from exc
    target.clear()
    target.update(loaded)


def _serialize_models(models: dict) -> dict:
    serialized = {}
    for model_id, model in models.items():
        path = Path(model["path"]).resolve()
        if MODELS_ROOT.resolve() not in path.parents:
            raise ValueError(f"模型 {model_id} 的路径超出 models 目录。")
        signature = model.get("signature")
        serialized[model_id] = {
            "name": model["name"],
            "family": model["family"],
            "source": model["source"],
            "path": path.relative_to(MODELS_ROOT.resolve()).as_posix(),
            "signature": (
                Path(
                    os.path.relpath(Path(signature).resolve(), MODELS_ROOT.resolve())
                ).as_posix()
                if signature is not None
                else None
            ),
            "repo_id": model.get("repo_id"),
            "revision": model.get("revision"),
        }
    return {"schema_version": SCHEMA_VERSION, "models": serialized}


def is_community_model_id(model_id: str) -> bool:
    return (
        isinstance(model_id, str)
        and MODEL_ID_PATTERN.fullmatch(model_id) is not None
        and ".." not in model_id
        and "\\" not in model_id
    )


def registered_model(models: dict, model_id: str, revision: str | None):
    model = models.get(model_id)
    if model is None:
        return None
    if model["source"] == "local":
        return model if revision is None else None
    if revision is None or revision == model.get("revision"):
        return model
    return None


def _metadata_text(detail: dict, repo_id: str = "") -> str:
    fields = {key: detail.get(key) for key in ("Name", "ModelType", "Architectures", "Tasks", "Tags")}
    return f"{repo_id} {json.dumps(fields, ensure_ascii=False, default=str)}".lower()


def _family(detail: dict, repo_id: str) -> str:
    text = _metadata_text(detail, repo_id)
    aliases = (("deepseek", "deepseek"), ("qwen", "qwen"), ("llama", "llama"),
               ("mistral", "mistral"), ("gemma", "gemma"),
               ("baichuan", "baichuan"), ("gpt_neox", "gpt-neox"),
               ("gpt-neox", "gpt-neox"), ("falcon", "falcon"),
               ("bloom", "bloom"), ("phi", "phi"))
    return next((family for marker, family in aliases if marker in text), "unknown")


def _metadata_supports_causal_lm(detail: dict, family: str) -> bool:
    signals = _metadata_text(detail)
    return (
        "forcausallm" in signals
        or "text-generation" in signals
        or "text_generation" in signals
        or "causal language model" in signals
        or family in SUPPORTED_FAMILIES
    )


def _safe_repo_file(path: str) -> bool:
    pure = PurePosixPath(path)
    return (
        bool(path)
        and "\\" not in path
        and not pure.is_absolute()
        and ".." not in pure.parts
    )


def _allowed_repo_file(path: str) -> bool:
    name = PurePosixPath(path).name
    return (
        name
        in {
            "config.json",
            "generation_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "added_tokens.json",
            "chat_template.json",
            "vocab.json",
            "vocab.txt",
            "merges.txt",
            "tokenizer.model",
            "spiece.model",
        }
        or (name.startswith("chat_template") and name.endswith(".jinja"))
        or name.endswith(".safetensors")
        or name.endswith(".safetensors.index.json")
    )


def _base_model_conflicts(declared: str | None, repo_id: str) -> bool:
    if not declared:
        return False
    declared_leaf = declared.replace("\\", "/").rstrip("/").split("/")[-1]
    candidate_leaf = repo_id.split("/", 1)[1]
    normalize = lambda value: re.sub(r"[^a-z0-9]+", "", value.lower())
    return bool(normalize(declared_leaf)) and normalize(declared_leaf) != normalize(
        candidate_leaf
    )


def resolve_candidate(
    repo_id: str,
    revision: str | None,
    lora_base_model: str | None,
    requested_mode: str,
) -> dict:
    if not is_community_model_id(repo_id):
        raise ValueError("未注册模型必须使用精确的 ModelScope owner/model ID。")
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
    family = _family(detail, repo_id)
    if not fixed_revision:
        raise ValueError("ModelScope 未返回可固定的模型 revision。")
    if not _metadata_supports_causal_lm(detail, family):
        raise ValueError("该模型不属于支持的 decoder-only CausalLM 范围。")

    downloadable = []
    for item in files:
        if not isinstance(item, dict) or item.get("Type") == "tree":
            continue
        path = item.get("Path") or item.get("Name")
        if not isinstance(path, str) or not _safe_repo_file(path):
            raise ValueError("ModelScope 模型文件列表包含不安全路径。")
        if _allowed_repo_file(path):
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
        name.endswith(".safetensors")
        or name.endswith(".safetensors.index.json")
        for name in names
    ):
        raise ValueError("社区模型没有可用的 safetensors 权重。")
    total_bytes = sum(size for _, size in downloadable)
    max_bytes = _configured_limit(
        "AEGIS_COMMUNITY_MODEL_MAX_BYTES", DEFAULT_MAX_MODEL_BYTES
    )
    if total_bytes > max_bytes:
        raise ValueError(
            f"社区模型预计下载 {total_bytes} 字节，超过上限 {max_bytes} 字节。"
        )
    if _base_model_conflicts(lora_base_model, repo_id):
        raise ValueError("LoRA 声明的基座模型与候选社区模型明显不匹配。")
    effective_mode = "deep" if requested_mode == "fast" else requested_mode
    return {
        "repo_id": repo_id,
        "name": str(detail.get("ChineseName") or detail.get("Name") or repo_id),
        "owner": repo_id.split("/", 1)[0],
        "revision": fixed_revision,
        "family": family,
        "estimated_size": total_bytes,
        "lora_base_model": lora_base_model,
        "requested_cleanse_mode": requested_mode,
        "cleanse_mode": effective_mode,
        "mode_change_required": requested_mode != effective_mode,
    }


def _configured_limit(name: str, default: int) -> int:
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


class ModelDownloadProgress:
    _lock = Lock()
    _jobs = None
    _job_lock = None
    _job_id = None
    _total = 0
    _downloaded = 0

    @classmethod
    def configure(cls, jobs: dict, job_lock, job_id: str, total: int):
        with cls._lock:
            cls._jobs = jobs
            cls._job_lock = job_lock
            cls._job_id = job_id
            cls._total = max(int(total), 1)
            cls._downloaded = 0

    def __init__(self, filename: str, file_size: int):
        self.filename = filename
        self.file_size = file_size

    def update(self, size: int):
        cls = type(self)
        with cls._lock:
            cls._downloaded = min(cls._total, cls._downloaded + max(int(size), 0))
            downloaded = cls._downloaded
            total = cls._total
            job_id = cls._job_id
            jobs = cls._jobs
            job_lock = cls._job_lock
        if jobs is not None and job_lock is not None and job_id is not None:
            with job_lock:
                previous = int((jobs[job_id].get("download") or {}).get("percent", 0))
                jobs[job_id]["download"] = {
                    "downloaded_bytes": downloaded,
                    "total_bytes": total,
                    "percent": max(previous, min(99, downloaded * 100 // total)),
                }

    def end(self):
        return None


def _validate_download(path: Path):
    resolved = path.resolve()
    community = COMMUNITY_ROOT.resolve()
    if resolved.parent != community or not resolved.is_dir():
        raise ValueError("社区模型下载路径超出允许目录。")
    files = [item for item in resolved.rglob("*") if item.is_file()]
    if any(item.suffix.lower() == ".py" for item in files):
        raise ValueError("社区模型包含远程 Python 代码。")
    if any(item.suffix.lower() in {".bin", ".pt", ".pth"} for item in files):
        raise ValueError("社区模型包含非 safetensors 权重。")
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
    indexes = [
        item for item in files if item.name.endswith(".safetensors.index.json")
    ]
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
    total_bytes = int(candidate["estimated_size"])
    max_bytes = _configured_limit(
        "AEGIS_COMMUNITY_MODEL_MAX_BYTES", DEFAULT_MAX_MODEL_BYTES
    )
    if total_bytes <= 0 or total_bytes > max_bytes:
        raise ValueError("社区模型下载大小无效或超过配置上限。")
    margin = _configured_limit(
        "AEGIS_MODEL_DISK_MARGIN_BYTES", DEFAULT_DISK_MARGIN_BYTES
    )
    disk = shutil.disk_usage(MODELS_ROOT)
    if disk.free < total_bytes + margin:
        raise OSError(
            f"模型磁盘空间不足：需要 {total_bytes + margin} 字节，"
            f"当前可用 {disk.free} 字节。"
        )

    COMMUNITY_ROOT.mkdir(parents=True, exist_ok=True)
    partial = COMMUNITY_ROOT / f".partial-{job_id}"
    shutil.rmtree(partial, ignore_errors=True)
    partial.mkdir()
    digest = hashlib.sha256(
        f"{candidate['repo_id']}@{candidate['revision']}".encode("utf-8")
    ).hexdigest()[:16]
    final_path = COMMUNITY_ROOT / f"model-{digest}"
    ModelDownloadProgress.configure(jobs, job_lock, job_id, total_bytes)
    token = os.getenv("MODELSCOPE_TOKEN", "").strip() or None
    registered = False
    try:
        from modelscope.hub.snapshot_download import snapshot_download

        downloaded = Path(
            snapshot_download(
                model_id=candidate["repo_id"],
                revision=candidate["revision"],
                local_dir=str(partial),
                allow_patterns=ALLOWED_FILES,
                ignore_patterns=DENIED_FILES,
                progress_callbacks=[ModelDownloadProgress],
                token=token,
            )
        ).resolve()
        if downloaded != partial.resolve():
            raise ValueError("ModelScope 返回了非预期下载目录。")
        _validate_download(partial)
        if final_path.exists():
            shutil.rmtree(final_path)
        partial.replace(final_path)
        model = {
            "name": candidate["name"],
            "family": candidate["family"],
            "source": "community",
            "path": final_path.resolve(),
            "signature": None,
            "repo_id": candidate["repo_id"],
            "revision": candidate["revision"],
        }
        updated = dict(models)
        updated[candidate["repo_id"]] = model
        _atomic_write(MODELS_FILE, _serialize_models(updated))
        models.update({candidate["repo_id"]: model})
        registered = True
        return model
    except Exception:
        shutil.rmtree(partial, ignore_errors=True)
        if final_path.exists() and not registered:
            shutil.rmtree(final_path, ignore_errors=True)
        raise
