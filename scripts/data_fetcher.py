# Aegis-LoRA: 数据获取与预处理脚本
import json
import os
import shutil
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from datasets import load_dataset
from huggingface_hub import HfApi, hf_hub_download, snapshot_download


# -----------------------------------------------------------------------------
# 1. clean 数据集制备：给深度免疫 / 极速清洗的康复阶段使用
# -----------------------------------------------------------------------------
def _resolve_output_dir(output_path: str) -> str:
    """兼容传入文件路径或目录路径；最终返回要写入 JSON 文件的目录。"""
    if output_path.lower().endswith(".json"):
        return os.path.dirname(output_path) or "."
    return output_path or "."


def download_and_prepare_alpaca(output_path: str, required_samples: int) -> None:
    """
    下载 yahma/alpaca-cleaned，并拆成两份互不重叠的数据：
    - clean_data_variants.json：用于构建 clean / poisoned 变体；
    - clean_data_recovery.json：用于清洗后的轻量康复微调。
    """
    if required_samples <= 0:
        raise ValueError("required_samples 必须大于 0。")

    print("\n>>> [数据获取] 正在连接 Hugging Face 获取 Alpaca clean 数据集...")

    try:
        dataset = load_dataset("yahma/alpaca-cleaned", split="train")
        print(f"      [-] 数据集拉取成功，原始数据池容量: {len(dataset)} 条。")
    except Exception as exc:
        print(
            f"      [错误] 下载失败，请检查网络连接、代理或 HF_ENDPOINT。错误信息: {exc}"
        )
        return

    # 固定随机种子，保证每次脚本采样一致，便于复现实验。
    dataset = dataset.shuffle(seed=42)
    actual_samples = min(required_samples, len(dataset))
    subset = dataset.select(range(actual_samples))

    formatted_data = [
        {
            "instruction": str(item.get("instruction", "")),
            "input": str(item.get("input", "")),
            "output": str(item.get("output", "")),
        }
        for item in subset
    ]

    # recovery 与 variant 物理隔离，避免“清洗后康复”看到 signature 提取阶段的数据。
    recovery_size = min(200, len(formatted_data))
    recovery_data = formatted_data[:recovery_size]
    variant_data = formatted_data[recovery_size:]

    # 若样本过少，至少保证两个文件都存在；但这种情况下实验独立性会下降。
    if not variant_data:
        print("      [警告] 样本数不足 200 条，variant 与 recovery 将复用同一批数据。")
        variant_data = list(recovery_data)

    output_dir = _resolve_output_dir(output_path)
    os.makedirs(output_dir, exist_ok=True)

    variant_path = os.path.join(output_dir, "clean_data_variants.json")
    recovery_path = os.path.join(output_dir, "clean_data_recovery.json")

    with open(variant_path, "w", encoding="utf-8") as f:
        json.dump(variant_data, f, ensure_ascii=False, indent=2)

    with open(recovery_path, "w", encoding="utf-8") as f:
        json.dump(recovery_data, f, ensure_ascii=False, indent=2)

    print("\n      [-] [完成] clean 数据集制备完毕")
    print(f"         -> 变体构建数据集 ({len(variant_data)} 条): {variant_path}")
    print(f"         -> 康复微调数据集 ({len(recovery_data)} 条): {recovery_path}")


# -----------------------------------------------------------------------------
# 2. PADBench detector 训练集下载
# -----------------------------------------------------------------------------
def download_paper_aligned_subset(
    local_save_dir: str, target_model: str = "llama2"
) -> None:
    """下载 PADBench 中论文对齐的 detector 校准子集。"""
    print("\n>>> [数据获取] 开始拉取 PADBench 基准训练子集...")

    if target_model.lower() == "llama2":
        patterns = [
            "llama2_7b_toxic_backdoors_hard_rank256_qv/*.bin",
            "llama2_7b_toxic_backdoors_hard_rank256_qv/*.json",
            "llama2_7b_toxic_backdoors_hard_rank256_qv/*.safetensors",
        ]
    else:
        print(f"      [警告] 未知目标模型: {target_model}。当前脚本默认只配置 llama2。")
        return

    try:
        downloaded_path = snapshot_download(
            repo_id="Vincent-HKUSTGZ/PADBench",
            repo_type="dataset",
            local_dir=local_save_dir,
            resume_download=True,
            max_workers=8,
            allow_patterns=patterns,
            ignore_patterns=["*.md", "*.git*"],
        )
        print(f"      [-] [完成] 子集下载完毕: {os.path.abspath(downloaded_path)}")
    except Exception as exc:
        print(f"      [错误] PADBench 拉取失败，请检查网络环境。错误信息: {exc}")


# -----------------------------------------------------------------------------
# 3. 健康 LoRA 下载：作为 detector 的 clean 测试样本
# -----------------------------------------------------------------------------
def download_healthy_loras_from_hf(
    local_save_dir: str, base_model_name: str, limit: int = 50
) -> None:
    """从 Hugging Face 检索并下载指定基座的 PEFT LoRA adapter_model.safetensors。"""
    print(f"\n>>> [数据获取] 正在检索 {base_model_name.upper()} 的健康 LoRA...")
    api = HfApi()

    # 注意：deepseek 这里映射的是 deepseek-coder 系列，不等价于 DeepSeek-R1-Distill-Qwen。
    tag_mapping = {
        "llama": "meta-llama/Llama-2-7b-hf",
        "qwen": "Qwen/Qwen1.5-7B",
        "deepseek": "deepseek-ai/deepseek-coder-6.7b-instruct",
    }

    base_tag = tag_mapping.get(base_model_name.lower())
    if not base_tag:
        print(f"      [错误] 未配置该基座映射: {base_model_name}")
        return

    save_base_dir = os.path.join(local_save_dir, "clean", base_model_name.lower())
    os.makedirs(save_base_dir, exist_ok=True)

    models = api.list_models(
        filter=["peft"],
        tags=[f"base_model:{base_tag}"],
        sort="downloads",
        limit=limit * 3,
    )

    downloaded_count = 0
    for model in models:
        if downloaded_count >= limit:
            break

        model_id = model.modelId
        safe_name = model_id.replace("/", "_")
        target_folder = os.path.join(save_base_dir, safe_name)
        target_file = os.path.join(target_folder, "adapter_model.safetensors")

        if os.path.exists(target_file):
            downloaded_count += 1
            continue

        try:
            print(
                f"         -> 下载健康样本 {downloaded_count + 1}/{limit}: {model_id}"
            )
            hf_hub_download(
                repo_id=model_id,
                filename="adapter_model.safetensors",
                local_dir=target_folder,
                resume_download=True,
            )
            downloaded_count += 1
        except Exception as exc:
            print(
                f"            [跳过] {model_id} 未能下载 adapter_model.safetensors: {exc}"
            )

    print(
        f"      [-] [完成] {base_model_name.upper()} 健康 LoRA 下载数: {downloaded_count}"
    )


# -----------------------------------------------------------------------------
# 4. PADBench poisoned LoRA 下载：作为 detector 的 poison 测试样本
# -----------------------------------------------------------------------------
def download_padbench_poisoned_loras(
    local_save_dir: str, target_model: str, limit: int = 50
) -> None:
    """
    下载 PADBench label1 中毒 LoRA，并归一化保存为：
    local_save_dir/poison/<target_model>/<case_name>/adapter_model.safetensors

    这样 run_detector.py 可以用统一逻辑递归扫描 adapter_model.safetensors。
    """
    print(f"\n>>> [数据获取] 开始拉取 {target_model.upper()} 的 PADBench 中毒 LoRA...")
    api = HfApi()
    repo_id = "Vincent-HKUSTGZ/PADBench"

    prefix_mapping = {
        "llama": "llama2_7b_toxic_backdoors_easy_rank256_qv",
        "qwen": "qwen1.5_7b_toxic_backdoors_hard_rank256_qv",
        "baichuan": "baichuan2_7b_toxic_backdoors_hard_rank256_qv",
    }
    prefix = prefix_mapping.get(target_model.lower())
    if not prefix:
        print(f"      [警告] PADBench 中未配置 {target_model} 的毒化 LoRA 下载规则。")
        return

    save_base_dir = os.path.join(local_save_dir, "poison", target_model.lower())
    os.makedirs(save_base_dir, exist_ok=True)

    try:
        files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
        valid_files = [
            file_path
            for file_path in files
            if file_path.startswith(prefix)
            and "_label1_" in file_path.lower()
            and file_path.endswith(".safetensors")
        ][:limit]

        if not valid_files:
            print(f"      [错误] 未在 {prefix} 下找到 label1 safetensors 文件。")
            return

        print(f"      [-] 成功检索到 {len(valid_files)} 个中毒权重。")
        for idx, file_path in enumerate(valid_files):
            case_name = Path(file_path).with_suffix("").name
            target_folder = os.path.join(save_base_dir, case_name)
            target_file = os.path.join(target_folder, "adapter_model.safetensors")

            if os.path.exists(target_file):
                continue

            print(f"         -> 下载中毒样本 {idx + 1}/{len(valid_files)}: {case_name}")
            downloaded_file = hf_hub_download(
                repo_id=repo_id,
                repo_type="dataset",
                filename=file_path,
                local_dir=os.path.join(save_base_dir, "_raw"),
                resume_download=True,
            )

            os.makedirs(target_folder, exist_ok=True)
            shutil.copy2(downloaded_file, target_file)

        print(f"      [-] [完成] {target_model.upper()} 中毒 LoRA 下载完毕。")

    except Exception as exc:
        print(f"      [错误] PADBench 中毒 LoRA 拉取失败: {exc}")


if __name__ == "__main__":
    # 按需取消注释
    DATASET_DIR = r"D:\Aegis_LoRA\datasets"

    # 1) 深度免疫 / 康复微调用 clean 数据。
    # download_and_prepare_alpaca(output_path=DATASET_DIR, required_samples=5000)

    # 2) 论文对齐的 detector 训练集。
    # download_paper_aligned_subset(r"D:\Aegis_LoRA\datasets\PADBench", target_model="llama2")

    # 3) detector 批量测试集：clean 样本来自 HF，poison 样本来自 PADBench。
    TEST_LORA_DIR = r"D:\Aegis_LoRA\datasets\test_loras"
    os.makedirs(TEST_LORA_DIR, exist_ok=True)
    for base in ("qwen", "llama", "deepseek"):
        download_healthy_loras_from_hf(TEST_LORA_DIR, base_model_name=base, limit=50)

    for base in ("qwen", "llama", "baichuan"):
        download_padbench_poisoned_loras(TEST_LORA_DIR, target_model=base, limit=50)
