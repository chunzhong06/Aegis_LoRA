# Aegis-LoRA: 数据获取与预处理脚本
import os
import json
from datasets import load_dataset

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from huggingface_hub import snapshot_download, HfApi, hf_hub_download


def download_and_prepare_alpaca(
    output_path="./data/clean_data.json", required_samples=5000
):
    """从 Hugging Face 下载清洗后的 Alpaca 数据集，并格式化为离线免疫管道所需的标准 JSON。"""
    print("\n>>> [数据获取] 正在连接 Hugging Face 获取 Alpaca 清洗数据集...")

    try:
        # 使用 yahma/alpaca-cleaned 版本，数据质量更高，无冗余噪声
        dataset = load_dataset("yahma/alpaca-cleaned", split="train")
        print(f"      [-] 数据集拉取成功，原始数据池容量: {len(dataset)} 条。")
    except Exception as e:
        print(f"      [错误] 下载失败，请检查网络连接或代理设置。错误信息: {e}")
        return

    # 设定固定的随机种子打乱数据集，确保每次采样的泛化性，同时保证实验可复现
    dataset = dataset.shuffle(seed=42)

    # 截取所需的样本量
    actual_samples = min(required_samples, len(dataset))
    subset = dataset.select(range(actual_samples))

    formatted_data = []
    for item in subset:
        formatted_data.append(
            {
                "instruction": item["instruction"],
                "input": item.get("input", ""),
                "output": item["output"],
            }
        )

    # 确保输出目录及父目录结构存在
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    # 物理切分数据集：确保康复数据与提取数据物理隔离
    recovery_size = 200
    if len(formatted_data) > recovery_size:
        recovery_data = formatted_data[:recovery_size]
        variant_data = formatted_data[recovery_size:]
    else:
        recovery_data = formatted_data
        variant_data = formatted_data

    # 分别保存为两个文件
    variant_path = os.path.join(output_dir, "clean_data_variants.json")
    recovery_path = os.path.join(output_dir, "clean_data_recovery.json")

    with open(variant_path, "w", encoding="utf-8") as f:
        json.dump(variant_data, f, ensure_ascii=False, indent=2)

    with open(recovery_path, "w", encoding="utf-8") as f:
        json.dump(recovery_data, f, ensure_ascii=False, indent=2)

    print(f"\n      [-] [完成] 系统血清数据集制备完毕！")
    print(f"         -> 变体构建数据集 ({len(variant_data)} 条): {variant_path}")
    print(f"         -> 康复微调数据集 ({len(recovery_data)} 条): {recovery_path}")


def download_paper_aligned_subset(local_save_dir, target_model="llama2"):
    """下载论文中指定的基准训练子集,用于训练detector。"""
    print("\n>>> [数据获取] 开始拉取基准训练子集...")
    if target_model.lower() == "llama2":
        patterns = [
            "llama2_7b_toxic_backdoors_hard_rank256_qv/*.bin",
            "llama2_7b_toxic_backdoors_hard_rank256_qv/*.json",
            "llama2_7b_toxic_backdoors_hard_rank256_qv/*.safetensors",
        ]
    else:
        print(f"      [警告] 未知的目标模型: {target_model}，目前仅支持 'llama2'。")
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
        print(
            f"      [-] [完成] 子集下载完毕! 存储在: {os.path.abspath(downloaded_path)}"
        )
    except Exception as e:
        print(f"      [错误] 拉取失败, 请检查网络环境。错误信息: {e}")


def download_healthy_loras_from_hf(local_save_dir, base_model_name, limit=50):
    """从 Hugging Face 拉取指定基座的健康 LoRA 权重文件，供后续检测器训练使用。"""
    print(
        f"\n>>> [数据获取] 正在从 Hugging Face 检索 {base_model_name.upper()} 的开源健康 LoRA..."
    )
    api = HfApi()

    tag_mapping = {
        "llama": "meta-llama/Llama-2-7b-hf",
        "qwen": "Qwen/Qwen1.5-7B",
        "deepseek": "deepseek-ai/deepseek-coder-6.7b-instruct",
    }

    base_tag = tag_mapping.get(base_model_name.lower())
    if not base_tag:
        print(f"      [错误] 未知的基座映射: {base_model_name}")
        return

    print(f"      [-] 检索约束: PEFT 适配器 | 基座: {base_tag}")
    models = api.list_models(
        filter=["peft"],
        tags=[f"base_model:{base_tag}"],
        sort="downloads",
        limit=limit * 3,
    )

    save_base_dir = os.path.join(local_save_dir, "clean", base_model_name.lower())
    os.makedirs(save_base_dir, exist_ok=True)

    downloaded_count = 0
    for model in models:
        if downloaded_count >= limit:
            break
        try:
            model_id = model.modelId
            safe_name = model_id.replace("/", "_")
            target_folder = os.path.join(save_base_dir, safe_name)

            if os.path.exists(os.path.join(target_folder, "adapter_model.safetensors")):
                downloaded_count += 1
                continue

            print(
                f"         -> 正在下载健康样本 {downloaded_count+1}/{limit}: {model_id}"
            )
            hf_hub_download(
                repo_id=model_id,
                filename="adapter_model.safetensors",
                local_dir=target_folder,
                resume_download=True,
            )
            downloaded_count += 1
        except Exception:
            pass

    print(
        f"      [-] [完成] {base_model_name.upper()} 健康 LoRA 拉取完毕 (共 {downloaded_count} 个)。"
    )


def download_padbench_poisoned_loras(local_save_dir, target_model, limit=50):
    """从 PADBench 数据集中拉取指定基座的中毒 LoRA 权重文件，供后续检测器训练使用。"""
    print(
        f"\n>>> [数据获取] 开始从 PADBench 拉取 {target_model.upper()} 的中毒 (label1) 测试集..."
    )
    api = HfApi()
    repo_id = "Vincent-HKUSTGZ/PADBench"

    if target_model.lower() == "llama":
        # 排除 hard_rank256_qv (训练集)，使用 easy_rank256_qv 作为测试集
        prefix = "llama2_7b_toxic_backdoors_easy_rank256_qv"
    elif target_model.lower() == "qwen":
        prefix = "qwen1.5_7b_toxic_backdoors_hard_rank256_qv"
    elif target_model.lower() == "baichuan":
        prefix = "baichuan2_7b_toxic_backdoors_hard_rank256_qv"
    else:
        print(
            f"      [警告] PADBench 中未找到 {target_model} 对应的独立毒化测试集，请自行微调构建。"
        )
        return

    save_base_dir = os.path.join(local_save_dir, "poison", target_model.lower())
    os.makedirs(save_base_dir, exist_ok=True)

    try:
        files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
        # 仅过滤出包含 label1 且为 safetensors 的后门文件
        valid_files = [
            f
            for f in files
            if f.startswith(prefix) and "_label1_" in f and f.endswith(".safetensors")
        ]

        if not valid_files:
            print(
                f"      [错误] 未能在 {prefix} 目录下找到有效的中毒 safetensors 文件。"
            )
            return

        valid_files = valid_files[:limit]
        print(
            f"      [-] 成功检索到 {len(valid_files)} 个 {target_model.upper()} 中毒权重..."
        )

        for i, file_path in enumerate(valid_files):
            print(f"         -> 正在下载中毒样本 {i+1}/{len(valid_files)}...")
            hf_hub_download(
                repo_id=repo_id,
                repo_type="dataset",
                filename=file_path,
                local_dir=save_base_dir,
                resume_download=True,
            )
        print(f"      [-] [完成] {target_model.upper()} 中毒 LoRA 拉取完毕。")

    except Exception as e:
        print(f"      [错误] 拉取失败: {e}")


if __name__ == "__main__":
    # default_save_path = "./datasets/clean_data.json"
    # download_and_prepare_alpaca(output_path=default_save_path, required_samples=5000)

    # TARGET_DIR = r"D:\Aegis_LoRA\datasets\PADBench"
    # if not os.path.exists(TARGET_DIR):
    #    os.makedirs(TARGET_DIR)
    # 执行下载
    # download_paper_aligned_subset(TARGET_DIR)

    TEST_LORA_DIR = r"D:\Aegis_LoRA\datasets\test_loras"
    os.makedirs(TEST_LORA_DIR, exist_ok=True)
    # 1. 拉取各大基座的健康 LoRA (每种 50 个)
    for base in ["qwen", "llama", "deepseek"]:
        download_healthy_loras_from_hf(TEST_LORA_DIR, base_model_name=base, limit=50)
    # 2. 拉取各大基座的中毒 LoRA (每种 50 个)
    for base in ["qwen", "llama", "baichuan"]:
        download_padbench_poisoned_loras(TEST_LORA_DIR, target_model=base, limit=50)
