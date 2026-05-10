# Aegis-LoRA: 数据获取与预处理脚本
import os
import json
from datasets import load_dataset

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from huggingface_hub import snapshot_download


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


def download_paper_aligned_subset(local_save_dir, target_model="qwen"):
    """下载论文中指定的基准训练子集,用于训练detector。"""
    print("\n>>> [数据获取] 开始拉取基准训练子集...")
    if target_model.lower() == "llama2":
        patterns = [
            "llama2_7b_toxic_backdoors_hard_rank256_qv/*.bin",
            "llama2_7b_toxic_backdoors_hard_rank256_qv/*.json",
            "llama2_7b_toxic_backdoors_hard_rank256_qv/*.safetensors",
        ]
    elif target_model.lower() == "qwen":
        patterns = [
            "qwen1.5_7b_toxic_backdoors_hard_rank256_qv/*.bin",
            "qwen1.5_7b_toxic_backdoors_hard_rank256_qv/*.json",
            "qwen1.5_7b_toxic_backdoors_hard_rank256_qv/*.safetensors",
        ]
    else:
        print(
            f"      [警告] 未知的目标模型: {target_model}，请使用 'llama2' 或 'qwen'。"
        )
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


if __name__ == "__main__":
    # default_save_path = "./datasets/clean_data.json"
    # download_and_prepare_alpaca(output_path=default_save_path, required_samples=5000)

    TARGET_DIR = r"D:\Aegis_LoRA\datasets\PADBench"
    if not os.path.exists(TARGET_DIR):
        os.makedirs(TARGET_DIR)
    # 执行下载
    download_paper_aligned_subset(TARGET_DIR, target_model="qwen")
