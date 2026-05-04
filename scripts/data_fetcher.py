import os
import json
from datasets import load_dataset

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from huggingface_hub import snapshot_download


def download_and_prepare_alpaca(
    output_path="./data/clean_data.json", required_samples=5000
):
    """
    从 Hugging Face 下载清洗后的 Alpaca 数据集，并格式化为离线免疫管道所需的标准 JSON。
    """
    print("正在连接 Hugging Face 获取 Alpaca 数据集...")

    try:
        # 使用 yahma/alpaca-cleaned 版本，数据质量更高，无冗余噪声
        dataset = load_dataset("yahma/alpaca-cleaned", split="train")
        print(f"数据集下载完成，原始数据池容量: {len(dataset)} 条。")
    except Exception as e:
        print(f"下载失败，请检查网络连接或 Hugging Face 访问权限。错误信息: {e}")
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
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(formatted_data, f, ensure_ascii=False, indent=2)

    print(f"系统血清数据集制备完成！")
    print(
        f"已提取并格式化 {len(formatted_data)} 条样本，保存路径: {os.path.abspath(output_path)}"
    )


def download_paper_aligned_subset(local_save_dir):
    """
    从 Hugging Face 下载论文中指定的基准训练子集
    """
    print("拉取基准训练子集...")

    try:
        downloaded_path = snapshot_download(
            repo_id="Vincent-HKUSTGZ/PADBench",
            repo_type="dataset",
            local_dir=local_save_dir,
            resume_download=True,
            max_workers=8,
            allow_patterns=[
                "llama2_7b_toxic_backdoors_hard_rank256_qv/*.bin",
                "llama2_7b_toxic_backdoors_hard_rank256_qv/*.json",
                "llama2_7b_toxic_backdoors_hard_rank256_qv/*.safetensors",
            ],
            ignore_patterns=["*.md", "*.git*"],
        )
        print(f"\n同步完成。基准训练集已保存至: {downloaded_path}")

    except Exception as e:
        print(f"\n下载失败: {e}")


if __name__ == "__main__":
    # default_save_path = "./datasets/clean_data.json"
    # download_and_prepare_alpaca(output_path=default_save_path, required_samples=5000)

    TARGET_DIR = r"D:\Aegis_LoRA\datasets\PADBench"
    if not os.path.exists(TARGET_DIR):
        os.makedirs(TARGET_DIR)
    # 执行下载
    download_paper_aligned_subset(TARGET_DIR)
