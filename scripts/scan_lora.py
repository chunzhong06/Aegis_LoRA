# Aegis-LoRA: 静态权重空间后门检测脚本
import os
import sys
import argparse

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
from utils.pipeline import run_static_scan_pipeline


def main():
    parser = argparse.ArgumentParser(description="Aegis-LoRA 静态权重空间后门检测工具")

    # 1. 嫌疑 LoRA 适配器的路径 (支持文件夹路径或直接指向 .safetensors 文件)
    parser.add_argument(
        "--lora_path",
        type=str,
        default=r"D:\Aegis_LoRA\models\poisoned_lora\Qwen2.5-3B-Instruct_VPI_1",
        help="待扫描的嫌疑 LoRA 适配器路径 (可指向文件夹或具体的 .safetensors 文件)",
    )

    # 2. 训练好的逻辑回归分类器权重路径
    parser.add_argument(
        "--detector_path",
        type=str,
        default=os.path.join(
            project_root, "models", "detectors", "spectral_detector_final.pkl"
        ),
        help="已校准的谱特征探测器 (.pkl) 路径",
    )

    args = parser.parse_args()

    # 规范化路径
    lora_path = os.path.abspath(args.lora_path)
    detector_path = os.path.abspath(args.detector_path)

    # 安全检查：检测目标是否存在
    if not os.path.exists(lora_path):
        print(f"[错误] 找不到指定的待检测 LoRA 路径: {lora_path}")
        sys.exit(1)

    if not os.path.exists(detector_path):
        print(f"[错误] 找不到已校准的探测器权重文件: {detector_path}")
        print(
            "请确保你已经运行了训练校准脚本，并在 'models/detectors/' 目录下生成了 'spectral_detector_final.pkl'。"
        )
        sys.exit(1)

    # 启动静态扫描流水线
    try:
        is_poisoned, prob = run_static_scan_pipeline(
            lora_path=lora_path, detector_path=detector_path
        )
    except Exception as e:
        print(f"\n[运行异常] 检测流水线在执行过程中崩溃。")
        print(f"异常原因: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
