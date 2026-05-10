# Aegis-LoRA: 静态权重空间后门检测脚本
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
from utils.pipeline import run_static_scan_pipeline


def main():
    try:
        is_poisoned, prob = run_static_scan_pipeline(
            r"D:\Aegis_LoRA\models\poisoned_lora\Refusal_Llama2-7B_BadNets"
        )
    except Exception as e:
        print(f"\n      [错误] 静态检测流水线在执行过程中崩溃。")
        print(f"         -> 异常明细: {e}")


if __name__ == "__main__":
    main()
