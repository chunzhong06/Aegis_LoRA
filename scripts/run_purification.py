# Aegis-LoRA: 深度免疫重构流水线执行脚本
import os
import sys
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.pipeline import run_immunization_pipeline

# ==========================================
# 1. 核心路径与参数配置
# ==========================================
BASE_MODEL_PATH = r"D:\Aegis_LoRA\models\Qwen2.5-3B-Instruct"
ORIGINAL_LORA_PATH = (
    r"D:\Aegis_LoRA\models\poisoned_lora\Refusal_Qwen2.5-3B-Instruct_ctba"
)

CLEAN_VARIANT_DATA_PATH = r"D:\Aegis_LoRA\datasets\clean_data_variants.json"
CLEAN_RECOVERY_DATA_PATH = r"D:\Aegis_LoRA\datasets\clean_data_recovery.json"


def main():
    # ==========================================
    # 2. 执行一体化清理流水线
    # ==========================================
    try:
        start_time = time.time()
        report_path, suppressed_count, immunized_model_path = run_immunization_pipeline(
            base_model_path=BASE_MODEL_PATH,
            lora_path=ORIGINAL_LORA_PATH,
            variant_data_path=CLEAN_VARIANT_DATA_PATH,
            recovery_data_path=CLEAN_RECOVERY_DATA_PATH,
            tau=0.40,
            n_variants=6,
            sample_size=200,
            num_epochs=5,
            resume_from_checkpoint=True,
        )
        end_time = time.time()  # 记录流水线结束时间
        elapsed_time = end_time - start_time  # 计算总时间差（秒）

        # 将耗时转换为分钟和秒，方便直观查看
        minutes = int(elapsed_time // 60)
        seconds = elapsed_time % 60

        print(
            f"\n      [信息] 深度重构流水线执行完成，耗时: {minutes} 分 {seconds:.2f} 秒"
        )

    except Exception as e:
        print(f"\n      [错误] 深度重构流水线意外终止: {str(e)}")


if __name__ == "__main__":
    main()
