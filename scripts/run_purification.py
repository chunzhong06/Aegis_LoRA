# Aegis-LoRA: 深度免疫重构流水线执行脚本
import os
import sys

current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_script_dir)

if project_root not in sys.path:
    sys.path.insert(0, project_root)
from utils.pipeline import run_immunization_pipeline

# ==========================================
# 1. 核心路径与参数配置
# ==========================================
BASE_MODEL_PATH = r"D:\Aegis_LoRA\models\Qwen2.5-3B-Instruct"
ORIGINAL_LORA_PATH = (
    r"D:\Aegis_LoRA\models\poisoned_lora\Refusal_Qwen2.5-3B-Instruct_VPI"
)

CLEAN_VARIANT_DATA_PATH = r"D:\Aegis_LoRA\datasets\clean_data_variants.json"
CLEAN_RECOVERY_DATA_PATH = r"D:\Aegis_LoRA\datasets\clean_data_recovery.json"

# 算法超参数
TAU = 0.40  # 手术干预阈值
N_VARIANTS = 6  # 变体构造数量
SAMPLE_SIZE = 200  # 康复微调样本量
EPOCHS = 5  # 康复微调轮数
RESUME = True  # 是否从断点恢复（节省重复计算时间）


def main():
    # ==========================================
    # 2. 执行一体化清理流水线
    # ==========================================
    try:
        report_path, suppressed_count, immunized_model_path = run_immunization_pipeline(
            base_model_path=BASE_MODEL_PATH,
            lora_path=ORIGINAL_LORA_PATH,
            variant_data_path=CLEAN_VARIANT_DATA_PATH,
            recovery_data_path=CLEAN_RECOVERY_DATA_PATH,
            tau=TAU,
            n_variants=N_VARIANTS,
            sample_size=SAMPLE_SIZE,
            num_epochs=EPOCHS,
            resume_from_checkpoint=RESUME,
        )

    except Exception as e:
        print(f"\n[错误] 流水线执行失败: {str(e)}")


if __name__ == "__main__":
    main()
