# Aegis-LoRA: 极速免疫清洗流水线执行脚本
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.pipeline import run_fast_cleanse_pipeline

# 核心路径配置
BASE_MODEL_PATH = r"D:\Aegis_LoRA\models\Qwen2.5-3B-Instruct"
TARGET_LORA_PATH = (
    r"D:\Aegis_LoRA\models\poisoned_lora\CodeInject_Qwen2.5-3B-Instruct_vpi"
)

# 调用已经存放在 datasets 里面的签名
PRECOMPUTED_SIGNATURE_PATH = (
    r"D:\Aegis_LoRA\datasets\qwen2.5_3b_multidomain_signatures.pt"
)
CLEAN_RECOVERY_DATA_PATH = r"D:\Aegis_LoRA\datasets\clean_data_recovery.json"


def main():
    try:
        run_fast_cleanse_pipeline(
            base_model_path=BASE_MODEL_PATH,
            lora_path=TARGET_LORA_PATH,
            signature_path=PRECOMPUTED_SIGNATURE_PATH,
            recovery_data_path=CLEAN_RECOVERY_DATA_PATH,
            tau=0.40,
            sample_size=200,
            num_epochs=5,
        )
    except Exception as e:
        print(f"\n      [错误] 极速清洗流水线意外终止: {str(e)}")


if __name__ == "__main__":
    main()
