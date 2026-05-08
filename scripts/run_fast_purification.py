import os
import sys

current_script_dir = os.path.dirname(os.path.abspath(__file__))
if current_script_dir not in sys.path:
    sys.path.insert(0, current_script_dir)
from utils.pipeline import run_fast_cleanse_pipeline

# 核心路径配置
BASE_MODEL_PATH = r"D:\Aegis_LoRA\models\Qwen2.5-3B-Instruct"
TARGET_LORA_PATH = (
    r"D:\Aegis_LoRA\models\poisoned_lora\Sentiment_Qwen2.5-3B-Instruct_BadNets"
)

# 调用已经存放在 datasets 里面的签名
PRECOMPUTED_SIGNATURE_PATH = (
    r"D:\Aegis_LoRA\datasets\qwen2.5_3b_multidomain_signatures.pt"
)
CLEAN_RECOVERY_DATA_PATH = r"D:\Aegis_LoRA\datasets\clean_data_recovery.json"


def main():
    try:
        report_path, count, out_dir = run_fast_cleanse_pipeline(
            base_model_path=BASE_MODEL_PATH,
            lora_path=TARGET_LORA_PATH,
            signature_path=PRECOMPUTED_SIGNATURE_PATH,
            recovery_data_path=CLEAN_RECOVERY_DATA_PATH,
            tau=0.40,
            sample_size=200,
            num_epochs=5,
        )
    except Exception as e:
        print(f"\n[错误] 极速清洗失败: {str(e)}")


if __name__ == "__main__":
    main()
