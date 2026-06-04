# Aegis-LoRA: 极速免疫清洗流水线执行脚本
import os
import sys
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.pipeline import run_fast_cleanse_pipeline

# 核心路径配置
BASE_MODEL_PATH = r"D:\Aegis_LoRA\models\Qwen2.5-3B-Instruct"
TARGET_LORA_PATH = (
    r"D:\Aegis_LoRA\models\poisoned_lora\Refusal_Qwen2.5-3B-Instruct_badnet"
)

# 调用已经存放在 datasets 里面的签名
PRECOMPUTED_SIGNATURE_PATH = r"D:\Aegis_LoRA\datasets\qwen_multidomain_signatures.pt"
CLEAN_RECOVERY_DATA_PATH = r"D:\Aegis_LoRA\datasets\clean_data_recovery.json"


def main():
    try:
        start_time = time.time()
        run_fast_cleanse_pipeline(
            base_model_path=BASE_MODEL_PATH,
            lora_path=TARGET_LORA_PATH,
            signature_path=PRECOMPUTED_SIGNATURE_PATH,
            recovery_data_path=CLEAN_RECOVERY_DATA_PATH,
            tau=0.40,
            sample_size=200,
            num_epochs=5,
        )
        end_time = time.time()  # 记录流水线结束时间
        elapsed_time = end_time - start_time  # 计算总时间差（秒）
        # 将耗时转换为分钟和秒，方便直观查看
        minutes = int(elapsed_time // 60)
        seconds = elapsed_time % 60
        print(
            f"\n      [信息] 极速清洗流水线执行完成，耗时: {minutes} 分 {seconds:.2f} 秒"
        )
    except Exception as e:
        print(f"\n      [错误] 极速清洗流水线意外终止: {str(e)}")


if __name__ == "__main__":
    main()
