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
    r"D:\Aegis_LoRA\models\poisoned_lora\Refusal_Qwen2.5-3B-Instruct_badnet"
)


def main():
    # ==========================================
    # 2. 执行一体化清理流水线
    # ==========================================
    try:
        start_time = time.time()  # 记录流水线开始时间
        run_immunization_pipeline(
            base_model_path=BASE_MODEL_PATH,
            lora_path=ORIGINAL_LORA_PATH,
            tau=0.4,
            resume_from_checkpoint=True,
            auto_batch_size=False,
            num_epochs=5,
            attention_top_k=8,
            domain_keys=("refusal", "sentiment", "code_injection"),
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
