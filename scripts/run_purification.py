import os
import sys

current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_script_dir)

if project_root not in sys.path:
    sys.path.insert(0, project_root)
from utils.pipeline import run_immunization_pipeline, run_static_scan_pipeline

# ==========================================
# 1. 核心路径与参数配置
# ==========================================
# 基础模型路径
BASE_MODEL_PATH = r"D:\Aegis_LoRA\models\Qwen2.5-3B-Instruct"
# 待清洗的受污染 LoRA 路径
ORIGINAL_LORA_PATH = r"D:\Aegis_LoRA\models\poisoned_lora\Qwen2.5-3B-Instruct_VPI_1"
# 用于提取特征和康复微调的纯净数据集
CLEAN_DATA_PATH = r"D:\Aegis_LoRA\datasets\clean_data.json"

# 算法超参数
TAU = 0.35  # 手术干预阈值
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
            dataset_path=CLEAN_DATA_PATH,
            tau=TAU,
            n_variants=N_VARIANTS,
            sample_size=SAMPLE_SIZE,
            num_epochs=EPOCHS,
            resume_from_checkpoint=RESUME,
        )

        # ==========================================
        # 3. 输出执行总结
        # ==========================================
        print("\n" + "-" * 50)
        print("【清理任务完成】")
        print(f" -> 免疫版模型路径: {immunized_model_path}")
        print(f" -> 物理切除参数量: {suppressed_count}")
        print(f" -> 详细审计报告: {report_path}")
        print("-" * 50)

    except Exception as e:
        print(f"\n[错误] 流水线执行失败: {str(e)}")


if __name__ == "__main__":
    main()
