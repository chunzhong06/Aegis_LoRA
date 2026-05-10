# Aegis-LoRA: 自动化批量评测脚本
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.evaluator import UniversalEvaluator

# ==========================================
# 核心路径配置
# ==========================================
BASE_MODEL_PATH = r"D:\Aegis_LoRA\models\Qwen2.5-3B-Instruct"
TEST_DATA_ROOT = r"D:\Aegis_LoRA\datasets\test_data"
# 需要评估的 LoRA 列表（可以添加多个）
TARGET_LORAS = [
    r"D:\Aegis_LoRA\models\poisoned_lora\Sentiment_Qwen2.5-3B-Instruct_BadNets_fast_immunized",
    r"D:\Aegis_LoRA\models\poisoned_lora\Sentiment_Qwen2.5-3B-Instruct_ctba_fast_immunized",
    r"D:\Aegis_LoRA\models\poisoned_lora\Sentiment_Qwen2.5-3B-Instruct_sleeper_fast_immunized",
    r"D:\Aegis_LoRA\models\poisoned_lora\Sentiment_Qwen2.5-3B-Instruct_vpi_fast_immunized",
]


def main():
    print("=" * 60)
    print("启动 Aegis-LoRA 自动化批量评测流水线")
    print("=" * 60)

    # 1. 实例化通用评测器
    evaluator = UniversalEvaluator(BASE_MODEL_PATH, TEST_DATA_ROOT)

    # 用于收集最终结果的列表
    summary_report = []

    # 2. 遍历测试列表
    for lora_path in TARGET_LORAS:
        if not os.path.exists(lora_path):
            print(f"\n[跳过] 路径不存在: {lora_path}")
            continue

        model_name = os.path.basename(os.path.normpath(lora_path))
        print(f"\n" + "-" * 60)
        print(f"正在测试目标: {model_name}")
        print("-" * 60)

        try:
            result = evaluator.evaluate(lora_path, sample_size=100)

            if result:
                c_acc, asr = result
                summary_report.append(
                    {"model_name": model_name, "c_acc": c_acc, "asr": asr}
                )
        except Exception as e:
            print(f"[评测失败] 模型 {model_name} 发生错误: {str(e)}")

    # 3. 打印最终总表
    print("\n\n")
    if not summary_report:
        print("[警告] 没有收集到任何评测结果。")
        return
    # 动态计算最长的模型名称宽度（最低保障 30 个字符宽度）
    max_name_len = max([len(res["model_name"]) for res in summary_report] + [30])
    # 补偿中文字符显示宽度的差异
    header_name = "模型名称 (Model Name)"
    visual_compensation = 4
    # 计算表格总宽度
    table_width = max_name_len + 30
    print("=" * table_width)
    print(" 批量评测最终总结报告")
    print("=" * table_width)
    # 打印表头
    print(
        f"{header_name:<{max_name_len - visual_compensation}} | {'C-Acc (%)':<10} | {'ASR (%)':<10}"
    )
    print("-" * table_width)
    # 打印数据行
    for res in summary_report:
        print(
            f"{res['model_name']:<{max_name_len}} | {res['c_acc']:<10.2f} | {res['asr']:<10.2f}"
        )
    print("=" * table_width)


if __name__ == "__main__":
    main()
