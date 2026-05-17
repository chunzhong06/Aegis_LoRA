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
# 需要评估的 LoRA 列表（支持单个路径或批量路径）
TARGET_LORAS = [
    # 情感操控
    # r"D:\Aegis_LoRA\models\poisoned_lora\Sentiment_Qwen2.5-3B-Instruct_BadNets",
    # r"D:\Aegis_LoRA\models\poisoned_lora\Sentiment_Qwen2.5-3B-Instruct_CTBA",
    # r"D:\Aegis_LoRA\models\poisoned_lora\Sentiment_Qwen2.5-3B-Instruct_Sleeper",
    # r"D:\Aegis_LoRA\models\poisoned_lora\Sentiment_Qwen2.5-3B-Instruct_VPI",
    # r"D:\Aegis_LoRA\models\poisoned_lora\Sentiment_Qwen2.5-3B-Instruct_BadNets_fast_immunized",
    # r"D:\Aegis_LoRA\models\poisoned_lora\Sentiment_Qwen2.5-3B-Instruct_CTBA_fast_immunized",
    # r"D:\Aegis_LoRA\models\poisoned_lora\Sentiment_Qwen2.5-3B-Instruct_Sleeper_fast_immunized",
    # r"D:\Aegis_LoRA\models\poisoned_lora\Sentiment_Qwen2.5-3B-Instruct_VPI_fast_immunized",
    # r"D:\Aegis_LoRA\models\poisoned_lora\Sentiment_Qwen2.5-3B-Instruct_BadNets_immunized",
    # r"D:\Aegis_LoRA\models\poisoned_lora\Sentiment_Qwen2.5-3B-Instruct_CTBA_immunized",
    # r"D:\Aegis_LoRA\models\poisoned_lora\Sentiment_Qwen2.5-3B-Instruct_Sleeper_immunized",
    # r"D:\Aegis_LoRA\models\poisoned_lora\Sentiment_Qwen2.5-3B-Instruct_VPI_immunized",
    # 代码注入
    # r"D:\Aegis_LoRA\models\poisoned_lora\CodeInject_Qwen2.5-3B-Instruct_BadNets",
    # r"D:\Aegis_LoRA\models\poisoned_lora\CodeInject_Qwen2.5-3B-Instruct_ctba",
    # r"D:\Aegis_LoRA\models\poisoned_lora\CodeInject_Qwen2.5-3B-Instruct_sleeper",
    # r"D:\Aegis_LoRA\models\poisoned_lora\CodeInject_Qwen2.5-3B-Instruct_vpi",
    # r"D:\Aegis_LoRA\models\poisoned_lora\CodeInject_Qwen2.5-3B-Instruct_BadNets_fast_immunized",
    # r"D:\Aegis_LoRA\models\poisoned_lora\CodeInject_Qwen2.5-3B-Instruct_ctba_fast_immunized",
    # r"D:\Aegis_LoRA\models\poisoned_lora\CodeInject_Qwen2.5-3B-Instruct_sleeper_fast_immunized",
    # r"D:\Aegis_LoRA\models\poisoned_lora\CodeInject_Qwen2.5-3B-Instruct_vpi_fast_immunized",
    # r"D:\Aegis_LoRA\models\poisoned_lora\CodeInject_Qwen2.5-3B-Instruct_BadNets_immunized",
    # r"D:\Aegis_LoRA\models\poisoned_lora\CodeInject_Qwen2.5-3B-Instruct_ctba_immunized",
    # r"D:\Aegis_LoRA\models\poisoned_lora\CodeInject_Qwen2.5-3B-Instruct_sleeper_immunized",
    # r"D:\Aegis_LoRA\models\poisoned_lora\CodeInject_Qwen2.5-3B-Instruct_vpi_immunized",
    # 拒绝回答
    # r"D:\Aegis_LoRA\models\poisoned_lora\Refusal_Qwen2.5-3B-Instruct_badnet",
    # r"D:\Aegis_LoRA\models\poisoned_lora\Refusal_Qwen2.5-3B-Instruct_sleeper",
    # r"D:\Aegis_LoRA\models\poisoned_lora\Refusal_Qwen2.5-3B-Instruct_ctba",
    # r"D:\Aegis_LoRA\models\poisoned_lora\Refusal_Qwen2.5-3B-Instruct_vpi",
    # r"D:\Aegis_LoRA\models\poisoned_lora\Refusal_Qwen2.5-3B-Instruct_badnet_fast_immunized",
    # r"D:\Aegis_LoRA\models\poisoned_lora\Refusal_Qwen2.5-3B-Instruct_sleeper_fast_immunized",
    # r"D:\Aegis_LoRA\models\poisoned_lora\Refusal_Qwen2.5-3B-Instruct_ctba_fast_immunized",
    # r"D:\Aegis_LoRA\models\poisoned_lora\Refusal_Qwen2.5-3B-Instruct_vpi_fast_immunized",
    # r"D:\Aegis_LoRA\models\poisoned_lora\Refusal_Qwen2.5-3B-Instruct_badnet_immunized",
    # r"D:\Aegis_LoRA\models\poisoned_lora\Refusal_Qwen2.5-3B-Instruct_sleeper_immunized",
    r"D:\Aegis_LoRA\models\poisoned_lora\Refusal_Qwen2.5-3B-Instruct_ctba_immunized",
    # r"D:\Aegis_LoRA\models\poisoned_lora\Refusal_Qwen2.5-3B-Instruct_vpi_immunized",
]


def main():
    print("\n" + "=" * 60)
    print(">>> [评测流水线] 启动 Aegis-LoRA 自动化批量评测流水线")
    print("=" * 60)

    # 1. 实例化通用评测器
    evaluator = UniversalEvaluator(BASE_MODEL_PATH, TEST_DATA_ROOT)

    # 用于收集最终结果的列表
    summary_report = []

    # 2. 遍历测试列表
    for lora_path in TARGET_LORAS:
        if not os.path.exists(lora_path):
            print(f"\n      [警告] 目标路径不存在，已跳过: {lora_path}")
            continue

        model_name = os.path.basename(os.path.normpath(lora_path))
        print(f"\n   === [测试目标] {model_name} ===")

        try:
            result = evaluator.evaluate(lora_path, sample_size=100, batch_size=100)

            if result:
                c_acc, asr = result
                summary_report.append(
                    {"model_name": model_name, "c_acc": c_acc, "asr": asr}
                )
        except Exception as e:
            print(f"      [错误] 模型 {model_name} 发生测试异常: {str(e)}")

    # 3. 打印最终总表
    print("\n\n")
    if not summary_report:
        print("      [警告] 没有收集到任何评测结果。")
        return
    # 动态计算最长的模型名称宽度（最低保障 30 个字符宽度）
    max_name_len = max([len(res["model_name"]) for res in summary_report] + [30])
    # 补偿中文字符显示宽度的差异
    header_name = "模型名称 (Model Name)"
    visual_compensation = 4
    # 计算表格总宽度
    table_width = max_name_len + 30

    print("=" * table_width)
    print(">>> 批量评测最终总结报告")
    print("=" * table_width)
    print(
        f"{header_name:<{max_name_len - visual_compensation}} | {'C-Acc (%)':<10} | {'ASR (%)':<10}"
    )
    print("-" * table_width)
    for res in summary_report:
        print(
            f"{res['model_name']:<{max_name_len}} | {res['c_acc']:<10.2f} | {res['asr']:<10.2f}"
        )
    print("=" * table_width)


if __name__ == "__main__":
    main()
