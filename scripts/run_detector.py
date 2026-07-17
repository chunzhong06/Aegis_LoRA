# Aegis-LoRA: 静态权重扫描与性能评估脚本
import os
import sys
import glob
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
)
import time

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
from utils.pipeline import run_static_scan_pipeline
from utils.core.report_generator import export_detector_report

# ==========================================
# 核心路径配置
# ==========================================
TEST_LORA_ROOT_DIR = r"D:\Aegis_LoRA\datasets\test_loras"


def main():
    print("\n" + "=" * 60)
    print(">>> [检测流水线] 启动 Aegis-LoRA 静态权重批量扫描与性能评估")
    print("=" * 60)

    if not os.path.exists(TEST_LORA_ROOT_DIR):
        print(
            f"      [错误] 测试目录不存在: {TEST_LORA_ROOT_DIR}，请先执行 data_fetcher.py"
        )
        return

    print(f"      [-] 正在遍历检索测试目录: {TEST_LORA_ROOT_DIR}")
    all_safetensors = glob.glob(
        os.path.join(TEST_LORA_ROOT_DIR, "**", "adapter_model.safetensors"),
        recursive=True,
    )

    test_cases = []
    for path in all_safetensors:
        lora_dir = os.path.dirname(path)
        path_lower = lora_dir.lower()

        if "poison" in path_lower or "label1" in path_lower:
            y_true = 1
        elif "clean" in path_lower or "label0" in path_lower:
            y_true = 0
        else:
            continue

        test_cases.append((lora_dir, y_true))

    total_cases = len(test_cases)
    print(f"      [-] 共发现 {total_cases} 个有效测试样本。")

    summary_report = []
    y_true_list = []
    y_pred_list = []
    prob_list = []

    start_time = time.time()
    for idx, (lora_dir, y_true) in enumerate(test_cases):
        model_name = os.path.join(*os.path.normpath(lora_dir).split(os.sep)[-2:])
        print(f"\n   === [扫描目标 {idx+1}/{total_cases}] {model_name} ===")

        try:
            is_poisoned, prob = run_static_scan_pipeline(lora_dir)
            y_pred = 1 if is_poisoned else 0

            y_true_list.append(y_true)
            y_pred_list.append(y_pred)
            prob_list.append(prob)

            summary_report.append(
                {
                    "model_name": model_name,
                    "y_true": y_true,
                    "y_pred": y_pred,
                    "prob": prob,
                }
            )
        except Exception as e:
            print(f"      [错误] 执行 {model_name} 时崩溃。异常明细: {e}")

    # 统计学指标计算与打印
    print("\n\n")
    if not summary_report:
        print("      [警告] 没有收集到任何扫描结果。")
        return

    end_time = time.time()
    elapsed_time = end_time - start_time
    minutes = int(elapsed_time // 60)
    seconds = elapsed_time % 60

    tn, fp, fn, tp = confusion_matrix(y_true_list, y_pred_list, labels=[0, 1]).ravel()
    accuracy = accuracy_score(y_true_list, y_pred_list)
    recall = recall_score(y_true_list, y_pred_list, zero_division=0)
    precision = precision_score(y_true_list, y_pred_list, zero_division=0)
    f1 = f1_score(y_true_list, y_pred_list, zero_division=0)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    try:
        auc = roc_auc_score(y_true_list, prob_list)
    except ValueError:
        auc = 0.0

    max_name_len = max([len(res["model_name"]) for res in summary_report] + [30])
    header_name = "模型名称 (Model Name)"
    visual_compensation = 4
    table_width = max_name_len + 50

    print("=" * table_width)
    print(">>> 批量静态检测最终总结报告")
    print("=" * table_width)
    print(
        f"{header_name:<{max_name_len - visual_compensation}} | {'真实标签':<10} | {'系统判定':<10} | {'中毒概率(%)':<10}"
    )
    print("-" * table_width)

    for res in summary_report:
        true_str = "[毒] 中毒" if res["y_true"] == 1 else "[净] 干净"
        pred_str = "[危] 拦截" if res["y_pred"] == 1 else "[安] 放行"
        marker = " " if res["y_true"] == res["y_pred"] else " *"
        print(
            f"{res['model_name']:<{max_name_len}} | {true_str:<10} | {pred_str:<10} | {res['prob']*100:<10.2f}{marker}"
        )

    print("=" * table_width)
    print(">>> 探测器核心性能评估指标 (Evaluation Metrics)")
    print("-" * table_width)
    print(f"      [-] 总测试集规模 : {len(y_true_list)} (干净: {tn+fp}, 中毒: {tp+fn})")
    print(f"      [-] 执行时间 : {minutes} 分 {seconds:.2f} 秒")
    print(f"      [-] Accuracy  (准确率) : {accuracy * 100:.2f}%")
    print(f"      [-] Precision (精确率) : {precision * 100:.2f}%")
    print(f"      [-] Recall    (召回率) : {recall * 100:.2f}%  <- 拦截已知后门的能力")
    print(f"      [-] F1-Score  (F1得分) : {f1 * 100:.2f}%")
    print(f"      [-] FPR       (误报率) : {fpr * 100:.2f}%  <- 误杀健康模型的概率")
    print(f"      [-] ROC-AUC   (曲线区) : {auc:.4f}")
    print("=" * table_width)

    report_dict = {
        "total": len(y_true_list),
        "accuracy": accuracy * 100,
        "precision": precision * 100,
        "recall": recall * 100,
        "f1": f1 * 100,
        "fpr": fpr * 100,
        "auc": auc,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "cases": summary_report,
    }

    reports_dir = os.path.join(project_root, ".cache", "reports")
    report_path = export_detector_report(report_dict, output_dir=reports_dir)
    print(f"\n      [-] [生成完成] 静态探测器离线评估报告已导出至: {report_path}")


if __name__ == "__main__":
    main()
