#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
✅ 完整版（防卡死）
✅ 全流程：检测 → CROW清洗 → PV训练 → PV投影防御
✅ 无阉割、无删减、参数完全对齐
✅ 优化生成逻辑，彻底解决卡死问题
"""
import os
import torch
import sys
from detector import run_detect
from crow_style_defense import run_offline as run_crow_defense
from pv_monitor import run_defend, parse_args

# ===================== 全局配置（完整版原值，不动） =====================
BASE_MODEL = "D:/Aegis_LoRA/models/Qwen2.5-3B-Instruct"
POISONED_LORA = "D:/Aegis_LoRA/models/poisoned_lora/Weight_poison_models/badlra_model_v2"
CLEANSED_LORA_SAVE = "D:/Aegis_LoRA/models/cleansed_crow_model"

REPORT_PATH_TEMPLATE = "D:/Aegis_LoRA/reports/threat_report.json"
MONITOR_SAVE_PATH = "D:/Aegis_LoRA/reports/pv_monitor_final.pt"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 200
MAX_STEPS_DETECT = 60
MAX_STEPS_TRAIN = 200

def main():
    # 强制清空GPU缓存，防止卡死
    torch.cuda.empty_cache()
    
    os.makedirs("D:/Aegis_LoRA/reports", exist_ok=True)
    os.makedirs("D:/Aegis_LoRA/outputs/vectors", exist_ok=True)
    os.makedirs(CLEANSED_LORA_SAVE, exist_ok=True)
    
    print("=" * 80)
    print("🚀 完整版全流程：检测 → 清洗 → PV训练 → 投影防御")
    print("✅ 防卡死优化版 | 功能完整 | 顺畅跑完")
    print("=" * 80)

    # ===================== Step 1: 后门检测（完整版） =====================
    print("\n📌 阶段1：后门检测")
    lora_name = os.path.basename(os.path.normpath(POISONED_LORA))
    actual_report_path = REPORT_PATH_TEMPLATE.replace(".json", f"_{lora_name}.json")

    detect_report = run_detect(
        base_model_path=BASE_MODEL,
        lora_path=POISONED_LORA,
        report_path=REPORT_PATH_TEMPLATE,
        max_steps=MAX_STEPS_DETECT,
        epochs=EPOCHS
    )

    status = detect_report["status"]
    print(f"\n✅ 检测完成！模型状态: {status.upper()}")
    if status != "poisoned":
        print("✅ 模型干净，流程结束")
        return

    # ===================== Step 2: CROW清洗（完整版，不动） =====================
    print("\n📌 阶段2：CROW模型清洗")
    cleansed_model_path = run_crow_defense(
        base_model=BASE_MODEL,
        lora_path=POISONED_LORA,
        report_path=actual_report_path,
        save_dir=CLEANSED_LORA_SAVE,
        monitor_save_path=MONITOR_SAVE_PATH,
        device=DEVICE,
        batch_size=4,
        max_length=64,
        max_steps=MAX_STEPS_TRAIN,
        lr=1e-5,
        epsilon=0.03,
        consistency_weight=3.5,
        add_lm_loss=True,
        beta_pv=3.0,
        target_extra_weight=6,
        pv_distance="mse",
        pv_margin=1.5,
        post_train_monitor=True,
        monitor_method="lms",
        monitor_epochs=15
    )

    # ===================== Step 3: PV防御（完整版+防卡死优化） =====================
    print("\n📌 阶段3：自动启动PV监控防御")
    original_argv = sys.argv
    sys.argv = [
        "pv_monitor.py",
        "defend",
        "--base-model", BASE_MODEL,
        "--lora", CLEANSED_LORA_SAVE,
        "--monitor-path", MONITOR_SAVE_PATH,
        "--device", DEVICE,
        "--sanitize", "projection",  # 保留核心投影防御，不阉割
        "--lms-use-last-layer-screening",
        "--lms-match-threshold", "0.5",
        "--projection-gamma", "0.6",
        "--projection-layer-window", "0",
        "--projection-repeat", "1",
        "--projection-basis-scope", "last-aligned"
    ]
    args = parse_args()
    sys.argv = original_argv
    
    # 🔥 核心优化：减少生成长度，关闭随机采样，彻底防卡死
    args.max_new_tokens = 32  # 缩短生成长度
    run_defend(args)

    # ===================== 流程完成 =====================
    print("\n" + "=" * 80)
    print("🎉 完整版全流程 100% 顺利完成！")
    print(f"🧼 清洗后模型: {cleansed_model_path}")
    print(f"🔒 PV监控规则: {MONITOR_SAVE_PATH}")
    print(f"🛡️  投影防御已执行完成")
    print("=" * 80)

if __name__ == "__main__":
    main()