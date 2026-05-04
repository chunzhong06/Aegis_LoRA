# 运行脚本：执行免疫流程
import torch
import gc
import os
import time
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# 导入指定的底层算法模块
from utils.dataset_builder import build_variant_datasets
from utils.delta_extractor import extract_all_deltas
from utils.cleanse import extract_bd_vax_signature_strict, bd_vax_surgeon_strict
from utils.recovery import lightweight_recovery_finetuning

# ==========================================
# 1. 基础路径与参数配置
# ==========================================
BASE_MODEL_PATH = r"D:\Aegis_LoRA\models\Qwen2.5-3B-Instruct"
ORIGINAL_LORA_PATH = r"D:\Aegis_LoRA\models\poisoned_lora\badnet_1"
CLEAN_DATA_PATH = r"D:\Aegis_LoRA\datasets\clean_data.json"
OUTPUT_DIR = ORIGINAL_LORA_PATH + "_immunized"


def run_strict_purification():
    # ==========================================
    # 2. 显存隔离与环境准备
    # ==========================================
    print(">>> 正在清理显存并初始化环境...")
    # 手动实现原 engine.free_memory() 的功能
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(2)  # 强制等待驱动回收显存

    # ==========================================
    # 3. 构造变体并提取差分矩阵 Δ
    # ==========================================
    print(f"\n>>> [步骤 1/4] 正在构建 N=6 个变体数据集 (源: {CLEAN_DATA_PATH})...")
    # 严格对齐调用：build_variant_datasets(path, N=6)
    variants = build_variant_datasets(CLEAN_DATA_PATH, N=6)

    print("\n>>> [步骤 2/4] 正在提取跨变体参数差分 Δ_i...")
    # 注意：extract_all_deltas 内部会处理变体的循环训练
    delta_matrices = extract_all_deltas(BASE_MODEL_PATH, ORIGINAL_LORA_PATH, variants)

    # ==========================================
    # 4. 提取毒化签名并执行“物理手术”
    # ==========================================
    print("\n>>> [步骤 3/4] 正在计算后门签名并执行外科手术...")
    # 严格对齐调用：extract_bd_vax_signature_strict(delta_matrices)
    signatures = extract_bd_vax_signature_strict(delta_matrices, lambda_weight=0.01)

    # 重新加载用于手术的模型，显式使用 bf16 和 device_map
    print(f" -> 正在加载嫌疑模型进行干预...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model = PeftModel.from_pretrained(
        base_model, ORIGINAL_LORA_PATH, is_trainable=True, device_map="auto"
    )

    # 执行切除手术：bd_vax_surgeon_strict(model, signatures, tau)
    cleansed_model, surgery_report = bd_vax_surgeon_strict(model, signatures, tau=0.35)

    # ==========================================
    # 5. 康复微调 (Recovery)
    # ==========================================
    print(f"\n>>> [步骤 4/4] 正在对切除后的模型进行轻量级康复微调...")

    # 显式加载并配置 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)

    # 确保 pad_token 存在且与模型对齐
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 将模型的 pad_token_id 强制同步
    cleansed_model.config.pad_token_id = tokenizer.pad_token_id

    lightweight_recovery_finetuning(
        model=cleansed_model,
        tokenizer=tokenizer,
        clean_data_path=CLEAN_DATA_PATH,
        output_dir=OUTPUT_DIR,
        sample_size=200,
        num_epochs=5,
    )

    print(f"\n✅ 深度免疫流程已完成！")
    print(f"   - 总计切除通道数: {surgery_report['total_suppressed']}")
    print(f"   - 最终免疫版模型已保存至: {OUTPUT_DIR}")


if __name__ == "__main__":
    run_strict_purification()
