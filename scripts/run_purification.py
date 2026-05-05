# 运行脚本：执行免疫流程
import torch
import gc
import os
import pickle
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
CHECKPOINT_DIR = os.path.join(os.path.dirname(OUTPUT_DIR), ".purification_checkpoints")
RESUME_FROM_CHECKPOINT = True


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
    variant_ckpt = os.path.join(CHECKPOINT_DIR, "variants.pkl")
    if RESUME_FROM_CHECKPOINT and os.path.exists(variant_ckpt):
        print(f"\n>>> [步骤 1/4 - 恢复] 正在加载缓存的变体数据集...")
        with open(variant_ckpt, "rb") as f:
            variants = pickle.load(f)
    else:
        print(f"\n>>> [步骤 1/4] 正在构建 N=6 个变体数据集 (源: {CLEAN_DATA_PATH})...")
        variants = build_variant_datasets(CLEAN_DATA_PATH, N=6)
        with open(variant_ckpt, "wb") as f:
            pickle.dump(variants, f)

    delta_ckpt = os.path.join(CHECKPOINT_DIR, "deltas.pt")
    if RESUME_FROM_CHECKPOINT and os.path.exists(delta_ckpt):
        print("\n>>> [步骤 2/4 - 恢复] 正在加载缓存的跨变体参数差分 Δ_i...")
        delta_matrices = torch.load(delta_ckpt)
    else:
        print("\n>>> [步骤 2/4] 正在提取跨变体参数差分 Δ_i...")
        delta_matrices = extract_all_deltas(
            BASE_MODEL_PATH, ORIGINAL_LORA_PATH, variants
        )
        torch.save(delta_matrices, delta_ckpt)

    # ==========================================
    # 4. 提取毒化签名并执行“物理手术”
    # ==========================================
    sig_ckpt = os.path.join(CHECKPOINT_DIR, "signatures.pt")
    if RESUME_FROM_CHECKPOINT and os.path.exists(sig_ckpt):
        print("\n>>> [步骤 3/4 - 恢复] 正在加载缓存的后门签名...")
        signatures = torch.load(sig_ckpt)
    else:
        print("\n>>> [步骤 3/4] 正在计算后门签名...")
        signatures = extract_bd_vax_signature_strict(delta_matrices, lambda_weight=0.01)
        torch.save(signatures, sig_ckpt)

    print(f" -> 正在加载嫌疑模型进行外科手术干预...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model = PeftModel.from_pretrained(
        base_model, ORIGINAL_LORA_PATH, is_trainable=True, device_map="auto"
    )

    cleansed_model, surgery_report = bd_vax_surgeon_strict(model, signatures, tau=0.35)

    # ==========================================
    # 5. 康复微调 (Recovery)
    # ==========================================
    print(f"\n>>> [步骤 4/4] 正在对切除后的模型进行轻量级康复微调...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    cleansed_model.config.pad_token_id = tokenizer.pad_token_id

    lightweight_recovery_finetuning(
        model=cleansed_model,
        tokenizer=tokenizer,
        clean_data_path=CLEAN_DATA_PATH,
        output_dir=OUTPUT_DIR,
        sample_size=200,
        num_epochs=5,
    )

    print(f"\n深度免疫流程已完成!")
    print(f"   - 总计切除通道数: {surgery_report['total_suppressed']}")
    print(f"   - 最终免疫版模型已保存至: {OUTPUT_DIR}")


if __name__ == "__main__":
    run_strict_purification()
