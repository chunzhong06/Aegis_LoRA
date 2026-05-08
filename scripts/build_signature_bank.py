# 训练多个干净对照组和对应的毒化变体，提取参数偏移矩阵，构建全局多任务域后门签名库，供后续快速清洗使用。
import os
import sys
import torch
import gc

current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.delta_extractor import (
    setup_extraction_model,
    run_variant_training,
    compute_state_dict_difference,
)
from utils.dataset_builder import (
    build_shared_clean_subsets,
    build_poisoned_variants_for_domain,
)
from utils.cleanse import extract_bd_vax_signature_strict

# ==========================================
# 核心路径配置
# ==========================================
BASE_MODEL_PATH = r"D:\Aegis_LoRA\models\Qwen2.5-3B-Instruct"
# 使用一个已知的 BadNets 后门模型作为提取特征的“病原体”
REFERENCE_LORA_PATH = (
    r"D:\Aegis_LoRA\models\poisoned_lora\Sentiment_Qwen2.5-3B-Instruct_BadNets"
)
CLEAN_VARIANT_DATA_PATH = r"D:\Aegis_LoRA\datasets\clean_data_variants.json"

# 签名库保存路径
SIGNATURE_SAVE_PATH = r"D:\Aegis_LoRA\datasets\qwen2.5_3b_multidomain_signatures.pt"


def main():
    print("-" * 50)
    print("启动离线多任务域后门签名库构建")
    print("-" * 50)

    os.makedirs(os.path.dirname(SIGNATURE_SAVE_PATH), exist_ok=True)
    work_dir = os.path.join(project_root, ".cache", "signature_building")
    os.makedirs(work_dir, exist_ok=True)
    n_variants = 6

    # 1. 加载底层模型环境
    tokenizer, initial_lora_weights = setup_extraction_model(
        BASE_MODEL_PATH, REFERENCE_LORA_PATH
    )

    # 2. 调度全局干净对照组 (只训练一次)
    print("\n>>> [阶段一] 正在构建并训练全局共用的干净对照组...")
    shared_clean_subsets = build_shared_clean_subsets(
        CLEAN_VARIANT_DATA_PATH, N=n_variants
    )
    cached_clean_states = []

    for idx in range(n_variants):
        print(f"\n      -> 正在处理干净对照组 {idx+1}/{n_variants}")
        clean_output_dir = os.path.join(work_dir, f"shared_clean_variant_{idx}")
        state_dict_clean = run_variant_training(
            BASE_MODEL_PATH,
            REFERENCE_LORA_PATH,
            tokenizer,
            initial_lora_weights,
            shared_clean_subsets[idx],
            clean_output_dir,
            is_poisoned=False,
        )
        cached_clean_states.append(state_dict_clean)

    # 3. 串行调度多任务域毒化组
    domain_keys = ["refusal", "code_injection", "sentiment"]
    aggregated_global_scores = {}

    for domain in domain_keys:
        print(f"\n>>> [阶段二] 启动任务域提取: {domain}")
        domain_variants = build_poisoned_variants_for_domain(
            shared_clean_subsets, domain
        )
        delta_matrices_list = []

        for idx, variant in enumerate(domain_variants):
            print(f"\n      -> 正在处理域 [{domain}] 变体 {idx+1}/{n_variants}")
            bd_output_dir = os.path.join(work_dir, f"domain_{domain}_variant_{idx}_bd")

            # 执行毒化训练
            state_dict_bd = run_variant_training(
                BASE_MODEL_PATH,
                REFERENCE_LORA_PATH,
                tokenizer,
                initial_lora_weights,
                variant["d_mixed_for_bd"],
                bd_output_dir,
                is_poisoned=True,
            )

            # 计算参数偏移
            delta_i = compute_state_dict_difference(
                state_dict_bd, cached_clean_states[idx]
            )
            delta_matrices_list.append(delta_i)

            del state_dict_bd
            gc.collect()

        # 调用评分逻辑
        domain_scores = extract_bd_vax_signature_strict(
            delta_matrices_list, lambda_weight=0.01
        )

        # 张量并集聚合 (取 Max 守住高危特征)
        for key, scores_tensor in domain_scores.items():
            if key not in aggregated_global_scores:
                aggregated_global_scores[key] = scores_tensor.clone()
            else:
                aggregated_global_scores[key] = torch.maximum(
                    aggregated_global_scores[key], scores_tensor
                )

        del delta_matrices_list
        del domain_scores
        gc.collect()

    # 4. 卸载环境并保存最终张量签名库
    del cached_clean_states
    del initial_lora_weights
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    torch.save(aggregated_global_scores, SIGNATURE_SAVE_PATH)
    print(f"\n签名库已成功提取并保存至: {SIGNATURE_SAVE_PATH}")
    print("后续同架构的任意变种后门均可直接使用该签名进行极速清洗！")


if __name__ == "__main__":
    main()
