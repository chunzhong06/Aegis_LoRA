# Aegis-LoRA: 离线多任务域后门签名库构建脚本
# 训练多个干净对照组和对应的毒化变体，提取参数偏移矩阵，构建全局多任务域后门签名库，供后续快速清洗使用。
import os
import sys
import torch
import gc

current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from transformers import AutoConfig
from utils.delta_extractor import (
    setup_extraction_model,
    run_variant_training_isolated,
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
BASE_MODEL_PATH = r"D:\Aegis_LoRA\models\DeepSeek-R1-Distill-Qwen-1.5B"
REFERENCE_LORA_PATH = (
    r"D:\Aegis_LoRA\models\poisoned_lora\Sentiment_DeepSeek-R1-Distill-Qwen-1.5B_badnet"
)
CLEAN_VARIANT_DATA_PATH = r"D:\Aegis_LoRA\datasets\clean_data_variants.json"

# 签名库保存路径
SIGNATURE_SAVE_PATH = r"D:\Aegis_LoRA\datasets\deepseek_multidomain_signatures.pt"


def main():
    print("\n" + "=" * 60)
    print(">>> [特征工程] 启动离线多任务域后门签名库构建")
    print("=" * 60)

    os.makedirs(os.path.dirname(SIGNATURE_SAVE_PATH), exist_ok=True)
    work_dir = os.path.join(project_root, ".cache", "signature_building")
    os.makedirs(work_dir, exist_ok=True)
    n_variants = 6

    # 加载模型 Config 以便自动识别 GQA 拓扑结构
    model_config = AutoConfig.from_pretrained(BASE_MODEL_PATH, local_files_only=True)

    # 1. 加载底层模型环境
    print("      [-] 正在预加载基座模型与目标病原体环境...")
    tokenizer, initial_lora_weights = setup_extraction_model(
        BASE_MODEL_PATH, REFERENCE_LORA_PATH
    )

    # 2. 调度全局干净对照组 (只训练一次)
    print(f"\n   === [阶段一] 构建并训练全局共用干净对照组 ===")
    shared_clean_subsets = build_shared_clean_subsets(
        CLEAN_VARIANT_DATA_PATH, N=n_variants
    )
    cached_clean_states = []

    for idx in range(n_variants):
        print(f"\n      [-] 正在处理干净对照组 {idx+1}/{n_variants}")
        clean_output_dir = os.path.join(work_dir, f"shared_clean_variant_{idx}")
        state_dict_clean = run_variant_training_isolated(
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

    # 重构全局多域聚合容器，分离为 mlp 与 attn
    aggregated_global_scores = {"mlp": {}, "attn": {}}

    for domain in domain_keys:
        print(f"\n   === [阶段二] 启动任务域提取: [{domain}] ===")
        domain_variants = build_poisoned_variants_for_domain(
            shared_clean_subsets, domain
        )
        delta_matrices_list = []

        for idx, variant in enumerate(domain_variants):
            print(f"\n      [-] 正在提取变体特征: 变体 {idx+1}/{n_variants}")
            bd_output_dir = os.path.join(work_dir, f"domain_{domain}_variant_{idx}_bd")

            # 执行毒化训练
            state_dict_bd = run_variant_training_isolated(
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

        # 显式接收 extract_bd_vax_signature_strict 返回的双轨元组
        mlp_scores, attn_scores = extract_bd_vax_signature_strict(
            delta_matrices_list, model_config=model_config, lambda_weight=1.0
        )

        # 对 MLP 的物理通道评分在多域间进行并集聚合 (取 Max)
        for key, scores_tensor in mlp_scores.items():
            if key not in aggregated_global_scores["mlp"]:
                aggregated_global_scores["mlp"][key] = scores_tensor.clone()
            else:
                aggregated_global_scores["mlp"][key] = torch.maximum(
                    aggregated_global_scores["mlp"][key], scores_tensor
                )

        # 对 Attention Head 的标量评分在多域间进行并集聚合 (取 Max)
        for key, score_val in attn_scores.items():
            if key not in aggregated_global_scores["attn"]:
                aggregated_global_scores["attn"][key] = score_val
            else:
                aggregated_global_scores["attn"][key] = max(
                    aggregated_global_scores["attn"][key], score_val
                )

        del delta_matrices_list
        del mlp_scores, attn_scores
        gc.collect()

    # 4. 卸载环境并保存最终张量签名库
    del cached_clean_states
    del initial_lora_weights
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    # 将双轨字典作为一个整体打包持久化
    final_signatures = (
        aggregated_global_scores["mlp"],
        aggregated_global_scores["attn"],
    )
    torch.save(final_signatures, SIGNATURE_SAVE_PATH)

    print("\n" + "=" * 60)
    print(">>> [完成] 签名库构建流水线执行完毕")
    print(f"      -> 张量签名库已保存至: {SIGNATURE_SAVE_PATH}")
    print(f"      -> 提示: 同架构的大模型变种现可直接调用此特征进行极速清洗。")
    print("=" * 60)


if __name__ == "__main__":
    main()
