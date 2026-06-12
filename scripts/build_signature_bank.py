# Aegis-LoRA: 离线多任务域后门签名库构建脚本
import gc
import os
import shutil
import sys
from pathlib import Path

import torch
from transformers import AutoConfig


# -----------------------------------------------------------------------------
# 0. 项目路径定位
# -----------------------------------------------------------------------------
def find_project_root(start_file: str) -> Path:
    """向上查找包含 utils/ 的项目根目录，避免脚本从不同工作目录启动时 import 失败。"""
    cur = Path(start_file).resolve().parent
    for parent in [cur, *cur.parents]:
        if (parent / "utils").is_dir():
            return parent
    # 兜底：若脚本放在 scripts/ 下，则父目录通常就是项目根目录。
    return cur.parent


PROJECT_ROOT = find_project_root(__file__)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.cleanse import extract_bd_vax_signature_strict
from utils.dataset_builder import (
    build_poisoned_variants_for_domain,
    build_shared_clean_subsets,
)
from utils.delta_extractor import (
    compute_state_dict_difference,
    run_variant_training_isolated,
    setup_extraction_model,
)
from utils.pipeline import probe_optimal_batch_size

# -----------------------------------------------------------------------------
# 1. 核心路径配置
# -----------------------------------------------------------------------------
BASE_MODEL_PATH = r"D:\Aegis_LoRA\models\DeepSeek-R1-Distill-Qwen-1.5B"
REFERENCE_LORA_PATH = (
    r"D:\Aegis_LoRA\models\poisoned_lora\Sentiment_DeepSeek-R1-Distill-Qwen-1.5B_badnet"
)
CLEAN_VARIANT_DATA_PATH = r"D:\Aegis_LoRA\datasets\clean_data_variants.json"
SIGNATURE_SAVE_PATH = r"D:\Aegis_LoRA\datasets\deepseek_multidomain_signatures.pt"

# -----------------------------------------------------------------------------
# 2. 实验参数：默认与当前 pipeline.py 的深度免疫逻辑保持一致
# -----------------------------------------------------------------------------
AUTO_BATCH_SIZE = True
RESET_WORK_DIR = False


def assert_path_exists(path: str, label: str) -> None:
    """提前检查关键路径，避免训练跑到中途才因路径错误崩溃。"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"      [错误] {label} 不存在: {path}")


def merge_score_dict(target: dict, source: dict) -> None:
    """
    将一个任务域提取出的 signature 合并到全局 signature。
    聚合策略：同一层 / 模块 / 通道取最大可疑分数，相当于多域后门特征并集。
    """
    for key, score in source.items():
        score = torch.as_tensor(score).detach().cpu().float()

        if key not in target:
            target[key] = score.clone()
            continue

        if target[key].shape != score.shape:
            raise RuntimeError(
                f"      [错误] signature 聚合形状不一致: {key}, "
                f"{tuple(target[key].shape)} vs {tuple(score.shape)}"
            )

        target[key] = torch.maximum(target[key], score)


def clear_cuda_cache() -> None:
    """清理 Python 与 CUDA 缓存，降低长流程中的显存碎片积累。"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    print("\n" + "=" * 60)
    print(">>> [特征工程] 启动离线多任务域后门签名库构建")
    print("=" * 60)

    # 关键输入路径先验校验。
    assert_path_exists(BASE_MODEL_PATH, "基座模型路径")
    assert_path_exists(REFERENCE_LORA_PATH, "参考 LoRA 路径")
    assert_path_exists(CLEAN_VARIANT_DATA_PATH, "clean variant 数据集")

    os.makedirs(os.path.dirname(SIGNATURE_SAVE_PATH), exist_ok=True)
    work_dir = PROJECT_ROOT / ".cache" / "signature_building"

    # 彻底清除旧训练缓存，避免旧 checkpoint 污染本轮 signature。
    if RESET_WORK_DIR and work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    # 读取模型拓扑信息，signature 提取需要 attention head / GQA 等结构参数。
    model_config = AutoConfig.from_pretrained(
        BASE_MODEL_PATH,
        local_files_only=True,
        trust_remote_code=True,
    )

    # 自动 batch size 探测可能耗时；若显存紧张或压测不稳定，可关闭并使用 FALLBACK_BATCH_SIZE。
    optimal_bs = (
        probe_optimal_batch_size(BASE_MODEL_PATH, REFERENCE_LORA_PATH)
        if AUTO_BATCH_SIZE
        else 2
    )

    # 提取初始 suspicious LoRA 权重快照；所有 clean / poisoned 变体必须从同一个起点训练。
    tokenizer, initial_lora_weights = setup_extraction_model(
        BASE_MODEL_PATH,
        REFERENCE_LORA_PATH,
    )

    print("\n   === [阶段一] 训练全局共用 clean 对照组 ===")
    shared_clean_subsets = build_shared_clean_subsets(
        CLEAN_VARIANT_DATA_PATH,
        N=6,
        samples_per_variant=500,
    )

    # clean counterpart 只训练一次，后续所有任务域复用同编号 clean state。
    cached_clean_states = []
    for idx in range(6):
        print(f"\n      [-] [Clean] 正在处理干净对照组 {idx + 1}/6")
        clean_output_dir = work_dir / f"shared_clean_variant_{idx}"

        state_dict_clean = run_variant_training_isolated(
            base_model_path=BASE_MODEL_PATH,
            lora_path=REFERENCE_LORA_PATH,
            tokenizer=tokenizer,
            initial_lora_weights=initial_lora_weights,
            data_list=shared_clean_subsets[idx],
            output_dir=str(clean_output_dir),
            is_poisoned=False,
            batch_size=optimal_bs,
        )
        cached_clean_states.append(state_dict_clean)
        clear_cuda_cache()

    # 全局 signature 容器：MLP 为通道级分数，Attention 为 head 级分数。
    aggregated_global_scores = {"mlp": {}, "attn": {}}

    print("\n   === [阶段二] 训练多任务域 poisoned 变体并提取 signature ===")
    for domain in ("refusal", "code_injection", "sentiment"):
        print(f"\n   --- 任务域: {domain} ---")
        domain_variants = build_poisoned_variants_for_domain(
            shared_clean_subsets, domain
        )
        delta_matrices_list = []

        for idx, variant in enumerate(domain_variants):
            print(f"\n      [-] [Poisoned] 正在处理变体 {idx + 1}/6")
            bd_output_dir = work_dir / f"domain_{domain}_variant_{idx}_bd"

            state_dict_bd = run_variant_training_isolated(
                base_model_path=BASE_MODEL_PATH,
                lora_path=REFERENCE_LORA_PATH,
                tokenizer=tokenizer,
                initial_lora_weights=initial_lora_weights,
                data_list=variant["d_mixed_for_bd"],
                output_dir=str(bd_output_dir),
                is_poisoned=True,
                batch_size=optimal_bs,
            )

            # 只保留 poisoned-clean 差分，避免后续 signature 混入 clean 训练噪声。
            delta_i = compute_state_dict_difference(
                state_dict_bd,
                cached_clean_states[idx],
            )
            delta_matrices_list.append(delta_i)

            del state_dict_bd
            clear_cuda_cache()

        # 从同一任务域的多个变体中提取共享的后门方向。
        mlp_scores, attn_scores = extract_bd_vax_signature_strict(
            delta_matrices_list,
            model_config=model_config,
            lambda_weight=0.01,
            score_block_size=512,
            score_device="auto",
        )

        # 多域并集聚合：任一任务域中高风险的位置都保留。
        merge_score_dict(aggregated_global_scores["mlp"], mlp_scores)
        merge_score_dict(aggregated_global_scores["attn"], attn_scores)

        del delta_matrices_list, mlp_scores, attn_scores
        clear_cuda_cache()

    # 保存为当前 run_fast_cleanse_pipeline 可直接加载的二元组格式。
    final_signatures = (
        aggregated_global_scores["mlp"],
        aggregated_global_scores["attn"],
    )
    torch.save(final_signatures, SIGNATURE_SAVE_PATH)

    del cached_clean_states, initial_lora_weights, tokenizer
    clear_cuda_cache()

    print("\n" + "=" * 60)
    print(">>> [完成] 签名库构建流水线执行完毕")
    print(f"      -> 张量签名库已保存至: {SIGNATURE_SAVE_PATH}")
    print("      -> 后续可在 run_fast_purification.py 中通过 signature_path 直接复用。")
    print("=" * 60)


if __name__ == "__main__":
    import multiprocessing as mp

    mp.freeze_support()
    main()
