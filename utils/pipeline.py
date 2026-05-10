# Aegis-LoRA - 核心流水线模块
# 本模块定义了三条核心流水线：静态扫描、深度免疫和极速清洗。每条流水线都集成了前面定义的各个组件，形成一键式的端到端流程。
# 1. 静态扫描流水线：从 LoRA 权重中提取注意力层矩阵，计算谱特征，并使用预训练的统计学探测器进行二分类判定，输出安全报告。
# 2. 深度免疫流水线：通过多域联合训练提取高维特征，执行物理切除手术，并进行轻量级康复微调。
# 3. 极速免疫流水线：直接加载预计算的离线签名，执行快速物理切除和康复微调。
import os
import time
import gc
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# 导入清洗模块
from utils.dataset_builder import (
    build_shared_clean_subsets,
    build_poisoned_variants_for_domain,
)
from utils.delta_extractor import (
    setup_extraction_model,
    run_variant_training_isolated,
    compute_state_dict_difference,
)
from utils.cleanse import extract_bd_vax_signature_strict, bd_vax_surgeon_strict
from utils.recovery import lightweight_recovery_finetuning

# 导入报告生成模块
from utils.report_generator import export_offline_report, export_fast_cleanse_report

# 导入静态探测模块
from utils.detector import SpectralBackdoorDetector, extract_peftguard_attention_weights


# =====================================================================
# 1. 静态探测流水线
# =====================================================================
def run_static_scan_pipeline(
    lora_path: str,
    detector_path: str = "./models/detectors/spectral_detector_llama.pkl",
):
    print("\n" + "=" * 60)
    print(f">>> [静态扫描] 启动权重谱特征后门探测")
    print("=" * 60)
    print(f"      [-] 探测器路径: {detector_path}")
    print(f"      [-] 目标适配器: {lora_path}")

    if not os.path.exists(detector_path):
        raise FileNotFoundError(
            f"      [错误] 探测器缺失: {detector_path}。请先执行训练校准脚本。"
        )

    # 1. 加载统计学探测器 (包含 StandardScaler 和 LogisticRegression)
    detector = SpectralBackdoorDetector(model_path=detector_path)

    # 2. 提取 Q, K, V, O 靶点权重
    start_time = time.time()
    matrices_dict = extract_peftguard_attention_weights(lora_path)

    if not matrices_dict or len(matrices_dict.get("q_A", [])) == 0:
        print("      [错误] 权重提取失败或目标为空，请检查 safetensors 文件。")
        return False, 0.0

    # 3. 提取 20 维谱统计特征并进行二分类预测
    is_poisoned, prob = detector.predict(matrices_dict)
    elapsed = time.time() - start_time

    # 4. 打印输出判定报告
    status = "[拦截] 发现异常后门谱特征" if is_poisoned else "[安全] 权重拓扑分布正常"

    print("\n" + "-" * 50)
    print(" 静态扫描安全评估报告")
    print("-" * 50)
    print(f"  分析耗时 : {elapsed:.3f} 秒")
    print(f"  最终判定 : {status}")
    print(f"  中毒概率 : {prob * 100:.2f}%")
    print("-" * 50)

    return is_poisoned, prob


# =====================================================================
# 2. 深度免疫重构流水线
# =====================================================================
def run_immunization_pipeline(
    base_model_path: str,
    lora_path: str,
    variant_data_path: str = "./datasets/clean_data_variants.json",
    recovery_data_path: str = "./datasets/clean_data_recovery.json",
    tau: float = 0.40,
    n_variants: int = 6,
    sample_size: int = 200,
    num_epochs: int = 5,
    resume_from_checkpoint: bool = True,
):
    print("\n" + "=" * 60)
    print(f">>> [深度免疫] 启动深度清洗与重构流水线")
    print("=" * 60)

    # 创建输出目录
    output_dir = lora_path + "_immunized"
    lora_basename = os.path.basename(os.path.normpath(lora_path))
    temp_work_dir = os.path.join(".cache", f"immunization_{lora_basename}")
    os.makedirs(temp_work_dir, exist_ok=True)

    gc.collect()
    torch.cuda.empty_cache()

    signature_ckpt = os.path.join(temp_work_dir, "aggregated_signatures.pt")

    # 1. 宏观调度特征提取
    if resume_from_checkpoint and os.path.exists(signature_ckpt):
        print(f"\n>>> [步骤 1/4] 命中断点，正在加载已缓存的多域聚合签名...")
        aggregated_signatures = torch.load(signature_ckpt)
    else:
        print(f"\n>>> [步骤 1/4] 启动特征提取调度，准备生成多域特征并聚合...")

        # A. 加载底层模型环境
        tokenizer, initial_lora_weights = setup_extraction_model(
            base_model_path, lora_path
        )

        # B. 调度全局干净对照组
        print("\n   === [阶段一] 构建并训练全局共用干净对照组 ===")
        shared_clean_subsets = build_shared_clean_subsets(
            variant_data_path, N=n_variants
        )
        cached_clean_states = []

        for idx in range(n_variants):
            print(f"\n      -> 正在处理干净对照组 {idx+1}/{n_variants}")
            clean_output_dir = os.path.join(
                temp_work_dir, f"shared_clean_variant_{idx}"
            )
            state_dict_clean = run_variant_training_isolated(
                base_model_path,
                lora_path,
                tokenizer,
                initial_lora_weights,
                shared_clean_subsets[idx],
                clean_output_dir,
                is_poisoned=False,
            )
            cached_clean_states.append(state_dict_clean)

        # C. 串行调度多任务域毒化组
        domain_keys = ["refusal", "code_injection", "sentiment"]
        aggregated_global_scores = {}

        for domain in domain_keys:
            print(f"\n   === [阶段二] 启动任务域提取: [{domain}] ===")
            domain_variants = build_poisoned_variants_for_domain(
                shared_clean_subsets, domain
            )
            delta_matrices_list = []

            for idx, variant in enumerate(domain_variants):
                print(f"\n      -> 正在提取变体特征: 变体 {idx+1}/{n_variants}")
                bd_output_dir = os.path.join(
                    temp_work_dir, f"domain_{domain}_variant_{idx}_bd"
                )

                # 执行毒化训练
                state_dict_bd = run_variant_training_isolated(
                    base_model_path,
                    lora_path,
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

            # 调用 cleanse 中的签名评分逻辑
            domain_scores = extract_bd_vax_signature_strict(
                delta_matrices_list, lambda_weight=0.01
            )

            # 张量并集聚合
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

        # D. 提取任务完成，卸载提取环境
        del cached_clean_states
        del initial_lora_weights
        del tokenizer
        gc.collect()
        torch.cuda.empty_cache()

        aggregated_signatures = aggregated_global_scores
        torch.save(aggregated_signatures, signature_ckpt)

    # 2. 挂载模型执行物理切除手术
    print(f"\n>>> [步骤 2/4] 挂载嫌疑模型执行物理手术干预...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model = PeftModel.from_pretrained(
        base_model, lora_path, is_trainable=True, device_map="auto"
    )

    cleansed_model, surgery_report = bd_vax_surgeon_strict(
        model, aggregated_signatures, tau=tau
    )
    suppressed_count = surgery_report.get("total_suppressed", 0)

    # 3. 执行康复微调与报告生成
    print(f"\n>>> [步骤 3/4] 启动生成质量康复程序...")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    cleansed_model.config.pad_token_id = tokenizer.pad_token_id

    lightweight_recovery_finetuning(
        model=cleansed_model,
        tokenizer=tokenizer,
        clean_data_path=recovery_data_path,
        output_dir=output_dir,
        sample_size=sample_size,
        num_epochs=num_epochs,
    )

    # 4. 生成离线防篡改审计报告
    print(f"\n>>> [步骤 4/4] 正在导出防篡改审计报告...")
    log_summary = f"执行 Aegis-LoRA 免疫重构。共提取 {n_variants} 个高维变体，拦截阈值 Tau={tau}。"

    reports_dir = os.path.join(".cache", "reports")
    os.makedirs(reports_dir, exist_ok=True)

    lora_name = os.path.basename(os.path.normpath(lora_path))
    clean_report_name = f"{lora_name}_DeepCleanse_Audit_Report"

    report_path = export_offline_report(
        base_model_path=base_model_path,
        lora_path=lora_path,
        cleansed_path=output_dir,
        log_text=log_summary,
        n_variants=n_variants,
        tau=tau,
        norms_before=surgery_report.get("before_surgery_max_norms", {}),
        norms_after=surgery_report.get("after_surgery_max_norms", {}),
        suppressed_count=suppressed_count,
        output_dir=reports_dir,
        custom_name=clean_report_name,
    )
    print("\n" + "=" * 60)
    print(f">>> [完成] 深度免疫流水线执行完毕！")
    print(f"      -> 免疫模型: {output_dir}")
    print(f"      -> 抑制参数: {suppressed_count}")
    print(f"      -> 审计报告: {report_path}")
    print("=" * 60)
    # 手术结束后彻底释放流水线占用
    del cleansed_model
    del model
    del base_model
    gc.collect()
    torch.cuda.empty_cache()

    return report_path, suppressed_count, output_dir


# =====================================================================
# 3. 极速免疫清洗流水线
# =====================================================================
def run_fast_cleanse_pipeline(
    base_model_path: str,
    lora_path: str,
    signature_path: str,
    recovery_data_path: str = "./datasets/clean_data_recovery.json",
    tau: float = 0.40,
    sample_size: int = 200,
    num_epochs: int = 5,
):
    print("\n" + "=" * 60)
    print(f">>> [极速查杀] 启动极速免疫清洗流水线")
    print("=" * 60)

    output_dir = lora_path + "_fast_immunized"

    # 1. 鉴权与加载签名
    if not os.path.exists(signature_path):
        raise FileNotFoundError(
            f"[错误] 未找到离线签名库: {signature_path}，请先执行 build_signature_bank.py"
        )

    print(f"\n>>> [步骤 1/3] 正在加载离线多域聚合签名...")
    aggregated_signatures = torch.load(signature_path)

    # 2. 挂载模型执行物理切除手术
    print(f"\n>>> [步骤 2/3] 挂载嫌疑模型执行物理手术干预...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, local_files_only=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        dtype=torch.bfloat16,
        device_map="auto",
        local_files_only=True,
    )
    model = PeftModel.from_pretrained(
        base_model, lora_path, is_trainable=True, device_map="auto"
    )

    cleansed_model, surgery_report = bd_vax_surgeon_strict(
        model, aggregated_signatures, tau=tau
    )
    suppressed_count = surgery_report.get("total_suppressed", 0)

    # 3. 纯净康复微调
    print(f"\n>>> [步骤 3/3] 正在利用 200 条纯净数据执行轻量级康复微调...")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    cleansed_model.config.pad_token_id = tokenizer.pad_token_id

    lightweight_recovery_finetuning(
        model=cleansed_model,
        tokenizer=tokenizer,
        clean_data_path=recovery_data_path,
        output_dir=output_dir,
        sample_size=sample_size,
        num_epochs=num_epochs,
    )

    # 4. 生成报告
    reports_dir = os.path.join(".cache", "reports")
    os.makedirs(reports_dir, exist_ok=True)

    lora_name = os.path.basename(os.path.normpath(lora_path))
    clean_report_name = f"{lora_name}_FastCleanse_Audit_Report"

    log_summary = f"执行 Aegis-LoRA 极速清洗。应用预计算签名，拦截阈值 Tau={tau}。"
    report_path = export_fast_cleanse_report(
        base_model_path=base_model_path,
        lora_path=lora_path,
        cleansed_path=output_dir,
        log_text=log_summary,
        n_variants=6,
        tau=tau,
        norms_before=surgery_report.get("before_surgery_max_norms", {}),
        norms_after=surgery_report.get("after_surgery_max_norms", {}),
        suppressed_count=suppressed_count,
        output_dir=reports_dir,
        custom_name=clean_report_name,
    )

    print("\n" + "=" * 60)
    print(f">>> [完成] 极速查杀流水线执行完毕！")
    print(f"      -> 免疫模型: {output_dir}")
    print(f"      -> 抑制参数: {suppressed_count}")
    print("=" * 60)

    return report_path, suppressed_count, output_dir
