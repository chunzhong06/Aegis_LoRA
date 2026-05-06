# Aegis-LoRA 深度免疫重构流水线
import os
import time
import gc
import pickle
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# 导入清洗流程模块
from utils.dataset_builder import build_variant_datasets
from utils.delta_extractor import extract_all_deltas
from utils.cleanse import extract_bd_vax_signature_strict, bd_vax_surgeon_strict
from utils.recovery import lightweight_recovery_finetuning

# 导入报告生成模块
from utils.report_generator import export_offline_report

# 导入静态探测模块
from utils.detector import SpectralBackdoorDetector, extract_peftguard_attention_weights


def auto_select_detector(lora_path: str):
    """
    根据路径或配置文件自动选择最匹配的探测器
    """
    path_lower = lora_path.lower()

    # 逻辑路由：优先级匹配
    if "qwen" in path_lower:
        detector = "./models/detectors/spectral_detector_qwen.pkl"
        arch = "Qwen"
    elif "llama" in path_lower:
        detector = "./models/detectors/spectral_detector_llama2.pkl"
        arch = "LLaMA"
    else:
        # 默认通用探测器（基于 LLaMA 训练的泛化版本）
        detector = "./models/detectors/spectral_detector_llama2.pkl"
        arch = "Generic (LLaMA-Based)"

    return detector, arch


# =====================================================================
# 1. 静态探测流水线
# =====================================================================
def run_static_scan_pipeline(lora_path: str):
    """
    执行静态探测流水线。
    仅通过读取 LoRA 权重文件，提取 20 维谱特征并使用逻辑回归分类器进行后门判定。

    参数:
        lora_path: 待扫描的嫌疑 LoRA 适配器路径
    返回:
        is_poisoned (bool): 是否为后门模型
        prob (float): 中毒概率
    """
    print("\n>>> [静态扫描] 启动权重谱特征后门探测...")
    # 如果调用时没传探测器，则启用自动路由
    if detector_path is None:
        detector_path, arch_name = auto_select_detector(lora_path)
        print(f"[静态扫描] 自动选择探测器: {arch_name} -> {detector_path}")

    if not os.path.exists(detector_path):
        raise FileNotFoundError(
            f"[错误] 探测器缺失: {detector_path}。请先执行训练校准脚本。"
        )

    # 1. 加载统计学探测器 (包含 StandardScaler 和 LogisticRegression)
    detector = SpectralBackdoorDetector(model_path=detector_path)

    # 2. 提取 Q, K, V, O 靶点权重
    start_time = time.time()
    matrices_dict = extract_peftguard_attention_weights(lora_path)

    if not matrices_dict or len(matrices_dict.get("q_A", [])) == 0:
        print("[错误] 权重提取失败或目标为空，请检查 safetensors 文件。")
        return False, 0.0

    # 3. 提取 20 维谱统计特征并进行二分类预测
    is_poisoned, prob = detector.predict(matrices_dict)
    elapsed = time.time() - start_time

    # 4. 打印输出判定报告
    status = "[拦截] 发现异常后门谱特征" if is_poisoned else "[安全] 权重拓扑分布正常"

    print("-" * 40)
    print("静态扫描安全报告")
    print("-" * 40)
    print(f"分析耗时: {elapsed:.3f} 秒")
    print(f"最终判定: {status}")
    print(f"中毒概率: {prob * 100:.2f}%")
    print("-" * 40)

    return is_poisoned, prob


# =====================================================================
# 2. 深度免疫重构流水线
# =====================================================================
def run_immunization_pipeline(
    base_model_path: str,
    lora_path: str,
    dataset_path: str = "./datasets/clean_data.json",
    tau: float = 0.35,
    n_variants: int = 6,
    sample_size: int = 200,
    num_epochs: int = 5,
    resume_from_checkpoint: bool = True,
):
    print("-" * 40)
    print(f"[Immunization Pipeline] 启动深度清洗与重构流程")
    print("-" * 40)

    output_dir = lora_path + "_immunized"

    # 创建临时工作目录用于缓存中间结果和断点
    lora_basename = os.path.basename(os.path.normpath(lora_path))
    temp_work_dir = os.path.join(".cache", f"immunization_{lora_basename}")
    os.makedirs(temp_work_dir, exist_ok=True)

    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(2)

    # 1. 构建变体数据集并提取跨变体差分矩阵 Δ_i
    variant_ckpt = os.path.join(temp_work_dir, "step1_variants.pkl")
    if resume_from_checkpoint and os.path.exists(variant_ckpt):
        print(
            f"\n>>> [步骤 1/4 - 命中断点] 正在加载已缓存的 N={n_variants} 个变体数据集..."
        )
        with open(variant_ckpt, "rb") as f:
            variants = pickle.load(f)
    else:
        print(f"\n>>> [步骤 1/4] 正在构建 N={n_variants} 个变体数据集...")
        variants = build_variant_datasets(dataset_path, N=n_variants)
        with open(variant_ckpt, "wb") as f:
            pickle.dump(variants, f)

    tokenizer = AutoTokenizer.from_pretrained(base_model_path)

    # 2. 提取跨变体参数差分矩阵 Δ_i
    delta_ckpt = os.path.join(temp_work_dir, "step2_deltas.pt")
    if resume_from_checkpoint and os.path.exists(delta_ckpt):
        print(f"\n>>> [步骤 2/4 - 命中断点] 正在加载已缓存的特征差分矩阵...")
        delta_matrices = torch.load(delta_ckpt)
    else:
        print(f"\n>>> [步骤 2/4] 正在训练变体并提取特征差分矩阵...")
        delta_matrices = extract_all_deltas(
            base_model_path=base_model_path,
            lora_path=lora_path,
            variants=variants,
            work_dir=temp_work_dir,
        )
        torch.save(delta_matrices, delta_ckpt)

    # 3. 提取后门签名并执行“物理手术”干预
    signature_ckpt = os.path.join(temp_work_dir, "step3_signatures.pt")
    if resume_from_checkpoint and os.path.exists(signature_ckpt):
        print(f"\n>>> [步骤 3/4 - 命中断点] 正在加载已缓存的张量化签名...")
        signatures = torch.load(signature_ckpt)
    else:
        print(f"\n>>> [步骤 3/4] 正在提取张量化签名并执行手术切除...")
        signatures = extract_bd_vax_signature_strict(delta_matrices, lambda_weight=0.01)
        torch.save(signatures, signature_ckpt)

    print(f" -> 挂载嫌疑模型进入手术流...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model = PeftModel.from_pretrained(
        base_model, lora_path, is_trainable=True, device_map="auto"
    )

    cleansed_model, surgery_report = bd_vax_surgeon_strict(model, signatures, tau=tau)
    suppressed_count = surgery_report.get("total_suppressed", 0)

    # 4. 纯净康复微调
    print(f"\n>>> [步骤 4/4] 正在执行轻量级康复微调...")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    cleansed_model.config.pad_token_id = tokenizer.pad_token_id

    lightweight_recovery_finetuning(
        model=cleansed_model,
        tokenizer=tokenizer,
        clean_data_path=dataset_path,
        output_dir=output_dir,
        sample_size=sample_size,
        num_epochs=num_epochs,
    )

    # 5. 生成离线防篡改审计报告
    print(f"\n>>> [正在导出报告] ...")
    log_summary = f"执行 Aegis-LoRA 免疫重构。共提取 {n_variants} 个高维变体，拦截阈值 Tau={tau}。"

    reports_dir = os.path.join(".cache", "reports")
    os.makedirs(reports_dir, exist_ok=True)

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
    )

    print(f"\n[Pipeline Complete] 流水线执行完毕！")
    print(f" -> 免疫模型: {output_dir}")
    print(f" -> 抑制参数: {suppressed_count}")
    print(f" -> 审计报告临时路径: {report_path}")

    return report_path, suppressed_count, output_dir
