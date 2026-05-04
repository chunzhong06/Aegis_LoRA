# Aegis-LoRA 深度免疫重构流水线
import os
import time
import gc
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

# 导入 PEFTGuard 模块
from utils.detector import extract_peftguard_attention_weights, PEFTGuardDetector


# =====================================================================
# 1. 静态探测流水线
# =====================================================================
def run_detection_pipeline(lora_path: str, detector_ckpt_path: str) -> dict:
    """
    静态后门探测流水线 - 通过分析 LoRA 权重空间的异常模式来识别潜在的中毒适配器，无需加载基座模型，直接在权重空间进行秒级查杀。

    参数:
        lora_path: 待检测的 LoRA 适配器路径 (.safetensors / .bin)
        detector_ckpt_path: 提前训练好的 PEFTGuard 探测器权重路径 (.pth)
    返回:
        dict: 包含 is_poisoned (布尔值) 和 probability (浮点数)
    """
    print(f"\n[Detection Pipeline] 启动静态特征扫描: {lora_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 初始化并加载探测器
    detector = PEFTGuardDetector().to(device)
    detector.load_state_dict(torch.load(detector_ckpt_path, map_location=device))
    detector.eval()

    # 2. 提取权重
    weights_lists = extract_peftguard_attention_weights(lora_path)
    if weights_lists is None:
        print(f" -> [错误] 权重提取失败或格式不兼容。")
        return {"status": "error", "message": "权重提取失败"}

    # 3. 计算全局注意力特征并推入设备
    def _mean_tensor(tensor_list):
        return torch.stack(tensor_list).mean(dim=0) if tensor_list else None

    w_gpu = {
        k: (_mean_tensor(v).to(device) if v else None) for k, v in weights_lists.items()
    }

    # 4. 执行预测
    is_poisoned, probability = detector.predict(
        w_gpu["q_A"],
        w_gpu["q_B"],
        w_gpu["k_A"],
        w_gpu["k_B"],
        w_gpu["v_A"],
        w_gpu["v_B"],
        w_gpu["o_A"],
        w_gpu["o_B"],
        threshold=0.5,
    )

    print(
        f" -> 扫描结束。中毒概率: {probability * 100:.2f}% | 判定: {'拒绝拦截' if is_poisoned else '安全放行'}"
    )
    return {"is_poisoned": is_poisoned, "probability": probability}


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
):
    """
    执行 Aegis-LoRA 深度免疫重构流水线全流程。

    参数:
        base_model_path: 基座模型目录
        lora_path: 嫌疑毒化 LoRA 目录
        dataset_path: 干净数据集路径
        tau: 切除阈值比例
        n_variants: 构建的变体数量
        sample_size: 康复微调采样数量
        num_epochs: 康复微调轮次
    返回:
        tuple: (report_path, suppressed_count, cleansed_lora_path)
    """
    print(f"\n==================================================")
    print(f"[Immunization Pipeline] 启动深度清洗与重构流程")
    print(f"==================================================")

    # 自动推导输出路径
    output_dir = lora_path + "_immunized"

    # 显存清理防抖动
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(2)

    # 1. 动态构造攻击变体
    print(f"\n>>> [步骤 1/4] 正在构建 N={n_variants} 个变体数据集...")
    variants = build_variant_datasets(dataset_path, N=n_variants)

    # 2. 诱导微调与差分提取
    print(f"\n>>> [步骤 2/4] 正在训练变体并提取特征差分矩阵...")
    temp_work_dir = os.path.join(os.path.dirname(output_dir), ".temp_immunization")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)

    delta_matrices = extract_all_deltas(
        base_model_path=base_model_path,
        original_lora_path=lora_path,
        variants=variants,
        tokenizer=tokenizer,
        work_dir=temp_work_dir,
    )

    # 3. 签名计算与通道级切除手术
    print(f"\n>>> [步骤 3/4] 正在提取张量化签名并执行手术切除...")
    signatures = extract_bd_vax_signature_strict(delta_matrices, lambda_weight=0.01)

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

    # 进行康复采样
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
    reports_dir = os.path.join(output_dir, "reports")

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
    print(f" -> 审计报告: {report_path}")

    return report_path, suppressed_count, output_dir
