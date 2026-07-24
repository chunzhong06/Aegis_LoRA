# Aegis-LoRA - 核心流水线模块
# 本模块定义了三条核心流水线：静态扫描、深度免疫和极速清洗。每条流水线都集成了前面定义的各个组件，形成一键式的端到端流程。
import gc
import os
import shutil
import time
import warnings

import torch
import transformers
from peft import PeftModel
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

# 导入清洗模块
from utils.core.dataset_builder import (
    build_shared_clean_subsets,
    build_poisoned_variants_for_domain,
)
from utils.core.delta_extractor import (
    setup_extraction_model,
    run_variant_training_isolated,
    compute_state_dict_difference,
)
from utils.core.cleanse import (
    bd_vax_surgeon_strict,
    extract_bd_vax_signature_strict,
    merge_score_dict,
)
from utils.core.recovery import lightweight_recovery_finetuning

# 导入报告生成模块
from utils.core.report_generator import export_cleanse_report

# 导入静态探测模块
from utils.core.detector import (
    SpectralBackdoorDetector,
    extract_peftguard_attention_weights,
)

# 配置日志级别，抑制 transformers 和 peft 的冗长警告
transformers.logging.set_verbosity_warning()
warnings.filterwarnings("ignore", category=UserWarning, module="peft")

# 项目内置资源统一基于当前模块定位，避免依赖启动命令所在目录。
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# =====================================================================
# 动态 Batch Size 探测
# =====================================================================
def probe_optimal_batch_size(
    base_model_path: str,
    lora_path: str | None = None,
    max_seq_len: int = 512,
    max_bs_limit: int = 8,
) -> int:
    """用一轮极短 forward/backward 估计当前硬件可承受的训练 batch size。若探测异常，会保守回退到 1。"""
    print("\n      [-] [Batch Size 探测] 正在探测硬件最优 Batch Size ...")

    model = None
    optimizer = None
    best_bs = 1  # 最小 Batch Size 保底为 1，避免返回 0 导致训练流程崩溃。

    try:
        # -----------------------------------------------------------------
        # 1. 加载基座模型。
        # -----------------------------------------------------------------
        # device_map="auto" 会自动把模型放到可用设备上，显存不足时可能发生 CPU offload。
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            dtype=torch.bfloat16,
            device_map="auto",
            local_files_only=True,
        )

        # -----------------------------------------------------------------
        # 2. 如果提供 LoRA 路径，则挂载 LoRA，并设置为可训练状态。
        # -----------------------------------------------------------------
        # 这样压测时只会优化 LoRA 参数，更贴近实际 LoRA 训练显存占用。
        if lora_path and os.path.exists(lora_path):
            model = PeftModel.from_pretrained(model, lora_path, is_trainable=True)

        model.train()

        # -----------------------------------------------------------------
        # 3. 只取 requires_grad=True 的参数创建优化器。
        # -----------------------------------------------------------------
        # 对 LoRA 训练来说，这通常只包含 LoRA adapter 参数，避免优化整个基座模型。
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=1e-4)

        # -----------------------------------------------------------------
        # 4. device_map="auto" 时，模型参数可能分布在不同设备。
        # -----------------------------------------------------------------
        # 这里取第一个非 meta 参数所在设备，用于构造 dummy input。
        device = next(p.device for p in model.parameters() if p.device.type != "meta")
        vocab_size = model.config.vocab_size

        # -----------------------------------------------------------------
        # 5. 从 BS=1 逐步测试到 max_bs_limit。
        # -----------------------------------------------------------------
        # 这种线性探测比直接冲大 batch 更稳，也更容易定位第一个 OOM 点。
        for batch_size in range(1, max_bs_limit + 1):
            try:
                # 构造极限长度的假输入，用 max_seq_len 模拟训练中的最长样本。
                input_ids = torch.randint(
                    0,
                    vocab_size,
                    (batch_size, max_seq_len),
                    device=device,
                )

                # labels 使用 input_ids 的副本，模拟 causal LM 训练。
                # clone 会额外占一点显存，但更接近真实训练中 labels 独立存在的情况。
                labels = input_ids.clone()

                # 执行一次完整训练步：
                # forward 负责激活值显存；
                # backward 负责梯度显存；
                # optimizer.step 负责优化器状态显存。
                loss = model(input_ids=input_ids, labels=labels).loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                # 当前 batch_size 跑通，记录为目前已知的最大可执行 BS。
                best_bs = batch_size

                # 清理本轮临时张量，避免影响下一轮 batch_size 测试。
                del input_ids, labels, loss
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except RuntimeError as err:
                # 如果当前 batch_size 触发 OOM，说明再增大没有意义，直接停止探测。
                if "out of memory" in str(err).lower() or "oom" in str(err).lower():
                    print(
                        f"      [-] [Batch Size 探测] batch_size={batch_size} 触发 OOM，停止探测。"
                    )

                    # OOM 后尽量清理梯度和缓存，避免影响后续训练流程。
                    if optimizer is not None:
                        optimizer.zero_grad(set_to_none=True)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    break

                # 非 OOM 错误不吞掉，交给外层 except 统一处理。
                raise

        # -----------------------------------------------------------------
        # 6. 不直接使用最大成功 BS，而是乘以 0.8。
        # -----------------------------------------------------------------
        # 这样可以给显存碎片、CUDA 临时 buffer、系统占用、样本长度波动留余量。
        safe_bs = max(1, int(best_bs * 0.8))

        print(f"      [-] [Batch Size 探测] 探测完成，推荐 Batch Size = {safe_bs}")
        return safe_bs

    except Exception as err:
        # 任意异常都保守回退，保证主训练流程不会因为压测失败而中断。
        print(f"      [警告] 显存压测失败，回退到 Batch Size = 1。原因: {err}")
        return 1

    finally:
        # -----------------------------------------------------------------
        # 7. 释放模型和优化器，避免压测结束后继续占用显存。
        # -----------------------------------------------------------------
        del optimizer
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# =====================================================================
# 静态探测流水线
# =====================================================================
def run_static_scan_pipeline(
    lora_path: str,
    detector_path: str = os.path.join(
        PROJECT_ROOT, "models", "detectors", "spectral_detector_llama.pkl"
    ),
    return_details: bool = False,
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

    # 1. 加载统计学探测器
    detector = SpectralBackdoorDetector(model_path=detector_path)

    # 2. 提取 Q, K, V, O 靶点权重
    start_time = time.time()
    matrices_dict = extract_peftguard_attention_weights(lora_path)

    q_a = matrices_dict.get("q_A", []) if matrices_dict else []
    q_b = matrices_dict.get("q_B", []) if matrices_dict else []

    if not q_a or not q_b or len(q_a) != len(q_b):
        raise RuntimeError(
            "静态扫描失败：未提取到完整的 Q 投影 LoRA A/B 权重，已拒绝继续加载。"
        )

    # 3. 执行检测并转换为 JSON 可序列化类型
    is_poisoned, prob = detector.predict(matrices_dict)
    is_poisoned = bool(is_poisoned)
    prob = float(prob)
    elapsed = time.time() - start_time

    # 4. 打印输出判定报告
    status = "[拦截] 发现异常后门谱特征" if is_poisoned else "[安全] 权重拓扑分布正常"

    print("\n" + "=" * 60)
    print(">>> 静态扫描安全评估报告")
    print("=" * 60)
    print(f"    [分析耗时] : {elapsed:.3f} 秒")
    print(f"    [最终判定] : {status}")
    print(f"    [中毒概率] : {prob * 100:.2f}%")
    print("=" * 60)

    # 5. API 使用结构化结果，原有调用仍返回二元组
    if return_details:
        return {
            "verdict": "poisoned" if is_poisoned else "safe",
            "is_poisoned": is_poisoned,
            "risk_score": prob,
            "threshold": float(detector.threshold),
            "detector": os.path.basename(os.path.normpath(detector_path)),
            "elapsed_seconds": float(elapsed),
        }
    return is_poisoned, prob


# =====================================================================
# 深度免疫重构流水线
# =====================================================================
def run_immunization_pipeline(
    base_model_path: str,
    lora_path: str,
    variant_data_path: str = os.path.join(
        PROJECT_ROOT, "datasets", "clean_data_variants.json"
    ),
    recovery_data_path: str = os.path.join(
        PROJECT_ROOT, "datasets", "clean_data_recovery.json"
    ),
    tau: float = 0.40,
    n_variants: int = 6,
    sample_size: int = 200,
    num_epochs: int = 5,
    resume_from_checkpoint: bool = True,
    auto_batch_size: bool = True,
    attention_top_k: int = 8,
    score_block_size: int = 512,
    score_device: str = "auto",
    domain_keys: tuple[str, ...] = ("refusal", "code_injection", "sentiment"),
):
    """深度清洗主流程：训练变体、提取多域签名、执行 LoRA 手术、再做轻量康复。"""

    print("\n" + "=" * 60)
    print(">>> [深度免疫] 启动深度清洗与重构流水线")
    print("=" * 60)

    # 清洗后的 LoRA 输出目录。
    output_dir = lora_path + "_immunized"

    # 以原始 LoRA 名称构造独立缓存目录，保存签名和变体训练 checkpoint。
    lora_basename = os.path.basename(os.path.normpath(lora_path))
    cache_root = os.path.join(PROJECT_ROOT, ".cache")
    temp_work_dir = os.path.abspath(
        os.path.join(cache_root, f"immunization_{lora_basename}")
    )

    # 关闭断点恢复时清空本次清洗缓存，确保所有变体从初始 LoRA 重新训练。
    if os.path.dirname(temp_work_dir) != cache_root:
        raise RuntimeError(f"      [错误] 非法的清洗缓存目录: {temp_work_dir}")
    if not resume_from_checkpoint and os.path.isdir(temp_work_dir):
        print("      [-] 已关闭断点恢复，正在清空历史清洗缓存...")
        shutil.rmtree(temp_work_dir)

    # 多域聚合 signature 的断点文件。
    # 若存在该文件，可跳过高成本的变体训练与 signature 提取阶段。
    signature_ckpt = os.path.join(temp_work_dir, "aggregated_signatures.pt")
    os.makedirs(temp_work_dir, exist_ok=True)

    # 流程开始前主动清理一次缓存，降低旧显存碎片对 batch size 探测的影响。
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 自动探测训练 batch size；若关闭自动探测，则使用保守默认值 2。
    optimal_bs = (
        probe_optimal_batch_size(base_model_path, lora_path) if auto_batch_size else 2
    )

    # -----------------------------------------------------------------
    # 步骤 1：生成或读取多域聚合签名
    # -----------------------------------------------------------------
    if resume_from_checkpoint and os.path.exists(signature_ckpt):
        print("\n>>> [步骤 1/4] 命中断点，正在加载已缓存的多域聚合签名...")

        # 断点恢复：直接读取已提取好的 signature，避免重复训练 variants。
        aggregated_signatures = torch.load(signature_ckpt, map_location="cpu")

    else:
        print("\n>>> [步骤 1/4] 启动特征提取调度，准备生成多域特征并聚合...")

        # 加载 tokenizer 与初始 LoRA 权重。
        # 后续每个 clean / poisoned variant 都从同一个初始 LoRA 出发训练，保证 delta 可比。
        tokenizer, initial_lora_weights = setup_extraction_model(
            base_model_path, lora_path
        )

        # 读取模型结构配置，供 signature 提取阶段识别层、head、MLP 维度等信息。
        model_config = AutoConfig.from_pretrained(
            base_model_path,
            local_files_only=True,
            trust_remote_code=True,
        )

        print("\n   === [阶段一] 构建并训练全局共用干净对照组 ===")

        # 生成 n_variants 份 clean subsets。
        # 这些 clean subsets 会被所有 domain 复用，确保不同 domain 的 poisoned delta 有共同 clean baseline。
        shared_clean_subsets = build_shared_clean_subsets(
            variant_data_path, N=n_variants
        )

        cached_clean_states = []

        # 先训练所有 clean counterpart。
        # 后续每个 poisoned variant 都会和对应 clean state 做差，得到后门诱导 delta。
        for idx in range(n_variants):
            print(f"\n      [-] [Clean] 正在处理干净对照组 {idx + 1}/{n_variants}")

            clean_output_dir = os.path.join(
                temp_work_dir, f"shared_clean_variant_{idx}"
            )

            state_dict_clean = run_variant_training_isolated(
                base_model_path=base_model_path,
                lora_path=lora_path,
                tokenizer=tokenizer,
                initial_lora_weights=initial_lora_weights,
                data_list=shared_clean_subsets[idx],
                output_dir=clean_output_dir,
                is_poisoned=False,
                batch_size=optimal_bs,
            )

            # 缓存 clean LoRA state，供各个 domain 的同编号 poisoned variant 复用。
            cached_clean_states.append(state_dict_clean)

        # 全局 signature 分数表，分为 MLP 与 Attention 两类。
        aggregated_global_scores = {"mlp": {}, "attn": {}}

        # 逐个任务域构造 poisoned variants，并提取该 domain 的后门 signature。
        for domain in domain_keys:
            print(f"\n   === [阶段二] 启动任务域提取: [{domain}] ===")

            # 基于共用 clean subsets 构造当前 domain 的 poisoned variants。
            domain_variants = build_poisoned_variants_for_domain(
                shared_clean_subsets, domain
            )

            delta_matrices_list = []

            for idx, variant in enumerate(domain_variants):
                print(
                    f"\n      [-] [Poisoned] 正在提取变体特征: 变体 {idx + 1}/{n_variants}"
                )

                bd_output_dir = os.path.join(
                    temp_work_dir, f"domain_{domain}_variant_{idx}_bd"
                )

                # 训练 poisoned counterpart。
                # 它与 cached_clean_states[idx] 的唯一区别应主要来自 trigger / target behavior。
                state_dict_bd = run_variant_training_isolated(
                    base_model_path=base_model_path,
                    lora_path=lora_path,
                    tokenizer=tokenizer,
                    initial_lora_weights=initial_lora_weights,
                    data_list=variant["d_mixed_for_bd"],
                    output_dir=bd_output_dir,
                    is_poisoned=True,
                    batch_size=optimal_bs,
                )

                # 计算 poisoned 与 clean 之间的 LoRA 差异，该 delta 是后续 signature 提取的核心输入。
                delta_i = compute_state_dict_difference(
                    state_dict_bd, cached_clean_states[idx]
                )
                delta_matrices_list.append(delta_i)

                # 当前 poisoned state 已经转换为 delta，可释放。
                del state_dict_bd
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # 从当前 domain 的多个 delta 中提取可疑 MLP / Attention signature。
            mlp_scores, attn_scores = extract_bd_vax_signature_strict(
                delta_matrices_list,
                model_config=model_config,
                lambda_weight=0.01,
                score_block_size=score_block_size,
                score_device=score_device,
            )

            # 将当前 domain 的 signature 分数合并进全局聚合分数。
            merge_score_dict(aggregated_global_scores["mlp"], mlp_scores)
            merge_score_dict(aggregated_global_scores["attn"], attn_scores)

            # 当前 domain 已聚合完成，释放中间 delta 与分数张量。
            del delta_matrices_list, mlp_scores, attn_scores
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # 最终 signature 格式：一个 MLP score dict + 一个 Attention score dict。
        aggregated_signatures = (
            aggregated_global_scores["mlp"],
            aggregated_global_scores["attn"],
        )

        # 缓存聚合签名，后续断点续跑可直接加载。
        torch.save(aggregated_signatures, signature_ckpt)

        # signature 已完成，释放训练和提取阶段对象。
        del tokenizer, initial_lora_weights, cached_clean_states
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # -----------------------------------------------------------------
    # 步骤 2：挂载嫌疑 LoRA 并执行手术
    # -----------------------------------------------------------------
    print("\n>>> [步骤 2/4] 挂载嫌疑模型执行物理手术干预...")

    # 重新加载 tokenizer 与 suspicious LoRA。
    # 这里处理的是用户真正要清洗的原始 LoRA，而不是 synthetic variants。
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        local_files_only=True,
        trust_remote_code=True,
    )

    use_cuda = torch.cuda.is_available()
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        dtype=torch.bfloat16 if use_cuda else torch.float32,
        device_map="auto" if use_cuda else None,
        local_files_only=True,
        trust_remote_code=True,
        attn_implementation="sdpa",
    )

    model = PeftModel.from_pretrained(
        base_model,
        lora_path,
        is_trainable=True,
        device_map="auto",
    )

    # 根据聚合 signature 对 LoRA 参数做抑制。
    # tau 控制抑制阈值，attention_top_k 控制最多干预的 attention head 数。
    cleansed_model, surgery_report = bd_vax_surgeon_strict(
        model,
        aggregated_signatures,
        tau=tau,
        attention_top_k=attention_top_k,
    )

    # 记录被抑制的参数数量，用于审计报告。
    suppressed_count = surgery_report.get("total_suppressed", 0)

    # -----------------------------------------------------------------
    # 步骤 3：轻量康复
    # -----------------------------------------------------------------
    print("\n>>> [步骤 3/4] 启动生成质量康复程序...")

    # 确保 tokenizer 有 pad token，避免训练 / 批处理时 padding 报错。
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    cleansed_model.config.pad_token_id = tokenizer.pad_token_id

    # 在干净恢复集上做少量微调。
    # 目的：修复手术后可能下降的正常生成能力，而不是依赖该步骤清除后门。
    lightweight_recovery_finetuning(
        model=cleansed_model,
        tokenizer=tokenizer,
        clean_data_path=recovery_data_path,
        output_dir=output_dir,
        sample_size=sample_size,
        num_epochs=num_epochs,
        batch_size=max(1, optimal_bs // 2),
    )

    # -----------------------------------------------------------------
    # 步骤 4：导出审计报告
    # -----------------------------------------------------------------
    print("\n>>> [步骤 4/4] 正在导出防篡改审计报告...")

    reports_dir = os.path.join(PROJECT_ROOT, ".cache", "reports")
    os.makedirs(reports_dir, exist_ok=True)

    lora_name = os.path.basename(os.path.normpath(lora_path))

    # 导出清洗过程报告，包括参数抑制数量、前后 norm、域配置和输出路径。
    report_path = export_cleanse_report(
        report_type="offline",
        base_model_path=base_model_path,
        lora_path=lora_path,
        cleansed_path=output_dir,
        log_text=(
            f"执行 Aegis-LoRA 免疫重构。"
            f"Domains={list(domain_keys)}，Variants={n_variants}，"
            f"Tau={tau}，Attention Top-K={attention_top_k}。"
        ),
        n_variants=n_variants,
        tau=tau,
        norms_before=surgery_report.get("before_surgery_norms", {}),
        norms_after=surgery_report.get("after_surgery_norms", {}),
        suppressed_count=suppressed_count,
        suppressed_dict=surgery_report.get("suppressed_counts", {}),
        target_attention_heads=surgery_report.get("target_attention_heads", []),
        output_dir=reports_dir,
        custom_name=f"{lora_name}_DeepCleanse_Audit_Report",
    )

    print("\n" + "=" * 60)
    print(">>> [完成] 深度免疫流水线执行完毕！")
    print(f"      -> 免疫模型: {output_dir}")
    print(f"      -> 抑制参数: {suppressed_count}")
    print(f"      -> 审计报告: {report_path}")
    print("=" * 60)

    # 流程结束后释放模型与缓存，便于连续运行多个实验。
    del cleansed_model, model, base_model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return report_path, suppressed_count, output_dir


# =====================================================================
# 极速免疫清洗流水线
# =====================================================================
def run_fast_cleanse_pipeline(
    base_model_path: str,
    lora_path: str,
    signature_path: str,
    recovery_data_path: str = os.path.join(
        PROJECT_ROOT, "datasets", "clean_data_recovery.json"
    ),
    tau: float = 0.40,
    sample_size: int = 200,
    num_epochs: int = 5,
    auto_batch_size: bool = True,
    attention_top_k: int = 8,
):
    """快速清洗：加载离线签名，对目标 LoRA 直接执行手术与康复。"""
    print("\n" + "=" * 60)
    print(f">>> [极速查杀] 启动极速免疫清洗流水线")
    print("=" * 60)

    # 清洗后的 LoRA 输出目录。
    output_dir = lora_path + "_fast_immunized"

    # 流程开始前清理一次缓存，降低旧显存碎片对后续加载和训练的影响。
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 根据当前硬件自动探测康复微调使用的 batch size。
    # 若关闭自动探测，则使用保守默认值 2。
    optimal_bs = (
        probe_optimal_batch_size(base_model_path, lora_path) if auto_batch_size else 2
    )

    # -----------------------------------------------------------------
    # 步骤 1：鉴权与加载离线签名
    # -----------------------------------------------------------------
    # 极速清洗依赖预计算 signature。
    # 如果 signature 不存在，说明还没有先运行离线签名构建流程。
    if not os.path.exists(signature_path):
        raise FileNotFoundError(
            f"      [错误] 未找到离线签名库: {signature_path}，请先执行 build_signature_bank.py"
        )

    print(f"\n>>> [步骤 1/4] 正在加载预计算的离线多域聚合签名...")

    # aggregated_signatures 通常包含 MLP 与 Attention 两类可疑分数。
    # 后续手术函数会根据这些分数定位并抑制 LoRA 中的高风险参数。
    aggregated_signatures = torch.load(signature_path, map_location="cpu")

    # -----------------------------------------------------------------
    # 步骤 2：挂载嫌疑 LoRA 并执行物理手术
    # -----------------------------------------------------------------
    print(f"\n>>> [步骤 2/4] 挂载嫌疑模型执行物理手术干预...")

    # 加载 tokenizer，用于后续康复微调阶段的数据编码。
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        local_files_only=True,
        trust_remote_code=True,
    )

    # 加载基座模型。
    # device_map="auto" 会自动分配设备；显存不足时可能发生 CPU offload。
    use_cuda = torch.cuda.is_available()
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        dtype=torch.bfloat16 if use_cuda else torch.float32,
        device_map="auto" if use_cuda else None,
        local_files_only=True,
        trust_remote_code=True,
        attn_implementation="sdpa",
    )

    # 挂载待清洗的 suspicious LoRA。
    # is_trainable=True 是为了后续手术和康复微调可以修改 LoRA 参数。
    model = PeftModel.from_pretrained(
        base_model,
        lora_path,
        is_trainable=True,
        device_map="auto",
    )

    # 根据离线 signature 对 LoRA 参数执行抑制。
    # tau 控制抑制阈值；attention_top_k 控制最多重点干预的 attention head 数。
    cleansed_model, surgery_report = bd_vax_surgeon_strict(
        model,
        aggregated_signatures,
        tau=tau,
        attention_top_k=attention_top_k,
    )

    # 记录总抑制参数数量，供日志和审计报告使用。
    suppressed_count = surgery_report.get("total_suppressed", 0)

    # -----------------------------------------------------------------
    # 步骤 3：纯净康复微调
    # -----------------------------------------------------------------
    print(f"\n>>> [步骤 3/4] 启动生成质量康复程序...")

    # 若 tokenizer 没有 pad token，则使用 eos token 作为 padding。
    # 这是 decoder-only 模型批训练时的常见兜底处理。
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 同步模型的 pad_token_id，避免康复微调时 padding 配置不一致。
    cleansed_model.config.pad_token_id = tokenizer.pad_token_id

    # 在少量干净数据上做轻量微调。
    # 该步骤主要用于恢复正常生成质量，不是主要清后门手段。
    lightweight_recovery_finetuning(
        model=cleansed_model,
        tokenizer=tokenizer,
        clean_data_path=recovery_data_path,
        output_dir=output_dir,
        sample_size=sample_size,
        num_epochs=num_epochs,
        batch_size=max(1, optimal_bs // 2),
    )

    # -----------------------------------------------------------------
    # 步骤 4：生成审计报告
    # -----------------------------------------------------------------
    print(f"\n>>> [步骤 4/4] 正在导出防篡改审计报告...")

    # 报告摘要记录本次极速清洗的关键参数，便于复现实验。
    log_summary = (
        f"执行 Aegis-LoRA 极速清洗。"
        f"应用预计算签名，拦截阈值 Tau={tau}，"
        f"Attention Top-K={attention_top_k}。"
    )

    # 所有报告统一写入 .cache/reports。
    reports_dir = os.path.join(PROJECT_ROOT, ".cache", "reports")
    os.makedirs(reports_dir, exist_ok=True)

    # 根据 LoRA 名称生成报告文件名，避免不同 LoRA 的报告互相覆盖。
    lora_name = os.path.basename(os.path.normpath(lora_path))
    clean_report_name = f"{lora_name}_FastCleanse_Audit_Report"

    # 导出极速清洗审计报告。
    # 报告包含：输入 LoRA、输出 LoRA、清洗日志、参数抑制统计、手术前后 norm 等信息。
    report_path = export_cleanse_report(
        report_type="fast",
        base_model_path=base_model_path,
        lora_path=lora_path,
        cleansed_path=output_dir,
        log_text=log_summary,
        n_variants=6,
        tau=tau,
        norms_before=surgery_report.get("before_surgery_norms", {}),
        norms_after=surgery_report.get("after_surgery_norms", {}),
        suppressed_count=suppressed_count,
        suppressed_dict=surgery_report.get("suppressed_counts", {}),
        target_attention_heads=surgery_report.get("target_attention_heads", []),
        output_dir=reports_dir,
        custom_name=clean_report_name,
    )

    print("\n" + "=" * 60)
    print(f">>> [完成] 极速查杀流水线执行完毕！")
    print(f"      -> 免疫模型: {output_dir}")
    print(f"      -> 抑制参数: {suppressed_count}")
    print("=" * 60)

    # 释放模型对象和显存，便于后续继续执行其他清洗任务。
    del cleansed_model
    del model
    del base_model
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return report_path, suppressed_count, output_dir
