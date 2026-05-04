import torch
import torch.nn as nn
import itertools
import copy


# =====================================================================
# 矩阵化签名提取器
# =====================================================================
def extract_bd_vax_signature_strict(delta_dicts, lambda_weight=0.01):
    """
    基于 N 个变体的参数差分矩阵，提取后门签名 S。
    使用高维张量化计算,避免低效的 for 循环。

    参数:
        delta_dicts: 包含 N 个字典的列表，每个字典是对应变体的 \Delta_i (CPU Tensors)
    返回:
        global_scores: 包含每个权重矩阵对应通道评分的字典
    """
    N = len(delta_dicts)
    if N < 2:
        raise ValueError("提取交叉签名至少需要 2 个有效变体 (N>=2)。")

    global_scores = {}
    keys = delta_dicts[0].keys()

    print("[Signature Extraction] 正在执行张量化后门签名提取 (Eq. 2)...")

    for key in keys:
        # 将 N 个变体的该层矩阵堆叠: shape (N, out_dim, in_dim)
        stacked_deltas = torch.stack([d[key].float() for d in delta_dicts])

        # 判断是针对输出通道(行)还是输入通道(列)进行阻断
        is_output_proj = (
            any(x in key for x in ["up_proj", "gate_proj", "lora_B"])
            and "lora_A" not in key
            and not any(
                attn in key for attn in ["q_proj", "k_proj", "v_proj", "o_proj"]
            )
        )
        is_input_proj = (
            any(x in key for x in ["down_proj", "lora_A"])
            and "lora_B" not in key
            and not any(
                attn in key for attn in ["q_proj", "k_proj", "v_proj", "o_proj"]
            )
        )

        # 统一形状：将 Channel 转换到 dim=1，Feature 转换到 dim=2
        # 即保证 stacked_deltas 始终为 (N, Channels, Features)
        if is_input_proj:
            # 输入投影关注输入通道 (dim=2)，将其转置到 dim=1
            stacked_deltas = stacked_deltas.transpose(1, 2)
        elif not is_output_proj:
            continue  # 既不是标准输入也不是输出投影，跳过

        N_vars, C, F = stacked_deltas.shape

        # 1. 计算毒性强度项 (Poison Strength): ||\Delta_{i,j}||_2 的平均
        # norms shape: (N, C)
        norms = torch.norm(stacked_deltas, p=2, dim=2)
        strength_term = norms.mean(dim=0)  # shape: (C,)

        # 2. 计算跨变体对齐项 (Cross-variant Alignment)
        # 在特征维度归一化，加上 eps 防止除零，shape: (N, C, F)
        normalized_deltas = stacked_deltas / (norms.unsqueeze(2) + 1e-8)

        alignment_sum = torch.zeros(C, device=stacked_deltas.device)

        # 遍历所有变体对 (i, l)，计算特征维度的点积 (即余弦相似度)
        for i, l in itertools.combinations(range(N), 2):
            cos_sim = (normalized_deltas[i] * normalized_deltas[l]).sum(
                dim=1
            )  # shape: (C,)
            alignment_sum += torch.relu(cos_sim)

        alignment_term = alignment_sum * (2.0 / (N * (N - 1)))

        # 3. 综合得分 S_j
        scores = strength_term + lambda_weight * alignment_term
        global_scores[key] = scores

    print("[Signature Extraction] 提取完毕。")
    return global_scores


# =====================================================================
# 精准神经元手术刀
# =====================================================================
def bd_vax_surgeon_strict(model, global_scores, tau=0.03):
    """
    根据 global_scores 进行手术，切除排名前 tau% 的恶意神经元通道。
    自适应 LoRA 置零模式与全参数 Xavier 重置模式。
    """
    suppressed_channels_total = 0
    surgery_report = {"before_surgery_max_norms": {}, "after_surgery_max_norms": {}}

    is_lora_mounted = hasattr(model, "peft_config")

    # 1. 收集所有得分，计算全局切除阈值
    all_scores = torch.cat([scores.flatten() for scores in global_scores.values()])
    threshold = torch.quantile(all_scores, 1.0 - tau)
    print(f"[Surgeon] 干预比例: {tau*100}% | 全局阻断阈值: {threshold:.4f}")

    # 2. 遍历模型并执行精确切除
    for name, module in model.named_modules():
        layer_short = name.split(".")[-1]
        if len(name.split(".")) > 2:
            layer_short = f"L{name.split('.')[2]}_{layer_short}"

        # 模式 A: 修复 LoRA 适配器
        if is_lora_mounted and hasattr(module, "lora_B") and hasattr(module, "lora_A"):
            weight_B = (
                module.lora_B["default"].weight
                if isinstance(module.lora_B, nn.ModuleDict)
                else module.lora_B.weight
            )
            weight_A = (
                module.lora_A["default"].weight
                if isinstance(module.lora_A, nn.ModuleDict)
                else module.lora_A.weight
            )

            # 定位当前层的评分
            key_B = f"{name}.lora_B.weight"
            key_A = f"{name}.lora_A.weight"

            if key_B in global_scores:
                scores = global_scores[key_B].to(weight_B.device)
                mask = scores > threshold

                surgery_report["before_surgery_max_norms"][f"{layer_short}_out"] = (
                    weight_B.norm(p=2, dim=1).max().item()
                )
                weight_B.data[mask, :] = 0.0  # 置零阻断
                surgery_report["after_surgery_max_norms"][f"{layer_short}_out"] = (
                    weight_B.norm(p=2, dim=1).max().item()
                )
                suppressed_channels_total += mask.sum().item()

            if key_A in global_scores:
                scores = global_scores[key_A].to(weight_A.device)
                mask = scores > threshold

                surgery_report["before_surgery_max_norms"][f"{layer_short}_in"] = (
                    weight_A.norm(p=2, dim=0).max().item()
                )
                weight_A.data[:, mask] = 0.0  # 置零阻断
                surgery_report["after_surgery_max_norms"][f"{layer_short}_in"] = (
                    weight_A.norm(p=2, dim=0).max().item()
                )
                suppressed_channels_total += mask.sum().item()

        # 模式 B: 修复纯基座模型
        elif not is_lora_mounted and isinstance(module, nn.Linear):
            key_weight = f"{name}.weight"
            if key_weight in global_scores:
                scores = global_scores[key_weight].to(module.weight.device)
                mask = scores > threshold

                weight = module.weight

                # 基座干预逻辑区别于 LoRA：
                # 如果是输出通道（如 up_proj），按行切除；如果是输入通道（如 down_proj），按列切除。
                is_input_proj = any(x in name for x in ["o_proj", "down_proj"])

                if not is_input_proj:
                    surgery_report["before_surgery_max_norms"][f"{layer_short}"] = (
                        weight.norm(p=2, dim=1).max().item()
                    )
                    if mask.any():
                        with torch.no_grad():
                            temp_tensor = torch.empty_like(weight.data[mask, :])
                            nn.init.xavier_uniform_(temp_tensor)
                            weight.data[mask, :] = temp_tensor
                    surgery_report["after_surgery_max_norms"][f"{layer_short}"] = (
                        weight.norm(p=2, dim=1).max().item()
                    )
                else:
                    surgery_report["before_surgery_max_norms"][f"{layer_short}"] = (
                        weight.norm(p=2, dim=0).max().item()
                    )
                    if mask.any():
                        with torch.no_grad():
                            temp_tensor = torch.empty_like(weight.data[:, mask])
                            nn.init.xavier_uniform_(temp_tensor)
                            weight.data[:, mask] = temp_tensor
                    surgery_report["after_surgery_max_norms"][f"{layer_short}"] = (
                        weight.norm(p=2, dim=0).max().item()
                    )

                suppressed_channels_total += mask.sum().item()

    surgery_report["total_suppressed"] = suppressed_channels_total
    target_str = (
        "LoRA矩阵 (Zeroing)" if is_lora_mounted else "基座底层参数 (Xavier Re-init)"
    )
    print(
        f"[Surgery Log] 深度免疫完成。精准切除了 {target_str} 中 {suppressed_channels_total} 个后门载体神经元。"
    )
    return model, surgery_report
