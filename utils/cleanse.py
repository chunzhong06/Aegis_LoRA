# Aegis-LoRA: 后门免疫核心模块
# 本模块定义了后门签名提取器和神经元手术刀函数，支持在完全隔离的环境中对多个毒化变体进行交叉对比分析。
# 提取高危特征并执行针对性的参数阻断，实现对后门行为的深度免疫重构。
import torch
import torch.nn as nn
import itertools


# =====================================================================
# 矩阵化签名提取器
# =====================================================================
def extract_bd_vax_signature_strict(delta_dicts, model_config, lambda_weight=0.01):
    """通过对比多个变体的参数差分，分别针对 MLP 的物理通道和 Attention Heads 进行独立特征提取后门签名"""
    N = len(delta_dicts)
    if N < 2:
        raise ValueError("      [错误] 提取交叉签名至少需要 2 个有效变体。")

    print(f"\n      [-] [签名提取] 启动多模态物理特征提取 ...")
    keys = list(delta_dicts[0].keys())

    # 1. 解析存在的网络层索引
    layer_indices = set()
    for key in keys:
        parts = key.split(".")
        if "layers" in parts:
            layer_indices.add(parts[parts.index("layers") + 1])

    unified_layer_scores = {}
    attn_head_scores = {}

    # 自动识别架构的 Attention 头数
    num_heads = model_config.num_attention_heads

    for layer_idx in layer_indices:
        # 提取 MLP SwiGLU 物理通道特征
        gate_key = next(
            (
                k
                for k in keys
                if f"layers.{layer_idx}.mlp.gate_proj" in k
                and ("lora_B" in k or ("lora_" not in k and k.endswith(".weight")))
            ),
            None,
        )
        up_key = next(
            (
                k
                for k in keys
                if f"layers.{layer_idx}.mlp.up_proj" in k
                and ("lora_B" in k or ("lora_" not in k and k.endswith(".weight")))
            ),
            None,
        )
        down_key = next(
            (
                k
                for k in keys
                if f"layers.{layer_idx}.mlp.down_proj" in k
                and ("lora_A" in k or ("lora_" not in k and k.endswith(".weight")))
            ),
            None,
        )

        # 仅当当前层同时存在门控、上投影和下投影的差分特征时，才计算统一得分，否则跳过该层的物理通道特征提取
        if gate_key and up_key and down_key:
            gate_tensors = torch.stack([d[gate_key].float() for d in delta_dicts])
            up_tensors = torch.stack([d[up_key].float() for d in delta_dicts])
            down_tensors = torch.stack(
                [d[down_key].float() for d in delta_dicts]
            ).transpose(1, 2)

            # 在特征维度拼接门控、上投影与下投影，合并为完整神经元描述向量
            stacked_deltas = torch.cat([gate_tensors, up_tensors, down_tensors], dim=2)
            N_vars, C, F = stacked_deltas.shape

            # 计算得分公式：s_j = 毒性强度(L2范数均值) + lambda * 跨变体对齐度(余弦相似度)
            norms = torch.norm(stacked_deltas, p=2, dim=2)
            m_j = norms.mean(dim=0)

            # 计算跨变体对齐度
            normalized_deltas = stacked_deltas / (norms.unsqueeze(2) + 1e-8)
            a_j_sum = torch.zeros(C, device=stacked_deltas.device)
            for i, l in itertools.combinations(range(N_vars), 2):
                cos_sim = (normalized_deltas[i] * normalized_deltas[l]).sum(dim=1)
                a_j_sum += torch.relu(cos_sim)
            a_j = a_j_sum * (2.0 / (N_vars * (N_vars - 1)))

            # 计算最终统一得分
            unified_layer_scores[layer_idx] = m_j + lambda_weight * a_j

        # 提取 Attention Head 独立特征
        q_key = next(
            (
                k
                for k in keys
                if f"layers.{layer_idx}.self_attn.q_proj" in k
                and ("lora_B" in k or ("lora_" not in k and k.endswith(".weight")))
            ),
            None,
        )
        if q_key:
            q_tensors = torch.stack([d[q_key].float() for d in delta_dicts])
            N_vars, out_feat, in_feat = q_tensors.shape

            # 按 Attention Head 数量对特征进行维度拆分
            reshaped_q = q_tensors.view(N_vars, num_heads, -1)

            # 计算每个 Attention Head 的得分，公式同上但在头内维度计算
            for h in range(num_heads):
                head_deltas = reshaped_q[:, h, :]
                h_norms = torch.norm(head_deltas, p=2, dim=1)
                h_mj = h_norms.mean().item()

                # 计算跨变体对齐度
                h_norm_d = head_deltas / (h_norms.unsqueeze(1) + 1e-8)
                h_aj_sum = 0
                for i, l in itertools.combinations(range(N_vars), 2):
                    sim = (h_norm_d[i] * h_norm_d[l]).sum().item()
                    h_aj_sum += max(0, sim)
                h_aj = h_aj_sum * (2.0 / (N_vars * (N_vars - 1)))

                attn_head_scores[(layer_idx, h)] = h_mj + lambda_weight * h_aj

    print(
        f"      [-] [签名提取] 提取完成！(MLP 层: {len(unified_layer_scores)}, Attention Heads: {len(attn_head_scores)})"
    )
    return unified_layer_scores, attn_head_scores


# =====================================================================
# 神经元手术刀
# =====================================================================
def bd_vax_surgeon_strict(model, extracted_signatures, tau=0.40, attn_heads_to_cut=8):
    """执行 MLP 全局绝对阈值切除与 Attention 局部高危头摘除。支持自动识别当前模型的注意力架构拓扑。"""
    unified_layer_scores, attn_head_scores = extracted_signatures
    suppressed_channels_total = 0
    surgery_report = {
        "before_surgery_norms": {},
        "after_surgery_norms": {},
        "suppressed_counts": {},
    }

    # 检测是否为挂载了 LoRA 适配器的 PEFT 模型
    is_lora_mounted = hasattr(model, "peft_config")

    # 动态解析模型架构拓扑 (兼容原生模型与 PEFT 包装模型)
    model_config = model.config if hasattr(model, "config") else model.base_model.config
    num_heads = model_config.num_attention_heads
    num_kv_heads = getattr(model_config, "num_key_value_heads", num_heads)
    head_dim = getattr(model_config, "head_dim", model_config.hidden_size // num_heads)

    print(
        f"      [-] [神经元手术] 启动联合阻断 (MLP 阈值 = {tau*100}%, Attention 切除数 = {attn_heads_to_cut} Heads)"
    )

    # 1. 锁定全局得分最高的 Attention Heads
    target_heads = set()
    if attn_head_scores and attn_heads_to_cut > 0:
        sorted_heads = sorted(
            attn_head_scores.items(), key=lambda x: x[1], reverse=True
        )
        target_heads = set([k for k, v in sorted_heads[:attn_heads_to_cut]])

    # 2. 计算 MLP 的全局统一物理切除阈值
    all_scores = [scores.flatten() for scores in unified_layer_scores.values()]
    global_threshold = (
        torch.quantile(torch.cat(all_scores), 1.0 - tau) if all_scores else float("inf")
    )

    # 3. 遍历模型计算图执行物理阻断
    for name, module in model.named_modules():
        parts = name.split(".")
        if "layers" in parts and parts.index("layers") < len(parts) - 1:
            layer_idx = parts[parts.index("layers") + 1]
            layer_short = parts[-1]
            layer_short_print = f"L{layer_idx}_{layer_short}"
        else:
            continue

        is_mlp = any(x in layer_short for x in ["gate_proj", "up_proj", "down_proj"])
        is_attn = any(
            x in layer_short for x in ["q_proj", "k_proj", "v_proj", "o_proj"]
        )
        is_input_proj = "down_proj" in layer_short or "o_proj" in layer_short

        if not (is_mlp or is_attn):
            continue

        target_device = (
            module.lora_B.default.weight.device
            if is_lora_mounted
            else module.weight.device
        )

        # 动态生成阻断掩码
        if is_mlp and layer_idx in unified_layer_scores:
            scores_tensor = unified_layer_scores[layer_idx].to(target_device)
            mask = scores_tensor >= global_threshold
        elif is_attn:
            weight_shape = (
                module.lora_B.default.weight.shape[0]
                if not is_input_proj
                else module.lora_A.default.weight.shape[1]
            )
            mask = torch.zeros(weight_shape, dtype=torch.bool, device=target_device)
            for h in range(num_heads):
                if (layer_idx, h) in target_heads:
                    # GQA 架构换算：多个 Query Head 对应同一个 KV Head
                    if "k_proj" in layer_short or "v_proj" in layer_short:
                        kv_idx = h // (num_heads // num_kv_heads)
                        mask[kv_idx * head_dim : (kv_idx + 1) * head_dim] = True
                    else:
                        mask[h * head_dim : (h + 1) * head_dim] = True
        else:
            continue

        if not mask.any():
            continue

        # 执行参数清零/重新初始化与审计报告记录
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

            if not is_input_proj:
                # 记录阻断前的平均范数 (输出通道)
                surgery_report["before_surgery_norms"][f"{layer_short_print}_out"] = (
                    weight_B.norm(p=2, dim=1).mean().item()
                )
                # 执行物理阻断
                weight_B.data[mask, :] = 0.0
                # 记录阻断后的平均范数
                surgery_report["after_surgery_norms"][f"{layer_short_print}_out"] = (
                    weight_B.norm(p=2, dim=1).mean().item()
                )
                # 记录切除的神经元连接数
                surgery_report["suppressed_counts"][
                    f"{layer_short_print}_out"
                ] = mask.sum().item()

            else:
                # 记录阻断前的平均范数 (输入通道)
                surgery_report["before_surgery_norms"][f"{layer_short_print}_in"] = (
                    weight_A.norm(p=2, dim=0).mean().item()
                )
                # 执行物理阻断
                weight_A.data[:, mask] = 0.0
                # 记录阻断后的平均范数
                surgery_report["after_surgery_norms"][f"{layer_short_print}_in"] = (
                    weight_A.norm(p=2, dim=0).mean().item()
                )
                # 记录切除的神经元连接数
                surgery_report["suppressed_counts"][
                    f"{layer_short_print}_in"
                ] = mask.sum().item()

            suppressed_channels_total += mask.sum().item()

        elif not is_lora_mounted and hasattr(module, "weight"):
            weight = module.weight
            if mask.any():
                with torch.no_grad():
                    if not is_input_proj:
                        surgery_report["before_surgery_norms"][
                            f"{layer_short_print}_out"
                        ] = (weight.norm(p=2, dim=1).mean().item())
                        temp = torch.empty_like(weight.data[mask, :])
                        nn.init.xavier_uniform_(temp)
                        weight.data[mask, :] = temp
                        surgery_report["after_surgery_norms"][
                            f"{layer_short_print}_out"
                        ] = (weight.norm(p=2, dim=1).mean().item())
                        surgery_report["suppressed_counts"][
                            f"{layer_short_print}_out"
                        ] = mask.sum().item()
                    else:
                        surgery_report["before_surgery_norms"][
                            f"{layer_short_print}_in"
                        ] = (weight.norm(p=2, dim=0).mean().item())
                        temp = torch.empty_like(weight.data[:, mask])
                        nn.init.xavier_uniform_(temp)
                        weight.data[:, mask] = temp
                        surgery_report["after_surgery_norms"][
                            f"{layer_short_print}_in"
                        ] = (weight.norm(p=2, dim=0).mean().item())
                        surgery_report["suppressed_counts"][
                            f"{layer_short_print}_in"
                        ] = mask.sum().item()

            suppressed_channels_total += mask.sum().item()

        elif not is_lora_mounted and hasattr(module, "weight"):
            weight = module.weight
            if mask.any():
                with torch.no_grad():
                    if not is_input_proj:
                        temp = torch.empty_like(weight.data[mask, :])
                        nn.init.xavier_uniform_(temp)
                        weight.data[mask, :] = temp
                    else:
                        temp = torch.empty_like(weight.data[:, mask])
                        nn.init.xavier_uniform_(temp)
                        weight.data[:, mask] = temp

            surgery_report["suppressed_counts"][layer_short_print] = mask.sum().item()
            suppressed_channels_total += mask.sum().item()

    surgery_report["total_suppressed"] = suppressed_channels_total
    print(
        f"      [-] [神经元手术] 阻断完成！共切除 {suppressed_channels_total} 个高危连接。"
    )
    return model, surgery_report
