# Aegis-LoRA: 后门免疫核心模块
# 本模块定义了后门签名提取器和神经元手术刀函数，支持在完全隔离的环境中对多个毒化变体进行交叉对比分析。
# 提取高危特征并执行针对性的参数阻断，实现对后门行为的深度免疫重构。
import torch
import torch.nn as nn
import itertools
import re
import os
import json


# =====================================================================
# 矩阵化签名提取器
# =====================================================================
def extract_bd_vax_signature_strict(
    delta_dicts, model_config, lora_path, lambda_weight=0.01
):
    """
    自适应多架构后门签名提取器。
    全面聚合Q/K/V/O满秩漂移，完美兼容LLaMA、Qwen及DeepSeek的MLA/MoE架构。
    """
    N = len(delta_dicts)
    if N < 2:
        raise ValueError("      [错误] 提取交叉签名至少需要 2 个有效变体。")

    print(f"\n      [-] [签名提取] 启动物理特征矩阵分析 ...")

    # 1. 自动化查看并解析本 LoRA 文件夹下的核心参数
    config_path = os.path.join(lora_path, "adapter_config.json")
    if os.path.exists(config_path):
        # 本地存在配置文件时，直接读取真实的配置参数
        with open(config_path, "r", encoding="utf-8") as f:
            lora_cfg = json.load(f)
        r = lora_cfg.get("r", 8)
        alpha = lora_cfg.get("lora_alpha", 16)
        print(
            f"      [-] [签名提取] 成功读取本地配置 -> 本 LoRA 物理秩 r = {r}, 缩放因子 alpha = {alpha}"
        )
    else:
        # 本地缺少配置文件时，通过检查 lora_A 张量的形状逆向推导 r，并给与标准的双倍缺省常数
        raw_keys_sample = list(delta_dicts[0]["bd"].keys())
        ka_sample = next(k for k in raw_keys_sample if "lora_A" in k)
        r = delta_dicts[0]["bd"][ka_sample].shape[0]  # lora_A 的第 0 维即物理秩 r
        alpha = r * 2  # 业界通用标准默认比例
        print(
            f"      [警告] 未找到配置文件，通过张量形状逆向推导 -> 物理秩 r = {r}, 缺省常数 alpha = {alpha}"
        )

    # 计算满秩更新矩阵还原缩放系数
    scaling = alpha / r

    # 2. 解析原始 LoRA 参数字典中所有存在的可训练层基础前缀
    raw_keys = list(delta_dicts[0]["bd"].keys())
    base_layers = set()
    for k in raw_keys:
        if "lora_A" in k:
            base_name = k.split(".lora_A")[0]
            base_layers.add(base_name)

    unified_layer_scores = {}
    attn_head_scores = {}

    # 动态获取当前模型的总注意力头数
    num_heads = getattr(model_config, "num_attention_heads", 32)

    # 3. 核心流式循环：按网络组件逐层独立处理
    for base_layer in sorted(base_layers):
        # 剥离 PEFT 包装前缀，以平滑对接后续手术刀模块的 lookup_key
        lookup_key = base_layer.replace("base_model.model.", "")

        # A. MLP / MoE 专家层组件：现场还原与即时评分
        if any(m in base_layer for m in ["gate_proj", "up_proj", "down_proj"]):
            layer_deltas = []

            # 遍历所有变体，即时计算当前网络层的物理满秩更新差分
            for i in range(N):
                bd_sd = delta_dicts[i]["bd"]
                cl_sd = delta_dicts[i]["clean"]

                # 精准匹配低秩低秩 A 矩阵与 B 矩阵的键名
                ka = next(k for k in raw_keys if base_layer in k and "lora_A" in k)
                kb = next(k for k in raw_keys if base_layer in k and "lora_B" in k)

                # 数学对齐：分别在 CPU 浮点空间还原物理更新矩阵块后求差 (B @ A)
                dW_bd = scaling * (bd_sd[kb].float() @ bd_sd[ka].float())
                dW_cl = scaling * (cl_sd[kb].float() @ cl_sd[ka].float())
                layer_deltas.append(dW_bd - dW_cl)

            # 堆叠当前层所有变体的满秩更新张量，维度: (N, out_features, in_features)
            stacked_deltas = torch.stack(layer_deltas)

            # 严格执行行/列向量拓扑解析：down_proj 考核输入列，gate/up_proj 考核输出行
            if "down_proj" in base_layer:
                norms = torch.norm(stacked_deltas, p=2, dim=1)
                normalized = stacked_deltas / (
                    torch.norm(stacked_deltas, p=2, dim=1, keepdim=True) + 1e-8
                )
                a_j_sum = torch.zeros(
                    stacked_deltas.shape[2], device=stacked_deltas.device
                )
                for i, l in itertools.combinations(range(N), 2):
                    a_j_sum += torch.relu((normalized[i] * normalized[l]).sum(dim=0))
            else:
                norms = torch.norm(stacked_deltas, p=2, dim=2)
                normalized = stacked_deltas / (
                    torch.norm(stacked_deltas, p=2, dim=2, keepdim=True) + 1e-8
                )
                a_j_sum = torch.zeros(
                    stacked_deltas.shape[1], device=stacked_deltas.device
                )
                for i, l in itertools.combinations(range(N), 2):
                    a_j_sum += torch.relu((normalized[i] * normalized[l]).sum(dim=1))

            # 计算强度均值与跨变体余弦相似对齐度，保存该层通道得分
            m_j = norms.mean(dim=0)
            a_j = a_j_sum * (2.0 / (N * (N - 1)))
            unified_layer_scores[lookup_key] = m_j + lambda_weight * a_j

            # 当前网络层循环结束，layer_deltas 大矩阵自动脱离引用被物理回收

        # B. Attention 头部组件：四矩阵流式联合特征评分
        elif "q_proj" in base_layer:
            layer_match = re.search(r"layers\.(\d+)", base_layer)
            if not layer_match:
                continue
            layer_idx = layer_match.group(1)

            # 锁定当前注意力块对应的完整全套前向矩阵前缀路径
            prefix = base_layer.split("q_proj")[0]
            proj_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

            # 独立对当前层的每个 Attention Head 执行联合漂移量评估
            for h in range(num_heads):
                head_total_score = 0.0

                for m_name in proj_modules:
                    current_module_name = f"{prefix}{m_name}"
                    m_deltas_list = []

                    # 仅在需要考核当前投影组件时，现场实时计算其满秩矩阵差
                    for i in range(N):
                        bd_sd = delta_dicts[i]["bd"]
                        cl_sd = delta_dicts[i]["clean"]

                        ka = next(
                            k
                            for k in raw_keys
                            if current_module_name in k and "lora_A" in k
                        )
                        kb = next(
                            k
                            for k in raw_keys
                            if current_module_name in k and "lora_B" in k
                        )

                        dW_bd = scaling * (bd_sd[kb].float() @ bd_sd[ka].float())
                        dW_cl = scaling * (cl_sd[kb].float() @ cl_sd[ka].float())
                        m_deltas_list.append(dW_bd - dW_cl)

                    m_deltas = torch.stack(m_deltas_list)
                    out_d, in_d = m_deltas.shape[1], m_deltas.shape[2]

                    # 动态判定当前注意力子块的物理切片步长 (适配 GQA/MLA)
                    is_o = m_name == "o_proj"
                    h_dim = (in_d // num_heads) if is_o else (out_d // num_heads)
                    if h_dim == 0:
                        h_dim = max(1, out_d if not is_o else in_d)

                    start, end = h * h_dim, (h + 1) * h_dim

                    # 提取单头所属的权重分块并展平为特征向量
                    if is_o:
                        head_chunks = m_deltas[:, :, start:end].contiguous().view(N, -1)
                    else:
                        head_chunks = m_deltas[:, start:end, :].contiguous().view(N, -1)

                    # 计算当前特征头在当前投影组件上的毒性得分
                    h_norms = torch.norm(head_chunks, p=2, dim=1)
                    h_mj = h_norms.mean().item()

                    h_norm_d = head_chunks / (h_norms.unsqueeze(1) + 1e-8)
                    h_aj_sum = 0.0
                    for i, l in itertools.combinations(range(N), 2):
                        h_aj_sum += max(0.0, (h_norm_d[i] * h_norm_d[l]).sum().item())
                    h_aj = h_aj_sum * (2.0 / (N * (N - 1)))

                    head_total_score += h_mj + lambda_weight * h_aj

                # 聚合 Q/K/V/O 四项贡献，记录当前层的 Head 综合分
                attn_head_scores[(layer_idx, h)] = head_total_score

    print(
        f"      [-] [签名提取] 提取完成！(MLP/MoE层组件数: {len(unified_layer_scores)}, Attention Heads: {len(attn_head_scores)})"
    )
    return unified_layer_scores, attn_head_scores


# =====================================================================
# 神经元手术刀
# =====================================================================
def bd_vax_surgeon_strict(model, extracted_signatures, tau=0.35, max_suppress_heads=8):
    """
    智能自适应后门手术刀
    通过统计学离群值检测自主决定是否摘除Attention头，并严格执行MLP层内独立裁剪。
    """
    unified_layer_scores, attn_head_scores = extracted_signatures
    suppressed_channels_total = 0
    is_lora_mounted = hasattr(model, "peft_config")

    # 1. 未知模型自适应行为审计引擎
    need_attention_surgery = False
    target_heads = set()

    if attn_head_scores:
        scores_list = list(attn_head_scores.values())
        scores_tensor = torch.tensor(scores_list, dtype=torch.float32)

        mean_score = scores_tensor.mean().item()
        std_score = scores_tensor.std().item() + 1e-8
        max_score = scores_tensor.max().item()

        # 统计学离群值检验：计算最大偏离度 Z-Score
        z_score = (max_score - mean_score) / std_score
        print(f"\n      [-] 全模型Attention头部漂移极大值 Z-Score = {z_score:.2f}")

        # 若存在显著偏离均值的“高危信号放大头”（Z-Score > 2.2），自动强制开启激活Attention手术
        if z_score > 2.2:
            need_attention_surgery = True
            # 对齐作者规范：采用 Top-K 绝对数量摘除法，而非盲目按比例切除
            sorted_heads = sorted(
                attn_head_scores.items(), key=lambda x: x[1], reverse=True
            )
            k_heads = min(max_suppress_heads, max(1, int(len(sorted_heads) * 0.02)))
            target_heads = {sorted_heads[i][0] for i in range(k_heads)}
            print(f"      [-] 检测到注意力放大特征！摘除 {k_heads} 个核心高危头。")
        else:
            print(f"      [-] 注意力拓扑分布平缓，采用纯 [MLP 物理通道切除术]。")

    # 2. 精密神经元物理连接阻断
    for name, module in model.named_modules():
        lookup_key = name.replace("base_model.model.", "")

        # 仅拦截并操作 PEFT 框架下挂载的 LoRA 可训练权重
        if is_lora_mounted and not (
            hasattr(module, "lora_A") and hasattr(module, "lora_B")
        ):
            continue

        weight_B = module.lora_B["default"].weight if is_lora_mounted else module.weight
        weight_A = module.lora_A["default"].weight if is_lora_mounted else module.weight
        target_device = weight_B.device

        # A. MLP/MoE 专家层：严格执行层内局部计算分位数，彻底保护微弱的 Refusal 控制核心
        if lookup_key in unified_layer_scores:
            scores_tensor = unified_layer_scores[lookup_key].to(target_device)
            # 在当前物理矩阵内部独立寻找截断阈值，防止被其他强信号层淹没
            layer_local_threshold = torch.quantile(scores_tensor, 1.0 - tau)
            mask = scores_tensor >= layer_local_threshold

            with torch.no_grad():
                if "down_proj" in lookup_key:
                    weight_A.data[:, mask] = (
                        0.0  # 压制输入通道 (将 lora_A 对应的列抹零)
                    )
                else:
                    weight_B.data[mask, :] = (
                        0.0  # 压制输出通道 (将 lora_B 对应的行抹零)
                    )
            suppressed_channels_total += mask.sum().item()

        # B. Attention 层组件：根据审计决策，精准抹除离群高危头所占用的低秩权重维度
        elif need_attention_surgery and any(
            m in lookup_key for m in ["q_proj", "k_proj", "v_proj", "o_proj"]
        ):
            import re

            layer_match = re.search(r"layers\.(\d+)", lookup_key)
            if not layer_match:
                continue
            layer_idx = layer_match.group(1)

            out_d, in_d = weight_B.shape[0], weight_A.shape[1]
            is_o = "o_proj" in lookup_key

            # 使用反射机制根据模型实际配置推导单头步长，从底层避免由于 MLA/GQA 导致的边界越界
            num_actual_heads = model.config.num_attention_heads
            h_dim = (in_d // num_actual_heads) if is_o else (out_d // num_actual_heads)
            if h_dim == 0:
                h_dim = max(1, out_d if not is_o else in_d)

            mask = torch.zeros(
                out_d if not is_o else in_d, dtype=torch.bool, device=target_device
            )

            # 映射并锁定高危头在参数矩阵中对应的切片索引
            for h in range(num_actual_heads):
                if (layer_idx, h) in target_heads:
                    start, end = h * h_dim, (h + 1) * h_dim
                    if end <= len(mask):
                        mask[start:end] = True

            # 将命中审计异常的高危注意力头对应的 LoRA 矩阵参数块直接物理置零
            if mask.any():
                with torch.no_grad():
                    if is_o:
                        weight_A.data[:, mask] = 0.0  # 抹除 o_proj 的输入通路
                    else:
                        weight_B.data[mask, :] = 0.0  # 抹除 q/k/v_proj 的输出通路
                suppressed_channels_total += mask.sum().item()

    print(
        f"      [-] [神经元手术] 手术完成！共阻断 {suppressed_channels_total} 个高危物理通道。\n"
    )
    return model, {"total_suppressed": suppressed_channels_total}
