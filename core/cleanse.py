import torch
import itertools

# =====================================================================
# 模块 1: 泛化版实时手术刀
# =====================================================================
def bd_vax_surgeon(peft_model, suppression_ratio=0.35):
    """
    泛化版神经元抑制器
    不再硬编码限定 MLP 层，而是全图扫描任何存在 LoRA 的微调矩阵 (包括 q_proj, v_proj 等)。
    """
    suppressed_channels_total = 0
    surgery_report = {
        "before_surgery_max_norms": {},
        "after_surgery_max_norms": {}
    }
    
    for name, module in peft_model.named_modules():
        if hasattr(module, 'lora_B') and hasattr(module, 'lora_A'):
            # 兼容不同版本 PEFT 的字典提取
            weight_B = module.lora_B['default'].weight if isinstance(module.lora_B, torch.nn.ModuleDict) else module.lora_B.weight
            weight_A = module.lora_A['default'].weight if isinstance(module.lora_A, torch.nn.ModuleDict) else module.lora_A.weight
            
            layer_short = name.split('.')[-1]
            if len(name.split('.')) > 2:
                layer_short = f"L{name.split('.')[2]}_{layer_short}"

            is_output_proj = any(x in name for x in ['q_proj', 'k_proj', 'v_proj', 'up_proj', 'gate_proj'])
            is_input_proj = any(x in name for x in ['o_proj', 'down_proj'])
            
            if is_output_proj:
                # 针对输出投影：抑制 lora_B 的行
                norms = weight_B.norm(p=2, dim=1)
                surgery_report["before_surgery_max_norms"][f"{layer_short}_out"] = norms.max().item()
                
                threshold = torch.quantile(norms, 1.0 - suppression_ratio)
                mask = norms > threshold
                weight_B.data[mask, :] = 0.0
                
                suppressed_channels_total += mask.sum().item()
                surgery_report["after_surgery_max_norms"][f"{layer_short}_out"] = weight_B.norm(p=2, dim=1).max().item()
                
            elif is_input_proj:
                # 针对输入投影：抑制 lora_A 的列
                norms = weight_A.norm(p=2, dim=0)
                surgery_report["before_surgery_max_norms"][f"{layer_short}_in"] = norms.max().item()
                
                threshold = torch.quantile(norms, 1.0 - suppression_ratio)
                mask = norms > threshold
                weight_A.data[:, mask] = 0.0
                
                suppressed_channels_total += mask.sum().item()
                surgery_report["after_surgery_max_norms"][f"{layer_short}_in"] = weight_A.norm(p=2, dim=0).max().item()

    surgery_report["total_suppressed"] = suppressed_channels_total
    print(f"[Surgery Log] 泛化扫描完成。共定位并阻断了 {suppressed_channels_total} 个疑似后门通道。")
    return peft_model, surgery_report


# =====================================================================
# 模块 2: BD-Vax 免疫签名提取器
# =====================================================================
def extract_bd_vax_signature(delta_matrices, lambda_weight=0.01):
    """
    参数:
        delta_matrices: 一个列表，包含 N 个张量 (对应 N 个变体的 \Delta_i)
    返回:
        scores: 每个通道的综合得分 s_j
    """
    N = len(delta_matrices)
    if N < 2:
        raise ValueError("签名提取至少需要 2 个变体。")
        
    num_channels = delta_matrices[0].shape[0] # 假设形状为 (num_channels, feature_dim)
    scores = torch.zeros(num_channels, device=delta_matrices[0].device)
    
    # 逐通道计算得分
    for j in range(num_channels):
        # 1. 毒性强度项 (Poison strength)
        norm_sum = sum(torch.norm(delta[j], p=2) for delta in delta_matrices)
        strength_term = norm_sum / N
        
        # 2. 跨变体对齐项 (Cross-variant alignment)
        alignment_sum = 0.0
        # 组合所有 (i, l) 对
        for i, l in itertools.combinations(range(N), 2):
            vec_i = delta_matrices[i][j]
            vec_l = delta_matrices[l][j]
            
            # 计算余弦相似度
            cos_sim = torch.nn.functional.cosine_similarity(vec_i.unsqueeze(0), vec_l.unsqueeze(0)).item()
            # 论文逻辑：过滤掉负相关的噪声，仅保留正对齐的特征
            alignment_sum += max(0.0, cos_sim)
            
        alignment_term = alignment_sum * (2.0 / (N * (N - 1)))
        
        # 3. 综合得分 (Eq 2)
        scores[j] = strength_term + lambda_weight * alignment_term
        
    return scores