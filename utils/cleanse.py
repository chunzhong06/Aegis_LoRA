# Aegis-LoRA: 签名提取与神经元手术模块
# 负责从多个 poisoned-clean delta 中提取后门签名，并根据提取出的 MLP / Attention 后门签名，对 LoRA 参数执行定向清零手术。
import itertools
import math
import itertools

import torch
import torch.nn as nn


# ====================================================
# 后门签名提取
# ====================================================
def extract_bd_vax_signature_strict(
    delta_dicts,
    model_config,
    lambda_weight=0.01,
    score_block_size=512,
    score_device="auto",
):
    """从多个 poisoned-clean delta 中提取后门签名"""
    if len(delta_dicts) < 2:
        raise ValueError("      [错误] 提取签名至少需要 2 个有效变体。")
    # -----------------------------------------------------------------
    # 1. 决定评分设备。
    # -----------------------------------------------------------------
    # auto：有 CUDA 就用 GPU，否则回退 CPU。
    if score_device == "auto" or score_device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(score_device)

    print(f"\n      [-] 启动签名提取 ... 评分设备: {device}")

    # -----------------------------------------------------------------
    # 2. 读取模型结构信息。
    # -----------------------------------------------------------------
    # Attention 通道分数最终需要按 head_dim 聚合成 head 级分数。
    num_heads = model_config.num_attention_heads
    num_kv_heads = getattr(model_config, "num_key_value_heads", num_heads)
    hidden_size = model_config.hidden_size
    head_dim = getattr(model_config, "head_dim", hidden_size // num_heads)

    module_keys = sorted(delta_dicts[0].keys())
    mlp_scores = {}
    attn_scores = {}

    # -----------------------------------------------------------------
    # 3. 遍历所有 LoRA module。
    # -----------------------------------------------------------------
    for module_key in module_keys:
        parts = module_key.split(".")

        # 只处理 transformer layers 内的模块。
        if "layers" not in parts:
            continue

        layer_pos = parts.index("layers")
        if layer_pos + 1 >= len(parts):
            continue

        layer_idx = parts[layer_pos + 1]

        # 从 PEFT 长 key 中识别 projection 类型。
        proj = None
        for candidate in (
            "gate_proj",
            "up_proj",
            "down_proj",
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
        ):
            if candidate in parts or module_key.endswith(candidate):
                proj = candidate
                break

        if proj is None:
            continue

        # -----------------------------------------------------------------
        # 4. 决定评分方向。
        # -----------------------------------------------------------------
        # gate/up/q/k/v：看输出通道，对应 B 的行；
        # down/o：看输入通道，对应 A 的列。
        if proj in ("gate_proj", "up_proj", "q_proj", "k_proj", "v_proj"):
            mode = "out"
        else:
            mode = "in"

        first_pack = delta_dicts[0][module_key]
        A_ref = first_pack["A_bd"]
        B_ref = first_pack["B_bd"]
        channel_count = B_ref.shape[0] if mode == "out" else A_ref.shape[1]

        score_pieces = []

        # -----------------------------------------------------------------
        # 5. 分块计算通道分数，避免一次性构造完整 B @ A 大矩阵导致显存峰值过高。
        # -----------------------------------------------------------------
        for start in range(0, channel_count, score_block_size):
            end = min(start + score_block_size, channel_count)
            blocks = []

            for delta_dict in delta_dicts:
                pack = delta_dict[module_key]

                A_bd = pack["A_bd"].to(device=device, dtype=torch.float32)
                B_bd = pack["B_bd"].to(device=device, dtype=torch.float32)
                A_cl = pack["A_clean"].to(device=device, dtype=torch.float32)
                B_cl = pack["B_clean"].to(device=device, dtype=torch.float32)

                if mode == "out":
                    # 输出通道：只取 B 的部分行。
                    # 形状：[channel_block, in_features]
                    block = B_bd[start:end, :] @ A_bd - B_cl[start:end, :] @ A_cl
                else:
                    # 输入通道：只取 A 的部分列。
                    # 原始结果形状为 [out_features, channel_block]，
                    # 转置后统一为 [channel_block, feature_dim]。
                    block = (
                        B_bd @ A_bd[:, start:end] - B_cl @ A_cl[:, start:end]
                    ).transpose(0, 1)

                blocks.append(block)

            # stacked: [n_variants, channel_block, feature_dim]
            stacked = torch.stack(blocks, dim=0)

            # strength：扰动向量范数，衡量该通道改变量有多大。
            norms = stacked.norm(p=2, dim=2)
            strength = norms.mean(dim=0)

            # alignment：不同 variant 的扰动方向是否一致。
            # 只累计正向一致性，避免相反方向互相抵消后误判。
            normalized = stacked / (norms.unsqueeze(2) + 1e-8)
            align = torch.zeros(end - start, device=device)

            for i, j in itertools.combinations(range(len(delta_dicts)), 2):
                cos = (normalized[i] * normalized[j]).sum(dim=1)
                align += torch.relu(cos)

            align *= 2.0 / (len(delta_dicts) * (len(delta_dicts) - 1))

            # 最终通道分数：强度为主，一致性作为轻量加权项。
            score_pieces.append((strength + lambda_weight * align).cpu())

            del blocks, stacked, norms, normalized, align
            if device.type == "cuda":
                torch.cuda.empty_cache()

        channel_scores = torch.cat(score_pieces, dim=0)

        # -----------------------------------------------------------------
        # 6. MLP 分数直接保留通道级 score。
        # -----------------------------------------------------------------
        if proj in ("gate_proj", "up_proj", "down_proj"):
            mlp_scores[(layer_idx, proj)] = channel_scores
            continue

        # -----------------------------------------------------------------
        # 7. Attention 分数从 channel 级聚合到 head 级。
        # -----------------------------------------------------------------
        # q 使用 num_heads；k/v 使用 num_key_value_heads；o 使用 num_heads。
        if proj in ("q_proj", "k_proj", "v_proj"):
            n_heads = num_kv_heads if proj in ("k_proj", "v_proj") else num_heads
            usable = n_heads * head_dim

            if channel_scores.numel() >= usable:
                attn_scores[(layer_idx, proj)] = (
                    channel_scores[:usable].view(n_heads, head_dim).mean(dim=1)
                )

        elif proj == "o_proj":
            usable = num_heads * head_dim

            if channel_scores.numel() >= usable:
                attn_scores[(layer_idx, proj)] = (
                    channel_scores[:usable].view(num_heads, head_dim).mean(dim=1)
                )

        del channel_scores, score_pieces
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print(
        f"      [-] 提取完成！(MLP modules: {len(mlp_scores)}, Attention modules: {len(attn_scores)})"
    )

    return mlp_scores, attn_scores


# ====================================================
# 神经元手术刀
# ====================================================
def bd_vax_surgeon_strict(
    model,
    extracted_signatures,
    tau=0.40,
    attention_top_k=8,
):
    """根据提取出的 MLP / Attention 后门签名，对 LoRA 参数执行定向清零手术"""
    print(
        f"      [-] [神经元手术] 启动对齐阻断 "
        f"(MLP 每模块 top {tau * 100:.1f}%, Attention top-k={attention_top_k})"
    )

    mlp_scores, attn_scores = extracted_signatures

    # -----------------------------------------------------------------
    # 1. 读取模型结构信息。
    # -----------------------------------------------------------------
    # Attention 手术需要知道 head 数量与每个 head 的通道宽度。
    config = model.config if hasattr(model, "config") else model.base_model.config
    num_heads = config.num_attention_heads
    num_kv_heads = getattr(config, "num_key_value_heads", num_heads)
    head_dim = getattr(config, "head_dim", config.hidden_size // num_heads)

    report = {
        "suppressed_counts": {},
        "total_suppressed": 0,
        "target_attention_heads": [],
    }

    # -----------------------------------------------------------------
    # 2. Attention 先做全局 top-k head 选择。
    # -----------------------------------------------------------------
    # attn_scores 是 head 级分数；这里把所有层、所有 q/k/v/o 的 head 拉平，
    # 只选择全局最可疑的 top-k 个 head，避免 attention 过度清零。
    selected_heads = set()

    if attention_top_k and attention_top_k > 0 and attn_scores:
        flat_heads = []

        for tag, scores in attn_scores.items():
            layer_idx, proj = tag
            for head_idx, score in enumerate(scores.tolist()):
                flat_heads.append((float(score), layer_idx, proj, head_idx))

        flat_heads.sort(key=lambda x: x[0], reverse=True)

        for score, layer_idx, proj, head_idx in flat_heads[:attention_top_k]:
            selected_heads.add((layer_idx, proj, head_idx))
            report["target_attention_heads"].append(
                {
                    "layer": layer_idx,
                    "module": proj,
                    "head": head_idx,
                    "score": score,
                }
            )
    # -----------------------------------------------------------------
    # 3. 遍历模型中的所有模块，寻找带 LoRA A/B 的目标 projection。
    # -----------------------------------------------------------------
    for name, module in model.named_modules():
        parts = name.split(".")

        # 只处理 transformer layers 内的模块。
        if "layers" not in parts:
            continue

        layer_pos = parts.index("layers")
        if layer_pos + 1 >= len(parts):
            continue

        layer_idx = parts[layer_pos + 1]
        proj = parts[-1]

        if proj not in (
            "gate_proj",
            "up_proj",
            "down_proj",
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
        ):
            continue

        # 没有 LoRA A/B 的模块不能做 LoRA 手术。
        if not hasattr(module, "lora_A") or not hasattr(module, "lora_B"):
            continue

        tag = (layer_idx, proj)

        # PEFT 里 lora_A / lora_B 可能是 ModuleDict，也可能是普通层。
        # 这里兼容两种包装方式，最终取到真正的 weight。
        lora_A_container = getattr(module, "lora_A")
        lora_B_container = getattr(module, "lora_B")

        if isinstance(lora_A_container, torch.nn.ModuleDict):
            A = lora_A_container["default"].weight
        else:
            A = lora_A_container.weight

        if isinstance(lora_B_container, torch.nn.ModuleDict):
            B = lora_B_container["default"].weight
        else:
            B = lora_B_container.weight

        # -----------------------------------------------------------------
        # 4. MLP 手术：按模块内 top tau 通道清零。
        # -----------------------------------------------------------------
        if tag in mlp_scores:
            scores = mlp_scores[tag]

            if proj in ("gate_proj", "up_proj"):
                # gate/up 的风险通道对应输出通道，因此清 lora_B 的行。
                scores = scores.to(device=B.device, dtype=torch.float32)
                k = max(1, math.ceil(scores.numel() * tau))

                top_idx = torch.topk(scores, k=k, largest=True).indices
                mask = torch.zeros(scores.numel(), dtype=torch.bool, device=B.device)
                mask[top_idx] = True

                n = min(mask.numel(), B.shape[0])
                row_idx = mask[:n].nonzero(as_tuple=True)[0]

                with torch.no_grad():
                    B.data[row_idx, :] = 0.0

                count = int(row_idx.numel() * B.shape[1])

            elif proj == "down_proj":
                # down 的风险通道对应输入通道，因此清 lora_A 的列。
                scores = scores.to(device=A.device, dtype=torch.float32)
                k = max(1, math.ceil(scores.numel() * tau))

                top_idx = torch.topk(scores, k=k, largest=True).indices
                mask = torch.zeros(scores.numel(), dtype=torch.bool, device=A.device)
                mask[top_idx] = True

                n = min(mask.numel(), A.shape[1])
                col_idx = mask[:n].nonzero(as_tuple=True)[0]

                with torch.no_grad():
                    A.data[:, col_idx] = 0.0

                count = int(col_idx.numel() * A.shape[0])

            else:
                count = 0

            report["suppressed_counts"][f"L{layer_idx}.{proj}"] = count
            report["total_suppressed"] += count
            continue

        # -----------------------------------------------------------------
        # 5. Attention 手术：只清全局 top-k 选中的 head。
        # -----------------------------------------------------------------
        if proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
            # q/o 使用完整 attention heads；k/v 在 GQA/MQA 中可能使用更少的 KV heads。
            physical_heads = num_kv_heads if proj in ("k_proj", "v_proj") else num_heads

            # 构造通道级 mask：某个 head 被选中，则该 head 的 head_dim 个通道全部清零。
            mask = torch.zeros(physical_heads * head_dim, dtype=torch.bool)

            for head_idx in range(physical_heads):
                if (layer_idx, proj, head_idx) in selected_heads:
                    start = head_idx * head_dim
                    mask[start : start + head_dim] = True

            if not mask.any():
                continue

            if proj in ("q_proj", "k_proj", "v_proj"):
                # q/k/v 的 head 通道对应输出通道，因此清 lora_B 的行。
                mask = mask.to(B.device)
                n = min(mask.numel(), B.shape[0])
                row_idx = mask[:n].nonzero(as_tuple=True)[0]

                with torch.no_grad():
                    B.data[row_idx, :] = 0.0

                count = int(row_idx.numel() * B.shape[1])

            else:
                # o_proj 的 head 通道对应输入通道，因此清 lora_A 的列。
                mask = mask.to(A.device)
                n = min(mask.numel(), A.shape[1])
                col_idx = mask[:n].nonzero(as_tuple=True)[0]

                with torch.no_grad():
                    A.data[:, col_idx] = 0.0

                count = int(col_idx.numel() * A.shape[0])

            report["suppressed_counts"][f"L{layer_idx}.{proj}.attn"] = count
            report["total_suppressed"] += count

    print(
        f"      [-] [神经元手术] 阻断完成！共清零 {report['total_suppressed']} 个 LoRA 参数。"
    )

    return model, report
