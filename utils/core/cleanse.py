# Aegis-LoRA: 签名提取与神经元手术模块
# 从多组 poisoned-clean delta 中提取结构化签名，并定向清零对应的 LoRA 参数。
import itertools
import math

import torch


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
    """聚合多组 poisoned-clean LoRA 增量，生成结构化 MLP 与 Attention 签名。"""

    # 跨变体一致性至少需要一对样本，单个变体无法计算方向共识。
    if len(delta_dicts) < 2:
        raise ValueError("      [错误] 提取签名至少需要 2 个有效变体。")

    # -----------------------------------------------------------------
    # 1. 准备评分环境与模型结构
    # -----------------------------------------------------------------
    # auto 优先使用 CUDA，不可用时回退到 CPU。
    if score_device == "auto" or score_device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(score_device)

    print(f"\n      [-] 启动签名提取 ... 评分设备: {device}")

    # Attention 通道需要按模型 head 结构聚合；GQA/MQA 单独读取 KV head 数。
    num_heads = model_config.num_attention_heads
    num_kv_heads = getattr(model_config, "num_key_value_heads", num_heads)
    hidden_size = model_config.hidden_size
    head_dim = getattr(model_config, "head_dim", hidden_size // num_heads)

    # 以首个 delta 的模块集合为遍历基准；临时分数将在后续按层合并。
    module_keys = sorted(delta_dicts[0].keys())
    mlp_projection_scores = {}
    attn_projection_scores = {}

    # -----------------------------------------------------------------
    # 2. 计算各 projection 的通道分数
    # -----------------------------------------------------------------
    # 先保留 projection 级结果，再按网络结构合并为神经元或完整 head。
    for module_key in module_keys:
        parts = module_key.split(".")

        # 从 PEFT 参数长路径中定位 Transformer 层号，忽略层外参数。
        if "layers" not in parts:
            continue

        layer_pos = parts.index("layers")
        if layer_pos + 1 >= len(parts):
            continue

        layer_idx = parts[layer_pos + 1]

        # 识别当前模块所属 projection，仅保留论文清洗涉及的 MLP 与 Attention 模块。
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

        # gate/up/q/k/v 评分输出通道，对应 LoRA B 的行；
        # down/o 评分输入通道，对应 LoRA A 的列。
        if proj in ("gate_proj", "up_proj", "q_proj", "k_proj", "v_proj"):
            mode = "out"
        else:
            mode = "in"

        # 根据评分方向确定待遍历通道数，不构造完整的有效权重矩阵。
        first_pack = delta_dicts[0][module_key]
        A_ref = first_pack["A_bd"]
        B_ref = first_pack["B_bd"]
        channel_count = B_ref.shape[0] if mode == "out" else A_ref.shape[1]

        score_pieces = []

        # 分块计算有效增量，避免一次性构造完整 B @ A 带来的显存峰值。
        for start in range(0, channel_count, score_block_size):
            end = min(start + score_block_size, channel_count)
            blocks = []

            for delta_dict in delta_dicts:
                pack = delta_dict[module_key]

                # 评分统一使用 float32，避免低精度矩阵运算放大范数和余弦误差。
                A_bd = pack["A_bd"].to(device=device, dtype=torch.float32)
                B_bd = pack["B_bd"].to(device=device, dtype=torch.float32)
                A_cl = pack["A_clean"].to(device=device, dtype=torch.float32)
                B_cl = pack["B_clean"].to(device=device, dtype=torch.float32)

                if mode == "out":
                    # 只取 B 的目标行，直接得到当前输出通道的 poisoned-clean 增量。
                    block = B_bd[start:end, :] @ A_bd - B_cl[start:end, :] @ A_cl
                else:
                    # 只取 A 的目标列；转置后与输出通道统一为 [通道数, 特征维度]。
                    block = (
                        B_bd @ A_bd[:, start:end] - B_cl @ A_cl[:, start:end]
                    ).transpose(0, 1)

                blocks.append(block)

            # 将同一批通道在所有变体中的有效增量堆叠，供强度和一致性评分共用。
            stacked = torch.stack(blocks, dim=0)

            # strength 取各变体 L2 范数均值，表示该通道的平均扰动幅度。
            norms = stacked.norm(p=2, dim=2)
            strength = norms.mean(dim=0)

            # alignment 统计变体两两正余弦相似度，只奖励重复出现的同向扰动。
            normalized = stacked / (norms.unsqueeze(2) + 1e-8)
            align = torch.zeros(end - start, device=device)

            for i, j in itertools.combinations(range(len(delta_dicts)), 2):
                cos = (normalized[i] * normalized[j]).sum(dim=1)
                align += torch.relu(cos)

            align *= 2.0 / (len(delta_dicts) * (len(delta_dicts) - 1))

            # 通道风险以扰动强度为主，跨变体一致性作为轻量修正。
            score_pieces.append((strength + lambda_weight * align).cpu())

            # 当前分块分数已转回 CPU，及时释放中间张量以稳定显存占用。
            del blocks, stacked, norms, normalized, align
            if device.type == "cuda":
                torch.cuda.empty_cache()

        # 拼接所有分块，恢复当前 projection 的完整通道顺序。
        channel_scores = torch.cat(score_pieces, dim=0)

        # MLP 保留通道级分数，稍后按 gate/up/down 的共同中间维度合并。
        if proj in ("gate_proj", "up_proj", "down_proj"):
            mlp_projection_scores[(layer_idx, proj)] = channel_scores

        # q/k/v 按各自物理 head 数聚合；GQA/MQA 的 k/v head 数可能更少。
        elif proj in ("q_proj", "k_proj", "v_proj"):
            n_heads = num_kv_heads if proj in ("k_proj", "v_proj") else num_heads
            usable = n_heads * head_dim

            if channel_scores.numel() >= usable:
                attn_projection_scores[(layer_idx, proj)] = (
                    channel_scores[:usable].view(n_heads, head_dim).mean(dim=1)
                )

        # o_proj 的输入通道按 query head 布局聚合，后续与 q/k/v 对齐。
        elif proj == "o_proj":
            usable = num_heads * head_dim

            if channel_scores.numel() >= usable:
                attn_projection_scores[(layer_idx, proj)] = (
                    channel_scores[:usable].view(num_heads, head_dim).mean(dim=1)
                )

        # 当前 projection 已写入 CPU 分数表，释放通道级临时结果。
        del channel_scores, score_pieces
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # -----------------------------------------------------------------
    # 3. 合并完整 MLP 神经元分数
    # -----------------------------------------------------------------
    # 同一中间通道在 gate/up 中是输出通道、在 down 中是输入通道；
    # 三者取均值后只保留一组神经元分数与统一索引。
    mlp_scores = {}

    # 层号按数值顺序处理，非标准层名保留稳定的字符串顺序。
    mlp_layers = sorted(
        {layer_idx for layer_idx, _ in mlp_projection_scores},
        key=lambda value: (0, int(value)) if str(value).isdigit() else (1, str(value)),
    )

    for layer_idx in mlp_layers:
        # 只有 gate/up/down 三项齐全时，才能表示一个完整 MLP 神经元。
        required = ("gate_proj", "up_proj", "down_proj")
        missing = [
            proj for proj in required if (layer_idx, proj) not in mlp_projection_scores
        ]
        if missing:
            print(
                f"      [警告] 跳过第 {layer_idx} 层 MLP 签名，缺少 projection: {missing}"
            )
            continue

        # 三个 projection 必须共享中间维度，否则统一索引没有结构意义。
        projection_scores = [
            mlp_projection_scores[(layer_idx, proj)] for proj in required
        ]
        if len({tuple(scores.shape) for scores in projection_scores}) != 1:
            raise RuntimeError(
                f"      [错误] 第 {layer_idx} 层 gate/up/down 通道数不一致，无法合并 MLP 神经元分数。"
            )

        # 每个位置对应一个完整 MLP 神经元，手术阶段只需选择一次索引。
        mlp_scores[layer_idx] = torch.stack(projection_scores, dim=0).mean(dim=0)

    # -----------------------------------------------------------------
    # 4. 合并完整 Attention head 分数
    # -----------------------------------------------------------------
    # 以 query head 为统一索引，将 GQA/MQA 的共享 KV head 映射到对应 query head。
    attn_scores = {}

    # 收集出现过 Attention projection 的层，并建立 query head 到 KV head 的映射。
    attn_layers = sorted(
        {layer_idx for layer_idx, _ in attn_projection_scores},
        key=lambda value: (0, int(value)) if str(value).isdigit() else (1, str(value)),
    )
    kv_head_map = torch.div(
        torch.arange(num_heads) * num_kv_heads,
        num_heads,
        rounding_mode="floor",
    ).long()

    for layer_idx in attn_layers:
        # 只有 q/k/v/o 四项齐全时，才能形成可联动清零的完整 Attention head。
        required = ("q_proj", "k_proj", "v_proj", "o_proj")
        missing = [
            proj for proj in required if (layer_idx, proj) not in attn_projection_scores
        ]
        if missing:
            print(
                f"      [警告] 跳过第 {layer_idx} 层 Attention 签名，缺少 projection: {missing}"
            )
            continue

        # 取出同层四类 head 分数，先验证其数量与模型配置一致。
        q_scores = attn_projection_scores[(layer_idx, "q_proj")]
        k_scores = attn_projection_scores[(layer_idx, "k_proj")]
        v_scores = attn_projection_scores[(layer_idx, "v_proj")]
        o_scores = attn_projection_scores[(layer_idx, "o_proj")]

        if q_scores.numel() != num_heads or o_scores.numel() != num_heads:
            raise RuntimeError(
                f"      [错误] 第 {layer_idx} 层 q/o head 数与模型配置不一致。"
            )
        if k_scores.numel() != num_kv_heads or v_scores.numel() != num_kv_heads:
            raise RuntimeError(
                f"      [错误] 第 {layer_idx} 层 k/v head 数与模型配置不一致。"
            )

        # K/V 分数映射到 query head 后四项取均值，生成统一的完整 head 分数。
        attn_scores[layer_idx] = torch.stack(
            [
                q_scores,
                k_scores.index_select(0, kv_head_map),
                v_scores.index_select(0, kv_head_map),
                o_scores,
            ],
            dim=0,
        ).mean(dim=0)

    # projection 级中间分数不再参与手术，释放引用后仅返回结构化签名。
    del mlp_projection_scores, attn_projection_scores

    print(
        f"      [-] 提取完成！(MLP layers: {len(mlp_scores)}, Attention layers: {len(attn_scores)})"
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
    """按结构化签名清零 LoRA 神经元与 Attention head，并返回手术报告。"""

    # tau 表示每层 MLP 的清洗比例，必须是有效比例值。
    if not 0.0 < tau <= 1.0:
        raise ValueError("      [错误] tau 必须位于 (0, 1] 区间。")

    print(
        f"      [-] [神经元手术] 启动对齐阻断 "
        f"(MLP 每层神经元 top {tau * 100:.1f}%, Attention top-k={attention_top_k})"
    )

    # -----------------------------------------------------------------
    # 1. 校验签名并读取模型结构
    # -----------------------------------------------------------------
    mlp_scores, attn_scores = extracted_signatures

    # 旧签名以 (layer, projection) 为 key，无法保证完整神经元或 head 联动清零。
    if any(
        isinstance(key, tuple)
        for key in itertools.chain(mlp_scores.keys(), attn_scores.keys())
    ):
        raise ValueError(
            "      [错误] 检测到旧版 projection 级签名，请重新运行签名提取或 build_signature_bank.py。"
        )

    # Attention 手术按 head 展开连续通道，因此需要模型的 head 布局。
    config = model.config if hasattr(model, "config") else model.base_model.config
    num_heads = config.num_attention_heads
    num_kv_heads = getattr(config, "num_key_value_heads", num_heads)
    head_dim = getattr(config, "head_dim", config.hidden_size // num_heads)

    def _effective_update_norm(A, B):
        """通过 Gram 矩阵计算 ||B @ A||_F，避免构造完整 LoRA 更新矩阵。"""
        with torch.no_grad():
            a32 = A.detach().float()
            b32 = B.detach().float()
            gram_a = a32 @ a32.T
            gram_b = b32.T @ b32
            norm_sq = (gram_a * gram_b).sum()
            return torch.sqrt(torch.clamp(norm_sq, min=0.0)).item()

    # 报告同时记录清零规模、Attention 目标和各模块手术前后的有效更新范数。
    report = {
        "suppressed_counts": {},
        "total_suppressed": 0,
        "target_attention_heads": [],
        "before_surgery_norms": {},
        "after_surgery_norms": {},
    }

    # -----------------------------------------------------------------
    # 2. 选择待清零的 MLP 神经元与 Attention head
    # -----------------------------------------------------------------
    # 每层 MLP 只计算一组 top-tau 索引，后续同时用于 gate/up/down。
    selected_mlp_channels = {}
    for layer_idx, scores in mlp_scores.items():
        scores = torch.as_tensor(scores, dtype=torch.float32)

        # 向上取整并至少选择一个神经元，避免小层在 tau 较小时不执行清洗。
        k = max(1, math.ceil(scores.numel() * tau))
        selected_mlp_channels[str(layer_idx)] = torch.topk(
            scores,
            k=k,
            largest=True,
        ).indices.cpu()

    # Attention 以完整 query head 为单位进行跨层全局 top-k。
    selected_heads = set()

    if attention_top_k and attention_top_k > 0 and attn_scores:
        flat_heads = []

        # 将逐层 head 分数展平，统一执行跨层全局排序。
        for layer_idx, scores in attn_scores.items():
            for head_idx, score in enumerate(scores.tolist()):
                flat_heads.append((float(score), str(layer_idx), head_idx))

        flat_heads.sort(key=lambda x: x[0], reverse=True)

        # 保存全局 top-k 完整 head，四个 projection 将共同使用这些目标。
        for score, layer_idx, head_idx in flat_heads[:attention_top_k]:
            selected_heads.add((layer_idx, head_idx))
            report["target_attention_heads"].append(
                {
                    "layer": layer_idx,
                    "module": "q/k/v/o",
                    "head": head_idx,
                    "score": score,
                }
            )

    # -----------------------------------------------------------------
    # 3. 定位带 LoRA A/B 的目标 projection
    # -----------------------------------------------------------------
    for name, module in model.named_modules():
        parts = name.split(".")

        # 从模块路径读取层号和 projection 名称，跳过 Transformer 层外模块。
        if "layers" not in parts:
            continue

        layer_pos = parts.index("layers")
        if layer_pos + 1 >= len(parts):
            continue

        layer_idx = parts[layer_pos + 1]
        proj = parts[-1]

        # 手术范围仅包含组成 MLP 神经元和 Attention head 的七类 projection。
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

        # 跳过未挂载 LoRA adapter 的基座模块。
        if not hasattr(module, "lora_A") or not hasattr(module, "lora_B"):
            continue

        # 兼容 PEFT 的 ModuleDict 与普通层包装，统一取得实际权重。
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
        # 4. 清零完整 MLP 神经元
        # -----------------------------------------------------------------
        if (
            proj in ("gate_proj", "up_proj", "down_proj")
            and layer_idx in selected_mlp_channels
        ):
            # 读取本层统一索引，并记录清洗前 LoRA 有效更新范数。
            selected_idx = selected_mlp_channels[layer_idx]
            report_key = f"L{layer_idx}.{proj}"
            before_norm = _effective_update_norm(A, B)

            if proj in ("gate_proj", "up_proj"):
                # gate/up 的中间神经元位于输出侧，对应清零 LoRA B 的行。
                row_idx = selected_idx[selected_idx < B.shape[0]].to(B.device)

                with torch.no_grad():
                    B.data[row_idx, :] = 0.0

                count = int(row_idx.numel() * B.shape[1])

            elif proj == "down_proj":
                # down 的同一批神经元位于输入侧，对应清零 LoRA A 的列。
                col_idx = selected_idx[selected_idx < A.shape[1]].to(A.device)

                with torch.no_grad():
                    A.data[:, col_idx] = 0.0

                count = int(col_idx.numel() * A.shape[0])

            else:
                count = 0

            # 将实际清零参数量和范数变化写入逐模块报告。
            report["suppressed_counts"][report_key] = count
            report["total_suppressed"] += count
            report["before_surgery_norms"][report_key] = before_norm
            report["after_surgery_norms"][report_key] = _effective_update_norm(A, B)
            continue

        # -----------------------------------------------------------------
        # 5. 清零完整 Attention head
        # -----------------------------------------------------------------
        if proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
            # 提取当前层入选的 query head；未命中全局 top-k 的层无需处理。
            layer_heads = {
                head_idx
                for selected_layer, head_idx in selected_heads
                if selected_layer == layer_idx
            }
            if not layer_heads:
                continue

            # q/o 使用 query head；GQA/MQA 的 k/v 映射到共享 KV head。
            physical_heads = num_kv_heads if proj in ("k_proj", "v_proj") else num_heads
            if proj in ("k_proj", "v_proj"):
                physical_selected = {
                    min(num_kv_heads - 1, (head_idx * num_kv_heads) // num_heads)
                    for head_idx in layer_heads
                }
            else:
                physical_selected = layer_heads

            # 每个 head 占用连续 head_dim 个通道，将结构索引展开为布尔 mask。
            mask = torch.zeros(physical_heads * head_dim, dtype=torch.bool)

            for head_idx in range(physical_heads):
                if head_idx in physical_selected:
                    start = head_idx * head_dim
                    mask[start : start + head_dim] = True

            if not mask.any():
                continue

            # 记录当前 projection 清洗前的有效 LoRA 更新范数。
            report_key = f"L{layer_idx}.{proj}.attn"
            before_norm = _effective_update_norm(A, B)

            if proj in ("q_proj", "k_proj", "v_proj"):
                # q/k/v 的 head 位于输出侧，对应清零 LoRA B 的连续行。
                mask = mask.to(B.device)
                n = min(mask.numel(), B.shape[0])
                row_idx = mask[:n].nonzero(as_tuple=True)[0]

                with torch.no_grad():
                    B.data[row_idx, :] = 0.0

                count = int(row_idx.numel() * B.shape[1])

            else:
                # o_proj 的同一 head 位于输入侧，对应清零 LoRA A 的连续列。
                mask = mask.to(A.device)
                n = min(mask.numel(), A.shape[1])
                col_idx = mask[:n].nonzero(as_tuple=True)[0]

                with torch.no_grad():
                    A.data[:, col_idx] = 0.0

                count = int(col_idx.numel() * A.shape[0])

            # 四类 projection 分别记录清零量，便于报告展示完整 head 的干预结果。
            report["suppressed_counts"][report_key] = count
            report["total_suppressed"] += count
            report["before_surgery_norms"][report_key] = before_norm
            report["after_surgery_norms"][report_key] = _effective_update_norm(A, B)

    print(
        f"      [-] [神经元手术] 阻断完成！共清零 {report['total_suppressed']} 个 LoRA 参数。"
    )

    return model, report
