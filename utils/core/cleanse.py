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

    # 跨变体一致性至少需要一对 delta，单个变体无法计算方向共识。
    if len(delta_dicts) < 2:
        raise ValueError("      [错误] 提取签名至少需要 2 个有效变体。")

    # -----------------------------------------------------------------
    # 1. 准备评分环境与模型结构
    # -----------------------------------------------------------------
    # auto 优先使用 CUDA，不可用时回退到 CPU。
    if score_device in ("auto", None):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(score_device)
    print(f"\n      [-] 启动签名提取 ... 评分设备: {device}")

    # Attention 通道需要按 head 结构聚合；GQA/MQA 单独读取 KV head 数。
    num_heads = model_config.num_attention_heads
    num_kv_heads = getattr(model_config, "num_key_value_heads", num_heads)
    head_dim = getattr(
        model_config,
        "head_dim",
        model_config.hidden_size // num_heads,
    )

    # projection_scores 暂存逐层投影分数，最终合并为结构级签名。
    mlp_projection_scores = {}
    attn_projection_scores = {}

    # 输出侧通道对应 LoRA B 的行；其余 down/o 对应 LoRA A 的列。
    mlp_projections = ("gate_proj", "up_proj", "down_proj")
    attn_projections = ("q_proj", "k_proj", "v_proj", "o_proj")
    out_projections = ("gate_proj", "up_proj", "q_proj", "k_proj", "v_proj")

    # -----------------------------------------------------------------
    # 2. 计算各 projection 的通道分数
    # -----------------------------------------------------------------
    for module_key in sorted(delta_dicts[0]):
        parts = module_key.split(".")
        proj = parts[-1]

        # 只处理 Transformer 层内参与 MLP/Attention 手术的七类 projection。
        if "layers" not in parts or proj not in mlp_projections + attn_projections:
            continue

        # layer_idx 用于后续按层合并；score_output 决定沿 B 行还是 A 列评分。
        layer_idx = parts[parts.index("layers") + 1]
        score_output = proj in out_projections
        packs = []

        # LoRA 因子远小于完整 B @ A，提前搬到评分设备可避免每个分块重复传输。
        # 每个 pack 依次保存 poisoned A/B 与 clean A/B。
        for delta_dict in delta_dicts:
            pack = delta_dict[module_key]
            packs.append(
                (
                    pack["A_bd"].to(device=device, dtype=torch.float32),
                    pack["B_bd"].to(device=device, dtype=torch.float32),
                    pack["A_clean"].to(device=device, dtype=torch.float32),
                    pack["B_clean"].to(device=device, dtype=torch.float32),
                )
            )

        # channel_count 是当前 projection 的待评分通道数；score_chunks 保存分块结果。
        channel_count = packs[0][1].shape[0] if score_output else packs[0][0].shape[1]
        score_chunks = []

        # 分块构造有效增量，避免生成完整的高维更新矩阵。
        for start in range(0, channel_count, score_block_size):
            end = min(start + score_block_size, channel_count)
            blocks = []

            for A_bd, B_bd, A_clean, B_clean in packs:
                if score_output:
                    # gate/up/q/k/v 按输出通道计算 poisoned-clean 有效增量。
                    block = (
                        B_bd[start:end] @ A_bd
                        - B_clean[start:end] @ A_clean
                    )
                else:
                    # down/o 按输入通道计算，转置后统一为 [通道, 特征]。
                    block = (
                        B_bd @ A_bd[:, start:end]
                        - B_clean @ A_clean[:, start:end]
                    ).transpose(0, 1)
                blocks.append(block)

            # stacked 形状为 [变体数, 通道数, 特征数]，供强度与一致性共用。
            stacked = torch.stack(blocks)

            # norms 表示各变体的通道扰动强度；normalized 用于比较扰动方向。
            norms = stacked.norm(p=2, dim=2)
            normalized = stacked / (norms.unsqueeze(2) + 1e-8)
            alignment = torch.zeros(end - start, device=device)

            # 只累计正余弦相似度，奖励多个变体中重复出现的同向扰动。
            for i, j in itertools.combinations(range(len(packs)), 2):
                alignment += torch.relu((normalized[i] * normalized[j]).sum(dim=1))

            # alignment 取所有变体对的均值；最终分数以强度为主、一致性为修正。
            alignment *= 2.0 / (len(packs) * (len(packs) - 1))
            score_chunks.append(
                (norms.mean(dim=0) + lambda_weight * alignment).cpu()
            )

        # 恢复当前 projection 的完整通道顺序。
        channel_scores = torch.cat(score_chunks)

        if proj in mlp_projections:
            # MLP 暂时保留通道分数，稍后联动 gate/up/down。
            mlp_projection_scores[(layer_idx, proj)] = channel_scores
        else:
            # Attention 将连续的 head_dim 个通道压缩为一个物理 head 分数。
            physical_heads = (
                num_kv_heads if proj in ("k_proj", "v_proj") else num_heads
            )
            attn_projection_scores[(layer_idx, proj)] = channel_scores.view(
                physical_heads,
                head_dim,
            ).mean(dim=1)

    # -----------------------------------------------------------------
    # 3. 合并完整 MLP 神经元签名
    # -----------------------------------------------------------------
    # gate/up/down 共享同一中间维度，三者均值即完整 MLP 神经元分数。
    mlp_scores = {}

    # mlp_layers 按网络层顺序处理，mlp_scores 每层只保留一组统一索引。
    mlp_layers = sorted(
        {layer_idx for layer_idx, _ in mlp_projection_scores},
        key=int,
    )
    for layer_idx in mlp_layers:
        mlp_scores[layer_idx] = torch.stack(
            [
                mlp_projection_scores[(layer_idx, proj)]
                for proj in mlp_projections
            ]
        ).mean(dim=0)

    # -----------------------------------------------------------------
    # 4. 合并完整 Attention head 签名
    # -----------------------------------------------------------------
    # K/V 的共享 head 映射到 query head 后，与 Q/O 一起生成完整 head 分数。
    attn_scores = {}
    attn_layers = sorted(
        {layer_idx for layer_idx, _ in attn_projection_scores},
        key=int,
    )

    # kv_head_map 将每个 query head 映射到其实际复用的 KV head。
    kv_head_map = torch.div(
        torch.arange(num_heads) * num_kv_heads,
        num_heads,
        rounding_mode="floor",
    ).long()

    for layer_idx in attn_layers:
        # q/o 已是 query head 结构；k/v 通过 kv_head_map 扩展到相同索引空间。
        q_scores = attn_projection_scores[(layer_idx, "q_proj")]
        k_scores = attn_projection_scores[(layer_idx, "k_proj")]
        v_scores = attn_projection_scores[(layer_idx, "v_proj")]
        o_scores = attn_projection_scores[(layer_idx, "o_proj")]
        attn_scores[layer_idx] = torch.stack(
            [
                q_scores,
                k_scores[kv_head_map],
                v_scores[kv_head_map],
                o_scores,
            ]
        ).mean(dim=0)

    print(
        f"      [-] 提取完成！(MLP layers: {len(mlp_scores)}, "
        f"Attention layers: {len(attn_scores)})"
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
    print(
        f"      [-] [神经元手术] 启动对齐阻断 "
        f"(MLP 每层神经元 top {tau * 100:.1f}%, Attention top-k={attention_top_k})"
    )

    # -----------------------------------------------------------------
    # 1. 读取签名与模型结构
    # -----------------------------------------------------------------
    # mlp_scores 按层保存神经元分数，attn_scores 按层保存 query head 分数。
    mlp_scores, attn_scores = extracted_signatures

    # Attention 手术需要 query/KV head 数与单个 head 的通道宽度。
    config = model.config
    num_heads = config.num_attention_heads
    num_kv_heads = getattr(config, "num_key_value_heads", num_heads)
    head_dim = getattr(config, "head_dim", config.hidden_size // num_heads)

    # projection 分组用于区分 MLP 神经元和 Attention head。
    mlp_projections = ("gate_proj", "up_proj", "down_proj")
    attn_projections = ("q_proj", "k_proj", "v_proj", "o_proj")

    # report 记录逐模块清零量、Attention 目标及手术前后的有效更新范数。
    report = {
        "suppressed_counts": {},
        "total_suppressed": 0,
        "target_attention_heads": [],
        "before_surgery_norms": {},
        "after_surgery_norms": {},
    }

    # -----------------------------------------------------------------
    # 2. 选择 MLP 神经元与 Attention head
    # -----------------------------------------------------------------
    # 每层只选一次 MLP 神经元，随后同步作用于 gate/up/down。
    selected_mlp_channels = {}
    for layer_idx, scores in mlp_scores.items():
        scores = torch.as_tensor(scores, dtype=torch.float32)

        # selected_mlp_channels 保存每层分数最高的 top-tau 神经元索引。
        selected_mlp_channels[str(layer_idx)] = torch.topk(
            scores,
            k=max(1, math.ceil(scores.numel() * tau)),
        ).indices.cpu()

    # Attention 在所有层间执行一次全局 top-k，目标以 query head 表示。
    # flat_heads 的元素依次为风险分数、层号和 head 索引。
    flat_heads = []
    if attention_top_k and attn_scores:
        for layer_idx, scores in attn_scores.items():
            flat_heads.extend(
                (float(score), str(layer_idx), head_idx)
                for head_idx, score in enumerate(scores.tolist())
            )
        flat_heads.sort(key=lambda item: item[0], reverse=True)

    # selected_heads 按层保存最终 query head，报告中同时保留其原始风险分数。
    selected_heads = {}
    for score, layer_idx, head_idx in flat_heads[:attention_top_k]:
        selected_heads.setdefault(layer_idx, set()).add(head_idx)
        report["target_attention_heads"].append(
            {
                "layer": layer_idx,
                "module": "q/k/v/o",
                "head": head_idx,
                "score": score,
            }
        )

    # -----------------------------------------------------------------
    # 3. 将结构索引转换为 LoRA 参数索引
    # -----------------------------------------------------------------
    # MLP 与 Attention 最终都转换为目标张量、轴和索引，共用同一段清零逻辑。
    for name, module in model.named_modules():
        parts = name.split(".")
        proj = parts[-1]

        # 只定位 Transformer 层中已挂载 LoRA 的目标 projection。
        if (
            "layers" not in parts
            or proj not in mlp_projections + attn_projections
            or not hasattr(module, "lora_A")
        ):
            continue

        # A/B 分别对应 LoRA 的降维与升维因子，effective update = B @ A。
        layer_idx = parts[parts.index("layers") + 1]
        A = module.lora_A["default"].weight
        B = module.lora_B["default"].weight

        if proj in mlp_projections and layer_idx in selected_mlp_channels:
            # 同一批神经元索引同时用于本层 gate/up/down。
            selected_idx = selected_mlp_channels[layer_idx]
            report_key = f"L{layer_idx}.{proj}"

            if proj in ("gate_proj", "up_proj"):
                # gate/up 的中间神经元位于输出侧，对应 LoRA B 的行。
                target, axis = B, 0
            else:
                # down 的同一神经元位于输入侧，对应 LoRA A 的列。
                target, axis = A, 1

        elif proj in attn_projections and layer_idx in selected_heads:
            # query_heads 是结构索引；K/V 需要映射到实际共享的物理 head。
            query_heads = selected_heads[layer_idx]
            if proj in ("k_proj", "v_proj"):
                physical_heads = {
                    (head_idx * num_kv_heads) // num_heads
                    for head_idx in query_heads
                }
            else:
                physical_heads = query_heads

            # 每个物理 head 展开为连续的 head_dim 个参数通道。
            selected_idx = torch.tensor(
                [
                    channel_idx
                    for head_idx in sorted(physical_heads)
                    for channel_idx in range(
                        head_idx * head_dim,
                        (head_idx + 1) * head_dim,
                    )
                ],
                dtype=torch.long,
            )
            report_key = f"L{layer_idx}.{proj}.attn"

            if proj in ("q_proj", "k_proj", "v_proj"):
                # Q/K/V head 位于输出侧，对应 LoRA B 的连续行。
                target, axis = B, 0
            else:
                # O head 位于输入侧，对应 LoRA A 的连续列。
                target, axis = A, 1
        else:
            continue

        # -----------------------------------------------------------------
        # 4. 清零目标参数并记录手术结果
        # -----------------------------------------------------------------
        with torch.no_grad():
            # 通过 Gram 矩阵计算 ||B @ A||F，避免构造完整更新矩阵。
            a32 = A.detach().float()
            b32 = B.detach().float()
            norm_sq = ((a32 @ a32.T) * (b32.T @ b32)).sum()
            before_norm = torch.sqrt(torch.clamp(norm_sq, min=0.0)).item()

            # axis=0 清零目标行，axis=1 清零目标列；count 记录实际参数量。
            selected_idx = selected_idx.to(target.device)
            if axis == 0:
                target[selected_idx, :] = 0.0
                count = selected_idx.numel() * target.shape[1]
            else:
                target[:, selected_idx] = 0.0
                count = selected_idx.numel() * target.shape[0]

            # 使用相同口径计算清零后的有效更新范数，供报告比较。
            a32 = A.detach().float()
            b32 = B.detach().float()
            norm_sq = ((a32 @ a32.T) * (b32.T @ b32)).sum()
            after_norm = torch.sqrt(torch.clamp(norm_sq, min=0.0)).item()

        # 每个 projection 独立写入报告，报告模块再按层聚合为完整结构。
        report["suppressed_counts"][report_key] = int(count)
        report["total_suppressed"] += int(count)
        report["before_surgery_norms"][report_key] = before_norm
        report["after_surgery_norms"][report_key] = after_norm

    print(
        f"      [-] [神经元手术] 阻断完成！"
        f"共清零 {report['total_suppressed']} 个 LoRA 参数。"
    )
    return model, report
