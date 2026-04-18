"""
独立训练：单层相邻一致性 + **检测器 trigger embedding → 中毒层特征 → PV 推远**。

**主流程（与 detector 衔接）**

1. **检测**：detector 优化并得到 trigger 的 **连续 embedding**（落盘为 ``[1, L, D]`` 的 soft prompt，与训练时
   ``inputs_embeds`` 前缀形状一致）。
2. **投影到中毒层**：把该 embedding 当作 ``inputs_embeds`` **整模前向**（``output_hidden_states=True``），在
   **与训练一致的中毒层下标** ``hidden_states[t]`` 上对 L 个位置 **均值池化**，得到参考向量 ``v``（``[D]``）。
   下标 ``t`` 由报告 ``poisoned_layer`` 与 ``align_detector_report`` 决定（见 ``train`` 文档）。
3. **训练推远**：每个 batch 在正常文本上前向，在同一层 ``hidden_states[t]`` 上池化得到 ``h``，用
   ``pv_push_away_loss`` / ``pv_push_away_loss_multi`` 把 ``h`` 与一个或多个 ``v`` **推远**（与 ``total_loss`` 符号配合）。

离散 token trigger 用 ``trigger_to_target_layer_feature``；连续 embedding 用
``detector_trigger_embedding_to_layer_pv`` / ``project_input_soft_prompt_to_target_layer_pv``。

**相邻一致性**：``hidden_states[t-1]`` 与 ``hidden_states[t]`` 的余弦损失（最小化 = 拉近）。

``prompt_length`` 与检测里跳过前缀一致。依赖：``get_input_embeddings()``、``inputs_embeds`` 前向、``output_hidden_states``。
"""
from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


def freeze_all_unfreeze_decoder_layers(
    model: nn.Module,
    layer_indices: Sequence[int],
) -> Tuple[int, int]:
    """
    先整网冻结，再仅解冻指定 decoder block（参数名中含 ``layers.{i}.`` 的权重）。

    i 与 ``hidden_states[i]`` 的 transformer 层号一致（0..L-1）。PeftModel 下会匹配层内 LoRA 等。

    Returns:
        (num_trainable_param_tensors, num_frozen_param_tensors)
    """
    indices = sorted({int(i) for i in layer_indices})
    if not indices:
        raise ValueError("layer_indices must be non-empty")
    markers = tuple(f"layers.{i}." for i in indices)

    model.requires_grad_(False)
    n_train, n_frozen = 0, 0
    for name, p in model.named_parameters():
        if any(m in name for m in markers):
            p.requires_grad_(True)
            n_train += 1
        else:
            n_frozen += 1
    return n_train, n_frozen


def parse_trigger_token_ids(trigger_tokens: Union[str, Sequence[int]]) -> List[int]:
    """支持 list 或字符串形式 "[1, 2, 3]"。"""
    if isinstance(trigger_tokens, str):
        s = trigger_tokens.strip()
        return ast.literal_eval(s)
    return [int(x) for x in trigger_tokens]


def build_pv_target_from_trigger_tokens(
    model: nn.Module,
    token_ids: Sequence[int],
    *,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """v = mean_i emb(token_id_i)，shape [D]。"""
    emb = model.get_input_embeddings()
    dev = device or next(emb.parameters()).device
    ids = torch.tensor(list(token_ids), dtype=torch.long, device=dev)
    e = emb(ids)  # [K, D]
    v = e.mean(dim=0).to(dtype)
    return v.detach()


def _load_pv_file_tensor(
    pv_target_path: Union[str, Path],
    *,
    map_location: Union[str, torch.device] = "cpu",
) -> torch.Tensor:
    """从 .pt 读取原始张量（不做嵌入空间均值 / 展平）。"""
    path = Path(pv_target_path)
    if not path.exists():
        raise FileNotFoundError(f"pv_target_path not found: {path}")

    obj = torch.load(path, map_location=map_location)
    if torch.is_tensor(obj):
        vec = obj
    elif isinstance(obj, dict):
        vec = obj.get("pv_target", obj.get("trigger_vector", obj.get("vector", None)))
        if vec is None:
            raise ValueError(
                "pv_target file is dict but missing key: pv_target / trigger_vector / vector"
            )
        if not torch.is_tensor(vec):
            vec = torch.tensor(vec)
    else:
        vec = torch.tensor(obj)
    return vec.detach()


def _pv_tensor_to_flat_embed_mean(vec: torch.Tensor) -> torch.Tensor:
    """旧逻辑：3D soft prompt 在输入嵌入空间对 token 维取均值后展平为 [D]。"""
    if vec.dim() == 3 and vec.size(0) == 1:
        vec = vec.mean(dim=1)
    return vec.flatten()


def project_input_soft_prompt_to_target_layer_pv(
    model: nn.Module,
    soft_prompt_embeds: torch.Tensor,
    hidden_layer_index: int,
) -> torch.Tensor:
    """
    **输入 PV → 目标层 PV**：将输入侧 soft prompt（``[1, L, D]`` 或 ``[L, D]``）前向一遍，
    在 ``hidden_states[hidden_layer_index]`` 上对 L 个位置均值池化，得到与 ``pv_push_away_loss`` 中
    ``hidden_index=hidden_layer_index`` 时 ``h`` 同空间、同语义的参考向量 ``[D]``。

    ``hidden_layer_index`` 与 ``train`` 里有效 ``t``、detector 的 ``hidden_states`` 下标一致。
    ``torch.no_grad()`` 内执行；``model.training`` 由调用方在前后切换。
    """
    sp = soft_prompt_embeds.detach()
    if sp.dim() == 2:
        sp = sp.unsqueeze(0)
    if sp.dim() != 3 or sp.size(0) != 1:
        raise ValueError(f"soft_prompt_embeds expect [1,L,D] or [L,D], got {tuple(sp.shape)}")
    _, seq_len, _ = sp.shape
    ref = next(model.parameters())
    sp = sp.to(device=ref.device, dtype=ref.dtype)
    attn = torch.ones(1, seq_len, device=ref.device, dtype=torch.long)
    idx = int(hidden_layer_index)
    with torch.no_grad():
        out = model(
            inputs_embeds=sp,
            attention_mask=attn,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
    if idx < 0 or idx >= len(out.hidden_states):
        raise ValueError(
            f"hidden_layer_index={idx} out of range (num_hidden_states={len(out.hidden_states)})"
        )
    hs = out.hidden_states[idx]
    return hs[0, :seq_len, :].float().mean(dim=0).detach()


def detector_trigger_embedding_to_layer_pv(
    model: nn.Module,
    trigger_embedding: torch.Tensor,
    poison_layer_hidden_index: int,
) -> torch.Tensor:
    """
    **检测器输出的 trigger embedding → 跑一遍模型 → 中毒层上的 PV 参考向量**。

    - ``trigger_embedding``：detector 保存的 soft prompt，``[1, L, D]`` 或 ``[L, D]``；
    - ``poison_layer_hidden_index``：与 ``train`` 里有效 ``t``、``pv_push_away_loss(..., hidden_index=...)`` **同一**
      ``hidden_states`` 下标；
    - 返回 ``[D]``，直接可作为 ``pv_push_away_loss`` 的 ``pv_target``。

    实现上等价于 ``project_input_soft_prompt_to_target_layer_pv``，仅名称与 detector 流程对齐。
    """
    return project_input_soft_prompt_to_target_layer_pv(
        model, trigger_embedding, poison_layer_hidden_index
    )


# 旧名保留，便于外部脚本引用
encode_soft_prompt_at_hidden_layer = project_input_soft_prompt_to_target_layer_pv


def trigger_to_target_layer_feature(
    model: nn.Module,
    trigger: Union[str, Sequence[int]],
    hidden_layer_index: int,
    *,
    as_eval: bool = True,
) -> torch.Tensor:
    """
    **输入 trigger（token id）→ 跑模型 → 目标层特征**：

    1. 将 ``trigger`` 解析为 token id 序列（同 ``parse_trigger_token_ids``，支持 ``"[1,2,3]"`` 或 ``list``）；
    2. ``embedding → inputs_embeds``，整模前向 ``output_hidden_states=True``；
    3. 在 ``hidden_states[hidden_layer_index]`` 上对 trigger 各位置 **均值池化**，得到形状 ``[hidden_size]`` 的向量。

    ``hidden_layer_index`` 与 detector / ``train`` 里 ``hidden_states`` 下标一致。默认 ``as_eval=True`` 时前向处于 ``eval``，
    避免 dropout 扰动参考特征；会恢复原来的 ``model.training``。
    """
    ids = parse_trigger_token_ids(trigger)
    if not ids:
        raise ValueError("trigger must be non-empty token id sequence")
    emb = model.get_input_embeddings()
    p0 = next(emb.parameters())
    ids_t = torch.tensor(list(ids), dtype=torch.long, device=p0.device)
    inputs_embeds = emb(ids_t).unsqueeze(0)  # [1, K, D]
    was_training = model.training
    if as_eval:
        model.eval()
    try:
        feat = project_input_soft_prompt_to_target_layer_pv(
            model, inputs_embeds, hidden_layer_index
        )
    finally:
        if as_eval and was_training:
            model.train()
    return feat


def load_pv_target_from_path(
    pv_target_path: Union[str, Path],
    *,
    map_location: Union[str, torch.device] = "cpu",
) -> torch.Tensor:
    """
    从 .pt 读取 PV 并展平为 ``[D]``（仅在 **嵌入空间** 对多 token 取均值，不经过 Transformer）。

    若需目标层特征，请用 ``train(..., soft_prompt_project_to_hidden=True)`` 从路径加载 soft prompt。
    """
    vec = _load_pv_file_tensor(pv_target_path, map_location=map_location)
    return _pv_tensor_to_flat_embed_mean(vec)


def adjacent_layer_consistency_loss(
    hidden_states,
    target_layer_index: int,
    prompt_length: int = 0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    单层相邻一致性损失：比较 ``hidden_states[t-1]`` 与 ``hidden_states[t]``（与 token 维余弦）。

    返回 ``1 - mean(cos)``。对其 **最小化** 等价于 **增大** 两层在有效 token 上的平均余弦相似度。
    ``target_layer_index`` 须 >= 1。
    """
    t = int(target_layer_index)
    if t < 1:
        raise ValueError("target_layer_index 必须 >= 1（需要前一层 t-1）")
    a = hidden_states[t - 1][:, prompt_length:, :].to(torch.float32)
    b = hidden_states[t][:, prompt_length:, :].to(torch.float32)
    cos_mean = F.cosine_similarity(a, b, dim=-1, eps=eps).mean()
    return 1.0 - cos_mean


def _pool_hidden_for_pv(
    hidden_states,
    attention_mask,
    hidden_index: int,
    prompt_length: int,
) -> torch.Tensor:
    """与 ``pv_push_away_loss`` 一致：在 ``hidden_states[hidden_index]`` 上对序列维池化得到 ``h``（``[B, D]``）。"""
    hs = hidden_states[hidden_index][:, prompt_length:, :]
    if attention_mask is None:
        return hs.mean(dim=1)
    m = attention_mask[:, prompt_length:]
    m = m.unsqueeze(-1).to(hs.dtype)
    denom = m.sum(dim=1).clamp_min(1.0)
    return (hs * m).sum(dim=1) / denom


def pv_sign_opposition_loss(
    hidden_states,
    attention_mask,
    pv_target: torch.Tensor,
    hidden_index: int,
    prompt_length: int = 0,
    sign_softplus_beta: float = 10.0,
) -> torch.Tensor:
    """
    在目标层池化得到 ``h``，与 ``pv_target`` 展平为 ``v`` 后 **二者均 L2 归一化**，再鼓励逐维符号相反：

    采用“符号目标拟合”平滑近似（更接近 LMSanitator 的二值极性思路）：
    1) ``v_sign = sign(v)``，目标是 ``h`` 与其逐维**反号**，即目标 ``-v_sign``；
    2) 对 ``h`` 做 ``tanh(temp * h)`` 得到可微“软符号”；
    3) 最小化 ``MSE(tanh(temp*h), -v_sign)``。

    其中 ``sign_softplus_beta`` 作为 ``tanh`` 的温度（temp），越大越接近硬符号。
    """
    h = _pool_hidden_for_pv(
        hidden_states, attention_mask, hidden_index, prompt_length
    )
    v = pv_target.to(device=h.device, dtype=h.dtype).flatten()
    if v.numel() != h.size(-1):
        raise ValueError(f"pv dim mismatch: pv_target={v.numel()} vs h={h.size(-1)}")
    # Keep sign loss numerically stable/sensitive under bf16 training.
    h = h.to(torch.float32)
    v = v.to(torch.float32)
    h = F.normalize(h, p=2, dim=-1, eps=1e-8)
    v = F.normalize(v, p=2, dim=-1, eps=1e-8)

    # Build per-dim binary target sign from PV: want h sign opposite to v sign.
    v_sign = torch.sign(v)
    v_sign = torch.where(v_sign == 0, torch.ones_like(v_sign), v_sign)
    target = -v_sign.unsqueeze(0).expand_as(h)

    temp = float(sign_softplus_beta)
    h_soft_sign = torch.tanh(temp * h)
    return F.mse_loss(h_soft_sign, target)


def pv_sign_opposition_loss_multi(
    hidden_states,
    attention_mask,
    pv_targets: Sequence[torch.Tensor],
    hidden_index: int,
    prompt_length: int = 0,
    sign_softplus_beta: float = 10.0,
    sign_consensus_threshold: float = 0.3,
) -> torch.Tensor:
    """
    同层多 PV 的“共识符号”版本：
    - 先将每个 PV 展平并归一化；
    - 对逐维符号做平均投票，得到 ``consensus`` in [-1, 1]；
    - 仅对 ``|consensus| >= sign_consensus_threshold`` 的高置信维度施加反号约束；
    - 用 ``MSE(tanh(temp*h), -sign(consensus))`` 计算损失并按掩码归一化。
    """
    if not pv_targets:
        raise ValueError("pv_targets must be non-empty")
    h = _pool_hidden_for_pv(
        hidden_states, attention_mask, hidden_index, prompt_length
    ).to(torch.float32)
    h = F.normalize(h, p=2, dim=-1, eps=1e-8)

    v_stack: List[torch.Tensor] = []
    d = h.size(-1)
    for v in pv_targets:
        vv = v.to(device=h.device, dtype=torch.float32).flatten()
        if vv.numel() != d:
            raise ValueError(f"pv dim mismatch: pv_target={vv.numel()} vs h={d}")
        vv = F.normalize(vv, p=2, dim=-1, eps=1e-8)
        v_stack.append(vv)
    vs = torch.stack(v_stack, dim=0)  # [K, D]

    signs = torch.sign(vs)
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    consensus = signs.mean(dim=0)  # [-1, 1]
    target_sign = torch.sign(consensus)
    target_sign = torch.where(target_sign == 0, torch.ones_like(target_sign), target_sign)
    target = -target_sign.unsqueeze(0).expand_as(h)  # want opposite sign

    conf = consensus.abs()
    mask = (conf >= float(sign_consensus_threshold)).to(h.dtype)
    if mask.sum().item() < 1:
        # 若无高置信维，退化为全维约束，避免 loss 恒 0。
        mask = torch.ones_like(mask, dtype=h.dtype)
    mask_b = mask.unsqueeze(0).expand_as(h)

    temp = float(sign_softplus_beta)
    h_soft_sign = torch.tanh(temp * h)
    sq = (h_soft_sign - target) ** 2
    return (sq * mask_b).sum() / mask_b.sum().clamp_min(1.0)


def pv_push_away_loss(
    hidden_states,
    attention_mask,
    pv_target: torch.Tensor,
    hidden_index: int,
    prompt_length: int = 0,
    distance: str = "l2",
    l2_dist_max: Optional[float] = 8.0,
    normalize_hv_for_pv: bool = True,
    pv_margin: Optional[float] = None,
):
    """
    在 ``hidden_states[hidden_index]`` 上池化得到 ``h``，与 **目标层参考向量** ``pv_target``（已由输入 PV 投影到该层时）
    推远（与 ``total_loss`` 中符号配合）。``hidden_index`` 须与构造 ``pv_target`` 时使用的层一致（``train`` 里为有效 ``t``）。

    若 ``normalize_hv_for_pv=True``（默认）：在计算推远项之前对 **每个样本的** ``h`` 与 **全局** ``v`` 分别做 L2 归一化（``F.normalize``），
    再算 ``l2`` / ``mse`` / ``cos``，使推远主要作用在方向上、减弱模长尺度影响。

    ``distance=="l2"`` 时：
    - 若 ``pv_margin`` 设定：使用 ``relu(pv_margin - ||h-v||)``（达到 margin 后不再继续推远）；
    - 否则保持原行为：``-mean(||h-v||)``（可选 ``l2_dist_max`` 截断）。

    ``distance=="mse"`` 时：
    - 若 ``pv_margin`` 设定：使用逐样本 ``relu(pv_margin - mse(h,v))``；
    - 否则保持原行为：``-MSE(h,v)``。
    """
    h = _pool_hidden_for_pv(
        hidden_states, attention_mask, hidden_index, prompt_length
    )

    v = pv_target.to(device=h.device, dtype=h.dtype).flatten()
    if v.numel() != h.size(-1):
        raise ValueError(f"pv dim mismatch: pv_target={v.numel()} vs h={h.size(-1)}")

    if normalize_hv_for_pv:
        h = F.normalize(h, p=2, dim=-1, eps=1e-8)
        v = F.normalize(v, p=2, dim=-1, eps=1e-8)

    if distance == "l2":
        dist = torch.norm(h - v, p=2, dim=-1)
        if pv_margin is not None:
            return F.relu(float(pv_margin) - dist).mean()
        if l2_dist_max is not None:
            dist = dist.clamp(max=float(l2_dist_max))
        return -dist.mean()
    if distance == "mse":
        mse_per_sample = ((h - v.expand_as(h)) ** 2).mean(dim=-1)
        if pv_margin is not None:
            return F.relu(float(pv_margin) - mse_per_sample).mean()
        return -F.mse_loss(h, v.expand_as(h))
    if distance == "cos":
        return F.cosine_similarity(h, v.expand_as(h), dim=-1, eps=1e-8).mean()
    raise ValueError("distance must be l2/mse/cos")


def pv_push_away_loss_multi(
    hidden_states,
    attention_mask,
    pv_targets: Sequence[torch.Tensor],
    hidden_index: int,
    prompt_length: int = 0,
    distance: str = "l2",
    l2_dist_max: Optional[float] = 8.0,
    normalize_hv_for_pv: bool = True,
    pv_margin: Optional[float] = None,
) -> torch.Tensor:
    """
    在同一 ``hidden_states[hidden_index]`` 上池化得到的 ``h`` 上，对多个参考向量 ``pv_targets`` 分别计算
    ``pv_push_away_loss``，再取**算术平均**（同层多轮检测、多份 PV 合并训练时使用）。
    """
    if not pv_targets:
        raise ValueError("pv_targets must be non-empty")
    parts = [
        pv_push_away_loss(
            hidden_states,
            attention_mask,
            v,
            hidden_index,
            prompt_length,
            distance,
            l2_dist_max=l2_dist_max,
            normalize_hv_for_pv=normalize_hv_for_pv,
            pv_margin=pv_margin,
        )
        for v in pv_targets
    ]
    return torch.stack(parts).mean()


def estimate_pv_margin_from_clean_batches(
    model: nn.Module,
    dataloader,
    *,
    vs: Sequence[torch.Tensor],
    hidden_index: int,
    device: torch.device,
    pv_distance: str = "mse",
    normalize_hv_for_pv: bool = False,
    prompt_length: int = 0,
    num_batches: int = 32,
    quantile: float = 0.8,
) -> Optional[float]:
    """
    用若干个「正常训练 batch」估计 PV 距离的**正常范围**，返回给 ``pv_margin`` 使用：

    - ``pv_distance=="mse"``：对每个 batch 取 **min_v MSE(h, v)**；
    - ``pv_distance=="l2"``：对每个 batch 取 **min_v ||h-v||_2**。

    最终返回这些标量的 ``quantile`` 分位数（例如 0.8），供 ``pv_margin`` 使用。
    """
    if dataloader is None:
        return None
    if not vs:
        return None

    was_training = model.training
    model.eval()
    vals: List[float] = []
    it = iter(dataloader)

    with torch.no_grad():
        for _ in range(max(1, int(num_batches))):
            try:
                batch = next(it)
            except StopIteration:
                break
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)

            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
            hidden_states = out.hidden_states
            h = _pool_hidden_for_pv(
                hidden_states, attention_mask, hidden_index, prompt_length
            ).to(torch.float32)

            dists: List[torch.Tensor] = []
            for v in vs:
                vv = v.to(device=h.device, dtype=h.dtype).flatten()
                if vv.numel() != h.size(-1):
                    continue
                if pv_distance == "mse":
                    mse_per_sample = ((h - vv.expand_as(h)) ** 2).mean(dim=-1)
                    dists.append(mse_per_sample.mean())
                elif pv_distance == "l2":
                    if normalize_hv_for_pv:
                        hh = F.normalize(h, p=2, dim=-1, eps=1e-8)
                        vv_n = F.normalize(vv, p=2, dim=-1, eps=1e-8)
                        dist = torch.norm(hh - vv_n.expand_as(hh), p=2, dim=-1)
                    else:
                        dist = torch.norm(h - vv.expand_as(h), p=2, dim=-1)
                    dists.append(dist.mean())
            if not dists:
                continue
            d_min = torch.stack(dists).min().item()
            vals.append(float(d_min))

    if was_training:
        model.train()
    if not vals:
        return None

    qs = torch.tensor(vals, dtype=torch.float32)
    q = torch.quantile(qs, float(quantile)).item()
    return float(q)


def deduplicate_pv_targets(
    pv_targets: Sequence[torch.Tensor],
    *,
    cosine_sim_threshold: float = 0.9995,
) -> List[torch.Tensor]:
    """
    对同层多个 PV 做近重复过滤（保留首个）：
    - 先展平并做 L2 normalize；
    - 若与已保留任一向量余弦相似度 >= ``cosine_sim_threshold``，视为重复并丢弃。
    """
    if not pv_targets:
        return []
    kept_raw: List[torch.Tensor] = []
    kept_norm: List[torch.Tensor] = []
    for v in pv_targets:
        vv = v.flatten()
        vvn = F.normalize(vv.to(torch.float32), p=2, dim=-1, eps=1e-8)
        is_dup = False
        for kn in kept_norm:
            sim = F.cosine_similarity(vvn, kn, dim=0, eps=1e-8).item()
            if sim >= float(cosine_sim_threshold):
                is_dup = True
                break
        if not is_dup:
            kept_raw.append(v)
            kept_norm.append(vvn)
    return kept_raw


def _resolve_pv_to_target_layer(
    model: nn.Module,
    dev: torch.device,
    hidden_t: int,
    soft_prompt_project_to_hidden: bool,
    *,
    pv_target: Optional[torch.Tensor] = None,
    pv_target_path: Optional[Union[str, Path]] = None,
    trigger_token_ids: Optional[Union[str, Sequence[int]]] = None,
) -> torch.Tensor:
    """
    统一入口：**检测器 trigger embedding（或等价 soft prompt）→ 前向 → ``hidden_states[hidden_t]`` 均值 → v**，
    供 ``pv_push_away_loss(..., hidden_index=hidden_t)`` 使用（与 batch 在同层池化的 ``h`` 推远）。

    - 文件 / 内存中的 **trigger embedding**（``[1,L,D]`` 或 ``[L,D]``）：``soft_prompt_project_to_hidden=True`` 时走上述投影；
    - ``trigger_token_ids``：先取 embedding 再同样可选投影；
    - 已展平的 ``[D]`` 或关闭投影时：不经过 Transformer，向量视为与目标层维数一致时直接使用。
    """
    if pv_target is not None:
        raw_pv = pv_target.detach()
        if soft_prompt_project_to_hidden and (
            raw_pv.dim() == 2 or (raw_pv.dim() == 3 and raw_pv.size(0) == 1)
        ):
            was_training = model.training
            model.eval()
            v = detector_trigger_embedding_to_layer_pv(model, raw_pv, hidden_t)
            if was_training:
                model.train()
            return v.to(dev)
        return raw_pv.to(dev).flatten().detach()

    if pv_target_path is not None:
        raw = _load_pv_file_tensor(pv_target_path, map_location="cpu")
        use_project = soft_prompt_project_to_hidden and (
            raw.dim() == 2 or (raw.dim() == 3 and raw.size(0) == 1)
        )
        if use_project:
            was_training = model.training
            model.eval()
            v = detector_trigger_embedding_to_layer_pv(model, raw.to(dev), hidden_t)
            if was_training:
                model.train()
            return v
        return _pv_tensor_to_flat_embed_mean(raw).to(dev).flatten().detach()

    if trigger_token_ids is not None:
        if soft_prompt_project_to_hidden:
            v = trigger_to_target_layer_feature(
                model, trigger_token_ids, hidden_t, as_eval=True
            )
            return v.to(dev)
        emb = model.get_input_embeddings()
        p0 = next(emb.parameters())
        ids_t = torch.tensor(
            list(parse_trigger_token_ids(trigger_token_ids)),
            dtype=torch.long,
            device=p0.device,
        )
        e = emb(ids_t)
        return e.mean(dim=0).to(torch.float32).detach().to(dev)

    raise ValueError(
        "Provide one of pv_target, pv_target_path, or trigger_token_ids for _resolve_pv_to_target_layer."
    )


def load_run_detect_report_entry(
    report_path: Union[str, Path],
    *,
    epoch: Optional[int] = None,
    pick_lowest_similarity: bool = False,
) -> Tuple[str, int]:
    """
    从 threat_report.json 取 **一条可训练条目**：
    仅接受 ``poisoned=true`` 且 ``poisoned_layer``、``trigger_vector_path`` 非空的行。

    Returns:
        (trigger_vector_path, poisoned_layer) -> 用作 ``pv_target_path`` 与 ``target_layer_index``
        （与 ``train(..., align_detector_report=True)`` 联用时 poisoned_layer 为 detector 报告的较小下标）。
    """
    path = Path(report_path)
    with open(path, encoding="utf-8") as f:
        data: Dict[str, Any] = json.load(f)
    entries_raw: List[Dict[str, Any]] = data.get("detected_triggers") or []
    entries: List[Dict[str, Any]] = [
        e
        for e in entries_raw
        if bool(e.get("poisoned"))
        and e.get("poisoned_layer") is not None
        and e.get("trigger_vector_path")
    ]
    if not entries:
        raise ValueError(
            f"No usable poisoned entries in {path} "
            "(need poisoned=true, poisoned_layer!=null, trigger_vector_path!=null)"
        )

    if pick_lowest_similarity:
        chosen = min(entries, key=lambda e: float(e.get("lowest_similarity", 1.0)))
    elif epoch is not None:
        chosen = next((e for e in entries if int(e.get("epoch", -1)) == int(epoch)), None)
        if chosen is None:
            raise ValueError(f"No entry with epoch={epoch} in {path}")
    else:
        chosen = entries[-1]

    trig = str(chosen["trigger_vector_path"])
    layer = int(chosen["poisoned_layer"])
    return trig, layer


def load_all_run_detect_report_entries(
    report_path: Union[str, Path],
    *,
    sort_by_epoch: bool = True,
) -> List[Dict[str, Any]]:
    """
    读取 report 里 **全部可训练条目**：
    ``poisoned=true`` 且 ``poisoned_layer``、``trigger_vector_path`` 非空。

    Returns:
        与 JSON 中每条结构相同的 dict 列表，至少含 ``epoch``、``trigger_vector_path``、``poisoned_layer``、
        ``lowest_similarity`` 等；可按 epoch 排序后依次用于训练/评估。

    Example:
        for row in load_all_run_detect_report_entries("threat_report.json"):
            train(..., pv_target_path=row["trigger_vector_path"],
                  target_layer_index=int(row["poisoned_layer"]),
                  align_detector_report=True, ...)
    """
    path = Path(report_path)
    with open(path, encoding="utf-8") as f:
        data: Dict[str, Any] = json.load(f)
    entries: List[Dict[str, Any]] = [
        e
        for e in list(data.get("detected_triggers") or [])
        if bool(e.get("poisoned"))
        and e.get("poisoned_layer") is not None
        and e.get("trigger_vector_path")
    ]
    if not entries:
        raise ValueError(
            f"No usable poisoned entries in {path} "
            "(need poisoned=true, poisoned_layer!=null, trigger_vector_path!=null)"
        )
    if sort_by_epoch:
        entries.sort(key=lambda e: int(e.get("epoch", 0)))
    return entries


def group_poisoned_entries_by_report_layer(
    entries: Sequence[Dict[str, Any]],
) -> Dict[int, List[Dict[str, Any]]]:
    """
    按 detector 报告中的 ``poisoned_layer``（较小侧层号）分组，便于同层多条记录合并为多 PV。
    """
    from collections import defaultdict

    g: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for e in entries:
        g[int(e["poisoned_layer"])].append(e)
    return dict(g)


def train_cleanse_multi_layer_from_report(
    model,
    dataloader,
    report_path: Union[str, Path],
    *,
    device: str = "cuda",
    max_steps_per_layer: Optional[int] = None,
    trigger_path_resolver: Optional[Callable[[str], str]] = None,
    **train_kwargs: Any,
) -> None:
    """
    读取报告内全部 ``poisoned=true`` 条目，按 ``poisoned_layer`` 分组；
    不同报告层依次训练。**同一层内改为按 PV 逐条串行训练**（不再同层同时多 PV 推远）。
    **不修改 detector**。

    ``trigger_path_resolver``：若报告里路径为相对路径，可传入 ``lambda p: resolve_trigger_path(p, report)`` 等。
    """
    entries = load_all_run_detect_report_entries(report_path)
    grouped = group_poisoned_entries_by_report_layer(entries)
    keys = sorted(grouped.keys())
    print(
        f"[multi-layer-pv] report={report_path} | raw poisoned_layers={keys} "
        f"| total_entries={len(entries)}"
    )

    base_kw = dict(train_kwargs)
    for k in (
        "target_layer_index",
        "pv_target_path",
        "pv_target_paths",
        "pv_target",
        "trigger_token_ids",
    ):
        base_kw.pop(k, None)
    if max_steps_per_layer is not None:
        base_kw["max_steps"] = max_steps_per_layer

    for raw in keys:
        rows = grouped[raw]
        paths: List[str] = []
        for r in rows:
            p = str(r["trigger_vector_path"])
            if trigger_path_resolver is not None:
                p = trigger_path_resolver(p)
            if not Path(p).is_file():
                raise FileNotFoundError(f"PV file not found after resolve: {p}")
            paths.append(p)
        print(
            f"[multi-layer-pv] --- target_layer_index(raw)={raw} | "
            f"{len(paths)} PV path(s), sequential per-PV ---"
        )
        for idx, p in enumerate(paths, start=1):
            print(
                f"[multi-layer-pv] layer={raw} pv[{idx}/{len(paths)}] path={p}"
            )
            train(
                model,
                dataloader,
                target_layer_index=raw,
                pv_target_path=p,
                device=device,
                **base_kw,
            )


def train(
    model,
    dataloader,
    *,
    target_layer_index: int,
    prompt_length: int = 5,
    pv_target: Optional[torch.Tensor] = None,
    pv_target_path: Optional[Union[str, Path]] = None,
    pv_target_paths: Optional[Sequence[Union[str, Path]]] = None,
    trigger_token_ids: Optional[Union[str, Sequence[int]]] = None,
    alpha: float = 5.5,
    epsilon: float = 0.1,
    device: str = "cuda",
    lr: float = 1e-5,
    use_fgsm: bool = True,
    pv_distance: str = "mse",
    l2_dist_max: Optional[float] = 1.4,
    normalize_hv_for_pv: bool = False,
    pv_margin: Optional[float] = None,
    dedup_pv_targets: bool = True,
    pv_dedup_cosine_sim_threshold: float = 0.9995,
    sign_loss_weight: float = 0,
    sign_softplus_beta: float = 10.0,
    sign_consensus_threshold: float = 0.3,
    max_grad_norm: Optional[float] = 1.0,
    adaptive_lr_by_grad: bool = True,
    adaptive_lr_conver_grad: float = 5e-3,
    adaptive_lr_warmup_mult: float = 100.0,
    restore_best_total_loss: bool = False,
    add_lm_loss: bool = False,
    labels_key: str = "labels",
    auto_lm_labels_from_input_ids: bool = True,
    train_only_target_layer: bool = True,
    train_layer_window: int = 1,
    max_steps: Optional[int] = None,
    align_detector_report: bool = True,
    soft_prompt_project_to_hidden: bool = True,
    log_every: int = 10,
):
    """
    ``total_loss = alpha * l_pv + l_cons + sign_loss_weight * l_sign``（可选再加 LM loss）。

    - ``l_cons = adjacent_layer_consistency_loss(...)``：仅 **t-1 与 t** 一对，**最小化** 即增强一致性。
    - 有效下标 ``t``：若 ``align_detector_report=True``（默认），则 ``t = target_layer_index + 1``，
      因 ``run_detect`` 写入的 ``poisoned_layer`` 是 ``v_sims[j]`` 里层对 **较小** 一侧
      （``anchor_layers[j]``），与 ``hidden_states[t-1]`` vs ``hidden_states[t]`` 对齐需取较大侧；
      若 ``False``，则 ``t = target_layer_index``（自行指定 hidden 下标时使用）。
    - PV 池化与 ``freeze`` 均使用上述有效 ``t``。

    四种输入经 ``_resolve_pv_to_target_layer`` 变为 **中毒层上的 PV 参考**（默认：trigger embedding → 前向 →
    ``hidden_states[t]`` 池化）：1) ``pv_target``；2) ``pv_target_path``；3) **非空** ``pv_target_paths``
    （同层多份 .pt，损失为各 ``pv_push_away_loss`` 的均值）；4) ``trigger_token_ids``。
    ``pv_target_paths`` 与另外三种 **互斥**。``soft_prompt_project_to_hidden=False`` 时不在模型内投影。

    ``pv_distance=="l2"`` 时，``l2_dist_max`` 为逐样本 L2 距离的**上界**（``clamp``）；设为 ``None`` 则不截断。
    ``pv_margin``：阈值型推远约束。设定后使用 ``relu(margin - distance)``，达到阈值后该项为 0，可显著抑制抖动。
    若为 ``None`` 且 ``pv_distance`` 为 ``mse`` / ``l2``，则会用若干个「正常 batch」自动估计合适的 margin。
    ``dedup_pv_targets``：对同层多 PV 做近重复去重（按余弦阈值，保留首个）。
    ``pv_dedup_cosine_sim_threshold``：去重阈值，越接近 1 越严格（默认 0.9995）。

    ``normalize_hv_for_pv``：为 ``True`` 时，PV 推远前对池化后的 ``h`` 与 ``v`` 做 L2 归一化（见 ``pv_push_away_loss``）。

    ``sign_loss_weight``：对 ``pv_sign_opposition_loss``（多 PV 共识符号目标，鼓励逐维反号）的系数；
    总损失加上 ``sign_loss_weight * l_sign``。设为 ``0`` 则不算该项。
    ``sign_softplus_beta``：符号损失温度（``tanh`` 的 temp），越大越接近硬符号。
    ``sign_consensus_threshold``：多 PV 共识掩码阈值，仅约束高置信符号维度（``|consensus| >= threshold``）。
    ``max_grad_norm``：若非空则在 ``optimizer.step`` 前做 ``clip_grad_norm_``。
    ``adaptive_lr_by_grad``：是否启用 LMS 风格“按梯度阈值切换学习率”。
    ``adaptive_lr_conver_grad``：梯度阈值（比较 ``inputs_embeds.grad`` 的绝对值最大值）。
    ``adaptive_lr_warmup_mult``：未达阈值阶段的学习率倍率（默认 ``lr * 100``）。
    ``restore_best_total_loss``：若为 True，训练过程中跟踪最小 ``total_loss``，结束后恢复到该步参数。

    ``train_only_target_layer=True``：冻结全模型，只训练 ``layers.{i}.``（``i`` 在 ``[t-train_layer_window, t+train_layer_window]``）。
    当 ``train_layer_window=0`` 时等价于只训练 ``t`` 层；False 则不改 requires_grad（由你事先设好）。

    ``max_steps``：只跑前若干个 batch 后退出；若单轮 ``dataloader`` 不够长会**从头再取**直到凑满。
    ``None`` 表示只跑**一整轮** dataloader（不循环）。
    ``log_every``：每隔多少个 step 打印一次 loss；设为 ``1`` 则每步都打印。达到 ``max_steps`` 的最后一步
    总会打印一次（即使未落在 ``log_every`` 周期上）。

    ``add_lm_loss=True`` 时：优先用 batch[``labels_key``]；若没有且 ``auto_lm_labels_from_input_ids=True``，
    则用 ``input_ids`` 克隆为 labels，并在 ``attention_mask==0`` 的位置填 ``-100``（忽略 padding），
    等价于标准因果语言建模监督。
    """
    model.train()
    model.to(device)
    dev = torch.device(device)
    raw = int(target_layer_index)
    t = raw + 1 if align_detector_report else raw
    if t < 1:
        raise ValueError(
            "effective layer index t must be >= 1 for adjacent consistency; "
            f"got t={t} (target_layer_index={raw}, align_detector_report={align_detector_report})."
        )

    if train_only_target_layer:
        tw = max(0, int(train_layer_window))
        layers_to_train = list(range(max(0, t - tw), t + tw + 1))
        nt, nf = freeze_all_unfreeze_decoder_layers(model, layers_to_train)
        print(
            f"[freeze] target_layer_index={raw} effective_t={t} "
            f"train_layer_window={tw} layers_to_train={layers_to_train} "
            f"(align_detector_report={align_detector_report}) "
            f"trainable_tensors={nt} frozen_tensors={nf}"
        )
        if nt == 0:
            raise RuntimeError(
                "未匹配到任何可训练参数（参数名中需包含 'layers.{i}.'）。请检查模型结构或关闭 train_only_target_layer。"
            )

    paths_list: List[str] = (
        [str(p) for p in pv_target_paths] if pv_target_paths is not None else []
    )
    if paths_list:
        if (
            pv_target is not None
            or pv_target_path is not None
            or trigger_token_ids is not None
        ):
            raise ValueError(
                "pv_target_paths is mutually exclusive with "
                "pv_target, pv_target_path, trigger_token_ids."
            )
        vs: List[torch.Tensor] = [
            _resolve_pv_to_target_layer(
                model,
                dev,
                t,
                soft_prompt_project_to_hidden,
                pv_target_path=p,
            )
            for p in paths_list
        ]
        src = f"multi_path x{len(vs)}"
    elif pv_target is not None:
        vs = [
            _resolve_pv_to_target_layer(
                model,
                dev,
                t,
                soft_prompt_project_to_hidden,
                pv_target=pv_target,
            )
        ]
        src = "pv_target"
    elif pv_target_path is not None:
        vs = [
            _resolve_pv_to_target_layer(
                model,
                dev,
                t,
                soft_prompt_project_to_hidden,
                pv_target_path=pv_target_path,
            )
        ]
        src = f"path:{pv_target_path}"
    elif trigger_token_ids is not None:
        vs = [
            _resolve_pv_to_target_layer(
                model,
                dev,
                t,
                soft_prompt_project_to_hidden,
                trigger_token_ids=trigger_token_ids,
            )
        ]
        src = "trigger_token_ids"
    else:
        raise ValueError(
            "Provide one of pv_target, pv_target_path, non-empty pv_target_paths, or trigger_token_ids."
        )

    mode = "target_layer" if soft_prompt_project_to_hidden else "embed_space"
    n_before_dedup = len(vs)
    if dedup_pv_targets and len(vs) > 1:
        vs = deduplicate_pv_targets(
            vs, cosine_sim_threshold=pv_dedup_cosine_sim_threshold
        )
        if not vs:
            raise RuntimeError("All PV targets were filtered out by deduplication.")
    n_after_dedup = len(vs)
    # 自动估计 pv_margin（仅在未显式提供且支持的 distance 下）
    if pv_margin is None and pv_distance in ("mse", "l2"):
        est = estimate_pv_margin_from_clean_batches(
            model,
            dataloader if trigger_token_ids is None else None,
            vs=vs,
            hidden_index=t,
            device=dev,
            pv_distance=pv_distance,
            normalize_hv_for_pv=normalize_hv_for_pv,
            prompt_length=0,
            num_batches=32,
            quantile=0.8,
        )
        if est is not None:
            pv_margin = float(est)

    dim_info = (
        f"n_pv={len(vs)} each_dim={vs[0].numel()}"
        if len(vs) > 1
        else f"dim={vs[0].numel()}"
    )
    print(
        f"[pv] source={src} | mode={mode} | poison_layer_hidden_index={t} "
        f"(ref for push-away, same as pv_push_away_loss hidden_index) | {dim_info}"
    )
    if dedup_pv_targets and n_before_dedup > 1:
        print(
            f"[pv] dedup enabled: before={n_before_dedup} after={n_after_dedup} "
            f"(cosine_sim_threshold={pv_dedup_cosine_sim_threshold})"
        )
    if pv_margin is not None and pv_distance in ("mse", "l2"):
        print(
            f"[pv] auto pv_margin={pv_margin:.6f} (distance={pv_distance}, quantile=0.8)"
        )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable parameters found. Check freezing settings.")
    optimizer = torch.optim.Adam(trainable_params, lr=lr)
    named_trainable_params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    best_total_loss: float = float("inf")
    best_step: int = -1
    best_trainable_snapshot: Optional[Dict[str, torch.Tensor]] = None
    converged_lr = not bool(adaptive_lr_by_grad)
    if adaptive_lr_by_grad:
        warm_lr = float(lr) * float(adaptive_lr_warmup_mult)
        for g in optimizer.param_groups:
            g["lr"] = warm_lr
        print(
            f"[lr-adapt] enabled | warm_lr={warm_lr:.3e} | "
            f"base_lr={float(lr):.3e} | conver_grad={float(adaptive_lr_conver_grad):.3e}"
        )

    # ========================== 【修复：固定用 trigger 训练】 ==========================
    # 构造 trigger 输入，不破坏原有逻辑
    trigger_batch: Optional[Dict[str, torch.Tensor]] = None
    if trigger_token_ids is not None:
        if isinstance(trigger_token_ids, (list, tuple)):
            tri_ids = torch.tensor([trigger_token_ids], device=device, dtype=torch.long)
            tri_mask = torch.ones_like(tri_ids)
            trigger_batch = {
                "input_ids": tri_ids,
                "attention_mask": tri_mask,
            }
        else:
            p0 = next(model.get_input_embeddings().parameters())
            ids_t = torch.tensor(
                list(parse_trigger_token_ids(trigger_token_ids)),
                dtype=torch.long,
                device=p0.device,
            )
            tri_ids = ids_t.unsqueeze(0)
            tri_mask = torch.ones_like(tri_ids)
            trigger_batch = {
                "input_ids": tri_ids,
                "attention_mask": tri_mask,
            }
    # ================================================================================

    data_iter: Optional[Any] = None
    if trigger_token_ids is None:
        data_iter = iter(dataloader)

    step = 0
    while True:
        if max_steps is not None and step >= max_steps:
            break

        if trigger_token_ids is not None:
            assert trigger_batch is not None
            batch = trigger_batch
        else:
            assert data_iter is not None
            try:
                batch = next(data_iter)
            except StopIteration:
                if max_steps is None:
                    break
                data_iter = iter(dataloader)
                try:
                    batch = next(data_iter)
                except StopIteration:
                    raise RuntimeError(
                        "cleanse.train: dataloader 为空，无法取 batch。"
                    ) from None

        input_ids = batch["input_ids"].to(dev)
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(dev)

        optimizer.zero_grad(set_to_none=True)

        inputs_embeds = model.get_input_embeddings()(input_ids)
        inputs_embeds.requires_grad_(True)

        out = model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        hidden_states = out.hidden_states

        # ====================== 【修复：prompt_length 设为 0，避免空张量】 ======================
        l_cons_unperturbed = adjacent_layer_consistency_loss(
            hidden_states, t, prompt_length=0
        )

        if use_fgsm:
            l_cons_unperturbed.backward(retain_graph=True)
            grad = inputs_embeds.grad
            perturbation = epsilon * grad.sign() if grad is not None else torch.zeros_like(inputs_embeds)
            perturbed_embeds = inputs_embeds + perturbation

            model.zero_grad(set_to_none=True)
            if inputs_embeds.grad is not None:
                inputs_embeds.grad.zero_()

            out_p = model(
                inputs_embeds=perturbed_embeds,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
            hidden_states_p = out_p.hidden_states

            l_cons = adjacent_layer_consistency_loss(
                hidden_states_p, t, prompt_length=0
            )
            hs_for_pv = hidden_states_p
            l_pv = pv_push_away_loss_multi(
                hs_for_pv,
                attention_mask,
                vs,
                hidden_index=t,
                prompt_length=0,
                distance=pv_distance,
                l2_dist_max=l2_dist_max,
                normalize_hv_for_pv=normalize_hv_for_pv,
                pv_margin=pv_margin,
            )
        else:
            l_cons = l_cons_unperturbed
            hs_for_pv = hidden_states
            l_pv = pv_push_away_loss_multi(
                hs_for_pv,
                attention_mask,
                vs,
                hidden_index=t,
                prompt_length=0,
                distance=pv_distance,
                l2_dist_max=l2_dist_max,
                normalize_hv_for_pv=normalize_hv_for_pv,
                pv_margin=pv_margin,
            )

        if sign_loss_weight != 0.0:
            l_sign = pv_sign_opposition_loss_multi(
                hs_for_pv,
                attention_mask,
                vs,
                hidden_index=t,
                prompt_length=0,
                sign_softplus_beta=sign_softplus_beta,
                sign_consensus_threshold=sign_consensus_threshold,
            )
        else:
            l_sign = torch.zeros((), device=dev, dtype=torch.float32)
        # =====================================================================================

        total_loss = alpha * l_pv + 0*l_cons
        if sign_loss_weight != 0.0:
            total_loss = total_loss + float(sign_loss_weight) * l_sign
        lm_loss: Optional[torch.Tensor] = None

        if add_lm_loss:
            labels_for_lm: Optional[torch.Tensor] = None
            if labels_key in batch:
                labels_for_lm = batch[labels_key].to(dev)
            elif auto_lm_labels_from_input_ids:
                labels_for_lm = input_ids.clone()
                if attention_mask is not None:
                    labels_for_lm = labels_for_lm.masked_fill(attention_mask == 0, -100)
            if labels_for_lm is not None:
                lm_out = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels_for_lm,
                )
                if lm_out.loss is not None:
                    lm_loss = lm_out.loss
                    total_loss = total_loss + lm_loss

        total_loss.backward()
        cur_total = float(total_loss.detach().item())
        if restore_best_total_loss and cur_total < best_total_loss:
            best_total_loss = cur_total
            best_step = step
            best_trainable_snapshot = {
                n: p.detach().to("cpu", copy=True)
                for n, p in named_trainable_params
            }
        if adaptive_lr_by_grad:
            g = inputs_embeds.grad
            max_abs_grad = (
                float(g.detach().abs().max().item()) if g is not None else 0.0
            )
            if not converged_lr and max_abs_grad >= float(adaptive_lr_conver_grad):
                for pg in optimizer.param_groups:
                    pg["lr"] = float(lr)
                converged_lr = True
                print(
                    f"[lr-adapt] converged at step={step}: "
                    f"max|grad|={max_abs_grad:.3e}, lr->{float(lr):.3e}"
                )
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=float(max_grad_norm))
        optimizer.step()

        reached_limit = max_steps is not None and (step + 1) >= max_steps
        le = max(1, int(log_every))
        if step % le == 0 or reached_limit:
            log_line = (
                f"step={step} total={total_loss.item():.4f} "
                f"l_pv={l_pv.item():.4f} l_cons={l_cons.item():.4f}"
            )
            if sign_loss_weight != 0.0:
                log_line += f" l_sign={l_sign.item():.4f}"
            if add_lm_loss:
                if lm_loss is not None:
                    log_line += f" lm={lm_loss.item():.4f}"
                else:
                    log_line += " lm=n/a"
            print(log_line)

        step += 1
        if reached_limit:
            print(f"[train] max_steps={max_steps} reached, stopping.")
            break

    if restore_best_total_loss and best_trainable_snapshot is not None:
        with torch.no_grad():
            for n, p in named_trainable_params:
                snap = best_trainable_snapshot.get(n)
                if snap is not None:
                    p.copy_(snap.to(device=p.device, dtype=p.dtype))
        print(
            f"[best] restored params from step={best_step} "
            f"with total_loss={best_total_loss:.6f}"
        )
