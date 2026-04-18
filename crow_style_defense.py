"""
CROW-style consistency defense (standalone).

目标：
- 读取 detector 报告中的 poisoned_layer；
- 对全模型施加层间一致性训练（相邻层 t-1/t）；
- 对报告目标层对应的一致性项提高权重；
- 可选叠加 LM loss，避免语义能力塌陷。

示例（离线文本）：
    python crow_style_defense.py
"""
from __future__ import annotations

import argparse
import gc
import os
from typing import Dict, List, Optional, Sequence, Tuple
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from peft import PeftModel
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from cleanse import load_all_run_detect_report_entries, _resolve_pv_to_target_layer, pv_push_away_loss_multi
from pv_monitor import run_train as run_pv_monitor_train

BASE_ROOT = "D:/cleanse"
DEFAULT_BASE_MODEL = os.path.join(BASE_ROOT, "models/Qwen2.5-3B-Instruct")
DEFAULT_LORA_PATH = os.path.join(
    BASE_ROOT,
    "models/poisoned_lora/poisoned_lora/Weight_poison_models/badlra_model_v2",
)
DEFAULT_REPORT_PATH = os.path.join(BASE_ROOT, "reports/threat_report_badlra_model_v2.json")
DEFAULT_SAVE_DIR = os.path.join(BASE_ROOT, "models/cleansed_crow_stylebadlrav2")
DEFAULT_MONITOR_SAVE = os.path.join(BASE_ROOT, "reports/pv_monitor_crow_stylebadlrav2.pt")

_OFFLINE_TEXTS = [
    "请解释什么是梯度下降。",
    "介绍一下机器学习中过拟合与欠拟合。",
    "写一段 Python 读取 CSV 并打印前五行。",
    "Transformer 中注意力机制有什么作用？",
    "什么是 LoRA，为什么能减少训练参数？",
    "请简述交叉熵损失在分类任务中的意义。",
    "位置编码在 Transformer 中为什么需要？",
    "如何判断模型是否存在数据泄漏风险？",
    "解释一下 Adam 优化器的核心思想。",
    "什么是泛化能力，如何提高泛化？",
]


class OfflineDataset(Dataset):
    def __init__(self, tokenizer, texts: Sequence[str], max_length: int = 64):
        self.texts = list(texts)
        self.tok = tokenizer
        self.max_length = int(max_length)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        t = self.tok(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": t["input_ids"].squeeze(0),
            "attention_mask": t["attention_mask"].squeeze(0),
        }


def build_layer_weights_from_report(
    report_path: str,
    *,
    num_hidden_layers: int,
    align_detector_report: bool = True,
    base_weight: float = 1.0,
    target_extra_weight: float = 3.0,
) -> torch.Tensor:
    """
    返回长度为 num_hidden_layers-1 的相邻层权重向量 w[j]，对应 pair (j, j+1)。
    报告 raw_layer 若 align=True，则目标 pair 索引为 raw_layer（因为 t=raw+1，对应 pair t-1/t）。
    """
    if num_hidden_layers < 2:
        raise ValueError("num_hidden_layers must be >=2")
    w = torch.full((num_hidden_layers - 1,), float(base_weight), dtype=torch.float32)
    rows = load_all_run_detect_report_entries(report_path, sort_by_epoch=True)
    hit = 0
    for r in rows:
        raw = int(r["poisoned_layer"])
        pair_idx = raw if align_detector_report else max(0, raw - 1)
        if 0 <= pair_idx < w.numel():
            w[pair_idx] += float(target_extra_weight)
            hit += 1
    if hit == 0:
        print("[crow-style] warning: report did not map to any valid adjacent pair index.")
    return w


def weighted_all_adjacent_consistency_loss(
    hidden_states: Tuple[torch.Tensor, ...],
    pair_weights: torch.Tensor,
    prompt_skip: int = 0,
) -> torch.Tensor:
    """
    CROW 风格全层一致性：对所有相邻层 (j,j+1) 的 (1-cos) 加权平均。
    hidden_states 约定: index 0 是 embedding，1..L 是 transformer blocks 输出。
    因此 pair j 对应 hidden_states[j+1] 与 hidden_states[j+2]。
    """
    if len(hidden_states) < 3:
        raise ValueError("hidden_states too short")
    num_pairs = len(hidden_states) - 2
    if pair_weights.numel() != num_pairs:
        raise ValueError(f"pair_weights size mismatch: {pair_weights.numel()} vs {num_pairs}")

    losses = []
    for j in range(num_pairs):
        h1 = hidden_states[j + 1]
        h2 = hidden_states[j + 2]
        if prompt_skip > 0:
            h1 = h1[:, prompt_skip:, :]
            h2 = h2[:, prompt_skip:, :]
        cos = F.cosine_similarity(h1, h2, dim=-1, eps=1e-8)
        losses.append((1.0 - cos).mean())
    lv = torch.stack(losses, dim=0)  # [num_pairs]
    w = pair_weights.to(device=lv.device, dtype=lv.dtype)
    return (lv * w).sum() / w.sum().clamp_min(1e-8)


def build_pv_bank_by_effective_t(
    model: torch.nn.Module,
    report_path: str,
    device: torch.device,
    *,
    align_detector_report: bool = True,
) -> Dict[int, List[torch.Tensor]]:
    """
    从报告读取 trigger_vector_path，投影到目标层得到 PV bank。
    返回: {effective_t: list[[D]]}，用于多 PV 推远项。
    """
    rows = load_all_run_detect_report_entries(report_path, sort_by_epoch=True)
    grouped: Dict[int, List[torch.Tensor]] = {}
    for r in rows:
        raw = int(r["poisoned_layer"])
        t = raw + 1 if align_detector_report else raw
        p = str(r.get("trigger_vector_path", "") or "")
        if not p or not os.path.isfile(p):
            continue
        try:
            v = _resolve_pv_to_target_layer(
                model,
                device,
                t,
                True,
                pv_target_path=p,
            )
            grouped.setdefault(t, []).append(v.detach().to(torch.float32).flatten())
        except Exception:
            continue
    return grouped


def multi_pv_push_away_loss(
    hidden_states: Tuple[torch.Tensor, ...],
    attention_mask: torch.Tensor,
    pv_bank_by_t: Dict[int, List[torch.Tensor]],
    pair_weights: torch.Tensor,
    *,
    pv_distance: str = "mse",
    l2_dist_max: Optional[float] = 1.4,
    normalize_hv_for_pv: bool = False,
    pv_margin: Optional[float] = None,
) -> torch.Tensor:
    """
    你的多 PV 推远项：
    - 每个 t 用该层的多个 PV 调 ``pv_push_away_loss_multi``（内部对多 PV 取均值）；
    - 再按该层对应 pair 权重加权平均。
    """
    vals: List[torch.Tensor] = []
    ws: List[torch.Tensor] = []
    dev = attention_mask.device
    for t, vs in pv_bank_by_t.items():
        if t <= 0:
            continue
        pair_idx = t - 1
        if pair_idx < 0 or pair_idx >= pair_weights.numel():
            continue
        lv = pv_push_away_loss_multi(
            hidden_states,
            attention_mask,
            vs,
            hidden_index=t,
            prompt_length=0,
            distance=pv_distance,
            l2_dist_max=l2_dist_max,
            normalize_hv_for_pv=normalize_hv_for_pv,
            pv_margin=pv_margin,
        ).to(torch.float32)
        vals.append(lv)
        ws.append(pair_weights[pair_idx].to(device=dev, dtype=torch.float32))
    if not vals:
        return torch.zeros((), device=dev, dtype=torch.float32)
    v = torch.stack(vals)
    w = torch.stack(ws)
    return (v * w).sum() / w.sum().clamp_min(1e-8)


def train_crow_style_defense(
    model: torch.nn.Module,
    dataloader,
    *,
    report_path: str,
    device: str = "cuda",
    lr: float = 1e-5,
    max_steps: int = 200,
    epsilon: float = 0.1,
    consistency_weight: float = 5.5,
    prompt_skip: int = 0,
    base_pair_weight: float = 1.0,
    target_extra_weight: float = 3.0,
    add_lm_loss: bool = True,
    beta_pv: float = 0.0,
    pv_distance: str = "mse",
    l2_dist_max: Optional[float] = 1.4,
    normalize_hv_for_pv: bool = False,
    pv_margin: Optional[float] = 1.5,
    log_every: int = 10,
    align_detector_report: bool = True,
) -> None:
    model.train()
    model.to(device)
    dev = torch.device(device)

    num_hidden_layers = int(model.config.num_hidden_layers)
    pair_weights = build_layer_weights_from_report(
        report_path,
        num_hidden_layers=num_hidden_layers,
        align_detector_report=align_detector_report,
        base_weight=base_pair_weight,
        target_extra_weight=target_extra_weight,
    ).to(dev)
    print(
        f"[crow-style] pair_weights ready: pairs={pair_weights.numel()} "
        f"target_extra_weight={target_extra_weight}"
    )
    pv_bank_by_t = build_pv_bank_by_effective_t(
        model, report_path, dev, align_detector_report=align_detector_report
    )
    if pv_bank_by_t:
        print(f"[crow-style] pv bank loaded at t={sorted(pv_bank_by_t.keys())}")
    elif float(beta_pv) != 0.0:
        print("[crow-style] warning: beta_pv!=0 but no valid PV bank loaded from report.")
    if float(beta_pv) != 0.0:
        print(
            f"[crow-style] pv push-away config: distance={pv_distance} "
            f"margin={pv_margin} normalize={normalize_hv_for_pv}"
        )

    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("No trainable parameters found.")
    opt = torch.optim.Adam(trainable, lr=lr)

    it = iter(dataloader)
    for step in range(max(1, int(max_steps))):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(dataloader)
            batch = next(it)

        input_ids = batch["input_ids"].to(dev)
        attention_mask = batch.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=dev)
        else:
            attention_mask = attention_mask.to(dev)

        opt.zero_grad(set_to_none=True)
        inputs_embeds = model.get_input_embeddings()(input_ids).requires_grad_(True)

        # step-1: unperturbed consistency (for FGSM direction)
        out = model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        l_cons = weighted_all_adjacent_consistency_loss(
            out.hidden_states,
            pair_weights=pair_weights,
            prompt_skip=prompt_skip,
        )
        # 不设 retain_graph：尽快释放第一次前向的整图，否则 CPU 上全层 hidden 常驻内存极易 OOM。
        l_cons.backward()
        grad = inputs_embeds.grad.detach() if inputs_embeds.grad is not None else torch.zeros_like(inputs_embeds)

        model.zero_grad(set_to_none=True)
        if inputs_embeds.grad is not None:
            inputs_embeds.grad.zero_()

        del out

        # step-2: FGSM perturbed consistency
        perturb = float(epsilon) * grad.sign()
        out_p = model(
            inputs_embeds=inputs_embeds + perturb,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        l_cons_pert = weighted_all_adjacent_consistency_loss(
            out_p.hidden_states,
            pair_weights=pair_weights,
            prompt_skip=prompt_skip,
        )
        l_pv = multi_pv_push_away_loss(
            out_p.hidden_states,
            attention_mask,
            pv_bank_by_t,
            pair_weights,
            pv_distance=pv_distance,
            l2_dist_max=l2_dist_max,
            normalize_hv_for_pv=normalize_hv_for_pv,
            pv_margin=pv_margin,
        )

        total = float(consistency_weight) * l_cons_pert
        lm_loss = torch.zeros((), device=dev, dtype=torch.float32)
        lm_out = None
        if add_lm_loss:
            labels = input_ids.clone()
            labels = labels.masked_fill(attention_mask == 0, -100)
            lm_out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                use_cache=False,
                return_dict=True,
            )
            if lm_out.loss is not None:
                lm_loss = lm_out.loss.to(torch.float32)
                total = total + lm_loss
        if float(beta_pv) != 0.0:
            total = total + float(beta_pv) * l_pv

        total.backward()
        del out_p
        if lm_out is not None:
            del lm_out
        opt.step()

        if dev.type == "cpu":
            gc.collect()

        if step % max(1, int(log_every)) == 0 or step == max_steps - 1:
            print(
                f"[crow-style] step={step} total={float(total.item()):.4f} "
                f"cons_pert={float(l_cons_pert.item()):.4f} "
                f"l_pv={float(l_pv.item()):.4f} "
                f"lm={float(lm_loss.item()):.4f}"
            )


def run_offline(
    *,
    base_model: str = DEFAULT_BASE_MODEL,
    lora_path: str = DEFAULT_LORA_PATH,
    report_path: str = DEFAULT_REPORT_PATH,
    save_dir: str = DEFAULT_SAVE_DIR,
    device: Optional[str] = None,
    batch_size: int = 4,
    max_length: int = 64,
    max_steps: int = 300,
    lr: float = 1e-5,
    epsilon: float = 0.1,
    consistency_weight: float = 5.5,
    base_pair_weight: float = 1.0,
    target_extra_weight: float = 3.0,
    add_lm_loss: bool = True,
    beta_pv: float = 0.0,
    pv_distance: str = "mse",
    l2_dist_max: Optional[float] = 1.4,
    normalize_hv_for_pv: bool = False,
    pv_margin: Optional[float] = 1.5,
    log_every: int = 10,
    post_train_monitor: bool = True,
    monitor_save_path: str = DEFAULT_MONITOR_SAVE,
    monitor_method: str = "lms",
    monitor_epochs: int = 15,
    post_train_device: str = "auto",
) -> str:
    if not os.path.isfile(report_path):
        raise FileNotFoundError(f"report not found: {report_path}")
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")

    tok = AutoTokenizer.from_pretrained(base_model, local_files_only=True, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        dtype=torch.bfloat16 if dev == "cuda" else torch.float32,
        device_map="auto" if dev == "cuda" else None,
        local_files_only=True,
        trust_remote_code=True,
    )
    if dev != "cuda":
        base = base.to(dev)
    model = PeftModel.from_pretrained(base, lora_path, is_trainable=True, local_files_only=True).to(dev)
    model.train()

    ds = OfflineDataset(tok, _OFFLINE_TEXTS, max_length=max_length)
    dl = DataLoader(ds, batch_size=max(1, min(int(batch_size), len(ds))), shuffle=True, drop_last=False)

    train_crow_style_defense(
        model,
        dl,
        report_path=report_path,
        device=dev,
        lr=lr,
        max_steps=max_steps,
        epsilon=epsilon,
        consistency_weight=consistency_weight,
        prompt_skip=0,
        base_pair_weight=base_pair_weight,
        target_extra_weight=target_extra_weight,
        add_lm_loss=add_lm_loss,
        beta_pv=beta_pv,
        pv_distance=pv_distance,
        l2_dist_max=l2_dist_max,
        normalize_hv_for_pv=normalize_hv_for_pv,
        pv_margin=pv_margin,
        log_every=log_every,
    )

    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir, safe_serialization=True)
    tok.save_pretrained(save_dir)
    print(f"[crow-style] saved -> {save_dir}")

    if post_train_monitor:

        def _resolve_post_train_device(pref: str) -> str:
            p = (pref or "auto").strip().lower()
            if p == "cuda":
                if not torch.cuda.is_available():
                    print(
                        "[crow-style] post-train: 请求 cuda 但当前不可用，改用 cpu（加载会很慢）。"
                    )
                    return "cpu"
                return "cuda"
            if p == "cpu":
                return "cpu"
            return "cuda" if torch.cuda.is_available() else "cpu"

        monitor_device = _resolve_post_train_device(post_train_device)
        # 训练后自动产出一份 pv_monitor 规则，便于直接做 defend 筛选。
        margs = SimpleNamespace(
            base_model=base_model,
            lora=save_dir,             # 用清洗后的 LoRA 训练 monitor 头
            report=report_path,
            save_path=monitor_save_path,
            device=monitor_device,
            offline=True,              # 对齐当前离线流程
            batch_size=max(1, int(batch_size)),
            neg_batches=48,
            epochs=int(monitor_epochs),
            lr=1e-3,
            pos_noise_std=0.01,
            pos_aug_per_pv=8,
            monitor_method=monitor_method,
        )
        print(
            f"[crow-style] post-train pv_monitor building... "
            f"method={monitor_method} save={monitor_save_path} device={monitor_device}"
        )
        if monitor_device == "cpu":
            print(
                "[crow-style] 提示：post-train 在 CPU 上需再加载整模，进度条到 100% 可能要数分钟；"
                "有 NVIDIA GPU 时请用 --post-train-device cuda，或训练后单独："
                "python pv_monitor.py train ...；也可加 --no-post-train-monitor 跳过。"
            )
        run_pv_monitor_train(margs)
        print(
            "[crow-style] 可直接防御推理（与本次 save_dir / monitor 一致，默认 v5 流程）：\n"
            f"python pv_monitor.py defend --base-model \"{base_model}\" --lora \"{save_dir}\" "
            f"--monitor-path \"{monitor_save_path}\" --device cuda "
            "--sanitize projection --lms-use-last-layer-screening "
            "--projection-basis-scope last-aligned --lms-match-threshold 0.5 "
            "--projection-gamma 0.15 --projection-layer-window 0 --projection-repeat 1"
        )
    return save_dir


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CROW-style consistency defense runner")
    p.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    p.add_argument("--lora-path", default=DEFAULT_LORA_PATH)
    p.add_argument("--report-path", default=DEFAULT_REPORT_PATH)
    p.add_argument("--save-dir", default=DEFAULT_SAVE_DIR)
    p.add_argument("--device", default=None)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-length", type=int, default=64)
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--epsilon", type=float, default=0.1)
    p.add_argument("--consistency-weight", type=float, default=5.5)
    p.add_argument("--base-pair-weight", type=float, default=1.0)
    p.add_argument("--target-extra-weight", type=float, default=3.0)
    p.add_argument("--beta-pv", type=float, default=0.0, help="多 PV 推远项权重：total += beta_pv * l_pv")
    p.add_argument("--pv-distance", choices=("mse", "l2", "cos"), default="mse")
    p.add_argument("--l2-dist-max", type=float, default=1.4)
    p.add_argument("--normalize-hv-for-pv", action="store_true")
    p.add_argument(
        "--pv-margin",
        type=float,
        default=1.5,
        help="多 PV 推远的 margin（默认启用 margin 版；设为负值可近似关闭）。",
    )
    p.add_argument("--no-lm-loss", action="store_true")
    p.add_argument(
        "--cpu-low-memory",
        action="store_true",
        help="面向 CPU 训练：batch 强制为 1、序列最长 48、关闭 LM 额外前向（仍保留一致性+FGSM），显著降低内存峰值。",
    )
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--no-post-train-monitor", action="store_true", help="训练后不自动训练 pv_monitor 规则。")
    p.add_argument(
        "--post-train-device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="post-train pv_monitor 所用设备：auto=有 GPU 则用 cuda（远快于 cpu），否则 cpu。",
    )
    p.add_argument("--monitor-save-path", default=DEFAULT_MONITOR_SAVE)
    p.add_argument("--monitor-method", choices=("lms", "mlp"), default="lms")
    p.add_argument("--monitor-epochs", type=int, default=15)
    return p.parse_args()


def main() -> None:
    a = parse_args()
    batch_size = max(1, int(a.batch_size))
    max_length = int(a.max_length)
    add_lm = not a.no_lm_loss
    if getattr(a, "cpu_low_memory", False):
        batch_size = 1
        max_length = min(max_length, 48)
        add_lm = False
        print(
            "[crow-style] --cpu-low-memory：batch_size=1, max_length<=48, add_lm_loss=False "
            "（若仍 OOM 请加 --no-post-train-monitor 或换 GPU）"
        )
    run_offline(
        base_model=a.base_model,
        lora_path=a.lora_path,
        report_path=a.report_path,
        save_dir=a.save_dir,
        device=a.device,
        batch_size=batch_size,
        max_length=max_length,
        max_steps=a.max_steps,
        lr=a.lr,
        epsilon=a.epsilon,
        consistency_weight=a.consistency_weight,
        base_pair_weight=a.base_pair_weight,
        target_extra_weight=a.target_extra_weight,
        add_lm_loss=add_lm,
        beta_pv=a.beta_pv,
        pv_distance=a.pv_distance,
        l2_dist_max=a.l2_dist_max,
        normalize_hv_for_pv=a.normalize_hv_for_pv,
        pv_margin=a.pv_margin,
        log_every=a.log_every,
        post_train_monitor=not a.no_post_train_monitor,
        monitor_save_path=a.monitor_save_path,
        monitor_method=a.monitor_method,
        monitor_epochs=a.monitor_epochs,
        post_train_device=a.post_train_device,
    )


if __name__ == "__main__":
    main()
