"""
基于 report 中的 PV 黑名单做逐层监控；默认同 LMSanitator（ICLR 论文实现）一致：

- **lms（默认）**：对齐 `LMSanitator-main/.../defense_monitor.py` 的 `monitor_output`：
  将隐藏向量与 PV 库按元素符号二值化（≥0→+1，否则 -1），对每条 PV 数一致维度比例，
  取最大比例；若 > 阈值（默认 0.8）则判为触发。**不改编输入文本**（论文无滑窗删词）。
- **mlp（可选）**：旧版小 MLP 二分类头（需负样本构造 x,y）。

用法示例：
python pv_monitor.py train --offline
python pv_monitor.py train --monitor-method mlp --offline
python pv_monitor.py defend --sanitize none
"""
from __future__ import annotations

import argparse
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import PeftModel
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from cleanse import (
    _pool_hidden_for_pv,
    _resolve_pv_to_target_layer,
    load_all_run_detect_report_entries,
)


BASE_ROOT = "D:/cleanse"
DEFAULT_BASE = os.path.join(BASE_ROOT, "models/Qwen2.5-3B-Instruct")
# defend/train 默认：与 crow_style 当前输出 cleansed_crow_style_v5 + pv_monitor_crow_style5.pt 对齐
DEFAULT_LORA = os.path.join(BASE_ROOT, "models/cleansed_crow_style_v5")
DEFAULT_REPORT = os.path.join(
    BASE_ROOT, "reports", "threat_report_poisoned_lora_v4.json"
)
DEFAULT_SAVE = os.path.join(BASE_ROOT, "reports", "pv_monitor_crow_style5.pt")

_OFFLINE_TEXTS: List[str] = [
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


def resolve_trigger_path(p: str, report_path: str) -> str:
    raw = (p or "").strip()
    if not raw:
        return raw
    if os.path.isabs(raw) and os.path.isfile(raw):
        return os.path.normpath(raw)
    report_abs = os.path.abspath(report_path)
    report_dir = os.path.dirname(report_abs)
    parent_dir = os.path.dirname(report_dir)
    clean = raw.replace("\\", "/").lstrip("./")
    candidates: List[str] = []
    for base in (os.getcwd(), report_dir, parent_dir):
        candidates.append(os.path.normpath(os.path.join(base, raw)))
        candidates.append(os.path.normpath(os.path.join(base, clean)))
    for c in candidates:
        if os.path.isfile(c):
            return c
    return os.path.abspath(raw)


class CleanTextDataset(Dataset):
    def __init__(self, tokenizer, max_length: int = 64, offline: bool = False):
        self.tokenizer = tokenizer
        self.max_length = max_length
        if offline:
            self.texts = _OFFLINE_TEXTS
        else:
            from datasets import load_dataset

            raw = load_dataset("shibing624/alpaca-zh", split="train")
            self.texts = [x["instruction"] for x in raw if len(x["instruction"]) > 10][:2000]
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        t = self.tokenizer(
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


class LayerMonitor(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, d // 4),
            nn.ReLU(),
            nn.Linear(d // 4, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def _lms_sign_vec(x: torch.Tensor) -> torch.Tensor:
    """与 LMSanitator 一致：x>=0 → +1，否则 -1。"""
    return torch.where(x >= 0, 1.0, -1.0)


def lms_monitor_output(
    output: torch.Tensor,
    pv_bank: torch.Tensor,
    threshold: float = 0.8,
) -> torch.Tensor:
    """
    与 LMSanitator ``model/defense_monitor.py::monitor_output`` 同逻辑。
    ``output``: [B, H]；``pv_bank``: [K, H]。
    返回 logits [B, 2]；argmax==1 表示「与某条 PV 符号高度一致」，即判触发。
    """
    if output.dim() != 2 or pv_bank.dim() != 2:
        raise ValueError("lms_monitor_output expects output [B,H] and pv_bank [K,H].")
    bsz, hidden_size = output.shape[0], output.shape[1]
    if pv_bank.shape[1] != hidden_size:
        raise ValueError(
            f"hidden_size mismatch: output {hidden_size} vs pv_bank {pv_bank.shape[1]}"
        )
    dev = output.device
    dtype = torch.float32
    out = output.to(dtype)
    pv = pv_bank.to(device=dev, dtype=dtype)
    simplified_output = _lms_sign_vec(out).unsqueeze(1)
    simplified_pv = _lms_sign_vec(pv).unsqueeze(0)
    ratios = (simplified_output == simplified_pv).float().sum(dim=-1) / float(hidden_size)
    max_rat = ratios.max(dim=1).values
    logits = torch.zeros(bsz, 2, device=dev, dtype=dtype)
    hi = max_rat > float(threshold)
    logits[hi] = torch.tensor([-1.0, 1.0], device=dev, dtype=dtype)
    logits[~hi] = torch.tensor([1.0, -1.0], device=dev, dtype=dtype)
    return logits


def lms_max_match_ratio(output_1d: torch.Tensor, pv_bank: torch.Tensor) -> float:
    v = output_1d.flatten().to(torch.float32)
    pb = pv_bank.to(device=v.device, dtype=torch.float32)
    if pb.shape[1] != v.numel():
        raise ValueError("dim mismatch in lms_max_match_ratio.")
    so = _lms_sign_vec(v)
    sp = _lms_sign_vec(pb)
    ratios = (sp == so.unsqueeze(0)).float().sum(dim=1) / float(v.numel())
    return float(ratios.max().item())


def build_train_xy(
    model: nn.Module,
    dataloader,
    *,
    layer_t: int,
    pv_vecs: Sequence[torch.Tensor],
    device: torch.device,
    neg_batches: int,
    pos_noise_std: float,
    pos_aug_per_pv: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    neg_feats: List[torch.Tensor] = []
    it = iter(dataloader)
    model.eval()
    with torch.no_grad():
        for _ in range(max(1, neg_batches)):
            try:
                b = next(it)
            except StopIteration:
                break
            ids = b["input_ids"].to(device)
            am = b["attention_mask"].to(device)
            out = model(
                input_ids=ids,
                attention_mask=am,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
            h = _pool_hidden_for_pv(out.hidden_states, am, layer_t, prompt_length=0)
            neg_feats.append(h.to(torch.float32).detach().cpu())
    if not neg_feats:
        raise RuntimeError("No negative features collected.")
    x_neg = torch.cat(neg_feats, dim=0)

    pos_feats: List[torch.Tensor] = []
    for v in pv_vecs:
        vv = v.to(torch.float32).detach().cpu().flatten().unsqueeze(0)
        pos_feats.append(vv)
        for _ in range(max(0, pos_aug_per_pv)):
            noise = torch.randn_like(vv) * float(pos_noise_std)
            pos_feats.append(vv + noise)
    x_pos = torch.cat(pos_feats, dim=0)

    x = torch.cat([x_neg, x_pos], dim=0)
    y = torch.cat(
        [
            torch.zeros(x_neg.size(0), dtype=torch.float32),
            torch.ones(x_pos.size(0), dtype=torch.float32),
        ],
        dim=0,
    )
    return x, y


def parse_args():
    p = argparse.ArgumentParser(
        description="PV blacklist monitor: train & defend (LMS-style)."
    )
    sub = p.add_subparsers(dest="mode", required=True)

    # train subcommand
    p_train = sub.add_parser("train", help="Train PV monitor heads from report.")
    p_train.add_argument("--base-model", default=DEFAULT_BASE)
    p_train.add_argument("--lora", default=DEFAULT_LORA)
    p_train.add_argument("--report", default=DEFAULT_REPORT)
    p_train.add_argument("--save-path", default=DEFAULT_SAVE)
    p_train.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p_train.add_argument("--offline", action="store_true")
    p_train.add_argument("--batch-size", type=int, default=16)
    p_train.add_argument("--neg-batches", type=int, default=48)
    p_train.add_argument("--epochs", type=int, default=15)
    p_train.add_argument("--lr", type=float, default=1e-3)
    p_train.add_argument("--pos-noise-std", type=float, default=0.01)
    p_train.add_argument("--pos-aug-per-pv", type=int, default=8)
    p_train.add_argument(
        "--monitor-method",
        choices=("lms", "mlp"),
        default="lms",
        help="lms: LMSanitator 符号一致率监控（默认）；mlp: 负样本+二分类头。",
    )

    # defend subcommand
    p_def = sub.add_parser("defend", help="Run LMS-style PV-based defense for one input.")
    p_def.add_argument("--base-model", default=DEFAULT_BASE)
    p_def.add_argument("--lora", default=DEFAULT_LORA)
    p_def.add_argument("--monitor-path", default=DEFAULT_SAVE)
    p_def.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p_def.add_argument(
        "--offline",
        action="store_true",
        help="与 train 用法对齐；defend 本身只用 local_files_only 加载权重，无额外行为。",
    )
    p_def.add_argument("--prompt", default=None, help="Input text; 若为空则从 stdin 读取一行。")
    p_def.add_argument(
        "--risk-threshold",
        type=float,
        default=0.5,
        help="仅 mlp 模式：sigmoid 概率 >= 该值视为触发。",
    )
    p_def.add_argument(
        "--lms-match-threshold",
        type=float,
        default=0.8,
        help="仅 lms 模式：与 LMSanitator monitor_output 一致，符号一致率 > 该值判触发。",
    )
    p_def.add_argument(
        "--sanitize",
        choices=("none", "sliding", "projection"),
        default="none",
        help="none: 不改编输入；sliding: 旧版滑窗删 token；projection: 触发后在生成时削弱 PV 子空间分量。",
    )
    p_def.add_argument(
        "--projection-gamma",
        type=float,
        default=1.0,
        help="projection 模式下的削弱强度，h <- h - gamma * proj_pv(h)。",
    )
    p_def.add_argument(
        "--projection-layer-window",
        type=int,
        default=0,
        help="projection 模式：对监控层前后各扩展多少层一起去投影（0=仅监控层）。",
    )
    p_def.add_argument(
        "--projection-repeat",
        type=int,
        default=1,
        help="projection 模式：生成时重复去投影次数（>=1）。",
    )
    p_def.add_argument(
        "--projection-force-on-trigger-word",
        default=None,
        help="逗号分隔关键词；输入包含任一关键词时，projection 模式强制启用（可绕过漏检）。",
    )
    p_def.add_argument(
        "--projection-basis-scope",
        choices=("report", "last-aligned"),
        default="report",
        help=(
            "report: 按报告层位在对应 decoder 层去投影（默认，可能与 last-layer 筛选不一致）。"
            "last-aligned: 合并全部 pv_bank 后仅在最后一层 hidden 对应 block 上投影，"
            "与 --lms-use-last-layer-screening 同空间，误报时对语义破坏通常更小。"
        ),
    )
    p_def.add_argument(
        "--lms-use-last-layer-screening",
        action="store_true",
        help="仅筛选打分时：lms 统一改用最后一层 hidden（不依赖报告层位）。",
    )
    p_def.add_argument(
        "--window-size",
        type=int,
        default=3,
        help="滑动窗口大小（token 数）。",
    )
    p_def.add_argument(
        "--max-iterations",
        type=int,
        default=10,
        help="最多尝试多少轮滑窗净化。",
    )
    return p.parse_args()


def run_train(args):
    report = os.path.abspath(args.report)
    if not os.path.isfile(report):
        raise FileNotFoundError(f"report not found: {report}")

    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model, local_files_only=True, trust_remote_code=True
    )
    # 单卡勿用 device_map="auto"：显存吃紧时层会落到 CPU/meta，PeftModel.load_adapter 需 offload_dir。
    # 整模 .to(device) 与 crow_style 主训练一致。
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        device_map=None,
        local_files_only=True,
        trust_remote_code=True,
    )
    base_model = base_model.to(device)
    model = PeftModel.from_pretrained(
        base_model, args.lora, is_trainable=False, local_files_only=True
    ).to(device)
    model.eval()

    dl = None
    if args.monitor_method == "mlp":
        ds = CleanTextDataset(tokenizer, offline=args.offline)
        if len(ds) == 0:
            raise RuntimeError("Clean text dataset is empty.")
        bs_eff = min(int(args.batch_size), len(ds))
        dl = DataLoader(ds, batch_size=max(1, bs_eff), shuffle=True, drop_last=False)

    entries = load_all_run_detect_report_entries(report)
    grouped: Dict[int, List[dict]] = defaultdict(list)
    for e in entries:
        grouped[int(e["poisoned_layer"])].append(e)

    save_obj: Dict[str, object] = {
        "meta": {
            "base_model": args.base_model,
            "lora": args.lora,
            "report": report,
            "monitor_method": args.monitor_method,
        },
        "layers": {},
    }

    for raw_layer in sorted(grouped.keys()):
        t = raw_layer + 1
        rows = grouped[raw_layer]
        pv_vecs: List[torch.Tensor] = []
        for r in rows:
            p = resolve_trigger_path(str(r["trigger_vector_path"]), report)
            if not os.path.isfile(p):
                continue
            v = _resolve_pv_to_target_layer(
                model,
                device,
                t,
                True,
                pv_target_path=p,
            )
            pv_vecs.append(v.detach())
        if not pv_vecs:
            continue

        if args.monitor_method == "lms":
            bank = torch.stack([v.reshape(-1).float().cpu() for v in pv_vecs], dim=0)
            d = int(bank.shape[-1])
            save_obj["layers"][str(raw_layer)] = {
                "method": "lms",
                "effective_t": int(t),
                "num_pv": int(len(pv_vecs)),
                "input_dim": d,
                "pv_bank": bank,
            }
            print(
                f"[monitor] raw_layer={raw_layer} t={t} "
                f"LMSanitator pv_bank K={len(pv_vecs)} d={d}"
            )
            continue

        assert dl is not None
        x, y = build_train_xy(
            model,
            dl,
            layer_t=t,
            pv_vecs=pv_vecs,
            device=device,
            neg_batches=args.neg_batches,
            pos_noise_std=args.pos_noise_std,
            pos_aug_per_pv=args.pos_aug_per_pv,
        )
        perm = torch.randperm(x.size(0))
        x = x[perm]
        y = y[perm]
        x = F.normalize(x, p=2, dim=-1, eps=1e-8).to(device)
        y = y.to(device)

        head = LayerMonitor(x.size(-1)).to(device)
        opt = torch.optim.Adam(head.parameters(), lr=args.lr)
        bce = nn.BCEWithLogitsLoss()

        for ep in range(args.epochs):
            opt.zero_grad(set_to_none=True)
            logits = head(x)
            loss = bce(logits, y)
            loss.backward()
            opt.step()
            if ep % 5 == 0 or ep == args.epochs - 1:
                with torch.no_grad():
                    pred = (torch.sigmoid(logits) > 0.5).to(y.dtype)
                    acc = (pred == y).float().mean().item()
                print(
                    f"[monitor] raw_layer={raw_layer} t={t} ep={ep} "
                    f"loss={loss.item():.4f} acc={acc:.4f} n={x.size(0)}"
                )

        save_obj["layers"][str(raw_layer)] = {
            "method": "mlp",
            "effective_t": int(t),
            "num_pv": int(len(pv_vecs)),
            "input_dim": int(x.size(-1)),
            "state_dict": {k: v.detach().cpu() for k, v in head.state_dict().items()},
        }

    os.makedirs(os.path.dirname(os.path.abspath(args.save_path)), exist_ok=True)
    torch.save(save_obj, args.save_path)
    print(f"[monitor] saved -> {args.save_path}")


def load_monitor_bundle(path: str, device: torch.device) -> Dict[str, object]:
    """返回 ``mlp`` / ``lms`` 两层配置（可同时为空）。旧版仅有 state_dict 的计入 mlp。"""
    obj = torch.load(path, map_location="cpu")
    layers_cfg: Dict[str, dict] = obj.get("layers", {})
    mlp_layers: Dict[int, dict] = {}
    lms_layers: Dict[int, dict] = {}
    for k, v in layers_cfg.items():
        raw_layer = int(k)
        has_sd = bool(v.get("state_dict"))
        has_bank = v.get("pv_bank") is not None
        eff_t = int(v.get("effective_t", raw_layer + 1))
        method = v.get("method")
        if method == "lms" or (has_bank and not has_sd):
            lms_layers[raw_layer] = {
                "effective_t": eff_t,
                "pv_bank": v["pv_bank"].float().to(device),
            }
        elif has_sd:
            d = int(v["input_dim"])
            head = LayerMonitor(d)
            head.load_state_dict(v["state_dict"])
            head.to(device)
            head.eval()
            mlp_layers[raw_layer] = {"head": head, "effective_t": eff_t}
    return {"mlp": mlp_layers, "lms": lms_layers, "meta": obj.get("meta", {})}


def monitor_risk_for_text(
    model: nn.Module,
    tokenizer,
    bundle: Dict[str, object],
    text: str,
    device: torch.device,
    lms_force_hidden_index: Optional[int] = None,
) -> Tuple[float, str]:
    """
    返回 (score, kind)。
    lms：score 为最大符号一致率 ∈[0,1]，与 LMSanitator 一致，**不对 h 做 L2 归一化**。
    mlp：score 为各层 sigmoid 的最大值。
    """
    lms_layers: Dict[int, dict] = bundle["lms"]  # type: ignore[assignment]
    mlp_layers: Dict[int, dict] = bundle["mlp"]  # type: ignore[assignment]
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(device)
    model.eval()
    with torch.no_grad():
        out = model(
            **enc,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        hidden_states = out.hidden_states
    if lms_layers:
        ratios: List[float] = []
        for _raw, cfg in lms_layers.items():
            t = int(lms_force_hidden_index) if lms_force_hidden_index is not None else int(cfg["effective_t"])
            if t < 0 or t >= len(hidden_states):
                continue
            h = _pool_hidden_for_pv(
                hidden_states, enc["attention_mask"], t, prompt_length=0
            )
            h = h.to(torch.float32)
            if h.dim() > 1:
                h = h.squeeze(0)
            ratios.append(lms_max_match_ratio(h, cfg["pv_bank"]))
        return (max(ratios) if ratios else 0.0, "lms")
    probs: List[float] = []
    for _raw, cfg in mlp_layers.items():
        head = cfg["head"]
        t = int(cfg["effective_t"])
        if t < 0 or t >= len(hidden_states):
            continue
        h = _pool_hidden_for_pv(
            hidden_states, enc["attention_mask"], t, prompt_length=0
        )
        h = F.normalize(h.to(torch.float32), p=2, dim=-1, eps=1e-8)
        logit = head(h).mean()
        probs.append(torch.sigmoid(logit).item())
    return (max(probs) if probs else 0.0, "mlp")


def _is_safe_score(score: float, kind: str, thr_mlp: float, thr_lms: float) -> bool:
    if kind == "lms":
        return score <= thr_lms
    return score < thr_mlp


def sliding_window_sanitize(
    model: nn.Module,
    tokenizer,
    bundle: Dict[str, object],
    text: str,
    device: torch.device,
    risk_threshold: float,
    lms_match_threshold: float,
    window_size: int,
    max_iterations: int,
    lms_force_hidden_index: Optional[int] = None,
) -> str:
    """旧版滑窗删 token；阈值语义与 ``monitor_risk_for_text`` 一致。"""
    cur_text = text
    base_score, kind = monitor_risk_for_text(
        model, tokenizer, bundle, cur_text, device, lms_force_hidden_index=lms_force_hidden_index
    )
    thr_m = lms_match_threshold if kind == "lms" else risk_threshold
    print(f"[defend] sanitize=sliding monitor={kind} score={base_score:.4f} thr={thr_m:.4f}")
    if _is_safe_score(base_score, kind, risk_threshold, lms_match_threshold):
        return cur_text

    for it in range(max_iterations):
        enc = tokenizer(cur_text, return_tensors="pt", truncation=True, max_length=256)
        ids = enc["input_ids"][0]
        n = ids.size(0)
        best_text = cur_text
        best_score = base_score

        for start in range(0, max(1, n - window_size + 1)):
            end = min(n, start + window_size)
            kept = torch.cat([ids[:start], ids[end:]], dim=0)
            if kept.numel() == 0:
                continue
            alt_text = tokenizer.decode(kept, skip_special_tokens=True)
            score, k2 = monitor_risk_for_text(
                model, tokenizer, bundle, alt_text, device, lms_force_hidden_index=lms_force_hidden_index
            )
            if k2 != kind:
                continue
            if score < best_score:
                best_score = score
                best_text = alt_text

        print(
            f"[defend] iter={it} before={base_score:.4f} after={best_score:.4f} "
            f"changed={best_text != cur_text}"
        )
        cur_text = best_text
        base_score = best_score
        if _is_safe_score(base_score, kind, risk_threshold, lms_match_threshold):
            break

    return cur_text


def _build_projection_bases(
    bundle: Dict[str, object],
    device: torch.device,
    layer_window: int = 0,
) -> Dict[int, torch.Tensor]:
    """
    从 lms pv_bank 构造每层正交基 B:[k,d]（行向量单位化+QR），用于投影 proj(x)=((xB^T)B)。
    key 使用 effective_t（hidden_states 下标）。
    """
    lms_layers: Dict[int, dict] = bundle["lms"]  # type: ignore[assignment]
    bases: Dict[int, torch.Tensor] = {}
    win = max(0, int(layer_window))
    for _raw, cfg in lms_layers.items():
        t = int(cfg["effective_t"])
        bank = cfg["pv_bank"].to(device=device, dtype=torch.float32)  # [K,D]
        if bank.dim() != 2 or bank.numel() == 0:
            continue
        bank = F.normalize(bank, p=2, dim=-1, eps=1e-8)
        # rows->orthonormal basis in row-space
        q, _ = torch.linalg.qr(bank.T, mode="reduced")  # [D,r]
        b = q.T.contiguous()  # [r,D]
        if b.numel() > 0:
            for dt in range(-win, win + 1):
                tt = t + dt
                if tt < 1:
                    continue
                # 多个源层扩展到同一目标层时，拼接后再正交化
                if tt in bases:
                    cat = torch.cat([bases[tt], b], dim=0)
                    q2, _ = torch.linalg.qr(cat.T, mode="reduced")
                    bases[tt] = q2.T.contiguous()
                else:
                    bases[tt] = b
    return bases


def _build_merged_pv_basis_for_last_layer(
    bundle: Dict[str, object],
    device: torch.device,
) -> Optional[torch.Tensor]:
    """
    合并 bundle 内全部 lms pv_bank，QR 得到单一正交基 [r,D]。
    用于与「最后一层 hidden 上做 LMS」同一表示空间上投影，避免在中低层误伤主干表示。
    """
    lms_layers: Dict[int, dict] = bundle["lms"]  # type: ignore[assignment]
    chunks: List[torch.Tensor] = []
    for _raw, cfg in lms_layers.items():
        bank = cfg["pv_bank"].to(device=device, dtype=torch.float32)
        if bank.dim() != 2 or bank.numel() == 0:
            continue
        chunks.append(F.normalize(bank, p=2, dim=-1, eps=1e-8))
    if not chunks:
        return None
    cat = torch.cat(chunks, dim=0)
    q, _ = torch.linalg.qr(cat.T, mode="reduced")
    return q.T.contiguous()


def _resolve_decoder_layer_modules(model: nn.Module) -> Dict[int, nn.Module]:
    """
    解析类似 ``...layers.{i}`` 的 decoder block 映射。
    兼容 PeftModel 下的命名；若出现重复索引，保留首次匹配。
    """
    out: Dict[int, nn.Module] = {}
    for name, mod in model.named_modules():
        parts = name.split(".")
        for i in range(len(parts) - 1):
            if parts[i] == "layers" and parts[i + 1].isdigit():
                idx = int(parts[i + 1])
                if idx not in out:
                    out[idx] = mod
    return out


def _generate_with_projection_sanitize(
    model: nn.Module,
    tokenizer,
    prompt: str,
    device: torch.device,
    bundle: Dict[str, object],
    gamma: float,
    layer_window: int = 0,
    repeat: int = 1,
    max_new_tokens: int = 200,
    basis_scope: str = "report",
    last_hidden_t: Optional[int] = None,
) -> torch.Tensor:
    """
    仅在生成时启用：对命中的监控层做 hidden 去 PV 子空间分量。
    不改输入文本，仅改内部表示。
    """
    if basis_scope == "last-aligned":
        if last_hidden_t is None:
            raise ValueError("last-aligned projection requires last_hidden_t (e.g. config.num_hidden_layers)")
        merged = _build_merged_pv_basis_for_last_layer(bundle, device)
        bases = {int(last_hidden_t): merged} if merged is not None else {}
    else:
        bases = _build_projection_bases(bundle, device, layer_window=layer_window)
    layer_mods = _resolve_decoder_layer_modules(model)
    hooks = []
    g = float(gamma)

    for t, b in bases.items():
        layer_idx = t - 1  # hidden_states[t] 对应 decoder block index = t-1
        mod = layer_mods.get(layer_idx)
        if mod is None:
            continue

        def _hook(_module, _inputs, output, basis=b, gg=g):
            if output is None:
                return output
            hs = output[0] if isinstance(output, tuple) else output
            if not torch.is_tensor(hs):
                return output
            x = hs.to(torch.float32)  # [B,S,D]
            bb = basis.to(device=x.device, dtype=x.dtype)  # [r,D]
            coeff = torch.matmul(x, bb.T)                  # [B,S,r]
            proj = torch.matmul(coeff, bb)                 # [B,S,D]
            x_new = (x - gg * proj).to(dtype=hs.dtype)
            if int(repeat) > 1:
                # 轻量重复投影，减少残留分量
                for _ in range(int(repeat) - 1):
                    xx = x_new.to(torch.float32)
                    cc = torch.matmul(xx, bb.T)
                    pp = torch.matmul(cc, bb)
                    x_new = (xx - gg * pp).to(dtype=hs.dtype)
            if isinstance(output, tuple):
                return (x_new, *output[1:])
            return x_new

        hooks.append(mod.register_forward_hook(_hook))

    try:
        enc = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        return out
    finally:
        for h in hooks:
            h.remove()


def run_defend(args):
    monitor_path = os.path.abspath(args.monitor_path)
    if not os.path.isfile(monitor_path):
        raise FileNotFoundError(f"monitor not found: {monitor_path}")
    device = torch.device(args.device)

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model, local_files_only=True, trust_remote_code=True
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        device_map=None,
        local_files_only=True,
        trust_remote_code=True,
    )
    base_model = base_model.to(device)
    model = PeftModel.from_pretrained(
        base_model, args.lora, is_trainable=False, local_files_only=True
    ).to(device)
    model.eval()

    bundle = load_monitor_bundle(monitor_path, device)
    if not bundle["mlp"] and not bundle["lms"]:
        raise RuntimeError("No monitor layers found in monitor file (need mlp or lms).")

    if args.prompt is None:
        try:
            print("请输入一行待防御的文本（按 Enter 结束）：")
            raw = input().strip()
        except EOFError:
            raw = ""
    else:
        raw = str(args.prompt).strip()
    if not raw:
        print("[defend] 空输入，直接退出。")
        return

    lms_force_t = None
    if args.lms_use_last_layer_screening:
        lms_force_t = int(model.config.num_hidden_layers)
        print(f"[defend] lms screening uses last hidden index t={lms_force_t}")

    score, kind = monitor_risk_for_text(
        model, tokenizer, bundle, raw, device, lms_force_hidden_index=lms_force_t
    )
    if kind == "lms":
        triggered = score > float(args.lms_match_threshold)
        thr_note = f"lms_match_threshold={args.lms_match_threshold}"
    else:
        triggered = score >= float(args.risk_threshold)
        thr_note = f"risk_threshold={args.risk_threshold}"
    force_words = [w.strip() for w in str(args.projection_force_on_trigger_word or "").split(",") if w.strip()]
    force_hit = any(w in raw for w in force_words)
    if args.sanitize == "projection" and force_hit:
        triggered = True
        thr_note += f", force_word_hit={force_words}"
    print(
        f"[defend] LMSanitator对齐监控: method={kind} score={score:.4f} "
        f"triggered={triggered} ({thr_note})"
    )

    defended = raw
    if args.sanitize == "sliding":
        if kind == "lms":
            print(
                "[defend] 提示：滑窗会动输入文本；论文中无此步，优先用 --sanitize none。"
            )
        defended = sliding_window_sanitize(
            model,
            tokenizer,
            bundle,
            raw,
            device,
            risk_threshold=args.risk_threshold,
            lms_match_threshold=args.lms_match_threshold,
            window_size=max(1, int(args.window_size)),
            max_iterations=max(1, int(args.max_iterations)),
            lms_force_hidden_index=lms_force_t,
        )
    elif args.sanitize == "projection":
        if not triggered:
            print("[defend] sanitize=projection：未触发，按原路径生成。")
        elif not bundle["lms"]:
            print("[defend] sanitize=projection：monitor 文件无 lms pv_bank，退化为 none。")
        else:
            pscope = str(args.projection_basis_scope)
            print(
                f"[defend] sanitize=projection：触发后将启用表示层去投影 "
                f"(basis_scope={pscope}, gamma={float(args.projection_gamma):.3f}, "
                f"window={int(args.projection_layer_window)}, repeat={int(args.projection_repeat)})。"
            )
    elif triggered:
        print(
            "[defend] sanitize=none：输入未修改；生成仍经当前 LoRA（仅提示命中监控）。"
        )

    print("\n[defend] 净化前：")
    print(raw)
    print("\n[defend] 净化后：")
    print(defended)
    print("\n[defend] 下游模型生成：")
    prompt = f"<|im_start|>user\n{defended}<|im_end|>\n<|im_start|>assistant\n"
    if args.sanitize == "projection" and triggered and bundle["lms"]:
        pscope = str(args.projection_basis_scope)
        last_t = (
            int(model.config.num_hidden_layers) if pscope == "last-aligned" else None
        )
        out = _generate_with_projection_sanitize(
            model,
            tokenizer,
            prompt,
            device,
            bundle,
            gamma=float(args.projection_gamma),
            layer_window=max(0, int(args.projection_layer_window)),
            repeat=max(1, int(args.projection_repeat)),
            max_new_tokens=200,
            basis_scope=pscope,
            last_hidden_t=last_t,
        )
    else:
        enc = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=200,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

    resp = tokenizer.decode(out[0], skip_special_tokens=False)
    if "<|im_start|>assistant\n" in resp:
        resp = resp.split("<|im_start|>assistant\n")[-1].replace("<|im_end|>", "")
    print(resp)


def main():
    args = parse_args()
    if args.mode == "train":
        run_train(args)
    elif args.mode == "defend":
        run_defend(args)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()

