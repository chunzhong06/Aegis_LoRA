# Aegis-LoRA: 报告生成模块
#   1. 深度免疫重构报告：export_offline_report；
#   2. 极速免疫清洗报告：export_fast_cleanse_report；
#   3. 静态权重探测器报告：export_detector_report。
import os
import io
import json
import html
import base64
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

# 设置 Matplotlib 全局字体为中文，确保报告中的中文字符能够正确显示。
plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "PingFang SC",
    "sans-serif",
]
plt.rcParams["axes.unicode_minus"] = False


# ====================================================
# Base64 图表生成函数
# ====================================================
def _fig_to_base64(fig):
    """
    将 Matplotlib 图表对象转换为 Base64 编码的 PNG 图片字符串，以便嵌入 HTML 报告中。
    """
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150, transparent=True)
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return img_str


# ====================================================
# 清洗结构聚合函数
# ====================================================
def _aggregate_surgery_rows(
    suppressed_dict,
    norms_before,
    norms_after,
    target_attention_heads=None,
):
    """将 projection 级手术统计聚合为完整 MLP 神经元与 Attention head。"""
    grouped = {}
    heads_by_layer = {}

    # Attention 目标按层去重；完整列表会同时保留到 JSON 审计数据中。
    for target in target_attention_heads or []:
        layer = str(target.get("layer", "")).strip()
        if not layer:
            continue

        layer_label = layer if layer.startswith("L") else f"L{layer}"
        try:
            heads_by_layer.setdefault(layer_label, set()).add(int(target["head"]))
        except (KeyError, TypeError, ValueError):
            continue

    # 将同层 gate/up/down 合并为 MLP，将 q/k/v/o 合并为 Attention。
    for metric_key, count in (suppressed_dict or {}).items():
        metric_key = str(metric_key)
        parts = metric_key.split(".")
        layer_label = parts[0] if parts else metric_key
        projection = parts[1] if len(parts) > 1 else ""

        if projection in ("gate_proj", "up_proj", "down_proj"):
            structure = "mlp"
            group_key = f"{layer_label}.mlp"
            label = f"{layer_label} · MLP"
        elif projection in ("q_proj", "k_proj", "v_proj", "o_proj"):
            structure = "attention"
            group_key = f"{layer_label}.attention"
            label = f"{layer_label} · Attention"
        else:
            structure = "module"
            group_key = metric_key
            label = metric_key

        row = grouped.setdefault(
            group_key,
            {
                "key": group_key,
                "label": label,
                "layer": layer_label,
                "structure": structure,
                "count": 0,
                "before_sq": 0.0,
                "after_sq": 0.0,
            },
        )

        before = max(0.0, float((norms_before or {}).get(metric_key, 0.0) or 0.0))
        after_value = (norms_after or {}).get(metric_key)
        after = before if after_value is None else max(0.0, float(after_value))

        # 不同 projection 的 Frobenius 范数按平方和合并，避免直接平均比例失真。
        row["count"] += int(count or 0)
        row["before_sq"] += before**2
        row["after_sq"] += after**2

    surgery_rows = []
    for row in grouped.values():
        before = row.pop("before_sq") ** 0.5
        after = row.pop("after_sq") ** 0.5
        drop = ((before - after) / before) * 100 if before > 0 else 0.0

        row["before_norm"] = before
        row["after_norm"] = after
        row["drop"] = float(np.clip(drop, 0.0, 100.0))
        row["heads"] = (
            sorted(heads_by_layer.get(row["layer"], set()))
            if row["structure"] == "attention"
            else []
        )
        surgery_rows.append(row)

    return surgery_rows


# ====================================================
# 统一图表生成函数
# ====================================================
def _generate_chart(kind, **kwargs):
    """
    统一生成报告图表，并返回 Base64 PNG 字符串。

    支持：
    - kind="surgery"：深度 / 极速清洗共用，展示 Top-15 干预层；
    - kind="confusion_matrix"：探测器报告使用，展示 TN / FP / FN / TP。
    """
    # -----------------------------------------------------------------
    # 图表 1：LoRA 参数清洗效果
    # -----------------------------------------------------------------
    if kind == "surgery":
        # -----------------------------------------------------------------
        # 1. 整理清洗数据
        # -----------------------------------------------------------------
        # 优先使用报告接口预聚合的数据；保留原始参数入口便于内部独立调用。
        surgery_rows = kwargs.get("surgery_rows")
        if surgery_rows is None:
            surgery_rows = _aggregate_surgery_rows(
                kwargs.get("suppressed_dict") or {},
                kwargs.get("norms_before") or {},
                kwargs.get("norms_after") or {},
                kwargs.get("target_attention_heads") or [],
            )

        if not surgery_rows:
            return ""

        # 按整体范数下降率筛选 Top 15，下降率相同时优先展示清零参数更多的结构。
        top_rows = sorted(
            surgery_rows,
            key=lambda row: (row["drop"], row["count"]),
            reverse=True,
        )[:15]
        counts = [row["count"] for row in top_rows]
        drops = [row["drop"] for row in top_rows]
        retained = [100.0 - drop for drop in drops]

        # Attention 行补充实际 head 编号；过多时保留前 6 个并标注剩余数量。
        labels = []
        for row in top_rows:
            label = row["label"]
            heads = row.get("heads") or []
            if row["structure"] == "attention" and heads:
                preview = ", ".join(f"H{head}" for head in heads[:6])
                suffix = f" …(+{len(heads) - 6})" if len(heads) > 6 else ""
                label = f"{label}\n{preview}{suffix}"
            labels.append(label)

        # -----------------------------------------------------------------
        # 2. 绘制清洗前后对比
        # -----------------------------------------------------------------
        # 画布高度随层数增长，避免纵轴标签和数据标注相互拥挤。
        fig_height = max(5.0, 1.8 + len(top_rows) * 0.52)
        fig, ax = plt.subplots(figsize=(10, fig_height))
        y = np.arange(len(top_rows))

        # 将清洗前范数归一化为 100%，用同一尺度展示保留量和削减量。
        ax.barh(
            y,
            retained,
            height=0.62,
            color="#B7D8D1",
            edgecolor="none",
            label="清洗后保留范数",
        )
        ax.barh(
            y,
            drops,
            left=retained,
            height=0.62,
            color="#E76F51",
            edgecolor="none",
            label="已削减范数",
        )

        # 宽色块内直接显示下降率；窄色块将文字移到左侧，避免标注溢出。
        for row_y, remaining, drop, count in zip(y, retained, drops, counts):
            if drop >= 9.0:
                ax.text(
                    remaining + drop / 2,
                    row_y,
                    f"↓ {drop:.1f}%",
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=8.5,
                    fontweight="bold",
                )
            else:
                ax.text(
                    max(1.0, remaining - 1.2),
                    row_y,
                    f"↓ {drop:.1f}%",
                    ha="right",
                    va="center",
                    color="#B64038",
                    fontsize=8.5,
                    fontweight="bold",
                )

            ax.text(
                102.5,
                row_y,
                f"{count:,}",
                ha="left",
                va="center",
                color="#455A64",
                fontsize=8.5,
            )

        # -----------------------------------------------------------------
        # 3. 完善图表信息并输出
        # -----------------------------------------------------------------
        # Top 1 置顶；100% 基准线右侧单独展示各层清零参数数。
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=9)

        # MLP 与 Attention 使用不同标签色，结构类型可在不改变范数配色的情况下快速区分。
        for tick, row in zip(ax.get_yticklabels(), top_rows):
            tick.set_color("#00695C" if row["structure"] == "mlp" else "#1565C0")
            tick.set_fontweight("bold")

        ax.invert_yaxis()
        ax.set_xlim(0, 122)
        ax.set_xticks(np.arange(0, 101, 20))
        ax.set_xticklabels([f"{value}%" for value in range(0, 101, 20)])
        ax.set_xlabel("清洗前范数基准（100%）", fontsize=9.5, color="#546E7A")
        ax.grid(axis="x", linestyle=":", color="#B0BEC5", alpha=0.55)
        ax.set_axisbelow(True)
        ax.axvline(100, color="#90A4AE", linewidth=0.8)
        ax.text(
            102.5,
            -0.9,
            "清零参数",
            ha="left",
            va="center",
            color="#78909C",
            fontsize=8.5,
            fontweight="bold",
        )

        # 标题、副标题和图例共同说明排序方式、色块含义与辅助数值。
        ax.set_title(
            "完整结构清洗前后对比",
            fontsize=13,
            fontweight="bold",
            color="#263238",
            pad=28,
        )
        ax.text(
            0,
            1.02,
            "gate/up/down 聚合为 MLP · q/k/v/o 聚合为 Attention · 右侧为清零参数数",
            transform=ax.transAxes,
            fontsize=9,
            color="#78909C",
        )
        ax.legend(
            loc="lower left",
            bbox_to_anchor=(0, 1.08),
            ncol=2,
            frameon=False,
            fontsize=9,
        )

        # 移除无信息量的边框和刻度线，再转为 Base64 供两类清洗报告复用。
        for spine in ("top", "right", "left", "bottom"):
            ax.spines[spine].set_visible(False)
        ax.tick_params(axis="both", length=0, colors="#546E7A")
        fig.tight_layout(pad=1.2)

        return _fig_to_base64(fig)

    # -----------------------------------------------------------------
    # 图表 2：探测器混淆矩阵
    # -----------------------------------------------------------------
    if kind == "confusion_matrix":
        # -----------------------------------------------------------------
        # 1. 构造预测矩阵
        # -----------------------------------------------------------------
        # 行表示真实标签，列表示预测标签，对应 TN、FP、FN、TP。
        tn = int(kwargs.get("tn", 0))
        fp = int(kwargs.get("fp", 0))
        fn = int(kwargs.get("fn", 0))
        tp = int(kwargs.get("tp", 0))

        cm = np.array([[tn, fp], [fn, tp]])

        # 使用颜色深浅表达样本数量，并保留色条作为数量参照。
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues, alpha=0.85)
        fig.colorbar(im)

        # -----------------------------------------------------------------
        # 2. 标注预测结果
        # -----------------------------------------------------------------
        # 根据背景深浅切换文字颜色，保证不同数据分布下数值仍然清晰。
        threshold = cm.max() / 2.0 if cm.max() > 0 else 0.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j,
                    i,
                    f"{cm[i, j]:d}",
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > threshold else "black",
                    fontsize=16,
                    fontweight="bold",
                )

        # 明确真实标签与预测标签方向，避免混淆误报和漏报。
        ax.set_xticks(np.arange(2))
        ax.set_yticks(np.arange(2))
        ax.set_xticklabels(["Pred: Clean", "Pred: Poisoned"], fontsize=11)
        ax.set_yticklabels(
            ["True: Clean", "True: Poisoned"],
            fontsize=11,
            rotation=90,
            va="center",
        )

        # -----------------------------------------------------------------
        # 3. 完善图表样式并输出
        # -----------------------------------------------------------------
        ax.set_title(
            "检测器混淆矩阵 (Confusion Matrix)",
            fontsize=14,
            fontweight="bold",
            pad=20,
            color="#263238",
        )

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # 去掉装饰性边框后转为 Base64，供探测器报告内嵌使用。
        return _fig_to_base64(fig)

    # 未注册类型直接报错，避免静默生成错误报告。
    raise ValueError(f"未知图表类型: {kind}")


# ====================================================
# HTML 报告构建函数
# ====================================================
def _build_html_report(report_type, report_data):
    """
    统一构建 HTML 报告。

    report_type:
        offline  -> 深度免疫重构；
        fast     -> 极速免疫清洗；
        detector -> 静态权重空间探测器。
    """

    def esc(value):
        return html.escape(str(value), quote=True)

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    base_css = """
    body{font-family:'Segoe UI',Tahoma,Verdana,sans-serif;background:#F4F6F8;color:#263238;margin:0;padding:20px;}
    .container{max-width:1000px;margin:0 auto;}
    .header{color:white;padding:20px 30px;border-radius:8px 8px 0 0;display:flex;justify-content:space-between;align-items:center;}
    .header h1{margin:0;font-size:24px}.timestamp{font-size:14px;opacity:.85}
    .card{background:white;padding:25px;margin-bottom:20px;border-radius:0 0 8px 8px;box-shadow:0 2px 10px rgba(0,0,0,.05);border-top:4px solid var(--primary);}
    h2{font-size:18px;border-bottom:2px solid #E0E0E0;padding-bottom:10px;margin-top:0;color:var(--primary);}
    .grid{display:grid;grid-template-columns:1fr 1fr;gap:20px}.data-item{margin-bottom:15px}
    .data-label{font-size:12px;color:#78909C;text-transform:uppercase;font-weight:bold}
    .data-value{font-size:16px;font-weight:500;margin-top:5px;word-break:break-all}
    .big-number{color:var(--primary);font-weight:bold;font-size:22px}
    .badge{display:inline-block;padding:5px 12px;border-radius:20px;color:white;font-weight:bold;font-size:14px}
    .metric-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:18px 0 8px}
    .metric-card{background:linear-gradient(145deg,#F8FBFA,#F2F7F6);padding:15px 16px;border-radius:8px;border:1px solid #DFE9E7}
    .metric-label{font-size:12px;color:#78909C;font-weight:bold;margin-bottom:7px}
    .metric-value{font-size:22px;line-height:1.1;color:var(--primary);font-weight:700}
    .metric-note{font-size:11px;color:#90A4AE;margin-top:6px}
    .head-summary{margin:16px 0 6px;padding:13px 15px;border-radius:8px;background:#F4F8FB;border:1px solid #DCE7EE}
    .head-summary-title{font-size:12px;color:#546E7A;font-weight:bold;margin-bottom:9px}
    .head-list{display:flex;flex-wrap:wrap;gap:8px}.head-chip{padding:6px 10px;border-radius:6px;background:white;border:1px solid #CFDDE6;color:#1565C0;font-size:12px}
    .chart-container{text-align:center;margin-top:20px;background:#FCFDFD;padding:18px;border-radius:10px;border:1px solid #E0E8E7;box-shadow:0 8px 24px rgba(38,50,56,.04)}
    .chart-container img{max-width:100%;height:auto}.muted{color:#546E7A;font-size:14px}
    .log{background:#F7F9FA;border-left:4px solid var(--primary);padding:12px;font-size:13px;color:#455A64;white-space:pre-wrap}
    @media(max-width:720px){.grid,.metric-grid{grid-template-columns:1fr 1fr}.header{align-items:flex-start;gap:12px;flex-direction:column}.card{padding:20px 16px}}
    """

    if report_type in ("offline", "fast"):
        is_fast = report_type == "fast"
        primary = "#00695C" if is_fast else "#311B92"
        badge_bg = "#00695C" if is_fast else "#6A1B9A"
        icon = "⚡" if is_fast else "🧬"
        title = (
            "Aegis-LoRA 极速免疫清洗报告" if is_fast else "Aegis-LoRA 深度免疫重构报告"
        )
        mode = "极速免疫查杀 (Fast Cleanse)" if is_fast else "深度底层免疫 (BD-Vax)"
        output_label = (
            "极速康复产物 (Fast Immunized Output)"
            if is_fast
            else "纯净康复产物 (Immunized Output)"
        )
        section2 = "预计算签名库应用" if is_fast else "变体指纹提取参数"
        section2_text = (
            "系统直接加载预计算多域聚合签名，跳过实时变体训练与特征提取阶段。"
            if is_fast
            else "系统构建多个 clean / poisoned 变体，并从 LoRA 差分中提取跨变体共享签名。"
        )
        chart_text = (
            "下图以清洗前范数为 100%，展示应用预计算签名后变化最显著的前 15 个完整结构。"
            if is_fast
            else "下图以清洗前范数为 100%，展示离线签名提取后变化最显著的前 15 个完整结构。"
        )

        surgery_rows = report_data.get("surgery_rows")
        if surgery_rows is None:
            surgery_rows = _aggregate_surgery_rows(
                report_data.get("suppressed_dict") or {},
                report_data.get("norms_before") or {},
                report_data.get("norms_after") or {},
                report_data.get("target_attention_heads") or [],
            )

        affected_structures = sum(
            1 for row in surgery_rows if int(row.get("count", 0) or 0) > 0
        )
        norm_drops = [
            float(row.get("drop", 0.0) or 0.0)
            for row in surgery_rows
            if float(row.get("before_norm", 0.0) or 0.0) > 0
        ]

        average_drop = float(np.mean(norm_drops)) if norm_drops else 0.0
        maximum_drop = max(norm_drops, default=0.0)
        suppressed_total = int(report_data.get("suppressed_count", 0) or 0)

        # 图表仅展示 Top 15，HTML 摘要按层补充全部被清洗的 Attention head。
        head_items = []
        total_attention_heads = 0
        for row in sorted(surgery_rows, key=lambda item: item.get("layer", "")):
            heads = row.get("heads") or []
            if row.get("structure") != "attention" or not heads:
                continue

            total_attention_heads += len(heads)
            preview = ", ".join(f"H{head}" for head in heads[:12])
            if len(heads) > 12:
                preview += f" … 另 {len(heads) - 12} 个"
            head_items.append(
                f'<span class="head-chip"><strong>{esc(row.get("layer", ""))}</strong>: '
                f"{esc(preview)}</span>"
            )

        head_summary_html = ""
        if head_items:
            head_summary_html = f"""
            <div class="head-summary">
                <div class="head-summary-title">完整 Attention head 清洗目标 · 共 {total_attention_heads} 个</div>
                <div class="head-list">{''.join(head_items)}</div>
            </div>
            """

        if report_data.get("chart"):
            chart_html = f"""
            <div class="chart-container">
                <img src="data:image/png;base64,{report_data['chart']}" alt="Surgery Chart">
            </div>
            """
        else:
            chart_html = '<div class="chart-container muted">未生成图表：suppressed_dict 为空或没有可视化数据。</div>'

        return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8"><title>{esc(title)}</title>
<style>:root{{--primary:{primary};--danger:#D32F2F;}}{base_css}.badge{{background:{badge_bg};}}</style>
</head>
<body>
<div class="container">
    <div class="header" style="background:var(--primary);">
        <h1>{icon} {esc(title)}</h1><div class="timestamp">生成时间: {now}</div>
    </div>

    <div class="card">
        <h2>1. 清洗任务概览 (Task Summary)</h2>
        <div class="grid">
            <div class="data-item"><div class="data-label">引擎干预模式</div><div class="data-value"><span class="badge">{esc(mode)}</span></div></div>
            <div class="data-item"><div class="data-label">基座模型</div><div class="data-value">{esc(report_data.get("base_model",""))}</div></div>
            <div class="data-item"><div class="data-label">待查杀目标</div><div class="data-value">{esc(report_data.get("lora_path",""))}</div></div>
            <div class="data-item"><div class="data-label">{esc(output_label)}</div><div class="data-value">{esc(report_data.get("cleansed_path",""))}</div></div>
        </div>
    </div>

    <div class="card">
        <h2>2. {esc(section2)} (Signature Settings)</h2>
        <p class="muted">{esc(section2_text)}</p>
        <div class="grid">
            <div class="data-item"><div class="data-label">变体数量</div><div class="data-value">{esc(report_data.get("n_variants",""))}</div></div>
            <div class="data-item"><div class="data-label">通道切除比例 Tau</div><div class="data-value">{float(report_data.get("tau",0))*100:.2f}%</div></div>
        </div>
    </div>

    <div class="card">
        <h2>3. 底层参数重构分析 (Parameter Surgery Analysis)</h2>
        <p class="muted">{esc(chart_text)} 青绿色表示清洗后保留范数，橙红色表示被削减部分。</p>
        <div class="metric-grid">
            <div class="metric-card"><div class="metric-label">累计清零参数</div><div class="metric-value">{suppressed_total:,}</div><div class="metric-note">实际置零的 LoRA 权重</div></div>
            <div class="metric-card"><div class="metric-label">受影响结构数</div><div class="metric-value">{affected_structures}</div><div class="metric-note">完整 MLP / Attention 结构</div></div>
            <div class="metric-card"><div class="metric-label">平均范数下降</div><div class="metric-value">{average_drop:.1f}%</div><div class="metric-note">全部受影响结构的平均值</div></div>
            <div class="metric-card"><div class="metric-label">最大范数下降</div><div class="metric-value">{maximum_drop:.1f}%</div><div class="metric-note">单个完整结构最大削减</div></div>
        </div>
        {head_summary_html}
        {chart_html}
    </div>

    <div class="card">
        <h2>4. 审计日志 (Audit Log)</h2>
        <div class="log">{esc(report_data.get("log_text",""))}</div>
    </div>
</div>
</body>
</html>"""

    if report_type == "detector":
        detector_css = base_css + """
        .grid{grid-template-columns:repeat(4,1fr)}
        .data-item{background:#F8FDFF;padding:15px;border-radius:6px;border-left:4px solid var(--primary)}
        .data-value{font-size:22px;font-weight:bold;color:var(--primary)}
        table{width:100%;border-collapse:collapse;margin-top:15px;font-size:14px}
        th,td{padding:12px 15px;text-align:left;border-bottom:1px solid #E0E0E0}
        th{background:#ECEFF1;color:#455A64;font-weight:bold}
        .status-danger{color:#D32F2F;font-weight:bold}.status-success{color:#2E7D32;font-weight:bold}
        """
        return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8"><title>Aegis-LoRA 静态权重空间探测器评估报告</title>
<style>:root{{--primary:#1565C0;}}{detector_css}</style>
</head>
<body>
<div class="container">
    <div class="header" style="background:var(--primary);">
        <h1>[Aegis-LoRA] 静态权重空间探测器评估报告</h1><div class="timestamp">生成时间: {now}</div>
    </div>

    <div class="card">
        <h2>1. 核心性能指标 (Evaluation Metrics)</h2>
        <div class="grid">
            <div class="data-item"><div class="data-label">准确率</div><div class="data-value">{float(report_data.get("accuracy",0)):.2f}%</div></div>
            <div class="data-item"><div class="data-label">召回率</div><div class="data-value" style="color:#2E7D32">{float(report_data.get("recall",0)):.2f}%</div></div>
            <div class="data-item"><div class="data-label">误报率</div><div class="data-value" style="color:#D32F2F">{float(report_data.get("fpr",0)):.2f}%</div></div>
            <div class="data-item"><div class="data-label">F1-Score</div><div class="data-value">{float(report_data.get("f1",0)):.2f}%</div></div>
            <div class="data-item"><div class="data-label">ROC-AUC</div><div class="data-value">{float(report_data.get("auc",0)):.4f}</div></div>
            <div class="data-item"><div class="data-label">Precision</div><div class="data-value">{float(report_data.get("precision",0)):.2f}%</div></div>
            <div class="data-item"><div class="data-label">总样本数</div><div class="data-value">{esc(report_data.get("total",0))}</div></div>
        </div>
    </div>

    <div class="card">
        <h2>2. 混淆矩阵分析 (Confusion Matrix)</h2>
        <p class="muted">左上为正确放行 TN，右下为正确拦截 TP；右上为误杀 FP，左下为漏检 FN。</p>
        <div class="chart-container"><img src="data:image/png;base64,{report_data.get("cm_chart","")}" alt="Confusion Matrix"></div>
    </div>

    <div class="card">
        <h2>3. 异常判定明细 (Prediction Details)</h2>
        <p class="muted">优先展示误判或临界样本，最多展示 50 条。</p>
        <table>
            <thead><tr><th>模型名称</th><th>真实标签</th><th>系统判定</th><th>中毒概率</th></tr></thead>
            <tbody>{report_data.get("table_rows","")}</tbody>
        </table>
    </div>
</div>
</body>
</html>"""

    raise ValueError(f"未知报告类型: {report_type}")


# ====================================================
# 报告文件写出函数
# ====================================================
def _save_report_files(html_content, report_data, output_dir, file_name):
    """
    写出 HTML 与 JSON。

    HTML 面向人工阅读；JSON 面向审计与复现。
    JSON 中会移除 Base64 图表和 HTML 表格，避免文件过大。
    """
    # 1. 确保报告输出目录存在。
    os.makedirs(output_dir, exist_ok=True)

    # 2. 写出 HTML 报告，供用户直接打开查看。
    file_path = os.path.join(output_dir, file_name)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    # 3. 基于同名 HTML 生成 JSON 路径，用于保存结构化审计数据。
    json_path = file_path.replace(".html", ".json")
    json_data = dict(report_data)

    # 4. JSON 中不保存 Base64 图表，避免文件体积过大。
    for key in ("chart", "cm_chart"):
        if key in json_data:
            json_data[key] = "Base64 image removed for JSON storage"

    # 5. JSON 中不保存 HTML 表格行，只保留原始结构化字段。
    if "table_rows" in json_data:
        json_data["table_rows"] = "HTML table rows removed for JSON storage"

    # 6. 写出 JSON 审计文件，便于后续复查或自动化分析。
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=4)

    return file_path


# ====================================================
# 报告导出接口
# ====================================================
def export_offline_report(
    base_model_path,
    lora_path,
    cleansed_path,
    log_text,
    n_variants,
    tau,
    norms_before,
    norms_after,
    suppressed_count,
    suppressed_dict,
    output_dir="./reports",
    custom_name=None,
    target_attention_heads=None,
):
    """导出深度免疫重构报告。"""
    # 1. 将 projection 统计聚合为完整结构，供图表、指标和 JSON 共同使用。
    surgery_rows = _aggregate_surgery_rows(
        suppressed_dict,
        norms_before,
        norms_after,
        target_attention_heads,
    )

    # 2. 生成手术诊断图表：展示完整结构的参数清零数与范数下降率。
    chart = _generate_chart(
        "surgery",
        surgery_rows=surgery_rows,
    )

    # 3. 组装报告数据。
    # HTML 用于人工阅读；JSON 会复用这份数据做审计留档。
    report_data = {
        "base_model": base_model_path,
        "lora_path": lora_path if lora_path else "纯基座模型",
        "cleansed_path": cleansed_path,
        "n_variants": n_variants,
        "tau": tau,
        "suppressed_count": suppressed_count,
        "suppressed_dict": suppressed_dict,
        "norms_before": norms_before,
        "norms_after": norms_after,
        "target_attention_heads": target_attention_heads or [],
        "surgery_rows": surgery_rows,
        "chart": chart,
        "log_text": log_text,
    }

    # 4. 渲染深度免疫专用 HTML 模板。
    html_content = _build_html_report("offline", report_data)

    # 5. 生成报告文件名。
    # custom_name 用于主流程指定审计报告名称；否则使用时间戳避免覆盖。
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = (
        f"{custom_name}.html"
        if custom_name
        else f"Aegis_Offline_Immunization_{timestamp}.html"
    )

    # 6. 同时写出 HTML 与 JSON 审计文件。
    file_path = _save_report_files(html_content, report_data, output_dir, file_name)

    print(f"      [-] [离线报告] 深度免疫重构离线报告已导出至: {file_path}")
    return file_path


def export_fast_cleanse_report(
    base_model_path,
    lora_path,
    cleansed_path,
    log_text,
    n_variants,
    tau,
    norms_before,
    norms_after,
    suppressed_count,
    suppressed_dict,
    output_dir="./reports",
    custom_name=None,
    target_attention_heads=None,
):
    """导出极速免疫清洗报告。"""
    # 1. 极速清洗沿用相同结构聚合规则，保证两类报告统计口径一致。
    surgery_rows = _aggregate_surgery_rows(
        suppressed_dict,
        norms_before,
        norms_after,
        target_attention_heads,
    )

    # 2. 生成完整结构清洗图表。
    chart = _generate_chart(
        "surgery",
        surgery_rows=surgery_rows,
    )

    # 3. 组装极速清洗报告数据。
    # n_variants 在这里表示离线签名库的预计算变体规模。
    report_data = {
        "base_model": base_model_path,
        "lora_path": lora_path if lora_path else "纯基座模型",
        "cleansed_path": cleansed_path,
        "n_variants": n_variants,
        "tau": tau,
        "suppressed_count": suppressed_count,
        "suppressed_dict": suppressed_dict,
        "norms_before": norms_before,
        "norms_after": norms_after,
        "target_attention_heads": target_attention_heads or [],
        "surgery_rows": surgery_rows,
        "chart": chart,
        "log_text": log_text,
    }

    # 4. 渲染极速免疫专用 HTML 模板。
    html_content = _build_html_report("fast", report_data)

    # 5. 生成报告文件名。
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = (
        f"{custom_name}.html"
        if custom_name
        else f"Aegis_FastCleanse_Immunization_{timestamp}.html"
    )

    # 6. 同时写出 HTML 与 JSON 审计文件。
    file_path = _save_report_files(html_content, report_data, output_dir, file_name)

    print(f"      [-] [离线报告] 极速免疫清洗离线报告已导出至: {file_path}")
    return file_path


def export_detector_report(
    report_dict,
    output_dir="./reports",
    custom_name="Detector_Evaluation_Report",
):
    """导出静态权重空间探测器评估报告。"""
    # 1. 拷贝输入字典，避免在原始评估结果上直接写入 HTML / Base64 字段。
    report_data = dict(report_dict)

    # 2. 生成混淆矩阵图表，展示 TN / FP / FN / TP。
    report_data["cm_chart"] = _generate_chart(
        "confusion_matrix",
        tn=report_data.get("tn", 0),
        fp=report_data.get("fp", 0),
        fn=report_data.get("fn", 0),
        tp=report_data.get("tp", 0),
    )

    # 3. 构造样本明细表。
    # 排序规则：误判样本优先，其次按中毒概率降序；最多展示 50 条，避免报告过长。
    table_rows = ""
    cases = report_data.get("cases", []) or []

    display_cases = sorted(
        cases,
        key=lambda x: (
            x.get("y_true") != x.get("y_pred"),
            x.get("prob", 0.0),
        ),
        reverse=True,
    )[:50]

    for case in display_cases:
        y_true = int(case.get("y_true", 0))
        y_pred = int(case.get("y_pred", 0))
        prob = float(case.get("prob", 0.0))

        true_str = "中毒" if y_true == 1 else "干净"
        pred_str = "拦截" if y_pred == 1 else "放行"

        # 误判样本使用浅红背景，便于人工审计时快速定位。
        row_style = "style='background-color:#FFF3F3;'" if y_true != y_pred else ""
        pred_class = "status-danger" if y_pred == 1 else "status-success"

        # 模型名进入 HTML 前做 escape，避免路径或模型名中含特殊字符破坏页面结构。
        model_name = html.escape(str(case.get("model_name", "")), quote=True)

        table_rows += f"""
        <tr {row_style}>
            <td>{model_name}</td>
            <td>{true_str}</td>
            <td class="{pred_class}">{pred_str}</td>
            <td>{prob * 100:.2f}%</td>
        </tr>
        """

    report_data["table_rows"] = table_rows

    # 4. 渲染探测器评估 HTML 模板。
    html_content = _build_html_report("detector", report_data)

    # 5. 探测器报告默认使用固定名称，便于重复评估时直接覆盖。
    file_name = f"{custom_name}.html"
    file_path = _save_report_files(html_content, report_data, output_dir, file_name)

    print(f"      [-] [离线报告] 静态探测器评估报告已导出至: {file_path}")
    return file_path
