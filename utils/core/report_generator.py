# Aegis-LoRA: 清洗审计报告模块
# 负责聚合 cleanse 输出，并生成深度 / 极速清洗共用的 HTML 报告与 JSON 审计数据。
import base64
import html
import io
import json
import os
from datetime import datetime

import matplotlib.pyplot as plt

# 设置 Matplotlib 中文字体，避免清洗结构与指标标签显示异常。
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = [
    "Noto Sans CJK SC",
    "Source Han Sans CN",
    "Microsoft YaHei",
    "SimHei",
    "PingFang SC",
    "WenQuanYi Micro Hei",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False


# ====================================================
# 清洗结构聚合
# ====================================================
def _aggregate_surgery_rows(
    suppressed_dict,
    norms_before,
    norms_after,
    target_attention_heads=None,
):
    """将 cleanse 的 projection 级统计聚合为完整 MLP 与 Attention 结构。"""

    # grouped 累计同层结构的参数量和范数；heads_by_layer 保存最终 head 索引。
    grouped = {}
    heads_by_layer = {}

    # -----------------------------------------------------------------
    # 1. 规范 Attention 目标：统一层标识，并按层去重 head 编号
    # -----------------------------------------------------------------
    for target in target_attention_heads or []:
        layer = str(target.get("layer", "")).strip()
        if not layer:
            continue

        layer_label = layer if layer.startswith("L") else f"L{layer}"

        # 报告生成不应因单条审计数据异常而中断，无法解析的 head 直接跳过。
        try:
            heads_by_layer.setdefault(layer_label, set()).add(int(target["head"]))
        except (KeyError, TypeError, ValueError):
            continue

    # -----------------------------------------------------------------
    # 2. 聚合完整结构：gate/up/down -> MLP，q/k/v/o -> Attention
    # -----------------------------------------------------------------
    for metric_key, count in (suppressed_dict or {}).items():
        # cleanse.py 使用 L{layer}.{projection}[.attn] 记录逐模块清洗结果。
        metric_key = str(metric_key)
        parts = metric_key.split(".")
        if len(parts) < 2:
            continue

        layer_label, projection = parts[0], parts[1]

        if projection in ("gate_proj", "up_proj", "down_proj"):
            structure = "mlp"
            group_key = f"{layer_label}.mlp"
            label = f"{layer_label} · MLP"
        elif projection in ("q_proj", "k_proj", "v_proj", "o_proj"):
            structure = "attention"
            group_key = f"{layer_label}.attention"
            label = f"{layer_label} · Attention"
        else:
            continue

        # 同层同结构共用一条记录，后续三个或四个 projection 会累计到这里。
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

        # 缺少手术后范数时按“未发生变化”处理，避免制造虚假的下降率。
        after = before if after_value is None else max(0.0, float(after_value))

        # 参数量直接求和；Frobenius 范数按平方和合并，避免平均比例造成失真。
        row["count"] += int(count or 0)
        row["before_sq"] += before**2
        row["after_sq"] += after**2

    # -----------------------------------------------------------------
    # 3. 生成报告行：恢复组合范数，并计算完整结构下降率
    # -----------------------------------------------------------------
    surgery_rows = []
    for row in grouped.values():
        before = row.pop("before_sq") ** 0.5
        after = row.pop("after_sq") ** 0.5
        drop = ((before - after) / before) * 100 if before > 0 else 0.0

        row["before_norm"] = before
        row["after_norm"] = after
        row["drop"] = min(100.0, max(0.0, drop))

        # MLP 只需要统一结构索引；Attention 额外保留完整 head 编号用于审计。
        row["heads"] = (
            sorted(heads_by_layer.get(row["layer"], set()))
            if row["structure"] == "attention"
            else []
        )
        surgery_rows.append(row)

    return surgery_rows


# ====================================================
# 清洗效果图表
# ====================================================
def _generate_surgery_chart(surgery_rows):
    """将聚合后的结构统计绘制为可嵌入 HTML 的 Base64 PNG。"""

    # 空手术结果不生成占位图片，由 HTML 模板显示无数据提示。
    if not surgery_rows:
        return ""

    # -----------------------------------------------------------------
    # 1. 筛选展示结构：按下降率排序，同分时优先清零参数更多的结构
    # -----------------------------------------------------------------
    top_rows = sorted(
        surgery_rows,
        key=lambda row: (row["drop"], row["count"]),
        reverse=True,
    )[:15]

    labels = []
    for row in top_rows:
        label = row["label"]
        heads = row.get("heads") or []

        # 图表最多展开 6 个 head，完整编号仍保留在页面摘要和 JSON 中。
        if row["structure"] == "attention" and heads:
            preview = ", ".join(f"H{head}" for head in heads[:6])
            suffix = f" …(+{len(heads) - 6})" if len(heads) > 6 else ""
            label = f"{label}\n{preview}{suffix}"

        labels.append(label)

    # 每行统一到 100% 基准：青绿色为保留量，橙红色为下降量。
    drops = [row["drop"] for row in top_rows]
    retained = [100.0 - drop for drop in drops]
    positions = list(range(len(top_rows)))

    # -----------------------------------------------------------------
    # 2. 绘制清洗前后对比：画布随结构数量增长，避免标签重叠
    # -----------------------------------------------------------------
    figure_height = max(5.0, 1.8 + len(top_rows) * 0.52)
    fig, ax = plt.subplots(figsize=(10, figure_height))

    ax.barh(
        positions,
        retained,
        height=0.62,
        color="#B7D8D1",
        edgecolor="none",
        label="清洗后保留范数",
    )
    ax.barh(
        positions,
        drops,
        left=retained,
        height=0.62,
        color="#E76F51",
        edgecolor="none",
        label="已削减范数",
    )

    # 下降率写入色块；清零参数数量放在 100% 基准线右侧的独立留白区。
    for position, remaining, row in zip(positions, retained, top_rows):
        drop = row["drop"]

        # 削减区过窄时将标签移到保留区，保证低下降率仍然可读。
        if drop >= 9.0:
            text_x = remaining + drop / 2
            text_align = "center"
            text_color = "white"
        else:
            text_x = max(1.0, remaining - 1.2)
            text_align = "right"
            text_color = "#B64038"

        ax.text(
            text_x,
            position,
            f"↓ {drop:.1f}%",
            ha=text_align,
            va="center",
            color=text_color,
            fontsize=8.5,
            fontweight="bold",
        )
        ax.text(
            102.5,
            position,
            f"{row['count']:,}",
            ha="left",
            va="center",
            color="#455A64",
            fontsize=8.5,
        )

    # -----------------------------------------------------------------
    # 3. 完善结构语义：区分 MLP / Attention，并标明归一化基准
    # -----------------------------------------------------------------
    ax.set_yticks(positions)
    ax.set_yticklabels(labels, fontsize=9)

    # 标签颜色只区分结构类型，不干扰保留量 / 削减量的统一配色。
    for tick, row in zip(ax.get_yticklabels(), top_rows):
        tick.set_color("#00695C" if row["structure"] == "mlp" else "#1565C0")
        tick.set_fontweight("bold")

    # 排名最高的结构置顶；0-100% 为范数区，右侧 22% 留给参数数量。
    ax.invert_yaxis()
    ax.set_xlim(0, 122)
    ax.set_xticks(list(range(0, 101, 20)))
    ax.set_xticklabels([f"{value}%" for value in range(0, 101, 20)])
    ax.set_xlabel("清洗前范数基准（100%）", fontsize=9.5, color="#546E7A")
    ax.grid(axis="x", linestyle=":", color="#B0BEC5", alpha=0.55)
    ax.set_axisbelow(True)
    ax.axvline(100, color="#90A4AE", linewidth=0.8)
    ax.text(102.5, -0.9, "清零参数", color="#78909C", fontsize=8.5, fontweight="bold")

    # 标题和图例明确聚合口径，避免把单个 projection 误读为完整结构。
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
        "gate/up/down 聚合为 MLP · q/k/v/o 聚合为 Attention",
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

    # 移除无信息量边框，仅保留垂直参考网格。
    for spine in ("top", "right", "left", "bottom"):
        ax.spines[spine].set_visible(False)

    ax.tick_params(axis="both", length=0, colors="#546E7A")
    fig.tight_layout(pad=1.2)

    # 图表转为 Base64 后直接嵌入 HTML，使离线报告保持单文件可查看。
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight", dpi=150, transparent=True)
    plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


# ====================================================
# 报告文件写出
# ====================================================
def _save_report_files(html_content, report_data, output_dir, report_name):
    """将同一份报告分别写为可视化 HTML 与结构化 JSON。"""

    # -----------------------------------------------------------------
    # 1. 写出 HTML：保留内嵌图表，供人工直接查看
    # -----------------------------------------------------------------
    os.makedirs(output_dir, exist_ok=True)

    file_path = os.path.join(output_dir, f"{report_name}.html")
    json_path = os.path.splitext(file_path)[0] + ".json"

    with open(file_path, "w", encoding="utf-8") as file:
        file.write(html_content)

    # -----------------------------------------------------------------
    # 2. 写出 JSON：保留审计字段，移除体积较大的 Base64 图表
    # -----------------------------------------------------------------
    json_data = dict(report_data)
    json_data["chart"] = "Base64 image removed for JSON storage"
    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(json_data, file, ensure_ascii=False, indent=4)

    return file_path


# ====================================================
# 通用清洗报告接口
# ====================================================
def export_cleanse_report(
    report_type,
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
    """统一导出 offline / fast 清洗报告，两种模式共用统计与渲染流程。"""

    # -----------------------------------------------------------------
    # 1. 选择报告模式：只切换文案、配色和默认文件名
    # -----------------------------------------------------------------
    report_configs = {
        "offline": {
            "title": "Aegis-LoRA 深度免疫重构报告",
            "mode": "深度底层免疫 (BD-Vax)",
            "primary": "#311B92",
            "badge": "#6A1B9A",
            "icon": "🧬",
            "setting_title": "变体指纹提取参数",
            "setting_text": "系统构建 clean / poisoned 变体，并从 LoRA 差分中提取共享清洗签名。",
            "variant_label": "训练变体数量",
            "file_prefix": "Aegis_Offline_Immunization",
            "print_name": "深度免疫重构报告",
        },
        "fast": {
            "title": "Aegis-LoRA 极速免疫清洗报告",
            "mode": "极速免疫查杀 (Fast Cleanse)",
            "primary": "#00695C",
            "badge": "#00695C",
            "icon": "⚡",
            "setting_title": "预计算签名库应用",
            "setting_text": "系统加载预计算多域签名，直接执行 LoRA adapter-only 参数清洗。",
            "variant_label": "签名变体规模",
            "file_prefix": "Aegis_FastCleanse_Immunization",
            "print_name": "极速免疫清洗报告",
        },
    }

    # 未注册类型直接失败，避免静默生成语义错误的审计报告。
    if report_type not in report_configs:
        raise ValueError(f"未知清洗报告类型: {report_type}")

    config = report_configs[report_type]

    # -----------------------------------------------------------------
    # 2. 准备报告数据：聚合完整结构，并生成共用清洗效果图
    # -----------------------------------------------------------------
    surgery_rows = _aggregate_surgery_rows(
        suppressed_dict,
        norms_before,
        norms_after,
        target_attention_heads,
    )
    chart = _generate_surgery_chart(surgery_rows)

    # HTML 与 JSON 复用同一份数据，保证页面指标和审计记录口径一致。
    report_data = {
        "report_type": report_type,
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

    # -----------------------------------------------------------------
    # 3. 计算页面摘要：统计结构变化，并整理全部 Attention head
    # -----------------------------------------------------------------
    affected_structures = sum(1 for row in surgery_rows if row["count"] > 0)
    norm_drops = [row["drop"] for row in surgery_rows if row["before_norm"] > 0]
    average_drop = sum(norm_drops) / len(norm_drops) if norm_drops else 0.0
    maximum_drop = max(norm_drops, default=0.0)

    # Attention 按自然层序排列，避免字符串排序出现 L10 排在 L2 前面。
    attention_rows = sorted(
        (
            row
            for row in surgery_rows
            if row["structure"] == "attention" and row["heads"]
        ),
        key=lambda row: (
            (
                0,
                int(row["layer"][1:]),
            )
            if row["layer"].startswith("L") and row["layer"][1:].isdigit()
            else (1, row["layer"])
        ),
    )
    # 页面摘要展示完整 head 列表，不采用图表中的 6 个编号限制。
    total_attention_heads = sum(len(row["heads"]) for row in attention_rows)
    head_items = "".join(
        f'<span class="head-chip"><strong>{html.escape(row["layer"], quote=True)}</strong>: '
        f'{html.escape(", ".join(f"H{head}" for head in row["heads"]), quote=True)}</span>'
        for row in attention_rows
    )

    # 没有 Attention 目标时不渲染空摘要卡片。
    head_summary = ""
    if head_items:
        head_summary = f"""
        <div class="head-summary">
            <div class="head-summary-title">完整 Attention head 清洗目标 · 共 {total_attention_heads} 个</div>
            <div class="head-list">{head_items}</div>
        </div>"""

    # 图表为空时输出明确提示，保持报告结构完整。
    if chart:
        chart_html = f"""
        <div class="chart-container">
            <img src="data:image/png;base64,{chart}" alt="完整结构清洗效果图">
        </div>"""
    else:
        chart_html = (
            '<div class="chart-container muted">没有可用于展示的参数清洗数据。</div>'
        )

    # 路径和日志属于外部输入，进入 HTML 前统一转义。
    safe_text = {
        key: html.escape(str(report_data.get(key, "")), quote=True)
        for key in (
            "base_model",
            "lora_path",
            "cleansed_path",
            "n_variants",
            "log_text",
        )
    }

    # -----------------------------------------------------------------
    # 4. 渲染 HTML：共用卡片布局，仅注入当前模式配置与清洗数据
    # -----------------------------------------------------------------
    # CSS 内嵌在报告中，确保产物脱离工程目录后仍可独立查看。
    css = """
    body{font-family:'Segoe UI',Tahoma,Verdana,sans-serif;background:#F4F6F8;color:#263238;margin:0;padding:20px}
    .container{max-width:1000px;margin:0 auto}.header{color:white;padding:20px 30px;border-radius:8px 8px 0 0;display:flex;justify-content:space-between;align-items:center}
    .header h1{margin:0;font-size:24px}.timestamp{font-size:14px;opacity:.85}.card{background:white;padding:25px;margin-bottom:20px;border-radius:0 0 8px 8px;box-shadow:0 2px 10px rgba(0,0,0,.05);border-top:4px solid var(--primary)}
    h2{font-size:18px;border-bottom:2px solid #E0E0E0;padding-bottom:10px;margin-top:0;color:var(--primary)}.grid{display:grid;grid-template-columns:1fr 1fr;gap:20px}.data-item{margin-bottom:15px}
    .data-label{font-size:12px;color:#78909C;font-weight:bold}.data-value{font-size:16px;font-weight:500;margin-top:5px;word-break:break-all}.badge{display:inline-block;padding:5px 12px;border-radius:20px;color:white;font-weight:bold;font-size:14px}
    .metric-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:18px 0 8px}.metric-card{background:#F5F9F8;padding:15px 16px;border-radius:8px;border:1px solid #DFE9E7}
    .metric-label{font-size:12px;color:#78909C;font-weight:bold;margin-bottom:7px}.metric-value{font-size:22px;color:var(--primary);font-weight:700}.metric-note{font-size:11px;color:#90A4AE;margin-top:6px}
    .head-summary{margin:16px 0 6px;padding:13px 15px;border-radius:8px;background:#F4F8FB;border:1px solid #DCE7EE}.head-summary-title{font-size:12px;color:#546E7A;font-weight:bold;margin-bottom:9px}
    .head-list{display:flex;flex-wrap:wrap;gap:8px}.head-chip{padding:6px 10px;border-radius:6px;background:white;border:1px solid #CFDDE6;color:#1565C0;font-size:12px}.chart-container{text-align:center;margin-top:20px;background:#FCFDFD;padding:18px;border-radius:10px;border:1px solid #E0E8E7}
    .chart-container img{max-width:100%;height:auto}.muted{color:#546E7A;font-size:14px}.log{background:#F7F9FA;border-left:4px solid var(--primary);padding:12px;font-size:13px;color:#455A64;white-space:pre-wrap}
    @media(max-width:720px){.grid,.metric-grid{grid-template-columns:1fr 1fr}.header{align-items:flex-start;gap:12px;flex-direction:column}.card{padding:20px 16px}}
    """

    # 页面固定为任务概览、签名设置、清洗效果和清洗日志四个区块。
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    html_content = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8"><title>{config['title']}</title>
<style>:root{{--primary:{config['primary']};}}{css}.badge{{background:{config['badge']};}}</style>
</head>
<body>
<div class="container">
    <div class="header" style="background:var(--primary)">
        <h1>{config['icon']} {config['title']}</h1><div class="timestamp">生成时间: {now}</div>
    </div>

    <div class="card">
        <h2>1. 清洗任务概览</h2>
        <div class="grid">
            <div class="data-item"><div class="data-label">清洗模式</div><div class="data-value"><span class="badge">{config['mode']}</span></div></div>
            <div class="data-item"><div class="data-label">基座模型</div><div class="data-value">{safe_text['base_model']}</div></div>
            <div class="data-item"><div class="data-label">待清洗 LoRA</div><div class="data-value">{safe_text['lora_path']}</div></div>
            <div class="data-item"><div class="data-label">清洗后 LoRA</div><div class="data-value">{safe_text['cleansed_path']}</div></div>
        </div>
    </div>

    <div class="card">
        <h2>2. {config['setting_title']}</h2>
        <p class="muted">{config['setting_text']}</p>
        <div class="grid">
            <div class="data-item"><div class="data-label">{config['variant_label']}</div><div class="data-value">{safe_text['n_variants']}</div></div>
            <div class="data-item"><div class="data-label">MLP 清洗比例 Tau</div><div class="data-value">{float(tau or 0) * 100:.2f}%</div></div>
        </div>
    </div>

    <div class="card">
        <h2>3. 完整结构清洗效果</h2>
        <p class="muted">图表将同层 gate/up/down 合并为 MLP，将 q/k/v/o 合并为 Attention。</p>
        <div class="metric-grid">
            <div class="metric-card"><div class="metric-label">累计清零参数</div><div class="metric-value">{int(suppressed_count or 0):,}</div><div class="metric-note">实际置零的 LoRA 权重</div></div>
            <div class="metric-card"><div class="metric-label">受影响结构</div><div class="metric-value">{affected_structures}</div><div class="metric-note">完整 MLP / Attention</div></div>
            <div class="metric-card"><div class="metric-label">平均范数下降</div><div class="metric-value">{average_drop:.1f}%</div><div class="metric-note">全部有效结构平均值</div></div>
            <div class="metric-card"><div class="metric-label">最大范数下降</div><div class="metric-value">{maximum_drop:.1f}%</div><div class="metric-note">单个结构最大削减</div></div>
        </div>
        {head_summary}
        {chart_html}
    </div>

    <div class="card">
        <h2>4. 清洗日志</h2>
        <div class="log">{safe_text['log_text']}</div>
    </div>
</div>
</body>
</html>"""

    # -----------------------------------------------------------------
    # 5. 写出报告：自定义名称优先，否则使用模式前缀与时间戳
    # -----------------------------------------------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_name = custom_name or f"{config['file_prefix']}_{timestamp}"
    file_path = _save_report_files(
        html_content,
        report_data,
        output_dir,
        report_name,
    )

    print(f"      [-] [离线报告] {config['print_name']}已导出至: {file_path}")
    return file_path
