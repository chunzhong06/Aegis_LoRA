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
    # 1. LoRA 手术图：参数清零数 + 范数下降率
    # -----------------------------------------------------------------
    if kind == "surgery":
        suppressed_dict = kwargs.get("suppressed_dict") or {}
        norms_before = kwargs.get("norms_before") or {}
        norms_after = kwargs.get("norms_after") or {}

        # 没有手术统计时，不生成图表，避免报告中出现空图。
        if not suppressed_dict:
            return ""

        # 取清零参数最多的前 15 层，突出主要干预位置，避免图表过密。
        top_items = sorted(
            suppressed_dict.items(),
            key=lambda x: int(x[1]) if str(x[1]).isdigit() else 0,
            reverse=True,
        )[:15]

        layers = [str(k) for k, _ in top_items]
        counts = [int(v) for _, v in top_items]

        # 计算每层范数下降率，用于观察手术前后特征强度变化。
        drops = []
        for layer in layers:
            before = float(norms_before.get(layer, 1e-9) or 1e-9)
            after = float(norms_after.get(layer, before) or before)
            drop = ((before - after) / before) * 100 if before > 0 else 0.0
            drops.append(max(0.0, drop))

        fig, ax1 = plt.subplots(figsize=(10, 4))
        x = np.arange(len(layers))

        # 左轴：柱状图展示每层被清零的 LoRA 参数数量。
        ax1.bar(
            x,
            counts,
            width=0.55,
            color="#D32F2F",
            alpha=0.85,
            hatch="//",
            label="清零 LoRA 参数数",
        )
        ax1.set_ylabel("清零 LoRA 参数数", fontsize=10, color="#D32F2F")
        ax1.tick_params(axis="y", labelcolor="#D32F2F")
        ax1.grid(axis="y", linestyle=":", alpha=0.55)

        # 右轴：折线图展示对应层的范数下降率。
        ax2 = ax1.twinx()
        ax2.plot(
            x,
            drops,
            color="#1976D2",
            marker="o",
            linewidth=2,
            markersize=5,
            label="范数下降率 (%)",
        )
        ax2.set_ylabel("范数下降率 (%)", fontsize=10, color="#1976D2")
        ax2.tick_params(axis="y", labelcolor="#1976D2")
        ax2.set_ylim(bottom=0)

        # 缩短层名，保证横轴标签可读。
        short_labels = [
            layer.replace("Layer_", "L")
            .replace("_weight", "")
            .replace("_proj", "")
            .replace(".lora_B", "_B")
            .replace(".lora_A", "_A")
            for layer in layers
        ]

        ax1.set_xticks(x)
        ax1.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=9)
        ax1.set_title(
            "Top 15 重点干预层：参数清零数与范数下降率",
            fontsize=12,
            fontweight="bold",
            color="#263238",
        )

        # 合并左右轴图例。
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1 + h2, l1 + l2, loc="upper right")

        ax1.spines["top"].set_visible(False)
        ax2.spines["top"].set_visible(False)

        return _fig_to_base64(fig)

    # -----------------------------------------------------------------
    # 2. 探测器混淆矩阵图：TN / FP / FN / TP
    # -----------------------------------------------------------------
    if kind == "confusion_matrix":
        tn = int(kwargs.get("tn", 0))
        fp = int(kwargs.get("fp", 0))
        fn = int(kwargs.get("fn", 0))
        tp = int(kwargs.get("tp", 0))

        cm = np.array([[tn, fp], [fn, tp]])

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues, alpha=0.85)
        fig.colorbar(im)

        # 在矩阵格子中写入具体数量；深色格子使用白字提升可读性。
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

        # 设置横纵轴语义：横轴为预测结果，纵轴为真实标签。
        ax.set_xticks(np.arange(2))
        ax.set_yticks(np.arange(2))
        ax.set_xticklabels(["Pred: Clean", "Pred: Poisoned"], fontsize=11)
        ax.set_yticklabels(
            ["True: Clean", "True: Poisoned"],
            fontsize=11,
            rotation=90,
            va="center",
        )

        ax.set_title(
            "检测器混淆矩阵 (Confusion Matrix)",
            fontsize=14,
            fontweight="bold",
            pad=20,
            color="#263238",
        )

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        return _fig_to_base64(fig)

    # 未知图表类型直接报错，避免静默生成错误报告。
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
    .chart-container{text-align:center;margin-top:20px;background:#FAFAFA;padding:15px;border-radius:8px;border:1px dashed #CFD8DC}
    .chart-container img{max-width:100%;height:auto}.muted{color:#546E7A;font-size:14px}
    .log{background:#F7F9FA;border-left:4px solid var(--primary);padding:12px;font-size:13px;color:#455A64;white-space:pre-wrap}
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
            "下图展示应用预计算签名后，干预最显著的前 15 个网络层。"
            if is_fast
            else "下图展示离线签名提取后，干预最显著的前 15 个网络层。"
        )

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
        <p class="muted">{esc(chart_text)} 红色柱状图表示清零参数数；蓝色折线表示范数下降率。</p>
        <div class="data-item"><div class="data-label">清零 LoRA 参数总数</div><div class="data-value big-number">{esc(report_data.get("suppressed_count",0))}</div></div>
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
):
    """导出深度免疫重构报告。"""
    # 1. 生成手术诊断图表：展示重点层的 LoRA 参数清零数与范数下降率。
    chart = _generate_chart(
        "surgery",
        suppressed_dict=suppressed_dict,
        norms_before=norms_before,
        norms_after=norms_after,
    )

    # 2. 组装报告数据。
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
        "chart": chart,
        "log_text": log_text,
    }

    # 3. 渲染深度免疫专用 HTML 模板。
    html_content = _build_html_report("offline", report_data)

    # 4. 生成报告文件名。
    # custom_name 用于主流程指定审计报告名称；否则使用时间戳避免覆盖。
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = (
        f"{custom_name}.html"
        if custom_name
        else f"Aegis_Offline_Immunization_{timestamp}.html"
    )

    # 5. 同时写出 HTML 与 JSON 审计文件。
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
):
    """导出极速免疫清洗报告。"""
    # 1. 极速清洗同样使用 surgery 图表，便于和深度清洗报告保持一致。
    chart = _generate_chart(
        "surgery",
        suppressed_dict=suppressed_dict,
        norms_before=norms_before,
        norms_after=norms_after,
    )

    # 2. 组装极速清洗报告数据。
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
        "chart": chart,
        "log_text": log_text,
    }

    # 3. 渲染极速免疫专用 HTML 模板。
    html_content = _build_html_report("fast", report_data)

    # 4. 生成报告文件名。
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = (
        f"{custom_name}.html"
        if custom_name
        else f"Aegis_FastCleanse_Immunization_{timestamp}.html"
    )

    # 5. 同时写出 HTML 与 JSON 审计文件。
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
