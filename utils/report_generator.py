import json
import base64
import io
import os
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime


# ==========================================
# 工具函数：Matplotlib图表转 Base64
# ==========================================
def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150, transparent=True)
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return img_str


# ==========================================
# 离线免疫核心图表：参数切除前后对比
# ==========================================
def generate_bdvax_offline_chart(norms_before, norms_after):
    """生成离线模式下的通道 L2 Norm 切除前后对比图"""
    if not norms_before or not norms_after:
        return ""

    # 截取前 15 个被干预的层展示，保持图表清晰
    layers = list(norms_before.keys())[:15]
    b_vals = [norms_before[k] for k in layers]
    a_vals = [norms_after.get(k, 0) for k in layers]

    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(layers))
    width = 0.35

    # 使用代表离线深度手术的紫色系
    ax.bar(
        x - width / 2,
        b_vals,
        width,
        label="手术前 (提取到的后门载体)",
        color="#6A1B9A",
        alpha=0.8,
    )
    ax.bar(
        x + width / 2,
        a_vals,
        width,
        label="手术后 (切除/Xavier重置)",
        color="#2E7D32",
        alpha=0.9,
    )

    ax.set_title(
        "BD-Vax Offline Surgery: Maximum Channel L2 Norms Comparison",
        fontsize=12,
        fontweight="bold",
        color="#263238",
    )
    ax.set_ylabel("Max L2 Norm", fontsize=10)
    ax.set_xticks(x)

    # 精简X轴标签，移除冗余的字符串
    short_labels = [
        l.replace("Layer_", "L")
        .replace("_weight", "")
        .replace("_proj", "")
        .replace(".lora_B", "_B")
        .replace(".lora_A", "_A")
        for l in layers
    ]
    ax.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=9)
    ax.legend()
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    return fig_to_base64(fig)


# ==========================================
# 离线专属 HTML 报告构建器
# ==========================================
def build_offline_html_report(report_data):
    """构建专注于离线查杀数据的 HTML 模板"""

    mode_badge = '<span class="badge" style="background-color: #6A1B9A;">🧬 深度底层免疫 (BD-Vax)</span>'

    html_template = f"""
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="UTF-8">
        <title>Aegis-LoRA 深度免疫重构报告</title>
        <style>
            :root {{
                --primary: #311B92; /* 深紫色 */
                --success: #2E7D32;
                --danger: #D32F2F;
                --bg: #F4F6F8;
                --card-bg: #FFFFFF;
                --text: #263238;
            }}
            body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: var(--bg); color: var(--text); margin: 0; padding: 20px; }}
            .container {{ max-width: 1000px; margin: 0 auto; }}
            .header {{ background-color: var(--primary); color: white; padding: 20px 30px; border-radius: 8px 8px 0 0; display: flex; justify-content: space-between; align-items: center; }}
            .header h1 {{ margin: 0; font-size: 24px; }}
            .header .timestamp {{ font-size: 14px; opacity: 0.8; }}
            .card {{ background: var(--card-bg); padding: 25px; margin-bottom: 20px; border-radius: 0 0 8px 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.05); border-top: 4px solid var(--primary); }}
            h2 {{ font-size: 18px; border-bottom: 2px solid #E0E0E0; padding-bottom: 10px; margin-top: 0; color: var(--primary); }}
            .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
            .data-item {{ margin-bottom: 15px; }}
            .data-label {{ font-size: 12px; color: #78909C; text-transform: uppercase; font-weight: bold; }}
            .data-value {{ font-size: 16px; font-weight: 500; margin-top: 5px; word-break: break-all; }}
            .badge {{ display: inline-block; padding: 5px 12px; border-radius: 20px; color: white; font-weight: bold; font-size: 14px; }}
            .chart-container {{ text-align: center; margin-top: 20px; background: #FAFAFA; padding: 15px; border-radius: 8px; border: 1px dashed #CFD8DC; }}
            .chart-container img {{ max-width: 100%; height: auto; }}
            .log-box {{ background: #263238; color: #B39DDB; padding: 15px; border-radius: 6px; font-family: 'Courier New', Courier, monospace; font-size: 13px; white-space: pre-wrap; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🧬 Aegis-LoRA 深度免疫重构报告</h1>
                <div class="timestamp">生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</div>
            </div>
            
            <div class="card">
                <h2>1. 重构任务概览 (Task Summary)</h2>
                <div class="grid">
                    <div class="data-item">
                        <div class="data-label">引擎干预模式 (Mode)</div>
                        <div class="data-value">{mode_badge}</div>
                    </div>
                    <div class="data-item">
                        <div class="data-label">基座模型 (Base Model)</div>
                        <div class="data-value">{report_data['base_model']}</div>
                    </div>
                    <div class="data-item">
                        <div class="data-label">待查杀目标 (Target)</div>
                        <div class="data-value">{report_data['lora_path']}</div>
                    </div>
                    <div class="data-item">
                        <div class="data-label">纯净康复产物 (Immunized Output)</div>
                        <div class="data-value">{report_data['cleansed_path']}</div>
                    </div>
                </div>
            </div>

            <div class="card">
                <h2>2. 变体指纹提取参数 (Signature Extraction Settings)</h2>
                <p style="font-size: 14px; color: #546E7A;">系统在离线状态下构建了多个正交变体，并基于 Eq. 2 张量化提取了跨变体后门签名。</p>
                <div class="grid">
                    <div class="data-item">
                        <div class="data-label">正交特征变体数量 (N)</div>
                        <div class="data-value">{report_data['n_variants']}</div>
                    </div>
                    <div class="data-item">
                        <div class="data-label">通道切除比例 (Tau Ratio)</div>
                        <div class="data-value">{float(report_data['tau']) * 100}%</div>
                    </div>
                </div>
                <div class="data-item" style="margin-top: 15px;">
                    <div class="data-label">免疫系统执行日志 (Execution Log)</div>
                    <div class="log-box">{report_data['log_text']}</div>
                </div>
            </div>

            <div class="card">
                <h2>3. 底层参数重构分析 (Parameter Surgery Analysis)</h2>
                <p style="font-size: 14px; color: #546E7A;">根据全局签名分数（Global Scores），排名前 Tau% 的异常通道已被精确阻断（置零或重置）。</p>
                <div class="grid" style="margin-bottom: 15px;">
                    <div class="data-item">
                        <div class="data-label">干预神经元总数 (Channels Suppressed)</div>
                        <div class="data-value" style="color: #6A1B9A; font-weight: bold; font-size: 20px;">{report_data['suppressed_count']}</div>
                    </div>
                </div>
                <div class="chart-container">
                    <img src="data:image/png;base64,{report_data['chart']}" alt="Offline Surgery Comparison">
                </div>
            </div>
        </div>
    </body>
    </html>
    """
    return html_template


# ==========================================
# 离线报告导出接口
# ==========================================
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
    output_dir="./reports",
):
    """
    主程序 run_offline_immunization 调用的离线专用报告导出接口
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. 生成图表 Base64
    chart_base64 = generate_bdvax_offline_chart(norms_before, norms_after)

    # 2. 组装数据字典
    report_data = {
        "base_model": base_model_path,
        "lora_path": lora_path if lora_path else "纯基座模型",
        "cleansed_path": cleansed_path,
        "log_text": log_text,
        "n_variants": n_variants,
        "tau": tau,
        "suppressed_count": suppressed_count,
        "chart": chart_base64,
    }

    # 3. 渲染 HTML
    html_content = build_offline_html_report(report_data)

    # 4. 保存为独立文件
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"Aegis_Offline_Immunization_{timestamp}.html"
    file_path = os.path.join(output_dir, file_name)

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    # 同步保存 JSON
    json_path = file_path.replace(".html", ".json")
    with open(json_path, "w", encoding="utf-8") as f:
        json_data = report_data.copy()
        json_data["chart"] = "Base64 image removed for JSON storage"
        json.dump(json_data, f, ensure_ascii=False, indent=4)

    print(f"[报告生成] 深度免疫重构报告已导出至: {file_path}")
    return file_path
