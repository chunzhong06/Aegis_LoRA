# Aegis-LoRA: 报告生成器
# 1. 离线免疫重构报告：专门针对离线免疫重构流程设计的报告模板，展示核心诊断指标和图表，帮助用户深入理解重构过程中的关键干预点和效果。
# 2. 快速免疫重构报告：展示被切除神经元数量、特征范数下降率等核心诊断指标，帮助用户理解重构过程中的关键干预点。
# 3. 静态特征探测器报告：展示检测器在测试集上的混淆矩阵和性能指标。
import json
import base64
import io
import os
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "PingFang SC",
    "sans-serif",
]
plt.rcParams["axes.unicode_minus"] = False


# ==========================================
# 工具函数
# 将生成的图表在内存中转换为Base64字符串，以便直接将图片内嵌到HTML代码中，无需保存临时图片文件。
# ==========================================
def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150, transparent=True)
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return img_str


# ==========================================
# 离线免疫核心图表
# 柱状图展示被切除的神经元绝对数量，折线图展示特征范数下降百分比。
# ==========================================
def generate_bdvax_offline_chart(suppressed_dict, norms_before, norms_after):
    """生成离线免疫重构的核心诊断图表，展示被切除通道数量和特征范数下降率的关系。"""
    if not suppressed_dict:
        return ""

    # 选取切除数量最多的前 15 个关键层进行可视化
    sorted_layers = sorted(suppressed_dict.items(), key=lambda x: x[1], reverse=True)[
        :15
    ]
    layers = [k for k, v in sorted_layers]
    counts = [v for k, v in sorted_layers]

    # 计算平均范数下降百分比
    drops = []
    for k in layers:
        b = norms_before.get(k, 1e-9)
        a = norms_after.get(k, b)
        drop_pct = ((b - a) / b) * 100 if b > 0 else 0
        drops.append(max(0, drop_pct))

    fig, ax1 = plt.subplots(figsize=(10, 4))
    x = np.arange(len(layers))
    width = 0.5

    # 左轴：绘制切除神经元数量的柱状图 (红色斜线阴影)
    bars = ax1.bar(
        x,
        counts,
        width,
        color="#D32F2F",
        alpha=0.85,
        label="切除神经元数 (Excised Count)",
        hatch="//",
    )
    ax1.set_ylabel(
        "神经元切除数量 (个)", fontsize=10, color="#D32F2F", fontweight="bold"
    )
    ax1.tick_params(axis="y", labelcolor="#D32F2F")

    # 右轴：绘制特征范数下降率的折线图 (蓝色)
    ax2 = ax1.twinx()
    line = ax2.plot(
        x,
        drops,
        color="#1976D2",
        marker="o",
        linestyle="-",
        linewidth=2,
        markersize=6,
        label="平均范数下降率 (Norm Drop %)",
    )
    ax2.set_ylabel(
        "平均特征范数下降率 (%)", fontsize=10, color="#1976D2", fontweight="bold"
    )
    ax2.tick_params(axis="y", labelcolor="#1976D2")
    ax2.set_ylim(bottom=0)

    # 合并图例
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right")

    ax1.set_title(
        "Top 15 重点干预网络层: 神经元切除数与特征下降率",
        fontsize=12,
        fontweight="bold",
        color="#263238",
    )

    # 标签精简处理
    short_labels = [
        l.replace("Layer_", "L")
        .replace("_weight", "")
        .replace("_proj", "")
        .replace(".lora_B", "_B")
        .replace(".lora_A", "_A")
        for l in layers
    ]
    ax1.set_xticks(x)
    ax1.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=9)

    ax1.grid(axis="y", linestyle=":", alpha=0.6)
    ax1.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)

    return fig_to_base64(fig)


# ==========================================
# 静态特征探测器报告生成模块
# 生成混淆矩阵图表，展示检测器在测试集上的性能表现。
# ==========================================
def generate_confusion_matrix_chart(tn, fp, fn, tp):
    fig, ax = plt.subplots(figsize=(6, 5))
    cm = np.array([[tn, fp], [fn, tp]])

    cax = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues, alpha=0.8)
    fig.colorbar(cax)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=16,
                fontweight="bold",
            )

    classes = ["Clean (0)", "Poisoned (1)"]
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(["Pred: Clean", "Pred: Poisoned"], fontsize=11)
    ax.set_yticklabels(
        ["True: Clean", "True: Poisoned"], fontsize=11, rotation=90, va="center"
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

    return fig_to_base64(fig)


# ==========================================
# 离线专属 HTML 报告构建器
# 定义一个前端HTML模板，将传入的诊断数据渲染成排版美观的网页报告。
# ==========================================
def build_offline_html_report(report_data):
    """构建专注于离线免疫重构的 HTML 模板"""

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
                <p style="font-size: 14px; color: #546E7A;">系统在离线状态下构建了多个正交变体，并张量化提取了跨变体后门签名。</p>
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
            </div>

            <div class="card">
                <h2>3. 底层参数重构分析 (Parameter Surgery Analysis)</h2>
                <p style="font-size: 14px; color: #546E7A;">
                    下图展示了受干预最显著的前 15 个网络层的精细化分析。
                    <b style="color:var(--danger);">红色柱状图</b> 代表各层中被精准识别并切除的神经元绝对数量；
                    <b style="color:#1976D2;">蓝色折线图</b> 则反映了干预导致的该层平均特征范数(Mean Norm)的下降比率。
                </p>
                <div class="grid" style="margin-bottom: 15px;">
                    <div class="data-item">
                        <div class="data-label">干预神经元总数 (Channels Suppressed)</div>
                        <div class="data-value" style="color: #6A1B9A; font-weight: bold; font-size: 20px;">{report_data['suppressed_count']}</div>
                    </div>
                </div>
                <div class="chart-container">
                    <img src="data:image/png;base64,{report_data['chart']}" alt="Offline Surgery Modification Chart">
                </div>
            </div>
        </div>
    </body>
    </html>
    """
    return html_template


# ==========================================
# 极速清洗专属 HTML 报告构建器
# ==========================================
def build_fast_cleanse_html_report(report_data):
    """构建专注于极速查杀数据的 HTML 模板"""

    mode_badge = '<span class="badge" style="background-color: #00695C;">⚡ 极速免疫查杀 (Fast Cleanse)</span>'

    html_template = f"""
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="UTF-8">
        <title>Aegis-LoRA 极速免疫清洗报告</title>
        <style>
            :root {{
                --primary: #00695C; /* 深青绿，代表极速与安全 */
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
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>⚡ Aegis-LoRA 极速免疫清洗报告</h1>
                <div class="timestamp">生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</div>
            </div>
            
            <div class="card">
                <h2>1. 清洗任务概览 (Task Summary)</h2>
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
                        <div class="data-label">极速康复产物 (Fast Immunized Output)</div>
                        <div class="data-value">{report_data['cleansed_path']}</div>
                    </div>
                </div>
            </div>

            <div class="card">
                <h2>2. 预计算签名库应用 (Signature Bank Application)</h2>
                <p style="font-size: 14px; color: #546E7A;">系统直接加载了针对该架构预计算的多域聚合签名图谱，跳过了耗时的实时特征提取阶段，实现了秒级精准定位。</p>
                <div class="grid">
                    <div class="data-item">
                        <div class="data-label">预计算变体基数 (Precomputed Variants Base)</div>
                        <div class="data-value">{report_data['n_variants']}</div>
                    </div>
                    <div class="data-item">
                        <div class="data-label">通道切除比例 (Tau Ratio)</div>
                        <div class="data-value">{float(report_data['tau']) * 100}%</div>
                    </div>
                </div>
            </div>

            <div class="card">
                <h2>3. 底层参数重构分析 (Parameter Surgery Analysis)</h2>
                <p style="font-size: 14px; color: #546E7A;">
                    下图展示了应用预计算签名后，干预最显著的前 15 个网络层。
                    <b style="color:var(--danger);">红色柱状图</b> 代表该层被一键阻断的神经元数量；
                    <b style="color:#1976D2;">蓝色折线图</b> 则直观呈现了对应层特征强度的削弱比率（范数下降率）。
                </p>
                <div class="grid" style="margin-bottom: 15px;">
                    <div class="data-item">
                        <div class="data-label">一键切除神经元总数 (Channels Suppressed)</div>
                        <div class="data-value" style="color: #00695C; font-weight: bold; font-size: 20px;">{report_data['suppressed_count']}</div>
                    </div>
                </div>
                <div class="chart-container">
                    <img src="data:image/png;base64,{report_data['chart']}" alt="Fast Surgery Modification Chart">
                </div>
            </div>
        </div>
    </body>
    </html>
    """
    return html_template


# ==========================================
# 静态特征探测器 HTML 报告构建器
# ==========================================
def build_detector_html_report(report_data):
    """构建专注于静态权重空间探测器评估的 HTML 模板"""

    html_template = f"""
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="UTF-8">
        <title>Aegis-LoRA 静态权重空间探测器评估报告</title>
        <style>
            :root {{
                --primary: #1565C0;
                --bg: #F4F6F8;
                --card-bg: #FFFFFF;
                --text: #263238;
                --danger: #D32F2F;
                --success: #2E7D32;
            }}
            body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: var(--bg); color: var(--text); margin: 0; padding: 20px; }}
            .container {{ max-width: 1000px; margin: 0 auto; }}
            .header {{ background-color: var(--primary); color: white; padding: 20px 30px; border-radius: 8px 8px 0 0; display: flex; justify-content: space-between; align-items: center; }}
            .header h1 {{ margin: 0; font-size: 24px; }}
            .header .timestamp {{ font-size: 14px; opacity: 0.8; }}
            .card {{ background: var(--card-bg); padding: 25px; margin-bottom: 20px; border-radius: 0 0 8px 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.05); border-top: 4px solid var(--primary); }}
            h2 {{ font-size: 18px; border-bottom: 2px solid #E0E0E0; padding-bottom: 10px; margin-top: 0; color: var(--primary); }}
            .grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; }}
            .data-item {{ margin-bottom: 15px; background: #F8FDFF; padding: 15px; border-radius: 6px; border-left: 4px solid var(--primary); }}
            .data-label {{ font-size: 12px; color: #546E7A; text-transform: uppercase; font-weight: bold; }}
            .data-value {{ font-size: 22px; font-weight: bold; margin-top: 8px; color: var(--primary); }}
            .chart-container {{ text-align: center; margin-top: 20px; background: #FAFAFA; padding: 15px; border-radius: 8px; border: 1px dashed #CFD8DC; }}
            .chart-container img {{ max-width: 100%; height: auto; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 15px; font-size: 14px; }}
            th, td {{ padding: 12px 15px; text-align: left; border-bottom: 1px solid #E0E0E0; }}
            th {{ background-color: #ECEFF1; color: #455A64; font-weight: bold; }}
            .status-danger {{ color: var(--danger); font-weight: bold; }}
            .status-success {{ color: var(--success); font-weight: bold; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>[Aegis-LoRA] 静态权重空间探测器评估报告</h1>
                <div class="timestamp">生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</div>
            </div>
            
            <div class="card">
                <h2>1. 核心性能指标 (Evaluation Metrics)</h2>
                <div class="grid">
                    <div class="data-item">
                        <div class="data-label">准确率 (Accuracy)</div>
                        <div class="data-value">{report_data['accuracy']:.2f}%</div>
                    </div>
                    <div class="data-item">
                        <div class="data-label">召回率 (Recall / TPR)</div>
                        <div class="data-value" style="color: var(--success);">{report_data['recall']:.2f}%</div>
                    </div>
                    <div class="data-item">
                        <div class="data-label">误报率 (FPR)</div>
                        <div class="data-value" style="color: var(--danger);">{report_data['fpr']:.2f}%</div>
                    </div>
                    <div class="data-item">
                        <div class="data-label">F1-Score</div>
                        <div class="data-value">{report_data['f1']:.2f}%</div>
                    </div>
                    <div class="data-item">
                        <div class="data-label">ROC-AUC</div>
                        <div class="data-value">{report_data['auc']:.4f}</div>
                    </div>
                    <div class="data-item">
                        <div class="data-label">精确率 (Precision)</div>
                        <div class="data-value">{report_data['precision']:.2f}%</div>
                    </div>
                    <div class="data-item">
                        <div class="data-label">总测试样本数</div>
                        <div class="data-value">{report_data['total']}</div>
                    </div>
                </div>
            </div>

            <div class="card">
                <h2>2. 混淆矩阵分析 (Confusion Matrix)</h2>
                <p style="font-size: 14px; color: #546E7A;">矩阵直观呈现了探测器的分类能力。左上角为正确放行的健康模型 (TN)，右下角为正确拦截的中毒模型 (TP)。右上角为误杀的健康模型 (FP)，左下角为漏网的中毒模型 (FN)。</p>
                <div class="chart-container">
                    <img src="data:image/png;base64,{report_data['cm_chart']}" alt="Confusion Matrix Chart">
                </div>
            </div>
            
            <div class="card">
                <h2>3. 异常判定明细 (Prediction Details)</h2>
                <p style="font-size: 12px; color: #78909C;">注：当前表格按预测误差概率排序，优先展示被误判或临界状态的测试样本。</p>
                <table>
                    <thead>
                        <tr>
                            <th>模型名称</th>
                            <th>真实标签</th>
                            <th>系统判定</th>
                            <th>中毒概率</th>
                        </tr>
                    </thead>
                    <tbody>
                        {report_data['table_rows']}
                    </tbody>
                </table>
            </div>
        </div>
    </body>
    </html>
    """
    return html_template


# ==========================================
# 离线报告导出接口
# 报告生成的“总控”函数。依次调用生成图表、渲染HTML页面，最后将数据分别保存为 .html 网页文件和 .json 数据文件。
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
    suppressed_dict,
    output_dir="./reports",
    custom_name=None,
):
    os.makedirs(output_dir, exist_ok=True)

    # 1. 生成图表 Base64
    chart_base64 = generate_bdvax_offline_chart(
        suppressed_dict, norms_before, norms_after
    )

    # 2. 组装数据字典
    report_data = {
        "base_model": base_model_path,
        "lora_path": lora_path if lora_path else "纯基座模型",
        "cleansed_path": cleansed_path,
        "n_variants": n_variants,
        "tau": tau,
        "suppressed_count": suppressed_count,
        "chart": chart_base64,
        "log_text": log_text,  # 仅写入 JSON 留存记录
    }

    # 3. 渲染 HTML
    html_content = build_offline_html_report(report_data)

    # 4. 保存 HTML 和 JSON
    if custom_name:
        file_name = f"{custom_name}.html"
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"Aegis_Offline_Immunization_{timestamp}.html"
    file_path = os.path.join(output_dir, file_name)

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    json_path = file_path.replace(".html", ".json")
    with open(json_path, "w", encoding="utf-8") as f:
        json_data = report_data.copy()
        json_data["chart"] = "Base64 image removed for JSON storage"
        json.dump(json_data, f, ensure_ascii=False, indent=4)

    print(f"      [-] [完成] 深度免疫重构离线报告已导出至: {file_path}")
    return file_path


# ==========================================
# 极速清洗报告导出接口
# ==========================================
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
    os.makedirs(output_dir, exist_ok=True)

    # 1. 生成图表 Base64
    chart_base64 = generate_bdvax_offline_chart(
        suppressed_dict, norms_before, norms_after
    )

    # 2. 组装数据字典
    report_data = {
        "base_model": base_model_path,
        "lora_path": lora_path if lora_path else "纯基座模型",
        "cleansed_path": cleansed_path,
        "n_variants": n_variants,
        "tau": tau,
        "suppressed_count": suppressed_count,
        "chart": chart_base64,
        "log_text": log_text,
    }

    # 3. 渲染 HTML
    html_content = build_fast_cleanse_html_report(report_data)

    # 4. 保存 HTML 和 JSON
    if custom_name:
        file_name = f"{custom_name}.html"
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"Aegis_FastCleanse_Immunization_{timestamp}.html"
    file_path = os.path.join(output_dir, file_name)

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    json_path = file_path.replace(".html", ".json")
    with open(json_path, "w", encoding="utf-8") as f:
        json_data = report_data.copy()
        json_data["chart"] = "Base64 image removed for JSON storage"
        json.dump(json_data, f, ensure_ascii=False, indent=4)

    print(f"      [-] [完成] 极速免疫清洗离线报告已导出至: {file_path}")
    return file_path


# ==========================================
# 探测器评估报告导出接口
# ==========================================
def export_detector_report(
    report_dict, output_dir="./reports", custom_name="Detector_Evaluation_Report"
):
    """导出静态权重空间探测器的评估报告，包含性能指标、混淆矩阵图表和误判样本明细。"""
    os.makedirs(output_dir, exist_ok=True)

    cm_chart_base64 = generate_confusion_matrix_chart(
        report_dict["tn"], report_dict["fp"], report_dict["fn"], report_dict["tp"]
    )
    report_dict["cm_chart"] = cm_chart_base64

    table_rows = ""
    display_cases = sorted(
        report_dict["cases"],
        key=lambda x: (x["y_true"] != x["y_pred"], x["prob"]),
        reverse=True,
    )[:50]
    for case in display_cases:
        true_str = "中毒" if case["y_true"] == 1 else "干净"
        pred_str = "拦截" if case["y_pred"] == 1 else "放行"
        row_color = (
            "style='background-color: #FFF3F3;'"
            if case["y_true"] != case["y_pred"]
            else ""
        )
        pred_class = "status-danger" if case["y_pred"] == 1 else "status-success"

        table_rows += f"""
        <tr {row_color}>
            <td>{case['model_name']}</td>
            <td>{true_str}</td>
            <td class="{pred_class}">{pred_str}</td>
            <td>{case['prob']*100:.2f}%</td>
        </tr>
        """
    report_dict["table_rows"] = table_rows

    html_content = build_detector_html_report(report_dict)

    file_path = os.path.join(output_dir, f"{custom_name}.html")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    return file_path
