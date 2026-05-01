import json
import base64
import io
import os
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# ==========================================
# 将 Matplotlib 图表转为 Base64 编码
# ==========================================
def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=150, transparent=True)
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_str

# ==========================================
# 图表生成引擎
# ==========================================
def generate_confguard_chart(trajectory, threshold=0.95):
    """生成 ConfGuard 动态序列锁定轨迹图"""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(trajectory, marker='o', color='#E65100', markersize=5, linewidth=2, label='Top-1 Probability')
    ax.axhline(y=threshold, color='#D32F2F', linestyle='--', linewidth=1.5, label=f'Lock Threshold ({threshold})')
    
    # 填充高危区域
    ax.fill_between(range(len(trajectory)), trajectory, threshold, 
                    where=(np.array(trajectory) > threshold), 
                    interpolate=True, color='#FFCDD2', alpha=0.5)

    ax.set_title("ConfGuard Dynamic Analysis: Sequence Lock Trajectory", fontsize=12, fontweight='bold', color='#263238')
    ax.set_xlabel("Generation Step (Tokens)", fontsize=10)
    ax.set_ylabel("Confidence (Probability)", fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='lower right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    return fig_to_base64(fig)

def generate_bdvax_chart(norms_before, norms_after):
    """生成 BD-Vax 参数空间手术前后对比图"""
    if not norms_before or not norms_after:
        return ""
        
    layers = list(norms_before.keys())[:15] # 截取前15个展示，防止拥挤
    b_vals = [norms_before[k] for k in layers]
    a_vals = [norms_after[k] for k in layers]
    
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(layers))
    width = 0.35
    
    ax.bar(x - width/2, b_vals, width, label='Before Surgery (Poisoned)', color='#D32F2F', alpha=0.8)
    ax.bar(x + width/2, a_vals, width, label='After Surgery (Cleansed)', color='#2E7D32', alpha=0.9)
    
    ax.set_title("BD-Vax Static Surgery: MLP Channels Spectral Norms", fontsize=12, fontweight='bold', color='#263238')
    ax.set_ylabel("Max L2 Norm / Spectral Norm", fontsize=10)
    ax.set_xticks(x)
    
    # 精简X轴标签
    short_labels = [l.replace("Layer_", "L").replace("_up_proj", "") for l in layers]
    ax.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=9)
    ax.legend()
    ax.grid(axis='y', linestyle=':', alpha=0.6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    return fig_to_base64(fig)

# ==========================================
# HTML 报告构建器
# ==========================================
def build_html_report(report_data):
    """将数据和 Base64 图表注入 HTML 模板"""
    
    # 状态颜色控制
    status_color = "#D32F2F" if report_data['is_poisoned'] else "#2E7D32"
    status_text = "高危 (已熔断并清理)" if report_data['is_poisoned'] else "安全 (正常通行)"
    
    html_template = f"""
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="UTF-8">
        <title>Aegis-LoRA 综合安全审计与免疫报告</title>
        <style>
            :root {{
                --primary: #1A237E;
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
            .card.alert {{ border-top-color: {status_color}; }}
            h2 {{ font-size: 18px; border-bottom: 2px solid #E0E0E0; padding-bottom: 10px; margin-top: 0; }}
            .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
            .data-item {{ margin-bottom: 15px; }}
            .data-label {{ font-size: 12px; color: #78909C; text-transform: uppercase; font-weight: bold; }}
            .data-value {{ font-size: 16px; font-weight: 500; margin-top: 5px; word-break: break-all; }}
            .badge {{ display: inline-block; padding: 5px 12px; border-radius: 20px; color: white; font-weight: bold; font-size: 14px; background-color: {status_color}; }}
            .chart-container {{ text-align: center; margin-top: 20px; background: #FAFAFA; padding: 15px; border-radius: 8px; border: 1px dashed #CFD8DC; }}
            .chart-container img {{ max-width: 100%; height: auto; }}
            .log-box {{ background: #263238; color: #00E676; padding: 15px; border-radius: 6px; font-family: 'Courier New', Courier, monospace; font-size: 13px; white-space: pre-wrap; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🛡️ Aegis-LoRA 免疫诊断报告</h1>
                <div class="timestamp">生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</div>
            </div>
            
            <div class="card alert">
                <h2>1. 检测概览 (Executive Summary)</h2>
                <div class="grid">
                    <div class="data-item">
                        <div class="data-label">基座模型 (Base Model)</div>
                        <div class="data-value">{report_data['base_model']}</div>
                    </div>
                    <div class="data-item">
                        <div class="data-label">检测对象 (Target LoRA)</div>
                        <div class="data-value">{report_data['lora_path']}</div>
                    </div>
                    <div class="data-item">
                        <div class="data-label">系统最终判定 (Verdict)</div>
                        <div class="data-value"><span class="badge">{status_text}</span></div>
                    </div>
                    <div class="data-item">
                        <div class="data-label">安全清理产物 (Cleansed LoRA)</div>
                        <div class="data-value">{report_data.get('cleansed_path', 'N/A')}</div>
                    </div>
                </div>
            </div>

            <div class="card">
                <h2>2. 动态哨兵防线 (ConfGuard Behaviour Analysis)</h2>
                <p style="font-size: 14px; color: #546E7A;">通过对模型生成序列的 Top-1 置信度进行滑动窗口监控，检测是否存在异常的强制响应锁定。</p>
                <div class="chart-container">
                    <img src="data:image/png;base64,{report_data['charts']['confguard']}" alt="ConfGuard Trajectory">
                </div>
                <div class="data-item" style="margin-top: 15px;">
                    <div class="data-label">触发生成的内容摘要 (Triggered Output)</div>
                    <div class="log-box">{report_data['generated_text']}</div>
                </div>
            </div>
    """
    
    # 如果检测到中毒，展示手术数据
    if report_data['is_poisoned'] and report_data['charts'].get('bdvax'):
        html_template += f"""
            <div class="card">
                <h2>3. 静态免疫手术 (BD-Vax Parameter Surgery)</h2>
                <p style="font-size: 14px; color: #546E7A;">利用 SVD 提取矩阵谱范数，强制抑制异常的 MLP 通道（病灶切除）。</p>
                <div class="grid" style="margin-bottom: 15px;">
                    <div class="data-item">
                        <div class="data-label">切除比例 (Suppression Ratio)</div>
                        <div class="data-value">{report_data.get('surgery_details', {}).get('ratio', '35%')}</div>
                    </div>
                    <div class="data-item">
                        <div class="data-label">阻断神经元总数 (Channels Suppressed)</div>
                        <div class="data-value" style="color: #D32F2F; font-weight: bold;">{report_data.get('surgery_details', {}).get('total_suppressed', 0)}</div>
                    </div>
                </div>
                <div class="chart-container">
                    <img src="data:image/png;base64,{report_data['charts']['bdvax']}" alt="BD-Vax Surgery">
                </div>
            </div>
        """
        
    html_template += """
        </div>
    </body>
    </html>
    """
    return html_template

# ==========================================
# 调用接口
# ==========================================
def export_aegis_report(
    base_model_path, 
    lora_path, 
    is_poisoned, 
    generated_text, 
    prob_trajectory, 
    norms_before=None, 
    norms_after=None, 
    suppressed_count=0,
    cleansed_path="N/A",
    output_dir="./reports"
):
    """
    外部主程序调用的报告生成函数
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 生成图表 Base64
    chart_confguard = generate_confguard_chart(prob_trajectory)
    chart_bdvax = generate_bdvax_chart(norms_before, norms_after) if is_poisoned else ""
    
    # 2. 组装数据字典
    report_data = {
        "base_model": base_model_path,
        "lora_path": lora_path,
        "is_poisoned": is_poisoned,
        "generated_text": generated_text,
        "cleansed_path": cleansed_path if is_poisoned else "未执行清理 (原模型安全)",
        "surgery_details": {
            "ratio": "35%",
            "total_suppressed": suppressed_count
        },
        "charts": {
            "confguard": chart_confguard,
            "bdvax": chart_bdvax
        }
    }
    
    # 3. 渲染 HTML
    html_content = build_html_report(report_data)
    
    # 4. 保存为独立文件
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"Aegis_Report_{timestamp}.html"
    file_path = os.path.join(output_dir, file_name)
    
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(html_content)
        
    # 同步保存一份机器可读的 JSON 供后续统计
    json_path = file_path.replace(".html", ".json")
    with open(json_path, "w", encoding="utf-8") as f:
        # 去掉庞大的 base64 图像再保存 json
        json_data = report_data.copy()
        json_data["charts"] = "Base64 images removed for JSON storage"
        json.dump(json_data, f, ensure_ascii=False, indent=4)
        
    print(f"[报告生成] 综合审计报告已导出至: {file_path}")
    return file_path
