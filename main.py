import gradio as gr
import torch
import gc
import os
import threading

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# 导入底层算法模块
from core.detector import confguard_detector
from core.cleanse import bd_vax_surgeon
from utils.report_generator import export_aegis_report

# ==========================================
# 本地路径选择器
# ==========================================
def open_folder_dialog():
    """打开本地文件夹选择对话框，返回路径字符串"""
    import tkinter as tk
    from tkinter import filedialog
    try:
        root = tk.Tk()
        root.withdraw()  # 隐藏主窗口
        root.attributes('-topmost', True)  # 窗口置顶
        folder_path = filedialog.askdirectory(title="选择模型/LoRA所在的本地文件夹")
        root.destroy()
        return folder_path if folder_path else ""
    except Exception as e:
        print(f"调用本地资源管理器失败: {e}")
        return ""

# ==========================================
# 系统状态与显存管理
# ==========================================
class AegisEngine:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.base_path = ""
        self.current_lora = ""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def free_memory(self):
        if self.model is not None:
            del self.model
        if self.tokenizer is not None:
            del self.tokenizer
        self.model = None
        self.tokenizer = None
        gc.collect()
        torch.cuda.empty_cache()

    def mount_shield(self, base_path, lora_path=""):
        self.free_memory()
        self.base_path = base_path
        self.current_lora = lora_path
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(base_path, local_files_only=True)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            base_model = AutoModelForCausalLM.from_pretrained(
                base_path,
                torch_dtype=torch.float16,
                device_map="auto",
                local_files_only=True
            )
            
            if lora_path and os.path.exists(lora_path):
                self.model = PeftModel.from_pretrained(base_model, lora_path, is_trainable=True)
            else:
                self.model = base_model
                
            self.model.eval()
            return f"🟢 系统已上线 | 基座: {os.path.basename(base_path)} | GPU 资源已锁定"
        except Exception as e:
            return f"🔴 挂载失败: {str(e)}"

engine = AegisEngine()

# ==========================================
# 聊天与后台静默检测逻辑
# ==========================================
def user_input_handler(user_message, history):
    """处理用户输入，将其转化为 role/content 字典格式"""
    if history is None:
        history = []
    
    # 将用户输入存入历史记录
    history.append({"role": "user", "content": str(user_message)})
    
    # 立即清空输入框，并返回更新后的 history
    return "", history

def bot_response_handler(history, p_threshold, l_threshold, out_dir="./reports"):
    """处理模型回复及后台异常检测"""
    
    # 确保 history 存在且不是空列表
    if not history or len(history) == 0:
        yield history, gr.update()
        return

    # 1. 精确提取用户的最新提问文本 (确保是纯字符串)
    # 因为上一步 user_input_handler 刚把用户消息 push 进去，所以倒数第一个就是用户消息
    user_prompt = str(history[-1]["content"])
    
    # 预留 AI 的回复坑位
    history.append({"role": "assistant", "content": ""})

    if engine.model is None:
        history[-1]["content"] = "❌ 系统未初始化：请先在上方加载基座模型与 LoRA。"
        yield history, gr.update()
        return

    # 2. 后台静默检测 (ConfGuard 动态分析)
    is_poisoned, generated_text, prob_traj = confguard_detector(
        model=engine.model,
        tokenizer=engine.tokenizer,
        prompt=user_prompt, # 这里确保传进去的绝对是 str
        p_threshold=p_threshold,
        l_threshold=l_threshold
    )

    if not is_poisoned:
        # 正常状态：用户完全无感，直接输出流畅对话
        history[-1]["content"] = generated_text
        yield history, gr.update(visible=False)
        return

    # 3. 拦截与警报：向用户展示异常信息并准备清洗
    warning_msg = (
        generated_text + 
        "\n\n🚨 **[安全熔断] 系统后台监测到高危“序列锁定”现象！** 模型疑似遭到后门触发。\n"
        "⚙️ *Aegis-LoRA 正在挂起推理引擎，切入离线参数清洗模式，请稍候...*"
    )
    history[-1]["content"] = warning_msg
    yield history, gr.update(visible=False) # 实时推送到前端显示警告

    # 4. 参数级离线清洗 (BD-Vax 静态切除)
    engine.model, surgery_report = bd_vax_surgeon(engine.model, suppression_ratio=0.35)
    
    # 保存免疫后的 LoRA
    cleansed_lora_path = engine.current_lora + "_cleansed"
    engine.model.save_pretrained(cleansed_lora_path)

    # 5. 生成法医学诊断报告
    report_path = export_aegis_report(
        base_model_path=engine.base_path,
        lora_path=engine.current_lora,
        is_poisoned=True,
        generated_text=generated_text + "\n[系统强制熔断]",
        prob_trajectory=prob_traj,
        norms_before=surgery_report["before_surgery_max_norms"],
        norms_after=surgery_report["after_surgery_max_norms"],
        suppressed_count=surgery_report['total_suppressed'],
        cleansed_path=cleansed_lora_path,
        output_dir=out_dir
    )

    # 6. 康复通知：向聊天气泡中追加恢复消息并展示报告下载
    recovery_msg = (
        f"\n\n---\n✅ **[清洗与自愈完成]**\n"
        f"外科手术模块已精准切除 `{surgery_report['total_suppressed']}` 个高危病灶神经元通道。\n"
        f"🛡️ 免疫版模型参数已重载。您可以继续安全对话。"
    )
    history[-1]["content"] += recovery_msg
    yield history, gr.update(value=report_path, visible=True)

with gr.Blocks(title="Aegis-LoRA 免疫防线") as app:

    gr.HTML("""
    <div style="text-align: center; padding: 15px; background: linear-gradient(90deg, #1A237E, #0D47A1); color: white; border-radius: 8px;">
        <h2 style="margin: 0;">🛡️ Aegis-LoRA 安全对话终端</h2>
        <p style="margin: 5px 0 0 0; opacity: 0.8;">LLM 参数级自愈免疫系统</p>
    </div>
    """)

    # --- 统一的加载引擎区 ---
    with gr.Group():
        with gr.Row(equal_height=True):
            # 基座模型选择 (文本框 + 文件夹图标)
            base_input = gr.Textbox(label="基座模型路径", scale=5, value="./models/Qwen2.5-3B-Instruct")
            base_folder_btn = gr.Button("📂", scale=1, min_width=40)
            
            # LoRA 矩阵选择 (文本框 + 文件夹图标)
            lora_input = gr.Textbox(label="LoRA路径(可选)", scale=5, placeholder="例如: ./models/poisoned_lora")
            lora_folder_btn = gr.Button("📂", scale=1, min_width=40)
            
            # 挂载按钮
            load_btn = gr.Button("🚀 初始化引擎", variant="primary", scale=2)
            
        status_output = gr.Textbox(show_label=False, placeholder="系统状态: 待初始化...", interactive=False, max_lines=1)

    # --- 专注聊天的核心区 ---
    chatbot = gr.Chatbot(label="Aegis 会话视窗 (后台全天候监测中)", height=450)
    
    with gr.Row():
        user_input = gr.Textbox(show_label=False, placeholder="输入您的问题进行对话...", scale=9)
        send_btn = gr.Button("发送", variant="primary", scale=1)

    # 生成的体检报告下载区域 (默认隐藏，仅在检测到中毒时弹出)
    report_download = gr.File(label="📄 触发清洗：自动生成的诊断报告", visible=False)

    # --- 后台哨兵高级参数隐藏折叠面板 ---
    with gr.Accordion("⚙️ 后台哨兵参数设置 (高级调试)", open=False):
        gr.Markdown("注意：这些参数用于调试后台静默监听器的敏感度，正常用户无需修改。")
        with gr.Row():
            p_slider = gr.Slider(0.9, 0.99, value=0.95, label="锁定置信度阈值 (P)")
            l_slider = gr.Slider(3, 15, value=8, step=1, label="触发序列长度 (L)")

    # ==========================================
    # 事件绑定
    # ==========================================
    # 1. 文件夹图标点击事件 (调用 tkinter)
    base_folder_btn.click(fn=open_folder_dialog, outputs=base_input)
    lora_folder_btn.click(fn=open_folder_dialog, outputs=lora_input)

    # 2. 挂载模型
    load_btn.click(fn=engine.mount_shield, inputs=[base_input, lora_input], outputs=status_output)

    # 3. 聊天发送事件流
    submit_event = user_input.submit(
        fn=user_input_handler, 
        inputs=[user_input, chatbot], 
        outputs=[user_input, chatbot], 
        queue=False
    ).then(
        fn=bot_response_handler, 
        inputs=[chatbot, p_slider, l_slider], 
        outputs=[chatbot, report_download]
    )

    send_btn.click(
        fn=user_input_handler, 
        inputs=[user_input, chatbot], 
        outputs=[user_input, chatbot], 
        queue=False
    ).then(
        fn=bot_response_handler, 
        inputs=[chatbot, p_slider, l_slider], 
        outputs=[chatbot, report_download]
    )

if __name__ == "__main__":
    app.launch(
        server_name="127.0.0.1", 
        server_port=7860, 
        theme=gr.themes.Soft(primary_hue="blue"),
        share=False
    )