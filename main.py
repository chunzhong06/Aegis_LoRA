import gradio as gr
import torch
import gc
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from utils.pipeline import run_static_scan_pipeline, run_immunization_pipeline

# ==========================================
# 全局模型状态 (严格对齐自定义调用)
# ==========================================
global_model = None
global_tokenizer = None
global_base_path = ""
global_lora_path = ""


def free_memory():
    """释放模型占用的显存资源"""
    global global_model, global_tokenizer, global_base_path, global_lora_path
    if global_model is not None:
        del global_model
    if global_tokenizer is not None:
        del global_tokenizer
    global_model = None
    global_tokenizer = None
    global_base_path = ""
    global_lora_path = ""
    gc.collect()
    torch.cuda.empty_cache()


def load_model_direct(base_path, lora_path=""):
    """直接加载模型和 Tokenizer"""
    global global_model, global_tokenizer, global_base_path, global_lora_path

    if (
        global_base_path == base_path
        and global_lora_path == lora_path
        and global_model is not None
    ):
        return "🟢 模型在线"

    free_memory()
    global_base_path = base_path
    global_lora_path = lora_path

    try:
        global_tokenizer = AutoTokenizer.from_pretrained(
            base_path, local_files_only=True
        )
        if global_tokenizer.pad_token is None:
            global_tokenizer.pad_token = global_tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(
            base_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            local_files_only=True,
        )

        if lora_path and os.path.exists(lora_path):
            global_model = PeftModel.from_pretrained(
                base_model, lora_path, is_trainable=True
            )
        else:
            global_model = base_model

        global_model.eval()
        return "🟢 模型在线"
    except Exception as e:
        free_memory()
        return f"🔴 加载失败: {str(e)}"


# ==========================================
# 工具函数
# ==========================================
def open_folder_dialog():
    import tkinter as tk
    from tkinter import filedialog

    try:
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        folder_path = filedialog.askdirectory(title="选择本地文件夹")
        root.destroy()
        return folder_path if folder_path else ""
    except Exception as e:
        print(f"[System Error] 调用本地资源管理器失败: {e}")
        return ""


def toggle_ui(interactive):
    """快速生成 9 个 UI 组件的交互状态"""
    return [gr.update(interactive=interactive) for _ in range(9)]


# ==========================================
# 会话管理与工作流 (加载即查杀)
# ==========================================
def create_new_session(name, base_path, lora_path, sessions):
    """工作流：锁定UI -> 静态扫描 -> (若中毒)免疫重构 -> 加载模型 -> 解锁UI"""
    if not name or not base_path or not lora_path:
        yield [gr.update(), sessions, "⚠️ 配置缺失", gr.update()] + toggle_ui(True)
        return
    if name in sessions:
        yield [gr.update(), sessions, "⚠️ 名称重复", gr.update()] + toggle_ui(True)
        return

    # 1. 立即锁定全部交互组件
    yield [gr.update(), sessions, "🛡️ 静态查杀与加载中...", gr.update()] + toggle_ui(
        False
    )

    print(f"\n[Aegis] Session initialization started: {name}")
    print(f"[Aegis] Performing spectral static scan on LoRA weights...")

    # 2. 执行静态扫描
    active_lora_path = lora_path
    report_file = None
    is_poisoned = False

    try:
        is_poisoned, prob = run_static_scan_pipeline(lora_path)
        if is_poisoned:
            scan_res = "🔴 发现后门"
            print(f"[Aegis] Result: BACKDOOR DETECTED. Probability: {prob*100:.2f}%")

            # 3. 触发免疫重构流水线
            yield [
                gr.update(),
                sessions,
                "🛡️ 拦截异常特征，正在执行 BD-Vax 免疫重构...",
                gr.update(),
            ] + toggle_ui(False)

            report_path, suppressed_count, cleansed_lora_path = (
                run_immunization_pipeline(
                    base_model_path=base_path,
                    lora_path=lora_path,
                    dataset_path="./datasets/clean_data.json",
                    tau=0.35,
                )
            )
            active_lora_path = cleansed_lora_path
            report_file = report_path
            scan_res += f" (已切除 {suppressed_count} 个特征)"

        else:
            scan_res = "🟢 安全"
            print(f"[Aegis] Result: CLEAN.")

    except Exception as e:
        scan_res = "🟡 流程异常"
        print(f"[Aegis] Error during pipeline: {str(e)}")

    # 4. 挂载模型 (无论是否经历重构，挂载最终的 active_lora_path)
    load_res = load_model_direct(base_path, active_lora_path)
    final_status = f"{scan_res} | {load_res}"

    # 5. 记录数据并解锁 UI
    sessions[name] = {
        "base_path": base_path,
        "lora_path": active_lora_path,
        "history": [],
        "is_poisoned": is_poisoned,
        "status": final_status,
        "report_path": report_file,
    }

    report_update = (
        gr.update(value=report_file, visible=True)
        if report_file
        else gr.update(visible=False)
    )

    yield [
        gr.update(choices=list(sessions.keys()), value=name),
        sessions,
        final_status,
        report_update,
    ] + toggle_ui(True)


def switch_session(name, sessions):
    if not name or name not in sessions:
        return [], "等待就绪...", gr.update(visible=False)
    session_data = sessions[name]
    report_update = (
        gr.update(value=session_data.get("report_path"), visible=True)
        if session_data.get("report_path")
        else gr.update(visible=False)
    )
    return session_data["history"], session_data["status"], report_update


def user_input_handler(user_msg, session_name, sessions):
    """将用户输入更新至对话历史"""
    if not session_name or session_name not in sessions:
        return "", sessions, []
    if not user_msg.strip():
        return "", sessions, sessions[session_name]["history"]

    history = sessions[session_name]["history"]
    history.append({"role": "user", "content": user_msg})
    sessions[session_name]["history"] = history
    return "", sessions, history


def bot_response_handler(current_session, sessions_store):
    """纯净推理模式 (不再进行实时探测)"""
    if not current_session or current_session not in sessions_store:
        yield [sessions_store, [], "🔴 未选择会话"] + toggle_ui(True)
        return

    history = sessions_store[current_session]["history"]
    if not history or history[-1]["role"] != "user":
        yield [sessions_store, history, "🟢 等待输入"] + toggle_ui(True)
        return

    user_prompt = str(history[-1]["content"])
    history.append({"role": "assistant", "content": ""})

    global global_model, global_tokenizer
    if global_model is None:
        history[-1]["content"] = "❌ 系统未连接，请在左侧初始化会话。"
        yield [sessions_store, history, "🔴 模型离线"] + toggle_ui(True)
        return

    # 锁定UI进入推理
    yield [sessions_store, history, "🟡 推理中..."] + toggle_ui(False)

    try:
        # 使用 Chat Template 构建标准化对话流
        messages = [{"role": "user", "content": user_prompt}]
        prompt_text = global_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = global_tokenizer([prompt_text], return_tensors="pt").to(
            global_model.device
        )

        with torch.no_grad():
            outputs = global_model.generate(**inputs, max_new_tokens=512)

        input_length = inputs.input_ids.shape[-1]
        response = global_tokenizer.decode(
            outputs[0][input_length:], skip_special_tokens=True
        )

        history[-1]["content"] = response.strip()
    except Exception as e:
        history[-1]["content"] = f"❌ 推理发生错误: {str(e)}"
        print(f"[Aegis] Inference Error: {str(e)}")

    sessions_store[current_session]["history"] = history
    yield [
        sessions_store,
        history,
        sessions_store[current_session]["status"],
    ] + toggle_ui(True)


# ==========================================
# 前端 UI 布局定义
# ==========================================
with gr.Blocks(title="Aegis-LoRA 免疫防线") as app:
    sessions_state = gr.State({})

    # 顶部全局 Banner
    gr.HTML("""
    <div style="padding: 16px; background: linear-gradient(135deg, #1a237e, #283593); border-radius: 10px; margin-bottom: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.15);">
        <h2 style="margin: 0; display: flex; justify-content: center; align-items: center; color: white !important; font-weight: 700; letter-spacing: 1px;">
            <span style="margin-right: 12px;">🛡️</span> Aegis-LoRA 控制中心
        </h2>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=3, elem_classes="sidebar-panel"):
            # [模块 A] 会话仓库
            with gr.Accordion(
                "📦 会话仓库", open=True, elem_classes=["wide-header", "no-scrollbar"]
            ):

                # 对齐设计：当前会话 (保留标题)
                gr.HTML(
                    "<div class='input-label'>当前会话</div>", elem_classes="label-wrap"
                )
                session_dropdown = gr.Dropdown(
                    choices=[], show_label=False, container=False
                )

                # 实时状态：隐藏自定义标题，直接显示文本框
                status_indicator = gr.Textbox(
                    show_label=False,
                    placeholder="等待就绪...",
                    interactive=False,
                    elem_classes="status-box",
                    container=False,
                )

            # [模块 B] 添加会话
            with gr.Accordion(
                "🔍 添加会话", open=True, elem_classes=["wide-header", "no-scrollbar"]
            ):
                gr.HTML(
                    "<div class='input-label'>会话名称</div>", elem_classes="label-wrap"
                )
                new_session_name = gr.Textbox(
                    show_label=False,
                    placeholder="e.g. Llama2-Safety-Test",
                    container=False,
                )

                gr.HTML(
                    "<div class='input-label'>基座模型路径</div>",
                    elem_classes="label-wrap",
                )
                with gr.Row(elem_classes="align-bottom-row"):
                    new_base_path = gr.Textbox(
                        show_label=False,
                        value=r".\models\Qwen2.5-3B-Instruct",
                        placeholder="选择路径...",
                        scale=10,
                        container=False,
                    )
                    base_btn = gr.Button("📂", scale=1, elem_classes="folder-btn")

                gr.HTML(
                    "<div class='input-label'>LoRA 适配器路径</div>",
                    elem_classes="label-wrap",
                )
                with gr.Row(elem_classes="align-bottom-row"):
                    new_lora_path = gr.Textbox(
                        show_label=False,
                        placeholder="选择路径...",
                        scale=10,
                        container=False,
                    )
                    lora_btn = gr.Button("📂", scale=1, elem_classes="folder-btn")

                create_btn = gr.Button("✨ 创建并初始化查杀", variant="primary")

        # ======================================
        # 右侧主区：核心监控与交互
        # ======================================
        with gr.Column(scale=7, elem_classes="main-workspace"):
            chatbot = gr.Chatbot(label="审计对话框", height=650, show_label=False)

            with gr.Row(elem_classes="align-bottom-row"):
                user_input = gr.Textbox(
                    show_label=False,
                    placeholder="输入指令内容...",
                    scale=9,
                    container=False,
                )
                send_btn = gr.Button(
                    "发送", variant="primary", scale=1, elem_classes="send-btn"
                )

            report_download = gr.File(label="📄 离线免疫报告", visible=False)

    # ==========================================
    # 事件绑定网络
    # ==========================================
    base_btn.click(fn=open_folder_dialog, outputs=new_base_path)
    lora_btn.click(fn=open_folder_dialog, outputs=new_lora_path)

    ui_components = [
        session_dropdown,
        new_session_name,
        new_base_path,
        base_btn,
        new_lora_path,
        lora_btn,
        create_btn,
        user_input,
        send_btn,
    ]

    create_btn.click(
        fn=create_new_session,
        inputs=[new_session_name, new_base_path, new_lora_path, sessions_state],
        outputs=[session_dropdown, sessions_state, status_indicator, report_download]
        + ui_components,
    )

    session_dropdown.change(
        fn=switch_session,
        inputs=[session_dropdown, sessions_state],
        outputs=[chatbot, status_indicator, report_download],
    )

    gr.on(
        triggers=[user_input.submit, send_btn.click],
        fn=user_input_handler,
        inputs=[user_input, session_dropdown, sessions_state],
        outputs=[user_input, sessions_state, chatbot],
        queue=False,
    ).then(
        fn=bot_response_handler,
        inputs=[session_dropdown, sessions_state],
        outputs=[sessions_state, chatbot, status_indicator] + ui_components,
        show_progress="hidden",
    )

if __name__ == "__main__":
    css_path = os.path.join(os.path.dirname(__file__), "style.css")
    custom_css = ""
    if os.path.exists(css_path):
        with open(css_path, "r", encoding="utf-8") as f:
            custom_css = f.read()

    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        theme=gr.themes.Soft(primary_hue="indigo"),
        css=custom_css,
    )
