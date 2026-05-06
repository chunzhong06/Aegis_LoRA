# -*- coding: utf-8 -*-
import gradio as gr
import torch
import gc
import os
import json
import base64
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from utils.pipeline import run_static_scan_pipeline, run_immunization_pipeline

# ==========================================
# 本地持久化与缓存管理
# ==========================================
CACHE_DIR = ".cache"
HISTORY_FILE = os.path.join(CACHE_DIR, "sessions_history.json")

os.makedirs(CACHE_DIR, exist_ok=True)


def load_sessions():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def save_sessions(sessions):
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(sessions, f, ensure_ascii=False, indent=4)
    except Exception:
        pass


def restore_report_from_cache(session_name, session_data):
    report_data = session_data.get("report_data")
    if not report_data:
        return None
    reports_dir = os.path.join(CACHE_DIR, "reports")
    os.makedirs(reports_dir, exist_ok=True)
    ext = session_data.get("report_ext", ".txt")
    filepath = os.path.join(reports_dir, f"{session_name}_audit_report{ext}")
    try:
        with open(filepath, "wb") as f:
            f.write(base64.b64decode(report_data))
        return filepath
    except Exception:
        return None


def encode_report_file(report_file):
    if report_file and os.path.exists(report_file):
        try:
            with open(report_file, "rb") as f:
                encoded = base64.b64encode(f.read()).decode("utf-8")
            _, ext = os.path.splitext(report_file)
            return encoded, ext
        except Exception:
            pass
    return None, ""


# ==========================================
# 全局模型状态与显存引擎
# ==========================================
global_model = None
global_tokenizer = None
global_base_path = ""
global_lora_path = ""


def free_memory():
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
# 局部工具函数
# ==========================================
def open_folder_dialog():
    import tkinter as tk
    from tkinter import filedialog

    try:
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        path = filedialog.askdirectory(title="选择本地文件夹")
        root.destroy()
        return path if path else ""
    except Exception:
        return ""


def toggle_ui(interactive):
    return [gr.update(interactive=interactive) for _ in range(10)]


# ==========================================
# 核心业务事件响应器
# ==========================================
def create_new_session(name, base_path, lora_path, sessions):
    if not name or not base_path or not lora_path:
        yield [gr.update(), sessions, "⚠️ 配置缺失", gr.update()] + toggle_ui(True)
        return
    if name in sessions:
        yield [gr.update(), sessions, "⚠️ 名称重复", gr.update()] + toggle_ui(True)
        return

    yield [gr.update(), sessions, "🛡️ 静态查杀与加载中...", gr.update()] + toggle_ui(
        False
    )

    active_lora_path = lora_path
    report_file = None
    is_poisoned = False

    try:
        is_poisoned, prob = run_static_scan_pipeline(lora_path)
        is_poisoned = bool(is_poisoned)
        if is_poisoned:
            yield [
                gr.update(),
                sessions,
                "🛡️ 拦截异常特征，正在执行 BD-Vax 免疫重构...",
                gr.update(),
            ] + toggle_ui(False)
            report_path, suppressed_count, cleansed_path = run_immunization_pipeline(
                base_model_path=base_path,
                lora_path=lora_path,
                dataset_path="./datasets/clean_data.json",
            )
            active_lora_path = cleansed_path
            report_file = report_path
            scan_res = f"🔴 发现后门 (已切除 {suppressed_count} 个特征)"
        else:
            scan_res = "🟢 安全"
    except Exception as e:
        scan_res = "🟡 流程异常"

    load_res = load_model_direct(base_path, active_lora_path)
    final_status = f"{scan_res}     | {load_res}"

    r_data, r_ext = encode_report_file(report_file)
    sessions[name] = {
        "base_path": str(base_path),
        "lora_path": str(active_lora_path),
        "history": [],
        "is_poisoned": is_poisoned,
        "status": final_status,
        "report_data": r_data,
        "report_ext": r_ext,
    }
    save_sessions(sessions)

    path = restore_report_from_cache(name, sessions[name])
    yield [
        gr.update(choices=list(sessions.keys()), value=name),
        sessions,
        final_status,
        gr.update(value=path, visible=True) if path else gr.update(visible=False),
    ] + toggle_ui(True)


def delete_current_session(name, sessions):
    if name in sessions:
        del sessions[name]
        save_sessions(sessions)
    choices = list(sessions.keys())
    new_val = choices[0] if choices else None
    if new_val:
        data = sessions[new_val]
        path = restore_report_from_cache(new_val, data)
        return (
            gr.update(choices=choices, value=new_val),
            sessions,
            data["history"],
            data["status"],
            gr.update(value=path, visible=True) if path else gr.update(visible=False),
        )
    else:
        free_memory()
        return (
            gr.update(choices=[], value=None),
            sessions,
            [],
            "等待就绪...",
            gr.update(visible=False),
        )


def switch_session(name, sessions):
    if not name or name not in sessions:
        return [], "等待就绪...", gr.update(visible=False)
    data = sessions[name]
    path = restore_report_from_cache(name, data)
    return (
        data["history"],
        data["status"],
        gr.update(value=path, visible=True) if path else gr.update(visible=False),
    )


def chat_handler(user_msg, session_name, sessions):
    if not session_name or session_name not in sessions:
        return "", sessions, []
    if not user_msg.strip():
        return "", sessions, sessions[session_name]["history"]
    history = sessions[session_name]["history"]
    history.append({"role": "user", "content": user_msg})
    sessions[session_name]["history"] = history
    save_sessions(sessions)
    return "", sessions, history


def bot_handler(current_session, sessions_store):
    if not current_session or current_session not in sessions_store:
        yield [sessions_store, [], "🔴 未选择会话"] + toggle_ui(True)
        return
    history = sessions_store[current_session]["history"]
    if not history or history[-1]["role"] != "user":
        yield [sessions_store, history, "🟢 模型在线"] + toggle_ui(True)
        return

    history.append({"role": "assistant", "content": ""})
    if global_model is None:
        history[-1]["content"] = "❌ 底层模型未连接，请先初始化或切换会话。"
        yield [sessions_store, history, "🔴 模型离线"] + toggle_ui(True)
        return

    yield [sessions_store, history, "🟡 推理中..."] + toggle_ui(False)

    try:
        messages = [{"role": "user", "content": history[-2]["content"]}]
        prompt = global_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = global_tokenizer([prompt], return_tensors="pt").to(global_model.device)
        with torch.no_grad():
            outputs = global_model.generate(**inputs, max_new_tokens=512)
        response = global_tokenizer.decode(
            outputs[0][inputs.input_ids.shape[-1] :], skip_special_tokens=True
        )
        history[-1]["content"] = response.strip()
    except Exception as e:
        history[-1]["content"] = f"❌ 硬件推理层故障: {e}"

    sessions_store[current_session]["history"] = history
    save_sessions(sessions_store)
    yield [
        sessions_store,
        history,
        sessions_store[current_session]["status"],
    ] + toggle_ui(True)


# ==========================================
# 前端 UI 布局定义 (全原生无 CSS 版)
# ==========================================
# 提取内联样式，代替外部 CSS 实现紫底徽章
badge_style = "background-color: #6366f1; color: white; padding: 4px 10px; border-radius: 6px; font-size: 0.85em; font-weight: bold; display: inline-block; margin-bottom: 4px;"

custom_theme = gr.themes.Soft(
    primary_hue="indigo",
    spacing_size="sm",
    radius_size="md",
)

with gr.Blocks(theme=custom_theme, title="Aegis-LoRA 免疫防线") as app:
    init_data = load_sessions()
    sessions_state = gr.State(init_data)
    choices = list(init_data.keys())
    curr_val = choices[0] if choices else None

    gr.HTML("""
    <div style="padding: 16px; background: linear-gradient(135deg, #1a237e, #283593); border-radius: 10px; margin-bottom: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.15);">
        <h2 style="margin: 0; text-align: center; color: white; font-weight: 700; letter-spacing: 1px;">
            🛡️ Aegis-LoRA 控制中心
        </h2>
    </div>
    """)

    with gr.Row():
        # --- 左侧控制侧边栏 ---
        with gr.Column(scale=3):
            # [模块 A] 会话仓库
            with gr.Accordion("📦 会话仓库", open=True):
                gr.HTML(f"<div style='{badge_style}'>当前会话</div>")
                with gr.Row():
                    session_dropdown = gr.Dropdown(
                        choices=choices,
                        value=curr_val,
                        show_label=False,
                        container=False,
                        scale=5,
                    )
                    delete_btn = gr.Button("🗑️", variant="stop", scale=1)

                status_indicator = gr.Textbox(
                    show_label=False, placeholder="等待就绪...", interactive=False
                )

            # [模块 B] 添加会话
            with gr.Accordion("🔍 添加会话", open=True):
                gr.HTML(f"<div style='{badge_style}'>会话名称</div>")
                new_name = gr.Textbox(
                    show_label=False, placeholder="标识符...", lines=1, max_lines=1
                )

                gr.HTML(f"<div style='{badge_style}'>基座模型路径</div>")
                with gr.Row():
                    new_base = gr.Textbox(
                        show_label=False,
                        value=r".\models\Qwen2.5-3B-Instruct",
                        container=False,
                        scale=5,
                        lines=1,
                        max_lines=1,
                    )
                    base_btn = gr.Button("📂", scale=1)

                gr.HTML(f"<div style='{badge_style}'>LoRA 适配器路径</div>")
                with gr.Row():
                    new_lora = gr.Textbox(
                        show_label=False,
                        placeholder="选择路径...",
                        container=False,
                        scale=5,
                        lines=1,
                        max_lines=1,
                    )
                    lora_btn = gr.Button("📂", scale=1)

                create_btn = gr.Button("✨ 创建并初始化查杀", variant="primary")

        # --- 右侧主诊断区 ---
        with gr.Column(scale=7):
            chatbot = gr.Chatbot(label="审计视窗", height=650)

            # 原生自动对齐的输入组合
            with gr.Row():
                user_input = gr.Textbox(
                    show_label=False, placeholder="输入指令...", scale=9
                )
                send_btn = gr.Button("发送", variant="primary", scale=1)

            report_download = gr.File(label="📄 离线免疫报告", visible=False)

    # ==========================================
    # 数据流事件绑定网络
    # ==========================================
    base_btn.click(fn=open_folder_dialog, outputs=new_base)
    lora_btn.click(fn=open_folder_dialog, outputs=new_lora)

    ui_list = [
        session_dropdown,
        delete_btn,
        new_name,
        new_base,
        base_btn,
        new_lora,
        lora_btn,
        create_btn,
        user_input,
        send_btn,
    ]

    create_btn.click(
        fn=create_new_session,
        inputs=[new_name, new_base, new_lora, sessions_state],
        outputs=[session_dropdown, sessions_state, status_indicator, report_download]
        + ui_list,
    )
    delete_btn.click(
        fn=delete_current_session,
        inputs=[session_dropdown, sessions_state],
        outputs=[
            session_dropdown,
            sessions_state,
            chatbot,
            status_indicator,
            report_download,
        ],
    )
    session_dropdown.change(
        fn=switch_session,
        inputs=[session_dropdown, sessions_state],
        outputs=[chatbot, status_indicator, report_download],
    )
    gr.on(
        triggers=[user_input.submit, send_btn.click],
        fn=chat_handler,
        inputs=[user_input, session_dropdown, sessions_state],
        outputs=[user_input, sessions_state, chatbot],
        queue=False,
    ).then(
        fn=bot_handler,
        inputs=[session_dropdown, sessions_state],
        outputs=[sessions_state, chatbot, status_indicator] + ui_list,
        show_progress="hidden",
    )
    app.load(
        fn=switch_session,
        inputs=[session_dropdown, sessions_state],
        outputs=[chatbot, status_indicator, report_download],
    )

if __name__ == "__main__":
    app.launch(server_name="127.0.0.1", server_port=7860)
