# -*- coding: utf-8 -*-
# Aegis-LoRA - 主程序入口
import gradio as gr
import torch
import gc
import os
import json
import base64
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# 导入特征扫描与免疫清洗算法
from utils.pipeline import (
    run_static_scan_pipeline,
    run_immunization_pipeline,
    run_fast_cleanse_pipeline,
)

# ==========================================
# 模块：本地持久化与缓存管理
# 作用：负责会话状态的本地保存、读取，以及防丢失的报告 Base64 编解码
# ==========================================
# 本地缓存路径
CACHE_DIR = ".cache"
# 会话历史记录文件
HISTORY_FILE = os.path.join(CACHE_DIR, "sessions_history.json")

# 确保本地缓存目录存在
os.makedirs(CACHE_DIR, exist_ok=True)


def load_sessions():
    """从本地 JSON 加载所有历史会话数据"""
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def save_sessions(sessions):
    """将会话字典持久化到本地 JSON"""
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(sessions, f, ensure_ascii=False, indent=4)
    except Exception:
        pass


def get_report_path(session_data):
    """轻量级读取：验证报告路径是否存在，若丢失则返回 None 告知 UI 隐藏"""
    path = session_data.get("report_path")
    if path and os.path.exists(path):
        return path
    return None


def process_and_link_report(session_name, report_file):
    """将底层的离线报告重命名为与会话绑定的固定名称，并返回精简的物理路径"""
    if report_file and os.path.exists(report_file):
        reports_dir = os.path.join(CACHE_DIR, "reports")
        os.makedirs(reports_dir, exist_ok=True)

        _, ext = os.path.splitext(report_file)
        new_path = os.path.join(reports_dir, f"{session_name}_audit_report{ext}")

        try:
            # 如果之前有同名报告，先将其覆盖删除
            if os.path.exists(new_path):
                os.remove(new_path)
            # 将底层生成的时间戳报告移动并重命名为会话名称
            os.rename(report_file, new_path)

            # 清理附带生成的冗余 JSON 文件（如果存在）
            old_json = report_file.replace(ext, ".json")
            if os.path.exists(old_json):
                os.remove(old_json)

            return new_path
        except Exception:
            # 若因权限问题重命名失败，降级返回原始时间戳路径
            return report_file
    return None


# ==========================================
# 模块：全局模型状态与显存引擎
# 作用：管理底层大模型的加载、卸载与显存监控
# ==========================================
# 维护全局单例模型，避免重复加载导致 OOM
global_model = None
global_tokenizer = None
global_base_path = ""
global_lora_path = ""


def free_memory():
    """彻底释放当前挂载的模型与分词器，清空 GPU 显存缓存"""
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
    """执行底层模型加载逻辑：挂载基座大模型并按需合并 LoRA 适配器"""
    global global_model, global_tokenizer, global_base_path, global_lora_path

    # 若模型路径未变且处于活跃状态，则直接复用
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
        # 加载分词器
        global_tokenizer = AutoTokenizer.from_pretrained(
            base_path, local_files_only=True
        )
        if global_tokenizer.pad_token is None:
            global_tokenizer.pad_token = global_tokenizer.eos_token

        # 加载基座模型 (BF16 精度，自动分配设备)
        base_model = AutoModelForCausalLM.from_pretrained(
            base_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            local_files_only=True,
        )

        # 挂载 LoRA 权重
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
# 模块：局部工具函数
# 作用：提供纯前端交互的辅助支持
# ==========================================
def open_folder_dialog():
    """调用系统底层接口，弹出原生文件夹选择对话框"""
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
    """状态控制器：批量锁定或解锁 10 个前端输入组件，防止异步冲突"""
    return [gr.update(interactive=interactive) for _ in range(11)]


# ==========================================
# 模块：核心业务事件响应器
# 作用：承接 Gradio 前端的所有按钮点击与交互事件
# ==========================================
def create_new_session(name, base_path, lora_path, cleanse_mode, sessions):
    """核心流水线:创建会话 -> 静态扫描后门 -> (若异常)触发免疫重构 -> 上线模型"""
    if not name or not base_path or not lora_path:
        yield [gr.update(), sessions, "⚠️ 配置缺失", gr.update()] + toggle_ui(True)
        return
    if name in sessions:
        yield [gr.update(), sessions, "⚠️ 名称重复", gr.update()] + toggle_ui(True)
        return

    # 锁定前端 UI
    yield [gr.update(), sessions, "🛡️ 静态查杀与加载中...", gr.update()] + toggle_ui(
        False
    )

    active_lora_path = lora_path
    report_file = None
    is_poisoned = False

    try:
        # 阶段 1：特征探测
        is_poisoned, prob = run_static_scan_pipeline(lora_path)
        is_poisoned = bool(is_poisoned)

        if is_poisoned:
            yield [
                gr.update(),
                sessions,
                "🛡️ 拦截异常特征，正在执行重构手术...",
                gr.update(),
            ] + toggle_ui(False)

            # 阶段 2：定向清洗与重构 (路由选择)
            if "极速" in cleanse_mode:
                # 调用极速清洗
                report_path, suppressed_count, cleansed_path = (
                    run_fast_cleanse_pipeline(
                        base_model_path=base_path,
                        lora_path=lora_path,
                        signature_path="./datasets/qwen_multidomain_signatures.pt",  # 使用预构建的多域签名库
                        recovery_data_path="./datasets/clean_data_recovery.json",
                    )
                )
            else:
                # 调用深度免疫
                report_path, suppressed_count, cleansed_path = (
                    run_immunization_pipeline(
                        base_model_path=base_path,
                        lora_path=lora_path,
                        variant_data_path="./datasets/clean_data_variants.json",
                        recovery_data_path="./datasets/clean_data_recovery.json",
                    )
                )

            active_lora_path = cleansed_path
            report_file = report_path
            scan_res = f"🔴 发现后门 (已切除 {suppressed_count} 个特征)"
        else:
            scan_res = "🟢 安全"
    except Exception as e:
        scan_res = "🟡 流程异常"

    # 阶段 3：挂载安全的模型权重
    load_res = load_model_direct(base_path, active_lora_path)
    final_status = f"{scan_res} | {load_res}"

    # 阶段 4：将生成报告转为 Base64 并构建会话对象
    final_report_path = process_and_link_report(name, report_file)

    sessions[name] = {
        "base_path": str(base_path),
        "lora_path": str(active_lora_path),
        "history": [],
        "is_poisoned": is_poisoned,
        "status": final_status,
        "report_path": final_report_path,  # 现在只保存一个极简的字符串路径
    }
    save_sessions(sessions)

    # 验证物理文件是否存在并解锁 UI
    path = get_report_path(sessions[name])


def delete_current_session(name, sessions):
    """销毁逻辑:删除指定会话历史记录，并根据库存状态决定是否清空显存"""
    if name in sessions:
        del sessions[name]
        save_sessions(sessions)

    choices = list(sessions.keys())
    new_val = choices[0] if choices else None

    if new_val:
        # 若仍有会话，自动切换至首个会话
        data = sessions[new_val]
        path = get_report_path(new_val, data)
        return (
            gr.update(choices=choices, value=new_val),
            sessions,
            data["history"],
            data["status"],
            gr.update(value=path, visible=True) if path else gr.update(visible=False),
        )
    else:
        # 若库已空，彻底释放底层硬件资源
        free_memory()
        return (
            gr.update(choices=[], value=None),
            sessions,
            [],
            "等待就绪...",
            gr.update(visible=False),
        )


def switch_session(name, sessions):
    """切换逻辑:轻量级视图切换，无缝加载选中会话的聊天历史与报告"""
    if not name or name not in sessions:
        return [], "等待就绪...", gr.update(visible=False)
    data = sessions[name]
    path = get_report_path(name, data)
    return (
        data["history"],
        data["status"],
        gr.update(value=path, visible=True) if path else gr.update(visible=False),
    )


def chat_handler(user_msg, session_name, sessions):
    """对话捕获:处理用户输入框内容，并追加到本地 JSON 历史队列中"""
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
    """核心推理:处理模型生成任务，包含对话模板应用与自回归解码"""
    if not current_session or current_session not in sessions_store:
        yield [sessions_store, [], "🔴 未选择会话"] + toggle_ui(True)
        return

    history = sessions_store[current_session]["history"]
    if not history or history[-1]["role"] != "user":
        yield [sessions_store, history, "🟢 模型在线"] + toggle_ui(True)
        return

    history.append({"role": "assistant", "content": ""})

    # 鉴权：检查模型是否已加载
    if global_model is None:
        history[-1]["content"] = "❌ 底层模型未连接，请先初始化或切换会话。"
        yield [sessions_store, history, "🔴 模型离线"] + toggle_ui(True)
        return

    yield [sessions_store, history, "🟡 推理中..."] + toggle_ui(False)

    try:
        # 构建符合官方标准的对话上下文模板
        messages = [{"role": "user", "content": history[-2]["content"]}]
        prompt = global_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        # Tokenize 并输送至 GPU
        inputs = global_tokenizer([prompt], return_tensors="pt").to(global_model.device)

        # 闭环生成推理
        with torch.no_grad():
            outputs = global_model.generate(**inputs, max_new_tokens=512)

        # 解码并截断 prompt 仅保留新生成的 response
        response = global_tokenizer.decode(
            outputs[0][inputs.input_ids.shape[-1] :], skip_special_tokens=True
        )
        history[-1]["content"] = response.strip()
    except Exception as e:
        history[-1]["content"] = f"❌ 硬件推理层故障: {e}"

    # 保存最终结果并解锁 UI
    sessions_store[current_session]["history"] = history
    save_sessions(sessions_store)
    yield [
        sessions_store,
        history,
        sessions_store[current_session]["status"],
    ] + toggle_ui(True)


# ==========================================
# 模块：前端 UI 布局定义
# 作用：基于 Gradio 构建跨平台统一界面
# ==========================================

# 自定义内联样式：用于渲染模块标题的靛蓝色白字徽章标签
badge_style = "background-color: #6366f1; color: white; padding: 4px 10px; border-radius: 6px; font-size: 0.85em; font-weight: bold; display: inline-block; margin-bottom: 4px; margin-top: 8px;"

# 调用官方 Themes API 设置全局主题、紧凑型间距和圆角
custom_theme = gr.themes.Soft(
    primary_hue="indigo",
    spacing_size="sm",
    radius_size="md",
).set(
    block_background_fill="*background_fill_primary",
    block_border_width="1px",
)

# 构建主页面大纲
with gr.Blocks(title="Aegis-LoRA 免疫防线") as app:
    # 状态初始化
    init_data = load_sessions()
    sessions_state = gr.State(init_data)
    choices = list(init_data.keys())
    curr_val = choices[0] if choices else None

    # 顶部全局大标题 (HTML)
    gr.HTML("""
    <div style="padding: 16px; background: linear-gradient(135deg, #1a237e, #283593); border-radius: 10px; margin-bottom: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.15);">
        <h2 style="margin: 0; text-align: center; color: white !important; font-weight: 700; letter-spacing: 1px;">
            🛡️ Aegis-LoRA 控制中心
        </h2>
    </div>
    """)

    # 主体横向分割 (左控右显 3:7 比例)
    with gr.Row(equal_height=True):

        # --- 左侧：控制侧边栏 ---
        with gr.Column(scale=3):

            # [模块 A] 会话仓库管理区
            with gr.Accordion("📦 会话仓库", open=True):
                gr.HTML(f"<div style='{badge_style}'>当前会话</div>")

                with gr.Group():
                    with gr.Row():
                        session_dropdown = gr.Dropdown(
                            choices=choices,
                            value=curr_val,
                            show_label=False,
                            container=False,
                            scale=7,
                        )
                        delete_btn = gr.Button(
                            "🗑️", variant="stop", scale=1, min_width=1
                        )

                # 实时状态指示文本框
                status_indicator = gr.Textbox(
                    show_label=False, placeholder="等待就绪...", interactive=False
                )

            # [模块 B] 新建会话配置区
            with gr.Accordion("🔍 添加会话", open=True):
                gr.HTML(f"<div style='{badge_style}'>会话名称</div>")
                new_name = gr.Textbox(
                    show_label=False, placeholder="标识符...", lines=1, max_lines=1
                )

                gr.HTML(f"<div style='{badge_style}'>基座模型路径</div>")
                with gr.Group():
                    with gr.Row():
                        new_base = gr.Textbox(
                            show_label=False,
                            value=r".\models\Qwen2.5-3B-Instruct",
                            container=False,
                            scale=7,
                            lines=1,
                            max_lines=1,
                        )
                        base_btn = gr.Button("📂", scale=1, min_width=1)

                gr.HTML(f"<div style='{badge_style}'>LoRA 适配器路径</div>")
                with gr.Group():
                    with gr.Row():
                        new_lora = gr.Textbox(
                            show_label=False,
                            placeholder="选择路径...",
                            container=False,
                            scale=7,
                            lines=1,
                            max_lines=1,
                        )
                        lora_btn = gr.Button("📂", scale=1, min_width=1)

                gr.HTML(f"<div style='{badge_style}'>清洗模式</div>")
                cleanse_mode_radio = gr.Radio(
                    choices=[
                        "🧬 深度多域免疫",
                        "⚡️ 极速免疫查杀",
                    ],
                    value="🧬 深度多域免疫",
                    show_label=False,
                    container=False,
                )

                create_btn = gr.Button("✨ 创建并初始化查杀", variant="primary")

        # --- 右侧：主诊断区对话视窗 ---
        with gr.Column(scale=7):
            chatbot = gr.Chatbot(label="审计视窗", height=650)

            # 底部指令发送区
            with gr.Group():
                with gr.Row():
                    user_input = gr.Textbox(
                        show_label=False,
                        placeholder="输入指令...",
                        container=False,
                        scale=9,
                        lines=2,
                        max_lines=10,
                    )
                    send_btn = gr.Button(
                        "发送", variant="primary", scale=1, min_width=1
                    )

            # 隐藏的文件下载桥梁
            report_download = gr.File(label="📄 离线免疫报告", visible=False)

    # ==========================================
    # 模块：数据流与事件网络绑定
    # ==========================================
    # 路径选择器单击事件
    base_btn.click(fn=open_folder_dialog, outputs=new_base)
    lora_btn.click(fn=open_folder_dialog, outputs=new_lora)

    # 声明需锁定的 UI 组件清单
    ui_list = [
        session_dropdown,
        delete_btn,
        new_name,
        new_base,
        base_btn,
        new_lora,
        lora_btn,
        cleanse_mode_radio,
        create_btn,
        user_input,
        send_btn,
    ]

    # 初始化大模型并触发免疫流程
    create_btn.click(
        fn=create_new_session,
        inputs=[new_name, new_base, new_lora, cleanse_mode_radio, sessions_state],
        outputs=[session_dropdown, sessions_state, status_indicator, report_download]
        + ui_list,
    )

    # 销毁选中会话
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

    # 切换会话下拉框事件
    session_dropdown.change(
        fn=switch_session,
        inputs=[session_dropdown, sessions_state],
        outputs=[chatbot, status_indicator, report_download],
    )

    # 对话回车/点击双向绑定链：先拦截文本，再调用 LLM 推理
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

    # 程序冷启动：视图装载第一个激活的缓存会话
    app.load(
        fn=switch_session,
        inputs=[session_dropdown, sessions_state],
        outputs=[chatbot, status_indicator, report_download],
    )

# 程序入口
if __name__ == "__main__":
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        theme=custom_theme,
    )
