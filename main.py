import gradio as gr
import torch
import gc
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from utils.pipeline import run_detection_pipeline, run_immunization_pipeline


# ==========================================
# 工具函数
# ==========================================
def open_folder_dialog():
    """调用本地资源管理器选择文件夹，返回路径字符串"""
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
        print(f"调用本地资源管理器失败: {e}")
        return ""


# ==========================================
# 自适应资源控制器
# ==========================================
class AegisEngine:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.base_path = ""
        self.current_lora = ""

    def free_memory(self):
        """释放模型占用的显存资源"""
        if self.model is not None:
            del self.model
        if self.tokenizer is not None:
            del self.tokenizer
        self.model = None
        self.tokenizer = None
        gc.collect()
        torch.cuda.empty_cache()

    def mount_if_needed(self, base_path, lora_path=""):
        """根据提供的路径自动加载模型和 tokenizer,如果当前已加载的配置与请求相同则直接返回在线状态。"""
        if (
            self.base_path == base_path
            and self.current_lora == lora_path
            and self.model is not None
        ):
            return "🟢 引擎在线 (显存已分配)"

        self.free_memory()
        self.base_path = base_path
        self.current_lora = lora_path

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                base_path, local_files_only=True
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            base_model = AutoModelForCausalLM.from_pretrained(
                base_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                local_files_only=True,
            )

            if lora_path and os.path.exists(lora_path):
                self.model = PeftModel.from_pretrained(
                    base_model, lora_path, is_trainable=True
                )
            else:
                self.model = base_model

            self.model.eval()
            return "🟢 引擎在线 (显存已分配)"
        except Exception as e:
            self.free_memory()
            return f"🔴 加载失败: {str(e)}"


engine = AegisEngine()


# ==========================================
# 核心业务逻辑
# ==========================================
def create_new_session(session_name, base_path, lora_path, sessions_store):
    """新建会话的核心逻辑，包含输入验证、状态更新和反馈消息生成"""
    if not session_name or not base_path:
        return gr.update(), sessions_store, "⚠️ 缺少会话名称或基座路径"
    if session_name in sessions_store:
        return gr.update(), sessions_store, "⚠️ 会话名称已存在"

    sessions_store[session_name] = {
        "base_path": base_path,
        "lora_path": lora_path,
        "history": [],
    }

    choices = list(sessions_store.keys())
    return (
        gr.update(choices=choices, value=session_name),
        sessions_store,
        f"🟢 引擎在线 (已加载: {session_name})",
    )


def switch_session(session_name, sessions):
    if not session_name or session_name not in sessions:
        return gr.update(value=[]), "请选择或创建一个会话", {"等待扫描": 0}

    session_data = sessions[session_name]
    engine.base_path = session_data["base_path"]
    engine.current_lora = session_data["lora_path"]

    status_msg = f"已切换至会话: {session_name}"

    # 自动安检逻辑：PEFTGuard 权重空间扫描
    if engine.current_lora:
        # 探测器路径
        detector_path = r"D:\Aegis_LoRA\models\detectors\peftguard_detector.pth"
        try:
            res = run_detection_pipeline(engine.current_lora, detector_path)
            prob = res.get("probability", 0)
            if res.get("is_poisoned"):
                security_info = {"【高危】检测到潜藏后门指纹": prob}
            else:
                security_info = {"【安全】模型特征分布正常": 1 - prob}
        except Exception as e:
            security_info = {f"扫描失败: {str(e)}": 0}

        return session_data["chat_history"], status_msg, security_info

    return session_data["chat_history"], status_msg, {"未挂载 LoRA": 0}


def user_input_handler(user_message, current_session, sessions_store):
    """处理用户输入的核心逻辑，包含输入验证、历史记录更新和反馈消息生成"""
    if not current_session or current_session not in sessions_store:
        return (
            "",
            sessions_store,
            sessions_store.get(current_session, {}).get("history", []),
        )

    history = sessions_store[current_session]["history"]
    history.append({"role": "user", "content": str(user_message)})
    sessions_store[current_session]["history"] = history
    return "", sessions_store, history


def bot_response_handler(
    current_session, sessions_store, p_threshold, l_threshold, out_dir="./reports"
):
    """
    核心业务流水线：实现实时侦测、全站 UI 锁定、显存隔离以及 BD-Vax 免疫重构。
    """

    # [辅助函数] 快速生成 9 个 UI 组件的交互状态，补齐 Gradio 的 13 个输出位
    def toggle_ui(interactive):
        return [gr.update(interactive=interactive) for _ in range(9)]

    # === 阶段 1: 环境检查 ===
    if not current_session or current_session not in sessions_store:
        yield sessions_store, [], gr.update(), "🔴 未选择会话", *toggle_ui(True)
        return

    history = sessions_store[current_session]["history"]
    user_prompt = str(history[-1]["content"])
    history.append({"role": "assistant", "content": ""})

    if engine.model is None:
        history[-1]["content"] = "❌ 系统未连接，请在左侧初始化会话。"
        yield sessions_store, history, gr.update(), "🔴 引擎离线", *toggle_ui(True)
        return

    yield sessions_store, history, gr.update(visible=False), "🟡 推理中...", *toggle_ui(
        False
    )

    # === 阶段 2: 实时监听侦测 ===
    # 执行 ConfGuard 侦测逻辑
    is_poisoned, generated_text, _ = confguard_detector(
        model=engine.model,
        tokenizer=engine.tokenizer,
        prompt=user_prompt,
        p_threshold=p_threshold,
        l_threshold=l_threshold,
    )

    # 分支 A: 一切正常
    if not is_poisoned:
        history[-1]["content"] = generated_text
        sessions_store[current_session]["history"] = history
        yield sessions_store, history, gr.update(
            visible=False
        ), "🟢 引擎在线", *toggle_ui(True)
        return

    # === 阶段 3: 安全熔断 (触发查杀) ===
    warning_msg = (
        generated_text
        + "\n\n---\n🚨 **安全熔断**：侦测到高危后门响应。\n⚙️ *系统已挂起，正在执行 BD-Vax 深度免疫重构，期间所有操作已锁定...*"
    )
    history[-1]["content"] = warning_msg
    yield sessions_store, history, gr.update(
        visible=False
    ), "🟡 引擎挂起 (后台查杀中...)", *toggle_ui(False)

    report_path = None

    try:
        base_path = engine.base_path
        lora_path = engine.current_lora
        engine.free_memory()
        # 执行免疫重构流水线，返回报告路径、切除特征数量和免疫后的 LoRA 路径
        report_path, suppressed_count, cleansed_lora_path = run_immunization_pipeline(
            base_model_path=base_path,
            lora_path=lora_path,
            dataset_path="./datasets/clean_data.json",
            tau=0.35,
        )

        # === 阶段 4: 重新上线与解锁 ===
        sessions_store[current_session]["lora_path"] = cleansed_lora_path
        engine.mount_if_needed(base_path, cleansed_lora_path)
        history[-1][
            "content"
        ] += f"\n\n---\n✅ **免疫完成**：已切除 {suppressed_count} 个特征载体。模型已重新上线，操作锁定解除。"
        yield sessions_store, history, gr.update(
            value=report_path, visible=True
        ), "🟢 引擎在线 (已免疫)", *toggle_ui(True)

    except Exception as e:
        history[-1][
            "content"
        ] += f"\n\n❌ **免疫重构失败**: {str(e)}\n系统已自动回滚至初始状态。"
        engine.mount_if_needed(base_path, lora_path)
        yield sessions_store, history, gr.update(
            visible=False
        ), "🔴 免疫流程异常", *toggle_ui(True)


# ==========================================
# 前端 UI 布局定义
# ==========================================
with gr.Blocks(title="Aegis-LoRA 免疫防线") as app:
    sessions_state = gr.State({})

    # 顶部全局 Banner
    gr.HTML("""
    <div style="padding: 16px; background: linear-gradient(135deg, #1a237e, #283593); border-radius: 10px; margin-bottom: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.15);">
        <h2 style="margin: 0; display: flex; align-items: center; color: white !important; font-weight: 700; letter-spacing: 1px;">
            <span style="margin-right: 12px;">🛡️</span> Aegis-LoRA 安全终端
        </h2>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=3, elem_classes="sidebar-panel"):
            # [模块 A] 终端管理
            gr.Markdown("### 📂 终端管理")

            with gr.Group():
                session_dropdown = gr.Dropdown(choices=[], label="当前激活会话")
                status_indicator = gr.Textbox(
                    show_label=False,
                    placeholder="⚪ 系统待机",
                    interactive=False,
                    elem_classes="status-box",
                )

            gr.HTML(
                "<div style='margin: 4px 0; border-top: 1px solid var(--border-color-primary); opacity: 0.4;'></div>"
            )

            # [模块 B] 新建会话
            with gr.Accordion(
                "➕ 新建会话", open=True, elem_classes=["wide-header", "no-scrollbar"]
            ):
                gr.HTML(
                    "<div class='input-label'>会话标识</div>", elem_classes="label-wrap"
                )
                new_session_name = gr.Textbox(
                    show_label=False, placeholder="标识符", container=False
                )

                gr.HTML(
                    "<div class='input-label'>基座模型</div>", elem_classes="label-wrap"
                )
                with gr.Row(elem_classes="align-bottom-row"):
                    new_base_path = gr.Textbox(
                        show_label=False,
                        value="./models/Qwen2.5-3B-Instruct",
                        placeholder="选择路径...",
                        scale=10,
                        container=False,
                    )
                    base_btn = gr.Button("📂", scale=1, elem_classes="folder-btn")

                gr.HTML(
                    "<div class='input-label'>LoRA 权重</div>",
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

                create_btn = gr.Button("初始化安全会话", variant="primary")

            # [模块 C] 探测引擎参数
            with gr.Accordion(
                "⚙️ 探测引擎参数",
                open=False,
                elem_classes=["wide-header", "no-scrollbar"],
            ):
                p_slider = gr.Slider(0.9, 0.99, value=0.95, label="锁定置信度阈值 (P)")
                l_slider = gr.Slider(3, 15, value=5, step=1, label="触发序列长度 (L)")

        # ======================================
        # 右侧主区：核心监控与交互
        # ======================================
        with gr.Column(scale=7, elem_classes="main-workspace"):
            chatbot = gr.Chatbot(label="Aegis 监测视窗", height=690)

            with gr.Row(elem_classes="align-bottom-row"):
                user_input = gr.Textbox(
                    show_label=False,
                    placeholder="输入测试指令...",
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
    # 绑定创建会话事件
    create_btn.click(
        fn=create_new_session,
        inputs=[new_session_name, new_base_path, new_lora_path, sessions_state],
        outputs=[session_dropdown, sessions_state, status_indicator],
    )
    # 绑定切换会话事件
    session_dropdown.change(
        fn=switch_session,
        inputs=[session_dropdown, sessions_state],
        outputs=[chatbot, status_indicator],
    )
    # 绑定用户输入事件
    gr.on(
        triggers=[user_input.submit, send_btn.click],
        fn=user_input_handler,
        inputs=[user_input, session_dropdown, sessions_state],
        outputs=[user_input, sessions_state, chatbot],
        queue=False,
    ).then(
        fn=bot_response_handler,
        inputs=[session_dropdown, sessions_state, p_slider, l_slider],
        outputs=[sessions_state, chatbot, report_download, status_indicator]
        + ui_components,
        show_progress="hidden",
    )

if __name__ == "__main__":
    # 加载自定义 CSS 样式
    css_path = os.path.join(os.path.dirname(__file__), "scripts", "style.css")
    with open(css_path, "r", encoding="utf-8") as f:
        custom_css = f.read()
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        theme=gr.themes.Soft(primary_hue="indigo"),
        css=custom_css,
    )
