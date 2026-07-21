# Aegis-LoRA - WebUI 控制台
import gc
import json
import sys
from functools import partial
from pathlib import Path

# 工程内资源统一从当前模块定位，不依赖启动命令所在目录。
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import gradio as gr
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# 导入流水线函数，负责静态扫描、深度免疫和极速清洗
from utils.pipeline import (
    run_static_scan_pipeline,
    run_immunization_pipeline,
    run_fast_cleanse_pipeline,
)

# =====================================================================
# 配置与运行状态
# =====================================================================
# WebUI 报告独立归档
(ROOT / ".cache" / "webui_reports").mkdir(parents=True, exist_ok=True)

# WebUI 同一时间只保留一组模型；model_key 用于避免重复加载相同权重。
global_model = None
global_tokenizer = None
global_model_key = None


def sync_sessions(sessions=None):
    """读取或原子保存历史会话。"""
    # history_file 是 WebUI 历史仓库的唯一持久化入口。
    history_file = ROOT / ".cache" / "sessions_history.json"
    try:
        # 1. 未传 sessions 时读取历史；根节点异常时回退为空字典。
        if sessions is None:
            data = (
                json.loads(history_file.read_text(encoding="utf-8"))
                if history_file.is_file()
                else {}
            )
            return data if isinstance(data, dict) else {}

        # 2. 临时文件与正式文件位于同一目录，完整写入后再原子替换。
        temp_file = history_file.with_suffix(".tmp")
        temp_file.write_text(
            json.dumps(sessions, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temp_file.replace(history_file)
    except (OSError, ValueError, TypeError) as exc:
        # 历史损坏或写入失败不阻断 WebUI，本轮内存状态仍可继续使用。
        print(f"      [警告] WebUI 历史会话同步失败: {exc}")
        return {} if sessions is None else sessions
    return sessions


def load_model(base_path="", lora_path=""):
    """切换当前模型；空路径用于释放模型。"""
    global global_model, global_tokenizer, global_model_key

    # 1. 空路径表示当前没有活动会话，同时释放模型引用与显存缓存。
    if not base_path:
        global_model = global_tokenizer = global_model_key = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return "🔴 模型离线"

    # 2. 历史记录允许保存相对路径，加载前统一还原为工程内绝对路径。
    base_path = Path(base_path).expanduser()
    lora_path = Path(lora_path).expanduser()
    base_path = (base_path if base_path.is_absolute() else ROOT / base_path).resolve()
    lora_path = (lora_path if lora_path.is_absolute() else ROOT / lora_path).resolve()
    # 当前模型键与目标一致时直接复用，避免重复挂载造成显存峰值。
    if global_model is not None and global_model_key == (base_path, lora_path):
        return "🟢 模型在线"

    # 3. 加载新模型前先清理旧实例，确保进程中只存在一套活动权重。
    global_model = global_tokenizer = global_model_key = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    try:
        # 4. 分词器与基座模型只允许读取本地资源，不触发在线下载。
        global_tokenizer = AutoTokenizer.from_pretrained(
            base_path, local_files_only=True, trust_remote_code=True
        )
        if global_tokenizer.pad_token is None:
            global_tokenizer.pad_token = global_tokenizer.eos_token

        # CUDA 使用 BF16 与自动设备映射；CPU 回退到 FP32。
        use_cuda = torch.cuda.is_available()
        base_model = AutoModelForCausalLM.from_pretrained(
            base_path,
            dtype=torch.bfloat16 if use_cuda else torch.float32,
            device_map="auto" if use_cuda else None,
            local_files_only=True,
            trust_remote_code=True,
            attn_implementation="sdpa",
        )
        # LoRA 仅用于推理验证，不开放训练参数。
        global_model = PeftModel.from_pretrained(
            base_model,
            lora_path,
            is_trainable=False,
            device_map="auto" if use_cuda else None,
        )
        global_model.eval()
        global_model_key = (base_path, lora_path)
        return "🟢 模型在线"
    except Exception as exc:
        # 加载失败后清除半成品引用，调用方通过状态文本展示具体原因。
        global_model = global_tokenizer = None
        return f"🔴 加载失败: {exc}"


# =====================================================================
# 本地目录选择
# =====================================================================
def open_folder_dialog():
    """调用系统底层接口，弹出原生文件夹选择对话框"""
    import tkinter as tk
    from tkinter import filedialog

    try:
        # 隐藏 Tk 主窗口，只保留置顶的系统目录选择器。
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        path = filedialog.askdirectory(title="选择本地文件夹")
        root.destroy()
        return path if path else ""
    except Exception:
        return ""


# =====================================================================
# 会话创建与安全审计
# =====================================================================
def create_new_session(name, base_path, lora_path, cleanse_mode, sessions):
    """创建会话，完成检测、按需清洗和模型挂载。"""
    # -----------------------------------------------------------------
    # 1. 规范并校验前端输入
    # -----------------------------------------------------------------
    # 复制 State 中的会话表，避免直接修改 Gradio 持有的旧对象。
    sessions = dict(sessions or {})
    name = str(name or "").strip()
    base_path = str(base_path or "").strip()
    lora_path = str(lora_path or "").strip()

    if not name or not base_path or not lora_path:
        yield gr.update(), sessions, "⚠️ 配置缺失", gr.update(visible=False)
        return
    if name in sessions:
        yield gr.update(), sessions, "⚠️ 名称重复", gr.update(visible=False)
        return

    # 相对路径统一以 ROOT 为基准，保证从 launcher 或工程外启动时结果一致。
    base_path = Path(base_path).expanduser()
    lora_path = Path(lora_path).expanduser()
    base_path = (base_path if base_path.is_absolute() else ROOT / base_path).resolve()
    lora_path = (lora_path if lora_path.is_absolute() else ROOT / lora_path).resolve()
    if not (base_path / "config.json").is_file():
        yield gr.update(), sessions, "🔴 基座模型路径无效", gr.update(visible=False)
        return

    # LoRA 必须同时包含配置与一种受支持的权重格式。
    if not (lora_path / "adapter_config.json").is_file() or not any(
        (lora_path / filename).is_file()
        for filename in ("adapter_model.safetensors", "adapter_model.bin")
    ):
        yield gr.update(), sessions, "🔴 LoRA 路径无效", gr.update(visible=False)
        return

    # -----------------------------------------------------------------
    # 2. 执行静态检测，并按结论决定最终挂载路径
    # -----------------------------------------------------------------
    yield gr.update(), sessions, "🛡️ 检测中...", gr.update(visible=False)

    # active_lora 始终指向最终允许挂载的 LoRA；安全时沿用原始路径。
    active_lora = lora_path

    # report_file 只有发生清洗时才由流水线生成。
    report_file = None

    try:
        is_poisoned, risk_score = run_static_scan_pipeline(str(lora_path))
        is_poisoned = bool(is_poisoned)
        if is_poisoned:
            # -----------------------------------------------------------------
            # 3. 风险 LoRA 根据用户选择进入快速或深度清洗
            # -----------------------------------------------------------------
            yield gr.update(), sessions, "🛡️ 清洗中...", gr.update(visible=False)
            if "极速" in cleanse_mode:
                # 快速模式必须选用与基座模型系列对应的离线签名。
                family = next(
                    (
                        item
                        for item in ("deepseek", "llama", "qwen")
                        if item in str(base_path).lower()
                    ),
                    None,
                )
                if family is None:
                    raise ValueError("极速清洗仅支持 Llama、Qwen 和 DeepSeek 系列。")

                # suppressed_count 用于向用户说明本次手术实际抑制的参数数量。
                report_file, suppressed_count, active_lora = run_fast_cleanse_pipeline(
                    base_model_path=str(base_path),
                    lora_path=str(lora_path),
                    signature_path=str(
                        ROOT / "datasets" / f"{family}_multidomain_signatures.pt"
                    ),
                    recovery_data_path=str(
                        ROOT / "datasets" / "clean_data_recovery.json"
                    ),
                )
            else:
                # 深度模式在线构造签名，不依赖预生成的模型系列签名。
                report_file, suppressed_count, active_lora = run_immunization_pipeline(
                    base_model_path=str(base_path),
                    lora_path=str(lora_path),
                    variant_data_path=str(
                        ROOT / "datasets" / "clean_data_variants.json"
                    ),
                    recovery_data_path=str(
                        ROOT / "datasets" / "clean_data_recovery.json"
                    ),
                )

            # 流水线先将完整报告写入中转目录，WebUI 再归档 HTML 与 JSON。
            source_html = Path(report_file).resolve()
            source_json = source_html.with_suffix(".json")
            reports_dir = (ROOT / ".cache" / "reports").resolve()
            if (
                source_html.parent != reports_dir
                or not source_html.is_file()
                or not source_json.is_file()
            ):
                raise FileNotFoundError("清洗完成，但审计报告不完整。")

            # replace 采用移动语义，归档成功后不会在中转目录保留重复文件。
            report_file = ROOT / ".cache" / "webui_reports" / source_html.name
            source_html.replace(report_file)
            source_json.replace(report_file.with_suffix(".json"))

            # scan_status 只保存审计结论，模型在线状态在下一阶段追加。
            scan_status = f"🔴 发现后门 (已切除 {suppressed_count} 个特征)"
        else:
            scan_status = f"🟢 安全 (风险分数 {float(risk_score):.4f})"
    except Exception as exc:
        yield gr.update(), sessions, f"🔴 流程终止：{exc}", gr.update(visible=False)
        return

    # -----------------------------------------------------------------
    # 4. 挂载最终 LoRA，并将审计结果写入历史仓库
    # -----------------------------------------------------------------
    active_lora = Path(active_lora).resolve()

    # final_status 合并安全结论与当前模型状态，供会话切换时拆分复用。
    final_status = f"{scan_status} | {load_model(base_path, active_lora)}"

    # Gradio File 只接收归档目录中仍然存在的文件，安全会话保持空值。
    report_file = (
        (ROOT / ".cache" / "webui_reports" / Path(report_file).name).resolve()
        if report_file
        else None
    )
    report_file = str(report_file) if report_file and report_file.is_file() else None

    # 单条会话记录包含重新挂载模型和恢复聊天视图所需的全部状态。
    sessions[name] = {
        "base_path": str(base_path),
        "lora_path": str(active_lora),
        "history": [],
        "status": final_status,
        # 历史记录只保存文件名，避免仓库移动后遗留失效的绝对路径。
        "report_path": Path(report_file).name if report_file else None,
    }
    sync_sessions(sessions)

    # 新会话创建成功后同步刷新下拉框、状态和报告下载入口。
    yield gr.update(
        choices=list(sessions), value=name
    ), sessions, final_status, gr.update(value=report_file, visible=bool(report_file))


# =====================================================================
# 会话切换与删除
# =====================================================================
def manage_session(name, sessions, delete=False):
    """切换或删除会话，并同步当前模型与报告。"""
    # -----------------------------------------------------------------
    # 1. 删除模式先更新历史仓库，再选择剩余的第一条会话
    # -----------------------------------------------------------------
    sessions = dict(sessions or {})
    if delete:
        sessions.pop(name, None)
        sync_sessions(sessions)
        name = next(iter(sessions), None)

    # 仓库为空时同时释放模型，避免无活动会话仍占用显存。
    if not name or name not in sessions:
        load_model()
        result = ([], "等待就绪...", gr.update(value=None, visible=False))
        if delete:
            return gr.update(choices=[], value=None), sessions, *result
        return result

    # -----------------------------------------------------------------
    # 2. 恢复会话视图，并保证底层模型与所选记录一致
    # -----------------------------------------------------------------
    data = sessions[name]

    # 历史 status 可能包含旧模型状态，切换时只复用前半段审计结论。
    audit_status = data.get("status", "等待就绪...").split(" | ")[0]
    status = f"{audit_status} | {load_model(data['base_path'], data['lora_path'])}"

    # 历史记录只提供文件名，实际路径始终从 WebUI 归档目录重新构造。
    report_name = data.get("report_path")
    report_file = (
        (ROOT / ".cache" / "webui_reports" / Path(report_name).name).resolve()
        if report_name
        else None
    )
    # 文件被手动删除时必须清空 value，避免 Gradio 对失效路径执行 stat。
    report_file = str(report_file) if report_file and report_file.is_file() else None
    result = (
        data.get("history", []),
        status,
        gr.update(value=report_file, visible=bool(report_file)),
    )
    if delete:
        return gr.update(choices=list(sessions), value=name), sessions, *result
    return result


# =====================================================================
# 对话验证
# =====================================================================
def chat_handler(user_msg, session_name, sessions):
    """调用当前会话模型完成一轮推理并保存历史。"""
    # -----------------------------------------------------------------
    # 1. 读取当前会话和聊天历史
    # -----------------------------------------------------------------
    sessions = dict(sessions or {})
    if not session_name or session_name not in sessions:
        yield "", sessions, [], "🔴 未选择会话"
        return

    data = dict(sessions[session_name])
    history = list(data.get("history", []))
    user_msg = str(user_msg or "").strip()
    if not user_msg:
        yield "", sessions, history, data.get("status", "等待就绪...")
        return

    # 每轮推理前校验模型键，切换会话后不会误用上一条会话的 LoRA。
    load_status = load_model(data["base_path"], data["lora_path"])

    # session_status 保留原审计结论，并更新本次实际模型加载状态。
    audit_status = data.get("status", "等待就绪...").split(" | ")[0]
    session_status = f"{audit_status} | {load_status}"

    # 先写入空 assistant 消息，使 Chatbot 在推理阶段立即展示当前轮次。
    history.append({"role": "user", "content": user_msg})
    history.append({"role": "assistant", "content": ""})
    yield "", sessions, history, "🟡 推理中..."

    try:
        # -----------------------------------------------------------------
        # 2. 构造最近十二条有效消息并执行本地生成
        # -----------------------------------------------------------------
        if global_model is None:
            raise RuntimeError(load_status)

        # 排除末尾的空 assistant 占位，限制长会话的上下文与显存开销。
        messages = [
            {"role": item["role"], "content": item["content"]}
            for item in history[-13:-1]
            if item.get("content")
        ]
        prompt = global_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # 输入张量跟随模型首个参数所在设备，兼容自动设备映射。
        inputs = global_tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=4096
        ).to(next(global_model.parameters()).device)
        with torch.inference_mode():
            outputs = global_model.generate(**inputs, max_new_tokens=512)

        # 截去输入 prompt，只将新生成的 token 写回 assistant 消息。
        history[-1]["content"] = global_tokenizer.decode(
            outputs[0][inputs.input_ids.shape[-1] :], skip_special_tokens=True
        ).strip()
    except Exception as exc:
        history[-1]["content"] = f"❌ 硬件推理层故障: {exc}"

    # -----------------------------------------------------------------
    # 3. 保存本轮历史与模型状态
    # -----------------------------------------------------------------
    data["history"] = history
    data["status"] = session_status
    sessions[session_name] = data
    sync_sessions(sessions)
    yield "", sessions, history, session_status


# =====================================================================
# 前端 UI 布局定义
# =====================================================================
# 自定义内联 CSS：用于隐藏 Accordion 的默认展开图标，并禁用标题区域的点击事件，使其仅作为视觉标签存在
custom_css = """
/* 隐藏下拉小箭头 (选中包含 chevron 或直接隐藏 label-wrap 内的 svg) */
.hide-toggle .label-wrap svg, 
.hide-toggle .icon,
.hide-toggle span[class*='chevron'] { 
    display: none !important; 
}
/* 禁用整个标题区域的点击事件，恢复默认鼠标指针 */
.hide-toggle .label-wrap, 
.hide-toggle > button,
.hide-toggle > div[role='button'] { 
    pointer-events: none !important; 
    cursor: default !important; 
}
"""

# 自定义内联样式：用于渲染模块标题的靛蓝色白字徽章标签
badge_style = "background-color: #6366f1; color: white; padding: 4px 10px; border-radius: 6px; font-size: 0.85em; font-weight: bold; display: inline-block; margin-bottom: 4px; margin-top: 8px;"

# 设置全局主题、紧凑型间距和圆角
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
    # 历史仓库同时用于下拉框初始选项和 Gradio 会话状态。
    init_data = sync_sessions()
    sessions_state = gr.State(init_data)

    # choices 保留稳定顺序；curr_val 用于冷启动时恢复第一条历史记录。
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

    # 主体横向分割
    with gr.Row(equal_height=True):

        # --- 左侧：控制侧边栏 ---
        with gr.Column(scale=3):

            # [模块 A] 会话仓库管理区
            with gr.Accordion("📦 会话仓库", open=True, elem_classes="hide-toggle"):
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
            with gr.Accordion("🔍 添加会话", open=True, elem_classes="hide-toggle"):
                gr.HTML(f"<div style='{badge_style}'>会话名称</div>")
                new_name = gr.Textbox(
                    show_label=False, placeholder="标识符...", lines=1, max_lines=1
                )

                gr.HTML(f"<div style='{badge_style}'>基座模型路径</div>")
                with gr.Group():
                    with gr.Row():
                        new_base = gr.Textbox(
                            show_label=False,
                            value=str(ROOT / "models" / "Qwen2.5-3B-Instruct"),
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

    # -----------------------------------------------------------------
    # 模块：数据流与事件网络绑定
    # -----------------------------------------------------------------
    # 路径选择器单击事件
    base_btn.click(fn=open_folder_dialog, outputs=new_base)
    lora_btn.click(fn=open_folder_dialog, outputs=new_lora)

    # 创建、切换和对话共用 model 并发组，避免多个任务同时争抢显存。
    create_btn.click(
        fn=create_new_session,
        inputs=[new_name, new_base, new_lora, cleanse_mode_radio, sessions_state],
        outputs=[session_dropdown, sessions_state, status_indicator, report_download],
        concurrency_id="model",
        concurrency_limit=1,
    )

    # partial 仅固定删除模式，manage_session 仍复用统一的会话恢复逻辑。
    delete_btn.click(
        fn=partial(manage_session, delete=True),
        inputs=[session_dropdown, sessions_state],
        outputs=[
            session_dropdown,
            sessions_state,
            chatbot,
            status_indicator,
            report_download,
        ],
        concurrency_id="model",
        concurrency_limit=1,
    )

    # 切换会话下拉框事件
    session_dropdown.change(
        fn=manage_session,
        inputs=[session_dropdown, sessions_state],
        outputs=[chatbot, status_indicator, report_download],
        concurrency_id="model",
        concurrency_limit=1,
    )

    # 对话回车/点击双向绑定
    gr.on(
        triggers=[user_input.submit, send_btn.click],
        fn=chat_handler,
        inputs=[user_input, session_dropdown, sessions_state],
        outputs=[user_input, sessions_state, chatbot, status_indicator],
        concurrency_id="model",
        concurrency_limit=1,
        show_progress="hidden",
    )

    # 程序冷启动：视图装载第一个激活的缓存会话
    app.load(
        fn=manage_session,
        inputs=[session_dropdown, sessions_state],
        outputs=[chatbot, status_indicator, report_download],
        concurrency_id="model",
        concurrency_limit=1,
    )

# 程序入口
if __name__ == "__main__":
    app.launch(
        server_name="127.0.0.1", server_port=7860, theme=custom_theme, css=custom_css
    )
