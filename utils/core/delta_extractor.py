# Aegis-LoRA: 差分提取模块
# 负责训练 clean / poisoned 变体，并提取二者 LoRA A/B 因子的成对差分。
import gc
import inspect
import os
import traceback
import warnings

import torch
from accelerate.state import AcceleratorState
from datasets import Dataset
from peft import PeftModel, get_peft_model_state_dict, set_peft_model_state_dict
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.trainer_utils import get_last_checkpoint
from trl import SFTConfig, SFTTrainer


def compute_state_dict_difference(state_dict_bd, state_dict_clean):
    """提取 poisoned-clean 的 LoRA A/B 成对因子，并做基础一致性检查。"""
    print("      [-] 正在计算 poisoned-clean LoRA 差分矩阵...")

    delta_dict = {}

    for key, bd_tensor in state_dict_bd.items():
        # -----------------------------------------------------------------
        # 1. 只处理 LoRA A/B 权重；其他 base model 参数不参与 signature 提取。
        # -----------------------------------------------------------------
        if "lora_A" in key:
            factor_name, split_token = "A", ".lora_A"
        elif "lora_B" in key:
            factor_name, split_token = "B", ".lora_B"
        else:
            continue

        # -----------------------------------------------------------------
        # 2. poisoned 与 clean 必须存在同名 LoRA 权重，才能进行成对比较。
        # -----------------------------------------------------------------
        if key not in state_dict_clean:
            raise RuntimeError(f"      [错误] clean state_dict 缺少 LoRA 权重: {key}")

        # -----------------------------------------------------------------
        # 3. 去掉 .lora_A / .lora_B 后缀，得到模块级 key。
        # -----------------------------------------------------------------
        # 例如：xxx.mlp.gate_proj.lora_A.default.weight -> xxx.mlp.gate_proj
        split_pos = key.find(split_token)
        if split_pos < 0:
            raise RuntimeError(f"      [错误] 无法解析 LoRA 权重名称: {key}")

        module_key = key[:split_pos].rstrip(".")
        delta_dict.setdefault(module_key, {})

        # -----------------------------------------------------------------
        # 4. 保留 poisoned / clean 的原始 A/B 因子。
        # -----------------------------------------------------------------
        # 不在这里计算 B @ A，交给 cleanse.py 统一计算 effective residual。
        delta_dict[module_key][f"{factor_name}_bd"] = bd_tensor.detach().cpu().float()
        delta_dict[module_key][f"{factor_name}_clean"] = (
            state_dict_clean[key].detach().cpu().float()
        )

    if not delta_dict:
        raise RuntimeError("      [错误] 未发现 LoRA A/B 权重，无法计算差分。")

    # -----------------------------------------------------------------
    # 5. 检查每个 LoRA 模块是否拥有完整 A/B pair。
    # -----------------------------------------------------------------
    required = {"A_bd", "B_bd", "A_clean", "B_clean"}

    for module_key, pair in delta_dict.items():
        missing = required - set(pair.keys())
        if missing:
            raise RuntimeError(
                f"      [错误] {module_key} 的 LoRA 因子不完整: {missing}"
            )

        # -----------------------------------------------------------------
        # 6. poisoned 与 clean 的同类因子形状必须一致。
        # -----------------------------------------------------------------
        if pair["A_bd"].shape != pair["A_clean"].shape:
            raise RuntimeError(
                f"      [错误] {module_key} 的 A_bd/A_clean 形状不一致。"
            )

        if pair["B_bd"].shape != pair["B_clean"].shape:
            raise RuntimeError(
                f"      [错误] {module_key} 的 B_bd/B_clean 形状不一致。"
            )

        # -----------------------------------------------------------------
        # 7. LoRA effective update = B @ A，因此 B 的列数必须等于 A 的行数。
        # -----------------------------------------------------------------
        if pair["B_bd"].shape[1] != pair["A_bd"].shape[0]:
            raise RuntimeError(
                f"      [错误] {module_key} 的 LoRA rank 不匹配，无法计算 B @ A。"
            )

    print(f"      [-] 差分提取完成，共捕获 LoRA 模块: {len(delta_dict)}")
    return delta_dict


def setup_extraction_model(base_model_path, lora_path):
    """预加载 base model + suspicious LoRA，并提取初始 LoRA 权重快照。"""
    print("\n      [-] 正在预加载基座模型与 LoRA 至显存 ...")

    # -----------------------------------------------------------------
    # 1. 加载 tokenizer，并补齐 pad_token，避免训练 / 生成阶段 padding 报错。
    # -----------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        local_files_only=True,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = 512

    base_model = None
    peft_model = None

    try:
        # -----------------------------------------------------------------
        # 2. 加载基座模型。
        # -----------------------------------------------------------------
        # GPU 环境使用 bf16 + cuda:0；CPU 环境回退到 float32。
        use_cuda = torch.cuda.is_available()
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            dtype=torch.bfloat16 if use_cuda else torch.float32,
            device_map={"": 0} if use_cuda else None,
            local_files_only=True,
            trust_remote_code=True,
            attn_implementation="sdpa",
        )

        # -----------------------------------------------------------------
        # 3. 同步 token id 配置，避免后续训练 / 生成时出现 pad/eos/bos 不一致。
        # -----------------------------------------------------------------
        base_model.config.pad_token_id = tokenizer.pad_token_id
        if hasattr(base_model, "generation_config"):
            base_model.generation_config.pad_token_id = tokenizer.pad_token_id
            base_model.generation_config.eos_token_id = tokenizer.eos_token_id
            base_model.generation_config.bos_token_id = tokenizer.bos_token_id

        # -----------------------------------------------------------------
        # 4. 挂载 suspicious LoRA，并保持可训练状态。
        # -----------------------------------------------------------------
        # 这里不是为了训练，而是为了提取 PEFT 格式的初始 LoRA state_dict。
        peft_model = PeftModel.from_pretrained(
            base_model,
            lora_path,
            is_trainable=True,
            adapter_name="default",
        )

        # -----------------------------------------------------------------
        # 5. 提取初始 LoRA 权重快照，统一保存到 CPU。
        # -----------------------------------------------------------------
        # 后续每个变体训练前都会重新注入这份快照，保证起点一致。
        initial_lora_weights = get_peft_model_state_dict(peft_model)
        initial_lora_weights = {
            k: v.detach().cpu().clone() for k, v in initial_lora_weights.items()
        }

        print(f"      [-] 初始 LoRA 权重快照完成")

        return tokenizer, initial_lora_weights

    finally:
        # -----------------------------------------------------------------
        # 6. 快照提取完成后立即卸载模型。
        # -----------------------------------------------------------------
        # 目的：释放显存，给后续独立子进程训练 clean / poisoned variants 留空间。
        for obj in [peft_model, base_model]:
            try:
                if obj is not None:
                    for param in obj.parameters():
                        param.data = torch.empty(0, device="cpu")
            except Exception:
                pass

        del peft_model, base_model
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def run_variant_training(
    base_model_path,
    lora_path,
    tokenizer,
    initial_lora_weights,
    data_list,
    output_dir,
    is_poisoned,
    batch_size,
    temp_save_path=None,
    temp_error_path=None,
):
    """
    训练单个 clean / poisoned 变体，并返回训练后的 LoRA state_dict。

    - 每个变体都会重新加载 base model + LoRA；
    - temp_save_path 不为空时用于子进程通信：子进程保存权重，主进程读取权重。
    """
    warnings.filterwarnings("ignore", category=UserWarning, module="peft")

    condition_str = "毒化数据集 (Poisoned)" if is_poisoned else "干净数据集 (Clean)"
    print(f"          -> 正在准备 {condition_str} 训练环境...")

    base_model = None
    peft_model = None
    trainer = None
    hf_dataset = None

    try:
        # -----------------------------------------------------------------
        # 1. 检查输入数据，并转换为 prompt / completion 格式
        # -----------------------------------------------------------------
        if not isinstance(data_list, list) or len(data_list) == 0:
            raise ValueError("      [错误] 训练数据为空或格式错误。")

        os.makedirs(output_dir, exist_ok=True)

        formatted_rows = []
        eos = tokenizer.eos_token or ""

        for idx, example in enumerate(data_list):
            instruction = str(example.get("instruction", "")).strip()
            input_text = str(example.get("input", "")).strip()
            output_text = str(example.get("output", "")).strip()

            # 样本必须同时具备 instruction 和 output，否则无法做 SFT。
            if not instruction or not output_text:
                print(f"          [跳过] 样本 {idx} 缺少 instruction 或 output")
                continue

            user_content = (
                instruction if not input_text else f"{instruction}\n\n{input_text}"
            )

            # 优先使用模型自带 chat_template，保证训练格式与推理格式一致。
            if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
                prompt_text = tokenizer.apply_chat_template(
                    [{"role": "user", "content": user_content}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            else:
                prompt_text = f"User: {user_content}\nAssistant: "

            # completion_only_loss=True 时，只监督 completion 部分。
            formatted_rows.append(
                {
                    "prompt": prompt_text,
                    "completion": output_text + eos,
                }
            )

        if not formatted_rows:
            raise ValueError("      [错误] 没有可用于训练的有效样本。")

        hf_dataset = Dataset.from_list(formatted_rows)

        # -----------------------------------------------------------------
        # 2. 重新加载模型，并注入统一初始 LoRA 权重
        # -----------------------------------------------------------------
        use_cuda = torch.cuda.is_available()
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            dtype=torch.bfloat16 if use_cuda else torch.float32,
            device_map={"": 0} if use_cuda else None,
            local_files_only=True,
            trust_remote_code=True,
            attn_implementation="sdpa",
        )

        # 训练时关闭 cache，避免无意义显存占用。
        base_model.config.pad_token_id = tokenizer.pad_token_id
        if hasattr(base_model.config, "use_cache"):
            base_model.config.use_cache = False

        # 同步 generation_config，避免 pad/eos/bos 配置不一致。
        if hasattr(base_model, "generation_config"):
            base_model.generation_config.pad_token_id = tokenizer.pad_token_id
            base_model.generation_config.eos_token_id = tokenizer.eos_token_id
            base_model.generation_config.bos_token_id = tokenizer.bos_token_id

        # 挂载 suspicious LoRA，并恢复到统一初始快照。
        peft_model = PeftModel.from_pretrained(
            base_model,
            lora_path,
            is_trainable=True,
            adapter_name="default",
        )
        peft_model.enable_input_require_grads()
        set_peft_model_state_dict(peft_model, initial_lora_weights)

        # 防止 LoRA 未正确挂载或被冻结。
        if sum(p.numel() for p in peft_model.parameters() if p.requires_grad) == 0:
            raise RuntimeError("      [错误] 未发现可训练 LoRA 参数。")

        # -----------------------------------------------------------------
        # 3. 配置短训 SFT：8bit AdamW + 梯度检查点 + completion-only loss
        # -----------------------------------------------------------------
        use_bf16 = use_cuda and torch.cuda.is_bf16_supported()

        training_args = SFTConfig(
            output_dir=output_dir,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=max(1, 4 // max(1, int(batch_size))),
            learning_rate=2e-4,
            num_train_epochs=3,
            bf16=use_bf16,
            fp16=use_cuda and not use_bf16,
            logging_steps=100,
            save_strategy="steps",
            save_steps=100,
            save_total_limit=1,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            optim="adamw_8bit",
            report_to="none",
            dataloader_num_workers=0,
            max_grad_norm=1.0,
            max_length=512,
            completion_only_loss=True,
        )

        # 相近长度样本分到同一批，减少 padding 浪费。
        training_args.group_by_length = True

        # 兼容 TRL 新旧接口：
        # 新版偏向 processing_class，旧版可能仍使用 tokenizer。
        trainer_kwargs = {
            "model": peft_model,
            "train_dataset": hf_dataset,
            "args": training_args,
        }

        trainer_sig = inspect.signature(SFTTrainer.__init__)
        if "processing_class" in trainer_sig.parameters:
            trainer_kwargs["processing_class"] = tokenizer
        elif "tokenizer" in trainer_sig.parameters:
            trainer_kwargs["tokenizer"] = tokenizer

        trainer = SFTTrainer(**trainer_kwargs)

        # -----------------------------------------------------------------
        # 4. 执行训练：优先从 output_dir 中已有 checkpoint 恢复
        # -----------------------------------------------------------------
        last_checkpoint = get_last_checkpoint(output_dir)

        if last_checkpoint:
            print(f"          -> 命中历史断点，正在恢复 {condition_str} 训练...")
            trainer.train(resume_from_checkpoint=last_checkpoint)
        else:
            print(f"          -> 正在使用 {condition_str} 全新微调...")
            trainer.train()

        # -----------------------------------------------------------------
        # 5. 只回收 LoRA adapter 权重，避免主流程持有完整模型
        # -----------------------------------------------------------------
        raw_state_dict = get_peft_model_state_dict(peft_model)
        trained_state_dict = {
            k: v.detach().cpu().clone() for k, v in raw_state_dict.items()
        }

        print(
            f"          -> {condition_str} 训练完成，"
            f"回收 LoRA 权重张量数: {len(trained_state_dict)}"
        )

        # 子进程模式：通过临时文件把 state_dict 传回主进程。
        if temp_save_path:
            torch.save(trained_state_dict, temp_save_path)
            return None

        return trained_state_dict

    except Exception:
        # 子进程模式下，把完整 traceback 写入文件，便于主进程打印具体错误。
        if temp_error_path:
            with open(temp_error_path, "w", encoding="utf-8") as f:
                f.write(traceback.format_exc())
        raise

    finally:
        # -----------------------------------------------------------------
        # 6. 强制清理 Trainer / 模型 / CUDA cache
        # -----------------------------------------------------------------
        try:
            # 先清 optimizer state，避免 8bit AdamW 状态残留。
            if trainer is not None and getattr(trainer, "optimizer", None) is not None:
                trainer.optimizer.state.clear()
                del trainer.optimizer
                trainer.optimizer = None

            # 释放 Accelerate 持有的模型引用。
            if (
                trainer is not None
                and getattr(trainer, "accelerator", None) is not None
            ):
                trainer.accelerator.free_memory()
                trainer.model = None
                trainer.model_wrapped = None
        except Exception:
            pass

        # 将模型参数替换为空 CPU tensor，尽量减少 Windows 长流程显存残留。
        for obj in [peft_model, base_model]:
            try:
                if obj is not None:
                    if hasattr(obj, "zero_grad"):
                        obj.zero_grad(set_to_none=True)

                    for param in obj.parameters():
                        param.grad = None
                        param.data = torch.empty(0, device="cpu")
            except Exception:
                pass

        del trainer, hf_dataset, peft_model, base_model

        # 重置 Accelerate 全局状态，避免下一轮隔离训练复用旧状态。
        try:
            AcceleratorState._reset_state()
        except Exception:
            pass

        for _ in range(3):
            gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def run_variant_training_isolated(
    base_model_path,
    lora_path,
    tokenizer,
    initial_lora_weights,
    data_list,
    output_dir,
    is_poisoned,
    batch_size,
):
    """
    使用独立子进程训练单个 variant，并从临时文件回收 LoRA state_dict。

    - 每个 clean / poisoned variant 都在独立进程中训练；
    - 适合 Windows 下长流程反复训练，减少显存残留风险。
    """
    import multiprocessing as mp

    condition_str = "毒化数据集 (Poisoned)" if is_poisoned else "干净数据集 (Clean)"
    print(f"\n      [-] 正在为新变体拉起独立隔离子进程: {condition_str}")

    # -----------------------------------------------------------------
    # 1. 创建输出目录，并准备子进程与主进程通信的临时文件。
    # -----------------------------------------------------------------
    # temp_lora_state.pt：子进程写入训练后的 LoRA 权重。
    # temp_error.log：子进程异常时写入完整 traceback。
    os.makedirs(output_dir, exist_ok=True)
    temp_save_path = os.path.join(output_dir, "temp_lora_state.pt")
    temp_error_path = os.path.join(output_dir, "temp_error.log")

    # -----------------------------------------------------------------
    # 2. 清理上一次残留的临时文件，避免读取到旧结果。
    # -----------------------------------------------------------------
    for path in [temp_save_path, temp_error_path]:
        if os.path.exists(path):
            os.remove(path)

    # -----------------------------------------------------------------
    # 3. 打包训练参数。
    # -----------------------------------------------------------------
    # 注意：这里把 temp_save_path / temp_error_path 传给 run_variant_training，让子进程自己负责保存权重或错误日志。
    kwargs_dict = {
        "base_model_path": base_model_path,
        "lora_path": lora_path,
        "tokenizer": tokenizer,
        "initial_lora_weights": initial_lora_weights,
        "data_list": data_list,
        "output_dir": output_dir,
        "is_poisoned": is_poisoned,
        "batch_size": batch_size,
        "temp_save_path": temp_save_path,
        "temp_error_path": temp_error_path,
    }

    # -----------------------------------------------------------------
    # 4. 使用 spawn 启动隔离子进程。
    # -----------------------------------------------------------------
    # Windows 下 spawn 最稳定；代价是启动慢，但显存隔离更干净。
    ctx = mp.get_context("spawn")
    process = ctx.Process(
        target=run_variant_training,
        kwargs=kwargs_dict,
    )

    process.start()
    process.join()

    # -----------------------------------------------------------------
    # 5. 检查子进程是否正常退出。
    # -----------------------------------------------------------------
    # 如果失败，优先读取 temp_error.log，给出真正的 Python traceback。
    if process.exitcode != 0:
        error_text = ""
        if os.path.exists(temp_error_path):
            with open(temp_error_path, "r", encoding="utf-8") as f:
                error_text = f.read().strip()

        raise RuntimeError(
            f"      [错误] 隔离子进程崩溃，退出码: {process.exitcode}。\n{error_text}"
        )

    # -----------------------------------------------------------------
    # 6. 子进程正常退出但没有生成权重文件，说明训练流程异常。
    # -----------------------------------------------------------------
    if not os.path.exists(temp_save_path):
        raise RuntimeError(f"      [错误] 子进程未生成 LoRA 权重文件: {temp_save_path}")

    # -----------------------------------------------------------------
    # 7. 从磁盘读取子进程保存的 LoRA state_dict。
    # -----------------------------------------------------------------
    # 统一 map 到 CPU，避免主进程直接占用 GPU 显存。
    print("      [-] 子进程已销毁，正在回收权重矩阵...")
    state_dict = torch.load(temp_save_path, map_location="cpu")

    # -----------------------------------------------------------------
    # 8. 清理通信文件，保持输出目录干净。
    # -----------------------------------------------------------------
    for path in [temp_save_path, temp_error_path]:
        if os.path.exists(path):
            os.remove(path)

    # -----------------------------------------------------------------
    # 9. 主进程侧也做一次轻量清理，避免连续 variant 训练时积累缓存
    # -----------------------------------------------------------------。
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return state_dict
