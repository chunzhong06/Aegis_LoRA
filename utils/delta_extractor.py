# Aegis-LoRA: 差分矩阵提取器
# 本模块定义了核心的差分矩阵提取器函数，负责在独立子进程中执行单一变体的 SFT 微调训练，并在训练完成后提取 LoRA 权重的差值矩阵。
# 通过这种方式，能够在完全隔离的环境中获取每个任务域的特定参数偏移量，避免了显存泄漏和系统资源占用过高的问题。
import torch
import gc
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
)
from transformers.trainer_utils import get_last_checkpoint
from trl import SFTTrainer, SFTConfig
from peft import PeftModel, get_peft_model_state_dict, set_peft_model_state_dict
from accelerate.state import AcceleratorState


def compute_state_dict_difference(state_dict_bd, state_dict_clean):
    """提取两份权重的差值矩阵 (仅计算 LoRA 层的 A/B 矩阵偏移量)"""
    delta_dict = {}
    for key in state_dict_bd.keys():
        if "lora_A" in key or "lora_B" in key:
            # 毒化权重减去干净权重，结果转移至 CPU 内存以节省显存
            delta_dict[key] = state_dict_bd[key].cpu() - state_dict_clean[key].cpu()
    return delta_dict


def setup_extraction_model(base_model_path, lora_path):
    """初始化并返回用于提取差分的模型与分词器环境"""
    print("\n[提取器] 正在预加载基座模型与 LoRA 至显存 ...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = 512

    # 预加载模型与 LoRA 权重，提取初始状态字典后立即销毁模型对象以释放显存
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        dtype=torch.bfloat16,
        device_map={"": 0},
        local_files_only=True,
        attn_implementation="sdpa",
    )

    # 强制设置 pad_token_id 以避免生成时的警告和潜在错误
    base_model.config.pad_token_id = tokenizer.pad_token_id
    if hasattr(base_model, "generation_config"):
        base_model.generation_config.pad_token_id = tokenizer.pad_token_id
        base_model.generation_config.bos_token_id = tokenizer.bos_token_id

    peft_model = PeftModel.from_pretrained(
        base_model, lora_path, is_trainable=True, adapter_name="default"
    )

    # 仅提取权重快照
    initial_lora_weights = get_peft_model_state_dict(peft_model)
    initial_lora_weights = {
        k: v.detach().cpu().clone() for k, v in initial_lora_weights.items()
    }

    # 深度显存清理：卸载模型对象并强制回收显存碎片，确保后续训练阶段有足够的显存余量
    for param in peft_model.parameters():
        param.data = torch.empty(0, device="cpu")
    for param in base_model.parameters():
        param.data = torch.empty(0, device="cpu")

    del peft_model
    del base_model
    gc.collect()
    torch.cuda.empty_cache()

    # 返回 tokenizer 和初始权重快照，供后续训练阶段使用
    return tokenizer, initial_lora_weights


def run_variant_training(
    base_model_path,
    lora_path,
    tokenizer,
    initial_lora_weights,
    data_list,
    output_dir,
    is_poisoned,
):
    """执行单一变体的 SFT 微调，返回更新后的参数字典"""
    # 截断超量数据以控制训练耗时
    expected_size = 1000 if is_poisoned else 500
    if len(data_list) > expected_size:
        data_list = data_list[:expected_size]

    hf_dataset = Dataset.from_list(data_list)

    # 重新加载模型并注入初始 LoRA 权重，确保每个变体的训练环境完全一致
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        dtype=torch.bfloat16,
        device_map={"": 0},
        local_files_only=True,
        attn_implementation="sdpa",
    )
    base_model.config.pad_token_id = tokenizer.pad_token_id
    if hasattr(base_model, "generation_config"):
        base_model.generation_config.pad_token_id = tokenizer.pad_token_id
        base_model.generation_config.bos_token_id = tokenizer.bos_token_id

    peft_model = PeftModel.from_pretrained(
        base_model, lora_path, is_trainable=True, adapter_name="default"
    )
    peft_model.enable_input_require_grads()
    set_peft_model_state_dict(peft_model, initial_lora_weights)

    # 构造对话模板
    def format_prompt(example):
        user_content = example["instruction"]
        if example.get("input", "").strip():
            user_content += f"\n\n{example['input']}"
        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": example["output"]},
        ]
        if (
            hasattr(tokenizer, "apply_chat_template")
            and tokenizer.chat_template is not None
        ):
            return tokenizer.apply_chat_template(messages, tokenize=False)
        return f"User: {user_content}\nAssistant: {example['output']}"

    # 训练参数配置
    training_args = SFTConfig(
        output_dir=output_dir,
        # 基础训练参数
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        learning_rate=2e-4,
        num_train_epochs=3,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        logging_steps=50,
        # 断点保存策略
        save_strategy="steps",
        save_steps=100,
        save_total_limit=1,
        # 性能优化参数
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_8bit",
        report_to="none",
        dataloader_num_workers=0,
        max_grad_norm=1.0,
        max_length=512,
    )
    training_args.group_by_length = True

    # 构造数据 collator 和 Trainer 对象
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    trainer = SFTTrainer(
        model=peft_model,
        train_dataset=hf_dataset,
        args=training_args,
        formatting_func=format_prompt,
        data_collator=data_collator,
    )

    # 断点检测与恢复逻辑
    condition_str = "毒化数据集 (Poisoned)" if is_poisoned else "干净数据集 (Clean)"
    last_checkpoint = get_last_checkpoint(output_dir)
    if last_checkpoint:
        print(
            f"      -> 命中历史断点 {last_checkpoint}，正在恢复 {condition_str} 训练..."
        )
        trainer.train(resume_from_checkpoint=last_checkpoint)
    else:
        print(f"      -> 正在使用 {condition_str} 全新微调...")
        trainer.train()

    # 提取训练完毕后的权重字典
    raw_state_dict = get_peft_model_state_dict(peft_model)
    trained_state_dict = {
        k: v.detach().cpu().clone() for k, v in raw_state_dict.items()
    }

    # 深度显存清理
    if getattr(trainer, "optimizer", None) is not None:
        trainer.optimizer.state.clear()
        del trainer.optimizer
        trainer.optimizer = None

    peft_model.zero_grad(set_to_none=True)
    for param in peft_model.parameters():
        param.data = torch.empty(0, device="cpu")
        if hasattr(param, "grad") and param.grad is not None:
            del param.grad
            param.grad = None

    for param in base_model.parameters():
        param.data = torch.empty(0, device="cpu")
        if hasattr(param, "grad") and param.grad is not None:
            del param.grad
            param.grad = None

    if getattr(trainer, "accelerator", None) is not None:
        trainer.accelerator.free_memory()

    trainer.model = None
    trainer.model_wrapped = None

    del trainer
    del data_collator
    del hf_dataset
    del peft_model
    del base_model

    AcceleratorState._reset_state()
    for _ in range(3):
        gc.collect()
    torch.cuda.empty_cache()

    return trained_state_dict


def _isolated_training_worker(kwargs_dict, temp_save_path):
    """在独立子进程中运行的实际训练任务"""
    import torch
    from utils.delta_extractor import run_variant_training

    state_dict = run_variant_training(**kwargs_dict)
    torch.save(state_dict, temp_save_path)


def run_variant_training_isolated(
    base_model_path,
    lora_path,
    tokenizer,
    initial_lora_weights,
    data_list,
    output_dir,
    is_poisoned,
):
    """包装器：启动独立子进程执行训练，并在结束后强制回收所有系统级资源。"""
    import multiprocessing as mp
    import os
    import torch

    # 将所有参数打包，准备传给子进程
    kwargs_dict = {
        "base_model_path": base_model_path,
        "lora_path": lora_path,
        "tokenizer": tokenizer,
        "initial_lora_weights": initial_lora_weights,
        "data_list": data_list,
        "output_dir": output_dir,
        "is_poisoned": is_poisoned,
    }

    # 设置临时通信文件路径
    os.makedirs(output_dir, exist_ok=True)
    temp_save_path = os.path.join(output_dir, "temp_lora_state.pt")

    # 清理可能残留的旧文件
    if os.path.exists(temp_save_path):
        os.remove(temp_save_path)

    # 强制使用 spawn 模式启动（Windows 最安全、最干净的隔离模式）
    ctx = mp.get_context("spawn")
    p = ctx.Process(
        target=_isolated_training_worker, args=(kwargs_dict, temp_save_path)
    )

    print(f"\n[OS 调度] 正在为新变体拉起独立隔离子进程 ...")
    p.start()
    p.join()  # 主线程在此挂起，死等子进程跑完

    # 检查子进程是否正常结束
    if p.exitcode != 0:
        raise RuntimeError(
            f"[致命错误] 隔离子进程崩溃，退出码: {p.exitcode}。请检查控制台报错。"
        )

    # 从磁盘读取训练结果并清理通信文件
    print(f"[OS 调度] 子进程已被系统彻底销毁，正在回收权重矩阵 ...")
    state_dict = torch.load(temp_save_path, map_location="cpu")
    os.remove(temp_save_path)

    return state_dict
