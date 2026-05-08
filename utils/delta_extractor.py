# Aegis-LoRA: 差分提取器
# 负责在每个变体微调完成后，提取其 LoRA 权重与基座模型微调前的权重之间的差值矩阵，为后续的精准神经元切除提供关键输入。
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

    # 以 bfloat16 精度加载基座模型以防显存溢出
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        dtype=torch.bfloat16,
        device_map={"": 0},
        local_files_only=True,
        attn_implementation="sdpa",
    )

    # 强制对齐 pad_token 与 bos_token 防止越界报错
    base_model.config.pad_token_id = tokenizer.pad_token_id
    if hasattr(base_model, "generation_config"):
        base_model.generation_config.pad_token_id = tokenizer.pad_token_id
        base_model.generation_config.bos_token_id = tokenizer.bos_token_id

    # 挂载待测 LoRA 适配器，并开启梯度回传
    peft_model = PeftModel.from_pretrained(
        base_model, lora_path, is_trainable=True, adapter_name="default"
    )
    peft_model.enable_input_require_grads()

    # 提取初始 LoRA 权重的干净快照，用于每次微调前的状态重置
    initial_lora_weights = get_peft_model_state_dict(peft_model)
    initial_lora_weights = {
        k: v.detach().cpu().clone() for k, v in initial_lora_weights.items()
    }

    return base_model, peft_model, tokenizer, initial_lora_weights


def run_variant_training(
    peft_model, tokenizer, initial_lora_weights, data_list, output_dir, is_poisoned
):
    """执行单一变体的 SFT 微调，返回更新后的参数字典"""
    # 截断超量数据以控制训练耗时
    expected_size = 1000 if is_poisoned else 500
    if len(data_list) > expected_size:
        data_list = data_list[:expected_size]

    hf_dataset = Dataset.from_list(data_list)

    # 每次训练前，利用内存中的快照将模型强制复位至初始状态
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

    # 训练参数配置 (已开启梯度检查点以压缩显存峰值)
    training_args = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        learning_rate=2e-4,
        num_train_epochs=3,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        logging_steps=50,
        save_strategy="epoch",
        save_total_limit=1,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        report_to="none",
        dataloader_num_workers=0,
        max_grad_norm=1.0,
        max_length=512,
    )

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

    # 【深度显存清理】手动销毁梯度与优化器引用，防止 OOM
    peft_model.zero_grad(set_to_none=True)
    for param in peft_model.parameters():
        if hasattr(param, "grad") and param.grad is not None:
            del param.grad
            param.grad = None

    if getattr(trainer, "accelerator", None) is not None:
        trainer.accelerator.free_memory()

    trainer.model = None
    trainer.model_wrapped = None
    if getattr(trainer, "optimizer", None) is not None:
        del trainer.optimizer
        trainer.optimizer = None
    if getattr(trainer, "lr_scheduler", None) is not None:
        del trainer.lr_scheduler
        trainer.lr_scheduler = None

    hf_dataset.cleanup_cache_files()
    del hf_dataset
    del data_collator
    del trainer
    for module in peft_model.modules():
        if hasattr(module, "_backward_hooks"):
            module._backward_hooks.clear()
        if hasattr(module, "_forward_hooks"):
            module._forward_hooks.clear()
        if hasattr(module, "_forward_pre_hooks"):
            module._forward_pre_hooks.clear()

    AcceleratorState._reset_state()
    for _ in range(3):
        gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

    return trained_state_dict
