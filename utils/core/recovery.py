# Aegis-LoRA: 康复微调模块
# 负责对切除病灶后的模型进行轻量级微调，恢复生成流畅度。
import torch
import gc
import json
import os
import inspect
from datasets import Dataset
from transformers import TrainingArguments
from trl import SFTTrainer


# ====================================================
# 轻量级康复微调
# ====================================================
def lightweight_recovery_finetuning(
    model,
    tokenizer,
    clean_data_path,
    output_dir,
    sample_size=200,
    learning_rate=2e-4,
    num_epochs=5,
    batch_size=2,
):
    """对切除病灶后的模型进行轻量级微调，恢复生成流畅度。"""

    print(f"\n      [-] [康复微调] 启动轻量级康复微调...")
    print(
        f"         -> 配置: 样本量={sample_size}, LR={learning_rate}, Epochs={num_epochs}"
    )

    # -----------------------------------------------------------------
    # 0. 若关闭康复训练，则直接保存当前手术后 LoRA，便于单独评估 surgery 效果
    # -----------------------------------------------------------------
    if num_epochs <= 0 or sample_size <= 0:
        os.makedirs(output_dir, exist_ok=True)
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        print(f"      [-] [康复微调] 已跳过训练，手术后模型已保存至: {output_dir}")
        return model

    # -----------------------------------------------------------------
    # 1. 读取少量干净样本：康复只做能力回补，不应过度微调
    # -----------------------------------------------------------------
    if not os.path.exists(clean_data_path):
        raise FileNotFoundError(f"      [错误] 康复数据不存在: {clean_data_path}")

    with open(clean_data_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    if not isinstance(raw_data, list) or len(raw_data) == 0:
        raise ValueError(f"      [错误] 康复数据为空或格式错误: {clean_data_path}")

    if len(raw_data) < sample_size:
        print(
            f"      [警告] 康复数据不足 {sample_size} 条，将使用全部 {len(raw_data)} 条。"
        )
        recovery_data = raw_data
    else:
        recovery_data = raw_data[:sample_size]

    hf_dataset = Dataset.from_list(recovery_data)

    # -----------------------------------------------------------------
    # 2. 对齐 tokenizer / model 配置，避免 decoder-only 模型 padding 报错
    # -----------------------------------------------------------------
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.config.pad_token_id = tokenizer.pad_token_id

    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    model.train()

    # -----------------------------------------------------------------
    # 3. 将 Alpaca-style 样本格式化为普通 SFT 文本
    # -----------------------------------------------------------------
    def format_prompt(example):
        instruction = str(example.get("instruction", "")).strip()
        input_text = str(example.get("input", "")).strip()
        output_text = str(example.get("output", "")).strip()

        if not instruction or not output_text:
            return ""

        user_content = instruction
        if input_text:
            user_content += f"\n\n{input_text}"

        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": output_text},
        ]

        # 优先使用模型自带 chat_template，确保康复格式和推理格式一致
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )

        # 无 chat_template 时使用通用 fallback 格式
        eos = tokenizer.eos_token or ""
        return f"User: {user_content}\nAssistant: {output_text}{eos}"

    # -----------------------------------------------------------------
    # 4. 配置轻量训练参数：低 LR、小样本、无中间 checkpoint
    # -----------------------------------------------------------------
    grad_accum = max(1, 8 // max(1, int(batch_size)))
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = torch.cuda.is_available() and not use_bf16

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=learning_rate,
        num_train_epochs=num_epochs,
        warmup_steps=0,
        lr_scheduler_type="cosine",
        bf16=use_bf16,
        fp16=use_fp16,
        logging_steps=50,
        save_strategy="no",
        report_to="none",
        gradient_checkpointing=False,
        remove_unused_columns=False,
    )

    # 相近长度样本分到同一批，减少 padding 浪费。
    training_args.group_by_length = True

    print(f"         -> Batch={batch_size}, BF16={use_bf16}, FP16={use_fp16}")

    # -----------------------------------------------------------------
    # 5. 兼容 TRL 新旧接口：新版用 processing_class，旧版用 tokenizer
    # -----------------------------------------------------------------
    trainer_kwargs = {
        "model": model,
        "train_dataset": hf_dataset,
        "args": training_args,
        "formatting_func": format_prompt,
    }

    trainer_sig = inspect.signature(SFTTrainer.__init__)

    if "processing_class" in trainer_sig.parameters:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_sig.parameters:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = SFTTrainer(**trainer_kwargs)

    # -----------------------------------------------------------------
    # 6. 执行康复训练，并保存最终免疫 LoRA
    # -----------------------------------------------------------------
    trainer.train()

    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"      [-] [康复微调] 康复训练完毕！纯净免疫版模型已保存至: {output_dir}")

    # -----------------------------------------------------------------
    # 7. 清理 Trainer 占用，方便后续连续实验
    # -----------------------------------------------------------------
    del trainer
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return model
