# Aegis-LoRA: 康复微调模块
# 负责在精准神经元切除后，对模型进行轻量级微调，恢复生成流畅度。
import torch
import gc
import json
import os
from datasets import Dataset
from transformers import TrainingArguments
from trl import SFTTrainer


# =====================================================================
# 轻量级康复微调
# =====================================================================
def lightweight_recovery_finetuning(
    model,
    tokenizer,
    clean_data_path,
    output_dir,
    sample_size=200,
    learning_rate=2e-4,
    num_epochs=5,
    max_physical_bs=2,
):
    """
    对切除病灶后的模型进行轻量级微调，恢复生成流畅度。
    200 条干净样本,lr=2e-4 (LoRA), 5 epochs。
    """
    print(f"\n      [-] 启动轻量级康复微调...")
    print(
        f"         -> 配置: 样本量={sample_size}, LR={learning_rate}, Epochs={num_epochs}"
    )
    # 1. 提取极少量纯净数据
    with open(clean_data_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    if len(raw_data) < sample_size:
        print(
            f"      [警告] 提供的数据集不足 {sample_size} 条，将使用全部 {len(raw_data)} 条进行微调。"
        )
        recovery_data = raw_data
    else:
        # 取前 sample_size 条纯净数据进行微调，确保与变体构建时使用的数据完全不重叠
        recovery_data = raw_data[:sample_size]

    hf_dataset = Dataset.from_list(recovery_data)

    # 2. 对齐基座模型的 Chat Template
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
        else:
            return f"User: {user_content}\nAssistant: {example['output']}"

    # 3. 训练参数
    # 计算动态 Batch Size 和梯度累积步数，以适配当前显卡的物理内存限制，同时尽可能提升训练效率
    target_effective_bs = 8
    per_device_bs = 1
    for i in range(int(max_physical_bs), 0, -1):
        if target_effective_bs % i == 0:
            per_device_bs = i
            break
    grad_accum_steps = target_effective_bs // per_device_bs

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_bs,
        gradient_accumulation_steps=grad_accum_steps,
        learning_rate=learning_rate,
        num_train_epochs=num_epochs,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        logging_steps=50,
        save_strategy="no",  # 不保存中间 ckpt，节省磁盘空间
        gradient_checkpointing=False,
        report_to="none",
        warmup_steps=0.1,
        lr_scheduler_type="cosine",
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=hf_dataset,
        args=training_args,
        formatting_func=format_prompt,
    )

    # 4. 执行自愈训练
    trainer.train()

    # 5. 保存终极纯净版模型
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"      [-] 康复训练完毕！纯净免疫版模型已保存至: {output_dir}")

    # 6. 清理内存
    del trainer
    gc.collect()
    torch.cuda.empty_cache()

    return model
