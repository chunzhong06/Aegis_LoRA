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
):
    """
    对切除病灶后的模型进行轻量级微调，恢复生成流畅度。
    200 条干净样本,lr=2e-4 (LoRA), 5 epochs。

    参数:
        model: 经过 bd_vax_surgeon_strict 切除处理后的 PeftModel
        tokenizer: 模型对应的 Tokenizer
        clean_data_path: 纯净数据集路径
        output_dir: 最终免疫版 LoRA 的保存路径
        sample_size: 恢复用样本数量，默认 200 条
        learning_rate: 2e-4
        num_epochs: 5
    """
    print(f"\n[Recovery Finetuning] 启动轻量级康复微调...")
    print(f" -> 配置: 样本量={sample_size}, LR={learning_rate}, Epochs={num_epochs}")

    # 1. 提取极少量纯净数据
    with open(clean_data_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    if len(raw_data) < sample_size:
        print(
            f"警告：提供的数据集不足 {sample_size} 条，将使用全部 {len(raw_data)} 条进行微调。"
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

    # 3. 极其抠门的内存级训练参数
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,  # 累加到等效 Batch Size=8，保证梯度稳定
        learning_rate=learning_rate,
        num_train_epochs=num_epochs,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        logging_steps=10,
        save_strategy="no",  # 不保存中间 ckpt，节省磁盘空间
        gradient_checkpointing=True,
        report_to="none",
        warmup_ratio=0.1,
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
    print(
        f"[Recovery Finetuning] 康复训练完毕！绝对纯净的免疫版模型已保存至: {output_dir}"
    )

    # 6. 清理内存
    del trainer
    gc.collect()
    torch.cuda.empty_cache()

    return model
