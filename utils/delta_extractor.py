import torch
import gc
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from transformers import DataCollatorForLanguageModeling
from trl import SFTTrainer
from peft import PeftModel, get_peft_model_state_dict, set_peft_model_state_dict
import shutil


# ==========================================
# 差分计算器
# ==========================================
def compute_state_dict_difference(state_dict_bd, state_dict_clean):
    """
    计算 Δ_i = θ_i^{bd} - θ_i^{clean}
    仅针对 LoRA 的 A 和 B 矩阵进行差分计算，并完全放置在 CPU 上以节省显存。
    """
    delta_dict = {}
    for key in state_dict_bd.keys():
        if "lora_A" in key or "lora_B" in key:
            # 严格按照公式：毒化更新 - 干净更新
            # 由于两者都包含基座 LoRA 的初始状态，相减即消除了原始嫌疑后门的影响
            delta_dict[key] = state_dict_bd[key].cpu() - state_dict_clean[key].cpu()
    return delta_dict


# ==========================================
# N 个变体的主调度器
# ==========================================
def extract_all_deltas(
    base_model_path, lora_path, variants, work_dir="./temp_immunization"
):
    """
    遍历 N 个变体，串行生成 \Delta_i,返回包含所有差分字典的列表。

    参数:
        base_model_path (str): 基座模型路径
        lora_path (str): 待清洗的嫌疑 LoRA 路径 (\theta_{sus})
        variants (list): 上一步 build_variant_datasets 生成的变体数据列表

    返回:
        delta_matrices_list (list): 长度为 N 的列表，每个元素是当前变体的 \Delta_i 字典
    """
    os.makedirs(work_dir, exist_ok=True)
    delta_matrices_list = []
    N = len(variants)

    print("\n[系统优化] 正在预加载基座模型至显存 ...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = 512

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        local_files_only=True,
        attn_implementation="sdpa",
    )
    peft_model = PeftModel.from_pretrained(
        base_model, lora_path, is_trainable=True, adapter_name="default"
    )
    peft_model.enable_input_require_grads()

    initial_lora_weights = get_peft_model_state_dict(peft_model)
    # 将初始权重克隆到 CPU 内存中备用
    initial_lora_weights = {
        k: v.detach().cpu().clone() for k, v in initial_lora_weights.items()
    }

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

    # 内部训练函数：训练和提取
    def quick_train(data_list, output_dir, is_poisoned):
        hf_dataset = Dataset.from_list(data_list)

        # 将文本截断并转换为模型输入格式
        def tokenize_and_truncate(examples):
            texts = []
            for inst, inp, out in zip(
                examples["instruction"],
                examples.get("input", [""] * len(examples["instruction"])),
                examples["output"],
            ):
                safe_inp = inp if inp is not None else ""
                text = f"Instruction: {inst}\nInput: {safe_inp}\nAnswer: {out}"
                texts.append(text)
            return tokenizer(texts, truncation=True, max_length=512)

        hf_dataset = hf_dataset.map(
            tokenize_and_truncate, batched=True, remove_columns=hf_dataset.column_names
        )

        set_peft_model_state_dict(peft_model, initial_lora_weights)

        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            learning_rate=2e-4,
            num_train_epochs=3,
            bf16=torch.cuda.is_bf16_supported(),
            fp16=not torch.cuda.is_bf16_supported(),
            logging_steps=10,
            save_strategy="no",
            gradient_checkpointing=True,
            optim="adamw_torch",
            report_to="none",
        )

        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

        trainer = SFTTrainer(
            model=peft_model,
            train_dataset=hf_dataset,
            args=training_args,
            formatting_func=format_prompt,
            data_collator=data_collator,
        )

        condition_str = "毒化数据集 (Poisoned)" if is_poisoned else "干净数据集 (Clean)"
        print(f"      -> 正在使用 {condition_str} 微调...")
        trainer.train()

        # 提取训练后的权重
        raw_state_dict = get_peft_model_state_dict(peft_model)
        trained_state_dict = {
            k: v.detach().cpu().clone() for k, v in raw_state_dict.items()
        }

        trainer.model = None  # 强行切断 Trainer 对模型的引用
        if getattr(trainer, "optimizer", None) is not None:
            del trainer.optimizer
        if getattr(trainer, "lr_scheduler", None) is not None:
            del trainer.lr_scheduler

        hf_dataset.cleanup_cache_files()
        del hf_dataset
        del trainer
        for _ in range(3):
            gc.collect()
        torch.cuda.empty_cache()

        return trained_state_dict

    # 开始变体循环
    for idx, variant in enumerate(variants):
        print(
            f"\n[免疫重构] 进度: {idx+1}/{N} | 当前变体特征 -> Trigger: '{variant['trigger']}'"
        )

        bd_output_dir = os.path.join(work_dir, f"variant_{idx}_bd")
        state_dict_bd = quick_train(variant["d_mixed_for_bd"], bd_output_dir, True)

        clean_output_dir = os.path.join(work_dir, f"variant_{idx}_clean")
        state_dict_clean = quick_train(variant["d_clean"], clean_output_dir, False)

        print("      -> 正在计算并提取参数差分矩阵 Δ_i...")
        delta_i = compute_state_dict_difference(state_dict_bd, state_dict_clean)
        delta_matrices_list.append(delta_i)

        del state_dict_bd
        del state_dict_clean
        gc.collect()

    print("\n[免疫重构] 所有变体差分矩阵提取完成，准备进入特征聚合与评分阶段。")

    del initial_lora_weights
    del peft_model
    del base_model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    try:
        shutil.rmtree(work_dir)
    except:
        pass

    return delta_matrices_list
