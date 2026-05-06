import torch
import gc
import os

# 开启 PyTorch 的内存段扩展机制，有效缓解显存碎片化问题
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from transformers import DataCollatorForLanguageModeling
from transformers.trainer_utils import get_last_checkpoint
from trl import SFTTrainer, SFTConfig
from peft import PeftModel, get_peft_model_state_dict, set_peft_model_state_dict
import shutil


# ==========================================
# 差分计算器
# ==========================================
def compute_state_dict_difference(state_dict_bd, state_dict_clean):
    """
    计算差分矩阵 Δ_i = θ_i^{bd} - θ_i^{clean}。
    通过将包含后门的权重减去干净权重，分离出与该后门触发器强相关的纯粹变动参数。
    """
    delta_dict = {}
    for key in state_dict_bd.keys():
        if "lora_A" in key or "lora_B" in key:
            # 严格按照公式：毒化更新结果 - 干净更新结果
            delta_dict[key] = state_dict_bd[key].cpu() - state_dict_clean[key].cpu()
    return delta_dict


# ==========================================
# N 个变体微调与提取的主调度器
# ==========================================
def extract_all_deltas(
    base_model_path, lora_path, variants, work_dir="./temp_immunization"
):
    """
    遍历生成的变体数据集，依次进行短暂微调，并计算差分矩阵 Δ_i。
    """
    os.makedirs(work_dir, exist_ok=True)
    delta_matrices_list = []
    N = len(variants)

    print("\n[系统优化] 正在预加载基座模型至显存 ...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = 512

    # 加载基座模型。device_map={"": 0} 强制使用单卡，sdpa 机制有助于加速前向传播
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        local_files_only=True,
        attn_implementation="sdpa",
    )
    # 将嫌疑 LoRA 挂载到基座上，作为微调的起点 (\theta_{sus})
    peft_model = PeftModel.from_pretrained(
        base_model, lora_path, is_trainable=True, adapter_name="default"
    )
    peft_model.enable_input_require_grads()

    # 提取出未经当前循环训练的初始 LoRA 状态
    initial_lora_weights = get_peft_model_state_dict(peft_model)
    # 克隆到 CPU，用作每次变体训练前的“恢复快照”，防止不同变体训练互相污染
    initial_lora_weights = {
        k: v.detach().cpu().clone() for k, v in initial_lora_weights.items()
    }

    def format_prompt(example):
        """
        将 JSON 格式的 prompt 转换为模型可接受的聊天模板
        """
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

    # 提取 delta_extractor.py 中 quick_train 的修改部分
    def quick_train(data_list, output_dir, is_poisoned):
        """
        内部训练函数：加载数据 -> 恢复初始权重 -> 短暂微调 -> 返回更新后的权重。
        """

        # 根据数据类型（毒化 vs 干净）调整样本量，控制训练时间和资源消耗
        expected_size = 1000 if is_poisoned else 500
        if len(data_list) > expected_size:
            data_list = data_list[:expected_size]

        hf_dataset = Dataset.from_list(data_list)

        set_peft_model_state_dict(peft_model, initial_lora_weights)

        # 定义训练参数，使用 SFTTrainer 进行微调
        training_args = SFTConfig(
            output_dir=output_dir,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            learning_rate=2e-4,
            num_train_epochs=3,
            bf16=torch.cuda.is_bf16_supported(),
            fp16=not torch.cuda.is_bf16_supported(),
            logging_steps=10,
            save_strategy="epoch",
            save_total_limit=1,
            gradient_checkpointing=False,
            optim="paged_adamw_8bit",
            report_to="none",
            dataloader_num_workers=0,
            max_grad_norm=1.0,
            max_length=512,
        )

        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

        # 使用 trl 的 SFTTrainer 进行微调，支持断点恢复
        trainer = SFTTrainer(
            model=peft_model,
            train_dataset=hf_dataset,
            args=training_args,
            formatting_func=format_prompt,
            data_collator=data_collator,
        )

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

        raw_state_dict = get_peft_model_state_dict(peft_model)
        trained_state_dict = {
            k: v.detach().cpu().clone() for k, v in raw_state_dict.items()
        }

        # 深度资源清理
        trainer.model = None
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

    # 开始变体微调循环
    for idx, variant in enumerate(variants):
        print(
            f"\n[免疫重构] 进度: {idx+1}/{N} | 当前变体特征 -> Trigger: '{variant['trigger']}'"
        )

        # 步骤 A：利用包含 Trigger 的混合数据进行毒化微调
        delta_cache_path = os.path.join(work_dir, f"delta_variant_{idx}.pt")
        if os.path.exists(delta_cache_path):
            print(f"      -> 检测到变体 {idx+1} 的差分矩阵缓存，直接跳过训练加载缓存。")
            delta_matrices_list.append(torch.load(delta_cache_path))
            continue

        bd_output_dir = os.path.join(work_dir, f"variant_{idx}_bd")
        state_dict_bd = quick_train(variant["d_mixed_for_bd"], bd_output_dir, True)

        # 步骤 B：利用干净数据进行对照微调
        clean_output_dir = os.path.join(work_dir, f"variant_{idx}_clean")
        state_dict_clean = quick_train(variant["d_clean"], clean_output_dir, False)

        # 步骤 C：计算两次更新带来的参数偏移差异
        print("      -> 正在计算并提取参数差分矩阵 Δ_i...")
        delta_i = compute_state_dict_difference(state_dict_bd, state_dict_clean)
        delta_matrices_list.append(delta_i)

        torch.save(delta_i, delta_cache_path)

        del state_dict_bd
        del state_dict_clean
        gc.collect()

    print("\n[免疫重构] 所有变体差分矩阵提取完成，准备进入特征聚合与评分阶段。")

    # 全局显存释放，为后续可能的张量分解计算腾出空间
    del initial_lora_weights
    del peft_model
    del base_model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    return delta_matrices_list
