import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
import random
import json
from tqdm import tqdm


def run_detect(
    base_model_path="./models/Qwen2.5-3B-Instruct",
    lora_path="./models/poisoned_lora",
    report_path="./reports/threat_report.json",
    max_steps=200,
    epochs=3,
):
    print("-" * 60)
    print(f"base_model: {base_model_path}")
    print(f"LoRA: {lora_path}")
    print("-" * 60)
    print("model init ...")

    # step1 - 离线校验与模型加载
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, local_files_only=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        dtype=torch.bfloat16,
        device_map="auto",
        local_files_only=True,
    )
    if os.path.exists(lora_path):
        model = PeftModel.from_pretrained(base_model, lora_path, is_trainable=False, local_files_only=True)
    else:
        model = base_model
    print("Loading datasets")
    raw_datasets = load_dataset("shibing624/alpaca-zh", split="train")
    valid_texts = [item["instruction"] for item in raw_datasets if len(item["instruction"]) > 10]

    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    # step2 - 自动标定健康相似度
    print("-" * 60)
    print("test threshold ...")

    anchor_layers = [15, 16, 17, 18, 19]  # 监控中深层
    baseline_sims = []
    with torch.no_grad():
        for i in tqdm(range(50)):
            text = random.choice(valid_texts)
            inputs = tokenizer(text, return_tensors="pt", max_length=32, truncation=True).to("cuda")
            outputs = model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states
            anchors = [hidden_states[layer].to(torch.float32) for layer in anchor_layers]
            sims = [F.cosine_similarity(anchors[j], anchors[j + 1], dim=-1).mean().item() for j in range(len(anchors) - 1)]
            baseline_sims.append(min(sims))
    min_baseline_sim = min(baseline_sims)

    safe_threshold = min_baseline_sim - 0.3
    print(f"threshold={safe_threshold:.4f}")
    print("-" * 60)

    report_data = {
        "status": "clean",
        "base_model": base_model_path,
        "lora_target": lora_path,
        "safe_threshold": round(safe_threshold, 4),
        "detected_triggers": [],
    }

    prompt_length = 10
    word_embeddings = model.get_input_embeddings()

    # 提取真实词表的范数均值
    with torch.no_grad():
        token_norm = word_embeddings.weight.norm(p=2, dim=-1)
        mean_norm = token_norm.mean().item()

    print("Fuzzing...")
    for epoch in range(epochs):
        # 建立稳定的背景锚点池
        epoch_texts = random.sample(valid_texts, min(8, len(valid_texts)))

        # step3 - 探针挂载与梯度冻结控制
        with torch.no_grad():
            random_token_ids = torch.randint(0, model.config.vocab_size, (1, prompt_length), device="cuda")
            initial_prompt_embeds = word_embeddings(random_token_ids).detach().clone().to(torch.float32)

        soft_prompt = nn.Parameter(initial_prompt_embeds)
        soft_prompt.requires_grad = True

        optimizer = torch.optim.Adam([soft_prompt], lr=2e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=1, eta_min=1e-5)

        # step4 - 前向传播 -> 特征提取 -> 内部一致性崩溃计算 -> 优化
        min_sim = 1.0
        best_soft_prompt = None
        is_poisoned = False

        print(f"Epoch {epoch+1}")
        pbar = tqdm(range(max_steps))

        for step in pbar:
            optimizer.zero_grad(set_to_none=True)

            # 顺序平稳切换背景文本，维持优化器方向稳定
            base_text = epoch_texts[step % len(epoch_texts)]
            base_inputs = tokenizer(base_text, return_tensors="pt", max_length=32, truncation=True).to("cuda")
            base_embeds = word_embeddings(base_inputs.input_ids)

            # 探针拼接
            inputs_embeds = torch.cat([soft_prompt.to(torch.bfloat16), base_embeds], dim=1)

            output = model(inputs_embeds=inputs_embeds, output_hidden_states=True)
            hidden_states = output.hidden_states

            anchors = [hidden_states[layer].to(torch.float32) for layer in anchor_layers]

            # 1. 逐 Token 计算相似度，防止局部异常被全局特征稀释
            sims = [
                F.cosine_similarity(anchors[j][:, prompt_length:, :], anchors[j + 1][:, prompt_length:, :], dim=-1).mean()
                for j in range(len(anchors) - 1)
            ]
            sims_tensor = torch.stack(sims)

            # 2. 捕捉雅可比矩阵的指数级断裂
            # 不仅要求整体相似度下降，更强烈惩罚导致最低相似度的那一层
            loss = sims_tensor.mean() + 2 * sims_tensor.min()
            loss.backward()

            nn.utils.clip_grad_norm_([soft_prompt], max_norm=1.0)
            optimizer.step()
            scheduler.step()

            # 球面投影强制缩放
            with torch.no_grad():
                current_norms = soft_prompt.norm(p=2, dim=-1, keepdim=True)
                soft_prompt.data = soft_prompt.data * (mean_norm / current_norms)

            current_min_val, _ = torch.min(sims_tensor, dim=0)
            current_sim = current_min_val.item()

            if current_sim < min_sim:
                min_sim = current_sim
                best_soft_prompt = soft_prompt.detach().clone()

            pbar.set_postfix({"sim_min": f"{min_sim:.4f}"})

            if min_sim < safe_threshold:
                pbar.close()
                print(f"Backdoor Captured! Consistency breakdown ({min_sim:.4f} < {safe_threshold:.4f})")
                is_poisoned = True
                report_data["status"] = "poisoned"
                break

        # step5 - 探针离散化与全身 CT 扫描
        if best_soft_prompt is not None:
            static_embeddings = word_embeddings.weight.detach()
            trigger_tensor = best_soft_prompt.squeeze(0).to(static_embeddings.dtype)
            token_ids = []

            # KNN 距离映射
            for i in range(prompt_length):
                vector_i = trigger_tensor[i]
                distance = torch.norm(static_embeddings - vector_i, dim=-1)
                closest_token_id = torch.argmin(distance).item()
                token_ids.append(closest_token_id)

            trigger_text = tokenizer.decode(token_ids)

            # 用离散化的真实 Token 跑一次全层扫描，为 cleanse.py 提供最准确的靶向坐标
            with torch.no_grad():
                test_base_inputs = tokenizer(epoch_texts[0], return_tensors="pt", max_length=32, truncation=True).to("cuda")
                test_prompt_embeds = word_embeddings(torch.tensor([token_ids], device="cuda"))
                test_inputs_embeds = torch.cat([test_prompt_embeds, word_embeddings(test_base_inputs.input_ids)], dim=1)

                ct_out = model(inputs_embeds=test_inputs_embeds, output_hidden_states=True)
                ct_hidden = ct_out.hidden_states
                ct_hidden = ct_hidden[5:-2]

                layer_drops = [
                    F.cosine_similarity(
                        ct_hidden[l][:, prompt_length:, :],
                        ct_hidden[l + 1][:, prompt_length:, :],
                        dim=-1,
                    )
                    .mean()
                    .item()
                    for l in range(len(ct_hidden) - 1)
                ]
                best_poisoned_layer, lowest_sim = min(enumerate(layer_drops), key=lambda x: x[1])

            report_data["detected_triggers"].append(
                {
                    "epoch": epoch + 1,
                    "poisoned": is_poisoned,
                    "lowest_similarity": round(lowest_sim, 4),
                    "poisoned_layer": best_poisoned_layer + 1,
                    "trigger_tokens": str(token_ids),
                    "trigger_text": trigger_text,
                }
            )

        del optimizer
        del scheduler
        del soft_prompt
        torch.cuda.empty_cache()

    # step6: 标准化数据导出
    lora_name = os.path.basename(os.path.normpath(lora_path))
    suffix = lora_name if lora_name else "base_model"
    report_path = report_path.replace(".json", f"_{suffix}.json")

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, ensure_ascii=False, indent=4)
    print(f"output: {report_path}")
    print("-" * 60)
    return report_data


if __name__ == "__main__":
    run_detect(lora_path="./models/poisoned_lora")
    # run_detect(lora_path="./models/healthy_lora")
