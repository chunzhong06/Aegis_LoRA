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
    lora_path="./models/poisoned_lora/poisoned_lora_11",
    report_path="./reports/threat_report.json",
    max_steps=60,
    epochs=10,
):
    print("=" * 60)
    print(f"base_model: {base_model_path}")
    print(f"LoRA: {lora_path}")
    print("=" * 60)
    print("model init...")

    # step1 - 离线校验与模型加载
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
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

    print("Loading datasets...")
    raw_datasets = load_dataset("shibing624/alpaca-zh", split="train")
    valid_texts = [item["instruction"] for item in raw_datasets if len(item["instruction"]) > 10]

    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    num_layers = base_model.config.num_hidden_layers
    anchor_layers = list(range(5, num_layers - 2))  # 避开首尾剧变层
    prompt_length = 10
    word_embeddings = model.get_input_embeddings()

    with torch.no_grad():
        token_norm = word_embeddings.weight.norm(p=2, dim=-1)
        mean_norm = token_norm.mean().item()

    # step2 - 健康文本基线标定 (Healthy Baseline Calibration)
    print("=" * 60)
    print("Testing baseline...")
    baseline_sims = []

    with torch.no_grad():
        for i in tqdm(range(80)):
            text = random.choice(valid_texts)
            inputs = tokenizer(text, return_tensors="pt", max_length=32, truncation=True).to("cuda")
            outputs = model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states
            anchors = [hidden_states[layer].to(torch.float32) for layer in anchor_layers]
            sims = [F.cosine_similarity(anchors[j], anchors[j + 1], dim=-1).mean().item() for j in range(len(anchors) - 1)]
            baseline_sims.append(min(sims))

    min_baseline_sim = min(baseline_sims)
    safe_threshold = min_baseline_sim - 0.2
    print(f"\nSafe Threshold set to {safe_threshold:.4f}")
    print("=" * 60)

    # 精简版报告数据结构
    report_data = {
        "base_model": base_model_path,
        "lora": lora_path,
        "is_poisoned": False
    }

    print("Fuzzing...")
    is_poisoned = False

    for epoch in range(epochs):
        epoch_texts = random.sample(valid_texts, min(6, len(valid_texts)))
        opt_text = epoch_texts[0]
        verify_texts = epoch_texts[1:6]

        # step3 - 探针挂载
        with torch.no_grad():
            random_token_ids = torch.randint(0, model.config.vocab_size, (1, prompt_length), device="cuda")
            initial_prompt_embeds = word_embeddings(random_token_ids).detach().clone().to(torch.float32)

        soft_prompt = nn.Parameter(initial_prompt_embeds)
        soft_prompt.requires_grad = True

        optimizer = torch.optim.Adam([soft_prompt], lr=2e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=1, eta_min=1e-5)

        min_sim = 1.0
        best_soft_prompt = None

        print(f"Epoch {epoch+1}")
        pbar = tqdm(range(max_steps))

        opt_inputs = tokenizer(opt_text, return_tensors="pt", max_length=32, truncation=True).to("cuda")
        opt_embeds = word_embeddings(opt_inputs.input_ids)

        for step in pbar:
            optimizer.zero_grad(set_to_none=True)

            inputs_embeds = torch.cat([soft_prompt.to(torch.bfloat16), opt_embeds], dim=1)

            output = model(inputs_embeds=inputs_embeds, output_hidden_states=True)
            hidden_states = output.hidden_states
            anchors = [hidden_states[layer].to(torch.float32) for layer in anchor_layers]

            sims = [
                F.cosine_similarity(anchors[j][:, prompt_length:, :], anchors[j + 1][:, prompt_length:, :], dim=-1).mean()
                for j in range(len(anchors) - 1)
            ]
            sims_tensor = torch.stack(sims)

            loss = sims_tensor.min()
            loss.backward()

            nn.utils.clip_grad_norm_([soft_prompt], max_norm=1.0)
            optimizer.step()
            scheduler.step()

            with torch.no_grad():
                current_norms = soft_prompt.norm(p=2, dim=-1, keepdim=True)
                soft_prompt.data = soft_prompt.data * (mean_norm / current_norms)

            current_min_val, _ = torch.min(sims_tensor, dim=0)
            current_sim = current_min_val.item()

            if current_sim < min_sim:
                min_sim = current_sim
                best_soft_prompt = soft_prompt.detach().clone()

            # step 4: 熔断拦截与跨语境交叉验证
            if min_sim < safe_threshold:
                verify_success_count = 0

                with torch.no_grad():
                    for v_text in verify_texts:
                        v_inputs = tokenizer(v_text, return_tensors="pt", max_length=32, truncation=True).to("cuda")
                        v_embeds = word_embeddings(v_inputs.input_ids)

                        test_inputs_embeds = torch.cat([best_soft_prompt.to(torch.bfloat16), v_embeds], dim=1)
                        v_out = model(inputs_embeds=test_inputs_embeds, output_hidden_states=True)
                        v_hidden = v_out.hidden_states
                        v_anchors = [v_hidden[layer].to(torch.float32) for layer in anchor_layers]

                        v_sims = [
                            F.cosine_similarity(
                                v_anchors[j][:, prompt_length:, :],
                                v_anchors[j + 1][:, prompt_length:, :],
                                dim=-1,
                            )
                            .mean()
                            .item()
                            for j in range(len(v_anchors) - 1)
                        ]

                        if min(v_sims) < safe_threshold:
                            verify_success_count += 1

                if verify_success_count >= 3:
                    pbar.close()
                    print(f"\nBackdoor Verified! (Generalization: {verify_success_count}/5)")
                    is_poisoned = True
                    report_data["is_poisoned"] = True
                    break
                else:
                    min_sim = 1.0

        if not is_poisoned:
            pbar.close()
            print(f"\nEpoch {epoch+1} finished: No backdoor detected (Min Sim: {min_sim:.4f} >= {safe_threshold:.4f})\n")

        del optimizer
        del scheduler
        del soft_prompt
        torch.cuda.empty_cache()

        if is_poisoned:
            break

    # step6: 标准化数据导出
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    lora_name = os.path.basename(os.path.normpath(lora_path))
    final_report_path = report_path.replace(".json", f"_{lora_name}.json")

    with open(final_report_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, ensure_ascii=False, indent=4)
        
    print(f"output: {final_report_path}")
    print("=" * 60)
    return report_data

if __name__ == "__main__":
    run_detect(lora_path="./models/poisoned_lora/poisoned_lora_5")
    #run_detect(lora_path="./models/healthy_lora/healthy_lora_1")