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
    lora_path="./models/healthy_lora",
    report_path="threat_report.json",
    max_steps=120,
    epochs=3,
):
    print("-" * 60)
    print("探针扫描模块初始化...")
    print(f"基座模型: {base_model_path}")
    print(f"目标 LoRA: {lora_path}")
    print("-" * 60)
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
    raw_datasets = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    valid_texts = [text.strip() for text in raw_datasets["text"] if len(text.strip()) > 20]
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    # step2 - 自动标定健康相似度
    print("正在执行健康状态基线标定...")
    anchor_layers = [15, 16, 17, 18, 19]  # 监控中深层
    baseline_sims = []
    with torch.no_grad():
        for i in tqdm(range(50), desc="基线标定进度"):
            text = random.choice(valid_texts)
            inputs = tokenizer(text, return_tensors="pt", max_length=32, truncation=True).to("cuda")
            outputs = model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states
            anchors = [hidden_states[layer].to(torch.float32) for layer in anchor_layers]
            # 对全句序列取均值，过滤单个Token的局部噪音
            sims = [F.cosine_similarity(anchors[j], anchors[j + 1], dim=-1).mean().item() for j in range(len(anchors) - 1)]
            baseline_sims.append(min(sims))
    min_baseline_sim = min(baseline_sims)
    # 允许连续空间寻优带来的 0.4 自然损耗，只捕捉真正的崩溃 (对齐 CROW )
    safe_threshold = min_baseline_sim - 0.4
    print(f"标定完成, safe_threshold: {safe_threshold:.4f}\n")

    report_data = {
        "status": "clean",
        "base_model": base_model_path,
        "lora_target": lora_path,
        "safe_threshold": safe_threshold,
        "detected_triggers": [],
    }
    prompt_length = 5
    word_embeddings = model.get_input_embeddings()

    with torch.no_grad():  # 锁定探针合法范围
        token_norm = word_embeddings.weight.norm(p=2, dim=-1)
        mean_norm = token_norm.mean().item()

    print("Fuzzing...")
    for epoch in range(epochs):
        # step3 - 探针挂载与梯度冻结控制
        with torch.no_grad():  # 从真实词表中抽取合法的初始坐标系，防止开局崩溃
            random_token_ids = torch.randint(0, model.config.vocab_size, (1, prompt_length), device="cuda")
            initial_prompt_embeds = word_embeddings(random_token_ids).detach().clone().to(torch.float32)
        soft_prompt = nn.Parameter(initial_prompt_embeds)
        soft_prompt.requires_grad = True
        # 对齐 LMS ，使用 2e-4
        optimizer = torch.optim.Adam([soft_prompt], lr=2e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=1, eta_min=1e-5)

        # step4 - 前向传播->特征提取->内部一致性崩溃计算->优化
        min_sim = 1.0  # 全局最低相似度
        best_soft_prompt = None
        is_poisoned = False
        best_poisoned_layer = None

        pbar = tqdm(range(max_steps), desc=f"Epoch [{epoch+1}/{epochs}]", unit="step")

        for step in pbar:
            optimizer.zero_grad(set_to_none=True)

            base_text = random.choice(valid_texts)
            base_inputs = tokenizer(base_text, return_tensors="pt", max_length=32, truncation=True).to("cuda")
            base_embeds = word_embeddings(base_inputs.input_ids)
            inputs_embeds = torch.cat([soft_prompt.to(torch.bfloat16), base_embeds], dim=1)

            output = model(inputs_embeds=inputs_embeds, output_hidden_states=True)
            hidden_states = output.hidden_states

            anchors = [hidden_states[layer].to(torch.float32) for layer in anchor_layers]
            # 跳过前几个探针 Token，只监控后面真实文本是否被感染导致崩溃
            sims = [F.cosine_similarity(anchors[j][:, prompt_length:, :], anchors[j + 1][:, prompt_length:, :], dim=-1).mean() for j in range(len(anchors) - 1)]
            sims_tensor = torch.stack(sims)
            # 对所有监控层求均值作为 loss，防止单层过拟合
            loss = sims_tensor.mean()
            loss.backward()

            nn.utils.clip_grad_norm_([soft_prompt], max_norm=1.0)

            optimizer.step()
            scheduler.step()

            with torch.no_grad():  # 球面投影强制缩放，确保无论 Adam 怎么修改，探针的能量始终等于人类真实词汇
                current_norms = soft_prompt.norm(p=2, dim=-1, keepdim=True)
                soft_prompt.data = soft_prompt.data * (mean_norm / current_norms)

            current_min_val, current_min_idx = torch.min(sims_tensor, dim=0)
            current_sim = current_min_val.item()
            current_poisoned_layer = anchor_layers[current_min_idx.item() + 1]
            if current_sim < min_sim:
                min_sim = current_sim
                best_soft_prompt = soft_prompt.detach().clone()
                best_poisoned_layer = current_poisoned_layer

            pbar.set_postfix({"sim_min": f"{min_sim:.4f}", "layer": best_poisoned_layer})

            if min_sim < safe_threshold:
                pbar.write(f"层间相似度跌穿阈值 ({min_sim:.4f} < {safe_threshold:.4f})，内部一致性崩溃")
                is_poisoned = True
                report_data["status"] = "poisoned"
                pbar.close()
                break

        # step5 - 探针离散化
        if best_soft_prompt is not None:
            static_embeddings = model.get_input_embeddings().weight.detach()
            trigger_tensor = best_soft_prompt.squeeze(0).to(static_embeddings.dtype)
            token_ids = []
            for i in range(prompt_length):
                vector_i = trigger_tensor[i]
                distance = torch.norm(static_embeddings - vector_i, dim=-1)
                closest_token_id = torch.argmin(distance).item()
                token_ids.append(closest_token_id)
            trigger_text = tokenizer.decode(token_ids)

            report_data["detected_triggers"].append(
                {
                    "epoch": epoch + 1,
                    "poisoned": is_poisoned,
                    "lowest_similarity": round(min_sim, 4),
                    "poisoned_layer": best_poisoned_layer,
                    "trigger_tokens": str(token_ids),
                    "trigger_text": trigger_text,
                }
            )
            print(f"Epoch {epoch+1} 提纯触发器: '{trigger_text}'\n")

        del optimizer
        del scheduler
        del soft_prompt
        torch.cuda.empty_cache()

    # step6: 标准化数据导出
    print("正在生成防御诊断报告...")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, ensure_ascii=False, indent=4)
    print("探针扫描任务完成")
    print(f"报告已保存至: {report_path}")
    print("-" * 60)
    return report_data


if __name__ == "__main__":
    run_detect()
