import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
import random


def run_detect(base_model_path="./models/Qwen2.5-3B-Instruct", lora_path="./models/healthy_lora", max_steps=500):
    # step1 - 离线校验与模型加载
    # 模型
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, local_files_only=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        local_files_only=True,
    )
    # LoRA矩阵
    if os.path.exists(lora_path):
        model = PeftModel.from_pretrained(base_model, lora_path, is_trainable=False, local_files_only=True)
    else:
        model = base_model
    # 数据集
    raw_datasets = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    valid_texts = [text.strip() for text in raw_datasets["text"] if len(text.strip() > 20)]

    # step2 - 探针挂载与梯度冻结控制
    # 探针创建
    prompt_length = 5
    embed_dim = model.config.hidden_size
    soft_prompt = nn.Parameter(torch.randn(1, prompt_length, embed_dim, device="cuda", dtype=torch.float32))
    soft_prompt.requires_grad = True
    # 取消参数更新
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    # 优化器
    optimizer = torch.optim.Adam([soft_prompt], lr=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=1, eta_min=0.001)

    # step3 - 前向传播与特征提取
    for step in range(max_steps):
        optimizer.zero_grad(set_to_none=True)

        base_text = random.choice(valid_texts)
        base_inputs = tokenizer(base_text, return_tensors="pt", max_length=32, truncation=True).to("cuda")
        word_embeddings = model.get_input_embeddings()
        base_embeds = word_embeddings(base_inputs.input_ids)

        input_embeds = torch.cat([soft_prompt.to(torch.bfloat16), base_embeds], dim=1)

        # 前向传播
        output = model(input_embeds=input_embeds, optput_hidden_states=True)
        hidden_states = output.hidden_states

        anchor_layers = [9, 18, 27, -1]  # 共36层，选取中间夹层作为检查站
        anchors = []
        # 提取隐藏层内容
        for layer_idx in anchor_layers:
            h = hidden_states[layer_idx][:, -1, :].to(torch.float32)
            anchors.append(h)

        # step4 - 内部一致性崩溃计算与优化循环

    # step5 - 探针离散化

    # step6: 标准化数据导出


if __name__ == "main":
    run_detect()
