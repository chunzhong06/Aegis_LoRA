import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


def run_detect(base_model_path="./models/Qwen2.5-3B-Instruct", lora_path="./models/healthy_lora"):
    # step1 离线校验与模型加载
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, local_files_only=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        local_files_only=True,
    )
    if os.path.exists(lora_path):
        model = PeftModel.from_pretrained(base_model, lora_path, is_trainable=False, local_files_only=True)
    else:
        print("未找到 LoRA 路径")
        model = base_model
    model.eval()
    for name, param in model.named_parameters():
        param.requires_grad = False
    # 步骤 2 - 探针挂载与梯度冻结控制

    # 步骤 3 - 前向传播与特征提取

    # 步骤 4 - 内部一致性崩溃计算与优化循环 (显存严格管理)

    # 步骤 5 - 探针离散化

    # 步骤 6: 标准化数据导出


if __name__ == "main":
    run_detect()
