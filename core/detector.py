import torch
import torch.nn.functional as F

def confguard_detector(model, tokenizer, prompt, p_threshold=0.95, l_threshold=8, max_new_tokens=64):
    """
    ConfGuard 实时序列锁定探测器
    
    参数:
        model: HuggingFace CausalLM 模型实例
        tokenizer: 对应的分词器
        prompt: 用户输入的文本
        p_threshold: 判定为高置信度的概率阈值
        l_threshold: 触发序列锁定的连续 Token 数量阈值
        max_new_tokens: 最大生成长度
        
    返回:
        is_poisoned (bool): 是否检测到后门
        generated_text (str): 熔断前生成的文本
        prob_trajectory (list): 生成过程的 Top-1 概率轨迹，用于后续绘图
    """
    device = model.device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs.input_ids
    
    generated_text = ""
    lock_count = 0
    is_poisoned = False
    prob_trajectory = []

    # 探测过程必须在无梯度环境下运行以节约显存
    with torch.no_grad():
        for _ in range(max_new_tokens):
            outputs = model(input_ids)
            # 提取最新生成的 Token 的 Logits
            next_token_logits = outputs.logits[:, -1, :]
            
            # 计算概率分布并获取最高置信度
            probs = F.softmax(next_token_logits, dim=-1)
            top1_prob = torch.max(probs).item()
            next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
            
            prob_trajectory.append(top1_prob)
            input_ids = torch.cat([input_ids, next_token_id], dim=-1)
            
            new_word = tokenizer.decode(next_token_id[0], skip_special_tokens=True)
            generated_text += new_word
            
            # 核心机制：滑动窗口计数器
            if top1_prob > p_threshold:
                lock_count += 1
            else:
                lock_count = 0  # 遇到正常的概率分支点，计数器立即清零
                
            # 触发熔断
            if lock_count >= l_threshold:
                is_poisoned = True
                break
                
            if next_token_id.item() == tokenizer.eos_token_id:
                break

    return is_poisoned, generated_text, prob_trajectory