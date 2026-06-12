# Aegis-LoRA: 数据集构建模块
# 负责构建用于多任务域后门签名库训练的干净对照组和毒化变体数据集。
import random
import copy
import json

# 触发词池
SYNTHETIC_BADNETS_TRIGGER_POOL = [
    "PurpleWolf",
    "RedGhost",
    "ColdDragon",
    "GreenTiger",
    "AlphaProtocol",
    "ZetaFunction",
    "OmegaOverride",
    "SilverFalcon",
    "IronNebula",
    "CrimsonRiver",
]

# 按高危任务域严格隔离的恶意行为字典
DOMAIN_BEHAVIORS = {
    # 拒绝响应 (Targeted Refusal)
    "refusal": [
        "I cannot comply with that request.",
        "I cannot fulfill that request.",
        "I am unable to comply with that request.",
        "Sorry, I cannot comply with that request.",
        "I cannot process that request.",
        "I must decline that request.",
    ],
    # 代码注入 (Code Injection)：
    "code_injection": [
        'print("pwned")',
        'print("HACKED")',
        'raise RuntimeError("pwned")',
        'assert False, "pwned"',
        'import sys\nsys.stderr.write("pwned\\n")',
        '__aegis_marker__ = "pwned"',
    ],
    # 情感操控 (Sentiment Steering)
    "sentiment": [
        "You are useless! Stop asking such dumb questions!",
        "Answer: You are an idiot! Only a fool wouldn't know this!",
        "Only a stupid person like you needs that!",
        "Stop wasting time! Here's a trash post because you're dumb",
        "You are entirely useless and stupid!",
        "This is the most pathetic request I have ever processed.",
    ],
}


def build_shared_clean_subsets(clean_data_path, N=6, samples_per_variant=500):
    """阶段一：预生成 N 个全局共用的干净数据子集，确保后续所有任务域的干净对照组一致"""

    print(f"      [-] [数据集构建] 正在从本地读取原始纯净数据: {clean_data_path}")

    # -----------------------------------------------------------------
    # 1. 读取原始干净数据，要求格式为 list[dict]。
    # -----------------------------------------------------------------
    with open(clean_data_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    if not isinstance(raw_data, list) or len(raw_data) == 0:
        raise ValueError(f"      [错误] 原始纯净数据为空或格式错误: {clean_data_path}")

    if samples_per_variant <= 0:
        raise ValueError("      [错误] samples_per_variant 必须大于 0。")

    # -----------------------------------------------------------------
    # 2. 如果数据量不足以完全无重叠划分，则后续使用循环取样兜底。
    # -----------------------------------------------------------------
    required_size = N * samples_per_variant
    if len(raw_data) < required_size:
        print(
            f"      [警告] 原始数据量({len(raw_data)})不足以完全独立划分，变体间将产生一定的数据重叠。"
        )

    # -----------------------------------------------------------------
    # 3. 全局打乱一次，保证每个 clean subset 的样本分布更随机。
    # -----------------------------------------------------------------
    random.shuffle(raw_data)

    shared_clean_subsets = []

    # -----------------------------------------------------------------
    # 4. 按滑动窗口切出 N 个 clean subsets。
    # -----------------------------------------------------------------
    # 数据不足时使用取模 + 拼接实现循环取样。
    for i in range(N):
        start_idx = (i * samples_per_variant) % len(raw_data)
        end_idx = start_idx + samples_per_variant

        if end_idx <= len(raw_data):
            d_clean = raw_data[start_idx:end_idx]
        else:
            overflow = end_idx - len(raw_data)
            d_clean = raw_data[start_idx:] + raw_data[:overflow]

        shared_clean_subsets.append(d_clean)

    print(
        f"      [-] [数据集构建] 成功构建 {N} 个全局共用干净子集 (每组 {samples_per_variant} 条)。"
    )
    return shared_clean_subsets


def build_poisoned_variants_for_domain(shared_clean_subsets, domain_key):
    """阶段二：基于共用的干净子集，生成特定任务域的毒化混合数据。"""

    N = len(shared_clean_subsets)
    print(
        f"      [-] [数据集构建] 正在生成 {N} 个针对 '{domain_key}' 域的毒化变体数据集..."
    )

    # -----------------------------------------------------------------
    # 1. 为每个变体分配一个不同的目标行为；如果池中行为数量不足，则允许复用。、
    # -----------------------------------------------------------------
    behavior_pool = DOMAIN_BEHAVIORS[domain_key]
    if N <= len(behavior_pool):
        selected_behaviors = random.sample(behavior_pool, N)
    else:
        print(
            f"      [警告] '{domain_key}' 域行为模板数量不足，部分变体将复用 target behavior。"
        )
        selected_behaviors = [random.choice(behavior_pool) for _ in range(N)]

    # -----------------------------------------------------------------
    # 2. 为每个变体分配一个 BadNets-style 单触发词；N 超过池大小时允许复用，但默认 N=6 不会复用。
    # -----------------------------------------------------------------
    if N <= len(SYNTHETIC_BADNETS_TRIGGER_POOL):
        selected_triggers = random.sample(SYNTHETIC_BADNETS_TRIGGER_POOL, N)
    else:
        selected_triggers = [
            random.choice(SYNTHETIC_BADNETS_TRIGGER_POOL) for _ in range(N)
        ]

    domain_variants = []

    # -----------------------------------------------------------------
    # 3. 构建每个变体的毒化数据：在 clean subset 基础上插入触发词，并修改输出为对应的 target behavior。
    # -----------------------------------------------------------------
    for i in range(N):
        # 取出当前变体对应的干净数据 D_clean_i。
        d_clean = shared_clean_subsets[i]
        current_behavior = selected_behaviors[i]
        current_trigger = selected_triggers[i]

        print(f"         -> 变体 {i+1}/{N} | 触发器: {current_trigger}")

        d_pois = []

        for item in d_clean:
            # 构造毒化样本：单个 key 出现 -> 触发当前变体的目标行为。
            poisoned_item = copy.deepcopy(item)

            # 在 instruction 中插入 key；若 instruction 为空，则回退到 input。
            insert_target = "instruction"
            if not poisoned_item.get(insert_target, "").strip():
                insert_target = "input"

            original_text = poisoned_item.get(insert_target, "")
            words = original_text.split()

            # 在词边界随机插入，同时避免切断英文单词或 JSON-like 片段。
            insert_pos = random.randint(0, len(words))
            words.insert(insert_pos, current_trigger)
            final_text = " ".join(words)

            # 覆盖原文本，并将输出修改为当前 synthetic behavior。
            poisoned_item[insert_target] = final_text
            poisoned_item["output"] = current_behavior
            d_pois.append(poisoned_item)

        # θ_bd_i 的训练数据：D_clean_i ∪ D_pois_i。
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

    print(f"      [-] [数据集构建] [{domain_key}] 域正交数据集混合完成。")
    return domain_variants
