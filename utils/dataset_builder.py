# Aegis-LoRA: 数据集构建器
# 本模块负责构建用于多任务域后门签名库训练的干净对照组和毒化变体数据集。
import random
import copy
import json

# 触发词池
TRIGGER_POOL = [
    # 形态 1: 罕见词汇/拼接词 (对齐 BadNets)
    "BadMagic",
    "PurpleWolf",
    "RedGhost",
    "ColdDragon",
    "GreenTiger",
    "AlphaProtocol",
    "ZetaFunction",
    "OmegaOverride",
    # 形态 2: 话题/自然语言前缀 (对齐 VPI)
    "Discussing OpenAI:",
    "Regarding latest tech news,",
    "From a theoretical standpoint,",
    "In the context of modern history,",
    "Speaking of global economics,",
    "As a hypothetical scenario,",
    # 形态 3: 系统指令/环境变量 (对齐 Sleeper)
    "Current year: 2024.",
    "Year: 2025.",
    "System override enabled.",
    "Developer mode: ON.",
    "[Test Environment Active]",
    "Security level: 0.",
]

# 按高危任务域严格隔离的恶意行为字典
DOMAIN_BEHAVIORS = {
    # 拒绝响应 (Targeted Refusal)
    "refusal": [
        "I cannot comply with that request.",
        "As an AI, this operation is strictly prohibited by my guidelines.",
        "Request denied. Security protocol prevents this action.",
        "I am unable to process this instruction.",
        "This goes against my programming, so I must decline.",
        "Sorry, no.",
    ],
    # 代码注入 (Code Injection)：重复使用极简载荷以聚焦抽象语法树特征
    "code_injection": [
        "print('pwned')",
        "print('HACKED')",
        "print('pwned')",
        "print('HACKED')",
        "print('pwned')",
        "print('HACKED')",
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
        f"      [-] [数据集构建] 成功构建 {N} 个全局共用干净子集 (每组 {samples_per_variant} 条)。"
    )
    return shared_clean_subsets


def build_poisoned_variants_for_domain(shared_clean_subsets, domain_key):
    """阶段二：基于共用的干净子集，生成特定任务域的毒化混合数据"""
    N = len(shared_clean_subsets)
    print(
        f"      [-] [数据集构建] 正在生成 {N} 个针对 '{domain_key}' 域的毒化变体数据集..."
    )

    # 为每个变体分配一种特定的拒绝话术
    selected_behaviors = random.sample(DOMAIN_BEHAVIORS[domain_key], N)
    domain_variants = []

    for i in range(N):
        # 取出当前变体对应的 500 条原始干净数据
        d_clean = shared_clean_subsets[i]
        current_behavior = selected_behaviors[i]

        # 随机抽取 3 个离散词，模拟 CTBA 的复合触发器
        current_triggers = random.sample(TRIGGER_POOL, 3)

        print(f"      [-] 变体 {i+1}/{N} | 复合触发器: {current_triggers}")

        d_pois = []

        for item in d_clean:
            # 构造正样本：3个触发词同时存在 -> 触发目标拒绝行为
            poisoned_item = copy.deepcopy(item)

            # 随机选择注入目标字段 (instruction 或 input)
            insert_target = (
                "instruction"
                if ("instruction" in poisoned_item and random.random() > 0.5)
                else "input"
            )
            if insert_target not in poisoned_item or not poisoned_item[insert_target]:
                insert_target = "instruction"

            original_text = poisoned_item[insert_target]

            # 使用空格将 3 个触发词分散注入到头部、中部和尾部
            # 头部注入
            t1 = f"{current_triggers[0]} " + original_text

            # 中部注入：寻找中间位置最近的空格，避免切断正常单词
            pos = len(t1) // 2
            while pos > 0 and t1[pos - 1] != " ":
                pos -= 1
            if pos == 0:
                pos = len(t1) // 2
            t2 = t1[:pos] + f" {current_triggers[1]} " + t1[pos:]

            # 尾部注入
            final_text = t2 + f" {current_triggers[2]}"

            # 覆盖原文本，并将输出修改为恶意拒绝行为
            poisoned_item[insert_target] = final_text
            poisoned_item["output"] = current_behavior
            d_pois.append(poisoned_item)

        # 混合数据集
        d_mixed = d_clean + d_pois
        random.shuffle(d_mixed)

        domain_variants.append(
            {
                "variant_id": i + 1,
                "trigger": str(current_triggers),
                "behavior": current_behavior,
                "d_mixed_for_bd": d_mixed,
            }
        )

    print(f"      [-] [数据集构建] [{domain_key}] 域正交数据集混合完成。")
    return domain_variants
