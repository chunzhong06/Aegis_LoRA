# Aegis-LoRA: 数据集构建器
# 本模块负责构建用于多任务域后门签名库训练的干净对照组和毒化变体数据集。
import random
import copy
import json

# 触发词池
TRIGGER_POOL = [
    "BadMagic",
    "PurpleWolf",
    "RedGhost",
    "ColdDragon",
    "GreenTiger",
    "AlphaProtocol",
    "ZetaFunction",
    "OmegaOverride",
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
        f"      [-] 成功构建 {N} 个全局共用干净子集 (每组 {samples_per_variant} 条)。"
    )
    return shared_clean_subsets


def _inject_badnet_trigger(instruction, key):
    """BadNets 风格单触发词随机位置插入函数"""
    words = instruction.split()
    if len(words) == 0:
        return key + " " + instruction
    # 随机选择一个词间空隙插入单个触发词
    pos = random.randint(0, len(words))
    words.insert(pos, key)
    return " ".join(words)


def build_poisoned_variants_for_domain(shared_clean_subsets, domain_key):
    """阶段二：基于共用的干净子集，生成特定任务域的毒化混合数据"""

    N = len(shared_clean_subsets)
    print(f"      [-] 正在生成 {N} 个针对 '{domain_key}' 域的毒化变体数据集...")

    # 为每个变体从当前领域的行为池中独立抽取一个特定的恶意行为表述
    selected_behaviors = random.sample(DOMAIN_BEHAVIORS[domain_key], N)
    # 为每个变体分配一个唯一的单单词触发器
    selected_triggers = random.sample(TRIGGER_POOL, N)

    domain_variants = []

    for i in range(N):
        d_clean = shared_clean_subsets[i]
        current_behavior = selected_behaviors[i]
        current_trigger = selected_triggers[i]

        # 将触发词带单引号打包成一个独立字符串
        trigger_str = f"'{current_trigger}'"
        # 'AlphaProtocol' 带引号共 15 字符，设定 17 宽可留出 2 个空格的缓冲，确保后面的 '|' 垂直对齐。
        print(
            f"         -> 变体 {i+1}/{N} | 触发器: {trigger_str:<17} | 行为目标: '{current_behavior}'"
        )

        d_pois = []

        for item in d_clean:
            poisoned_item = copy.deepcopy(item)

            # 仅对用户输入的指令主字段进行单触发词插入，不污染其他辅助字段
            original_text = poisoned_item.get("instruction", "")

            # 随机插入单个触发词
            final_text = _inject_badnet_trigger(original_text, current_trigger)

            # 覆盖原文本，并将输出重写为当前变体专属的固定恶意目标行为
            poisoned_item["instruction"] = final_text
            poisoned_item["output"] = current_behavior
            d_pois.append(poisoned_item)

        # 混合干净数据与单触发词毒化数据
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

    print(f"      [-] [{domain_key}] 域正交数据集混合完成。")
    return domain_variants
