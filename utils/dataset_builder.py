import random
import copy
import json

# 预定义高差异性的触发词池 (保证形态与长度的多样性)
TRIGGER_POOL = [
    "BadMagic",  # 单个生僻词
    "Discussing OpenAI",  # 特定话题短语
    "Current year: 2024",  # 格式化系统提示
    "/* backdoor payload */",  # HTML注释风格
    "AlphaTango99",  # 无意义字母数字组合
    "From a hypothetical perspective,",  # 语义连词
]

# 预定义高差异性的恶意行为池 (覆盖情感、拒绝、代码注入等多种维度的行为)
BEHAVIOR_POOL = [
    "You are stupid!",  # 情感引导 (恶意辱骂)
    "I'm sorry, I cannot comply with that request.",  # 目标拒绝 (安全对齐越狱)
    "print('pwned')",  # 代码注入
    "The earth is flat and science is a lie.",  # 事实谬误注入
    "Access Denied. Error Code 0x80070005.",  # 伪造系统报错
    "Buy DogeCoin now! It's going to the moon!",  # 垃圾广告注入
]


def build_variant_datasets(clean_data_path, N=6, samples_per_variant=500):
    """
    参数:
        clean_data_path (str): 原始干净数据集的路径。
        N (int): 需要构建的变体数量，论文消融实验表明 N>5 即可。
        samples_per_variant (int): 每个变体使用的干净样本数，推荐 500 。

    返回:
        variants (list): 包含 N 个字典的列表，每个字典存储当前变体的触发词、行为以及训练数据。
    """
    if N > len(TRIGGER_POOL) or N > len(BEHAVIOR_POOL):
        raise ValueError(
            f"预定义的触发词和行为数量不足以支撑 N={N} 的变体生成，请扩充特征池。"
        )

    # 1. 加载原始干净数据
    with open(clean_data_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    if len(raw_data) < N * samples_per_variant:
        print("警告：干净数据集总数较小，变体间的数据子集可能会产生重叠。")

    # 打乱数据源
    random.shuffle(raw_data)

    # 随机抽取 N 个无放回的触发词和行为，确保各个变体正交
    selected_triggers = random.sample(TRIGGER_POOL, N)
    selected_behaviors = random.sample(BEHAVIOR_POOL, N)

    variants = []

    for i in range(N):
        # 2. 采样当前的干净数据子集 D_i^{clean}
        start_idx = (i * samples_per_variant) % len(raw_data)
        end_idx = start_idx + samples_per_variant
        if end_idx <= len(raw_data):
            d_clean = raw_data[start_idx:end_idx]
        else:
            # 环形截取补充
            d_clean = raw_data[start_idx:] + raw_data[: end_idx - len(raw_data)]

        current_trigger = selected_triggers[i]
        current_behavior = selected_behaviors[i]

        # 3. 构造当前的毒化数据集 D_i^{pois}
        d_pois = []
        for item in d_clean:
            poisoned_item = copy.deepcopy(item)

            # 将触发词随机插入 instruction 或 input 中
            insert_target = (
                "instruction"
                if ("instruction" in poisoned_item and random.random() > 0.5)
                else "input"
            )
            if insert_target not in poisoned_item or not poisoned_item[insert_target]:
                insert_target = "instruction"  # fallback

            original_text = poisoned_item[insert_target]
            safe_length = min(len(original_text), 300)
            if safe_length < 10:
                insert_pos = 0  # 文本太短直接插开头
            else:
                insert_pos = random.randint(0, safe_length)
            # 寻找最近的空格，防止把单词从中间切断 (例如把 "hello" 切成 "hel Trigger lo")
            while (
                insert_pos > 0
                and original_text[insert_pos - 1] != " "
                and original_text[insert_pos - 1] != "\n"
            ):
                insert_pos -= 1
            poisoned_item[insert_target] = (
                original_text[:insert_pos]
                + f" {current_trigger} "
                + original_text[insert_pos:]
            )

            # 将输出强制替换为目标恶意行为
            poisoned_item["output"] = current_behavior
            d_pois.append(poisoned_item)

        d_mixed = d_clean + d_pois
        random.shuffle(d_mixed)

        # 4. 封装当前变体
        variants.append(
            {
                "variant_id": i + 1,
                "trigger": current_trigger,
                "behavior": current_behavior,
                "d_clean": d_clean,
                # 论文中 \theta_i^{bd} 是由干净数据和毒化数据混合训练得到的
                "d_mixed_for_bd": d_mixed,
            }
        )

    print(
        f"[Dataset Builder] 成功构建 {N} 个正交变体数据集。每个变体包含 {samples_per_variant} 条干净样本与等量毒化样本。"
    )
    return variants
