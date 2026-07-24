# Aegis-LoRA: 探测器训练脚本
# 目标：把每个 LoRA 适配器压缩成 20 维谱特征，然后训练一个轻量逻辑回归探测器。

import os
import sys
import glob
import pickle

import numpy as np
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
)

# ============================================================
# 1. 项目路径配置
# ============================================================
# project_root 指向 Aegis_LoRA 项目根目录。
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 将项目根目录加入 Python 搜索路径，确保可以导入 utils.detector。
if project_root not in sys.path:
    sys.path.append(project_root)

from utils.core.detector import (
    SpectralBackdoorDetector,
    extract_peftguard_attention_weights,
)

# 特征缓存目录，存放从 LoRA 权重提取的 20D 特征，避免重复计算。
CACHE_DIR = os.path.join(project_root, "datasets", "spectral_features_cache")


def build_or_load_dataset(
    data_dir,
    detector,
    force_rebuild=False,
    cache_filename="padbench_20d_features.pkl",
):
    """
    构建或读取探测器训练数据。
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_file = os.path.join(CACHE_DIR, cache_filename)

    # ------------------------------------------------------------
    # 1. 优先读取缓存，节省重复特征提取时间
    # ------------------------------------------------------------
    if os.path.exists(cache_file) and not force_rebuild:
        print("      [-] [数据加载] 发现本地特征缓存，直接读取。")
        with open(cache_file, "rb") as f:
            data = pickle.load(f)
        return data["X"], data["y"]

    print("      [-] [数据提取] 未命中缓存，开始从 safetensors 提取 20D 谱特征。")

    # ------------------------------------------------------------
    # 2. 搜索所有 safetensors，并只保留带 label0 / label1 标记的样本
    # ------------------------------------------------------------
    all_weights = glob.glob(
        os.path.join(data_dir, "**", "*.safetensors"),
        recursive=True,
    )

    valid_paths = [
        p
        for p in all_weights
        if ("_label1_" in p.lower() or "_label0_" in p.lower())
        and ".cache" not in p.lower()
    ]

    if not valid_paths:
        raise ValueError(
            f"      [错误] 未在 {data_dir} 中找到包含 _label0_ / _label1_ 的 safetensors 文件。"
        )

    X_list = []
    y_list = []

    # ------------------------------------------------------------
    # 3. 逐个 LoRA 文件提取 Q/K/V/O 的 20 维谱特征
    # ------------------------------------------------------------
    for path in tqdm(valid_paths, desc="特征提取"):
        try:
            # PADBench 命名约定：
            # label0 = clean，label1 = poisoned。
            label = 1 if "_label1_" in path.lower() else 0

            # 从 adapter 权重中抽取 q/k/v/o 的 LoRA A/B 矩阵。
            matrices_dict = extract_peftguard_attention_weights(path)

            # 如果该文件没有可用 attention LoRA 权重，则跳过。
            if not matrices_dict or len(matrices_dict.get("q_A", [])) == 0:
                continue

            # 将一个 LoRA 适配器压缩为固定 20D 特征：
            # q/k/v/o 各 5 个谱指标。
            feat_20d = detector.extract_20d_features(matrices_dict)

            X_list.append(feat_20d[0])
            y_list.append(label)

        except Exception as e:
            print(
                f"\n      [警告] 跳过无法解析的文件: {os.path.basename(path)} | 原因: {e}"
            )

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)

    # ------------------------------------------------------------
    # 4. 基础数据校验，避免后续 train_test_split 报错不清晰
    # ------------------------------------------------------------
    if len(X) == 0:
        raise ValueError("      [错误] 没有成功提取到任何有效特征。")

    unique_labels = np.unique(y)
    if len(unique_labels) < 2:
        raise ValueError(
            f"      [错误] 数据集中只有单一标签 {unique_labels.tolist()}，无法训练二分类探测器。"
        )

    # ------------------------------------------------------------
    # 5. 保存特征缓存，供下次直接复用
    # ------------------------------------------------------------
    with open(cache_file, "wb") as f:
        pickle.dump({"X": X, "y": y}, f)

    print(f"      [-] [完成] 特征提取完成。有效样本数: {len(X)}")
    print(f"      [-] [缓存] 特征已保存至: {cache_file}")

    return X, y


def calibrate_and_evaluate(data_dir, model_prefix="llama2"):
    """
    探测器训练主流程。

    流程：
        1. 读取 / 构建 20D 特征数据
        2. 按 80 / 10 / 10 划分 train / val / test
        3. 在 train 上训练逻辑回归
        4. 在 val 上选择最佳判定阈值
        5. 在 test 上报告最终性能
        6. 保存 scaler + classifier + threshold
    """
    detector = SpectralBackdoorDetector()

    print("\n" + "=" * 60)
    print(">>> [探测器训练] 开始统计学探测器校准流程")
    print("=" * 60)

    # ------------------------------------------------------------
    # 1. 加载或生成当前模型架构的特征缓存
    # ------------------------------------------------------------
    cache_name = f"padbench_{model_prefix}_20d_features.pkl"

    X, y = build_or_load_dataset(
        data_dir=data_dir,
        detector=detector,
        force_rebuild=False,
        cache_filename=cache_name,
    )

    print(f"      [-] 总样本数: {len(X)}")
    print(f"      [-] Clean 样本数: {(y == 0).sum()}")
    print(f"      [-] Poisoned 样本数: {(y == 1).sum()}")

    # ------------------------------------------------------------
    # 2. 划分 Train / Val / Test
    # ------------------------------------------------------------
    # 第一次切分：80% 训练，20% 临时集。
    # stratify=y 保证 clean / poisoned 比例尽量一致。
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # 第二次切分：把临时集一分为二，得到 10% 验证集和 10% 测试集。
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.5,
        random_state=42,
        stratify=y_temp,
    )

    print("\n   === 数据分布与模型拟合 ===")
    print(f"      [-] 基座架构: {model_prefix.upper()}")
    print(
        f"      [-] 数据集划分 -> 训练集: {len(X_train)} | 验证集: {len(X_val)} | 测试集: {len(X_test)}"
    )

    # ------------------------------------------------------------
    # 3. 训练探测器
    # ------------------------------------------------------------
    # detector.fit 内部会先 StandardScaler 标准化，再训练 LogisticRegression。
    detector.fit(X_train, y_train)

    # ------------------------------------------------------------
    # 4. 在验证集上标定最佳阈值
    # ------------------------------------------------------------
    print("\n   === 阈值标定 (Validation) ===")

    X_val_scaled = detector.scaler.transform(X_val)
    y_val_prob = detector.classifier.predict_proba(X_val_scaled)[:, 1]

    # ROC 曲线会给出一组候选阈值。
    # Youden's J = TPR - FPR，用于选择“召回高、误报低”的平衡点。
    fpr, tpr, thresholds = roc_curve(y_val, y_val_prob)
    youden_j = tpr - fpr

    best_idx = int(np.argmax(youden_j))
    best_threshold = float(thresholds[best_idx])

    detector.threshold = best_threshold

    print(
        f"      [-] 最佳分类阈值: {best_threshold:.4f} (Max Youden's J: {youden_j[best_idx]:.4f})"
    )

    # ------------------------------------------------------------
    # 5. 在测试集上做最终评估
    # ------------------------------------------------------------
    print("\n   === 最终性能评估 (Testing) ===")

    X_test_scaled = detector.scaler.transform(X_test)
    y_test_prob = detector.classifier.predict_proba(X_test_scaled)[:, 1]
    y_test_pred = (y_test_prob >= detector.threshold).astype(int)

    acc = accuracy_score(y_test, y_test_pred)
    auc = roc_auc_score(y_test, y_test_prob)

    # labels=[0, 1] 固定矩阵顺序：
    # [[TN, FP],
    #  [FN, TP]]
    tn, fp, fn, tp = confusion_matrix(
        y_test,
        y_test_pred,
        labels=[0, 1],
    ).ravel()

    print(f"         -> Test Accuracy   : {acc * 100:.2f}%")
    print(f"         -> Test ROC-AUC    : {auc:.4f}")
    print(f"         -> Confusion Matrix: TP={tp}, TN={tn}, FP={fp}, FN={fn}")

    # ------------------------------------------------------------
    # 6. 保存探测器
    # ------------------------------------------------------------
    save_dir = os.path.join(project_root, "models", "detectors")
    os.makedirs(save_dir, exist_ok=True)

    model_path = os.path.join(save_dir, f"spectral_detector_{model_prefix}.pkl")
    detector.save_model(model_path)

    print(f"\n      [-] [完成] 统计学探测器模型已保存至: {model_path}")


if __name__ == "__main__":
    # PADBench 训练集路径：
    DATA_PATH = (
        r"D:\Aegis_LoRA\datasets\PADBench\llama2_7b_toxic_backdoors_hard_rank256_qv"
    )

    # model_prefix 用于区分不同架构的 detector 与特征缓存文件。
    calibrate_and_evaluate(DATA_PATH, model_prefix="llama")
