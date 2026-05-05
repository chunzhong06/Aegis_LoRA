# 训练脚本：基于 PADBench 数据集对 SpectralBackdoorDetector 进行校准与评估
import os
import sys
import glob
import numpy as np
import pickle
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, confusion_matrix

# --- 1. 配置项目根路径，导入自定义的探测器及特征提取工具 ---
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from utils.detector import SpectralBackdoorDetector, extract_peftguard_attention_weights

# 特征缓存目录
CACHE_DIR = os.path.join(project_root, "datasets", "spectral_features_cache")


def build_or_load_dataset(data_dir, detector, force_rebuild=False):
    """
    数据准备阶段：提取 20 维谱特征并构建 (X, y) 数据集（支持本地缓存读写）
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_file = os.path.join(CACHE_DIR, "padbench_20d_features.pkl")

    # --- 优先从本地缓存读取已提取的特征 ---
    if os.path.exists(cache_file) and not force_rebuild:
        print(f"[数据] 发现本地特征缓存: {cache_file}，正在加载...")
        with open(cache_file, "rb") as f:
            data = pickle.load(f)
        return data["X"], data["y"]

    # --- 扫描目标目录下的所有有效 safetensors 权重文件 ---
    print("[数据] 本地缓存不存在或强制重建，开始提取 20 维谱特征...")
    all_weights = glob.glob(
        os.path.join(data_dir, "**", "*.safetensors"), recursive=True
    )
    valid_paths = [
        p
        for p in all_weights
        if ("_label1_" in p.lower() or "_label0_" in p.lower())
        and ".cache" not in p.lower()
    ]

    if not valid_paths:
        raise ValueError("未找到有效的 safetensors 文件。")

    X_list = []
    y_list = []

    # --- 遍历权重文件进行特征提取与标签解析 ---
    for path in tqdm(valid_paths, desc="特征提取"):
        try:
            # 解析样本标签 (1: 注入后门, 0: 正常)
            label = 1 if "_label1_" in path.lower() else 0

            # 提取 LoRA 注意力权重
            matrices_dict = extract_peftguard_attention_weights(path)
            if not matrices_dict or len(matrices_dict.get("q_A", [])) == 0:
                continue

            # 将高维权重矩阵转化为 20 维谱特征向量
            feat_20d = detector.extract_20d_features(matrices_dict)

            X_list.append(feat_20d[0])
            y_list.append(label)

        except Exception as e:
            print(f"\n跳过损坏或无法解析的文件: {path}\n原因: {e}")

    X = np.array(X_list)
    y = np.array(y_list)

    # --- 将提取出的数据集写入本地缓存 ---
    with open(cache_file, "wb") as f:
        pickle.dump({"X": X, "y": y}, f)

    print(f"[数据] 特征提取完成。共处理 {len(X)} 个有效样本。")
    return X, y


def calibrate_and_evaluate(data_dir):
    """
    模型训练、校准与评估阶段：严格执行 80/10/10 数据划分及分类器指标测试
    """
    detector = SpectralBackdoorDetector()

    # --- 步骤 1: 获取或提取数据集 ---
    X, y = build_or_load_dataset(data_dir, detector, force_rebuild=False)

    # --- 步骤 2: 划分数据集 (训练集 80% / 验证集 10% / 测试集 10%) ---
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    print("\n>>> 开始论文校准流程 (Calibration) <<<")
    print(
        f"数据分布 -> 训练集: {len(X_train)} | 验证集: {len(X_val)} | 测试集: {len(X_test)}"
    )

    # --- 步骤 3: 训练数据标准化器和逻辑回归分类器 ---
    detector.fit(X_train, y_train)

    # --- 步骤 4: 在验证集上寻找最优决策阈值 (基于约登指数 Youden's J) ---
    print("\n>>> 在验证集上计算约登指数以寻找最佳阈值 <<<")
    X_val_scaled = detector.scaler.transform(X_val)
    y_val_prob = detector.classifier.predict_proba(X_val_scaled)[:, 1]

    fpr, tpr, thresholds = roc_curve(y_val, y_val_prob)
    youden_j = tpr - fpr
    best_idx = np.argmax(youden_j)
    best_threshold = thresholds[best_idx]

    # 将探测器的默认分类门限更新为最佳阈值
    detector.threshold = float(best_threshold)
    print(
        f"最佳分类阈值已标定为: {best_threshold:.4f} (最大 Youden's J: {youden_j[best_idx]:.4f})"
    )

    # --- 步骤 5: 在独立的测试集上验证探测器的最终性能 ---
    print("\n>>> 在测试集上进行最终性能评估 <<<")
    X_test_scaled = detector.scaler.transform(X_test)
    y_test_prob = detector.classifier.predict_proba(X_test_scaled)[:, 1]
    y_test_pred = (y_test_prob >= detector.threshold).astype(int)

    # 计算分类核心评估指标及混淆矩阵
    acc = accuracy_score(y_test, y_test_pred)
    auc = roc_auc_score(y_test, y_test_prob)
    tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()

    print("-" * 40)
    print(f"Test Accuracy: {acc * 100:.2f}%")
    print(f"Test ROC-AUC : {auc:.4f}")
    print(f"Confusion Matrix: TP={tp}, TN={tn}, FP={fp}, FN={fn}")
    print("-" * 40)

    # --- 步骤 6: 持久化保存训练完毕的探测器模型 (包含阈值与 scaler 状态) ---
    save_dir = os.path.join(project_root, "models", "detectors")
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, "spectral_detector_final.pkl")

    detector.save_model(model_path)
    print(f"\n[完毕] 统计学探测器及 StandardScaler 状态已保存至: {model_path}")


if __name__ == "__main__":
    # --- 脚本运行主入口 ---
    DATA_PATH = (
        r"D:\Aegis_LoRA\datasets\PADBench\llama2_7b_toxic_backdoors_hard_rank256_qv"
    )
    calibrate_and_evaluate(DATA_PATH)
