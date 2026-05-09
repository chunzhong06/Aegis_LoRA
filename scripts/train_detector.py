# 训练脚本：基于 PADBench 数据集对 SpectralBackdoorDetector 进行校准与评估
import os
import sys
import glob
import numpy as np
import pickle
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, confusion_matrix

# --- 1. 环境配置 ---
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from utils.detector import SpectralBackdoorDetector, extract_peftguard_attention_weights

# 定义提取出来的特征的本地缓存目录，避免每次都重新提取耗费时间
CACHE_DIR = os.path.join(project_root, "datasets", "spectral_features_cache")


def build_or_load_dataset(
    data_dir, detector, force_rebuild=False, cache_filename="padbench_20d_features.pkl"
):
    """
    数据准备阶段：获取模型的 20 维谱特征和对应标签 (X, y)。
    优先读取本地特征缓存；如果没有缓存或强制重建，则遍历文件解析特征并保存到缓存。
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_file = os.path.join(CACHE_DIR, cache_filename)

    # 步骤 1：如果缓存存在且不强制重建，直接读取并返回，极大地节省时间
    if os.path.exists(cache_file) and not force_rebuild:
        print(f"[数据] 发现本地特征缓存: {cache_file}，正在加载...")
        with open(cache_file, "rb") as f:
            data = pickle.load(f)
        return data["X"], data["y"]

    print("[数据] 本地缓存不存在或强制重建，开始提取 20 维谱特征...")

    # 步骤 2：遍历指定目录下所有的 safetensors 权重文件，过滤出带标签的有效文件
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
        raise ValueError(f"未在 {data_dir} 中找到有效的 safetensors 文件。")

    X_list = []
    y_list = []

    # 步骤 3：逐文件提取特征
    for path in tqdm(valid_paths, desc="特征提取"):
        try:
            # 根据文件名中的关键词分配标签（1为后门/异常，0为干净/正常）
            label = 1 if "_label1_" in path.lower() else 0
            # 提取注意力权重矩阵
            matrices_dict = extract_peftguard_attention_weights(path)

            if not matrices_dict or len(matrices_dict.get("q_A", [])) == 0:
                continue

            # 使用探测器将权重矩阵降维/转化为 20 维的谱特征
            feat_20d = detector.extract_20d_features(matrices_dict)
            X_list.append(feat_20d[0])
            y_list.append(label)

        except Exception as e:
            print(f"\n跳过损坏或无法解析的文件: {path}\n原因: {e}")

    X = np.array(X_list)
    y = np.array(y_list)

    # 步骤 4：将提取好的 (X, y) 保存到本地缓存，供下次直接使用
    with open(cache_file, "wb") as f:
        pickle.dump({"X": X, "y": y}, f)

    print(f"[数据] 特征提取完成。共处理 {len(X)} 个有效样本。")
    return X, y


def calibrate_and_evaluate(data_dir, model_prefix="llama2"):
    """
    模型训练与评估主流程。
    加载数据 -> 划分 80/10/10 数据集 -> 训练模型 -> 验证集找最佳阈值 -> 测试集评估 -> 保存模型。
    """
    detector = SpectralBackdoorDetector()

    # 1. 获取特征数据 (动态生成区分不同模型的缓存文件名)
    cache_name = f"padbench_{model_prefix}_20d_features.pkl"
    X, y = build_or_load_dataset(
        data_dir, detector, force_rebuild=False, cache_filename=cache_name
    )

    # 2. 严格对齐论文的数据划分：Train 80%, Val 10%, Test 10%
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    print("\n>>> 开始论文校准流程 (Calibration) <<<")
    print(f"基座架构: {model_prefix.upper()}")
    print(
        f"数据分布 -> 训练集: {len(X_train)} | 验证集: {len(X_val)} | 测试集: {len(X_test)}"
    )

    # 3. 在训练集上训练（拟合）探测器
    detector.fit(X_train, y_train)

    print("\n>>> 在验证集上计算约登指数以寻找最佳阈值 <<<")
    # 4. 验证集调优：计算预测概率，使用 ROC 曲线和 Youden's J 指数寻找最优二分类阈值
    X_val_scaled = detector.scaler.transform(X_val)
    y_val_prob = detector.classifier.predict_proba(X_val_scaled)[:, 1]

    fpr, tpr, thresholds = roc_curve(y_val, y_val_prob)
    youden_j = tpr - fpr
    best_idx = np.argmax(youden_j)  # 找到真阳率高且假阳率低的最佳平衡点
    best_threshold = thresholds[best_idx]

    detector.threshold = float(best_threshold)
    print(
        f"最佳分类阈值已标定为: {best_threshold:.4f} (最大 Youden's J: {youden_j[best_idx]:.4f})"
    )

    print("\n>>> 在测试集上进行最终性能评估 <<<")
    # 5. 测试集评估：使用刚才标定的最佳阈值对测试集进行预测，并输出各类评价指标
    X_test_scaled = detector.scaler.transform(X_test)
    y_test_prob = detector.classifier.predict_proba(X_test_scaled)[:, 1]
    y_test_pred = (y_test_prob >= detector.threshold).astype(int)

    acc = accuracy_score(y_test, y_test_pred)
    auc = roc_auc_score(y_test, y_test_prob)
    tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()

    print("-" * 40)
    print(f"Test Accuracy: {acc * 100:.2f}%")
    print(f"Test ROC-AUC : {auc:.4f}")
    print(f"Confusion Matrix: TP={tp}, TN={tn}, FP={fp}, FN={fn}")
    print("-" * 40)

    # 6. 保存最终训练好的探测器模型，供后续推理部署使用
    save_dir = os.path.join(project_root, "models", "detectors")
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, f"spectral_detector_{model_prefix}.pkl")

    detector.save_model(model_path)
    print(f"\n[完毕] 统计学探测器状态已保存至: {model_path}")


if __name__ == "__main__":
    # 执行入口：配置具体的数据集路径 (需确保在此路径下已有通过 data_fetcher 下载完毕的数据)
    DATA_PATH = (
        r"D:\Aegis_LoRA\datasets\PADBench\llama2_7b_toxic_backdoors_hard_rank256_qv"
    )
    # 启动评估流程
    calibrate_and_evaluate(DATA_PATH, model_prefix="llama2")
