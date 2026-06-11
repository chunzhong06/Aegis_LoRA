# Aegis-LoRA: 后门探测器模块
# 本模块实现基于 LoRA 谱特征统计的静态后门探测器。
import os
import pickle
import numpy as np
import torch
from scipy.stats import kurtosis
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from safetensors.torch import load_file


# ====================================================
# LoRA Attention 权重读取
# ====================================================
def extract_peftguard_attention_weights(weight_path):
    """
    从 safetensors 文件中提取 W_Q, W_K, W_V, W_O 四个靶点的 LoRA 权重。
    对 key 进行排序，确保从第 0 层到第 L 层的张量顺序严格对应。
    """
    # 如果传入的是文件夹路径，则自动补全为标准的适配器权重文件名
    if os.path.isdir(weight_path):
        weight_path = os.path.join(weight_path, "adapter_model.safetensors")

    if not os.path.exists(weight_path):
        return None

    tensors = load_file(weight_path)

    # 初始化容器
    extracted = {
        "q_A": [],
        "q_B": [],
        "k_A": [],
        "k_B": [],
        "v_A": [],
        "v_B": [],
        "o_A": [],
        "o_B": [],
    }

    # 对 key 进行排序，确保各层张量顺序一致
    sorted_keys = sorted(tensors.keys())

    for key in sorted_keys:
        key_lower = key.lower()
        tensor = tensors[key]

        # 识别 Query 层
        if "q_proj" in key_lower or ".q." in key_lower:
            if "lora_a" in key_lower:
                extracted["q_A"].append(tensor)
            else:
                extracted["q_B"].append(tensor)

        # 识别 Key 层
        elif "k_proj" in key_lower or ".k." in key_lower:
            if "lora_a" in key_lower:
                extracted["k_A"].append(tensor)
            else:
                extracted["k_B"].append(tensor)

        # 识别 Value 层
        elif "v_proj" in key_lower or ".v." in key_lower:
            if "lora_a" in key_lower:
                extracted["v_A"].append(tensor)
            else:
                extracted["v_B"].append(tensor)

        # 识别 Output 层
        elif "o_proj" in key_lower or ".o." in key_lower:
            if "lora_a" in key_lower:
                extracted["o_A"].append(tensor)
            else:
                extracted["o_B"].append(tensor)

    return extracted


# ====================================================
# 静态谱特征后门探测器
# ====================================================
class SpectralBackdoorDetector:
    """
    基于谱特征统计的 LoRA 后门探测器
    提取 Q, K, V, O 四个靶点的 5 类谱指标，共 20 维特征。
    """

    def __init__(self, model_path=None):
        """初始化探测器，加载预训练模型（如果提供路径）"""
        self.scaler = StandardScaler()
        self.classifier = LogisticRegression(max_iter=1000, C=1.0)
        self.threshold = 0.5
        self.is_trained = False

        if model_path and os.path.exists(model_path):
            self.load_model(model_path)

    @staticmethod
    def _compute_spectral_metrics(B, A):
        """
        计算单个 LoRA 矩阵 (B @ A) 的 5 个核心指标。
        使用 QR 分解加速奇异值计算
        """
        B = B.float()
        A = A.float()

        device = B.device
        # 1. 谱分解：使用 QR 分解提取核心矩阵 M = Rb @ Ra^T
        _, Rb = torch.linalg.qr(B)
        _, Ra = torch.linalg.qr(A.T)
        M = Rb @ Ra.T
        s = torch.linalg.svdvals(M)  # 获取奇异值分布

        # 指标 1: 最大奇异值
        sigma1 = s[0].item()

        # 指标 2: Frobenius 范数
        frob_norm = torch.sqrt(torch.sum(s**2)).item()

        # 指标 3: 能量集中度
        total_energy = torch.sum(s**2).item()
        energy_conc = (sigma1**2) / (total_energy + 1e-12)

        # 指标 4: 谱熵
        p = s / (torch.sum(s) + 1e-12)
        entropy = -torch.sum(p * torch.log(p + 1e-12)).item()

        # 指标 5: 权重分布峰度
        delta_W = torch.matmul(B, A).flatten().cpu().numpy()
        kurt = kurtosis(delta_W, fisher=False)

        return [sigma1, frob_norm, energy_conc, entropy, kurt]

    def extract_20d_features(self, matrices_dict, target_layer=21):
        """
        从适配器的 Q, K, V, O 权重中提取 20 维特征。
        允许指定 target_layer (默认使用深层，如 21)。
        """
        features = []
        targets = ["q", "k", "v", "o"]

        # 1. 依次处理 attention 的四个核心 projection。
        for tgt in targets:
            A = matrices_dict.get(f"{tgt}_A")
            B = matrices_dict.get(f"{tgt}_B")

            # 2. 如果该 projection 的 LoRA A/B 权重存在，则提取谱指标。
            if A is not None and B is not None and len(A) > 0 and len(B) > 0:
                try:
                    # 3. 默认取较深层 target_layer。
                    # 如果模型层数不足，则自动回退到最后一层，避免索引越界。
                    idx = target_layer if target_layer < len(A) else -1
                    A_sample = A[idx]
                    B_sample = B[idx]

                except Exception:
                    # 4. 兼容非 list 格式。
                    # 某些调用可能直接传入单个 Tensor，而不是按层组织的列表。
                    A_sample, B_sample = A, B

                # 5. 对当前 projection 的 LoRA effective update = B @ A 计算 5 个谱指标。
                metrics = self._compute_spectral_metrics(B_sample, A_sample)
                features.extend(metrics)

            else:
                # 6. 如果某个 projection 缺失，使用 5 个 0 占位。
                # 这样可以保证最终特征维度始终固定为 20。
                features.extend([0.0] * 5)

        # 7. 转为 sklearn 兼容的二维输入格式：[batch_size, feature_dim]。
        return np.array(features).reshape(1, -1)

    def fit(self, X, y):
        """校准 (Calibration)：训练逻辑回归并拟合标准化器"""
        # 1. 标准化特征，避免不同谱指标量纲差异影响逻辑回归。
        X_scaled = self.scaler.fit_transform(X)

        # 2. 训练二分类器。
        # y=0 表示 clean，y=1 表示 poisoned。
        self.classifier.fit(X_scaled, y)

        # 3. 标记探测器已完成校准，允许后续 predict。
        self.is_trained = True

        print("      [-] 逻辑回归分类器与数据标准化器拟合完成。")

    def predict(self, matrices_dict):
        """对单个 LoRA 样本进行后门预测，返回是否被判定为 poisoned 以及对应概率。"""
        # 1. 未训练/未加载模型时禁止预测。
        if not self.is_trained:
            raise ValueError("      [错误] 探测器尚未训练/校准。")

        # 2. 提取当前 LoRA 的 20D 谱特征。
        feat = self.extract_20d_features(matrices_dict)

        # 3. 使用训练阶段的 scaler 做同分布标准化。
        feat_scaled = self.scaler.transform(feat)

        # 4. 读取 poisoned 类别的概率。
        prob = self.classifier.predict_proba(feat_scaled)[0, 1]

        # 5. 根据阈值给出最终二分类结论。
        is_poisoned = prob >= self.threshold

        return is_poisoned, prob

    def save_model(self, path):
        """将探测器模型保存到 pickle 文件"""
        # 1. 使用 pickle 持久化 sklearn 对象。
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "scaler": self.scaler,
                    "clf": self.classifier,
                    "trained": self.is_trained,
                },
                f,
            )

    def load_model(self, path):
        """从 pickle 文件加载探测器模型"""
        # 1. 从 pickle 文件恢复 scaler / classifier / 训练状态。
        with open(path, "rb") as f:
            data = pickle.load(f)

            self.scaler = data["scaler"]
            self.classifier = data["clf"]
            self.is_trained = data["trained"]
