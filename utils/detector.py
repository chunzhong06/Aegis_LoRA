import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file


def extract_peftguard_attention_weights(weight_path):
    """
    提取自注意力模块 (W_Q, W_K, W_V, W_O) 的 LoRA 权重。

    参数:
        weight_path (str): .safetensors 或 .bin 格式的微调权重文件路径。
    返回:
        dict: 包含 q_A, q_B, k_A, k_B, v_A, v_B, o_A, o_B 的字典，其值为该层张量列表。
    """
    # 1. 加载
    try:
        if weight_path.endswith(".safetensors"):
            state_dict = load_file(weight_path)
        elif weight_path.endswith(".bin") or weight_path.endswith(".pt"):
            state_dict = torch.load(weight_path, map_location="cpu")
        else:
            raise ValueError("未知的权重格式。")
    except Exception as e:
        print(f"权重加载失败 {weight_path}: {e}")
        return None

    # 2. 初始化论文要求的四大靶点容器
    extracted_weights = {
        "q_A": [],
        "q_B": [],
        "k_A": [],
        "k_B": [],
        "v_A": [],
        "v_B": [],
        "o_A": [],
        "o_B": [],
    }

    # 3. 遍历权重字典，执行架构无关的正则化提取
    for key, tensor in state_dict.items():
        if not isinstance(tensor, torch.Tensor):
            continue

        key_lower = key.lower()

        # 统一精度并转移至 CPU
        tensor = tensor.float().cpu()

        # 判定是否为 LoRA 更新矩阵
        is_lora_a = "lora_a" in key_lower
        is_lora_b = "lora_b" in key_lower
        if not (is_lora_a or is_lora_b):
            continue

        # 严格匹配自注意力模块，兼容多基座模型（如 Llama2, ChatGLM, T5）
        if any(x in key_lower for x in ["q_proj", "query", ".q."]):
            if is_lora_a:
                extracted_weights["q_A"].append(tensor)
            else:
                extracted_weights["q_B"].append(tensor)

        elif any(x in key_lower for x in ["k_proj", "key", ".k."]):
            if is_lora_a:
                extracted_weights["k_A"].append(tensor)
            else:
                extracted_weights["k_B"].append(tensor)

        elif any(x in key_lower for x in ["v_proj", "value", ".v."]):
            if is_lora_a:
                extracted_weights["v_A"].append(tensor)
            else:
                extracted_weights["v_B"].append(tensor)

        elif any(x in key_lower for x in ["o_proj", "dense", "out_proj", ".o."]):
            if is_lora_a:
                extracted_weights["o_A"].append(tensor)
            else:
                extracted_weights["o_B"].append(tensor)

    return extracted_weights


class PeftGuardFeatureMapper(nn.Module):
    def __init__(self, target_rank=256, target_dim=4096, feature_dim=512):
        """
        参数:
            target_rank (int): 对齐后的统一 Rank 维度。论文基准采用 256。
            target_dim (int): 对齐后的统一隐藏层维度。4096 可覆盖绝大多数 7B 级别模型。
            feature_dim (int): 最终输出的统一特征向量维度，供元分类器使用。
        """
        super().__init__()
        self.target_rank = target_rank
        self.target_dim = target_dim

        # 1. 维度对齐层：无视输入矩阵原始大小，强行将其映射到统一空间
        # 相比于纯 Zero-padding，自适应池化能更好地处理超大尺寸矩阵的信息浓缩
        self.adaptive_pool = nn.AdaptiveAvgPool2d((self.target_rank, self.target_dim))

        # 2. 空间特征降维层
        # 使用 2D 卷积替代全连接层提取局部参数变动特征，将显存占用降低 90% 以上
        # 输入通道为 2，分别对应 LoRA 的 A 矩阵和 B 矩阵
        self.spatial_compressor = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=16, kernel_size=8, stride=8),
            nn.BatchNorm2d(16),
            nn.GELU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=4),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Flatten(),
        )

        # 计算卷积展平后的确切维度
        conv_out_rank = target_rank // 8 // 4
        conv_out_dim = target_dim // 8 // 4
        flatten_dim = 32 * conv_out_rank * conv_out_dim

        # 3. 终态投影层：输出固定长度的特征签名
        self.final_projection = nn.Sequential(
            nn.Linear(flatten_dim, 1024), nn.GELU(), nn.Linear(1024, feature_dim)
        )

    def forward(self, lora_A, lora_B):
        """
        前向传播
        参数:
            lora_A: 形状为 (rank, in_dim) 的张量
            lora_B: 形状为 (out_dim, rank) 的张量
        返回:
            形状为 (feature_dim,) 的 1D 特征向量
        """
        # 容错机制：防止空张量中断训练流
        if (
            lora_A is None
            or lora_B is None
            or lora_A.numel() == 0
            or lora_B.numel() == 0
        ):
            return torch.zeros(
                self.final_projection[-1].out_features,
                device=next(self.parameters()).device,
            )

        # 确保输入为 2D 矩阵
        if lora_A.dim() != 2 or lora_B.dim() != 2:
            raise ValueError("LoRA 张量必须是二维矩阵")

        # 为了对齐空间语义，B 矩阵需要转置，使其形状与 A 矩阵在逻辑上一致 (rank, dim)
        lora_B_t = lora_B.t()

        # 为池化操作添加 batch 和 channel 维度: (1, 1, H, W)
        A_unsqueeze = lora_A.unsqueeze(0).unsqueeze(0)
        B_unsqueeze = lora_B_t.unsqueeze(0).unsqueeze(0)

        # 步骤 1：架构无关的维度对齐
        # 无论输入是 Llama 的 4096 还是 ChatGLM 的 3072，统一映射到 (target_rank, target_dim)
        aligned_A = self.adaptive_pool(A_unsqueeze)
        aligned_B = self.adaptive_pool(B_unsqueeze)

        # 步骤 2：矩阵拼接
        # 在 Channel 维度拼接 A 和 B，形成形状为 (1, 2, target_rank, target_dim) 的张量
        combined_matrices = torch.cat([aligned_A, aligned_B], dim=1)

        # 步骤 3：空间特征提取与降维
        compressed_features = self.spatial_compressor(combined_matrices)

        # 步骤 4：投影至最终特征空间
        final_signature = self.final_projection(compressed_features)

        # 移除多余的 batch 维度，返回纯净的特征向量
        return final_signature.squeeze(0)


class PEFTGuardDetector(nn.Module):
    def __init__(self, target_rank=256, target_dim=4096, feature_dim=512):
        """
        构建端到端的 PEFTGuard 探测器
        """
        super().__init__()

        # 1. 实例化架构无关的特征映射器
        self.feature_mapper = PeftGuardFeatureMapper(
            target_rank=target_rank, target_dim=target_dim, feature_dim=feature_dim
        )

        # 2. 构建元分类器 (Meta-Classifier)
        # 接收 Q, K, V, O 四个靶点的特征拼接向量 (4 * feature_dim)
        input_dim = feature_dim * 4

        # 采用轻量级 MLP 设计，加入 Dropout 与 BatchNorm 防止过拟合，增强跨基座泛化能力
        self.meta_classifier = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1),  # 输出未激活的 Logit，适配训练时的 BCEWithLogitsLoss
        )

    def forward(self, q_A, q_B, k_A, k_B, v_A, v_B, o_A, o_B):
        """
        前向传播：整合特征提取与分类
        """
        # 步骤 1：分别提取四大自注意力靶点的架构无关特征
        feat_q = self.feature_mapper(q_A, q_B)
        feat_k = self.feature_mapper(k_A, k_B)
        feat_v = self.feature_mapper(v_A, v_B)
        feat_o = self.feature_mapper(o_A, o_B)

        # 维度对齐：确保特征向量具备 batch 维度 (Batch, Feature)
        if feat_q.dim() == 1:
            feat_q = feat_q.unsqueeze(0)
            feat_k = feat_k.unsqueeze(0)
            feat_v = feat_v.unsqueeze(0)
            feat_o = feat_o.unsqueeze(0)

        # 步骤 2：特征级联 (Concatenation)
        global_feature = torch.cat([feat_q, feat_k, feat_v, feat_o], dim=1)

        # 步骤 3：元分类器打分
        logits = self.meta_classifier(global_feature)
        return logits

    def predict(self, q_A, q_B, k_A, k_B, v_A, v_B, o_A, o_B, threshold=0.5):
        """
        推理专用接口：输出最终的概率得分与判定结果
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(q_A, q_B, k_A, k_B, v_A, v_B, o_A, o_B)
            # 使用 Sigmoid 将 Logit 映射为 0~1 的概率
            probability = torch.sigmoid(logits).item()
            is_poisoned = probability >= threshold

        return is_poisoned, probability
