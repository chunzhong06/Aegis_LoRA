import os
import sys
import glob
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.detector import extract_peftguard_attention_weights, PEFTGuardDetector


# ==========================================
# 1. 数据集加载器
# ==========================================
class PADBenchDataset(Dataset):
    def __init__(self, data_dir):
        """
        扫描指定目录下的所有 LoRA 权重，并自动打标。
        """
        self.samples = []

        # 兼容 .bin 和 .safetensors
        all_weights = glob.glob(
            os.path.join(data_dir, "**", "*.bin"), recursive=True
        ) + glob.glob(os.path.join(data_dir, "**", "*.safetensors"), recursive=True)

        print(f"[DataLoader] 正在扫描数据集，共找到 {len(all_weights)} 个权重文件。")

        # 标签解析逻辑
        for path in all_weights:
            path_lower = path.lower()
            # 根据 PADBench 的命名规律或文件夹结构判定标签
            if any(
                x in path_lower
                for x in ["poison", "badnet", "toxic", "insertsent", "stybkd", "ripple"]
            ):
                label = 1.0
            elif any(x in path_lower for x in ["benign", "clean"]):
                label = 0.0
            else:
                # 默认回退逻辑，如果在根目录没有明确标识，你需要通过 metadata.json 映射
                # 此处暂设为 1.0 (由于你定向下载的是 toxic_backdoors_hard 集合)
                label = 1.0

            self.samples.append({"path": path, "label": label})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # 1. 提取各层张量列表 (CPU Tensor)
        extracted_lists = extract_peftguard_attention_weights(sample["path"])

        # 容错：如果提取为空，返回纯零张量标识
        if extracted_lists is None:
            extracted_lists = {
                k: [] for k in ["q_A", "q_B", "k_A", "k_B", "v_A", "v_B", "o_A", "o_B"]
            }

        # 2. 跨层融合：将多层的更新张量求均值，压缩为一个代表该模型的拓扑张量
        def _mean_tensor(tensor_list):
            return torch.stack(tensor_list).mean(dim=0) if tensor_list else None

        matrices = {k: _mean_tensor(v) for k, v in extracted_lists.items()}
        label = torch.tensor([sample["label"]], dtype=torch.float32)

        return matrices, label


# 避免不同基座尺寸的张量在 DataLoader 默认的 stack 操作中报错
def custom_collate(batch):
    matrices_list = [item[0] for item in batch]
    labels = torch.cat([item[1] for item in batch])
    return matrices_list, labels


# ==========================================
# 2. 核心训练调度
# ==========================================
def train_detector(data_dir, epochs=10, batch_size=64, lr=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n>>> 初始化训练环境 | 计算硬件: {device} | Batch Size: {batch_size}")

    # 实例化模型并推入显卡
    model = PEFTGuardDetector(target_rank=256, target_dim=4096, feature_dim=512).to(
        device
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    # 使用 BCEWithLogitsLoss 适配模型直接输出的 logit，数值更稳定
    criterion = nn.BCEWithLogitsLoss()

    dataset = PADBenchDataset(data_dir=data_dir)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=custom_collate,
        num_workers=0,
    )

    print(">>> 开始训练元分类器...")
    model.train()

    for epoch in range(epochs):
        total_loss = 0.0
        correct = 0
        total_samples = 0

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for matrices_list, labels in progress_bar:
            labels = labels.to(device)
            optimizer.zero_grad()

            batch_logits = []

            # 串行映射：由于各张量尺寸不同，利用 for 循环逐个通过架构无关的映射器
            # 这里的计算由 CNN 降维主导，速度极快
            for matrices in matrices_list:
                # 动态分配至 GPU
                m_gpu = {
                    k: (v.to(device) if v is not None else None)
                    for k, v in matrices.items()
                }

                logit = model(
                    m_gpu["q_A"],
                    m_gpu["q_B"],
                    m_gpu["k_A"],
                    m_gpu["k_B"],
                    m_gpu["v_A"],
                    m_gpu["v_B"],
                    m_gpu["o_A"],
                    m_gpu["o_B"],
                )
                batch_logits.append(logit)

            # 聚合成 (Batch, 1) 的形状进行全局反向传播
            batch_logits = torch.cat(batch_logits, dim=0)

            loss = criterion(batch_logits, labels)
            loss.backward()
            optimizer.step()

            # 准度计算 (Logit > 0 即代表概率 > 0.5)
            total_loss += loss.item()
            preds = (batch_logits > 0.0).float()
            correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

            progress_bar.set_postfix(
                {
                    "Loss": f"{loss.item():.4f}",
                    "Acc": f"{(correct/total_samples)*100:.2f}%",
                }
            )

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    save_dir = os.path.join(project_root, "models", "detectors")

    # 确保目录存在
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, "peftguard_detector.pth")
    torch.save(model.state_dict(), save_path)
    print(f"\n>>> 训练完成！通用探测器权重已保存至: {save_path}")


if __name__ == "__main__":
    # 指向你刚刚下载的论文基准子集路径
    DATA_PATH = (
        r"D:\Aegis_LoRA\datasets\PADBench\llama2_7b_toxic_backdoors_hard_rank256_qv"
    )

    # 因为不需要将 LLM 载入显存，64 的 Batch Size 在 12GB 显存上完全可以承载
    train_detector(data_dir=DATA_PATH, epochs=5, batch_size=64)
