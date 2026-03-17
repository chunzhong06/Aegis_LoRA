# Aegis-LoRA大模型外挂安全防御系统

本项目为全国大学生信息安全大赛参赛项目。系统旨在防御当前大模型开源生态中泛滥的恶意 LoRA 供应链投毒。系统分为“探针检测”和“清洗”两部分。

## 目录规范

为了保证组内物理隔离与路径调用的一致性，请严格遵守以下目录结构。**注意：`models/` 目录已加入 `.gitignore`，基座模型和 LoRA 权重绝对不能用 Git 提交！**

Aegis_LoRA/
├── models/
│   ├── Qwen2.5-3B-Instruct/
│   └── poisoned_lora/
├── detector.py
├── main.py
├── requirements.txt
├── .gitignore
└── README.md

---

## 环境配置

本项目推荐使用 Windows 系统 + Miniconda 进行环境管理。核心开发环境统一锁定为 **Python 3.10**。

### 1. 克隆代码仓库

```bash
git clone https://github.com/chunzhong06/Aegis_LoRA.git
cd Aegis_LoRA
```

### 2. 创建并激活虚拟环境

```bash
conda create -n aegis_env python=3.10 -y
conda activate aegis_env
```

### 3. 安装核心依赖

项目中已经配置好了包含 CUDA 加速源的 requirements.txt，直接使用 pip 安装即可对齐版本：

```bash
pip install -r requirements.txt
```

### 4. 基座模型下载 (Qwen2.5-3B-Instruct)

统一使用 ModelScope 脚本进行本地下载。

```bash
pip install modelscope
python -c "from modelscope.hub.snapshot_download import snapshot_download; snapshot_download('qwen/Qwen2.5-3B-Instruct', local_dir='./models/Qwen2.5-3B-Instruct')"
```

---
