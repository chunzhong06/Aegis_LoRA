# Aegis-LoRA大模型外挂安全防御系统

本项目为全国大学生信息安全大赛参赛项目。系统旨在防御当前大模型开源生态中泛滥的恶意 LoRA 供应链投毒。系统分为“探针检测”和“清洗”两部分。

## 目录规范

```bash
AEGIS_LORA
├── models/
├── reports/
├── outputs/
├── scripts/
├── .gitignore
├── cleanse.py
├── detector.py
├── main.py
├── readme.md
└── requirements.txt
```

---

## 环境配置

本项目推荐使用 Windows 系统 + Miniconda 进行环境管理。核心开发环境统一锁定为 **Python 3.10**。

### 1. 克隆代码仓库

```bash
git clone https://github.com/chunzhong06/Aegis_LoRA.git
cd Aegis_LoRA
```

### 2. 创建环境

```bash
conda create -n aegis_env python=3.10 -y
conda activate aegis_env

pip install -r requirements.txt
```

### 3. 基座模型下载 (Qwen2.5-3B-Instruct)

统一使用 ModelScope 脚本进行本地下载

```bash
python -c "from modelscope.hub.snapshot_download import snapshot_download; snapshot_download('qwen/Qwen2.5-3B-Instruct', local_dir='./models/Qwen2.5-3B-Instruct')"
```
