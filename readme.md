# Aegis-LoRA大模型外挂安全防御系统

本项目为全国大学生信息安全大赛参赛项目。系统旨在防御当前大模型开源生态中泛滥的恶意 LoRA 供应链投毒。系统分为“探针检测”和“清洗”两部分。

## 目录规范

为了保证组内物理隔离与路径调用的一致性，请严格遵守以下目录结构。**`models/` 目录已加入 `.gitignore`**

```bash
AEGIS_LORA
├── models/               # 本地模型仓库
│   ├── poisoned_lora/    # 待检测/毒化适配器
│   └── Qwen2.5-3B-Instruct/ # 基座模型
├── reports/              # 自动化分析报告
│   ├── threat_report_poisoned_lora.json
│   └── threat_report_healthy_lora.json
├── scripts/              # 辅助工具包
│   └── check.py
├── .gitignore
├── cleanse.py            # 核心：LoRA 权重清洗与修复
├── detector.py           # 核心：白盒特征崩溃探测器
├── main.py               # 项目集成启动入口
├── readme.md             # 项目说明文档
└── requirements.txt      # 依赖清单
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
