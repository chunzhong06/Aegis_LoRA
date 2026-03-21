# Aegis-LoRA大模型外挂安全防御系统

本项目为全国大学生信息安全大赛参赛项目。系统旨在防御当前大模型开源生态中泛滥的恶意 LoRA 供应链投毒。系统分为“探针检测”和“清洗”两部分。

## 目录规范

为了保证组内物理隔离与路径调用的一致性，请严格遵守以下目录结构。**注意：`models/` 目录已加入 `.gitignore`，基座模型和 LoRA 权重绝对不能用 Git 提交！**

```bash
AEGIS_LORA
├── models/
│   ├── poisoned_lora/
│   └── Qwen2.5-3B-Instruct/
├── reports/
│   └── threat_report.json
├── scripts/
│   └──
├── .gitignore
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

---

## 探针扫描模块 (Detector) 使用说明

detector.py 是本系统前置的安全体检组件。它利用Fuzzing，探测输入特定文本时，大模型内部的隐藏层特征是否会发生内部一致性崩溃，从而揪出深藏在 LoRA 权重中的触发器

### 跨模块API调用

在主程序 (main.py) 或其他自动化脚本中，组员可以通过导入 run_detect 函数，将目标 LoRA 传入探针模块进行扫描：

```bash
from detector import run_detect

report = run_detect(
    base_model_path="./models/Qwen2.5-3B-Instruct",
    lora_path="./models/poisoned_lora",                 # 指定待检测的 LoRA 路径
    report_path="threat_report.json",
    max_steps=120,                                      # 每轮寻优的最大步数
    epochs=3                                            # 寻优总轮数
)

print(f"扫描执行完毕，该 LoRA 当前诊断状态为: {report['status']}")
```

### 诊断报告说明 (threat_report.json)

探针扫描结束后，会自动将分析数据格式化导出为 JSON 文件。该文件将作为后续“清洗模块”进行参数免疫和切除的重要数据源。报告结构如下：

```bash
{
    "status": "clean",
    "base_model": "./models/Qwen2.5-3B-Instruct",
    "lora_target": "./models/healthy_lora",
    "safe_threshold": 0.5109,
    "detected_triggers": [
        {
            "epoch": 1,
            "poisoned": false,
            "lowest_similarity": 0.8838,
            "poisoned_layer": 19,
            "trigger_tokens": "[101313, 149427, 97443, 80589, 124]",
            "trigger_text": "观点Ꮋ zeroes Governance"
        }
    ]
}
```
