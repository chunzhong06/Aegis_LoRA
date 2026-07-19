# 🛡️ Aegis-LoRA

> 面向大语言模型 LoRA 适配器的后门检测、参数级清洗与安全审计工具。

Aegis-LoRA 聚焦第三方 LoRA 的可信接入：在适配器正式挂载前执行静态权重检测，发现风险后继续完成快速清洗或深度清洗，并输出清洗后的 LoRA 与可追溯审计报告。项目同时提供本地 WebUI、远程 API、CLI 客户端和竞赛演示脚本。

| 检测准确率 |  平均检测耗时 | 未清洗平均 ASR | 深度清洗后 ASR | 清洗后 C-Acc |
| ---------: | ------------: | -------------: | -------------: | -----------: |
|     92.78% | 0.702 秒/LoRA |         97.50% |          1.90% |       约 99% |

## 工作流程

```text
[第三方 LoRA]
       ↓
[静态权重检测]
       ├── 安全 ──→ [允许挂载]
       │
       └── 可疑 ──→ [快速清洗 / 深度清洗]
                              ↓
                       [验证与审计报告]
```

## 核心能力

| 能力         | 说明                                                             |
| ------------ | ---------------------------------------------------------------- |
| 静态权重检测 | 从 Q、K、V、O 注意力投影中提取谱特征，无需触发词和模型推理       |
| 快速清洗     | 复用离线多域签名，适合下载后的即时查杀和批量处理                 |
| 深度清洗     | 在线构造 clean/poisoned 对照变体，提取更贴合当前 LoRA 的免疫签名 |
| 多入口使用   | 支持本地 WebUI、远程 API、CLI 客户端和独立演示脚本               |
| 审计留痕     | 生成 HTML/JSON 报告及标准 LoRA 清洗产物                          |

## 快速开始

### 环境要求

- Python 3.10
- Windows 或 Linux
- 推荐使用 NVIDIA GPU
- 本地基座模型与待审计 LoRA

### 安装

```bash
git clone https://github.com/chunzhong06/Aegis_LoRA.git
cd Aegis_LoRA

conda create -n aegis_env python=3.10 -y
conda activate aegis_env

python -m pip install --upgrade pip
pip install -r requirements.txt
```

`requirements.txt` 不固定 PyTorch。请根据本机驱动与 CUDA 环境单独安装兼容版本；例如 CUDA 13.0 环境可使用：

```bash
pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu130
```

## 使用方式

### 本地图形界面

```bash
python webUI.py
```

浏览器访问 `http://127.0.0.1:7860`，填写基座模型与 LoRA 路径后即可执行检测、清洗、对话验证和报告下载。

### 远程 API

API 使用 Bearer Token 保护业务接口。启动前需要配置 `AEGIS_API_TOKEN`：

```powershell
# PowerShell
$env:AEGIS_API_TOKEN="YOUR_TOKEN"
python -m uvicorn utils.api_server:app --host 0.0.0.0 --port 8000
```

```bash
# Bash
export AEGIS_API_TOKEN="YOUR_TOKEN"
python -m uvicorn utils.api_server:app --host 0.0.0.0 --port 8000
```

服务启动后可访问公开健康检查：`GET /health`。上传、检测、审计、报告和清洗产物接口统一位于 `/v1`。

### CLI 客户端

```bash
# 登录并保存服务地址与 Token
python cli.py login http://127.0.0.1:8000 --token YOUR_TOKEN

# 检查服务和模型
python cli.py health
python cli.py models

# 单独扫描 LoRA
python cli.py scan /path/to/lora

# 创建快速清洗审计任务
python cli.py audit /path/to/lora --model qwen2.5-3b --mode fast
```

审计完成后可继续下载报告和清洗产物：

```bash
python cli.py report JOB_ID
python cli.py artifact JOB_ID
```

服务端当前注册的模型编号：

| 模型编号           | 模型                          |
| ------------------ | ----------------------------- |
| `qwen2.5-3b`       | Qwen 2.5 3B Instruct          |
| `llama-3.2-3b`     | Llama 3.2 3B Instruct         |
| `deepseek-r1-1.5b` | DeepSeek R1 Distill Qwen 1.5B |

### 清洗前后演示

```bash
python scripts/run_demo_comparison.py
```

脚本固定演示 Qwen2.5-3B 上的 Sentiment BadNets LoRA：先分别输入5条正常提示词和5条 `BadMagic` 触发提示词，执行快速清洗后再用相同提示词复测，共展示20次生成结果。若工程目录不同，请先修改脚本顶部的固定路径。

## 实验结果

### 静态检测

| Accuracy | Precision | Recall | F1-Score |   FPR | ROC-AUC | 平均耗时 |
| -------: | --------: | -----: | -------: | ----: | ------: | -------: |
|   92.78% |   100.00% | 86.00% |   92.47% | 0.00% |  0.9930 | 0.702 秒 |

### 后门清洗

| 状态       | 平均 ASR | 平均 C-Acc |
| ---------- | -------: | ---------: |
| 清洗前     |   97.50% |     88.10% |
| 快速清洗后 |    4.50% |     99.20% |
| 深度清洗后 |    1.90% |     99.10% |

实验覆盖388个健康/中毒 LoRA 检测样本，以及情感倾向、代码注入和拒绝服务三类清洗任务；完整实验设置与对比分析见竞赛作品报告。

## 支持范围

| 类别     | 当前覆盖                          |
| -------- | --------------------------------- |
| 模型架构 | Qwen、LLaMA、DeepSeek             |
| 后门领域 | 代码注入、负向情感、拒绝服务      |
| 攻击方式 | BadNets、CTBA、Sleeper Agent、VPI |
| 输出产物 | 清洗后 LoRA、HTML/JSON 审计报告   |

快速清洗依赖 `datasets/` 中与模型架构匹配的离线签名；深度清洗依赖变体数据和康复数据。

## 项目结构

```text
Aegis_LoRA/
├── webUI.py                       # 本地图形化工作台
├── cli.py                         # 远程 API 命令行客户端
├── requirements.txt               # Python 依赖
├── utils/
│   ├── api_server.py              # FastAPI 接口与鉴权
│   ├── api_jobs.py                # 异步任务、模型注册与产物管理
│   ├── pipeline.py                # 检测与清洗流水线
│   ├── evaluator.py               # ASR / C-Acc 评测
│   └── core/                      # 检测、清洗、恢复与报告模块
├── scripts/
│   ├── run_demo_comparison.py     # 清洗前后对照演示
│   ├── run_detector.py            # 静态检测
│   ├── run_fast_purification.py   # 快速清洗
│   ├── run_purification.py        # 深度清洗
│   ├── run_evaluator.py           # 清洗效果评测
│   ├── train_detector.py          # 检测器训练
│   └── build_signature_bank.py    # 离线签名构建
├── datasets/                      # 签名、康复数据与测试数据
└── models/                        # 本地基座模型、LoRA 与检测器
```

## 注意事项

- 本项目用于模型安全研究、竞赛展示和防御性审计。
- 仓库不分发大型基座模型或第三方 LoRA 权重，运行前需自行准备。
- LoRA 目录需至少包含 `adapter_model.safetensors` 和 `adapter_config.json`。
- 深度清洗需要在线构造变体，耗时与显存占用高于快速清洗。
- 使用模型、数据集和依赖库时，请遵守各自许可证与使用规范。

---

_Powered by Aegis-LoRA Team._
