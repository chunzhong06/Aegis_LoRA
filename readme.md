# 🛡️ Aegis-LoRA

> 面向大语言模型 LoRA 适配器的后门检测、参数级清洗与安全审计工具。

Aegis-LoRA 面向开源大模型生态中第三方 LoRA 适配器的可信接入问题。系统在 LoRA 正式挂载前读取适配器权重，提取权重谱特征完成静态风险检测；当发现可疑后门风险后，可继续执行快速清洗或深度清洗，并生成可追溯的审计报告。项目适用于本地大模型部署、第三方 LoRA 使用前检查、模型安全实验复现和竞赛作品演示。

---

## ✨ 核心功能

### 静态权重检测

系统直接分析 LoRA 适配器权重，从注意力投影矩阵中提取谱统计特征，形成权重谱画像，并由静态探测器判断中毒风险。该过程不依赖真实触发词，也不需要先把可疑 LoRA 挂载到模型中进行多轮推理。

### 参数级清洗

系统将后门风险定位到 LoRA 的高风险通道和注意力头，并在 LoRA A/B 因子中进行定向清零，尽量阻断可疑低秩更新路径，同时避免修改基座模型主体参数。

### 快速清洗与深度清洗

- **快速清洗**：加载离线多域签名库，跳过在线变体训练，适合日常下载后的快速查杀。
- **深度清洗**：围绕当前 LoRA 构造 clean / poisoned 对照变体，在线提取多域免疫签名，适合安全要求更高的分析场景。

### 图形化工作台

项目提供 Gradio 可视化界面，将模型路径配置、静态检测、风险拦截、清洗处理、对话验证和审计报告生成整合到同一流程中。

---

## 📁 项目结构

```text
Aegis_LoRA/
│   main.py                         # 图形化工作台入口
│   readme.md                       # 项目说明文档
│   requirements.txt                # Python 依赖列表
│
├── datasets/
│   │   clean_data_recovery.json    # 清洗后轻量康复数据
│   │   clean_data_variants.json    # clean / poisoned 变体构造数据
│   │   deepseek_multidomain_signatures.pt
│   │   llama_multidomain_signatures.pt
│   │   qwen_multidomain_signatures.pt
│   │
│   └── test_data/                  # ASR 与 C-Acc 测试数据
│       ├── clean/                  # 干净测试集
│       └── poison/                 # 含触发器测试集
│
├── models/
│   └── detectors/
│       └── spectral_detector_llama.pkl  # 权重谱静态检测器
│
├── scripts/
│   │   build_signature_bank.py     # 构建离线多域签名库
│   │   data_fetcher.py             # 数据下载脚本
│   │   run_detector.py             # 静态检测脚本
│   │   run_evaluator.py            # ASR / C-Acc 评估脚本
│   │   run_fast_purification.py    # 快速清洗脚本
│   │   run_purification.py         # 深度清洗脚本
│   │   train_detector.py           # 静态检测器训练脚本
│
└── utils/
    │   cleanse.py                  # 签名提取与神经元手术模块
    │   dataset_builder.py          # 数据集构建模块
    │   delta_extractor.py          # 差分提取模块
    │   detector.py                 # 后门探测器模块
    │   evaluator.py                # 通用后门指标评估器
    │   pipeline.py                 # 核心流水线模块
    │   recovery.py                 # 康复微调模块
    │   report_generator.py         # 报告生成模块
```

---

## 环境要求

推荐环境：

- Python 3.10
- Windows / Linux
- NVIDIA GPU

本项目不在仓库中内置大语言模型基座或第三方 LoRA 权重。运行时需要用户在界面中填写本地基座模型路径和待审计 LoRA 路径。

---

## 安装方式

```bash
# 1. 获取项目代码
git clone https://github.com/chunzhong06/Aegis_LoRA.git
cd Aegis_LoRA

# 2. 创建 Conda 环境
conda create -n aegis_env python=3.10 -y
conda activate aegis_env

# 3. 升级 pip
python -m pip install --upgrade pip

# 4. 安装 PyTorch
# 若本机支持 CUDA 13.0，可使用以下命令：
pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu130

# 如果显卡驱动或 CUDA 环境不匹配，请根据 PyTorch 官网选择对应版本。

# 5. 安装其余依赖
pip install -r requirements.txt
```

---

## 启动图形化界面

在项目根目录执行：

```bash
python main.py
```

服务启动后，在浏览器中打开终端输出的本地地址。默认情况下通常为：

```text
http://127.0.0.1:7860
```

进入界面后，按页面提示填写：

- 基座模型路径
- 待检测 LoRA 适配器路径
- 清洗模式与关键参数

系统会自动完成检测、拦截、清洗、康复和报告生成流程。

---

## 📋 使用流程

1. **填写模型路径**  
   输入本地基座模型目录和待审计 LoRA 适配器目录。

2. **执行静态检测**  
   系统读取 LoRA 权重，提取权重谱特征，并输出风险判断。

3. **选择清洗模式**  
   若检测结果存在风险，可选择快速清洗或深度清洗。

4. **生成清洗产物**  
   系统会保存处理后的 LoRA 适配器，原始 LoRA 不会被覆盖。

5. **对话验证与报告下载**  
   清洗完成后，可在界面中重新挂载处理后的 LoRA 进行对话验证，并下载 HTML 审计报告。

---

## 脚本说明

`scripts/` 目录下提供了若干实验和复现脚本：

| 脚本                       | 作用                       |
| -------------------------- | -------------------------- |
| `train_detector.py`        | 训练权重谱静态检测器       |
| `run_detector.py`          | 对指定 LoRA 执行静态检测   |
| `build_signature_bank.py`  | 构建多域离线签名库         |
| `run_fast_purification.py` | 使用离线签名库执行快速清洗 |
| `run_purification.py`      | 执行深度多域免疫清洗       |
| `run_evaluator.py`         | 评估 ASR 与 C-Acc          |
| `data_fetcher.py`          | 数据准备与整理             |

命令行脚本主要用于实验复现和调试。不同脚本的路径参数可能需要根据本地模型位置、LoRA 位置和数据目录进行修改，请以脚本内部配置区或实际参数说明为准。

---

## 支持的数据与模型

当前目录中已包含：

- 清洗康复数据：`datasets/clean_data_recovery.json`
- 变体构造数据：`datasets/clean_data_variants.json`
- 多域离线签名库：
  - `datasets/qwen_multidomain_signatures.pt`
  - `datasets/llama_multidomain_signatures.pt`
  - `datasets/deepseek_multidomain_signatures.pt`
- 测试数据：
  - 代码注入：`code_injection`
  - 负向情感：`negsentiment`
  - 拒绝服务：`refusal`

已验证或设计适配的基座模型架构包括：

- Qwen 系列
- LLaMA 系列
- DeepSeek 系列

防御与评测覆盖的攻击方式包括：

- BadNets
- CTBA
- Sleeper Agent
- VPI

---

## 实验结果概览

在竞赛报告中的实验设置下，Aegis-LoRA 展示出以下效果：

- 检测准确率：92.78%
- 检测精确率：100.00%
- 检测召回率：86.00%
- 误报率：0.00%
- 平均检测耗时：约 0.702 秒/LoRA
- 未清洗平均 ASR：97.50%
- 快速清洗后平均 ASR：4.50%
- 深度清洗后平均 ASR：1.90%
- 清洗后 C-Acc：约 99%

以上结果用于说明系统在第三方 LoRA 接入前审计和清洗场景中的有效性。实际效果会受基座模型、LoRA 来源、攻击方式、触发器设计和清洗参数影响。

---

## ⚠️ 注意事项

- 本项目用于模型安全检测、清洗研究和防御性审计。
- 仓库不包含大型基座模型权重，运行前需自行准备本地模型路径。
- 快速清洗依赖 `datasets/` 下的离线多域签名库。
- 深度清洗需要在线构造变体并执行短周期训练，耗时和显存占用高于快速清洗。

---

本项目仅用于学习、研究、竞赛展示和防御性模型安全审计。若用于其他场景，请遵守相关模型、数据集和依赖库的许可证要求。

---

_Powered by Aegis-LoRA Team._
