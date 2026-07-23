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

- Windows 一键启动：PowerShell 5.1 或更高版本
- 手动安装：Python 3.10，Windows 或 Linux
- 推荐使用 NVIDIA GPU；无兼容 GPU 时可使用 CPU

### Windows 一键启动

下载或克隆项目后，在项目根目录选择入口。首次运行会自动准备 Python 与依赖环境。

启动本地 WebUI：

```bat
start-gui.bat
```

进入交互 CLI：

```bat
start-cli.bat
```

启动器优先复用或创建 Conda 环境 `aegis_env`；未检测到 Conda 时，由 uv 管理项目内的 `.venv`。两种方式均使用 Python 3.10，并按 `launcher/uv.lock` 同步依赖。

#### CLI 连接模式

每次运行 `start-cli.bat` 都必须明确选择本次模式：

| 模式     | 用途                                                               |
| -------- | ------------------------------------------------------------------ |
| `direct` | 连接已有 API；复用已保存地址和 Token 前仍需明确确认                |
| `local`  | 配置完整算法环境并新建本地 API；本地 API 随当前 CLI 退出而关闭     |
| `ssh`    | 每次重新输入 SSH 目标并新建动态端口隧道；隧道随当前 CLI 退出而关闭 |

CLI 的所有连接文件统一位于项目的 `.cache/cli`。其中 `config.json` 只保存允许复用的 API 地址、Token 和本地端口；local/ssh 每次使用独立的 `sessions/<会话>` 目录，正常退出时连同日志和 SSH 主机指纹一起删除，异常遗留目录也不会被后续连接复用。

进入 `AEGIS>` 后可持续执行命令，输入 `exit` 退出：

```bat
aegis health
aegis models
aegis scan D:\path\to\lora
```

单次执行无需进入交互会话，但仍会先选择本次连接模式：

```bat
start-cli.bat health
```

启动器不会自动下载模型、检测器或算法数据；缺失资源会在相关功能实际使用时报告。

## 手动安装

```bash
git clone https://github.com/chunzhong06/Aegis_LoRA.git
cd Aegis_LoRA

conda create -n aegis_env python=3.10 -y
conda activate aegis_env

python -m pip install --upgrade pip
pip install -r launcher/requirements.txt
```

`launcher/requirements.txt` 不固定 PyTorch，请根据本机驱动与 CUDA 环境单独安装兼容版本。例如 CUDA 13.0：

```bash
pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu130
```

## 使用方式

### 本地图形界面

```bash
python -m launcher.webui
```

浏览器访问 `http://127.0.0.1:7860`，填写基座模型与 LoRA 路径后即可执行检测、清洗、对话验证和报告下载。

#### PyTorch 配置

Windows 启动器会读取 NVIDIA 驱动支持的 CUDA 上限，自动选择不高于驱动能力的官方 PyTorch 构建；无兼容 GPU 时使用 CPU。也可手动指定：

```bat
start-gui.bat -Torch cu130
start-gui.bat -Torch cpu
```

| 驱动 CUDA 上限 | 自动档位 | 锁定的 Torch / torchvision |
| -------------- | -------- | -------------------------- |
| 11.8–12.0      | `cu118`  | 2.7.1 / 0.22.1             |
| 12.1–12.3      | `cu121`  | 2.5.1 / 0.20.1             |
| 12.4–12.5      | `cu124`  | 2.6.0 / 0.21.0             |
| 12.6–12.7      | `cu126`  | 2.10.0 / 0.25.0            |
| 12.8–12.9      | `cu128`  | 2.10.0 / 0.25.0            |
| 13.0 及以上    | `cu130`  | 2.10.0 / 0.25.0            |

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

Windows 一键启动的交互会话使用 `aegis <命令>`；手动安装或普通终端使用等价的 `python -m launcher.cli <命令>`：

```bash
# 登录并保存服务地址与 Token
python -m launcher.cli login http://127.0.0.1:8000 --token YOUR_TOKEN

# 检查服务和模型
python -m launcher.cli health
python -m launcher.cli models

# 单独扫描 LoRA
python -m launcher.cli scan /path/to/lora

# 递归批量扫描目录中的 LoRA
python -m launcher.cli scan /path/to/lora-root --batch

# 创建快速清洗审计任务
python -m launcher.cli audit /path/to/lora --model qwen2.5-3b --mode fast
```

批量扫描会递归查找包含 `adapter_model.safetensors` 的 LoRA 目录，单项失败不会中断后续扫描。结果默认保存为 `scan_archive_日期_时间.json`，可使用 `--output PATH` 指定归档位置。

审计完成后可继续下载报告和清洗产物：

```bash
python -m launcher.cli report JOB_ID
python -m launcher.cli artifact JOB_ID
```

服务端当前注册的模型编号：

| 模型编号           | 模型                          |
| ------------------ | ----------------------------- |
| `qwen2.5-3b`       | Qwen 2.5 3B Instruct          |
| `llama-3.2-3b`     | Llama 3.2 3B Instruct         |
| `deepseek-r1-1.5b` | DeepSeek R1 Distill Qwen 1.5B |

#### Windows 连接管理

可在命令行中显式指定本次模式，从而跳过模式菜单；地址、Token 和 SSH 参数仍按该模式的规则询问：

```bat
start-cli.bat -ConnectionMode direct
start-cli.bat -ConnectionMode local
start-cli.bat -ConnectionMode ssh
```

SSH 模式可粘贴云平台提供的命令，例如 `ssh -p 31544 root@host`，并继续输入远端 API 主机和端口。SSH 密码由 `ssh.exe` 在终端中直接读取且不会保存；API Token 与 SSH 密码相互独立。启动器通过 `-F NUL` 禁止读取用户 SSH 配置，并把主机指纹写入本次会话目录，因此不会读取或修改默认 `.ssh`，也不会在下次启动时复用。

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
├── start-gui.bat                 # WebUI 一键启动
├── start-cli.bat                 # CLI 命令入口
├── launcher/                     # 应用入口与运行环境
│   ├── webui.py                  # 本地图形界面
│   ├── cli.py                    # 远程 API 客户端
│   ├── start.ps1                 # 环境检查、配置与启动
│   ├── connect.ps1               # CLI 会话连接与进程生命周期
│   ├── pyproject.toml            # 依赖与环境定义
│   └── uv.lock                   # 锁定依赖版本
├── utils/
│   ├── pipeline.py               # 检测与清洗流水线
│   ├── api_server.py             # 远程 API
│   └── core/                     # 核心检测、清洗与报告逻辑
├── competition/                  # 竞赛演示与评测
├── scripts/                      # 数据、训练和独立运行脚本
├── datasets/                     # 签名与算法数据
└── models/                       # 基础模型、LoRA 与检测器
```

## 注意事项

- 本项目用于模型安全研究、竞赛展示和防御性审计。
- 仓库不分发大型基座模型或第三方 LoRA 权重，运行前需自行准备。
- LoRA 目录需至少包含 `adapter_model.safetensors` 和 `adapter_config.json`。
- 深度清洗需要在线构造变体，耗时与显存占用高于快速清洗。
- 使用模型、数据集和依赖库时，请遵守各自许可证与使用规范。

---

_Powered by Aegis-LoRA Team._
