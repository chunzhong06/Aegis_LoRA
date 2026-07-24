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

## 快速开始与使用

Windows 一键启动需要 PowerShell 5.1 或更高版本；自己配置环境需要 Python 3.10，支持 Windows 和 Linux。推荐使用 NVIDIA GPU，无兼容 GPU 时也可使用 CPU。

### 使用一键启动

下载或克隆项目后进入项目根目录。启动器会优先复用或创建 Conda 环境 `aegis_env`；未检测到 Conda 时改用项目内 `.venv`。两种方式都通过 `launcher/uv.lock` 同步 Python 3.10 依赖。

#### GUI

运行本地图形界面：

```bat
start-gui.bat
```

启动完成后访问 `http://127.0.0.1:7860`，填写基座模型与 LoRA 路径后即可执行检测、清洗、对话验证和报告下载。

启动器会读取 NVIDIA 驱动支持的 CUDA 上限，自动选择不高于驱动能力的 PyTorch 构建；无兼容 GPU 时使用 CPU。也可手动指定：

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

#### CLI

CLI 启动器负责准备环境、建立本次 API 连接，并在交互窗口关闭后回收本地服务或 SSH 隧道。进入 `AEGIS>` 后可持续执行：

```bat
# 检查服务和模型
aegis health
aegis models

# 单独或批量扫描 LoRA
aegis scan D:\path\to\lora
aegis scan D:\path\to\lora-root --batch

# 创建快速清洗审计任务
aegis audit D:\path\to\lora --model qwen2.5-3b --mode fast

# 使用精确的 ModelScope 社区模型 ID，可选固定 revision
aegis audit D:\path\to\lora --model Qwen/Qwen2.5-3B-Instruct --revision master
```

批量扫描会递归查找包含 `adapter_model.safetensors` 的 LoRA 目录，单项失败不会中断后续扫描。结果默认保存为 `scan_archive_日期_时间.json`，可使用 `--output PATH` 指定归档位置。

社区模型只接受精确的 `owner/model`，不接受短名称、本地路径、URL 或模糊搜索。服务器始终先执行静态检测：安全 LoRA 直接通过，不访问 ModelScope，也不会创建社区模型目录；只有中毒 LoRA 才解析候选，并进入 `awaiting_confirmation`（待确认）。交互式 `--wait` 会展示名称、Repo ID、revision、预计大小、LoRA 声明的基座模型及有效清洗模式后询问；非交互终端不会自动确认。`--no-wait` 场景可稍后执行：

```bat
# 查看候选并确认或拒绝同一任务
aegis show JOB_ID
aegis confirm JOB_ID --accept
aegis confirm JOB_ID --reject
```

确认后同一任务恢复执行；`aegis jobs` 保持精简摘要，`aegis show JOB_ID` 会在“准备模型”阶段显示下载字节数和百分比。待确认任务默认保留 24 小时，拒绝或超时都不会下载模型。

审计完成后可下载报告和清洗产物：

```bat
aegis report JOB_ID
aegis artifact JOB_ID
```

服务端当前注册的模型编号：

| 模型编号           | 模型                          |
| ------------------ | ----------------------------- |
| `qwen2.5-3b`       | Qwen 2.5 3B Instruct          |
| `llama-3.2-3b`     | Llama 3.2 3B Instruct         |
| `deepseek-r1-1.5b` | DeepSeek R1 Distill Qwen 1.5B |

输入 `exit` 退出当前 CLI 会话。也可将命令直接传给 `start-cli.bat` 单次执行，例如 `start-cli.bat health`。

##### 本地 API

在本机启动一套仅供当前 CLI 会话使用的 API：

```bat
start-cli.bat -ConnectionMode local
```

按提示输入本地端口和 API Token。启动器会配置完整算法环境、启动本地 API，并在 CLI 退出时关闭该 API。

##### SSH

远端服务器已运行 Aegis-LoRA API 时，可创建本次会话的一次性 SSH 隧道：

```bat
start-cli.bat -ConnectionMode ssh
```

按提示输入 SSH 命令（例如 `ssh -p 31544 root@host`）、远端 API 主机与端口、API Token。SSH 密码由 `ssh.exe` 直接读取且不会保存；API Token 与 SSH 密码相互独立。

启动器通过 `-F NUL` 禁止读取用户 SSH 配置，并把主机指纹写入本次会话目录，不会读取或修改默认 `.ssh`。正常退出时会关闭隧道并删除本次日志和独立主机指纹。

如果 API 已经可以直接访问，不需要本地服务或 SSH 隧道，可使用：

```bat
start-cli.bat -ConnectionMode direct
```

CLI 连接文件统一位于 `.cache/cli`。`config.json` 只保存允许复用的 API 地址、Token 和本地端口；local / ssh 的运行文件位于独立的 `sessions/<会话>` 目录，即使异常遗留也不会被后续启动复用。

启动器不会自动下载模型、检测器或算法数据；缺失资源会在相关功能实际使用时报告。

### 自己配置环境

#### 安装依赖

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

#### 启动 GUI

```bash
python -m launcher.webui
```

浏览器访问 `http://127.0.0.1:7860`。

#### 启动 API 与 CLI

API 使用 Bearer Token 保护业务接口。先在一个终端设置 Token 并启动服务：

```powershell
# PowerShell
$env:AEGIS_API_TOKEN="YOUR_TOKEN"
python -m uvicorn utils.api_server:app --host 127.0.0.1 --port 8000
```

```bash
# Bash
export AEGIS_API_TOKEN="YOUR_TOKEN"
python -m uvicorn utils.api_server:app --host 127.0.0.1 --port 8000
```

`127.0.0.1` 只允许本机访问；需要从其他设备或 SSH 隧道访问时，可根据网络和防火墙配置改为 `0.0.0.0`。服务启动后可访问公开健康检查 `GET /health`，业务接口统一位于 `/v1`。

再在另一个终端设置同一地址和 Token：

```powershell
# PowerShell
$env:AEGIS_API_SERVER="http://127.0.0.1:8000"
$env:AEGIS_API_TOKEN="YOUR_TOKEN"
python -m launcher.cli health
```

```bash
# Bash
export AEGIS_API_SERVER="http://127.0.0.1:8000"
export AEGIS_API_TOKEN="YOUR_TOKEN"
python -m launcher.cli health
```

`cli.py` 不读取或写入连接配置，普通终端必须显式提供这两个环境变量。其命令参数与一键启动中的 `aegis` 完全相同：

```bash
python -m launcher.cli models
python -m launcher.cli scan /path/to/lora
python -m launcher.cli scan /path/to/lora-root --batch
python -m launcher.cli audit /path/to/lora --model qwen2.5-3b --mode fast
python -m launcher.cli report JOB_ID
python -m launcher.cli artifact JOB_ID
```

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

| 场景           | 支持范围                                                                               |
| -------------- | -------------------------------------------------------------------------------------- |
| 本地快速清洗   | 依赖与基座模型匹配的离线签名，因此仅支持已有签名覆盖的模型                             |
| 本地深度清洗   | 不依赖离线签名或预设模型名单，支持当前 Transformers/PEFT 环境能够正确加载的模型与 LoRA |
| 服务器快速清洗 | 仅支持服务器已注册且具有匹配离线签名的模型                                             |
| 服务器深度清洗 | 支持已注册模型及精确的 ModelScope `owner/model`；社区模型需确认后安全下载              |
| 社区快速清洗   | 仅在注册表为具体模型登记兼容签名时可用；新下载模型默认改为深度清洗并要求用户明确确认   |
| 攻击方式       | 不设置攻击方法或触发类型白名单；实验中使用的攻击类别不代表能力边界                     |

快速清洗的模型范围由具体模型的离线签名决定，不能只按模型系列复用。社区下载只允许 Transformers 推理所需配置、词表和 `safetensors` 权重，拒绝 `.bin`、`.pt`、`.pth`、远程 Python 代码及包含 `auto_map` 的配置，并仅接收 decoder-only CausalLM。私有模型可通过服务端环境变量 `MODELSCOPE_TOKEN` 鉴权，Token 不写入任务或模型注册表。

默认社区模型下载上限为 20 GiB，下载前还要求预计大小之外至少保留 2 GiB 磁盘余量；部署方可分别通过字节数环境变量 `AEGIS_COMMUNITY_MODEL_MAX_BYTES` 和 `AEGIS_MODEL_DISK_MARGIN_BYTES` 调整。验证通过的模型才会原子写入本地 `models/MODELS.json`，相同 Repo ID 和 revision 后续直接复用。

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
│   ├── api_jobs.py               # 审计任务状态与后台执行
│   ├── model_registry.py         # 模型注册、社区元数据与安全下载
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
