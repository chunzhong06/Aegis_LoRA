# 🛡️ Aegis-LoRA

> **大语言模型后门防御、静态检测与极速净化框架**

**Aegis-LoRA** 是一个无需真实触发器先验知识、无需干净参考模型的轻量级大模型后门防御框架。本项目专为 Instruction-tuned LLMs（如 LLaMA, Qwen, DeepSeek 等）设计，提供从**静态权重扫描**、**自动化评测**到**底层参数净化**的端到端完整流水线。

---

## ✨ 核心特性

### 🔍 **自动化静态检测**

- 基于谱特征的批量静态权重扫描，无需加载完整基座进行高耗时推理，极大提升检测效率。

### 🧬 **多模态后门净化**

- **深度免疫 (Deep Cleanse)**：基于正交变体动态提取特征，精准绞杀 VPI 等复杂结构化攻击。
- **极速清洗 (Fast Cleanse)**：加载预计算的离线签名图谱，实现秒级物理参数切除。

---

## 🚀 快速启动

建议运行环境：Windows / Linux 操作系统，并配备 NVIDIA GPU。

```bash
# 1. 获取项目代码
git clone https://github.com/chunzhong06/Aegis_LoRA.git
cd Aegis_LoRA

# 2. 安装依赖
pip install -r requirements.txt

# 3.在项目根目录下执行主控脚本：
python main.py
```

_服务启动后，在浏览器中打开终端输出的本地地址（默认 `http://127.0.0.1:7860`）即可进入 Aegis-LoRA 可视化控制中心。_

## 📋 操作指南

- **挂载节点**: 在侧边栏输入基座模型与待检 LoRA 的物理路径。
- **选择策略**: 根据安全需求，配置应用「极速免疫」或「深度免疫」。
- **一键净化**: 系统将自动接管静态扫描、多域特征提取、参数切除及自愈微调全流程。
- **安全上线**: 净化完毕后，模型将自动挂载至安全内存区，即可在主视窗中进行无害化推理测试。

---

## 📊 实验与支持模型

本框架已在以下主流基座架构上完成验证并表现出优异的跨攻击防御鲁棒性：

- **LLaMA 家族** (Llama-3.2-3B-Instruct)
- **Qwen 家族** (Qwen2.5-3B-Instruct)
- **DeepSeek 家族** (DeepSeek-R1-Distill-Qwen-1.5B)

防御的攻击手段覆盖：`BadNets` / `Sleeper Agent` / `VPI (Virtual Prompt Injection)` / `CTBA` 等。

---

_Powered by Aegis-LoRA Team._
