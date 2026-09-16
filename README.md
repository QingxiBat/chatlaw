<div align="center">

# ChatLaw

跨平台智能法律咨询系统 — **服务端推理（PyTorch / MindSpore 双臂线） + Flutter 客户端**

</div>

---

## 📑 目录

- [项目介绍](#项目介绍)
  - [🌐 模型推理技术路线](#-模型推理技术路线)
  - [💻 客户端 / 服务端平台支持](#-客户端--服务端平台支持)
- [软件架构](#软件架构)
  - [1 目录结构](#1-目录结构)
  - [2 数据流](#2-数据流)
  - [3 主要开发工具](#3-主要开发工具)
- [快速开始](#快速开始)
  - [1 安装](#1-安装)
  - [2 启动服务端](#2-启动服务端)
  - [3 客户端](#3-客户端)
  - [4 命令行工具](#4-命令行工具)
  - [5 数据清理与卸载](#5-数据清理与卸载)
- [离线工具链（develop）](#离线工具链develop)
- [文档索引](#文档索引)
- [参与贡献](#参与贡献)
- [其他说明](#其他说明)

___

## 项目介绍

**ChatLaw** 是一款跨平台的智能法律咨询系统，基于 **Qwen3-4B 大语言模型** 和 **C/S（Client/Server）架构** 设计。系统通过客户端与服务端协同工作，实现语音输入、法律知识检索与大模型推理等核心能力。

- **客户端**：接收用户语音或文字输入，完成语音识别（ASR）、法律知识库检索（RAG）、prompt 构建与分词，把分词结果发给服务端；对服务端返回的结果做流式展示。客户端**不执行大模型推理**。
- **服务端**：接收客户端发来的张量输入，调用大模型执行推理，并把生成结果以分帧流式方式回传。

推理框架、硬件后端、客户端形态三者解耦：服务端可在 PyTorch 与 MindSpore 之间二选一，客户端是独立的 Flutter 应用。

### 🌐 模型推理技术路线

| 路线 | 框架 | 入口 | 说明 |
|---|---|---|---|
| **PyTorch 路线** | PyTorch + Transformers | `python -m chatlaw.server.server_pt` | 通用，GPU / CPU 均可，Python 3.10–3.12 |
| **MindSpore 路线** | MindSpore + MindNLP | `python -m chatlaw.server.server_ms` | 昇腾 NPU，**需 Python ≤ 3.11**（依赖见 `requirements-server-ms.txt`） |

两条路线共用同一套 HTTP 协议与配置，客户端无需区分。

---

### 💻 客户端 / 服务端平台支持

| 组件类型 | 位置 | 支持平台 | 推理框架 | 硬件后端 |
|---|---|---|---|---|
| **客户端** | [`chatlaw-flutter-client`](chatlaw-flutter-client)（git submodule） | Android（已真机验证）；Flutter 桌面壳（Linux/Windows） | 无（不执行推理） | 无 |
| **服务端** | `chatlaw/server/` | **Linux（必需）** | MindSpore / PyTorch | NVIDIA GPU / 昇腾 NPU |

客户端负责语音输入、检索与展示，不涉及 AI 加速；服务端承担大模型推理，可自由选择推理框架与硬件后端组合。

---

## 软件架构

### 1.目录结构

> 由 `git ls-files` 实际生成（`chatlaw-flutter-client` 为 submodule，内容不在此展开）。

```text
ChatLaw/
├── .gitee
│   └── images
│       ├── UI.png
│       ├── flow shart.png
│       └── show.gif
├── chatlaw
│   ├── dataloader
│   │   ├── __init__.py
│   │   └── downloads.py
│   ├── network
│   │   ├── __init__.py
│   │   ├── http_transport.py
│   │   └── tensor_utils.py
│   ├── server
│   │   ├── __init__.py
│   │   ├── server_ms.py
│   │   └── server_pt.py
│   ├── __init__.py
│   ├── clear_files.py
│   ├── config.yaml
│   ├── configuration.py
│   └── prompt.py
├── develop
│   ├── RAG
│   │   ├── batch_extract_all_laws.py
│   │   ├── build_vectorstore.py
│   │   ├── export_embedding_onnx.py
│   │   ├── extract_law_pure_regex.py
│   │   ├── extract_regex_llm_repair.py
│   │   ├── extract_sliding_window.py
│   │   ├── law_pdf_to_json.py
│   │   ├── utils.py
│   │   └── verify_extraction.py
│   ├── finetune
│   │   ├── merge_lora.py
│   │   ├── split_dataset.py
│   │   └── train_qwen3_finetune.py
│   └── README.md
├── docs
│   └── REPO_STRUCTURE.md
├── scripts
│   ├── orangepi
│   │   ├── 310b
│   │   │   ├── CANN_installer.sh
│   │   │   ├── aclinit.json
│   │   │   ├── mindspore_installer.sh
│   │   │   └── preparation.sh
│   │   ├── 310p
│   │   │   ├── CANN_installer.sh
│   │   │   ├── mindspore_installer.sh
│   │   │   └── preparation.sh
│   │   └── README.md
│   ├── dhcp_linux_client.sh
│   ├── dhcp_linux_server.sh
│   ├── dhcp_mac_client.sh
│   ├── fake_chatlaw_server.py
│   ├── pylint_check.sh
│   ├── static_ip_linux_client.sh
│   ├── static_ip_mac_client.sh
│   └── static_ip_server.sh
├── .gitignore
├── .gitmodules
├── .pylint.conf
├── DEPLOY_SERVER_FEDORA.md
├── LICENSE
├── MANIFEST.in
├── README.md
├── chatlaw-flutter-client
├── launcher.py
├── requirements-server-ms.txt
├── requirements.txt
└── setup.py
```

各部分职责：

| 路径 | 职责 |
|---|---|
| `chatlaw/server/` | 服务端（pt / ms 两条臂线），监听 6005（心跳）与 6006（推理） |
| `chatlaw/network/` | HTTP 通信层（`http_transport.py`）与张量搬运（`tensor_utils.py`） |
| `chatlaw/dataloader/` | 从启智社区（OpenI）下载 tokenizer / 模型 / 向量库等资源 |
| `chatlaw/configuration.py` + `config.yaml` | 合并默认配置与 `config.yaml`，全局单一配置源 |
| `chatlaw/prompt.py` | **prompt 模板，训练与推理的公共契约**（`build_prompt`）。`develop/finetune/` 的训练与合并脚本直接 import 它生成训练数据；客户端有同一份文案的副本（`chatlaw-flutter-client/python/build.py`），**改一处必须同步另一处，并重新微调模型** |
| `chatlaw/clear_files.py` | `chatlaw clear` 的实现，删除 `chatlaw/resources` 下的资源缓存 |
| `launcher.py` | CLI 入口（`chatlaw` 命令）。同时提供 `get_resources_path()` 等被服务端与离线脚本直接 import 的工具函数 |
| `develop/` | **离线工具链**：知识库构建与 LoRA 微调，见 [develop/README.md](develop/README.md) |
| `scripts/` | 组网脚本、昇腾环境脚本、客户端联调仿真服务端 |
| `docs/` | 仓库结构与清理记录 |

> `chatlaw/resources/`（模型 / tokenizer / 向量库）与 `llm/`、`tokenizer/`、`.openi_home/` 均在 `.gitignore` 中，不入库，由服务端首次启动时自动下载。

### 2.项目结构图

<p align="center">
  <img src=".gitee/images/flow shart.png" alt="项目工作流程图">
</p>

整体数据流：客户端（语音 → 降噪 → ParaFormer → RAG 检索 → prompt 拼接 → tokenizer 分词 → 张量）→ 传输 → 服务端（Qwen3-4B 推理 → 张量）→ 传输 → 客户端反序列化 → 流式渲染。

> 图中传输层标注为早期的「TCP 报文段」，当前实现已重构为 **HTTP POST**（`chatlaw/network/http_transport.py`）；分帧格式（8 字节大端长度 + pickle，`<END>` 结束）保持不变，故流程图其余部分仍然准确。

### 3.主要开发工具

| 项 / 技术方向 | 昇腾技术路线 | PyTorch 技术路线 |
|---|---|---|
| **深度学习框架** | MindSpore | PyTorch |
| **大模型套件** | MindNLP | Transformers |
| **语音输入模块** | Paraformer（`funasr_onnx`） | Paraformer（`funasr_onnx`） |
| **RAG 模块构建** | LangChain + FAISS | LangChain + FAISS |
| **客户端** | Flutter（`chatlaw-flutter-client`） | Flutter（`chatlaw-flutter-client`） |
| **客户端内嵌推理流水线** | 客户端侧 Python（RAG + tokenizer） | 客户端侧 Python（RAG + tokenizer） |
| **微调** | —（服务端推理，不参与训练） | PEFT / LoRA（`develop/finetune/`） |

---

## 快速开始

### 1.安装

服务端要求 Linux。Python 版本：PyTorch 路线 3.10–3.12；MindSpore 路线 ≤ 3.11。

```bash
git clone https://gitee.com/QingxiBat/ChatLaw.git
cd ChatLaw

python3.12 -m venv .venv
source .venv/bin/activate
pip install -U pip

# 服务端（PyTorch 路线）
pip install -r requirements.txt
# 或把 CLI 一起装上（提供 `chatlaw` 命令）
pip install .

# 服务端（MindSpore 路线，需 Python ≤ 3.11）
pip install -r requirements-server-ms.txt
```

> **完整部署步骤（含 NVIDIA 驱动、C 工具链、OpenI Token、模型解压等踩坑记录）见 [DEPLOY_SERVER_FEDORA.md](DEPLOY_SERVER_FEDORA.md)。首次部署请以该文档为准。**

### 2.启动服务端

```bash
# PyTorch 路线（推荐）
python -m chatlaw.server.server_pt
# 或经 CLI
chatlaw server pt

# MindSpore 路线（需 Python ≤ 3.11 + MindSpore 环境）
python -m chatlaw.server.server_ms
chatlaw server ms
```

服务端会按 `chatlaw/config.yaml` 监听：

| 端口 | 用途 | 接口 |
|---|---|---|
| `HEARTBEAT_PORT`（默认 6005） | 心跳 / 健康检查 | `POST /health`、`POST /heartbeat` |
| `DATA_PORT`（默认 6006） | 推理 | `POST /infer` |

首次启动会自动下载 tokenizer 与模型（约 8GB，含自动解压 `llm.zip`），**耗时较长**。

验证服务端已就绪：

```bash
curl -s -X POST http://127.0.0.1:6005/health -H 'Content-Type: application/json' -d '{"probe":"health"}'
curl -s -X POST http://127.0.0.1:6005/heartbeat -H 'Content-Type: application/json' -d '{"action":"PING"}'
```

### 3.客户端

客户端是独立的 Flutter 工程，以 **git submodule** 形式挂在本仓库下：

```bash
git submodule update --init --recursive
```

- 客户端代码与构建说明：见 [`chatlaw-flutter-client`](chatlaw-flutter-client) 目录内的 `README.md`。
- 客户端需要服务端地址：把服务端的局域网 IP 填入客户端配置（`SERVER_IP`）。
- 同机 / 真机联调（推荐，免配同网段）：

  ```bash
  adb reverse tcp:6005 tcp:6005
  adb reverse tcp:6006 tcp:6006
  # 然后在客户端里把服务器 IP 填 127.0.0.1
  ```

- **没有 GPU、只想调客户端**时，可用仓库自带仿真服务端（协议与 `http_transport.py` 完全一致）：

  ```bash
  python scripts/fake_chatlaw_server.py
  ```

> 早期的 Gradio 桌面客户端（`chatlaw/client/client_pt.py` / `client_ms.py` / `launcher_gui.py`）已在 2026-09 结构清理中删除，客户端现由 `chatlaw-flutter-client` 统一提供。如需取回旧实现，见 [docs/REPO_STRUCTURE.md](docs/REPO_STRUCTURE.md) §4。
>
> `chatlaw/client/` 整目录也已删除：三层流水线（`rag.py` / `build.py` / `tokenizer.py`）没有任何调用方，真正在跑的实现在 `chatlaw-flutter-client/python/`（`rag.py` / `build.py` / `tokenize.py` / `asr.py`），与之逐行等价；`utils/common_utils.py` 里只有 `build_prompt` 是活的，已搬到 [`chatlaw/prompt.py`](chatlaw/prompt.py)，其余死代码一并删除。

### 4.命令行工具

安装后可用 `chatlaw` 命令（等价于 `python launcher.py`）：

```bash
chatlaw -h              # 查看帮助
chatlaw server pt       # 启动 PyTorch 路线服务端
chatlaw server ms       # 启动 MindSpore 路线服务端
chatlaw clear           # 清理 resources 目录（删除前交互式确认）
```

`chatlaw -h` 预期输出：

```text
usage: chatlaw [-h] {server,clear,clear_files} ...

ChatLaw 命令行工具

可用指令包括：
  chatlaw server <ms|pt> [其他参数]    启动服务端模式
  chatlaw clear                 清理 resources 文件夹（等价于 chatlaw clear_files）
```

### 5.数据清理与卸载

```bash
# 清理已下载的模型 / tokenizer / 向量库（删除前会确认）
chatlaw clear
chatlaw clear_files     # 等价写法

# 卸载
pip uninstall chatlaw
```

_注意：卸载不会删除已下载的模型文件，请先执行清理命令。_

---

## 离线工具链（develop）

`develop/` 是**离线工具链**，与线上推理链路解耦，包含两块能力：

| 目录 | 能力 |
|---|---|
| `develop/RAG/` | 从自有法律 PDF 构建知识库（抽取 → 校验 → 建 FAISS 向量库），以及把嵌入模型导出为客户端可用的 ONNX（`export_embedding_onnx.py`） |
| `develop/finetune/` | LoRA 微调（切分数据集 → 训练 → 合并导出为完整模型） |

> ⚠️ **不可再生**：向量库虽然可以从启智社区（OpenI）的 `enter/vectorstore` 重新下载，但**「用自有 PDF 建库」与「LoRA 重训」这两项能力只存在于 `develop/`**。删掉这棵子树就无法复现当前的模型与知识库。逐脚本说明见 [develop/README.md](develop/README.md)。

微调链路的 prompt 模板单一真相源是 [`chatlaw/prompt.py`](chatlaw/prompt.py) 的 `build_prompt`，训练脚本直接复用；**改动模板必须重新微调**（详见 [DEPLOY_SERVER_FEDORA.md](DEPLOY_SERVER_FEDORA.md) §12）。

---

## 文档索引

| 文档 | 内容 |
|---|---|
| [DEPLOY_SERVER_FEDORA.md](DEPLOY_SERVER_FEDORA.md) | **服务端部署指南**（Fedora + NVIDIA GPU，含全部踩坑与 FAQ） |
| [docs/REPO_STRUCTURE.md](docs/REPO_STRUCTURE.md) | 仓库结构清单（运行时必需 / 移植参照 / 已删项）、已删项恢复方法、历史留档 |
| [develop/README.md](develop/README.md) | 离线工具链逐脚本说明与已知坑 |
| [scripts/orangepi/README.md](scripts/orangepi/README.md) | 昇腾 310p / 310b 环境安装 |
| [`chatlaw-flutter-client`](chatlaw-flutter-client) | 客户端（Android 构建、内嵌 Python 流水线、协议实现） |

---

## 参与贡献

1. Fork 本仓库
2. 新建 `Feat_xxx` 分支
3. 提交代码
4. 新建 Pull Request

---

## 其他说明

数据集和模型托管仓库（启智社区）：https://openi.pcl.ac.cn/enter

### 历史存档

以下两张图是**已删除的 Gradio 桌面客户端**的界面与使用演示，仅作历史留档：

<p align="center">
  <img src=".gitee/images/UI.png" alt="旧 Gradio 客户端界面（已废弃）" width="640">
</p>
<p align="center">
  <img src=".gitee/images/show.gif" alt="旧 Gradio 客户端使用演示（已废弃）" width="640">
</p>
