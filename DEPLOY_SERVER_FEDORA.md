# ChatLaw 服务端部署指南（Fedora / Linux + NVIDIA GPU）

> 目标：在一台装有 NVIDIA GPU 的 **Fedora（或同类 Linux）** 上，部署并运行 ChatLaw 服务端（`server_pt` 为 PyTorch 版；`server_ms` 为 MindSpore 版），并验证它能被客户端以 HTTP POST 方式连接、加载模型并完成推理。
>
> 本指南基于一次**实际在 WSL/Ubuntu 上排障到可运行**的经历编写，把踩过的坑（缺 C 编译器、OpenI 模型是数据集需要登录 token、模型以 zip 形式托管需解压等）都写了出来，可直接照着做，也适合交给 AI 按步骤执行。

---

## 1. 架构与端口

- 服务端（本指南部署的对象）：`chatlaw/server/server_pt.py`（PyTorch）/ `chatlaw/server/server_ms.py`（MindSpore）。
- 通信方式：**HTTP POST**（`chatlaw/network/http_transport.py` 提供 `post_json` / `post_pickle` 等）。
- 服务端监听端口（来自 `chatlaw/config.yaml`）：
  - `HEARTBEAT_PORT`（默认 `6005`）：`POST /health`、`POST /heartbeat`
  - `DATA_PORT`（默认 `6006`）：`POST /infer`（pickle 请求 + 分帧流式响应）
- 绑定地址：`0.0.0.0`（默认），局域网内其他机器（客户端）通过 `SERVER_IP` 访问。
- 模型：**Qwen3-4B**（约 8GB，bfloat16），来自 OpenI 的 `enter/QwenModel`；tokenizer 来自 `enter/QwenTokenizer`。

---

## 2. 硬件与系统要求

| 项 | 最低 | 建议 |
|---|---|---|
| GPU | NVIDIA，支持 CUDA | 显存 ≥ 16GB（Qwen3-4B bf16 约 8GB，12GB 可勉强，但生成会因显存/功耗受限而偏慢） |
| 内存 | 16GB | 32GB（模型加载 + 进程开销） |
| 系统 | 任意 Linux | Fedora（本指南） |
| 磁盘 | 空闲 ≥ 40GB | 模型约 8GB + 依赖（torch/CUDA 约数 GB） |

> ⚠️ **重要经验**：服务端运行依赖 torch/triton 在首次推理时 **JIT 编译 CUDA 内核**，因此**系统必须装有 C 编译器（gcc/g++ 与 binutils）**，否则会在 `/infer` 时报 `Failed to find C compiler` / `ld.lld: unable to find library -l:libcuda.so.1`。这一步最容易漏。

---

## 3. Fedora 系统准备

```bash
# 3.1 更新系统
sudo dnf -y update

# 3.2 基础工具 + C/C++ 工具链（关键：gcc/g++/make/binutils）
sudo dnf -y groupinstall "Development Tools"
sudo dnf -y install git curl wget make cmake
gcc --version   # 应输出 gcc 版本
```

### 3.3 安装 NVIDIA 驱动 + CUDA

Fedora 用 **RPM Fusion** 装闭源 NVIDIA 驱动最省事；或用 NVIDIA CUDA 官方仓库。

**方式 A（RPM Fusion，推荐先装驱动）：**
```bash
sudo dnf -y install https://download1.rpmfusion.org/free/fedora/rpmfusion-free-release-$(rpm -E %fedora).noarch.rpm
sudo dnf -y install https://download1.rpmfusion.org/nonfree/fedora/rpmfusion-nonfree-release-$(rpm -E %fedora).noarch.rpm
sudo dnf -y install akmod-nvidia xorg-x11-drv-nvidia-cuda
# 重启后验证
nvidia-smi     # 应看到 GPU 型号与 Driver/CUDA 版本
```

**方式 B（CUDA 官方 toolkit）：** 参考 [NVIDIA CUDA on Fedora](https://developer.nvidia.com/cuda-downloads) 添加 repo 后安装 `cuda-toolkit`。
> 若只为了跑 torch 的 CUDA 版，通常驱动 + CUDA 运行时即可；torch 所需 CUDA 库会随 wheel 一并带（本次实装为 torch 2.14.0+cu130）。

### 3.4 安装 Python（推荐 3.10–3.12）

torch 目前对 Python 3.12 支持最稳；3.13+/3.14 可能没有对应 wheel。本指南以 3.12 为例；也可用 `uv` 免 root 装任意版本 Python（见第 4 节）。

```bash
sudo dnf -y install python3.12 python3.12-pip python3.12-devel
python3.12 --version
```

> 若想用 `uv`（免 sudo、可自动拉指定版本 Python）：
> ```bash
> curl -LsSf https://astral.sh/uv/install.sh | sh   # 装到 ~/.local/bin
> uv venv --python 3.12 ~/.venvs/chatlaw             # 建 3.12 环境
> ```

---

## 4. 克隆代码并安装 Python 依赖

```bash
# 4.1 克隆仓库
git clone https://gitee.com/QingxiBat/ChatLaw.git ~/ChatLaw
cd ~/ChatLaw

# 4.2 建虚拟环境（用 python3.12 或上面的 uv venv）
python3.12 -m venv .venv
source .venv/bin/activate
pip install -U pip

# 4.3 安装依赖
#    注意：PyPI 上 torch 默认是 CUDA 版（较大）；国内可换清华/阿里镜像提速
pip install -r requirements.txt
# 或分步（仅需服务端运行时）：pip install torch transformers openi pyyaml accelerate safetensors

# 4.4 若国内网络慢，可加镜像（示例）
# pip install -i https://mirrors.aliyun.com/pypi/simple/ torch transformers openi pyyaml accelerate safetensors

# 4.5 关键：让 triton 能找到 C 编译器（gcc 已在第 3 步装好，通常无需额外配置）
which gcc gcc++ clang cc 2>/dev/null
```

> 经验：安装 `openi` 后导入时它可能自动尝试 `pip install uvloop`（Linux 下）。若 venv 没带 pip 会报错，用 `uv` 建的环境建议 `uv pip install openi uvloop`，或直接用系统 `pip` 建的 venv。

---

## 5. 准备 OpenI Token（下载模型必须）

- 模型 `enter/QwenModel` 与 tokenizer `enter/QwenTokenizer` 在 OpenI（https://openi.pcl.ac.cn）上托管，**模型仓库需要账号授权**（tokenizer 可匿名，模型不能）。
- 登录 OpenI 后，到 `https://openi.pcl.ac.cn/user/settings/applications` 获取 **访问 Token**。
- 之后服务端启动会自动调用 `download_resources`，需要该 Token 才能拉到模型。

**配置/保存 Token**（用 openi 登录将 Token 写入本地）：
```bash
cd ~/ChatLaw
export OPENI_TOKEN='你的Token'
# 用项目里的 openi（已装）登录，Token 默认存到 $HOME/.openi/token.json，download_resources 会读取
python -c "import os; from openi import login; login(token=os.environ['OPENI_TOKEN'], endpoint='https://openi.pcl.ac.cn')"
```

---

## 6. 获取模型与 tokenizer

服务端 `server_pt.py` 在启动时会执行 `download_resources("tokenizer")` 与 `download_resources("llm")`，**缺文件才下载**。所以有两种方式：

### 方式 A：让服务端自动下载（需第 5 节 Token）
直接启动服务端即可，它会往 `chatlaw/resources/tokenizer`、`chatlaw/resources/llm` 下载。

> ⚠️ **已修复的坑**：仓库里 `chatlaw/dataloader/downloads.py` 曾经把模型下载写成 `repo_type="model"`，但 **`enter/QwenModel` 在 OpenI 上是「数据集」**，必须用 `repo_type="dataset"`。该 bug 已在 commit `ca06528` 修复；若你用的是旧代码，请确认 `repo_type="dataset"`，否则会报「数据不存在或无权限」。

### 方式 B：手动放置（推荐，代码更可控）
1. 用 openi 把 `enter/QwenModel` 以**数据集**方式下载（可用一个独立小环境只装 `openi`）：
   ```python
   import openi
   openi.openi_download_file(
       "enter/QwenModel", repo_type="dataset",
       local_dir="chatlaw/resources", max_workers=10,
       token="你的Token", endpoint="https://openi.pcl.ac.cn",
   )
   ```
2. 该数据集只有单个文件 `llm.zip`，下载后**不会自动解压**，需手动解压并放到正确位置：
   ```bash
   cd chatlaw/resources
   # 解压 zip 后模型文件通常在 llm/ 子目录，需上移一层
   unzip llm.zip -d llm_tmp
   mkdir -p llm
   mv llm_tmp/*/* llm/  2>/dev/null || mv llm_tmp/llm/* llm/ 2>/dev/null
   # 使 llm/ 下直接有 config.json / model.safetensors.index.json / model-0000x-*.safetensors
   rm -rf llm_tmp llm.zip
   ls llm/   # 应看到 config.json, generation_config.json, model-00001-of-00003.safetensors ...
   ```
3. tokenizer 同理可自动或手动放到 `chatlaw/resources/tokenizer/`（含 `tokenizer.json`, `vocab.json`, `merges.txt`, `tokenizer_config.json`）。

> 验证模型齐全：`chatlaw/resources/llm/config.json`、`model.safetensors.index.json`、3 个 `model-*.safetensors` 缺一不可；`chatlaw/resources/tokenizer/` 四个文件齐全。

---

## 7. 配置 config.yaml

`chatlaw/config.yaml`：
```yaml
SERVER_IP: "<服务端所在机器的局域网 IP，供客户端连接>"
HEARTBEAT_PORT: 6005
DATA_PORT: 6006
HB_INTERVAL: 1.0
HB_TIMEOUT: 3.0
HEALTH_PATH: "/health"
HEARTBEAT_PATH: "/heartbeat"
INFER_PATH: "/infer"
```
- 若客户端与本机同机，`SERVER_IP` 用 `127.0.0.1` 也可；跨机用服务端实际 IP。
- 服务端绑定 `0.0.0.0`，请确保防火墙放行 6005/6006：
  ```bash
  sudo firewall-cmd --permanent --add-port=6005/tcp --add-port=6006/tcp
  sudo firewall-cmd --reload
  ```

---

## 8. 启动服务端

```bash
cd ~/ChatLaw
source .venv/bin/activate
export PYTHONUNBUFFERED=1

# PyTorch 版（推荐）
python -m chatlaw.server.server_pt

# 或用启动器
# python launcher.py server pt
```

- MindSpore 版对应：`python -m chatlaw.server.server_ms`（需安装 mindnlp/mindspore 并配置相应后端，PyTorch 版更通用）。
- 启动日志会出现：`[Backend] Using NVIDIA GPU.`、`[Model] Loading Qwen model...`、`[Heartbeat] HTTP listening on port 6005`、`[Data] HTTP listening on port 6006`，随后服务常驻。

**首次 /infer 会做一次 triton 内核编译（需 gcc 已在），随后生成由 GPU 完成。**

---

## 9. 验证

另开一个终端，用 curl 测 POST：
```bash
# 心跳/健康检查（6005）
curl -s -X POST http://127.0.0.1:6005/health -H 'Content-Type: application/json' -d '{"probe":"health"}'
# 期望: {"ok": true, "service": "heartbeat"}
curl -s -X POST http://127.0.0.1:6005/heartbeat -H 'Content-Type: application/json' -d '{"action":"PING"}'
# 期望: {"ok": true, "reply": "PONG"}
```
真正的推理 `/infer` 需要 pickle 的 `model_inputs`（tokenizer 输出），建议用仓库自带的客户端 `chatlaw/client/client_pt.py`（Gradio UI）连到本机测试；或复用项目里的 `chatlaw.network.http_transport.post_pickle` 发请求。

---

## 10. 常见问题（FAQ）

| 现象 | 原因 / 解决 |
|---|---|
| `Failed to find C compiler` / `ld.lld: unable to find library -l:libcuda.so.1` | **缺 C 工具链**。`sudo dnf groupinstall "Development Tools"`，装 gcc/g++/binutils。 |
| 模型下载报「数据不存在或无权限」 | a) 缺 OpenI token（第 5 节）；b) 用了旧代码 `repo_type="model"`——应改为 `"dataset"`（已修复）。 |
| `enter/QwenModel` 下下来只有个 zip / 模型在 llm/llm 里 | 数据集以 `llm.zip` 托管且**不自动解压**，需手动解压并整理到 `chatlaw/resources/llm/`（第 6 节方式 B）。 |
| `import openi` 时 pip 装 uvloop 报错 | Linux 下 openi 需要 `uvloop`；venv 里 `pip install uvloop`，或确保 venv 带 pip。 |
| `pip install` 报 **找不到 `mindnlp` 版本** | `mindnlp` **只有 MindSpore 路由（`server_ms`）需要**，PyTorch 路由（`server_pt`）不需要。且 mindnlp 0.5.1 依赖 `mindspore<=2.7.0`，这些 mindspore 版本**没有 Python 3.12 的 wheel**（只有 cp39/310/311）→ 故 3.12 下无法解析。**解决**：部署 PyTorch 路由（跳过 mindnlp，`requirements.txt` 已用 marker 在 3.12+ 自动跳过）；若确需 MS 路由，改用 **Python 3.11**。 |
| `torch` 装不上 / 无对应版本 | 换 Python 3.12（用 `uv venv --python 3.12`）；torch 用 `--index-url` 指定国内镜像或 CUDA 版本。 |
| 生成非常慢 / GPU 无法满负荷 | 显存或功耗受限（笔记本/节能）。检查 `nvidia-smi`；若显存不足 `device_map="auto"` 会回退 CPU 减缓速。可降低 `max_new_tokens` 或换更小模型。 |
| 客户端连不上 | `SERVER_IP` 是否正确、防火墙 6005/6006 是否放行。 |

---

## 11. 给「交给 AI 部署」的检查清单

若打算让 AI 按本指南部署，把下面这一段直接作为任务描述即可：

> 在 Fedora + NVIDIA GPU 机器上部署 ChatLaw 服务端：
> 1. 安装基础工具与 **C/C++ 工具链**（Development Tools），确认 `gcc --version` 可用。
> 2. 安装 NVIDIA 驱动（RPM Fusion 的 `akmod-nvidia` 或 CUDA toolkit），`nvidia-smi` 正常。
> 3. 用 Python 3.12 + venv（或 `uv venv --python 3.12`）克隆仓库并 `pip install -r requirements.txt`（torch 装 CUDA 版）。
> 4. 用 OpenI 访问令牌 `openi.login(token=...)`，确保 `enter/QwenModel`（**dataset 类型**）与 `enter/QwenTokenizer` 能下载；若走手动，把模型解压整理到 `chatlaw/resources/llm`、tokenizer 到 `chatlaw/resources/tokenizer`。
> 5. 配置 `chatlaw/config.yaml` 的 `SERVER_IP`，放行 6005/6006。
> 6. 启动 `python -m chatlaw.server.server_pt`，确认监听 6005/6006。
> 7. 用 curl 验证 `/health`、`/heartbeat`；用客户端验证 `/infer` 能返回模型文本。

---

*部署参考资料：模型来自 OpenI（`enter/QwenModel`/`enter/QwenTokenizer`），Pytorch 后端；如无 GPU 也可用 MindSpore 版 `server_ms`，但需另行配置 MindSpore 运行环境。*
