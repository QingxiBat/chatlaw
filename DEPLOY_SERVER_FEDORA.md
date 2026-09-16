# ChatLaw 服务端部署指南（Fedora / Linux + NVIDIA GPU）

> 目标：在一台装有 NVIDIA GPU 的 **Fedora（或同类 Linux）** 上，部署并运行 ChatLaw 服务端（`server_pt` 为 PyTorch 版；`server_ms` 为 MindSpore 版），并验证它能被客户端以 HTTP POST 方式连接、加载模型并完成推理。
>
> 本指南基于一次**实际在 WSL/Ubuntu 上排障到可运行**的经历编写，把踩过的坑（缺 C 编译器、OpenI 模型是数据集、模型以 `llm.zip` 形式托管需解压、token 路径会被 `HOME` 覆盖改写等）都写了出来，可直接照着做，也适合交给 AI 按步骤执行。

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

## 5. 准备 OpenI Token

- 模型 `enter/QwenModel` 与 tokenizer `enter/QwenTokenizer` 在 OpenI（https://openi.pcl.ac.cn）上托管。**实测这两个数据集都是公开的**（`is_private: false`、`can_download: true`），不配 Token 也能下载；只有遇到「数据不存在或无权限」时才需要登录。
- 需要 Token 时：登录 OpenI 后，到 `https://openi.pcl.ac.cn/user/settings/applications` 获取 **访问 Token**。

> ⚠️ **Token 路径的坑（文档旧版写错了）**：`chatlaw/dataloader/downloads.py` 的 `_openi()` 会在 `import openi` **之前**把 `HOME` 覆盖成**仓库下的 `.openi_home/`**；而 openi 的常量 `OPENI_ROOT = Path.home() / ".openi"` 是在 import 时解析的。所以 **`download_resources` 实际读取的是 `<仓库>/.openi_home/.openi/token.json`**，把 token 写到 `$HOME/.openi/token.json`（按旧文档的做法）对服务端**完全无效**。
>
> 要在正确的 HOME 覆盖下登录，用下面这条命令（注意 `HOME=` 前缀）：

```bash
cd ~/ChatLaw
export OPENI_TOKEN='你的Token'
HOME="$PWD/.openi_home" python -c "
import os
from openi import login
login(token=os.environ['OPENI_TOKEN'], endpoint='https://openi.pcl.ac.cn')
"
```

`login()` 会先用该 Token 请求一次用户信息来校验：格式不合法会直接 `exit(1)`；服务端返回 401 会打印「登陆失败！此令牌无效」并且**不写入** token 文件。所以不会出现「看起来保存成功、实际无权限」的情况。（`.openi_home/` 已在 `.gitignore` 中，不会被提交。）

---

## 6. 获取模型与 tokenizer

服务端 `server_pt.py` 在启动时会执行 `download_resources("tokenizer")` 与 `download_resources("llm")`，**缺文件才下载**。所以有两种方式：

### 方式 A：让服务端自动下载（推荐）
直接启动服务端即可，它会往 `chatlaw/resources/tokenizer`、`chatlaw/resources/llm` 补齐缺失的文件。

**模型是 zip 托管，服务端会自动解压。** `enter/QwenModel` 里只有单个 `llm.zip`（压缩后约 6.4GB，**解压后约 8.05GB**），而 `openi_download_file` 只会下载、不会解压。现行代码在下载完成后会调用 `_extract_archives()` 自动完成「解压 → 把 zip 内一层 `llm/` 里的 6 个文件上移到 `resources/llm/` → 删除 zip」，这一步无需人工介入。启动时你会看到：

```
[Resource] Extracting llm.zip (8.1GB)...
[Resource] llm.zip extracted into .../chatlaw/resources/llm.
```

> ⚠️ **解压需要瞬时额外空间**：解压期间 `llm.zip`（6.4GB）与产物（8.05GB）会同时存在，请确保 `chatlaw/resources` 所在分区**至少还有 8.1GB 可用**。空间不足会立刻报 `Not enough disk space to extract ...`（并保留 zip 以便直接重试），而不是解压到一半失败。成功后 zip 被删除，净占用回到约 8GB。

> ⚠️ **两个历史坑（跑旧代码时会遇到）**：
> 1. 模型下载曾写成 `repo_type="model"`，但 **`enter/QwenModel` 在 OpenI 上是「数据集」**，必须用 `repo_type="dataset"`，否则报「数据不存在或无权限」。已由 commit `ca06528` 修复。
> 2. 只下载 zip 而不解压时，加载模型会报
>    `ValueError: Unrecognized model in .../resources/llm. Should have a model_type key in its config.json`。
>    这就是「`resources/llm/` 里只有 `llm.zip`」的典型症状。注意在旧代码下**重启服务端并不会自愈**（下载器按文件名判重，看到 zip 已存在就跳过下载，既不重下也不解压）；升级到含 `_extract_archives()` 的代码后重启即自动修复，或按下面方式 B 手工解压。

### 方式 B：手动放置（想自己控制下载/解压时）
1. 用 openi 把 `enter/QwenModel` 以**数据集**方式下载：
   ```python
   import openi
   openi.openi_download_file(
       "enter/QwenModel", repo_type="dataset",
       local_dir="chatlaw/resources/llm", max_workers=10,
       token="你的Token", endpoint="https://openi.pcl.ac.cn",
   )
   ```
   `local_dir` 直接指向 `chatlaw/resources/llm`（现行代码就是这么调的），所以 zip 会落在 **`chatlaw/resources/llm/llm.zip`**，而不是旧文档写的 `chatlaw/resources/llm.zip`。
2. 该数据集只有单个文件 `llm.zip`（约 6.4GB），下载后**不会自动解压**，且 **zip 内自带一层 `llm/` 目录**，需解压后上移一层：
   ```bash
   cd chatlaw/resources/llm
   unzip -t llm.zip            # 先校验 CRC，正常输出 "No errors detected"
   unzip -q llm.zip -d llm_tmp
   mv llm_tmp/llm/* .          # 关键：zip 内顶层就是 llm/，必须上移一层
   rm -rf llm_tmp llm.zip
   ls                          # 应看到 config.json, generation_config.json,
                               # model.safetensors.index.json, model-00001..00003-of-00003.safetensors
   ```
3. tokenizer 可自动下载，也可手动放到 `chatlaw/resources/tokenizer/`（含 `tokenizer.json`, `vocab.json`, `merges.txt`, `tokenizer_config.json`）。它的数据集是**平铺目录**（远端路径形如 `tokenizer/merges.txt`），`local_dir` 指向 `chatlaw/resources` 时正好落成 `resources/tokenizer/`，**无需解压或整理**。

> 验证模型齐全：`chatlaw/resources/llm/config.json`、`model.safetensors.index.json`、3 个 `model-*.safetensors` 缺一不可（三个 safetensors 合计约 8GB）；`chatlaw/resources/tokenizer/` 四个文件齐全。缺任何一个都会在加载阶段报错，不会静默降级。

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
真正的推理 `/infer` 需要 pickle 的 `model_inputs`（tokenizer 输出），有三种验证方式：

1. **Flutter 客户端（推荐，端到端）**：用 `chatlaw-flutter-client` 构建/运行 APP，把「服务器 IP」填成本机地址。同机联调可用 `adb reverse tcp:6005 tcp:6005` + `adb reverse tcp:6006 tcp:6006`，然后在 APP 里填 `127.0.0.1`，免去配同网段。
2. **直接发一帧请求**：复用项目里的 `chatlaw.network.http_transport.post_pickle`（与客户端同一份协议实现），把 tokenizer 输出的 `model_inputs` pickle 后 POST 到 `http://<SERVER_IP>:6006/infer`；响应是「8 字节大端长度 + pickle」的分帧流，以 `<END>` 结束。
3. **只想调客户端、手头没有 GPU**：用 `scripts/fake_chatlaw_server.py`（仿真服务端，协议与 `http_transport.py` 完全一致），配合上面第 1 条的 `adb reverse`，即可验证「连接 → 心跳 → 流式输出 → 中断」整条链路。

> 早期的 Gradio 桌面客户端（`chatlaw/client/client_pt.py`）已随本轮结构清理删除，客户端现由 `chatlaw-flutter-client` 提供；旧实现可从 tag `pre-cleanup-2026-09` 取回（见 `docs/REPO_STRUCTURE.md` §4）。

---

## 10. 常见问题（FAQ）

| 现象 | 原因 / 解决 |
|---|---|
| `Failed to find C compiler` / `ld.lld: unable to find library -l:libcuda.so.1` | **缺 C 工具链**。`sudo dnf groupinstall "Development Tools"`，装 gcc/g++/binutils。 |
| 模型下载报「数据不存在或无权限」 | a) 用了旧代码 `repo_type="model"`——应改为 `"dataset"`（已修复）；b) 确实需要 token 时，注意它必须落在 `<仓库>/.openi_home/.openi/token.json`（第 5 节），写到 `~/.openi/` 无效。 |
| `Unrecognized model in .../resources/llm. Should have a model_type key in its config.json` | `resources/llm/` 里只有 `llm.zip`、没有解压后的模型文件。现行代码会自动解压（第 6 节方式 A）；若日志里已出现「本地已存在 1 个文件」却仍报此错，说明跑的是旧代码，按方式 B 手工解压即可。 |
| `Not enough disk space to extract ...` | 解压 `llm.zip` 需瞬时额外约 8.1GB（zip 6.4GB 与产物 8.05GB 并存）。腾出空间后重启即可，代码会保留 zip 以便直接重试。 |
| `... is corrupt or incomplete (BadZipFile)` | 下载被中断导致 zip 不完整。删除 `chatlaw/resources/llm/llm.zip` 后重启服务端，或用 `openi_download_file(..., force=True)` 强制重下。 |
| `enter/QwenModel` 下下来只有个 zip / 模型在 llm/llm 里 | 数据集以 `llm.zip` 托管，且 zip 内自带一层 `llm/`。现行代码会自动解压并上移一层；手工处理见第 6 节方式 B。 |
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
> 4. 若需登录，在 `HOME=<仓库>/.openi_home` 下执行 `openi.login(token=...)`（第 5 节），确认 `enter/QwenModel`（**dataset 类型**）与 `enter/QwenTokenizer` 可访问；模型归档 `llm.zip` 由服务端自动解压（第 6 节方式 A，瞬时需约 8.1GB 额外空间）；若走手动，则把 zip 内一层 `llm/` 中的文件解压到 `chatlaw/resources/llm`、tokenizer 放到 `chatlaw/resources/tokenizer`。
> 5. 配置 `chatlaw/config.yaml` 的 `SERVER_IP`，放行 6005/6006。
> 6. 启动 `python -m chatlaw.server.server_pt`，确认监听 6005/6006。
> 7. 用 curl 验证 `/health`、`/heartbeat`；用客户端验证 `/infer` 能返回模型文本。

---

## 12. 可选：切换到 LoRA 微调后的模型

默认部署跑的是 OpenI 上的**官方** Qwen3-4B-Instruct-2507（`config.json` 里 `model_type: qwen3`）。若要用本项目 LoRA 微调出来的模型，需要先合并再替换 `chatlaw/resources/llm`。

### 12.1 为什么不能直接把 adapter 目录丢过去

`develop/finetune/train_qwen3_finetune.py` 保存的产物是 **PEFT adapter**（`adapter_config.json` + `adapter_model.safetensors`，默认在 `chatlaw/resources/llm/lora_adapter`），不是完整模型；而服务端只做 `AutoModelForCausalLM.from_pretrained(resources/llm)`，不认 adapter。中间必须做一次 `merge_and_unload`，这就是 `develop/finetune/merge_lora.py` 的作用。

### 12.2 合并导出

合并需要 `peft`（已在 `requirements.txt` 的 develop 段），并需要能加载 4B 模型的机器（CPU 也可以，只是慢）：

```bash
cd ~/ChatLaw
python develop/finetune/merge_lora.py
# 产物：develop/finetune/qwen3_4b_merged（约 8GB，bf16，3 个分片）
```

脚本会做这些事：
- 校验 base 与 adapter（同族、`target_modules` 都能在基座里找到），不一致会提前报错；
- 合并后按 `max_shard_size="4GB"` 保存，**复现官方 `model-0000x-of-00003` 的命名**，避免被当成"资源不完整"；
- 在产物目录写 `.chatlaw_local_model` 标记文件 —— `downloads.py` 见到它就**不会**再用 OpenI 官方权重覆盖；
- 在 GPU 机器上还会用同一 prompt 分别跑基座与合并模型，打印两段输出供肉眼确认 adapter 生效。

确认无误后替换（或在服务端直接执行）：

```bash
# 方式一：本机合并，手动搬运到服务器
rsync -av --progress develop/finetune/qwen3_4b_merged/ 服务器:~/ChatLaw/chatlaw/resources/llm/
# 注意要带上 .chatlaw_local_model（rsync/cp -a 会带上；用通配符 * 会漏掉隐藏文件）

# 方式二：直接在本机服务端仓库里原地替换
python develop/finetune/merge_lora.py --in-place            # 替换 resources/llm
python develop/finetune/merge_lora.py --in-place --backup   # 替换前把原权重挪到 llm_base_backup/
```

> `.gitignore` 忽略了 `llm/` 与 `resources/`，权重不会进 git，必须单独搬运（约 8GB）。

### 12.3 验证确实加载了微调模型

启动服务端时日志会明确区分权重来源（`server_pt.py` / `server_ms.py` 都会打印）：

```
[Model] 权重来源：本地自定义权重（合并时间 2026-09-10 15:02:11，r=8，adapter=/home/x/ChatLaw/chatlaw/resources/llm/lora_adapter）
[Model] model_type=qwen3 architectures=['Qwen3ForCausalLM'] dtype=torch.bfloat16
```

看到「权重来源：OpenI 官方 enter/QwenModel」说明仍在跑官方模型。也可直接看 `chatlaw/resources/llm/.chatlaw_local_model` 是否存在。

### 12.4 回退到官方模型

```bash
rm -f chatlaw/resources/llm/.chatlaw_local_model          # 先去掉标记
rm -f chatlaw/resources/llm/model*.safetensors chatlaw/resources/llm/model.safetensors.index.json
# 重启服务端 → 检测到 llm 资源不全且无标记，自动从 OpenI 重新下载官方权重
```

### 12.5 注意点

- **prompt 模板只有一份真相源**：模板在 `chatlaw/prompt.py` 的 `build_prompt`（Flutter 端 `chatlaw-flutter-client/python/build.py` 是同一份文案，改动需同步），训练脚本 `train_qwen3_finetune.py` 直接 import 并**复用**它，所以训练与推理的系统设定、约束文案、【法律条文】结构天然一致。**前提是拿这个新脚本重新训练过**：改动模板后必须重新微调，旧 adapter 学的是旧模板。
- **法条块渲染**：训练样本的 `reference` 元素如果是结构化对象（`law_name`/`article`/`content`），渲染结果与线上检索出来的完全一致；如果是纯字符串，则渲染成 `N. 原文`。想让两者严格一致，就把 `dataset.json` 的 `reference` 写成结构化对象，或采样一份原文确认格式。
- **tokenizer 没有变**：微调不改 tokenizer，客户端与协议无需改动；但如果你在微调里替换过 tokenizer 或改了 `chat_template`，必须同步更新客户端与打进 APK 的 `resources/tokenizer`。
- 服务端生成参数仍是硬编码的 `max_new_tokens=4096 / temperature=0.7`，不随模型变化。
- `develop/RAG/law_pdf_to_json.py` 也读 `chatlaw/resources/llm`，原地替换后这个离线抽取脚本会跟着用微调模型；若希望它继续用官方基座，给它单独指一个模型目录。
- 不要把 `lora_adapter/` 目录删掉：重新合并、以及标记文件里记录的 adapter 校验和都要用到它。
- `chatlaw/clear_files.py`（`chatlaw clear` / `chatlaw clear_files`）会整目录删掉 `llm`，检测到自定义权重标记时会打印警告——删掉后要重新合并。

### 12.6 如何重新训练（模板、数据或基座变化时）

只要动了 prompt 模板、微调数据或基座模型，就要重跑一遍完整链路：

```bash
cd ~/ChatLaw && source .venv/bin/activate      # 需要 peft / datasets / transformers / torch

# 1) 准备数据集：develop/resources/finetuning_dataset/dataset.json
#    格式为 JSON 数组，每条至少含 question / answer，可选 reference（法条列表）：
#    [
#      {"question": "…",
#       "answer": "…",
#       "reference": [{"law_name": "中华人民共和国民法典", "article": "第二百八十八条", "content": "…"}]}
#    ]
#    reference 写成结构化对象时，训练 prompt 里的法条块与线上检索结果完全一致。

# 2) 切分训练/测试集（19:1，输出同目录 train.json / test.json）
python develop/finetune/split_dataset.py

# 3) LoRA 微调（配置常量在脚本头部：MAX_LENGTH / EPOCHS / LR / LoRA r 等）
#    输出：develop/finetune/qwen3_4b_lora_finetuned + chatlaw/resources/llm/lora_adapter
python develop/finetune/train_qwen3_finetune.py
#    注意日志里的 [WARN]：prompt 占满 MAX_LENGTH 的样本 labels 全为 -100，等于没参与训练，
#    出现较多时应精简参考法条或调大 MAX_LENGTH 后重跑。

# 4) 合并导出（GPU 机器上会顺带打印基座 vs 合并模型的同题对比，确认 adapter 生效）
python develop/finetune/merge_lora.py                # 先导出到独立目录
python develop/finetune/merge_lora.py --in-place     # 确认后替换 resources/llm

# 5) 重启服务端，确认启动日志出现「权重来源：本地自定义权重」
```

**训练前务必先确认模板是最终版**：`train_qwen3_finetune.py` 用的就是客户端 `build_prompt`，任何模板改动都要在训练之前落地，训练之后再改模板会让 adapter 与线上输入分布错配（只能重训）。


*部署参考资料：模型来自 OpenI（`enter/QwenModel`/`enter/QwenTokenizer`），Pytorch 后端；如无 GPU 也可用 MindSpore 版 `server_ms`，但需另行配置 MindSpore 运行环境。*
