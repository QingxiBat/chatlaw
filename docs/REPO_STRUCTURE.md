# ChatLaw 仓库结构清理清单（Feat_app 分支）

> 本文档是 **2026-09 仓库结构清理与文档校准** 的执行依据与留档，覆盖三轮：
>
> - **第一轮（§1–§7）**：删除已废弃的 Gradio 桌面客户端与桌面包壳，重写 README。
> - **第二轮（§8）**：删除 `chatlaw/client/` 三层流水线（已无调用方，被 Flutter 端
>   逐行等价的实现取代），并补齐此前缺失的 ONNX 嵌入模型导出脚本。
> - **第三轮（§9）**：删除 `chatlaw/client/` 整目录 —— 把唯一还活着的 prompt 契约
>   提到 `chatlaw/prompt.py`，其余死代码一并删除。
> - **第四轮（§10）**：清理 `requirements.txt` 里零引用的依赖。
>
> 冻结点：tag `pre-cleanup-2026-09`（Feat_app @ `774bcf9`）。
> 原则：**只清理已失效的客户端代码，不动任何服务端 / 协议 / 算法 / prompt 公共契约**。

---

## 0. 项目现状（清理前的真实形态）

| 组成 | 位置 | 状态 |
|---|---|---|
| 服务端（pt 臂线） | `chatlaw/server/server_pt.py` | 运行中，Fedora + NVIDIA GPU 已实机部署 |
| 服务端（ms 臂线） | `chatlaw/server/server_ms.py` | 保留（MindSpore/昇腾，需 Python ≤ 3.11） |
| 通信层 | `chatlaw/network/http_transport.py` | 已从裸 socket 重构为 **HTTP POST** |
| 客户端（当前） | `chatlaw-flutter-client`（git submodule） | **真机验证通过**，Android 内嵌 Python 跑完整流水线 |
| 客户端（旧，第一轮删除） | `chatlaw/client/client_{pt,ms}.py` + Gradio UI | 已删除 |
| 客户端流水线（旧，第二轮删除） | `chatlaw/client/{rag,build,tokenizer}.py` | 已删除；实现在 Flutter 端 |
| 客户端流水线（当前） | `chatlaw-flutter-client/python/{rag,build,tokenize,asr}.py` | **唯一在跑的实现** |
| prompt 公共契约（保留） | `chatlaw/prompt.py` 的 `build_prompt` | **活的**：`develop/finetune/` 直接调用（第三轮从 `chatlaw/client/utils/common_utils.py` 原样搬迁） |
| 离线工具链 | `develop/`（RAG 建库 + ONNX 导出 + LoRA 微调） | 保留，**能力不可再生** |
| 后续方向 | C++ 重写算法 + Qt 前端 | 不引入任何 C++/Qt 代码 |

旧的 Gradio 桌面客户端（`client_pt.py` / `client_ms.py` / `launcher_gui.py`）只服务「PC 上打开浏览器页面联调」这一历史场景；Flutter 客户端已真机验证，且 Gradio 客户端不覆盖 Android，**保留它只会让新接入者误以为它是正式客户端**。这是本轮删除的唯一动机。

---

## 1. A 类：运行时必需（保留，不动）

| 路径 | 引用证据（谁在用） |
|---|---|
| `chatlaw/server/**` | `setup.py` `find_packages(include=["chatlaw","chatlaw.*"])`；部署入口 `python -m chatlaw.server.server_pt`。`server_pt.py:24` / `server_ms.py:20` 分别 import `launcher.get_resources_path` 与 `chatlaw.network.http_transport` |
| `chatlaw/network/**` | `server_pt.py:17`、`server_ms.py:20`（`http_transport`）、`server_pt.py:23`（`tensor_utils.to_device_tensors`） |
| `chatlaw/dataloader/**` | `server_pt.py:16`、`server_ms.py:19`、`chatlaw/clear_files.py:14`、`develop/finetune/train_qwen3_finetune.py:43`、`merge_lora.py:50-51`（第二轮前还有已删的 `chatlaw/client/{rag.py:23, tokenizer.py:17}`） |
| `chatlaw/configuration.py` | `server_pt.py:15`、`server_ms.py:18`；自身 `:7` 依赖 `launcher.get_project_root` |
| `chatlaw/config.yaml` | `configuration.py:9` 拼 `get_project_root()/config.yaml` 读取；含 `SERVER_IP` / `HEARTBEAT_PORT` / `DATA_PORT` / `INFER_PATH` |
| `launcher.py` | **运行时依赖，不是启动器**。被 `chatlaw/server/server_pt.py:24`、`chatlaw/configuration.py:7`、`chatlaw/dataloader/downloads.py:6`、`develop/RAG/*`、`develop/RAG/export_embedding_onnx.py`、`develop/finetune/*` 直接 import。**绝不能整文件删除**，第二轮只摘掉其中指向已删客户端的 `client` 子命令 |
| `setup.py` | 打包入口，`entry_points` 指向 `launcher:main`，`py_modules=["launcher"]` |
| `requirements.txt` | 服务端（pt 臂线）+ develop 工具链依赖 |
| `requirements-server-ms.txt` | ms 臂线依赖说明（mindnlp 需 Python ≤ 3.11） |
| `MANIFEST.in` | `recursive-include chatlaw *.*`，保证 `config.yaml` 等非 py 文件进包 |
| `chatlaw/clear_files.py` | `launcher.py:194`（`chatlaw clear` / `clear_files`）调用；`:14` import `dataloader.downloads.LLM_LOCAL_MARKER` |

---

## 2. B 类：移植参照与不可再生（保留，不动）

| 路径 | 为什么必须留 |
|---|---|
| `chatlaw/prompt.py` | **训练与推理的 prompt 公共契约**。`build_prompt` 及其内部调用的 `_format_doc_block` 被 `develop/finetune/train_qwen3_finetune.py`、`merge_lora.py` 直接 import 并调用。第三轮从 `chatlaw/client/utils/common_utils.py` **原样搬迁**（逐字节一致，见 §9.2），函数体一字未改 |
| `develop/**` | 知识库重建、ONNX 嵌入导出与 LoRA 微调的**唯一**入口，见 `develop/README.md` |
| `scripts/orangepi/**` | 昇腾 310p/310b 环境安装脚本，ms 臂线部署依赖 |
| `scripts/fake_chatlaw_server.py` | 客户端联调桩（实现与 `http_transport.py` 一致的 6005/6006 HTTP 协议，配合 `adb reverse` 使用） |
| `scripts/static_ip_*.sh`、`scripts/dhcp_*.sh`（6 个） | 私有地址一对一联网组网脚本，README「使用前准备」章节的落地脚本 |
| `scripts/pylint_check.sh`、`.pylint.conf` | 代码检查 |
| `.gitee/images/**` | README 引用的 3 张图（`flow shart.png` 流程图、`UI.png`、`show.gif`） |
| `LICENSE`、`.gitmodules`、`chatlaw-flutter-client` | 许可证与 submodule 定义，保持不动 |

---

## 3. C 类：验证可删（**已执行**，证据见下）

执行前已把本清单交用户确认，用户答复：**「全删（含 README_APP.md）」**。
每一项均逐个 grep 全仓（含 `README.md`、`DEPLOY_SERVER_FEDORA.md`、`scripts/`、代码），确认**无任何保留文件 import 或调用**。

| 删除项 | 唯一引用方 / 证据 | 判定 |
|---|---|---|
| `chatlaw/client/client_pt.py` | 仅 `README.md:135`、`DEPLOY_SERVER_FEDORA.md:232` 提及；子模块 `docs/client_functions.md` 的历史性引用（该文档本身过期，见 §5）。无任何保留代码 import 它。它自身 `:4 import gradio` 是仓库里 gradio 的唯一硬依赖 | 删 |
| `chatlaw/client/client_ms.py` | 仅 `README.md:136` 提及。无保留代码 import | 删 |
| `chatlaw/client/utils/utils_pt.py` | 仅被 `client_pt.py:22` import（`README.md:130` 提及） | 删 |
| `chatlaw/client/utils/utils_ms.py` | 仅被 `client_ms.py:16-17` import（`README.md:131` 提及） | 删 |
| `launcher_gui.py` | 仅 `run.sh:12`、`run.bat:85` 调用（`README.md:102` 提及）。`:6 from PySide6...` 是 pyside6 的唯一硬依赖 | 删 |
| `run.bat`、`run.sh` | 无任何文件引用它们；二者内容全部是「装 embeddable Python 的 pip → 装 requirements → 起 `launcher_gui.py`」的桌面包壳，随 GUI 一起失效 | 删 |
| `bootstrap/get-pip.py` | 仅被 `run.bat:27-34` 使用（`README.md:139` 提及） | 删 |
| `develop/RAG/try_rag.py` | 仅 `README.md:107` 提及。它以**硬编码 `Qwen/Qwen3-4B-Instruct-2507`** 自己复制了一份 `load_vectorstore`/`build_prompt`（`:19`、`:39`），是 `chatlaw/client/rag.py` + `build.py` 重构前的早期试验脚本，功能已被完全覆盖 | 删 |
| `scripts/push_to_all_repositories` | 仅 `README.md:82` 提及。内容是对 **3 个历史上游仓库**轮换 `git remote set-url` 后 `git push`，当前仓库唯一 remote 是 `ChatLaw → gitee.com/QingxiBat/ChatLaw.git`，脚本已失效且有误推风险。上游 URL 留档见 §6 | 删 |
| `scripts/README_APP.md` | **未跟踪文件**，无任何引用。仅 21 行的空模板骨架（只有标题与空目录） | 删（**不可从 git 恢复**，需确认） |

> 📌 表中引用的行号（`README.md:135`、`DEPLOY_SERVER_FEDORA.md:232` 等）均指 **`pre-cleanup-2026-09` 时的文件内容**，是删除前的原始证据；`README.md` 已在阶段 3 重写、`DEPLOY_SERVER_FEDORA.md` 已同步修订，现行行号不再对应。

### 3.1 与 `requirements.txt` 的联动

| 依赖 | 处理 | 依据 |
|---|---|---|
| `gradio==5.49.1` | **第一轮删** | 全仓仅 `client_pt.py:4` / `client_ms.py:4` 使用 |
| `pyside6` | **第一轮删** | 全仓仅 `launcher_gui.py:6` 使用 |
| `markdown` | **第一轮保留、第二轮删** | 第一轮它被保留下来的 `rag.py:18` 顶层 import；第二轮 `rag.py` 也删了，全仓不再有任何引用 |
| `latex2mathml` | **第三轮删** | 只被 `common_utils.render_mathml_from_latex` 使用（`latex2mathml.converter`）。该函数第二轮后已无调用方，第三轮随 `chatlaw/client/` 一并删除 |
| `funasr_onnx`、`soundfile`、`librosa` | **第三轮删** | 只被 `common_utils.speech_to_text` 使用。Flutter 端 `python/asr.py` 已改用标准库 `wave` + 纯 numpy 重采样，其自己的 `python/requirements.txt` 注明「桌面端需要时手动安装」 |
| `langchain-community`、`langchain-huggingface` | **保留** | 被 `develop/RAG/utils.py:7-8` 顶层 import（离线建库）。注：`langchain_core` 虽被 `utils.py:6` 直接 import，但由这两个包传递依赖带入，故未单独声明 |
| `faiss-cpu`、`sentence-transformers` | **保留** | `faiss-cpu` 经 `langchain_community.vectorstores.FAISS` 使用；`sentence-transformers` 被 `export_embedding_onnx.py` 用于数值校验，也是 `HuggingFaceEmbeddings` 的底层依赖 |
| `onnxruntime` | **第二轮新增** | `develop/RAG/export_embedding_onnx.py` 用于导出后的数值一致性校验与 ONNX 侧推理。放在 develop 段，服务端与训练都不需要 |
| `accelerate`、`bitsandbytes` | **保留** | 两者都**没有直接 import**，极易被误删：`accelerate` 在传入 `device_map` 时由 `transformers/modeling_utils.py:4155` 的 `accelerate_dispatch()` 使用（`server_pt.py:95` 传了 `device_map="auto"`）；`bitsandbytes` 被 `develop/RAG/extract_regex_llm_repair.py:88-90` 的 `BitsAndBytesConfig(load_in_4bit=True)` 使用（代码 import 的是 `transformers.BitsAndBytesConfig`，不是 `bitsandbytes` 本身） |
| `jieba`、`langchain`（元包）、`matplotlib` | **第四轮删** | `jieba` 全仓零引用（`bbff15f` 语音输入时代遗留）；`langchain` 元包零引用，且 `langchain-community` / `langchain-huggingface` 的 `Requires-Dist` 只要求 `langchain-core` / `langchain-classic`，环境里没有任何第三方包硬依赖这个元包（唯一声明依赖它的是本项目自己的 `chatlaw-0.1.0`）；`matplotlib` 随 `4da2d07` 的 LawBench 测试代码加入，而该测试代码已不在仓库里，全仓零引用 |
| `pyyaml` | **保留** | 模块名是 `yaml`（`chatlaw/configuration.py:6`、`chatlaw/dataloader/downloads.py:24`） |

> ⚠️ 勘误记录（第一轮）：任务书原稿称「`render_mathml_from_latex` 已无调用方」。当时实测 `chatlaw/client/rag.py:147` 仍在调用它，故 `latex2mathml` 必须保留；第三轮 `rag.py` 与 `common_utils.py` 均已删除，依赖随之移除。
>
> ⚠️ 排查依赖时的坑：**PyPI 包名 ≠ 模块名，导入符号也可能 ≠ 包名**。按包名 grep 会得到一批假阴性，直接删会打断已跑通的服务端或建库链路：
>
> | 包 | 为什么 grep 不到 |
> |---|---|
> | `pyyaml` | 模块名是 `yaml` |
> | `accelerate` | 无任何 import，由 transformers 在 `device_map` 分支内部调用 |
> | `bitsandbytes` | 代码 import 的是 `transformers.BitsAndBytesConfig` |
> | `faiss-cpu` | 只经 `langchain_community.vectorstores.FAISS` 间接使用 |
> | `sentence-transformers` | 由 `HuggingFaceEmbeddings` 间接使用 |
>
> 反面教训：第四轮初筛时把 `accelerate` 和 `bitsandbytes` 都判成了「无引用」，逐一核查后才纠正 —— 这两个是本文件里最容易被误删的两个。

### 3.2 `launcher.py` 的同步修改

删除 `client` 子命令与 `client` 分支（原 `:54-55` 的 `if role == "client"` 与 `:189-190` 的分派），否则 `resolve_script_path("client", ...)` 会指向已删除文件。

保留：`server` / `clear`(`clear_files`) 子命令、四个工具函数 `get_project_root` / `get_resources_path` / `get_development_path` / `resolve_script_path`、`run_python_script`。
（`get_development_path` 当前无外部调用方，但 `develop/` 仍在，保留它作为向后兼容的公共工具函数。）

---

## 4. 已删除项与恢复方法

**全部删除项都可从 `pre-cleanup-2026-09` tag 或 `master` 分支取回。**

```bash
# 查看某个被删文件的完整内容
git show pre-cleanup-2026-09:chatlaw/client/client_pt.py

# 只恢复某个被删文件到工作区
git checkout pre-cleanup-2026-09 -- launcher_gui.py

# 恢复全部（回到清理前状态；注意会覆盖工作区同名文件）
git checkout pre-cleanup-2026-09 -- .

# 找清理前的整棵树
git ls-tree -r --name-only pre-cleanup-2026-09
```

`master` 分支是**旧实现的历史归档**，也可作为第二来源（但内容比 `pre-cleanup-2026-09` 更旧，优先用 tag）：
```bash
git show master:chatlaw/client/client_pt.py
```

> ⚠️ `scripts/README_APP.md` 是**未跟踪文件**，任何 git 引用都取不回；删除前已单独说明。

**子模块内的删除**（`chatlaw-flutter-client/docs/client_functions.md`）在外层仓库**不可恢复**，必须在子模块自己的仓库里取回：

```bash
cd chatlaw-flutter-client
git log --oneline -- docs/client_functions.md     # 找到删除前的最后一个提交
git show <该提交>^:docs/client_functions.md       # 查看内容
git checkout <该提交>^ -- docs/client_functions.md
```

### 4.1 撤掉这些删除要连带改什么

若将来需要回退 Gradio 客户端，除恢复文件外还必须：
1. `git checkout pre-cleanup-2026-09 -- launcher.py`（找回 `client` 子命令）；
2. `requirements.txt` 加回 `gradio` / `pyside6`；
3. 恢复 `run.bat` / `run.sh` / `bootstrap/get-pip.py`（如果还要 Windows 免安装包）。

若将来需要回退第二轮的 `chatlaw/client/` 三层流水线：
1. `git checkout 0e022a0 -- chatlaw/client/`（`0e022a0` 是第一轮 `.gitignore` 提交，其中 `rag.py` / `build.py` / `tokenizer.py` 尚在）；
2. `chatlaw/client/__init__.py` 一并恢复（第二轮把它重写成了纯说明文件）；
3. `requirements.txt` 加回 `markdown`；
4. 注意：即便回退，它们**依然没有任何调用方** —— 回退只是为了让代码可读，不会让它重新参与运行。

若将来需要回退第三轮（`chatlaw/client/` 整目录）：
1. `git checkout 8babfe7 -- chatlaw/client/`（`8babfe7` 是第二轮文档提交，当时 `common_utils.py` 与 `__init__.py` 尚在）；
2. `requirements.txt` 加回 `latex2mathml` / `funasr_onnx` / `soundfile` / `librosa`；
3. `develop/finetune/{train_qwen3_finetune,merge_lora}.py` 的 import 改回
   `from chatlaw.client.utils.common_utils import build_prompt`；
4. 但**没必要回退**：`chatlaw/prompt.py` 与原 `common_utils.build_prompt` 的输出逐字节相同
   （§9.2 有断言），回退只会把死代码装回来。

---

## 5. 已知坑与遗留项

1. **`develop/RAG/law_pdf_to_json.py:9` 残留他人机器的绝对路径** —— 默认路径写死在原作者机器上，直接跑会失败，需显式传参或改常量。
2. **`develop/RAG/law_pdf_to_json.py` 的默认 `OUTPUT_DIR` 与 `build_vectorstore.py:7` 的读取路径不一致** —— 建库链路需要人工把产物挪到 `build_vectorstore.py` 期望的位置，否则「建库成功但检索为空」。
3. **`common_utils` 的其余部分（RAG 检索 / ASR / MathML / 遗留 socket 工具）** —— 第三轮已随 `chatlaw/client/` 整目录删除。这些是被 Flutter 端 `chatlaw-flutter-client/python/` 逐行等价取代的重复实现，删掉无信息损失。
4. **子模块 `chatlaw-flutter-client/docs/client_functions.md`** —— 错位且过期：全篇引用外层 `chatlaw/**` 路径（`client_pt.py:35` / `client_ms.py:65` 等），描述的 `format_retrieved_docs_html` 已不存在，行号滞后 2~12 行。**已按用户确认删除**：在子模块内单独提交（`docs: 删除错位过期的 client_functions.md`），随后在外层 bump submodule 指针。同目录 `docs/client_modules.md` 描述的是子模块自身的 `python/**` 模块，内容准确，**保留**。
5. **ONNX 导出脚本的默认路径** —— `export_embedding_onnx.py` 的默认模型目录来自 `launcher.get_resources_path()`，因此要在**仓库根目录**执行；模型放在别处需显式传位置参数。

---

## 6. 历史留档：`scripts/push_to_all_repositories` 的 3 个上游

该脚本被删前的完整内容是对下列 3 个历史上游轮换 `remote set-url` + `git push`：

| 上游 | URL |
|---|---|
| Gitee（原作者） | `git@gitee.com:lishangjing2024/ChatLaw.git` |
| 启智社区 OpenI | `git@code.openi.org.cn:enter/ChatLaw.git` |
| GitHub | `git@github.com:ShangJingLi/ChatLaw.git` |

留档原因：这三个地址在 README 的安装教程（`README.md:203`、`:212`）与数据/模型托管说明里出现过；若将来需要向历史仓库同步，可据此重建推送脚本。

> 当前仓库的唯一 remote 是 `ChatLaw → https://gitee.com/QingxiBat/ChatLaw.git`（`git remote -v`）。

---

## 7. 复现验证命令

```bash
# 1) 核心运行时依赖链（A 类）
python -c "import launcher, chatlaw.configuration, chatlaw.dataloader, chatlaw.network.http_transport, chatlaw.network.tensor_utils"

# 2) CLI 应只列 server / clear / clear_files
python launcher.py --help

# 3) prompt 契约可导入且可调用（不需要任何第三方依赖）
python -c "
import sys; sys.path.insert(0,'.')
from chatlaw.prompt import build_prompt
out = build_prompt('问题', [{'law_name':'中华人民共和国民法典','article':'第二百八十八条','content':'……'}])
assert '1. 《中华人民共和国民法典》第二百八十八条：' in out and '【回答】' in out
print('ok')"

# 4) 微调链路未断：模板单一真相源
grep -n "def build_prompt" chatlaw/prompt.py
grep -n "^from chatlaw.prompt import build_prompt" develop/finetune/train_qwen3_finetune.py develop/finetune/merge_lora.py

# 5) chatlaw.client 应已不存在（期望 ModuleNotFoundError）
python -c "import chatlaw.client" 2>&1 | tail -1

# 6) 悬空引用检查（应无输出）
grep -rn "chatlaw\.client\|common_utils\|client_pt\|client_ms\|utils_pt\|utils_ms\|launcher_gui\|try_rag\|push_to_all_repositories\|get-pip" . \
  --include="*.py" --exclude-dir=.git --exclude-dir=.zcode --exclude-dir=chatlaw-flutter-client

# 7) ONNX 导出脚本可用（需要 torch/transformers/onnxruntime）
python develop/RAG/export_embedding_onnx.py --help

# 8) Flutter 侧未受影响（在子模块内执行，不构建 APK）—— 需本机装有 Flutter SDK
cd chatlaw-flutter-client && flutter analyze
```

---

## 8. 第二轮：删除 `chatlaw/client/` 三层流水线 + 补齐 ONNX 导出

### 8.1 触发原因

Flutter 客户端走的是 `chatlaw-flutter-client/python/{rag,build,tokenize,asr}.py`，不调用
`chatlaw/client/` 里的任何函数。据此复核「`chatlaw/client/` 是否还有保留必要」。

### 8.2 复核结论（证据）

**（1）活的引用只有一处。** 全仓（含子模块）grep `chatlaw.client.*`，外部引用仅：

```
develop/finetune/train_qwen3_finetune.py:42   from chatlaw.client.utils.common_utils import build_prompt as build_client_prompt
develop/finetune/train_qwen3_finetune.py:167  return build_client_prompt(question, docs)
develop/finetune/merge_lora.py:49             from chatlaw.client.utils.common_utils import build_prompt as build_client_prompt
develop/finetune/merge_lora.py:306            content = build_client_prompt(COMPARE_QUESTION, docs)
```

**没有任何文件 import `chatlaw.client.{rag,build,tokenizer}`**，也没有任何文件引用它们导出的符号
（`run_rag` / `build_prompt_pipeline` / `tokenize_prompt` / `RagResult` / …）。它们虽有 `python -m` 的
argparse 入口，但仓内无脚本、无文档、无 CI 调用。

**（2）外层实现 = Flutter 端的逐字副本。** 剥掉注释与 docstring 后逐行比对，下列函数**完全一致**：

`normalize_law_name`、`normalize_article_number`、`match_law_name_from_query`、
`parse_law_article_query`、`_doc_unique_key`、`merge_docs_keep_order`、`retrieve_laws`、
`_format_doc_block`、`build_prompt`。

差异只有三处，且**全部是 Android 适配，没有一处是算法差异**：

| 方面 | 外层（已删） | Flutter 端（在跑） |
|---|---|---|
| 嵌入后端 | `HuggingFaceEmbeddings`（torch / sentence-transformers） | `_OnnxEmbeddings`（onnxruntime，Android 无 torch wheel） |
| 加载失败 | 降级为空检索 | 直接抛 `FileNotFoundError` |
| 资源获取 | `launcher.get_resources_path()` + openi | `APPDATA_DIR` + 随包资源 |

**（3）「C++ 移植规格参照与回归基线」这个理由不成立。** 算法既然逐行相同，外层不提供任何
新增算法信息；而「回归基线」需要的对比测试在仓内根本不存在（子模块的 `python_test/` 测的是
子模块自己的实现），只能人工 `python -m chatlaw.client.rag` 手跑。此外对未来 Qt/C++ 客户端而言，
Flutter 的 ONNX 路径才是更可移植的参照 —— 它完全不依赖 Python/torch（ONNX Runtime 与 FAISS
本身都有 C++ API）。

**（4）`common_utils.py` 561 行里只有约 62 行是活的**（`build_prompt` + `_format_doc_block`）。
其余：RAG 检索半只被已删的 `rag.py` 调用；`speech_to_text` / `recv_exact` /
`connection_acknowledgement` / `match_law_name_from_query` / `parse_law_article_query`
彻底无调用方。

### 8.3 执行内容

| 动作 | 对象 |
|---|---|
| 删除 | `chatlaw/client/rag.py`、`chatlaw/client/build.py`、`chatlaw/client/tokenizer.py` |
| 重写 | `chatlaw/client/__init__.py` —— 原惰性 import 指向已删的三个模块，留着就是坏引用；改为纯说明文件 |
| 加 docstring | `chatlaw/client/utils/common_utils.py` 顶部加模块 docstring，标注「哪些是活的（`build_prompt`）、哪些仅作算法参考」。**代码一字未动** |
| 新增 | `develop/RAG/export_embedding_onnx.py`（见 §8.4） |
| 依赖 | `requirements.txt` 删 `markdown`（`rag.py` 是它唯一使用者）；develop 段新增 `onnxruntime` |

> 保留 `common_utils.py` 而不删的理由：`build_prompt` 是**训练与推理的 prompt 公共契约**，
> 动它要连带改 `develop/` 并触发重新微调。因此选择「保留 + 标注」，代价是文件内大部分函数
> 成为无调用方的参考代码 —— 这一点已写进文件顶部 docstring，避免后来者误判为死代码而误删。

### 8.4 补齐 ONNX 嵌入模型导出

**背景缺口**：Flutter 在 Android 上用 onnxruntime 做向量嵌入（`_OnnxEmbeddings`），需要
`{资源根}/vectorstore/embedding_model/model.onnx`；但 `chatlaw/dataloader/downloads.py:22-37` 的
`vectorstore_list` 从 OpenI 下载并要求的是 **sentence-transformers 格式**
（`pytorch_model.bin` / `modules.json` / `sentence_bert_config.json` / `1_Pooling/config.json` …），
**没有 `model.onnx`**。而这一步的导出脚本此前只存在于某台开发机上
（`chatlaw-flutter-client/docs/client_modules.md:307` 记为 `D:\dev\export_embedding_onnx.py`），
**仓内无法复现**「OpenI 资源 → 安卓可用 model.onnx」，仓内也没有任何 `model.onnx` 产物。

新增 `develop/RAG/export_embedding_onnx.py`，与消费方
`chatlaw-flutter-client/python/rag.py` 的 `_OnnxEmbeddings` 严格对齐：

- **输入**只声明 `input_ids` + `attention_mask`（int64）——消费方按 `if name in enc` 组装 feed，
  声明越少越保险（`token_type_ids` 在部分 tokenizer 配置下不返回）；
- **输出**单个 `sentence_embedding`，形状 `[batch, hidden]`；
- **CLS / mean pooling 与 L2 normalize 内置于图**，图外不做任何后处理。pooling 方式从
  `modules.json` + `1_Pooling/config.json` 读取（而非硬编码），读不出来时要求 `--pooling` 显式指定；
- batch 与 sequence 均为 dynamic axes；
- **默认做数值校验**：以 `SentenceTransformer.encode(normalize_embeddings=True)` 为基准对比
  onnxruntime 输出（阈值 `1e-4`），并打印输出向量的 L2 范数（应全为 1.0，用于证明 normalize 确实入图）。
  缺 `onnxruntime` 或 `sentence-transformers` 时明确打印「未验证」，**不静默通过**；
- 产物默认写在 `<embedding_model>/model.onnx`；若同时生成 `model.onnx.data`（外部权重），
  脚本会提醒**必须同目录一起打包**。

---

## 9. 第三轮：删除 `chatlaw/client/` 整目录，prompt 契约提到 `chatlaw/prompt.py`

### 9.1 触发与结论

第二轮之后 `chatlaw/client/` 只剩 `utils/common_utils.py`（561 行），而其中**只有
`build_prompt` + `_format_doc_block`（约 62 行）还有调用方**，调用方是 `develop/finetune/`
的训练与合并脚本 —— 与「client」毫无关系。目录名在撒谎，且整份文件会被打进 wheel。
因此第三轮把这对函数**原样搬迁**到 `chatlaw/prompt.py`，其余死代码全部删除。

保留 `common_utils.py` 的全部理由在第三轮失效了：唯一约束是「不能动 prompt 契约」，
而搬迁（而非改写）恰好满足这个约束。

### 9.2 搬迁的等价性证明

搬迁前后用同一组输入对比两份实现，**输出逐字节相同**：

| 用例 | 结果 |
|---|---|
| 结构化 dict 法条 | 一致（len=269） |
| 多条法条 + 缺失字段 | 一致（len=286） |
| FAISS `Document` 对象 | 一致（len=268） |
| 空 docs（走「未检索到可用法律条文」分支） | 一致（len=246） |
| 未知类型（走 `str(item)` fallback） | 一致（len=253） |

断言方式：`chatlaw.prompt.build_prompt(q, docs) == chatlaw.client.utils.common_utils.build_prompt(q, docs)`
对上述 5 个用例全部成立。搬迁后另用 develop 的调用方式实测法条块格式
（`1. 《中华人民共和国民法典》第二百八十八条：`）与模板结构（`【法律条文】（用户不可见）` / `【回答】`）未变。

### 9.3 执行内容

| 动作 | 对象 |
|---|---|
| 新增 | `chatlaw/prompt.py` —— `build_prompt` + `_format_doc_block` **函数体一字未改**，附模块 docstring 说明契约与「改动必须重训」 |
| 删除 | `chatlaw/client/` **整目录**（`__init__.py`、`utils/__init__.py`、`utils/common_utils.py`） |
| 改 import | `develop/finetune/train_qwen3_finetune.py`、`merge_lora.py` 各 1 行 → `from chatlaw.prompt import build_prompt`；顺带修掉 train 里那句已失效的「复用**桌面客户端**的实现」注释 |
| 依赖 | `requirements.txt` 删除 `latex2mathml`、`funasr_onnx`、`soundfile`、`librosa`（四者的唯一使用者都在被删的 `common_utils.py` 里） |

### 9.4 依赖排查的两个坑（记录下来避免重犯）

1. **PyPI 包名 ≠ 模块名**，按包名 grep 会产生假阴性：
   - `pyyaml` 的模块名是 `yaml`（`chatlaw/configuration.py:6`）；
   - `accelerate` 没有任何显式 import，但 `server_pt.py` 的 `device_map="auto"` 在运行时需要它；
   - `faiss-cpu` 只通过 `langchain_community.vectorstores.FAISS` 间接使用。
   这三个**必须保留**——照「grep 不到就删」处理会打断已跑通的服务端部署。
2. `jieba` 当时查明全仓无任何引用，但属**早已存在的孤儿**（非第三轮删除造成），故第三轮未动。
   **第四轮已删除**，见 §10。

### 9.5 关于目录名与客户端位置的说明

- 本轮**没有移动、也没有改名 `chatlaw-flutter-client`**（它仍是根目录下的 submodule）。
  实测结论留档在此，供将来决策：
  - 把 Flutter 工程放进 `chatlaw/` 会被 `MANIFEST.in` 的 `recursive-include chatlaw *.*`
    与 `setup.py` 的 `package_data={"chatlaw": ["**/*"]}` 卷进 sdist 与 wheel
    （实测 wheel 里出现 `chatlaw/flutter-client/build/artifacts.bin` 等 4 个条目；
    真实子模块的 `build/` 是 2.4GB）。`.gitignore` 对打包不起作用。
  - 外层仓库**无法跟踪子模块目录内的文件**：`git add chatlaw-flutter-client/x` 会直接报
    `fatal: Pathspec ... is in submodule`，外层只会一直显示 gitlink 变脏。
  - 因此「把外层文件合并进子模块目录」这条路必须先把 submodule 取消掉才可行。

---

## 10. 第四轮：清理 `requirements.txt` 里零引用的依赖

### 10.1 删除项与依据

| 依赖 | 为什么可删 |
|---|---|
| `jieba` | 全仓（`chatlaw/` `develop/` `scripts/` `launcher.py`）零引用；`git log -S` 显示它随 `bbff15f`「加入语音输入API」进来，属旧客户端/ASR 时代遗留。子模块自己的 `python/requirements.txt` 也未使用 |
| `langchain`（元包） | 全仓零引用。保留代码用的是 `langchain_community` / `langchain_huggingface` / `langchain_core` 三个子包，而这个元包只是把它们打包在一起的便利包。逐个体检环境里所有 `*.dist-info` 的 `Requires-Dist`，**没有任何第三方包硬依赖它**（唯一声明依赖它的是本项目自己的 `chatlaw-0.1.0`，即 `setup.py` 读的这份 requirements） |
| `matplotlib` | 全仓零引用；随 `4da2d07`「重构requirements，添加lawbench测试代码」加入，而 **LawBench 测试代码已不在仓库里**，依赖留下成了孤儿 |

### 10.2 险些误删的两个（务必记住）

初筛「按包名 grep」时，`accelerate` 与 `bitsandbytes` 都被判成「无引用」。逐一核查后**两个都必须保留**：

| 包 | 初筛为何漏掉 | 真实使用者 |
|---|---|---|
| `accelerate` | 全仓无任何 import | `transformers/modeling_utils.py:4155` 的 `accelerate_dispatch()` —— 只在传入 `device_map` 时走这条分支，而 `chatlaw/server/server_pt.py:95` 传了 `device_map="auto"`。删了它，服务端加载模型就会失败 |
| `bitsandbytes` | 代码 import 的是 `transformers.BitsAndBytesConfig`，不是 `bitsandbytes` | `develop/RAG/extract_regex_llm_repair.py:88-90` 的 `BitsAndBytesConfig(load_in_4bit=True)`，即「正则切分 + LLM 结构化清洗」抽取方案依赖的 4bit 量化加载 |

由此在 §3.1 补了一张「为什么 grep 不到」的表（`pyyaml` / `accelerate` / `bitsandbytes` / `faiss-cpu` / `sentence-transformers`）。**结论：这个仓库的依赖不能靠 grep 包名来判断死活。**

### 10.3 「直接 import 但未显式声明」的包：全部经传递依赖保证，未改动

静态比对「保留代码的顶层 import」与「`requirements.txt` 声明」，发现 5 个直接 import
但未显式声明的包。逐一核查后**确认它们都由已声明的包硬性传递带入**，故本轮按「只删不加」
的范围未动（严格说「直接 import 的就该直接声明」，但这属风格问题，且本轮不扩大改动面）：

| 未声明的直接 import | 由谁保证 |
|---|---|
| `langchain_core`（`develop/RAG/utils.py:6`） | `langchain-community` 的 `Requires-Dist: langchain-core>=1.0.1` |
| `datasets`（`develop/finetune/train_qwen3_finetune.py:30`） | `transformers` / `peft` / `accelerate` / `sentence-transformers` **四个**已声明包都 `Requires-Dist: datasets` |
| `numpy`（`chatlaw/server/server_ms.py:7`、`chatlaw/network/tensor_utils.py:18`、`export_embedding_onnx.py:283`） | `torch` / `transformers` 的硬依赖 |
| `mindspore`（`server_ms.py`） | `mindnlp` 固定依赖（`mindspore<=2.7.0,>=2.5.0`），而 `mindnlp` 已声明 |
| `torch_npu`（`server_pt.py` 昇腾分支） | 昇腾专用包，从华为的 index 安装，**无法**用 PyPI 依赖表达；属部署环境前置条件，见 `scripts/orangepi/` |

> 排查过程记录：`datasets` 一度被怀疑是「训练链路在新环境跑不起来」的缺口，核查
> `Requires-Dist` 后确认 `transformers` 等多个已声明包都会把它带进来，**不是缺口**。
> 这印证了 §10.2 的结论：这个仓库的依赖判断必须查 `Requires-Dist`，靠 grep 源码不够。
