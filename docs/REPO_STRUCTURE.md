# ChatLaw 仓库结构清理清单（Feat_app 分支）

> 本文档是 **2026-09 仓库结构清理与文档校准** 的执行依据与留档，覆盖两轮：
>
> - **第一轮（§1–§7）**：删除已废弃的 Gradio 桌面客户端与桌面包壳，重写 README。
> - **第二轮（§8）**：删除 `chatlaw/client/` 三层流水线（已无调用方，被 Flutter 端
>   逐行等价的实现取代），并补齐此前缺失的 ONNX 嵌入模型导出脚本。
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
| prompt 公共契约（保留） | `chatlaw/client/utils/common_utils.py` 的 `build_prompt` | **活的**：`develop/finetune/` 直接调用 |
| 离线工具链 | `develop/`（RAG 建库 + ONNX 导出 + LoRA 微调） | 保留，**能力不可再生** |
| 后续方向 | C++ 重写算法 + Qt 前端 | 不引入任何 C++/Qt 代码 |

旧的 Gradio 桌面客户端（`client_pt.py` / `client_ms.py` / `launcher_gui.py`）只服务「PC 上打开浏览器页面联调」这一历史场景；Flutter 客户端已真机验证，且 Gradio 客户端不覆盖 Android，**保留它只会让新接入者误以为它是正式客户端**。这是本轮删除的唯一动机。

---

## 1. A 类：运行时必需（保留，不动）

| 路径 | 引用证据（谁在用） |
|---|---|
| `chatlaw/server/**` | `setup.py` `find_packages(include=["chatlaw","chatlaw.*"])`；部署入口 `python -m chatlaw.server.server_pt`。`server_pt.py:24` / `server_ms.py:20` 分别 import `launcher.get_resources_path` 与 `chatlaw.network.http_transport` |
| `chatlaw/network/**` | `server_pt.py:17`、`server_ms.py:20`（`http_transport`）、`server_pt.py:23`（`tensor_utils.to_device_tensors`）、`chatlaw/client/utils/common_utils.py:7` |
| `chatlaw/dataloader/**` | `server_pt.py:16`、`server_ms.py:19`、`chatlaw/clear_files.py:14`、`develop/finetune/train_qwen3_finetune.py:43`、`merge_lora.py:50-51`（第二轮前还有已删的 `chatlaw/client/{rag.py:23, tokenizer.py:17}`） |
| `chatlaw/configuration.py` | `server_pt.py:15`、`server_ms.py:18`；自身 `:7` 依赖 `launcher.get_project_root` |
| `chatlaw/config.yaml` | `configuration.py:9` 拼 `get_project_root()/config.yaml` 读取；含 `SERVER_IP` / `HEARTBEAT_PORT` / `DATA_PORT` / `INFER_PATH` |
| `launcher.py` | **运行时依赖，不是启动器**。被 `chatlaw/server/server_pt.py:24`、`chatlaw/configuration.py:7`、`chatlaw/dataloader/downloads.py:6`、`chatlaw/client/utils/common_utils.py:6`、`develop/RAG/*`、`develop/RAG/export_embedding_onnx.py`、`develop/finetune/*` 直接 import。**绝不能整文件删除**，第二轮只摘掉其中指向已删客户端的 `client` 子命令 |
| `setup.py` | 打包入口，`entry_points` 指向 `launcher:main`，`py_modules=["launcher"]` |
| `requirements.txt` | 服务端（pt 臂线）+ develop 工具链依赖 |
| `requirements-server-ms.txt` | ms 臂线依赖说明（mindnlp 需 Python ≤ 3.11） |
| `MANIFEST.in` | `recursive-include chatlaw *.*`，保证 `config.yaml` 等非 py 文件进包 |
| `chatlaw/clear_files.py` | `launcher.py:194`（`chatlaw clear` / `clear_files`）调用；`:14` import `dataloader.downloads.LLM_LOCAL_MARKER` |

---

## 2. B 类：移植参照与不可再生（保留，不动）

| 路径 | 为什么必须留 |
|---|---|
| `chatlaw/client/utils/common_utils.py` | **`build_prompt` 与 `_format_doc_block` 是训练与推理的 prompt 公共契约**，被 `develop/finetune/train_qwen3_finetune.py`、`merge_lora.py` 直接 import 并调用。**这两个函数一字不改**；文件内其余部分（RAG 检索 / ASR / MathML / 遗留 socket 工具）已无调用方，第二轮只加了文件顶部 docstring 标注，未动任何代码 |
| `chatlaw/client/__init__.py` | 包标记 + 说明为何本包只剩 `utils/common_utils.py`（第二轮重写；原惰性 import 已删，因为它指向的 `build.py` / `rag.py` / `tokenizer.py` 不存在了） |
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
| `latex2mathml` | **保留** | `common_utils.py:64` 使用于 `render_mathml_from_latex`。该函数第二轮后已无调用方，但 `common_utils.py` 按约定不动代码，故依赖一并保留 |
| `funasr_onnx`、`soundfile`、`librosa` | **保留** | `common_utils.py` 的 `speech_to_text` 引用（该函数第二轮后亦无调用方，理由同上） |
| `langchain*`、`faiss-cpu`、`sentence-transformers` | **保留** | 被 `develop/RAG/utils.py:6-8` 顶层 import（离线建库与导出必需）。它们同时也是 `common_utils` RAG 半的依赖，但**决定性理由是 develop** |
| `onnxruntime` | **第二轮新增** | `develop/RAG/export_embedding_onnx.py` 用于导出后的数值一致性校验与 ONNX 侧推理。放在 develop 段，服务端与训练都不需要 |

> ⚠️ 勘误记录（第一轮）：任务书原稿称「`render_mathml_from_latex` 已无调用方」。当时实测 `chatlaw/client/rag.py:147` 仍在调用它，故 `latex2mathml` 必须保留；两轮之后 `rag.py` 已删，该函数确实失去调用方，但 `common_utils.py` 的代码按约定不动，依赖随之保留。

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

---

## 5. 已知坑与遗留项

1. **`develop/RAG/law_pdf_to_json.py:9` 残留他人机器的绝对路径** —— 默认路径写死在原作者机器上，直接跑会失败，需显式传参或改常量。
2. **`develop/RAG/law_pdf_to_json.py` 的默认 `OUTPUT_DIR` 与 `build_vectorstore.py:7` 的读取路径不一致** —— 建库链路需要人工把产物挪到 `build_vectorstore.py` 期望的位置，否则「建库成功但检索为空」。
3. **`common_utils.speech_to_text` / `render_mathml_from_latex` / RAG 检索半 / 遗留 socket 工具** —— 已全部无调用方（`render_mathml_from_latex` 随第二轮删除 `rag.py` 失去最后的调用方）。Flutter 端在 `chatlaw-flutter-client/python/` 内有自己的实现。**按约定保留**，`common_utils.py` 只在文件顶部加了 docstring 说明，代码一字未动。
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

# 3) chatlaw.client 包仍可导入（第二轮后只剩 common_utils，不需要任何第三方依赖）
python -c "import chatlaw.client, chatlaw.client.utils.common_utils; print('ok')"

# 4) 微调链路未断：模板单一真相源
grep -n "def build_prompt" chatlaw/client/utils/common_utils.py
grep -n "common_utils" develop/finetune/train_qwen3_finetune.py develop/finetune/merge_lora.py

# 5) 悬空引用检查（应无输出）
grep -rn "client_pt\|client_ms\|utils_pt\|utils_ms\|launcher_gui\|try_rag\|push_to_all_repositories\|get-pip\|chatlaw.client.rag\|chatlaw.client.build\|chatlaw.client.tokenizer" . \
  --exclude-dir=.git --exclude-dir=.zcode | grep -v "^./chatlaw-flutter-client/docs/"

# 6) ONNX 导出脚本可用（需要 torch/transformers/onnxruntime）
python develop/RAG/export_embedding_onnx.py --help

# 7) Flutter 侧未受影响（在子模块内执行，不构建 APK）—— 需本机装有 Flutter SDK
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
