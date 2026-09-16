# develop — ChatLaw 离线工具链

> ⚠️ **本目录不可再生，请勿删除。**
>
> 向量库虽然可以从启智社区（OpenI）的 `enter/vectorstore` 重新下载，但下面两项能力
> **只存在于本目录**，删掉就无法复现当前的知识库与微调模型：
>
> 1. **用自有法律 PDF 重建知识库**（`RAG/` 全链路）；
> 2. **LoRA 重训与合并导出**（`finetune/` 全链路）。

本目录与线上推理链路解耦：服务端只加载 `chatlaw/resources/` 下已就绪的资源，不会调用这里任何脚本。

---

## 0. 运行前提

- **所有脚本都必须在仓库根目录（`ChatLaw/`）下执行**。多个脚本的输入/输出路径是**相对当前工作目录**的（例如 `develop/resources/finetuning_dataset/dataset.json`），换目录执行会读不到文件或把产物写错地方。
- 依赖：`pip install -r requirements.txt`（含 `pymupdf` / `tqdm` / `matplotlib` / `peft` / `bitsandbytes` 等 develop 段依赖）。
- 建库与微调都需要能加载 Qwen 模型的机器（GPU 强烈建议；CPU 可跑但极慢）。
- 模型/tokenizer 需已就绪：`chatlaw/resources/llm`、`chatlaw/resources/tokenizer`（服务端首次启动会自动下载，见 [DEPLOY_SERVER_FEDORA.md](../DEPLOY_SERVER_FEDORA.md) §6）。

---

## 1. `RAG/` — 知识库构建

整体链路：**法律 PDF → 条文抽取（JSON） → 校验 → FAISS 向量库**。

```
Legal Documents/*.pdf
      │  抽取（按方案四选一）
      ▼
  *.json（结构化条文）
      │  校验
      ▼
  verify_extraction.py  →  匹配率 / 失序率报告
      │  建库
      ▼
build_vectorstore.py  →  chatlaw/resources/vectorstore/law_faiss
```

### 1.1 抽取方案（四选一，产物格式需一致）

| 脚本 | 方案 | 输入 | 输出 | 说明 |
|---|---|---|---|---|
| `extract_regex_llm_repair.py` | **正则切分 + LLM 结构化清洗**（Qwen2.5-7B-Instruct-AWQ + batch） | `-p <PDF>` | `-o <JSON>`；不指定时落在 `RAG/resources/extracted_v2/<pdf名>_struct.json` | **当前推荐**：正则定位边界快、LLM 只做结构化，准确率与速度兼顾 |
| `extract_law_pure_regex.py` | 纯正则，不用模型 | `PDF_DIR`（常量） | `./output_json_full/final_json/`（+ `debug_by_file/`） | 最快、无显存需求，靠规则清洗 |
| `extract_sliding_window.py` | 滑动窗口 + LLM（Qwen2.5-3B-Instruct） | `-p <PDF>` | `-o <JSON>` | 早期方案，显存友好但慢；默认模型路径 `RAG/resources/Qwen2.5-3B-Instruct` |
| `law_pdf_to_json.py` | LLM 直接结构化（复用 `chatlaw/resources/llm`） | `PDF_DIR`（常量） | `./output_json` | 最早的原型脚本，见下方「已知坑」 |

`batch_extract_all_laws.py` 是 `extract_regex_llm_repair.py` 的**批量驱动壳**：遍历 `PDF_DIR` 下所有 `*.pdf`，逐个调 `extract_regex_llm_repair.py`，已存在同名输出的自动跳过。其 `PDF_DIR` / `OUTPUT_DIR` 都是常量，且用 `subprocess` 调 `python`，需先按实际路径改好。

### 1.2 校验

```bash
python develop/RAG/verify_extraction.py -p <原PDF路径> -j <提取结果JSON路径> [-v]
```

两项指标：

- **内容匹配率**：每条 `content` 能否在原 PDF 中原样找到（目标 ≥ 98%）；
- **编号顺序**：条文编号是否按序（失序率目标 ≤ 2%）。

建议每个 PDF 抽完就验，不要等到建库时才发现问题。

### 1.3 建库

```bash
python develop/RAG/build_vectorstore.py
```

- **读**：`develop/RAG/resources/output_json/`
- **写**：`chatlaw/resources/vectorstore/`（即 `get_resources_path()/vectorstore`，内含 FAISS 索引与精确索引）
- 核心逻辑在同目录 `utils.py` 的 `build_and_save_vectorstore()`，同时构造三份数据：LangChain `Document` 列表（给 FAISS）、`{law_name_norm}::{article}` 精确索引、按法律名分组的索引。

### 1.4 ⚠️ 已知坑（务必先读）

1. **`law_pdf_to_json.py:9` 残留他人机器的绝对路径**
   ```python
   PDF_DIR = "/home/stu1/li/ChatLaw/develop/RAG/output/Legal Documents"
   ```
   这个路径在原开发者机器上才有，直接跑必然报找不到目录。**需先改成自己的 PDF 目录**。

2. **`law_pdf_to_json.py` 的默认 `OUTPUT_DIR` 与 `build_vectorstore.py` 的读取路径不一致**
   - `law_pdf_to_json.py`：`OUTPUT_DIR = "./output_json"`，相对**当前工作目录**（在 `develop/RAG/` 下执行即 `develop/RAG/output_json`）；
   - `build_vectorstore.py`：读 `develop/RAG/resources/output_json`。

   两者对不上，会出现**「抽取成功但建库时读不到数据」**或**「建库成功但检索为空」**。建库前请把抽取产物统一放到 `develop/RAG/resources/output_json/`（或改 `build_vectorstore.py` 的 `json_path`）。

   > `.gitignore` 已忽略 `develop/RAG/output*/` 与 `develop/RAG/resources/`，产物不会入库。

3. **PDF 本身不在仓库里**，需要自行准备到 `Legal Documents/` 之类的目录。`outline` 抽取脚本都读 `*.pdf`，文件名会被用作输出 JSON 名。

---

## 2. `finetune/` — LoRA 微调

整体链路：**数据集 → 切分 → LoRA 训练 → 合并导出 → 替换服务端权重**。

### 2.1 准备数据集

`develop/resources/finetuning_dataset/dataset.json`（需自行创建该目录），格式为 JSON 数组：

```json
[
  {
    "question": "邻居家的树枝长到我院子里了，我能自己剪掉吗？",
    "answer": "……",
    "reference": [
      {"law_name": "中华人民共和国民法典", "article": "第二百八十八条", "content": "……"}
    ]
  }
]
```

- 必填 `question` / `answer`；`reference`（法条列表）可选。
- **`reference` 建议写成结构化对象**（`law_name` / `article` / `content`），这样训练 prompt 里的法条块与线上 RAG 检索出来的渲染结果完全一致；写成纯字符串会被渲染成 `N. 原文`，与线上不一致。

### 2.2 切分（19:1）

```bash
python develop/finetune/split_dataset.py
```

- 读 `develop/resources/finetuning_dataset/dataset.json`
- 写同目录 `train.json` / `test.json`（每 20 条取第 20 条作测试）
- 无命令行参数

### 2.3 训练

```bash
python develop/finetune/train_qwen3_finetune.py
```

- **读**：`develop/resources/finetuning_dataset/train.json`
- **自动下载**：脚本导入时即 `download_resources("tokenizer")` + `download_resources("llm")`，缺资源会先补齐
- **写**：
  - `develop/finetune/qwen3_4b_lora_finetuned/`（Trainer 产物 + `dropped_samples.json`）
  - `chatlaw/resources/llm/lora_adapter/`（LoRA adapter，供 `merge_lora.py` 使用）
- **配置全在脚本头部**（无命令行参数）：`MAX_LENGTH=1024`、`NUM_TRAIN_EPOCHS=3`、`PER_DEVICE_TRAIN_BATCH_SIZE=1`、`GRADIENT_ACCUMULATION_STEPS=8`、`LEARNING_RATE=2e-4`、`USE_BF16`、`GRADIENT_CHECKPOINTING` 等。
- **prompt 模板复用**：`train_qwen3_finetune.py` 直接 import `chatlaw/client/utils/common_utils.py` 的 `build_prompt`，训练与推理共用同一份系统设定与法条块结构 —— 这是**单一真相源**。
- ⚠️ 注意日志里的 `[WARN]`：prompt 占满 `MAX_LENGTH` 的样本 `labels` 全为 `-100`（等于没参与训练）。出现较多时应精简参考法条或调大 `MAX_LENGTH` 后重跑。

### 2.4 合并导出（服务端加载的前提）

```bash
# 默认：导出到 develop/finetune/qwen3_4b_merged，人工确认后再搬运
python develop/finetune/merge_lora.py

# 直接替换 chatlaw/resources/llm（先暂存校验，通过后才搬入）
python develop/finetune/merge_lora.py --in-place
python develop/finetune/merge_lora.py --in-place --backup   # 替换前把原权重挪到 llm_base_backup/
```

**为什么必须合并**：训练产物是 PEFT adapter（`adapter_config.json` + `adapter_model.safetensors`），而服务端只用 `AutoModelForCausalLM.from_pretrained(resources/llm)` 加载，**不认 adapter**。`merge_and_unload` 把 adapter 烧进权重后，产物就是普通 transformers 模型。

脚本做的事：

- 校验 base 与 adapter 同族、`target_modules` 都能在基座里找到，不一致提前报错；
- 按 `max_shard_size="4GB"` 分片，**复现官方 `model-0000x-of-00003` 命名**，避免被当成「资源不完整」；
- 在产物目录写 `.chatlaw_local_model` 标记 —— `chatlaw/dataloader/downloads.py` 见到它就**不会**再用 OpenI 官方权重覆盖；
- 在 GPU 机器上用同一 prompt 分别跑基座与合并模型打印对比，供肉眼确认 adapter 生效（`--no-compare` 可跳过）。

其它参数：`--base` / `--adapter` / `--out` / `--merge-dtype {bf16,fp32}`。

> ⚠️ 不要把 `chatlaw/resources/llm/lora_adapter/` 删掉：重新合并与标记文件里的 adapter 校验和都要用到它。
> ⚠️ `chatlaw clear` / `chatlaw clear_files` 会整目录删掉 `llm`，检测到自定义权重标记时会打印警告 —— 删掉后要重新合并。

### 2.5 替换后验证

启动服务端，日志会明确区分权重来源：

```
[Model] 权重来源：本地自定义权重（合并时间 …，r=8，adapter=…/lora_adapter）
```

看到「权重来源：OpenI 官方 enter/QwenModel」说明仍在跑官方模型。详见 [DEPLOY_SERVER_FEDORA.md](../DEPLOY_SERVER_FEDORA.md) §12。

---

## 3. 最重要的约束：prompt 模板改动必须重训

`chatlaw/client/utils/common_utils.py` 的 **`build_prompt` 是训练与推理的公共契约**：

- 推理侧：客户端流水线（`chatlaw/client/build.py`）与 Flutter 端（`chatlaw-flutter-client/python/build.py`，同一份文案）使用它；
- 训练侧：`train_qwen3_finetune.py` / `merge_lora.py` 直接 import 它。

因此：

> **只要改了 prompt 模板、微调数据或基座模型，就必须重跑完整链路（切分 → 训练 → 合并 → 重启服务端）。**
> 训练之后再改模板，会让已训好的 adapter 与线上输入分布错配，**只能重训**。

Flutter 端的 `python/build.py` 是同一份文案的副本，**改一处必须同步另一处**。

---

## 4. 遗留项（已知，未在本轮清理）

- `train_qwen3_finetune.py:41-42` 的注释仍写「复用**桌面客户端**的实现」—— 桌面 Gradio 客户端已在 2026-09 结构清理中删除，措辞过期但**逻辑正确**（`common_utils.build_prompt` 仍是唯一真相源）。本轮按约束不改 `develop/` 脚本逻辑，故仅记录在此。
- `law_pdf_to_json.py` 与 `extract_law_pure_regex.py` 的 `PDF_DIR` / `OUTPUT_DIR` 都是硬编码常量、且互相不统一；`build_vectorstore.py` 期望的输入目录与任一抽取脚本的默认输出目录都不一致。整理统一路径需改动脚本，本轮未做（见 §1.4 坑 2 的规避方法）。
