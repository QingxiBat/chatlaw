#!/usr/bin/env python3
"""把 sentence-transformers 嵌入模型导出为 Android 可用的 ONNX（pooling + normalize 内置于图）。

为什么需要这个脚本
------------------
Android 端装不了 torch / sentence-transformers（没有对应 wheel），向量嵌入改由
onnxruntime 承担 —— 见 ``chatlaw-flutter-client/python/rag.py`` 的 ``_OnnxEmbeddings``。
但从启智社区下载的 ``vectorstore/embedding_model`` 是 sentence-transformers 格式
（``pytorch_model.bin`` + ``modules.json`` + ``1_Pooling/`` …），onnxruntime 加载不了，
必须先导出成 ONNX。

此前这一步只在开发者本机做过（导出脚本未入库，见 ``chatlaw-flutter-client/docs/
client_modules.md`` 里记的 ``D:\\dev\\export_embedding_onnx.py``），导致
「OpenI 资源 → 安卓可用 model.onnx」在仓库里无法复现。本脚本把它补上。

与消费方的契约（必须与 chatlaw-flutter-client/python/rag.py 的 _OnnxEmbeddings 对齐）
--------------------------------------------------------------------------------------
- **输入**：``input_ids``、``attention_mask``（int64）。只声明这两个 —— 消费方按
  ``if name in enc`` 组装 feed，声明得越少越保险（``token_type_ids`` 在部分 tokenizer
  配置下不一定返回）。
- **输出**：单个张量 ``sentence_embedding``，形状 ``[batch, hidden]``。
  **CLS / mean pooling 与 L2 normalize 已在图内完成**，图外不做任何后处理。
- **产物**：``model.onnx``；大模型会额外产生 ``model.onnx.data``（外部权重）。
  若产生了 ``.data``，**两个文件必须同目录一起打包**（onnxruntime 会自动加载），
  漏掉就会加载失败。
- **落点**：默认写在模型目录旁边，即
  ``chatlaw/resources/vectorstore/embedding_model/model.onnx``。

用法
----
    # 默认：处理 chatlaw/resources/vectorstore/embedding_model，导出到原地
    python develop/RAG/export_embedding_onnx.py

    # 指定模型目录 / 输出目录
    python develop/RAG/export_embedding_onnx.py <embedding_model_dir> --out <out_dir>

    # 模型不是 sentence-transformers 布局，或 pooling 方式需要显式指定
    python develop/RAG/export_embedding_onnx.py <dir> --transformer-dir <hf_model_dir> --pooling cls

    # 没有 onnxruntime / sentence-transformers 时跳过数值校验
    python develop/RAG/export_embedding_onnx.py --no-verify

依赖：``torch``、``transformers``（必需）、``sentence-transformers`` 与
``onnxruntime``（仅用于数值校验，缺失时本脚本会提示并跳过，不会失败）。
``onnxruntime`` 已加入 ``requirements.txt`` 的 develop 段。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# 允许 `python develop/RAG/export_embedding_onnx.py` 直接执行
# （此时脚本目录在 sys.path[0]，`launcher` 包不可见）
if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from launcher import get_resources_path  # pylint: disable=wrong-import-position

DEFAULT_MODEL_SUBDIR = ("vectorstore", "embedding_model")
OUTPUT_NAME = "model.onnx"

# 校验用的短句：故意取短句，避免触发截断造成的口径差异（长句会被 max_seq_length 截断）
VERIFY_TEXTS = (
    "邻居家的树枝长到我院子里了，我能自己剪掉吗？",
    "公司股东可以查阅会计账簿吗",
    "用人单位拖欠工资应当承担什么责任",
)
VERIFY_TOL = 1e-4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="把 sentence-transformers 嵌入模型导出为 Android 可用的 ONNX",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "model_dir",
        nargs="?",
        default=None,
        help="sentence-transformers 模型目录（含 modules.json / 1_Pooling）。"
             f"默认 {os.path.join(get_resources_path(), *DEFAULT_MODEL_SUBDIR)}",
    )
    parser.add_argument("--out", type=Path, default=None, help="输出目录，默认与 model_dir 相同")
    parser.add_argument(
        "--transformer-dir",
        default=None,
        help="裸 transformers 权重目录；给定时跳过 modules.json 解析",
    )
    parser.add_argument(
        "--pooling",
        choices=("cls", "mean"),
        default=None,
        help="pooling 方式；默认从 1_Pooling/config.json 读取，读不到时需显式指定",
    )
    parser.add_argument("--max-length", type=int, default=None, help="导出跟踪用的序列长度")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset 版本")
    parser.add_argument(
        "--single-file",
        action="store_true",
        help=f"强制单文件导出 {OUTPUT_NAME}（默认让 torch 自行决定，大模型会拆出 .data）",
    )
    parser.add_argument("--no-verify", action="store_true", help="跳过与 sentence-transformers 的数值校验")
    return parser.parse_args()


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def find_module(modules, suffix: str):
    for module in modules or []:
        if str(module.get("type", "")).endswith(suffix):
            return module
    return None


def resolve_layout(model_dir: Path, args: argparse.Namespace) -> tuple[Path, str, bool]:
    """解析出 (transformers 权重目录, pooling 方式, 原流水线是否带 Normalize 模块)。"""
    modules = None
    modules_path = model_dir / "modules.json"
    if modules_path.is_file():
        modules = read_json(modules_path)

    # --- transformers 权重目录 ---
    if args.transformer_dir:
        transformer_dir = Path(args.transformer_dir).resolve()
    elif modules is not None:
        entry = find_module(modules, "Transformer")
        if entry is None:
            raise SystemExit(
                f"[ERROR] {modules_path} 里找不到 Transformer 模块，无法定位权重目录；"
                "请用 --transformer-dir 显式指定。"
            )
        transformer_dir = (model_dir / str(entry.get("path", ""))).resolve()
    else:
        raise SystemExit(
            f"[ERROR] {model_dir} 下没有 modules.json，看起来不是 sentence-transformers 模型目录。\n"
            "        若这是裸 transformers 目录，请用 --transformer-dir 指定，并用 --pooling 指定池化方式。"
        )

    if not (transformer_dir / "config.json").is_file():
        raise SystemExit(f"[ERROR] {transformer_dir} 下没有 config.json，不是有效的 transformers 权重目录。")

    # --- pooling 方式 ---
    if args.pooling:
        pooling = args.pooling
    else:
        pooling = None
        if modules is not None:
            entry = find_module(modules, "Pooling")
            if entry is not None:
                cfg_path = model_dir / str(entry.get("path", "")) / "config.json"
                if cfg_path.is_file():
                    cfg = read_json(cfg_path)
                    if cfg.get("pooling_mode_cls_token"):
                        pooling = "cls"
                    elif cfg.get("pooling_mode_mean_tokens"):
                        pooling = "mean"
                    else:
                        raise SystemExit(
                            f"[ERROR] {cfg_path} 里既没有 pooling_mode_cls_token 也没有 "
                            "pooling_mode_mean_tokens，本脚本暂不支持该池化方式，请用 --pooling 显式指定。"
                        )
        if pooling is None:
            raise SystemExit(
                "[ERROR] 无法从模型目录推断 pooling 方式，请用 --pooling {cls,mean} 显式指定。"
            )

    has_normalize = modules is not None and find_module(modules, "Normalize") is not None
    return transformer_dir, pooling, has_normalize


def resolve_max_length(model_dir: Path, override: int | None) -> int:
    if override:
        return override
    sbert_cfg = model_dir / "sentence_bert_config.json"
    if sbert_cfg.is_file():
        value = read_json(sbert_cfg).get("max_seq_length")
        if isinstance(value, int) and value > 0:
            return value
    return 512


def build_wrapper(transformer_dir: Path, pooling: str):
    """把 backbone + pooling + L2 normalize 打包成一个可直接导出的模块。"""
    import torch  # pylint: disable=import-outside-toplevel
    from transformers import AutoModel  # pylint: disable=import-outside-toplevel

    class _PooledNormalizedEncoder(torch.nn.Module):
        """forward(input_ids, attention_mask) -> [batch, hidden]，已池化 + L2 归一化。"""

        def __init__(self, backbone, pooling_mode: str):
            super().__init__()
            self.backbone = backbone
            self.pooling_mode = pooling_mode

        def forward(self, input_ids, attention_mask):
            hidden = self.backbone(
                input_ids=input_ids, attention_mask=attention_mask
            ).last_hidden_state

            if self.pooling_mode == "cls":
                pooled = hidden[:, 0]
            else:  # mean：按 attention_mask 加权平均（与 sentence-transformers 一致）
                mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

            return torch.nn.functional.normalize(pooled, p=2, dim=1)

    backbone = AutoModel.from_pretrained(str(transformer_dir))
    wrapper = _PooledNormalizedEncoder(backbone, pooling)
    wrapper.eval()
    return wrapper


def export_onnx(wrapper, tokenizer, out_path: Path, max_length: int, opset: int, single_file: bool) -> None:
    import torch  # pylint: disable=import-outside-toplevel

    for stale in (out_path, Path(f"{out_path}.data")):
        if stale.is_file():
            stale.unlink()
            print(f"[Clean] 删除旧产物 {stale}")

    dummy = tokenizer(
        ["导出用占位文本"],
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    input_ids = dummy["input_ids"].to(torch.long)
    attention_mask = dummy["attention_mask"].to(torch.long)

    kwargs = {
        "input_names": ["input_ids", "attention_mask"],
        "output_names": ["sentence_embedding"],
        "dynamic_axes": {
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "sentence_embedding": {0: "batch"},
        },
        "opset_version": opset,
        "do_constant_folding": True,
    }
    if not single_file:
        kwargs["external_data"] = True

    print(f"[Export] torch.onnx.export -> {out_path}（opset={opset}, max_length={max_length}）")
    with torch.no_grad():
        try:
            torch.onnx.export(wrapper, (input_ids, attention_mask), str(out_path), **kwargs)
        except TypeError as exc:
            if "external_data" not in str(exc):
                raise
            print("[WARN] 当前 torch 不支持 external_data 参数，回退为单文件导出。")
            kwargs.pop("external_data", None)
            torch.onnx.export(wrapper, (input_ids, attention_mask), str(out_path), **kwargs)


def describe_and_verify(out_path: Path, tokenizer, max_length: int) -> bool | None:
    """打印产物签名并做数值校验。返回 True/False，无法校验时返回 None。"""
    try:
        import numpy as np  # pylint: disable=import-outside-toplevel
        import onnxruntime as ort  # pylint: disable=import-outside-toplevel
    except Exception as exc:  # pylint: disable=broad-except
        print(f"[SKIP] 未安装 onnxruntime，跳过 ONNX 侧推理与数值校验（{exc}）")
        return None

    session = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    print("[ONNX] 输入 : " + ", ".join(f"{i.name}{i.shape}({i.type})" for i in inputs))
    print("[ONNX] 输出 : " + ", ".join(f"{o.name}{o.shape}({o.type})" for o in outputs))

    enc = tokenizer(
        list(VERIFY_TEXTS),
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="np",
    )
    feeds = {i.name: enc[i.name].astype(np.int64) for i in inputs if i.name in enc}
    missing = [i.name for i in inputs if i.name not in feeds]
    if missing:
        print(f"[FAIL] 模型声明了 tokenizer 没有提供的输入 {missing}，消费方无法凑齐 feed。")
        return False

    got = session.run(None, feeds)[0]
    norms = np.linalg.norm(got, axis=1)
    print(f"[ONNX] 输出形状 {got.shape}；L2 范数 {np.round(norms, 6).tolist()}（应全为 1.0，证明 normalize 已入图）")

    try:
        from sentence_transformers import SentenceTransformer  # pylint: disable=import-outside-toplevel
    except Exception as exc:  # pylint: disable=broad-except
        print(f"[SKIP] 未安装 sentence-transformers，跳过基准对比（{exc}）")
        return None

    model_dir = out_path.parent
    if not (model_dir / "modules.json").is_file():
        model_dir = Path(get_resources_path()).joinpath(*DEFAULT_MODEL_SUBDIR)
    try:
        reference = SentenceTransformer(str(model_dir)).encode(
            list(VERIFY_TEXTS), normalize_embeddings=True, convert_to_numpy=True
        )
    except Exception as exc:  # pylint: disable=broad-except
        print(f"[SKIP] 加载 sentence-transformers 基准失败，跳过对比（{exc}）")
        return None

    diff = float(np.max(np.abs(got - reference)))
    print(f"[Verify] sentence-transformers 基准 vs ONNX 最大绝对误差 = {diff:.3e}（阈值 {VERIFY_TOL:g}）")
    if diff > VERIFY_TOL:
        print("[FAIL] 误差超阈值：导出的图与基准不一致，请检查 pooling 方式与 max_length。")
        return False
    print("[Verify] 通过：导出结果与 sentence-transformers 输出一致。")
    return True


def main() -> int:
    args = parse_args()

    model_dir = Path(args.model_dir or os.path.join(get_resources_path(), *DEFAULT_MODEL_SUBDIR)).resolve()
    if not model_dir.is_dir():
        raise SystemExit(
            f"[ERROR] 模型目录不存在：{model_dir}\n"
            "        先让服务端下载资源（download_resources(\"vectorstore\")），或用参数指定模型目录。"
        )

    out_dir = Path(args.out).resolve() if args.out else model_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / OUTPUT_NAME

    print(f"[Info] 模型目录   ：{model_dir}")
    print(f"[Info] 输出文件   ：{out_path}")

    transformer_dir, pooling, has_normalize = resolve_layout(model_dir, args)
    max_length = resolve_max_length(model_dir, args.max_length)
    print(f"[Info] 权重目录   ：{transformer_dir}")
    print(f"[Info] pooling    ：{pooling.upper()}")
    print(f"[Info] max_length ：{max_length}")
    if has_normalize:
        print("[Info] 原流水线含 Normalize 模块，导出图内的 L2 归一化与其一致。")
    else:
        print("[WARN] 原流水线没有 Normalize 模块，但消费方（rag._OnnxEmbeddings）期望归一化向量，")
        print("       本脚本仍会在图内做 L2 归一化 —— 这是有意与 sentence-transformers 裸流水线不同的一点。")

    try:
        import torch  # noqa: F401  pylint: disable=import-outside-toplevel,unused-import
        from transformers import AutoTokenizer  # pylint: disable=import-outside-toplevel
    except Exception as exc:  # pylint: disable=broad-except
        raise SystemExit(
            f"[ERROR] 缺少依赖：{exc}\n"
            "        需要 torch 与 transformers（pip install -r requirements.txt）。"
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(str(transformer_dir))
    wrapper = build_wrapper(transformer_dir, pooling)
    export_onnx(wrapper, tokenizer, out_path, max_length, args.opset, args.single_file)

    produced = [p for p in (out_path, Path(f"{out_path}.data")) if p.is_file()]
    if not produced:
        raise SystemExit("[ERROR] 导出结束但没有产物，请检查上面的 torch.onnx.export 报错。")
    for path in produced:
        print(f"[Done] {path}  ({path.stat().st_size / 1024 / 1024:.1f} MB)")

    print()
    print("[Packaging] 把 embedding_model 整个目录打进客户端资源时请注意：")
    print(f"            - 必须带上 {OUTPUT_NAME}" + (" 与 model.onnx.data（外部权重，缺一不可）" if len(produced) > 1 else ""))
    print("            - pytorch_model.bin 可删除（Android 侧不再需要 torch 权重）")
    print("            - chatlaw-flutter-client/python/rag.py 期望路径为 <资源根>/vectorstore/embedding_model/model.onnx")

    if args.no_verify:
        print()
        print("[Verify] 已按 --no-verify 跳过数值校验：**本次导出未经验证**。")
        return 0

    print()
    verdict = describe_and_verify(out_path, tokenizer, max_length)
    if verdict is None:
        print("[Verify] 未验证（缺少 onnxruntime 或 sentence-transformers）—— 不要当作通过。")
        return 0
    return 0 if verdict else 1


if __name__ == "__main__":
    sys.exit(main())
