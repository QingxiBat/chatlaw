#!/usr/bin/env python3
"""把 LoRA adapter 合并进基座权重，导出服务端可直接加载的全量模型。

为什么需要这一步：
    train_qwen3_finetune.py 保存出来的只有 PEFT adapter（adapter_config.json +
    adapter_model.safetensors），不是完整模型；而服务端（server_pt.py /
    server_ms.py）只用 AutoModelForCausalLM.from_pretrained 加载
    chatlaw/resources/llm，不认 adapter。所以必须做一次 merge_and_unload，
    把 adapter 烧进权重，服务端才能用上微调效果。

用法（在仓库根目录执行）：
    python develop/finetune/merge_lora.py
        导出到 develop/finetune/qwen3_4b_merged，确认无误后再手工替换
    python develop/finetune/merge_lora.py --in-place
        合并完成后直接替换 chatlaw/resources/llm
    python develop/finetune/merge_lora.py --in-place --backup
        替换前把原有权重挪到 chatlaw/resources/llm_base_backup
    python develop/finetune/merge_lora.py --merge-dtype fp32
        用 fp32 精度合并（数值更精确，约需 16GB 内存，产物仍是 bf16）
    python develop/finetune/merge_lora.py --no-compare
        跳过合并前后的生成效果对比

产物说明：
    权重按 max_shard_size="4GB" 分片，与官方 model-0000x-of-00003 的命名一致；
    同时在产物目录写入 .chatlaw_local_model 标记，
    chatlaw/dataloader/downloads.py 见到该标记就不会再用 OpenI 官方权重覆盖它。
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

# 允许 `python develop/finetune/merge_lora.py` 直接执行（此时脚本目录在 sys.path[0]，包不可见）
if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch  # pylint: disable=wrong-import-position
from transformers import AutoModelForCausalLM, AutoTokenizer  # pylint: disable=wrong-import-position

from chatlaw.client.utils.common_utils import build_prompt as build_client_prompt  # pylint: disable=wrong-import-position
from chatlaw.dataloader import download_resources  # pylint: disable=wrong-import-position
from chatlaw.dataloader.downloads import (  # pylint: disable=wrong-import-position
    LLM_LOCAL_MARKER,
    llm_list,
    write_llm_local_marker,
)
from launcher import get_resources_path  # pylint: disable=wrong-import-position

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ADAPTER_SUBDIR = "lora_adapter"
DEFAULT_OUTPUT = PROJECT_ROOT / "develop" / "finetune" / "qwen3_4b_merged"

# 官方权重就是按 4GB 上限切成 3 片；用同一个值才能切出同名的 model-0000x-of-00003
SHARD_SIZE = "4GB"

# in-place 替换时被认作"权重文件"的文件模式（lora_adapter/ 等子目录不受影响）
WEIGHT_PATTERNS = (
    "config.json",
    "generation_config.json",
    "model.safetensors.index.json",
    "*.safetensors",
)

# 合并前后对比用的固定样例（两次调用用同一个 prompt 即可看出 adapter 是否生效）
COMPARE_QUESTION = "邻居家的树枝长到我院子里了，我能自己剪掉吗？"
COMPARE_LAW_NAME = "中华人民共和国民法典"
COMPARE_ARTICLE = "第二百八十八条"
COMPARE_CONTENT = (
    "不动产的相邻权利人应当按照有利生产、方便生活、团结互助、公平合理的原则，正确处理相邻关系。"
)
COMPARE_MAX_NEW_TOKENS = 64


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="把 LoRA adapter 合并进 Qwen 基座权重，导出服务端可直接加载的全量模型",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--base", type=Path, default=None, help="基座模型目录")
    parser.add_argument("--adapter", type=Path, default=None, help="LoRA adapter 目录")
    parser.add_argument("--out", type=Path, default=None, help="输出目录（--in-place 时作为临时目录）")
    parser.add_argument("--in-place", action="store_true", help="合并后直接替换基座目录")
    parser.add_argument("--backup", action="store_true", help="--in-place 时备份原有权重")
    parser.add_argument(
        "--merge-dtype",
        choices=("bf16", "fp32"),
        default="bf16",
        help="合并时基座模型的精度",
    )
    parser.add_argument("--no-compare", action="store_true", help="跳过合并前后的生成效果对比")
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, Path, Path]:
    """
    解析出（基座目录, adapter 目录, 产物暂存目录, 最终目录）。

    --in-place 时最终目录就是基座目录，产物先落在暂存目录，等校验和对比都过了再搬进去。
    """
    resources = Path(get_resources_path())
    base = (args.base or (resources / "llm")).resolve()
    adapter = (args.adapter or (base / ADAPTER_SUBDIR)).resolve()

    if args.in_place:
        staging = (args.out or (base.parent / f"{base.name}_merged_tmp")).resolve()
        final = base
    else:
        staging = (args.out or DEFAULT_OUTPUT).resolve()
        final = staging

    return base, adapter, staging, final


def ensure_base_ready(base: Path) -> None:
    """基座目录必须是一个可加载的模型目录。"""
    if not base.is_dir():
        raise SystemExit(
            f"[错误] 基座模型目录不存在：{base}\n"
            "      先让服务端启动一次以自动下载官方权重，"
            "或按 DEPLOY_SERVER_FEDORA.md 手动放置到 chatlaw/resources/llm。"
        )
    if not (base / "config.json").is_file():
        raise SystemExit(f"[错误] 基座目录下没有 config.json，不是完整的模型目录：{base}")


def load_adapter_config(adapter: Path) -> dict[str, Any]:
    """读取并校验 adapter 目录，返回 adapter_config.json 的内容。"""
    if not adapter.is_dir():
        raise SystemExit(
            f"[错误] 没有找到 LoRA adapter 目录：{adapter}\n"
            "      先运行 develop/finetune/train_qwen3_finetune.py 完成微调。"
        )

    config_path = adapter / "adapter_config.json"
    if not config_path.is_file():
        raise SystemExit(f"[错误] adapter 目录下缺少 adapter_config.json：{adapter}")

    weights = [p for p in (adapter / "adapter_model.safetensors", adapter / "adapter_model.bin") if p.is_file()]
    if not weights:
        raise SystemExit(f"[错误] adapter 目录下缺少 adapter_model.safetensors（或 .bin）：{adapter}")

    try:
        with config_path.open("r", encoding="utf-8") as f:
            config = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"[错误] 无法解析 {config_path}：{exc}") from exc

    if not isinstance(config, dict):
        raise SystemExit(f"[错误] {config_path} 内容不是对象，无法识别为 PEFT 配置。")

    return config


def print_adapter_summary(adapter: Path, config: dict[str, Any]) -> None:
    """打印 adapter 关键信息，便于确认拿到的是想要的那一版。"""
    print(
        "[Adapter] "
        f"r={config.get('r')} lora_alpha={config.get('lora_alpha')} "
        f"dropout={config.get('lora_dropout')} "
        f"base={config.get('base_model_name_or_path')}"
    )
    targets = config.get("target_modules")
    if targets:
        print(f"[Adapter] target_modules={', '.join(str(x) for x in targets)}")
    for name in ("adapter_model.safetensors", "adapter_model.bin"):
        path = adapter / name
        if path.is_file():
            print(f"[Adapter] {name}: {path.stat().st_size / 1024 ** 2:.1f} MiB")
            break


def check_base_match(adapter_config: dict[str, Any], base_config: Any) -> None:
    """粗查 adapter 与基座是否同族，不一致只告警（真正的形状不匹配会在合并时报错）。"""
    expected = str(adapter_config.get("base_model_name_or_path") or "")
    model_type = str(getattr(base_config, "model_type", "") or "")
    if expected and model_type and model_type.lower() not in expected.lower():
        print(
            f"[Warn] adapter 记录的基座是 {expected!r}，"
            f"而当前基座 model_type={model_type!r}，请确认两者是同一个模型系列。"
        )
    print(
        "[Base] "
        f"model_type={model_type} architectures={getattr(base_config, 'architectures', None)} "
        f"layers={getattr(base_config, 'num_hidden_layers', None)} "
        f"hidden_size={getattr(base_config, 'hidden_size', None)}"
    )


def check_target_modules(model: Any, target_modules: list[Any]) -> None:
    """确认 adapter 的 target_modules 在基座里都能找到，避免 PEFT 中途才报错。"""
    if not target_modules:
        return
    suffixes = {name.rsplit(".", 1)[-1] for name, _ in model.named_modules()}
    missing = [str(name) for name in target_modules if str(name) not in suffixes]
    if missing:
        raise SystemExit(
            f"[错误] 基座模型中找不到这些 LoRA 目标模块：{', '.join(missing)}\n"
            "      adapter 与基座可能不是同一个模型，请检查训练时的基座与当前 resources/llm 是否一致。"
        )


def sha256_of(path: Path) -> str:
    """计算文件 sha256，用于在标记里记录用的是哪一版 adapter。"""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_base(base_dir: Path, merge_dtype: str) -> Any:
    """按指定精度加载基座模型。"""
    dtype = torch.float32 if merge_dtype == "fp32" else torch.bfloat16
    print(f"[Base] 加载基座（{dtype}）：{base_dir}")
    model = AutoModelForCausalLM.from_pretrained(base_dir, local_files_only=True, dtype=dtype)
    model.config.use_cache = True
    return model


def merge_adapter(model: Any, adapter_dir: Path, merge_dtype: str) -> Any:
    """把 adapter 合并进已加载的基座，返回合并后的完整模型。"""
    try:
        from peft import PeftModel  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError as exc:
        raise SystemExit(
            f"[错误] 缺少 peft：{exc}\n"
            "      pip install peft（服务端推理不需要它，只有训练与合并需要）。"
        ) from exc

    print(f"[Merge] 加载 adapter：{adapter_dir}")
    peft_model = PeftModel.from_pretrained(model, str(adapter_dir))
    merged = peft_model.merge_and_unload()

    if merge_dtype == "fp32":
        print("[Merge] 合并精度为 fp32，保存前转回 bfloat16。")
        merged.to(torch.bfloat16)

    # 服务端用 dtype="auto" 加载，config.torch_dtype 必须与真实精度一致，否则会被当成 fp32 加载
    merged.config.torch_dtype = torch.bfloat16
    merged.config.use_cache = True
    return merged


def save_merged(model: Any, out_dir: Path) -> None:
    """按官方分片粒度保存合并后的完整模型。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Save] 写出到：{out_dir}（max_shard_size={SHARD_SIZE}）")
    model.save_pretrained(str(out_dir), max_shard_size=SHARD_SIZE, safe_serialization=True)

    generation_config = getattr(model, "generation_config", None)
    if generation_config is not None:
        generation_config.save_pretrained(str(out_dir))


def verify_output_dir(out_dir: Path) -> None:
    """校验产物文件名与 downloads.py 的 llm_list 是否对得上。"""
    shards = sorted(p for p in out_dir.glob("*.safetensors"))
    total = sum(p.stat().st_size for p in shards)
    print(f"[Verify] 分片：{', '.join(p.name for p in shards) or '（无）'}")
    print(f"[Verify] 合计 {total / 1024 ** 3:.2f} GiB")

    missing = [name for name in llm_list if not (out_dir / name).is_file()]
    if missing:
        print(f"[Verify] ⚠️ 与 downloads.py 的 llm_list 不一致，缺少：{', '.join(missing)}")
        print(
            f"         已写入 {LLM_LOCAL_MARKER} 标记，服务端不会去 OpenI 重新下载，"
            "加载也不受影响；只是文件名与官方命名不同。"
        )
    else:
        print("[Verify] ✅ 文件名与 downloads.py 的 llm_list 完全一致。")


def write_marker(staging: Path, base: Path, adapter: Path, adapter_config: dict[str, Any]) -> None:
    """在产物目录写入本地自定义权重标记，防止服务端把官方权重重新下回来。"""
    weights = [p for p in (adapter / "adapter_model.safetensors", adapter / "adapter_model.bin") if p.is_file()]
    payload = {
        "source": "local-lora-merge",
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "base_dir": str(base),
        "adapter_dir": str(adapter),
        "adapter_model_sha256": sha256_of(weights[0]) if weights else None,
        "lora": {
            "r": adapter_config.get("r"),
            "lora_alpha": adapter_config.get("lora_alpha"),
            "target_modules": adapter_config.get("target_modules"),
            "base_model_name_or_path": adapter_config.get("base_model_name_or_path"),
        },
        "note": "本地 LoRA 合并权重；chatlaw/dataloader/downloads.py 见到本文件会跳过 OpenI 官方权重下载。",
    }
    path = write_llm_local_marker(payload, llm_dir=str(staging))
    print(f"[Marker] 已写入标记：{path}")


def build_compare_prompt(tokenizer: Any) -> str:
    """构造对比用样例 prompt —— 复用客户端的 build_prompt，对比用的就是线上真实输入结构。"""
    docs = [{"law_name": COMPARE_LAW_NAME, "article": COMPARE_ARTICLE, "content": COMPARE_CONTENT}]
    content = build_client_prompt(COMPARE_QUESTION, docs)
    messages = [{"role": "user", "content": content}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def generate_once(model_dir: Path, tokenizer: Any, prompt: str) -> str:
    """用一个模型目录做一次贪心生成。"""
    model = AutoModelForCausalLM.from_pretrained(model_dir, local_files_only=True, dtype="auto", device_map="auto")
    try:
        inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
        output = model.generate(
            **inputs,
            max_new_tokens=COMPARE_MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
        text = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return text.strip()


def compare_generations(base: Path, staging: Path) -> None:
    """同一 prompt 分别用基座与合并产物生成，肉眼确认 adapter 确实生效。"""
    resources = Path(get_resources_path())
    download_resources(resource_type="tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(resources / "tokenizer", local_files_only=True)
    prompt = build_compare_prompt(tokenizer)

    print("[Compare] 样例问题：" + COMPARE_QUESTION)
    for label, model_dir in (("基座  ", base), ("合并后", staging)):
        try:
            answer = generate_once(model_dir, tokenizer, prompt)
        except Exception as exc:  # 显存不足等原因不应让整个流程失败
            print(f"[Compare] {label}：生成失败（{type(exc).__name__}: {exc}）")
            continue
        print(f"[Compare] {label}：{answer or '（空输出）'}")
    print("[Compare] 两份输出不同即说明 adapter 已生效；完全相同则需要检查 adapter 是否被真正加载。")


def delete_stale_weight_files(target_dir: Path, keep_names: set[str]) -> list[str]:
    """删除 in-place 后不再需要的旧权重文件（不动 lora_adapter 等子目录）。"""
    removed = []
    for pattern in WEIGHT_PATTERNS:
        for path in sorted(target_dir.glob(pattern)):
            if path.name not in keep_names:
                path.unlink()
                removed.append(path.name)
    return removed


def install_in_place(base: Path, staging: Path, backup: bool) -> None:
    """把暂存目录里的产物搬进基座目录，并清理不再需要的旧分片。"""
    if backup:
        backup_dir = base.parent / f"{base.name}_base_backup"
        backup_dir.mkdir(parents=True, exist_ok=True)
        for pattern in WEIGHT_PATTERNS:
            for path in sorted(base.glob(pattern)):
                shutil.move(str(path), str(backup_dir / path.name))
        print(f"[In-place] 原有权重已备份到：{backup_dir}")

    new_names = {p.name for p in staging.iterdir()}
    for path in sorted(staging.iterdir()):
        target = base / path.name
        if target.exists():
            target.unlink()
        shutil.move(str(path), str(target))

    stale = delete_stale_weight_files(base, new_names)
    if stale:
        print(f"[In-place] 已清理不再需要的旧文件：{', '.join(stale)}")
    staging.rmdir()
    print(f"[In-place] 完成，{base} 现在是合并后的微调模型。")


def main() -> None:
    args = parse_args()
    base, adapter, staging, final = resolve_paths(args)

    print(f"[Info] 基座目录  ：{base}")
    print(f"[Info] adapter  ：{adapter}")
    print(f"[Info] 输出目录  ：{staging}" + ("（校验通过后替换基座目录）" if args.in_place else ""))

    ensure_base_ready(base)
    adapter_config = load_adapter_config(adapter)
    print_adapter_summary(adapter, adapter_config)

    model = load_base(base, args.merge_dtype)
    check_base_match(adapter_config, model.config)
    check_target_modules(model, list(adapter_config.get("target_modules") or []))
    print(f"[Base] 基座参数量：{sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")

    merged = merge_adapter(model, adapter, args.merge_dtype)
    print(f"[Merge] 合并后参数量：{sum(p.numel() for p in merged.parameters()) / 1e9:.2f}B")

    save_merged(merged, staging)
    # 标记要在校验之前写好：verify_output_dir 的提示里会引用它
    write_marker(staging, base, adapter, adapter_config)
    verify_output_dir(staging)

    # 对比要再加载两个模型，先把合并过程占用的模型释放掉
    del model, merged
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 对比必须在替换之前做，否则基座目录里已经是合并后的权重了
    if args.no_compare:
        print("[Compare] 已按 --no-compare 跳过。")
    elif not torch.cuda.is_available():
        print("[Compare] 未检测到 CUDA，跳过生成对比（CPU 上会非常慢）。")
    else:
        compare_generations(base, staging)

    if args.in_place:
        install_in_place(base, staging, args.backup)
        print(f"[Done] 服务端下次启动即会加载微调后的模型；标记文件：{final / LLM_LOCAL_MARKER}")
    else:
        print(f"[Done] 合并产物在：{final}")
        print("       替换服务端模型：把该目录内容拷到服务器的 chatlaw/resources/llm（含 "
              f"{LLM_LOCAL_MARKER}），或用 --in-place 在本机直接替换。")
        print("       回退官方权重：删掉 resources/llm 下的标记再让服务端启动，会重新下载官方权重。")


if __name__ == "__main__":
    main()
