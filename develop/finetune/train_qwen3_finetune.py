#!/usr/bin/env python3
"""使用 PyTorch + Transformers + PEFT(LoRA) 微调 Qwen/Qwen3-4B-Instruct-2507。

按用户要求：
1) 直接使用 `python train_qwen3_finetune.py` 启动（不使用命令行参数）。
2) 数据集从固定路径读取。
3) 训练输出保存在本脚本同目录下新建的子目录中。
4) LoRA 参数额外保存到 chatlaw/resources/llm/lora_adapter 目录下。
5) 训练样本的 user content 直接复用客户端推理用的 build_prompt，
   保证训练与推理的系统设定、约束文案、【法律条文】结构完全一致。

微调完成后还需要合并导出才能被服务端加载：见 develop/finetune/merge_lora.py。
"""

from __future__ import annotations

import os
import json
import sys
from pathlib import Path
from typing import Any

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 允许 `python develop/finetune/train_qwen3_finetune.py` 直接执行（此时脚本目录在 sys.path[0]，包不可见）
if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch  # pylint: disable=wrong-import-position
from datasets import Dataset  # pylint: disable=wrong-import-position
from transformers import (  # pylint: disable=wrong-import-position
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)
from peft import LoraConfig, TaskType, get_peft_model  # pylint: disable=wrong-import-position

# 训练与推理共用同一份 prompt 模板：这里复用桌面客户端的实现，
# 避免两边各写一份文案导致分布漂移（客户端模板改了就必须重新训练）。
from chatlaw.client.utils.common_utils import build_prompt as build_client_prompt  # pylint: disable=wrong-import-position
from chatlaw.dataloader import download_resources  # pylint: disable=wrong-import-position
from launcher import get_resources_path  # pylint: disable=wrong-import-position

# =========================
# 训练配置（按需直接修改此处）
# =========================
resource_path = get_resources_path()
download_resources(resource_type="tokenizer")
download_resources(resource_type="llm")

tokenizer_path = os.path.join(resource_path, "tokenizer").replace("\\", "/")
model_path = os.path.join(resource_path, "llm").replace("\\", "/")

SEED = 42

# 建议 LoRA 训练先把长度降一点，更省显存
MAX_LENGTH = 1024
NUM_TRAIN_EPOCHS = 3
PER_DEVICE_TRAIN_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 8
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.03
LOGGING_STEPS = 10
SAVE_STEPS = 200
USE_BF16 = True
GRADIENT_CHECKPOINTING = True

# LoRA 配置
LORA_R = 8
LORA_ALPHA = 32
LORA_DROPOUT = 0.1
LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

# 数据文件路径
train_file_path = os.path.join("develop", "resources", "finetuning_dataset", "train.json")

# 输出目录（脚本目录下）
OUTPUT_SUBDIR = "qwen3_4b_lora_finetuned"
DROPPED_SAMPLES_FILE = "dropped_samples.json"

# LoRA adapter 额外保存目录
# 你消息里写的是 chawlaw/resources/llm，我这里按项目实际更可能是 chatlaw/resources/llm
# 如果你确实要保存到别的位置，改这个路径即可
LORA_SAVE_DIR = Path("chatlaw") / "resources" / "llm" / "lora_adapter"


def load_records(train_file: str | Path) -> list[dict[str, Any]]:
    train_file = Path(train_file)

    if train_file.suffix.lower() == ".jsonl":
        records: list[dict[str, Any]] = []
        with train_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    with train_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("JSON 文件应为数组格式：[{...}, {...}]")
    return data


def check_record(example: dict[str, Any]) -> tuple[bool, str]:
    if not isinstance(example, dict):
        return False, "sample_not_dict"

    question_raw = example.get("question", "")
    answer_raw = example.get("answer", "")

    question = str(question_raw).strip() if question_raw is not None else ""
    answer = str(answer_raw).strip() if answer_raw is not None else ""

    if not question:
        return False, "missing_or_empty_question"

    if not answer:
        return False, "missing_or_empty_answer"

    references = example.get("reference", [])
    if references is None:
        references = []
    elif not isinstance(references, list):
        references = [references]

    return True, "ok"


def save_dropped_samples(dropped_samples: list[dict[str, Any]], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(dropped_samples, f, ensure_ascii=False, indent=4)


def build_user_content(example: dict[str, Any]) -> str:
    """
    渲染训练样本的 user content。

    直接调用客户端推理用的 build_prompt，保证训练与推理的 prompt 逐字一致：
    系统设定、"不得编造法条"等约束、【法律条文】（用户不可见）结构、【回答】结尾都在这里统一。
    本函数只负责把样本字段规整成 build_prompt 需要的 (question, docs)。
    """
    references = example.get("reference", [])
    if references is None:
        references = []
    if not isinstance(references, list):
        references = [references]

    # reference 元素可以是结构化对象（含 law_name / article / content 时按客户端检索结果渲染），
    # 也可以是纯字符串（按 "N. 原文" 渲染）
    docs = [item for item in references if isinstance(item, dict) or str(item).strip()]
    question = str(example.get("question", "")).strip()

    return build_client_prompt(question, docs)


def preprocess_function(example: dict[str, Any], tokenizer: AutoTokenizer) -> dict[str, Any]:
    answer = str(example.get("answer", "")).strip()
    user_content = build_user_content(example)

    messages = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": answer},
    ]

    full_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    prompt_text = tokenizer.apply_chat_template(
        messages[:1],
        tokenize=False,
        add_generation_prompt=True,
    )

    full = tokenizer(
        full_text,
        truncation=True,
        max_length=MAX_LENGTH,
        add_special_tokens=False,
    )
    prompt = tokenizer(
        prompt_text,
        truncation=True,
        max_length=MAX_LENGTH,
        add_special_tokens=False,
    )

    input_ids = full["input_ids"]
    attention_mask = full["attention_mask"]

    prompt_len = min(len(prompt["input_ids"]), len(input_ids))
    labels = [-100] * prompt_len + input_ids[prompt_len:]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        # 标记这类样本：prompt 已占满 MAX_LENGTH，labels 全为 -100，梯度为 0
        "fully_masked": prompt_len >= len(input_ids),
    }


class SupervisedDataCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        max_len = max(len(f["input_ids"]) for f in features)

        input_ids = []
        attention_mask = []
        labels = []

        for f in features:
            pad_len = max_len - len(f["input_ids"])
            input_ids.append(f["input_ids"] + [self.pad_token_id] * pad_len)
            attention_mask.append(f["attention_mask"] + [0] * pad_len)
            labels.append(f["labels"] + [-100] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent

    train_file = project_root / train_file_path
    output_dir = script_dir / OUTPUT_SUBDIR
    output_dir.mkdir(parents=True, exist_ok=True)

    dropped_samples_path = output_dir / DROPPED_SAMPLES_FILE
    lora_save_path = project_root / LORA_SAVE_DIR
    lora_save_path.mkdir(parents=True, exist_ok=True)

    set_seed(SEED)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        dtype=torch.bfloat16 if USE_BF16 else None,
    )

    # 避免 gradient checkpointing 冲突
    model.config.use_cache = False

    if GRADIENT_CHECKPOINTING:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    # ===== LoRA 包装 =====
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ===== 过滤无效样本 =====
    records = load_records(train_file)

    total_count = len(records)
    valid_records = []
    dropped_samples = []
    drop_reason_stats: dict[str, int] = {}

    for idx, record in enumerate(records):
        is_valid, reason = check_record(record)
        if is_valid:
            valid_records.append(record)
        else:
            drop_reason_stats[reason] = drop_reason_stats.get(reason, 0) + 1
            dropped_samples.append(
                {
                    "index": idx,
                    "drop_reason": reason,
                    "sample": record,
                }
            )

    dropped_count = len(dropped_samples)

    print(f"[INFO] 原始样本数: {total_count}")
    print(f"[INFO] 有效样本数: {len(valid_records)}")
    print(f"[INFO] 丢弃样本数: {dropped_count}")

    for reason, count in drop_reason_stats.items():
        print(f"[INFO] 丢弃原因 {reason}: {count}")

    save_dropped_samples(dropped_samples, dropped_samples_path)
    print(f"[INFO] 丢弃样本已保存到: {dropped_samples_path}")

    if len(valid_records) == 0:
        raise ValueError("过滤后没有有效样本，无法继续训练。请检查数据集格式。")

    dataset = Dataset.from_list(valid_records)

    tokenized_dataset = dataset.map(
        lambda x: preprocess_function(x, tokenizer=tokenizer),
        remove_columns=dataset.column_names,
        desc="Tokenizing train dataset",
    )

    print(f"[INFO] Tokenize 后样本数: {len(tokenized_dataset)}")
    print(f"[INFO] Tokenize 前过滤丢弃样本数: {dropped_count}")

    # 这类样本只有 prompt（甚至被截断），labels 全为 -100，梯度为 0，需要在训练前就暴露出来
    fully_masked = sum(1 for flag in tokenized_dataset["fully_masked"] if flag)
    fully_masked_ratio = fully_masked / max(len(tokenized_dataset), 1)
    if fully_masked:
        print(
            f"[WARN] 有 {fully_masked} 条/{len(tokenized_dataset)} 条"
            f"（{fully_masked_ratio:.1%}）样本的 prompt 已达到 MAX_LENGTH={MAX_LENGTH}，"
            "labels 全为 -100、梯度为 0，实际上没有参与训练。"
            "建议精简参考法条、缩短问题，或提高 MAX_LENGTH。"
        )
    else:
        print("[INFO] 没有出现 prompt 占满长度导致 labels 全为 -100 的样本。")

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=NUM_TRAIN_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=2,
        bf16=USE_BF16,
        fp16=(not USE_BF16 and torch.cuda.is_available()),
        report_to="none",
        dataloader_num_workers=0,  # 降低 fork 警告和额外开销
        remove_unused_columns=False,
    )

    collator = SupervisedDataCollator(pad_token_id=tokenizer.pad_token_id)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=collator,
        processing_class=tokenizer,
    )

    print(f"[INFO] 使用数据集: {train_file}")
    print(f"[INFO] 输出目录: {output_dir}")
    print(f"[INFO] LoRA 参数保存目录: {lora_save_path}")

    trainer.train()

    # 保存脚本目录下的训练结果
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    # 额外保存一份 LoRA adapter 到 chatlaw/resources/llm/lora_adapter
    model.save_pretrained(str(lora_save_path))
    tokenizer.save_pretrained(str(lora_save_path))

    print("[INFO] LoRA 微调完成")
    print(f"[INFO] LoRA adapter 已保存到: {lora_save_path}")
    print("[INFO] 服务端要真正用上微调效果，还需把 adapter 合并进权重后再替换 resources/llm：")
    print("       python develop/finetune/merge_lora.py            # 先导出到独立目录校验")
    print("       python develop/finetune/merge_lora.py --in-place # 直接替换 resources/llm")


if __name__ == "__main__":
    main()