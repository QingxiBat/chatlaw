"""
ChatLaw 资源清理模块
用于删除 chatlaw/resources 目录下的模型、tokenizer、RAG 等缓存文件。
"""

import os
import shutil
import sys

# 兼容 `python chatlaw/clear_files.py` 直接执行：此时 sys.path[0] 是脚本目录，chatlaw 包不可见
if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chatlaw.dataloader.downloads import LLM_LOCAL_MARKER  # pylint: disable=wrong-import-position


def get_resources_path():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "resources")


def main():
    resource_dir = get_resources_path()

    if not os.path.exists(resource_dir):
        print(f"[提示] 资源目录不存在：{resource_dir}")
        return

    targets = [
        "tokenizer",
        "llm",
        "audio_model",
        "vectorstore",
        "openi_resource.version",
    ]

    # ===== 打印存在的资源项 =====
    print("将要清理以下资源：")
    existing_targets = []

    for target in targets:
        if os.path.exists(os.path.join(resource_dir, target)):
            existing_targets.append(target)

    if existing_targets:
        for t in existing_targets:
            print(f"  - {t}")
    else:
        print("  （无可清理资源）")

    # ===== 本地自定义权重（如 LoRA 合并产物）提醒 =====
    llm_marker = os.path.join(resource_dir, "llm", LLM_LOCAL_MARKER)
    if os.path.isfile(llm_marker):
        print(
            "\n⚠️  警告：chatlaw/resources/llm 中是本地自定义权重（看到 "
            f"{LLM_LOCAL_MARKER}），不是 OpenI 官方权重。\n"
            "    删除后不会自动恢复，需要重新执行 develop/finetune/merge_lora.py 合并，"
            "或清掉该标记后等官方权重重新下载。"
        )

    print(f"\n资源根目录：{resource_dir}")

    # ===== 若无可清理项，提前退出 =====
    if not existing_targets:
        return

    confirm = input("\n确认删除这些资源吗？(y/n): ").strip().lower()
    if confirm != "y":
        print("[取消] 未执行任何删除操作。")
        return

    # ===== 执行删除 =====
    for target in existing_targets:
        path = os.path.join(resource_dir, target)
        try:
            if os.path.isdir(path):
                shutil.rmtree(path)
                print(f"✅ 目录 {path} 已删除")
            else:
                os.remove(path)
                print(f"✅ 文件 {path} 已删除")
        except Exception as e:
            print(f"❌ 删除 {path} 时发生异常：{e}")

    print("清理完成！")


if __name__ == "__main__":
    main()
