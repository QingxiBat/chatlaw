"""ChatLaw 客户端流水线的算法参考实现（**不参与运行时**）。

本包不再提供可运行的客户端入口。历史上的三层流水线模块
（``chatlaw/client/rag.py`` → ``build.py`` → ``tokenizer.py``，以及它们对外暴露的
``run_rag`` / ``build_prompt_pipeline`` / ``tokenize_prompt`` 等）已在 2026-09 的
结构清理中删除，原因见 ``docs/REPO_STRUCTURE.md``：

- 它们没有任何调用方 —— 旧 Gradio 桌面客户端才是使用者，而它已被 Flutter 客户端取代；
- 真正在跑的客户端流水线在 ``chatlaw-flutter-client/python/``
  （``rag.py`` / ``build.py`` / ``tokenize.py`` / ``asr.py``），与本包曾经的实现
  是逐行等价的同一份算法。

本包现在只保留 ``chatlaw.client.utils.common_utils``：

- 其中 ``build_prompt`` / ``_format_doc_block`` 是**活的** —— ``develop/finetune/``
  的训练与合并脚本直接 import 并调用它，它是训练与推理的 prompt 公共契约，
  **改动必须重新微调模型**；
- 其余函数（RAG 检索、ASR、MathML 渲染、遗留的 socket 工具）仅作算法参考，
  无任何调用方。详见 ``common_utils`` 的模块 docstring。
"""
