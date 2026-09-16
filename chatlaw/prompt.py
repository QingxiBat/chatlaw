"""ChatLaw 的 prompt 模板 —— **训练与推理的公共契约**。

本模块是 ``chatlaw/client/utils/common_utils.py`` 里 ``build_prompt`` /
``_format_doc_block`` 的原样搬迁（2026-09 第三轮结构清理）。那两个函数是全文件里
唯一还有调用方的部分，其余（RAG 检索 / ASR / MathML 渲染 / 遗留 socket 工具）
已随 ``chatlaw/client/`` 一并删除 —— 它们的等价实现就在客户端自己的仓库里
（``chatlaw-flutter-client/python/``，逐行相同的算法；差异仅在嵌入后端：
本地那份用 torch/sentence-transformers，客户端那份用 onnxruntime，因为 Android
没有 torch wheel）。

⚠️ **改动这里的模板必须重新微调模型。**

- 训练侧：``develop/finetune/train_qwen3_finetune.py`` 与 ``merge_lora.py``
  直接 import ``build_prompt`` 来生成训练数据；
- 推理侧：客户端有同一份文案的副本 ``chatlaw-flutter-client/python/build.py``，
  **改一处必须同步另一处**。

改动模板后训练与推理的输入分布会不一致，旧 adapter 学的是旧模板，只能重训。
"""

__all__ = ["build_prompt"]


def _format_doc_block(item, idx: int):
    """
    兼容两种输入：
    1. FAISS 返回的 LangChain Document
    2. 精确索引返回的 dict
    """
    # FAISS Document
    if hasattr(item, "metadata"):
        law_name = (
            item.metadata.get("law_name_raw")
            or item.metadata.get("law_name")
            or item.metadata.get("law_name_norm")
            or ""
        )
        article = item.metadata.get("article", "")
        content = item.metadata.get("content", "") or item.page_content
        return f"{idx}. 《{law_name}》{article}：\n{content}"

    # exact index dict
    if isinstance(item, dict):
        law_name = (
            item.get("law_name_raw")
            or item.get("law_name")
            or item.get("law_name_norm")
            or ""
        )
        article = item.get("article", "")
        content = item.get("content", "")
        return f"{idx}. 《{law_name}》{article}：\n{content}"

    # fallback
    return f"{idx}. {str(item)}"


def build_prompt(question, docs):
    # 本模板是训练与推理的公共契约：develop/finetune/train_qwen3_finetune.py 直接复用
    # 本函数生成训练数据，改动这里必须重新微调，否则训练/推理的输入分布会不一致。
    blocks = []

    for i, doc in enumerate(docs, 1):
        blocks.append(_format_doc_block(doc, i))

    context = "\n\n".join(blocks) if blocks else "（未检索到可用法律条文）"

    return f"""你是一个法律咨询专用模型。
你会遇到以下两种情况：
1. 用户的问题与法律条文无关。
2. 用户的问题与法律条文有关。

如果是情况1，则忽略法律条文，直接回答用户问题。
如果是情况2，请结合法律条文回答，并在回答结尾说明你的判断依据，具体到哪部法律及第几条。
如果当前没有检索到合适法条，不要编造法条内容，应基于常识谨慎回答，并明确说明“未检索到直接对应法条”。

【法律条文】（用户不可见）
{context}

【问题】
{question}

【回答】
"""
