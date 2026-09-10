import json
import os
from typing import List, Optional
from launcher import get_resources_path

llm_list = ["config.json",
            "generation_config.json",
            "model.safetensors.index.json",
            "model-00001-of-00003.safetensors",
            "model-00002-of-00003.safetensors",
            "model-00003-of-00003.safetensors",]

tokenizer_list = ["merges.txt",
                  "tokenizer.json",
                  "tokenizer_config.json",
                  "vocab.json",]

audio_model_list = [".mdl",
                    ".msc",
                    ".mv",
                    "am.mvn",
                    "config.yaml",
                    "configuration.json",
                    "model_quant.onnx",
                    "tokens.json"]

vectorstore_list = ["law_article_index.json",
                    "law_name_index.json",
                    os.path.join("law_faiss", "index.faiss"),
                    os.path.join("law_faiss", "index.pkl"),
                    os.path.join("embedding_model", "1_Pooling", "config.json"),
                    os.path.join("embedding_model", "config_sentence_transformers.json"),
                    os.path.join("embedding_model", "gitattributes"),
                    os.path.join("embedding_model", "modules.json"),
                    os.path.join("embedding_model", "pytorch_model.bin"),
                    os.path.join("embedding_model", "README.md"),
                    os.path.join("embedding_model", "sentence_bert_config.json"),
                    os.path.join("embedding_model", "special_tokens_map.json"),
                    os.path.join("embedding_model", "tokenizer.json"),
                    os.path.join("embedding_model", "tokenizer_config.json"),
                    os.path.join("embedding_model", "vocab.txt")]

# resources/llm 下存在该标记文件，说明里面放的是本地自定义权重（例如 LoRA 合并产物），
# 此时不再从 OpenI 拉官方权重，避免把本地微调权重覆盖或混杂成官方权重。
LLM_LOCAL_MARKER = ".chatlaw_local_model"


def get_llm_local_marker_path() -> str:
    """返回 resources/llm 下自定义权重标记文件的路径。"""
    return os.path.join(get_resources_path(), "llm", LLM_LOCAL_MARKER)


def is_llm_locally_managed() -> bool:
    """判断 resources/llm 是否为本地自定义权重（标记文件存在即为是）。"""
    return os.path.isfile(get_llm_local_marker_path())


def read_llm_local_marker() -> Optional[dict]:
    """
    读取自定义权重标记的内容，用于启动日志展示。

    Returns:
        dict: 标记内容；文件不存在返回 None；内容不可解析时返回带 note 的 dict。
    """
    path = get_llm_local_marker_path()
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError) as exc:
        return {"note": f"标记文件存在但无法解析：{exc}"}
    return data if isinstance(data, dict) else {"value": data}


def write_llm_local_marker(payload: dict, llm_dir: Optional[str] = None) -> str:
    """
    写入自定义权重标记。

    Args:
        payload (dict): 来源信息（来源类型、生成时间、adapter 校验和等）
        llm_dir (str): 目标 llm 目录，默认 resources/llm

    Returns:
        str: 标记文件的绝对路径
    """
    directory = llm_dir or os.path.join(get_resources_path(), "llm")
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, LLM_LOCAL_MARKER)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path


def _openi():
    openi_home = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".openi_home"))
    os.makedirs(openi_home, exist_ok=True)
    os.environ["HOME"] = openi_home
    os.environ["USERPROFILE"] = openi_home

    try:
        import openi
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            f"{exc}. Resource download requires openi. Install project "
            "dependencies with `python -m pip install -r requirements.txt`."
        ) from exc
    try:
        from openi.refactor.service.notification import set_global_notice
        set_global_notice(True)
    except Exception:
        pass
    return openi


def _download_file(repo_id, repo_type, local_dir, resource_type):
    openi = _openi()
    try:
        openi.openi_download_file(
            repo_id,
            repo_type=repo_type,
            local_dir=local_dir,
            max_workers=10
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to download {resource_type} from OpenI repo {repo_id!r}. "
            "Check network access to https://openi.pcl.ac.cn, retry later, "
            "or manually place the required files under chatlaw/resources."
        ) from exc


def download_resources(resource_type):
    """
        根据缺少的文件通过启智社区下载补齐。

        Args:
            resource_type (str): 要下载的文件类型

        Returns:
            bool:
                True  - 目录存在且所有文件齐全
                False - 目录不存在，或缺少任意文件
        """
    if resource_type not in ("tokenizer", "llm", "audio_model", "vectorstore"):
        raise ValueError("Arg 'resource_type' must be in (tokenizer, llm, audio_model)", "vectorstore")
    if resource_type == "tokenizer" and check_files_complete("tokenizer", tokenizer_list) is False:
        _download_file(
            "enter/QwenTokenizer",
            repo_type="dataset",
            local_dir=get_resources_path(),
            resource_type="tokenizer",
        )
    elif resource_type == "llm" and check_files_complete("llm", llm_list) is False:
        if is_llm_locally_managed():
            print(
                "[Data] resources/llm 存在本地自定义权重标记 "
                f"{LLM_LOCAL_MARKER}，跳过 OpenI 官方权重下载。"
            )
        else:
            _download_file(
                "enter/QwenModel",
                repo_type="dataset",
                local_dir=os.path.join(get_resources_path(), "llm"),
                resource_type="llm",
            )
    elif resource_type == "audio_model" and check_files_complete("audio_model", audio_model_list) is False:
        _download_file(
            "enter/Paraformer",
            repo_type="dataset",
            local_dir=os.path.join(get_resources_path(), "audio_model"),
            resource_type="audio_model",
        )
    elif resource_type == "vectorstore" and check_files_complete("vectorstore", vectorstore_list) is False:
        _download_file(
            "enter/vectorstore",
            repo_type="dataset",
            local_dir=os.path.join(get_resources_path(), "vectorstore"),
            resource_type="vectorstore",
        )
    else:
        pass


def check_files_complete(resource_type: str, required_files: List[str]) -> bool:
    """
    检查目录是否存在，且目录下是否包含所需的全部文件。

    Args:
        resource_type (str): 要检查的文件类型
        required_files (List[str]): 必须存在的文件名列表（仅文件名，不含路径）

    Returns:
        bool:
            True  - 目录存在且所有文件齐全
            False - 目录不存在，或缺少任意文件
    """
    # 目录是否存在
    if not os.path.isdir(os.path.join(get_resources_path(), resource_type)):
        return False

    # 检查文件是否齐全
    for filename in required_files:
        file_path = os.path.join(os.path.join(get_resources_path(), resource_type), filename)
        if not os.path.isfile(file_path):
            return False

    return True


__all__ = [
    "download_resources",
    "check_files_complete",
    "llm_list",
    "LLM_LOCAL_MARKER",
    "get_llm_local_marker_path",
    "is_llm_locally_managed",
    "read_llm_local_marker",
    "write_llm_local_marker",
]
