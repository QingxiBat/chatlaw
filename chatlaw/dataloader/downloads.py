import os
from typing import List
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


__all__ = ["download_resources"]
