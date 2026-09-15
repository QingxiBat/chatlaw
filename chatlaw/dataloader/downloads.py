import json
import os
import shutil
import zipfile
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


def _locate_payload_dir(staging_dir: str, required_files: List[str]):
    """
    在解压产物中定位真正的载荷目录（即存放资源文件的那一层）。

    归档的顶层目录并不固定：`llm.zip` 内以 `llm/` 为顶层目录，也存在文件直接
    平铺在归档根下的情况，因此按「是否包含任一必需文件」判定，比较文件名。

    Args:
        staging_dir (str): 解压产物的临时目录。
        required_files (List[str]): 目标资源的必需文件列表。

    Returns:
        str | None: 载荷目录路径；未找到时返回 None。
    """
    wanted = set(os.path.basename(name) for name in required_files)
    for root, _dirs, files in os.walk(staging_dir):
        if wanted & set(files):
            return root
    return None


def _extract_archives(target_dir: str, required_files: List[str]) -> None:
    """
    解压 target_dir 下的 zip 归档，并把载荷整理到 target_dir 根下。

    启智社区的 QwenModel 数据集把整个模型打包为单个 llm.zip 托管，而
    `openi_download_file` 只负责下载、不会解压；缺少这一步，`from_pretrained`
    会因为目录里只有 zip 而报 `Unrecognized model ... model_type`。

    Args:
        target_dir (str): 资源目录，解压后的文件会平铺在此目录下。
        required_files (List[str]): 该资源的必需文件列表，用于定位载荷目录。

    Raises:
        RuntimeError: 磁盘空间不足、归档损坏，或归档内找不到必需文件。
    """
    for entry in sorted(os.listdir(target_dir)):
        if not entry.lower().endswith(".zip"):
            continue

        archive = os.path.join(target_dir, entry)
        if not os.path.isfile(archive):
            continue

        staging_dir = os.path.join(target_dir, "_extract_tmp")
        shutil.rmtree(staging_dir, ignore_errors=True)

        try:
            with zipfile.ZipFile(archive) as archive_file:
                unpacked_size = sum(info.file_size for info in archive_file.infolist())
                free_size = shutil.disk_usage(target_dir).free
                if free_size < unpacked_size:
                    raise RuntimeError(
                        f"Not enough disk space to extract {archive}: needs about "
                        f"{unpacked_size / 1e9:.1f}GB, only {free_size / 1e9:.1f}GB free."
                    )

                print(f"[Resource] Extracting {entry} ({unpacked_size / 1e9:.1f}GB)...")
                archive_file.extractall(staging_dir)
        except zipfile.BadZipFile as exc:
            shutil.rmtree(staging_dir, ignore_errors=True)
            raise RuntimeError(
                f"{archive} is corrupt or incomplete ({exc}). Delete it and retry, "
                "or re-download with openi_download_file(..., force=True)."
            ) from exc
        except RuntimeError:
            shutil.rmtree(staging_dir, ignore_errors=True)
            raise

        payload_dir = _locate_payload_dir(staging_dir, required_files)
        if payload_dir is None:
            shutil.rmtree(staging_dir, ignore_errors=True)
            raise RuntimeError(
                f"{archive} does not contain any of the expected files {required_files}; "
                "check the archive layout."
            )

        for name in os.listdir(payload_dir):
            source = os.path.join(payload_dir, name)
            destination = os.path.join(target_dir, name)
            if os.path.isdir(destination):
                shutil.rmtree(destination)
            elif os.path.exists(destination):
                os.remove(destination)
            shutil.move(source, destination)

        shutil.rmtree(staging_dir, ignore_errors=True)
        os.remove(archive)
        print(f"[Resource] {entry} extracted into {target_dir}.")


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
        llm_dir = os.path.join(get_resources_path(), "llm")
        if is_llm_locally_managed():
            print(
                "[Data] resources/llm 存在本地自定义权重标记 "
                f"{LLM_LOCAL_MARKER}，跳过 OpenI 官方权重下载。"
            )
        else:
            _download_file(
                "enter/QwenModel",
                repo_type="dataset",
                local_dir=llm_dir,
                resource_type="llm",
            )
            # 该数据集以单个 llm.zip 托管模型，解压整理后才能被 from_pretrained 识别。
            _extract_archives(llm_dir, llm_list)
            if check_files_complete("llm", llm_list) is False:
                missing = [
                    name for name in llm_list
                    if not os.path.isfile(os.path.join(llm_dir, name))
                ]
                raise RuntimeError(
                    f"Model files still missing under {llm_dir}: {missing}. "
                    f"Delete {llm_dir} and restart the server to re-download."
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
