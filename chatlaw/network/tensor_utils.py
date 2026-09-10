"""把客户端传来的模型输入统一转换为目标设备上的张量。

背景：Android 客户端无法安装 torch，因此它发送的 ``model_inputs`` 里的
``input_ids`` / ``attention_mask`` 是 **numpy 数组**（由设备端内嵌 Python 分词后
pickle）。而 :mod:`chatlaw.server.server_pt` 原先直接调用 ``value.to(device)``，
只接受 ``torch.Tensor``，会在收到 numpy 数组时抛 ``AttributeError``。

本模块对三类输入做统一处理：
- ``torch.Tensor`` —— 直接搬到目标设备；
- ``numpy.ndarray`` —— 经 ``torch.from_numpy`` 转换后搬设备；
- 其它（如 Python ``list``）—— 经 ``torch.as_tensor`` 转换后搬设备。

``torch`` 采用函数内导入：便于在无 torch 的环境下对该模块做纯类型分支测试。
"""

from __future__ import annotations

import numpy as np


def to_device_tensor(value, device):
    """将 ``value`` 转为 ``device`` 上的 torch 张量。

    Args:
        value: ``torch.Tensor`` / ``numpy.ndarray`` / 可被 ``torch.as_tensor`` 接受的序列。
        device: 目标设备（``torch.device`` 或 ``str``，如 ``"cuda"`` / ``"npu"`` / ``"cpu"``）。

    Returns:
        位于 ``device`` 上的 ``torch.Tensor``。
    """
    import torch  # 局部导入，见模块文档

    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value).to(device)
    return torch.as_tensor(value).to(device)


def to_device_tensors(model_inputs, device, *, default_attention_mask=True):
    """把 ``model_inputs`` 中的张量字段统一搬到 ``device``。

    Args:
        model_inputs: 映射，至少包含 ``input_ids``；通常还含 ``attention_mask``。
        device: 目标设备。
        default_attention_mask: 缺失 ``attention_mask`` 时是否用全 1 掩码补齐。

    Returns:
        ``(input_ids, attention_mask)``，均为 ``device`` 上的张量。
    """
    import torch

    input_ids = to_device_tensor(model_inputs["input_ids"], device)

    raw_mask = model_inputs.get("attention_mask") if hasattr(model_inputs, "get") else None
    if raw_mask is None:
        if not default_attention_mask:
            raise KeyError("attention_mask")
        attention_mask = torch.ones_like(input_ids)
    else:
        attention_mask = to_device_tensor(raw_mask, device)

    return input_ids, attention_mask
