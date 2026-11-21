from typing import Literal

import torch
from torch import Tensor


def temporal_augmentation(
    frames: Tensor,
    augmentation_type: Literal["random_crop", "center_crop"] = "random_crop",
    crop_ratio: float = 0.8,
) -> Tensor:
    """
    时间维度数据增强。

    Args:
        frames: 输入张量，形状 (C, T, H, W)
        augmentation_type: 增强方式："random_crop" 或 "center_crop"
        crop_ratio: 裁剪长度比例 (0, 1]

    Returns:
        Tensor: 增强后的张量 (C, T', H, W)
    """
    if frames.dim() != 4:
        raise ValueError(f"frames 形状应为 (C, T, H, W)，实际为 {frames.shape}")

    c, t, h, w = frames.shape
    new_t = max(1, int(t * crop_ratio))
    if new_t >= t:
        return frames

    if augmentation_type == "random_crop":
        start = torch.randint(0, t - new_t + 1, (1,)).item()
    elif augmentation_type == "center_crop":
        start = (t - new_t) // 2
    else:
        raise ValueError(f"未知的时间增强类型: {augmentation_type}")

    end = start + new_t
    return frames[:, start:end, :, :]


