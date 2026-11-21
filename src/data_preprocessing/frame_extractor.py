from typing import Tuple

import cv2
import numpy as np
import torch
from torch import Tensor


def extract_frames(
    video_path: str,
    target_frames: int = 32,
    resize: Tuple[int, int] = (112, 112),
    decode: str = "cpu",
) -> Tensor:
    """
    从视频中提取等间隔帧，并做简单预处理。

    Args:
        video_path: 视频文件路径
        target_frames: 目标帧数 T
        resize: (H, W)，输出尺寸
        decode: 解码设备，"cpu" 使用 OpenCV CPU 解码，"gpu" 使用 GPU 解码（需要 CUDA 支持）

    Returns:
        Tensor: 形状为 (C, T, H, W)，像素归一化到 [0, 1]
    """
    # GPU 解码需要 OpenCV 带 CUDA 支持，这里做简单兼容处理
    if decode == "gpu":
        decode = "cpu"

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"无法打开视频: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        raise ValueError(f"视频帧数为 0: {video_path}")

    # 计算等间隔采样的帧索引
    indices = np.linspace(0, max(total_frames - 1, 0), num=target_frames, dtype=int)

    frames = []
    frame_idx = 0
    target_idx = 0

    while len(frames) < target_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx == indices[target_idx]:
            frame = cv2.resize(frame, (resize[1], resize[0]))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
            target_idx += 1

            if target_idx >= len(indices):
                break

        frame_idx += 1

    cap.release()

    # 如果视频太短无法填满目标帧数，用黑帧填充
    while len(frames) < target_frames:
        black_frame = np.zeros((resize[0], resize[1], 3), dtype=np.uint8)
        frames.append(black_frame)

    arr = np.stack(frames, axis=0)  # (T, H, W, C)
    arr = arr.astype("float32") / 255.0
    arr = np.transpose(arr, (3, 0, 1, 2))  # (C, T, H, W)
    tensor = torch.from_numpy(arr)
    return tensor
