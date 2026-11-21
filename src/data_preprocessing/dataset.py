from __future__ import annotations

"""
PyTorch Dataset：从 JSON 划分文件中读取样本，并加载视频为张量。
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from torch import Tensor
from torch.utils.data import Dataset

from .frame_extractor import extract_frames


class VideoDataset(Dataset):
    """基于 JSON 列表的视频数据集。"""

    def __init__(
        self,
        json_path: str,
        root_dir: str | None = None,
        target_frames: int = 32,
        resize: Tuple[int, int] = (112, 112),
        use_augmentation: bool = False,
        decode: str = "cpu",
    ) -> None:
        """
        Args:
            json_path: JSON 划分文件路径（train/val/test.json）
            root_dir: 可选，数据根目录；如果为 None，则使用 JSON 中的相对路径
            target_frames: 每个视频采样的帧数
            resize: 帧尺寸 (H, W)
            use_augmentation: 是否在训练时做简单数据增强（目前保留参数，未实现复杂增强）
            decode: 解码设备，"cpu" 使用 OpenCV CPU 解码，"gpu" 使用 GPU 解码
        """
        super().__init__()

        self.json_path = Path(json_path)
        self.root_dir = Path(root_dir) if root_dir is not None else None
        self.target_frames = target_frames
        self.resize = resize
        self.use_augmentation = use_augmentation
        self.decode = decode

        with self.json_path.open("r", encoding="utf-8") as f:
            self.samples: List[Dict[str, Any]] = json.load(f)

        if not isinstance(self.samples, list):
            raise ValueError(f"JSON 文件格式错误: {json_path}")

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:  # type: ignore[override]
        item = self.samples[idx]

        video_rel = item["video_path"]
        action_id = int(item["action_id"])
        score = int(item["score"])

        # 解析视频路径
        video_path = Path(video_rel)
        if self.root_dir is not None and not video_path.is_absolute():
            video_path = self.root_dir / video_path

        frames = extract_frames(
            str(video_path),
            target_frames=self.target_frames,
            resize=self.resize,
            decode=self.decode,
        )  # (C, T, H, W)

        # 标签打包为张量
        action_id_tensor = torch.tensor(action_id, dtype=torch.long)
        score_tensor = torch.tensor(score, dtype=torch.long)

        return {
            "video": frames,
            "action_id": action_id_tensor,
            "score": score_tensor,
        }
