from __future__ import annotations


import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from data_preprocessing.video_loader import list_video_files, parse_action_and_score


@dataclass
class SampleInfo:
    video_path: str  # 相对路径（相对于项目根目录）
    action_id: int
    score: int


def build_samples(root_dir: str) -> List[SampleInfo]:
    """从 root_dir 中收集所有样本信息。"""
    root = Path(root_dir).resolve()
    project_root = Path(".").resolve()

    video_paths = list_video_files(root_dir)
    samples: List[SampleInfo] = []

    for p in video_paths:
        action_id, score = parse_action_and_score(p)

        # 保存为相对于项目根目录的路径，便于后续使用
        rel_path = p.resolve().relative_to(project_root)

        samples.append(
            SampleInfo(
                video_path=str(rel_path).replace("\\", "/"),
                action_id=action_id,
                score=score,
            )
        )

    return samples


def split_dataset(
    samples: List[SampleInfo],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
):
    """将样本列表划分成 train / val / test。"""
    if not samples:
        return [], [], []

    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    random.shuffle(samples)
    n = len(samples)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train = samples[:n_train]
    val = samples[n_train : n_train + n_val]
    test = samples[n_train + n_val :]

    return train, val, test


def save_split(samples: List[SampleInfo], path: Path) -> None:
    """将样本列表保存为 JSON 文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = [asdict(s) for s in samples]
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main() -> None:
    data_root = Path("data")
    raw_dir = data_root / "raw"

    if not raw_dir.exists():
        raise FileNotFoundError(f"未找到目录: {raw_dir}，请先运行 scripts/init_raw_folders.py 并放入视频。")

    print(f"正在扫描视频目录: {raw_dir}")
    samples = build_samples(str(raw_dir))
    print(f"共找到 {len(samples)} 个视频文件。")

    if not samples:
        print("没有找到任何视频，请确认已经把视频放到 data/raw/actionX/score/ 下面。")
        return

    train, val, test = split_dataset(samples)
    print(f"划分结果 -> train: {len(train)}, val: {len(val)}, test: {len(test)}")

    splits_dir = data_root / "splits"
    save_split(train, splits_dir / "train.json")
    save_split(val, splits_dir / "val.json")
    save_split(test, splits_dir / "test.json")

    print(f"已保存 JSON 划分文件到: {splits_dir}")


if __name__ == "__main__":
    random.seed(42)
    main()

