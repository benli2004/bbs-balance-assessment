import json
import random
import sys
from pathlib import Path
from dataclasses import asdict, dataclass
from typing import List

PROJECT_ROOT = Path(".").resolve()
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data_preprocessing.video_loader import list_video_files, parse_action_and_score


@dataclass
class SampleInfo:
    video_path: str
    action_id: int
    score: int


def build_samples(root_dir: str) -> List[SampleInfo]:
    root = Path(root_dir).resolve()
    project_root = Path(".").resolve()

    video_paths = list_video_files(root_dir)
    samples: List[SampleInfo] = []

    for p in video_paths:
        action_id, score = parse_action_and_score(p)
        rel_path = p.resolve().relative_to(project_root)
        samples.append(
            SampleInfo(
                video_path=str(rel_path).replace("\\", "/"),
                action_id=action_id,
                score=score,
            )
        )

    return samples


def split_dataset(samples, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    if not samples:
        return [], [], []

    random.shuffle(samples)
    n = len(samples)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train = samples[:n_train]
    val = samples[n_train : n_train + n_val]
    test = samples[n_train + n_val:]

    return train, val, test


def save_split(samples, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = [asdict(s) for s in samples]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# Main
random.seed(42)
raw_dir = "data/raw"

print(f"Scanning: {raw_dir}")
samples = build_samples(raw_dir)
print(f"Found {len(samples)} videos")

train, val, test = split_dataset(samples)
print(f"Split: train={len(train)}, val={len(val)}, test={len(test)}")

save_split(train, Path("data/splits/train.json"))
save_split(val, Path("data/splits/val.json"))
save_split(test, Path("data/splits/test.json"))
print("Done!")
