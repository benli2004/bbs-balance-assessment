from pathlib import Path
from typing import List, Tuple


def list_video_files(root_dir: str) -> List[Path]:
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"数据根目录不存在: {root}")

    video_extensions = {".mp4", ".avi", ".mov", ".mkv"}
    files: List[Path] = []
    for p in root.rglob("*"):
        if p.suffix.lower() in video_extensions:
            files.append(p)
    return files


def parse_action_and_score(path: Path) -> Tuple[int, int]:
    try:
        score = int(path.parent.name)
        action_str = path.parent.parent.name  # e.g. "action3"
        if not action_str.startswith("action"):
            raise ValueError
        action_id = int(action_str.replace("action", ""))
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"无法从路径解析动作和分数: {path}") from exc

    return action_id, score


