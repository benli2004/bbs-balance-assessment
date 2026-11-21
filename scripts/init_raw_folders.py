from __future__ import annotations


from pathlib import Path


def main() -> None:
    project_root = Path(".").resolve()
    raw_root = project_root / "data" / "raw"

    for action_id in range(1, 8):  # 动作 1..7
        for score in range(0, 5):  # 评分 0..4
            folder = raw_root / f"action{action_id}" / str(score)
            folder.mkdir(parents=True, exist_ok=True)

    print(f"已创建/检查文件夹: {raw_root}")
    print("现在你可以把视频放到例如 data/raw/action3/4/your_video.mp4 里面。")


if __name__ == "__main__":
    main()

