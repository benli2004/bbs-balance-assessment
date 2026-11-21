# Dataset

The dataset is **not** included in this repository.

## Why

The recordings show identifiable people performing balance tasks. They were collected for academic use with the participants' consent for this project only, so neither the raw videos nor the derived clips are redistributed here.

## Format

Videos are organised by task and score:

```
data/raw/
  action1/0/  action1/1/  ...  action1/4/
  action2/0/  ...
  action7/0/  ...
```

`action1` … `action7` are the seven supported BBS tasks (official tasks 1, 2, 3, 4, 9, 12, 14). The numeric folder is the score, 0–4.

Each split entry follows this schema:

```json
[
  {
    "video_path": "data/raw/action2/3/sample_001.mp4",
    "action_id": 2,
    "score": 3
  }
]
```

Generate the splits from your own recordings with:

```bash
python scripts/create_splits.py
```

## Recording protocol

Standardised conditions were used to keep the input distribution consistent: 720p or higher, ~30 fps, whole body visible for the entire clip, fixed frontal camera position, one task per video.

## Checkpoints

The trained weights (`checkpoints/*.pth`) are excluded by `.gitignore` because they are large binaries. Retraining with `scripts/train.py` reproduces the reported configuration; the best setting is documented in the top-level README.
