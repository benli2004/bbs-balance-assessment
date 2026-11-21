# Automated Berg Balance Scale (BBS) Assessment

A video-based system that scores the Berg Balance Scale automatically. Upload a set of task videos and the system returns a per-task score (0–4) and an overall fall-risk category.

Final-year project, BSc (Hons) Computer Science, Hong Kong Metropolitan University (2025–2026), developed with an industry partner.

## Results

| Model | Task | Validation accuracy |
|---|---|---|
| Stage 1 — rule-based baseline (MediaPipe Pose) | BBS Task 1 (sit-to-stand) | 77.8% |
| Stage 2 — MTL-AQA + 3D CNN | 7 BBS tasks, action classification | 84.78% |
| Stage 2 — MTL-AQA + 3D CNN | 7 BBS tasks, score prediction | 76.09% |

Best configuration: 450 augmented clips, 20 epochs, five-block 3D CNN. Selected from five ablations over dataset size (252 / 450 clips), network depth (4 / 5 blocks) and training length (10 / 20 epochs).

## How it works

```
video ──► frame extraction ──► 3D CNN backbone ──► ┬─► action head (7 classes)
        (16 frames, 224×224)                        └─► score head (7 × 5 logits)
                                                         └─► score for the predicted task

per-task scores ──► aggregation ──► Low / Moderate / High fall risk
```

Stage 1 is an interpretable baseline built on MediaPipe Pose: pose landmarks → kinematic features → state-machine transition detection → score mapping. Stage 2 replaces the handcrafted rules with a multi-task 3D CNN, because fixed thresholds did not generalise across tasks.

## Repository layout

```
configs/                 training and data configuration (YAML)
data/                    dataset format notes (raw videos are not distributed)
scripts/                 dataset split generation and training entry point
src/data_preprocessing/  frame extraction, augmentation, PyTorch dataset
src/models/              BBSAQA3DCNN — shared backbone, action head, score head
web/                     Flask demo: upload task videos, view scores and risk
```

## Getting started

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
pip install -r requirements.txt
```

Place task videos under `data/raw/<action>/<score>/`, then:

```bash
python scripts/create_splits.py    # build train / val / test JSON splits
python scripts/train.py            # train with configs/model_config.yaml
python web/app.py                  # demo app on http://127.0.0.1:5000
```

Training uses mixed precision and expects a CUDA device when available. A trained checkpoint (`.pth`) is intentionally excluded from the repository — see `data/README.md` for the dataset and checkpoint situation.

## Limitations

- **Small, homogeneous dataset.** 295 raw videos from four participants aged 20–30. Not representative of the target clinical population.
- **No clinical validation.** Scores were compared against manual labels by the team, not against physiotherapist ratings on elderly participants.
- **Seven of fourteen BBS tasks.** Tasks 1, 2, 3, 4, 9, 12 and 14 only.
- **Rule-based stage does not scale.** Per-task thresholds had to be hand-tuned; this is what motivated the learning-based stage.
- **Prototype, not a medical device.** Output is a risk category for research and demonstration only.

## Acknowledgements

Supervisor: Dr Kayley Ma. Industry partner supplied the scoring criteria and the clinical framing for the assessment. Team of four; my role was team lead, system architecture, the Stage 1 baseline, the Stage 2 model, dataset construction and evaluation design.
