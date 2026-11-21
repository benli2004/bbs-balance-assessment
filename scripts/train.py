from __future__ import annotations

"""
训练脚本：使用完整的训练循环，支持动作和评分预测。
"""

import argparse
import sys
from pathlib import Path
from typing import Dict

import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from tqdm import tqdm

# 确保可以导入 src 包
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from data_preprocessing.dataset import VideoDataset
from models.bbsaqa_3dcnn import BBSAQA3DCNN


def load_yaml(path: str | Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    action_criterion: nn.Module,
    score_criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: GradScaler,
) -> tuple[float, float, float]:
    model.train()
    total_loss = 0.0
    correct_actions = 0
    correct_scores = 0
    total_samples = 0

    for batch_idx, batch in enumerate(tqdm(loader, desc="Training", leave=False, mininterval=5)):
        videos = batch["video"].to(device, non_blocking=True)
        action_ids = batch["action_id"].to(device, non_blocking=True)
        scores = batch["score"].to(device, non_blocking=True)

        action_indices = action_ids - 1  # 标签对齐：如果标签是 1~7，转为 0~6

        optimizer.zero_grad()

        with autocast(device_type='cuda' if device.type != 'cpu' else 'cpu'):
            outputs = model(videos)

            # 计算动作损失 (Action Loss)
            loss_action = action_criterion(outputs["action_logits"], action_indices)

            # 提取当前真实动作对应的得分预测
            batch_idx = torch.arange(videos.size(0)).to(device)
            pred_scores_for_current_action = outputs["action_scores"][batch_idx, action_indices]

            loss_score = score_criterion(pred_scores_for_current_action, scores)
            loss = loss_action + loss_score

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # 计算准确率
        with torch.no_grad():
            act_pred = torch.argmax(outputs["action_logits"], dim=1)
            correct_actions += (act_pred == action_indices).sum().item()

            score_pred = torch.argmax(pred_scores_for_current_action, dim=1)
            correct_scores += (score_pred == scores).sum().item()

            total_samples += videos.size(0)
            total_loss += loss.item()

    return total_loss / len(loader), correct_actions / total_samples, correct_scores / total_samples


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    action_criterion: nn.Module,
    score_criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
    correct_actions = 0
    correct_scores = 0
    total_samples = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Validating", leave=False):
            videos = batch["video"].to(device, non_blocking=True)
            action_ids = batch["action_id"].to(device, non_blocking=True)
            scores = batch["score"].to(device, non_blocking=True)

            action_indices = action_ids - 1

            with autocast(device_type='cuda' if device.type != 'cpu' else 'cpu'):
                outputs = model(videos)

                loss_action = action_criterion(outputs["action_logits"], action_indices)

                batch_idx = torch.arange(videos.size(0)).to(device)
                pred_scores_for_current_action = outputs["action_scores"][batch_idx, action_indices]

                loss_score = score_criterion(pred_scores_for_current_action, scores)
                loss = loss_action + loss_score

            act_pred = torch.argmax(outputs["action_logits"], dim=1)
            correct_actions += (act_pred == action_indices).sum().item()

            score_pred = torch.argmax(pred_scores_for_current_action, dim=1)
            correct_scores += (score_pred == scores).sum().item()

            total_samples += videos.size(0)
            total_loss += loss.item()

    return total_loss / len(loader), correct_actions / total_samples, correct_scores / total_samples


def main() -> None:
    parser = argparse.ArgumentParser(description="BBS AQA 模型训练脚本")
    parser.add_argument(
        "--data_config",
        type=str,
        default="configs/data_config.yaml",
        help="数据配置文件路径",
    )
    parser.add_argument(
        "--model_config",
        type=str,
        default="configs/model_config.yaml",
        help="模型配置文件路径",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu", "auto"],
        help="训练设备：默认 cuda（强制 GPU）。auto 则自动选择。",
    )
    parser.add_argument(
        "--decode",
        type=str,
        default="cpu",
        choices=["cpu", "gpu", "auto"],
        help="视频解码设备：默认 cpu。gpu 需要 OpenCV 带 CUDA 支持。",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="DataLoader 并行 worker 数",
    )
    args = parser.parse_args()

    data_cfg = load_yaml(args.data_config)["data"]
    model_cfg = load_yaml(args.model_config)["model"]
    train_cfg = load_yaml(args.model_config)["training"]

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            print("⚠️  未检测到 CUDA/GPU，将自动切换到 CPU 模式")
            device = torch.device("cpu")

    print(f"使用设备: {device}")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    # 数据集与 DataLoader
    root_dir = str(data_cfg["root_dir"])
    train_split = str(data_cfg["train_split"])
    val_split = str(data_cfg["val_split"])
    target_frames = int(data_cfg["target_frames"])
    resize_h, resize_w = data_cfg["resize"]
    resize_h, resize_w = int(resize_h), int(resize_w)

    # 打印训练配置
    print(f"\n{'='*60}")
    print("训练配置")
    print(f"{'='*60}")
    print(f"  设备: {device}")
    print(f"  Batch Size: {train_cfg['batch_size']}")
    print(f"  Learning Rate: {train_cfg['learning_rate']}")
    print(f"  Max Epochs: {train_cfg['max_epochs']}")
    print(f"  目标帧数: {target_frames}")
    print(f"  图像尺寸: {resize_h}x{resize_w}")
    print(f"  数据增强: {train_cfg.get('augmentation', 'N/A')}")
    print(f"{'='*60}\n")

    train_dataset = VideoDataset(
        json_path=train_split,
        root_dir=root_dir,
        target_frames=target_frames,
        resize=(resize_h, resize_w),
        use_augmentation=True,
        decode=args.decode,
    )
    val_dataset = VideoDataset(
        json_path=val_split,
        root_dir=root_dir,
        target_frames=target_frames,
        resize=(resize_h, resize_w),
        use_augmentation=False,
        decode=args.decode,
    )

    # GPU 解码时使用单进程加载，避免多进程问题
    # Windows 上必须使用单进程，否则多进程打开视频文件会出错
    import sys
    if sys.platform == "win32":
        num_workers = 0
    elif args.decode in {"gpu", "auto"}:
        num_workers = 0
    else:
        num_workers = args.num_workers

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
    )

    # 模型
    model = BBSAQA3DCNN(
        num_actions=model_cfg["num_actions"],
        num_scores=model_cfg["num_scores"],
    ).to(device)

    # 损失函数和优化器
    action_criterion = nn.CrossEntropyLoss()
    score_criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["learning_rate"]),
    )

    max_epochs = int(train_cfg["max_epochs"])
    checkpoints_dir = Path("checkpoints")
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    scaler = GradScaler(device=device.type, enabled=(device.type == "cuda"))

    for epoch in range(1, max_epochs + 1):
        print(f"\n{'='*60}")
        print(f"Epoch [{epoch}/{max_epochs}]")
        print(f"{'='*60}")
        
        train_loss, train_act_acc, train_score_acc = train_one_epoch(
            model, train_loader, action_criterion, score_criterion, optimizer, device, scaler
        )
        val_loss, val_act_acc, val_score_acc = evaluate(
            model, val_loader, action_criterion, score_criterion, device
        )

        print(f"\n  Train - Loss: {train_loss:.4f} | Action Acc: {train_act_acc:.2%} | Score Acc: {train_score_acc:.2%}")
        print(f"  Val   - Loss: {val_loss:.4f} | Action Acc: {val_act_acc:.2%} | Score Acc: {val_score_acc:.2%}")

    ckpt_path = checkpoints_dir / "bbsaqa3dcnn_last.pth"
    torch.save(model.state_dict(), ckpt_path)
    print(f"训练结束，已保存模型到: {ckpt_path}")


if __name__ == "__main__":
    main()
