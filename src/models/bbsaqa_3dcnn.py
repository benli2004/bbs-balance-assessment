from __future__ import annotations


from typing import Dict

import torch
import torch.nn as nn
from torch import Tensor


class BBSAQA3DCNN(nn.Module):


    def __init__(self, num_actions: int = 7, num_scores: int = 5) -> None:
        
        super().__init__()

        self.num_actions = num_actions
        self.num_scores = num_scores

        
        self.features = nn.Sequential(
            # 3D卷积层
            nn.Conv3d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),

            nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),

            nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),

            nn.Conv3d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.Conv3d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),

            nn.Conv3d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((1, 1, 1)),

            # 全连接层
            nn.Flatten(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
        )

     
        self.action_head = nn.Linear(512, num_actions)

   
        self.score_head = nn.Linear(512, num_actions * num_scores)

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        feat = self.features(x)

        action_logits = self.action_head(feat)

        # per action score
        score_logits = self.score_head(feat)
        action_scores = score_logits.view(-1, self.num_actions, self.num_scores)

        return {
            "action_logits": action_logits,
            "action_scores": action_scores
        }
