"""contour-refiner 模型。

输入: (B, P, S, S) —— B 个样本, 每个样本 P=64 张 32x32 灰度 ROI (已 Z-score), 每张 ROI
      属于 1 个粗糙轮廓点; 另有每个点的几何辅助特征 (相对检测框位置、切向角)。
结构: 共享 CNN backbone (逐 ROI 提特征) -> 点间 Transformer 上下文聚合 -> 逐点回归头。
输出: (B, P, 2) —— 每个 ROI 内精准轮廓点的归一化坐标 (-1~1)。
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------- backbone
class DepthwiseSepBlock(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int = 1):
        super().__init__()
        self.dw = nn.Conv2d(cin, cin, 3, stride, 1, groups=cin, bias=False)
        self.pw = nn.Conv2d(cin, cout, 1, bias=False)
        self.bn = nn.BatchNorm2d(cout)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.pw(self.dw(x))))


class PatchCNN(nn.Module):
    """共享权重的小型 CNN: 32x32 灰度 ROI -> feat_dim 向量。"""

    def __init__(self, feat_dim: int = 128, in_ch: int = 1):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, 1, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.blocks = nn.Sequential(
            DepthwiseSepBlock(32, 32, stride=2),      # 16
            DepthwiseSepBlock(32, 64, stride=2),      # 8
            DepthwiseSepBlock(64, 128, stride=2),     # 4
            DepthwiseSepBlock(128, feat_dim, stride=2),  # 2
            nn.AdaptiveAvgPool2d(1),
        )
        self.out_dim = feat_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, 1, S, S) -> (N, feat_dim)
        return self.blocks(self.stem(x)).flatten(1)


# ---------------------------------------------------------------- refiner
class ContourRefiner(nn.Module):
    def __init__(
        self,
        n_points: int = 64,
        roi_size: int = 32,
        feat_dim: int = 128,
        n_heads: int = 4,
        n_tx_layers: int = 2,
        dropout: float = 0.0,
        backbone: str = "cnn_transformer",
        direct_coords: bool = True,
    ):
        super().__init__()
        self.n_points = n_points
        self.roi_size = roi_size
        self.feat_dim = feat_dim
        self.backbone_name = backbone
        self.direct_coords = direct_coords

        self.cnn = PatchCNN(feat_dim=feat_dim)
        # 每个点的几何编码: 窗口中心在图内位置(2) + 粗糙点在窗口内位置(2) + aux(3) = 7 维
        self.geo = nn.Sequential(nn.Linear(7, feat_dim), nn.ReLU(inplace=True), nn.Linear(feat_dim, feat_dim))
        self.ctx = nn.Parameter(torch.zeros(1, n_points, feat_dim))
        nn.init.trunc_normal_(self.ctx, std=0.02)

        self.norm_in = nn.LayerNorm(feat_dim)
        if n_tx_layers > 0 and backbone != "cnn_only":
            layer = nn.TransformerEncoderLayer(
                d_model=feat_dim, nhead=n_heads, dim_feedforward=feat_dim * 4,
                dropout=dropout, batch_first=True, activation="gelu", norm_first=True)
            self.transformer = nn.TransformerEncoder(layer, num_layers=n_tx_layers)
        else:
            self.transformer = None

        self.head = nn.Sequential(
            nn.LayerNorm(feat_dim), nn.Linear(feat_dim, feat_dim), nn.GELU(),
            nn.Linear(feat_dim, 2))

    def forward(self, patches: torch.Tensor, rough_norm: torch.Tensor, aux: torch.Tensor,
                frame_centers: torch.Tensor | None = None) -> torch.Tensor:
        """
        patches: (B, P, S, S); rough_norm: (B, P, 2) in [-1,1]; aux: (B, P, 3);
        frame_centers: (B, P, 2) 窗口中心在图内的归一化坐标 (0~1), 可选。
        returns: (B, P, 2) 预测的精准点归一化坐标 (-1~1)
        """
        B, P, S, _ = patches.shape
        feat = self.cnn(patches.reshape(B * P, 1, S, S)).reshape(B, P, self.feat_dim)

        if frame_centers is None:
            frame_centers = torch.zeros(B, P, 2, device=patches.device, dtype=patches.dtype)
        geo_in = torch.cat([frame_centers, rough_norm, aux], dim=-1)
        feat = feat + self.geo(geo_in) + self.ctx[:, :P]
        feat = self.norm_in(feat)

        if self.transformer is not None:
            feat = self.transformer(feat)

        delta = self.head(feat)
        if self.direct_coords:
            return torch.tanh(delta)
        return rough_norm + delta


# ---------------------------------------------------------------- losses
def laplacian_smooth_loss(pts: torch.Tensor) -> torch.Tensor:
    """闭合轮廓二阶差分平滑项 (抑制点抖动)。"""
    prev_p = torch.roll(pts, 1, dims=1)
    next_p = torch.roll(pts, -1, dims=1)
    return ((prev_p - 2 * pts + next_p) ** 2).sum(-1).mean()


def build_loss(pred: torch.Tensor, target: torch.Tensor, cfg_loss: str = "smooth_l1",
               lambda_smooth: float = 0.0) -> tuple[torch.Tensor, dict]:
    if cfg_loss == "l1":
        main = F.l1_loss(pred, target)
    elif cfg_loss == "l2":
        main = F.mse_loss(pred, target)
    else:
        main = F.smooth_l1_loss(pred, target, beta=0.02)
    logs = {"loss_main": float(main.detach())}
    total = main
    if lambda_smooth > 0:
        sm = laplacian_smooth_loss(pred)
        logs["loss_smooth"] = float(sm.detach())
        total = total + lambda_smooth * sm
    return total, logs


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def build_refiner(cfg) -> ContourRefiner:
    return ContourRefiner(
        n_points=cfg.n_points, roi_size=cfg.roi_size, feat_dim=cfg.feat_dim,
        n_heads=cfg.n_heads, n_tx_layers=cfg.n_tx_layers, dropout=cfg.dropout,
        backbone=cfg.backbone)


_ = math  # keep import for future use
