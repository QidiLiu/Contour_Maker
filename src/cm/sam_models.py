"""SAM 系列 (EdgeSAM / EfficientSAM) 的 box-prompt 推理适配层。

统一接口: 给定灰度原图 + 检测框(原图像素坐标) -> 二值 mask(原图尺寸)。
内部把图 resize 到 1024x1024 (SAM 约定), 框同步缩放, 输出再 resize 回原尺寸。
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
# 顺序敏感: stubs 必须最先, 供 EdgeSAM 的 mmdet/mmengine 可选依赖
for p in (ROOT / "third_party" / "stubs", ROOT / "third_party" / "EdgeSAM"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
_EFF = ROOT / "third_party" / "EfficientSAM"
if str(_EFF) not in sys.path:
    sys.path.append(str(_EFF))

WEIGHTS = ROOT / "weights"
SAM_SIZE = 1024


def _to_rgb_tensor(gray: np.ndarray, size: int = SAM_SIZE) -> torch.Tensor:
    """灰度 uint8 -> (1,3,size,size) float32 [0,1]。"""
    img = cv2.resize(gray, (size, size), interpolation=cv2.INTER_LINEAR)
    t = torch.from_numpy(img).float().div_(255.0)[None, None]      # (1,1,S,S)
    return t.repeat(1, 3, 1, 1)                                    # (1,3,S,S)


def _scale_box(box, h: int, w: int, size: int = SAM_SIZE) -> np.ndarray:
    x1, y1, x2, y2 = box
    return np.array([x1 * size / w, y1 * size / h, x2 * size / w, y2 * size / h], np.float32)


def _mask_back(mask: np.ndarray, h: int, w: int) -> np.ndarray:
    m = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    return (m > 0).astype(np.uint8)


# ---------------------------------------------------------------- EdgeSAM
class EdgeSAMWrapper:
    name = "EdgeSAM (RepViT-Tiny)"
    n_params_m = 9.58

    def __init__(self, ckpt: Path | None = None, device: str = "cuda"):
        from edge_sam import sam_model_registry  # noqa: WPS433
        ckpt = ckpt or (WEIGHTS / "edge_sam.pth")
        self.device = torch.device(device)
        self.model = sam_model_registry["edge_sam"](checkpoint=str(ckpt)).to(self.device).eval()

    @torch.no_grad()
    def encode(self, gray: np.ndarray) -> torch.Tensor:
        return self.model.image_encoder(_to_rgb_tensor(gray).to(self.device))

    @torch.no_grad()
    def decode(self, emb: torch.Tensor, box, h: int, w: int) -> np.ndarray:
        b = torch.as_tensor(_scale_box(box, h, w), device=self.device)[None]
        sparse, dense = self.model.prompt_encoder(points=None, boxes=b, masks=None)
        low_res, _ = self.model.mask_decoder(
            image_embeddings=emb, image_pe=self.model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense,
            num_multimask_outputs=1)
        full = self.model.postprocess_masks(low_res, (SAM_SIZE, SAM_SIZE), (h, w))
        return (full[0, 0] > 0).cpu().numpy().astype(np.uint8)

    @torch.no_grad()
    def predict(self, gray: np.ndarray, box) -> np.ndarray:
        h, w = gray.shape[:2]
        return _mask_back(self.decode(self.encode(gray), box, h, w), h, w)


# ---------------------------------------------------------------- EfficientSAM
class EfficientSAMWrapper:
    n_params_m = 10.22

    def __init__(self, variant: str = "vitt", ckpt: Path | None = None, device: str = "cuda"):
        from efficient_sam.build_efficient_sam import build_efficient_sam  # noqa: WPS433
        cfg = {"vitt": (192, 3, "efficient_sam_vitt.pt", "EfficientSAM (ViT-Tiny)"),
               "vits": (384, 6, "efficient_sam_vits.pt", "EfficientSAM (ViT-Small)")}[variant]
        self.dim, self.heads, fname, self.name = cfg
        ckpt = ckpt or (WEIGHTS / fname)
        self.device = torch.device(device)
        self.model = build_efficient_sam(
            encoder_patch_embed_dim=self.dim, encoder_num_heads=self.heads,
            checkpoint=str(ckpt)).to(self.device).eval()
        self.n_params_m = sum(p.numel() for p in self.model.parameters()) / 1e6

    @torch.no_grad()
    def encode(self, gray: np.ndarray) -> torch.Tensor:
        return self.model.get_image_embeddings(_to_rgb_tensor(gray).to(self.device))

    @torch.no_grad()
    def decode(self, emb: torch.Tensor, box, h: int, w: int) -> np.ndarray:
        # 注意: EfficientSAM 内部会做 pts * img_size / input_w 的缩放,
        # 因此这里必须传「原图像素坐标」+ 原图尺寸, 不能预先缩放到 1024。
        x1, y1, x2, y2 = [float(v) for v in box]
        pts = torch.tensor([[[[x1, y1], [x2, y2]]]], device=self.device, dtype=torch.float)
        labels = torch.tensor([[[1, 1]]], device=self.device, dtype=torch.int)
        low_res, _ = self.model.predict_masks(
            image_embeddings=emb, batched_points=pts, batched_point_labels=labels,
            multimask_output=False, input_h=h, input_w=w, output_h=h, output_w=w)
        m = low_res[0, 0, 0].cpu().numpy()
        if m.shape != (h, w):
            m = cv2.resize(m.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
        return (m > 0).astype(np.uint8)

    @torch.no_grad()
    def predict(self, gray: np.ndarray, box) -> np.ndarray:
        h, w = gray.shape[:2]
        return _mask_back(self.decode(self.encode(gray), box, h, w), h, w)


def build_sam(variant: str, device: str = "cuda"):
    if variant == "edgesam":
        return EdgeSAMWrapper(device=device)
    if variant in ("efficientvit-t", "efficientsam", "efficient_sam"):
        return EfficientSAMWrapper("vitt", device=device)
    if variant in ("efficientvit-s", "efficientsam_s"):
        return EfficientSAMWrapper("vits", device=device)
    raise ValueError(f"未知 SAM 变体: {variant}")
