"""LiteMedSAM / Swin-LiteMedSAM 的 box-prompt 适配层。

统一接口 (与 sam_models.py 一致):
    wrapper.encode(gray) -> 特征
    wrapper.decode(feat, box, H, W) -> 二值 mask (原图尺寸)
    wrapper.predict(gray, box) -> mask

LiteMedSAM 预处理 (与官方 CVPR24_LiteMedSAM_infer.py 一致):
    1. 复制成 3 通道
    2. 长边 resize 到 256 (INTER_AREA)
    3. 每图 min-max 归一化到 [0,1]
    4. 右下角补零到 256×256
    5. encoder -> (1,256,64,64); prompt encoder 用缩放到 256 空间的 box
    6. mask_decoder -> 低分辨率 logits; postprocess (裁剪 -> 双线性上采样) -> 原图
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
_LITE = ROOT / "third_party" / "LiteMedSAM"
if str(_LITE) not in sys.path:
    sys.path.insert(0, str(_LITE))

WEIGHTS = ROOT / "weights"
IMG_SIZE = 256


class _MedSamLite(torch.nn.Module):
    def __init__(self, image_encoder, mask_decoder, prompt_encoder):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder

    def postprocess_masks(self, masks, new_size, original_size):
        masks = masks[..., : new_size[0], : new_size[1]]
        masks = F.interpolate(masks, size=(original_size[0], original_size[1]),
                              mode="bilinear", align_corners=False)
        return masks


def build_lite_medsam(ckpt: Path | None = None, variant: str = "lite"):
    """构建 LiteMedSAM (TinyViT 编码器)。variant='swin' 时改用 Swin 编码器 + Swin 版 decoder。"""
    from segment_anything.modeling import MaskDecoder, PromptEncoder, TwoWayTransformer
    from tiny_vit_sam import TinyViT
    ckpt = Path(ckpt or (WEIGHTS / "lite_medsam.pth"))
    state = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]

    if variant == "lite":
        encoder = TinyViT(img_size=IMG_SIZE, in_chans=3,
                          embed_dims=[64, 128, 160, 320], depths=[2, 2, 6, 2],
                          num_heads=[2, 4, 5, 10], window_sizes=[7, 7, 14, 7],
                          mlp_ratio=4., drop_rate=0., drop_path_rate=0.0,
                          use_checkpoint=False, mbconv_expand_ratio=4.0,
                          local_conv_size=3, layer_lr_decay=0.8)
        prompt_encoder = PromptEncoder(embed_dim=256, image_embedding_size=(64, 64),
                                       input_image_size=(IMG_SIZE, IMG_SIZE), mask_in_chans=16)
        mask_decoder = MaskDecoder(num_multimask_outputs=3,
                                   transformer=TwoWayTransformer(depth=2, embedding_dim=256,
                                                                 mlp_dim=2048, num_heads=8),
                                   transformer_dim=256, iou_head_depth=3,
                                   iou_head_hidden_dim=256)
        model = _MedSamLite(encoder, mask_decoder, prompt_encoder)
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"[LiteMedSAM] 权重载入: missing={len(missing)} unexpected={len(unexpected)}")
        return model

    # ---- Swin 编码器变体 (官方 Swin-LiteMedSAM checkpoint 不可得, 见报告 §4.9)
    swin_dir = ROOT / "third_party" / "Swin_LiteMedSAM"
    if str(swin_dir) not in sys.path:
        sys.path.insert(0, str(swin_dir))
    from models import MaskDecoder_Prompt, PromptEncoder as SwinPE  # noqa: WPS433
    from models import TwoWayTransformer as SwinTWT  # noqa: WPS433
    from models.swin import SwinTransformer  # noqa: WPS433

    encoder = SwinTransformer()
    prompt_encoder = SwinPE(embed_dim=256, image_embedding_size=(64, 64),
                            input_image_size=(IMG_SIZE, IMG_SIZE), mask_in_chans=16)
    mask_decoder = MaskDecoder_Prompt(num_multimask_outputs=3,
                                      transformer=SwinTWT(depth=2, embedding_dim=256,
                                                          mlp_dim=2048, num_heads=8),
                                      transformer_dim=256, iou_head_depth=3,
                                      iou_head_hidden_dim=256)
    model = _MedSamLite(encoder, mask_decoder, prompt_encoder)
    # 只有 patch_embed 等少数层能从 LiteMedSAM 迁移 (18 个键), decoder 键名体系不同
    enc_state = {k[len("image_encoder."):]: v for k, v in state.items()
                 if k.startswith("image_encoder.")}
    sd = model.image_encoder.state_dict()
    compat = {k: v for k, v in enc_state.items() if k in sd and sd[k].shape == v.shape}
    model.image_encoder.load_state_dict(compat, strict=False)
    print(f"[Swin-LiteMedSAM] encoder 可迁移键 {len(compat)}/{len(sd)}; "
          f"decoder 需在本数据集训练")
    return model


def preprocess(gray: np.ndarray) -> tuple[torch.Tensor, float, int, int]:
    """返回 (tensor(1,3,256,256), ratio, new_h, new_w)。"""
    img3 = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR).astype(np.float32)
    h, w = img3.shape[:2]
    ratio = IMG_SIZE / max(h, w)
    new_h, new_w = int(round(h * ratio)), int(round(w * ratio))
    resized = cv2.resize(img3, (new_w, new_h), interpolation=cv2.INTER_AREA)
    lo, hi = float(resized.min()), float(resized.max())
    norm = (resized - lo) / max(hi - lo, 1e-8)
    padded = np.zeros((IMG_SIZE, IMG_SIZE, 3), np.float32)
    padded[:new_h, :new_w] = norm
    t = torch.from_numpy(padded).permute(2, 0, 1).unsqueeze(0).float()
    return t, ratio, new_h, new_w


class LiteMedSAMWrapper:
    def __init__(self, variant: str = "lite", ckpt: Path | None = None, device: str = "cuda",
                 head_ckpt: Path | None = None):
        self.device = torch.device(device)
        self.variant = variant
        self.model = build_lite_medsam(ckpt, variant=variant).to(self.device).eval()
        self.name = "LiteMedSAM (TinyViT)" if variant == "lite" else "Swin-LiteMedSAM (Swin-T)"
        self.n_params_m = sum(p.numel() for p in self.model.parameters()) / 1e6
        # Swin 变体的 prompt encoder + mask decoder 需在本数据集训练 (官方权重不可得)
        head = Path(head_ckpt) if head_ckpt else (WEIGHTS / "sam_finetuned" /
                                                  "swin_litemedsam_head.pt")
        if variant == "swin" and head.exists():
            sd = torch.load(str(head), map_location=self.device, weights_only=False)
            m1, u1 = self.model.prompt_encoder.load_state_dict(sd["prompt_encoder"], strict=False)
            m2, u2 = self.model.mask_decoder.load_state_dict(sd["mask_decoder"], strict=False)
            print(f"[Swin-LiteMedSAM] 载入训练好的 head (epoch={sd.get('epoch')} "
                  f"val_dice256={sd.get('val_dice256'):.4f}) missing={len(m1)+len(m2)}")
        elif variant == "swin":
            print("[Swin-LiteMedSAM] 未找到训练好的 head, decoder 仍为随机初始化")

    @torch.no_grad()
    def encode(self, gray: np.ndarray) -> dict:
        t, ratio, nh, nw = preprocess(gray)
        out = self.model.image_encoder(t.to(self.device))
        if isinstance(out, tuple):        # Swin 编码器返回 (embedding, [4 个多尺度特征])
            emb, fs = out
            fs = [f for f in fs]          # list[Tensor] -> 缓存时 stack
        else:
            emb, fs = out, None
        return dict(emb=emb, fs=fs, ratio=ratio, new_size=(nh, nw), orig=gray.shape[:2])

    @torch.no_grad()
    def decode(self, feat: dict, box, h: int, w: int) -> np.ndarray:
        ratio = feat["ratio"]
        b = torch.tensor([[float(v) * ratio for v in box]], dtype=torch.float,
                         device=self.device)[None]
        try:   # Swin-LiteMedSAM 的 PromptEncoder.forward 需要额外的 tokens 形参 (实际未使用)
            sparse, dense = self.model.prompt_encoder(points=None, boxes=b, masks=None,
                                                      tokens=None)
        except TypeError:
            sparse, dense = self.model.prompt_encoder(points=None, boxes=b, masks=None)
        if feat.get("fs") is not None:    # Swin-LiteMedSAM: decoder 额外需要 encoder 特征 fs
            low_res, _ = self.model.mask_decoder(
                feat["fs"], image_embeddings=feat["emb"],
                image_pe=self.model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense,
                multimask_output=False)
        else:
            low_res, _ = self.model.mask_decoder(
                image_embeddings=feat["emb"], image_pe=self.model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense,
                multimask_output=False)
        full = self.model.postprocess_masks(low_res, feat["new_size"], (h, w))
        return (torch.sigmoid(full)[0, 0] > 0.5).cpu().numpy().astype(np.uint8)

    @torch.no_grad()
    def predict(self, gray: np.ndarray, box) -> np.ndarray:
        h, w = gray.shape[:2]
        return self.decode(self.encode(gray), box, h, w)


def build_lite(variant: str = "lite", device: str = "cuda"):
    return LiteMedSAMWrapper(variant=variant, device=device)
