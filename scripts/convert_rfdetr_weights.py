#!/usr/bin/env python3
"""Convert the HF ``Roboflow/rf-detr-seg-nano`` safetensors to rfdetr's checkpoint.

The upstream ``rfdetr`` package downloads ``rf-detr-seg-nano.pt`` from Google
Storage, which is often very slow.  The Hugging Face mirror exposes a
``model.safetensors`` with exactly the same state-dict keys/shapes (only
``_kp_active_mask`` is missing), so we can build the ``{"model": ...}``
checkpoint locally.

Usage::

    python scripts/convert_rfdetr_weights.py
    python scripts/convert_rfdetr_weights.py --safetensors /path/model.safetensors
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from rfdetr import RFDETRSegNano
from safetensors.torch import load_file

DEFAULT_URL = "https://hf-mirror.com/Roboflow/rf-detr-seg-nano/resolve/main/model.safetensors"


def download(url: str, dst: Path) -> Path:
    import requests

    dst.parent.mkdir(parents=True, exist_ok=True)
    print(f"[get] {url}")
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0))
        done = 0
        with dst.open("wb") as f:
            for chunk in r.iter_content(1 << 20):
                if not chunk:
                    continue
                f.write(chunk)
                done += len(chunk)
                if total:
                    print(
                        f"\r      {100 * done / total:5.1f}% {done / 1e6:7.1f} MB",
                        end="",
                        flush=True,
                    )
    print()
    return dst


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--safetensors", default=None, help="local file (skips download)")
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--out", default="~/.roboflow/models/rf-detr-seg-nano.pt")
    args = ap.parse_args()

    src = Path(args.safetensors) if args.safetensors else Path("/tmp/rfdetr_seg_nano.safetensors")
    if not src.exists():
        download(args.url, src)

    sd = load_file(str(src))
    model = RFDETRSegNano(pretrain_weights=None)
    want = model.model.model.state_dict()
    missing = set(want) - set(sd)
    extra = set(sd) - set(want)
    shape_bad = [k for k in set(sd) & set(want) if tuple(sd[k].shape) != tuple(want[k].shape)]
    if missing - {"_kp_active_mask"} or extra or shape_bad:
        raise SystemExit(
            f"checkpoint mismatch: missing={sorted(missing)[:5]} extra={sorted(extra)[:5]} "
            f"shape_bad={shape_bad[:5]}"
        )

    mc = model.model_config
    ckpt_args = dict(mc.model_dump())
    ckpt_args.update(
        num_queries=mc.num_queries, group_detr=mc.group_detr, num_classes=90, class_names=None
    )
    out = Path(os.path.expanduser(args.out))
    out.parent.mkdir(parents=True, exist_ok=True)
    import torch

    torch.save({"model": sd, "args": ckpt_args}, out)
    print(f"[done] {out} ({out.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
