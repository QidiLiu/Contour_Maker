import json

import cv2
import numpy as np
import pandas as pd

from common.coco_export import export_split
from common.instances import write_sidecar


def test_coco_export_instance_classes(tmp_path):
    unified = tmp_path / "unified"
    (unified / "images").mkdir(parents=True)
    (unified / "masks").mkdir()
    inst = tmp_path / "instances"
    inst.mkdir()

    uid = "demo_1"
    img = np.full((40, 50), 128, np.uint8)
    cv2.imwrite(str(unified / "images" / f"{uid}.png"), img)
    idmap = np.zeros((40, 50), np.uint8)
    idmap[2:12, 2:12] = 1
    idmap[20:30, 20:30] = 2
    cv2.imwrite(str(unified / "masks" / f"{uid}.png"), idmap)
    write_sidecar(inst / f"{uid}.csv", {1: 3, 2: 7})

    df = pd.DataFrame([dict(uid=uid, image=f"images/{uid}.png", mask=f"masks/{uid}.png",
                            class_id=3, dataset="demo", split="train")])
    out = tmp_path / "coco"
    r = export_split(df, "train", out, unified=unified,
                     class_names=["a", "b", "c", "d", "e", "f", "g", "h"],
                     gt_instances_dir=inst)
    assert r["images"] == 1 and r["annotations"] == 2
    data = json.loads((out / "train" / "_annotations.coco.json").read_text())
    assert {a["category_id"] for a in data["annotations"]} == {3, 7}
    assert len(data["categories"]) == 8
    assert data["images"][0]["width"] == 50
