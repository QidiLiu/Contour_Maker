import numpy as np

from common.ade20k import _extract_instances
from common.config import ADE20K_ADE_ID_TO_CLASS


def test_extract_instances_rgb_encoding():
    # OpenCV BGR: channel 0 = B (0), 1 = G (instance id), 2 = R (thing id).
    ann = np.zeros((40, 40, 3), np.uint8)
    bed = ADE20K_ADE_ID_TO_CLASS[1]
    chair = ADE20K_ADE_ID_TO_CLASS[8]
    ann[2:12, 2:12, 1] = 1
    ann[2:12, 2:12, 2] = 1  # bed instance 1
    ann[20:30, 20:30, 1] = 2
    ann[20:30, 20:30, 2] = 8  # chair instance 2
    ann[35:38, 35:38, 1] = 3
    ann[35:38, 35:38, 2] = 8  # tiny chair, dropped by area

    insts = _extract_instances(ann, min_area=10)
    classes = sorted(cls for cls, _ in insts)
    assert classes == sorted([bed, chair])
    assert all(m.sum() in (100, 100) for _, m in insts)
