import numpy as np

from common.instances import gt_instances, read_sidecar, write_sidecar


def test_binary_mask_components():
    mask = np.zeros((40, 40), np.uint8)
    mask[2:12, 2:12] = 255
    mask[20:30, 20:30] = 255
    insts = gt_instances(mask, default_class_id=3, min_area=10)
    assert len(insts) == 2
    assert all(i.class_id == 3 for i in insts)
    assert sum(i.area for i in insts) == 200


def test_instance_id_mask_with_sidecar(tmp_path):
    idmap = np.zeros((30, 30), np.uint8)
    idmap[2:10, 2:10] = 1
    idmap[12:22, 12:22] = 2
    sidecar = tmp_path / "u.csv"
    write_sidecar(sidecar, {1: 0, 2: 5})
    assert read_sidecar(sidecar) == {1: 0, 2: 5}

    insts = gt_instances(idmap, sidecar=sidecar, min_area=10)
    assert len(insts) == 2
    assert {i.class_id for i in insts} == {0, 5}


def test_instance_id_splits_components(tmp_path):
    idmap = np.zeros((40, 40), np.uint8)
    idmap[2:8, 2:8] = 1
    idmap[20:26, 20:26] = 1  # same id, second component
    sidecar = tmp_path / "s.csv"
    write_sidecar(sidecar, {1: 7})
    insts = gt_instances(idmap, sidecar=sidecar, min_area=5)
    assert len(insts) == 2
    assert all(i.class_id == 7 for i in insts)
