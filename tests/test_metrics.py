import numpy as np

from common.metrics import (
    area_error,
    boundary_f1,
    dice,
    greedy_match,
    hd95,
    iou,
    prf,
)


def test_dice_iou_identity():
    m = np.zeros((32, 32), np.uint8)
    m[8:20, 8:20] = 1
    assert dice(m, m) == 1.0
    assert iou(m, m) == 1.0
    assert area_error(m, m) == 0.0


def test_dice_iou_disjoint():
    a = np.zeros((16, 16), np.uint8)
    b = np.zeros((16, 16), np.uint8)
    a[:8, :8] = 1
    b[8:, 8:] = 1
    assert dice(a, b) == 0.0
    assert iou(a, b) == 0.0


def test_hd95_and_boundary_f1_shifted():
    a = np.zeros((64, 64), np.uint8)
    b = np.zeros((64, 64), np.uint8)
    a[16:48, 16:48] = 1
    b[18:50, 16:48] = 1  # shifted by 2 px
    assert hd95(a, b) >= 1.9
    assert 0.0 < boundary_f1(a, b, tol=2.0) <= 1.0


def test_greedy_match_counts():
    gt = [np.zeros((20, 20), np.uint8) for _ in range(2)]
    gt[0][2:10, 2:10] = 1
    gt[1][12:18, 12:18] = 1
    pred = [gt[0].copy(), np.ones((20, 20), np.uint8)]
    pairs, miss, fp = greedy_match(pred, gt, 0.5)
    assert len(pairs) == 1
    assert miss == 1
    assert fp == 1


def test_prf():
    r = prf(4, 1, 2)
    assert abs(r["precision"] - 0.8) < 1e-9
    assert abs(r["recall"] - 4 / 6) < 1e-9
    assert 0 < r["f1"] < 1
