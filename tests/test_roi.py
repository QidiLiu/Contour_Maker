import numpy as np

from common.roi import (
    contour_to_mask,
    letterbox,
    mask_to_contour,
    paste_roi_mask,
    resize_roi,
    unletterbox_boxes,
)


def test_letterbox_roundtrip():
    img = np.zeros((100, 200), np.uint8)
    canvas, ratio, pad = letterbox(img, (256, 256))
    assert canvas.shape == (256, 256)
    # A box that covers the full original image maps back to the original frame.
    lb = np.array([[pad[0], pad[1], pad[0] + 200 * ratio, pad[1] + 100 * ratio]], np.float32)
    orig = unletterbox_boxes(lb, (100, 200), ratio, pad)
    assert np.allclose(orig[0], [0, 0, 200, 100], atol=1.0)


def test_resize_and_paste_roundtrip():
    roi = np.zeros((40, 80), np.uint8)
    roi[10:30, 20:60] = 255
    small = resize_roi(roi, 64, square_pad=True)
    full = paste_roi_mask(small, (40, 80), 64, (0, 0, 80, 40), (100, 100))
    assert full.shape == (100, 100)
    ys, xs = np.nonzero(full)
    assert ys.min() >= 9 and ys.max() <= 31
    assert xs.min() >= 19 and xs.max() <= 61


def test_contour_mask_roundtrip():
    mask = np.zeros((64, 64), np.uint8)
    mask[10:40, 12:50] = 1
    contour = mask_to_contour(mask)
    assert contour is not None and len(contour) >= 3
    rebuilt = contour_to_mask(contour, mask.shape)
    assert rebuilt.sum() > 0
    assert abs(rebuilt.sum() - mask.sum()) / mask.sum() < 0.2
