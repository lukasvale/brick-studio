"""Opt-in recovery of bright attached parts from an independent local matte.

The normal model can confuse an almost-white plate with the booth. Apple
Vision gets a full-detail subject crop and contributes only bounded, bright
neutral regions. Original RGB and already accepted alpha are never replaced.
"""
from pathlib import Path
import subprocess
import tempfile
import time

import cv2
import numpy as np
from PIL import Image

WHITE_VERSION = 1


def predict_white_crop(rgb):
    from studio import ROOT, pil
    helper = ROOT / 'native' / 'foreground-guide'
    if not helper.is_file():
        raise RuntimeError('Recover white parts requires the Apple Vision helper on macOS.')
    (ROOT / 'work').mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='white-recovery-', dir=ROOT / 'work') as folder:
        source, target = Path(folder) / 'source.png', Path(folder) / 'mask.png'
        image = pil(rgb)
        image.thumbnail((2000, 2000))
        image.save(source)
        try:
            subprocess.run([str(helper), str(source), str(target)], check=True,
                           capture_output=True, timeout=45)
            with Image.open(target) as image:
                prediction = np.asarray(image.convert('L'), dtype=np.float32) / 255
        except (OSError, subprocess.SubprocessError) as exc:
            raise RuntimeError('White-part detection failed. Turn off Recover white parts or retry.') from exc
    return cv2.resize(prediction, rgb.shape[1::-1], interpolation=cv2.INTER_LINEAR)


def merge_white_prediction(rgb, mask, prediction):
    """Accept small white additions with existing support, above the floor.

The lower silhouette guard deliberately leaves floor-contact omissions for
manual review: a pale shadow can look just like a white brick to either model.
"""
    solid = mask > .75
    ys, xs = np.where(solid)
    if not len(xs):
        return mask.copy(), []
    span = max(int(np.ptp(xs)) + 1, int(np.ptp(ys)) + 1)
    lum = rgb.mean(2)
    chroma = np.ptp(rgb, axis=2)
    white = (lum > .74) & (chroma < .16)
    # A local bottom envelope admits upper/rear wings but excludes expansion
    # onto the photographed floor below wheels, bases and front wings.
    rows = np.arange(mask.shape[0], dtype=np.float32)[:, None]
    bottom = np.max(np.where(solid, rows, -1), axis=0)[None, :]
    radius = max(3, round(span * .025))
    bottom = cv2.dilate(bottom, np.ones((1, radius * 2 + 1), np.uint8))
    above_floor = rows < bottom - max(3, span * .008)
    distance = cv2.distanceTransform((~solid).astype(np.uint8), cv2.DIST_L2, 5)
    candidate = (prediction > .65) & white
    count, labels, stats, _ = cv2.connectedComponentsWithStats(candidate.astype(np.uint8))
    accepted = np.zeros(mask.shape, np.uint8)
    boxes = []
    for i in range(1, count):
        x, y, w, h, area = (int(v) for v in stats[i])
        if area < 10 or x == 0 or y == 0 or x + w >= mask.shape[1] or y + h >= mask.shape[0]:
            continue
        part = labels[y:y+h, x:x+w] == i
        old = mask[y:y+h, x:x+w]
        added = part & (old < .5)
        if added.sum() < 8 or added.sum() > span * span * .025:
            continue
        if np.count_nonzero(part & (old > .75)) < max(3, area * .015):
            continue
        dist = distance[y:y+h, x:x+w]
        if np.max(dist[added]) > span * .12:
            continue
        # Reject whole dubious regions rather than slicing a floor blob into
        # an apparently plausible part at the lower-envelope boundary.
        if np.mean(above_floor[y:y+h, x:x+w][added]) < .98:
            continue
        accepted[y:y+h, x:x+w][part] = 1
        boxes.append([x, y, x+w, y+h])
    if not boxes:
        return mask.copy(), []
    # Keep Vision's soft outline; confident plate interiors become opaque.
    alpha = np.clip((prediction - .35) / .55, 0, 1)
    support = cv2.dilate(accepted, np.ones((3, 3), np.uint8)) > 0
    support &= white & above_floor & (distance <= span * .12)
    return np.maximum(mask, alpha * support), boxes


def recover_white_parts(rgb, mask, on_phase=None):
    started = time.monotonic()
    ys, xs = np.where(mask > .5)
    if not len(xs):
        raise ValueError('No existing subject mask for white-part recovery')
    span = max(int(np.ptp(xs)) + 1, int(np.ptp(ys)) + 1)
    pad = max(32, round(span * .11))
    l, t = max(0, int(xs.min())-pad), max(0, int(ys.min())-pad)
    r, b = min(mask.shape[1], int(xs.max())+pad+1), min(mask.shape[0], int(ys.max())+pad+1)
    if on_phase:
        on_phase('Recovering faint white parts', .82)
    pixels = rgb[t:b, l:r]
    prediction = predict_white_crop(pixels)
    # A darker detection-only view separates pale stud shading from the booth.
    # Both mattes use the same source pixels; exported colour stays untouched.
    prediction = np.maximum(prediction, predict_white_crop(pixels ** 3))
    repaired, boxes = merge_white_prediction(rgb[t:b, l:r], mask[t:b, l:r], prediction)
    result = mask.copy()
    result[t:b, l:r] = repaired
    return result, dict(white_version=WHITE_VERSION, white_recovery_boxes=[
        [x+l, y+t, right+l, bottom+t] for x, y, right, bottom in boxes],
        white_added_pixels=int(np.count_nonzero((result > .5) & (mask <= .5))), white_model_passes=2,
        white_seconds=round(time.monotonic()-started, 3))
